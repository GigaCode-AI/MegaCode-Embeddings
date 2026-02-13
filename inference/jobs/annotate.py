import os
import json
import glob
import shutil
from argparse import ArgumentParser
from multiprocessing import Process

import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt


system_prompt = """A code snippet and a text query are given.
Answer "yes" if snippet is relevant to the query, "no" otherwise.
The code snippet is relevant to the query if it helps to answer the query either fully or partially.

### Example 1:
Query:
function that raises a number to a power

Code:
```
def f(x, p):
    return x ** p
```
Answer: yes
Explanation: function performs exactly what the user asks.


### Example 2:
Query:
toggleDone method implementation

Code:
```
    final TodoAdapter adapter = this;
    final View view = convertView;
    final Todo todo = getItem(position);
    todo.setCompleted(true);
    dbHelper.updateTodo(todo);
}
```
Answer: no
Explanation: it is not possible to exactly tell that provided snippet belongs to `toggleDone` method or function

### Example 3:
Query:
Remove user from a role

Code:
```
void removeUser();
```
Answer: yes
Explanation: despite the absence of function's body, it's obvious by it's name what it should do, and it's probably what user is searching for.

### Example 4:
Query:
Kills the server

Code:
```
class BaseService:
def kill(self):
        raise NotImplementedError()
```
Answer: no
Explanation: despite the fact that the function is called "kill," it does not do what the query suggests.

### Example 5:
Query:
Format comp messages to start with >>

Code:
```
  return ['  ---'
         ,'    message: >'
         ,       pad(6, entry.log.join(' '))
         ,'  ...'
         ].join('\n') }
```
Answer: no
Explanation: this function formats somehow entry.log, but result doesn't start with >>

### Example 6:
Query:
Set all the rule parameters in one go.

Code:
```
		Type								= type;
		MinAge								= minAge;
		MaxAge								= maxAge;
		MinVelocity							= minVelocity;
		MaxVelocity							= maxVelocity;
		Damping								= damping;
	}
```
Answer: no
Explanation: here is implementation or setting some parameters to some member variables, but these parameters are not *rule*.

### Example 7:
Query:
Returns ArticleAdapter

Code:
```
class BaseArticleView(BaseFormView):
def adapter(self):
        return IArticleAdapter(self.context)
```
Answer: no
Explanation: expected ArticleAdapter to be returned, got IArticleAdapter

### Example 8:
Query:
#########################

Code:
```
	var jsonData = '{ "Method":"GeHotelAmenity"}';
	var resGeHotelAmenity = callAjax(urlToHandler, jsonData, false);
	if (resGeHotelAmenity.Status == 1) {
		allHotelAmenityData = resGeHotelAmenity.Data;
	}
	else {
		Admin.Modal("Error", resGeHotelAmenity.Message);
	}
}
```
Answer: no
Explanation: query is not informative. Each code snippet must be classified as negative to such query.

Answer with a single word without any comments and explanation."""

user_prompt = """Query:
{query}

Code:
```
{code}
```"""


def get_paths(shards_mask: str):
    paths = []
    for mask in shards_mask.split(","):
        paths_i = sorted(glob.glob(mask))
        assert len(paths_i) > 0, f"there are no paths matching mask {mask}"
        paths += paths_i
    return paths


def main(args):
    input_paths = get_paths(args.input_paths)

    # load data
    pairs = set()
    for path in tqdm.tqdm(input_paths):
        with open(path) as f:
            for line in f:
                x = json.loads(line)
                for d in x["candidates"][:args.k]:
                    pairs.add((x["question"], d["text"]))
    pairs = list(pairs)

    # setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_dir)

    # tokenize pairs
    q_ids = tokenizer([x[0] for x in pairs], truncation=True, max_length=args.max_tokens_query, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer([x[1] for x in pairs], truncation=True, max_length=args.max_tokens_code, add_special_tokens=False)["input_ids"]

    # tokenize template
    x = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    x = tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    i = x.index("{query}")
    j = x.index("{code}")
    parts = [x[:i], x[i + len("{query}"):j], x[j + len("{code}"):]]
    parts = tokenizer(parts, add_special_tokens=False)["input_ids"]

    # build prompts
    prompts = []
    for i in range(len(pairs)):
        p = parts[0] + q_ids[i] + parts[1] + c_ids[i] + parts[2]
        prompts.append(TokensPrompt(prompt_token_ids=p))

    # predict
    os.makedirs(args.temp_dir, exist_ok=True)

    def job_fn(job_id):
        indices_job = [i for i in range(len(pairs)) if i % args.nproc == job_id]
        prompts_job = [prompts[i] for i in indices_job]

        # setup model
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, (job_id * args.tp_size + i for i in range(args.tp_size))))
        model = LLM(model=args.pretrained_dir, max_model_len=args.max_model_len, tensor_parallel_size=args.tp_size)

        # generate
        # sampling params: https://huggingface.co/Qwen/Qwen3-32B#switching-between-thinking-and-non-thinking-mode
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=args.max_new_tokens, seed=0)
        outputs = model.generate(prompts_job, sampling_params)

        # save
        with open(os.path.join(args.temp_dir, f"job_{job_id}.jsonl"), "w") as f:
            for i in range(len(indices_job)):
                f.write(json.dumps({"idx": indices_job[i], "answer": outputs[i].outputs[0].text}) + "\n")

    jobs = []
    for i in range(args.nproc):
        job = Process(target=job_fn, args=(i,))
        job.start()
        jobs.append(job)
    for job in jobs:
        job.join()
    res = {}
    for i in range(args.nproc):
        with open(os.path.join(args.temp_dir, f"job_{i}.jsonl")) as f:
            for line in f:
                x = json.loads(line)
                res[x["idx"]] = x["answer"]
    answers = [res[i] for i in range(len(pairs))]
    shutil.rmtree(args.temp_dir)

    # write answer to preds
    pair2answer = dict(zip(pairs, answers))
    for path in tqdm.tqdm(input_paths, desc="save"):
        # read predictions
        tmp = []
        with open(path) as f:
            for line in f:
                tmp.append(json.loads(line))
        # write answer to top-k
        for x in tmp:
            for d in x["candidates"][:args.k]:
                d[args.answer_key] = pair2answer[(x["question"], d["text"])]
        # save
        with open(path, "w") as f:
            for x in tmp:
                f.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_paths", help="comma-sep, wildcards supported")
    parser.add_argument("--pretrained_dir")
    parser.add_argument("--temp_dir")
    parser.add_argument("--answer_key")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--tp_size", type=int, default=2)  # for qwen3-32b
    parser.add_argument("--max_tokens_query", type=int, default=64)
    parser.add_argument("--max_tokens_code", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=1024)  # >99% reasoning steps are within this limit
    main(parser.parse_args())
