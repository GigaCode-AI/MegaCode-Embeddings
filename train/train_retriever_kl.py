"""
Distill reranker in retriever

Line format of a single shard, e.g. "shard-00001-of-00128.jsonl":
{
    "question": List[int],
    "candidates": [
        {"context": List[int], "score": float}
    ]
}
"""
import os
import json
import glob
import random
import argparse
import logging
from typing import Union, Dict, List, Tuple

import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.integrations import MLflowCallback, WandbCallback
from safetensors.torch import load_file

# Check for tensorboard; otherwise nothing will crash but logs won't be written to tb
try:
    import tensorboard
except ImportError:
    print("install tb")
    raise

T = torch.Tensor


def get_logger(name: str = None, level: Union[int, str] = logging.INFO, log_path: str = None):
    """
    Format taken from hydra
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.setLevel(level)
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return logger


def str2bool(x: Union[str, bool]) -> bool:
    """
    For Python logger
    """
    if isinstance(x, bool):
        return x
    elif isinstance(x, str):
        if x.lower() == "true":
            return True
        elif x.lower() == "false":
            return False
        else:
            raise ValueError(f"invalid arg value: {x}")
    else:
        raise TypeError(f"invalid arg type: {type(x)}")


def tensorize(input_ids: List[List[int]], pad_token_id: int, device=None) -> Tuple[T, T]:
    m = max(map(len, input_ids))
    input_ids_t = torch.tensor([x + [pad_token_id] * (m - len(x)) for x in input_ids], device=device)
    attention_mask_t = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in input_ids], device=device)
    return input_ids_t, attention_mask_t


class TrainDataset(Dataset):
    def __init__(
            self,
            data: List[str],
            k: int,
            eos: int,
            max_tokens_query: int,
            max_tokens_code: int,
            seed: int = 0
    ):
        super().__init__()
        self.data = data
        self.k = k
        self.eos = eos
        self.max_tokens_query = max_tokens_query
        self.max_tokens_code = max_tokens_code
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = json.loads(self.data[i])
        docs = self.rng.sample(x["candidates"], self.k)
        return {
            "question": self.ensure_eos(x["question"][:self.max_tokens_query]),
            "candidates": [self.ensure_eos(d["context"][:self.max_tokens_code]) for d in docs],
            "scores": [d["score"] for d in docs]
        }

    def ensure_eos(self, ids: List[int]):
        if ids[-1] != self.eos:
            return ids + [self.eos]
        return ids


class TrainCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        questions = []
        candidates = []
        scores = []
        for x in batch:
            questions.append(x["question"])
            candidates += x["candidates"]
            scores += x["scores"]
        res = dict()
        res["input_ids_q"], res["attention_mask_q"] = tensorize(questions, self.pad_token_id)
        res["input_ids_c"], res["attention_mask_c"] = tensorize(candidates, self.pad_token_id)
        res["scores"] = torch.tensor(scores, dtype=torch.float32)
        return res


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        self.train_collator = kwargs.pop("train_collator")
        self.tau_student = kwargs.pop("tau_student")
        self.tau_teacher = kwargs.pop("tau_teacher")
        self.k = kwargs.pop("k")
        self.logger = kwargs.pop("logger")  # to log metrics to log file
        super().__init__(**kwargs)

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        self.logger.info(output)  # only change

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.train_collator,
            shuffle=True,
            drop_last=True
        )

    def compute_loss(self, model, inputs: Dict[str, T], return_outputs: bool = False, *args, **kwargs) -> T:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        assert not return_outputs
        # vectorize queries and documents
        q = self.vectorize(inputs["input_ids_q"], inputs["attention_mask_q"])  # [n, d]
        c = self.vectorize(inputs["input_ids_c"], inputs["attention_mask_c"])  # [nk, d]

        # compute student scores
        q = q[:, None, :]  # [n, 1, d]
        c = c.reshape(q.shape[0], self.k, -1).transpose(1, 2)  # [n, d, k]
        student_scores = (q @ c).squeeze(1)

        # compute loss
        teacher_scores = inputs["scores"].to(self.args.device).reshape(q.shape[0], -1)  # [n, k]
        loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        log_probs_student = torch.log_softmax(student_scores / self.tau_student, dim=-1)
        log_probs_teacher = torch.log_softmax(teacher_scores / self.tau_teacher, dim=-1)
        loss = loss_fn(input=log_probs_student, target=log_probs_teacher)

        # # write loss to logs
        # if (self.args.local_rank == 0) & (self.state.global_step % self.args.logging_steps == 0):
        #     self.logger.info({
        #         "loss": round(loss.item(), 6),
        #         "step": self.state.global_step,
        #         "epoch": round(self.state.epoch, 2),
        #     })
        return loss

    def vectorize(self, input_ids: T, attention_mask: T) -> T:
        """
        Get hidden states, take eos embedding, normalize.

        input_ids, attention_mask - [n, t]
        return - [n, d]
        """
        device = self.args.device
        outputs = self.model(input_ids.to(device), attention_mask.to(device))
        x = outputs.last_hidden_state
        xs = torch.arange(x.shape[0], device=device)
        ys = attention_mask.sum(1) - 1
        x = x[xs, ys]
        # RuntimeError: div(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.
        x = F.normalize(x)
        return x


def load_checkpoint(path: str):
    if path.endswith(".bin"):
        return torch.load(path, map_location="cpu")
    elif path.endswith(".safetensors"):
        return load_file(path, device="cpu")
    else:
        raise NotImplementedError


def main(args):
    # setup training args
    training_args = TrainingArguments(
        do_eval=False,
        output_dir=args.output_dir,
        overwrite_output_dir=False,  # default is false, but made it explicit
        max_steps=args.num_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=({"use_reentrant": False} if args.gradient_checkpointing else None),
        deepspeed=args.deepspeed,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,  # default value, made explicit to remember
        weight_decay=1e-4,
        logging_steps=args.logging_steps,
        save_total_limit=100,
        save_safetensors=args.save_safetensors,
        save_on_each_node=False,
        save_only_model=args.save_only_model
    )
    local_rank = training_args.local_rank
    world_size = training_args.world_size

    # setup logging
    if local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        log_level = logging.INFO
        log_path = os.path.join(args.output_dir, "train.log")
    else:
        log_level = logging.WARN
        log_path = None
    logger = get_logger(name="train", level=log_level, log_path=log_path)
    transformers.utils.logging.set_verbosity(logging.DEBUG)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # load train data
    # Lines are much lighter than dicts; json.loads is done at TrainDataset level
    train_data_lines = []
    paths = sorted(glob.glob(os.path.join(args.train_data_dir, "*.jsonl")))
    paths_rank = [path for i, path in enumerate(paths) if i % world_size == local_rank]
    for path in tqdm.tqdm(paths_rank, desc=f"[rank {local_rank}] load data", position=local_rank):
        flag = False
        with open(path) as f:
            for line in f:
                train_data_lines.append(line)
                if len(train_data_lines) == args.limit:
                    flag = True
                    break
        if flag:
            break

    # sync num of train examples
    # Otherwise there's no guarantee that steps per epoch will be the same across processes
    n = torch.tensor(len(train_data_lines)).to(training_args.device)
    ns = [torch.empty_like(n) for _ in range(world_size)]
    dist.all_gather(ns, n)
    logger.info(f"num training examples per process: {[x.item() for x in ns]}")
    dist.all_reduce(n, op=dist.ReduceOp.MIN)
    n = n.item()
    logger.info(f"synced number of training examples: {n}")
    train_data_lines = train_data_lines[:n]

    # setup data pipeline
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_dir)
    train_dataset = TrainDataset(
        train_data_lines,
        k=args.k,
        eos=tokenizer.eos_token_id,
        max_tokens_query=args.max_tokens_query,
        max_tokens_code=args.max_tokens_code,
        seed=228
    )
    train_collator = TrainCollator(pad_token_id=tokenizer.pad_token_id)

    # setup model
    model = AutoModel.from_pretrained(args.pretrained_dir, torch_dtype=torch.bfloat16)
    if args.checkpoint_path:
        logger.info(f"loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(load_checkpoint(args.checkpoint_path))

    # setup trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        train_collator=train_collator,
        tau_student=args.tau_student,
        tau_teacher=args.tau_teacher,
        k=args.k,
        logger=logger
    )
    trainer.remove_callback(MLflowCallback)
    trainer.remove_callback(WandbCallback)

    # train!
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--pretrained_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--num_steps", required=True, type=int)
    parser.add_argument("--save_steps", required=True, type=int, help="If validation is needed, set equal to eval_steps; otherwise desired save frequency")
    parser.add_argument("--tau_student", type=float, default=0.01, help="Divide by this")
    parser.add_argument("--tau_teacher", type=float, default=1.0, help="Divide by this")
    parser.add_argument("--max_tokens_query", type=int, default=64, help="Must match the value used when scoring candidates")
    parser.add_argument("--max_tokens_code", type=int, default=1024, help="Same as max_tokens_query")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64, help="Should be relatively large since one language pair is processed per micro-step; global batch should be at least 512")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--deepspeed", type=str, default="")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False)
    parser.add_argument("--save_safetensors", type=str2bool, default=True)
    parser.add_argument("--save_only_model", type=str2bool, default=False)
    parser.add_argument("--limit", type=int, default=-1)
    main(parser.parse_args())
