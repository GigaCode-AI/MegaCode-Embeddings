import os
import json
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# So hydra can access the classes that need to be instantiated
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import topk_dense, topk_sparse
from src.retrievers import RetrieverTfidf


@hydra.main(config_path="../config", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # instantiate model
    model = hydra.utils.instantiate(cfg.model)

    with open(cfg.data_path) as f:
        data = json.load(f)

    langs = ['python', 'cpp', 'java', 'typescript', 'rust', 'go']
    with open(cfg.output_path, "w") as fout:
        for lang in langs:
            for repo in data[lang]:
                queries = []
                labels = []
                for x in repo["needles"]:
                    queries.append(x["query"])
                    labels.append(f'{x["global_start_byte"]}_{x["global_end_byte"]}')
                snippets = []
                keys = []
                for file_path, funcs in repo["functions"].items():
                    for func in funcs:
                        content = repo["content"][file_path].encode()
                        snippets.append(content[func["start_byte"]:func["end_byte"]].decode())
                        keys.append(f'{func["global_start_byte"]}_{func["global_end_byte"]}')
                assert set(labels) < set(keys)

                # search top-k
                if isinstance(model, RetrieverTfidf):
                    X = model.fit_transform(snippets)
                    Q = model.transform(queries)
                    D, I = topk_sparse(Q @ X.T, cfg.k)
                else:
                    Q = model.vectorize([cfg.query_prefix + q for q in queries], cfg.max_tokens_query)
                    X = model.vectorize([cfg.passage_prefix + s for s in snippets], cfg.max_tokens_code)
                    D, I = topk_dense(Q @ X.T, cfg.k)

                # save
                for i in range(I.shape[0]):
                    x = {
                        "question": queries[i],
                        "label": labels[i],
                        "lang": lang,  # for grouping
                        "repo": repo["repo"],  # just in case
                        "candidates": [],
                    }
                    for j in range(I.shape[1]):
                        x["candidates"].append({
                            "id": keys[I[i, j]],
                            "text": snippets[I[i, j]],
                            "score": float(D[i, j])
                        })
                    fout.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
