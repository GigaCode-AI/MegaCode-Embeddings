"""
Same task as CSN, but with functions where:
1. Name is replaced with "Func", argument names with "arg_0", "arg_1", ...
2. All comments removed, not just docstrings
3. Python only
4. Positive must be found in the entire test set, not among 1000 candidates with 1 positive
"""
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

    queries = []
    snippets = []
    with open(cfg.data_path) as f:
        for line in f:
            x = json.loads(line)
            queries.append(" ".join(x["docstring_tokens"]))
            snippets.append(" ".join(x["function_tokens"]))

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
    with open(cfg.output_path, "w") as f:
        for i in range(I.shape[0]):
            x = {
                "question": queries[i],
                "label": i,
                "lang": "python",
                "candidates": []
            }
            for j in range(I.shape[1]):
                x["candidates"].append({
                    "id": int(I[i, j]),
                    "text": snippets[I[i, j]],
                    "score": float(D[i, j])
                })
            f.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
