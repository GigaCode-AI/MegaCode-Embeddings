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

    # load docs
    with open(os.path.join(cfg.data_path, "code_idx_map.txt")) as f:
        d = json.load(f)
    assert len(d) == len(set(d.values()))  # ensure 1->1 mapping
    key2snippet = {v: k for k, v in d.items()}
    keys = list(key2snippet.keys())
    snippets = list(key2snippet.values())

    # load queries
    queries = []
    labels = []
    with open(os.path.join(cfg.data_path, "cosqa-retrieval-test-500.json")) as f:
        d = json.load(f)
    for x in d:
        assert x["label"] == 1
        assert key2snippet[x["retrieval_idx"]] == x["code"]
        queries.append(x["doc"])
        labels.append(x["retrieval_idx"])

    # search top-k
    if isinstance(model, RetrieverTfidf):
        X = model.fit_transform(snippets)
        Q = model.transform(queries)
        D, I = topk_sparse(Q @ X.T, cfg.k)
    else:
        Q = model.vectorize([cfg.query_prefix + q for q in  queries], cfg.max_tokens_query)
        X = model.vectorize([cfg.passage_prefix + s for s in snippets], cfg.max_tokens_code)
        D, I = topk_dense(Q @ X.T, cfg.k)

    # save
    with open(cfg.output_path, "w") as f:
        for i in range(I.shape[0]):
            x = {
                "question": queries[i],
                "label": labels[i],
                "lang": "python",
                "candidates": []
            }
            for j in range(I.shape[1]):
                x["candidates"].append({
                    "id": keys[I[i, j]],
                    "text": snippets[I[i, j]],
                    "score": float(D[i, j])
                })
            f.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
