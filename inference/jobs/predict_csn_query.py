"""
Given a Bing query, find the relevant function.
Only (query, function) pairs where all annotators agreed the function is relevant to the query are included.
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

    langs = ["Go", "Java", "JavaScript", "PHP", "Python", "Ruby"]
    with open(cfg.output_path, "w") as fout:
        for lang in langs:
            # load queries with annotated positives
            queries = []
            with open(os.path.join(cfg.data_path, lang, "annotation.jsonl")) as f:
                for line in f:
                    x = json.loads(line)
                    x["urls"] = set(x["urls"])
                    queries.append(x)

            # load whole corpus
            functions = []
            with open(os.path.join(cfg.data_path, lang, "corpus.jsonl")) as f:
                for line in f:
                    functions.append(json.loads(line))

            # search top-k
            if isinstance(model, RetrieverTfidf):
                X = model.fit_transform([x["function"] for x in functions])
                Q = model.transform([x["query"] for x in queries])
                D, I = topk_sparse(Q @ X.T, cfg.k)
            else:
                Q = model.vectorize([cfg.query_prefix + x["query"] for x in queries], cfg.max_tokens_query)
                X = model.vectorize([cfg.passage_prefix + x["function"] for x in functions], cfg.max_tokens_code)
                D, I = topk_dense(Q @ X.T, cfg.k)

            # save
            for i in range(len(queries)):
                x = {
                    "question": queries[i]["query"],
                    "label": list(queries[i]["urls"]),
                    "lang": lang,
                    "candidates": []
                }
                for j in range(I.shape[1]):
                    x["candidates"].append({
                        "id": functions[I[i, j]]["url"],
                        "text": functions[I[i, j]]["function"],
                        "score": float(D[i, j])
                    })
                fout.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
