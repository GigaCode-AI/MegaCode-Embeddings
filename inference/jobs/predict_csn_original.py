"""
For each (query, positive) pair, 999 negatives are randomly sampled from the same language.
Task: find the positive among 1000 candidates.
"""
import json
import sys
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from scipy.sparse import csr_matrix, diags

# So hydra can access the classes that need to be instantiated
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrievers import RetrieverTfidf
from src.utils import topk_dense, topk_sparse


@hydra.main(config_path="../config", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # instantiate model
    model = hydra.utils.instantiate(cfg.model)

    langs = ["go", "java", "javascript", "php", "python", "ruby"]
    num_random_neg = 999
    bs = 1024
    with open(cfg.output_path, "w") as fout:
        for lang in langs:
            # load data
            queries = []
            snippets = []
            labels = []
            with open(f"{cfg.data_path}/{lang}/final/jsonl/test/{lang}_test_0.jsonl") as f:
                for line in f:
                    x = json.loads(line)
                    if x["partition"] == "test":  # sanity check
                        queries.append(" ".join(x["docstring_tokens"]))
                        snippets.append(" ".join(x["code_tokens"]))
                        labels.append(x["url"])

            # vectorize
            if isinstance(model, RetrieverTfidf):
                X = model.fit_transform(snippets)
                Q = model.transform(queries)

                # each vs each
                S = Q @ X.T

                # random candidates and diagonal
                rng = np.random.RandomState(228)
                n_rows, n_cols = len(queries), len(snippets)
                indices = [rng.choice(n_cols, size=num_random_neg, replace=False) for _ in range(n_rows)]
                indices = np.fromiter((i for row in indices for i in row), dtype=np.int64)
                indptr = np.arange(0, n_rows * num_random_neg + 1, num_random_neg, dtype=np.int64)
                data = np.ones(n_rows * num_random_neg, dtype=S.dtype)
                mask1 = csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))  # random candidates
                mask2 = diags(np.ones(n_rows, dtype=S.dtype)).tocsr()  # and diagonal

                # mask
                mask = mask1.maximum(mask2)  # equivalent of logical_or
                S = S.multiply(mask)
                D, I = topk_sparse(S, cfg.k)

                # save
                for i in range(I.shape[0]):
                    x = {
                        "question": queries[i],
                        "label": labels[i],
                        "lang": lang,
                        "candidates": []
                    }
                    for j in range(I.shape[1]):
                        x["candidates"].append({
                            "id": labels[I[i, j]],
                            "text": snippets[I[i, j]],
                            "score": float(D[i, j])
                        })
                    fout.write(json.dumps(x) + "\n")
            else:
                Q = model.vectorize([cfg.query_prefix + q for q in queries], cfg.max_tokens_query)
                X = model.vectorize([cfg.passage_prefix + s for s in snippets], cfg.max_tokens_code)

                # search top-k among 1 pos and 999 random
                n = len(queries)
                device = Q.device
                g = torch.Generator(device=device)
                g.manual_seed(228)
                for i in range(0, n, bs):
                    j = min(i + bs, n)
                    pos = (Q[i:j] * X[i:j]).sum(1)[:, None]  # [bs, 1]
                    idx_rand = torch.randint(0, n, size=(j - i, num_random_neg), device=device, generator=g)  # [bs, k]
                    neg = (Q[i:j, None] @ X[idx_rand].transpose(1, 2)).squeeze(1)  # [bs, 1, d] x [bs, k, d]
                    neg.masked_fill_(idx_rand == torch.arange(i, j, device=device)[:, None], -100.0)
                    scores = torch.cat([pos, neg], dim=1)  # [bs, k+1]
                    D, I = topk_dense(scores, cfg.k)
                    batch_idx = np.hstack([np.arange(i, j)[:, None], idx_rand.cpu().numpy()])
                    for ii in range(I.shape[0]):
                        x = {
                            "question": queries[i + ii],
                            "label": labels[i + ii],
                            "lang": lang,
                            "candidates": []
                        }
                        for jj in range(I.shape[1]):
                            idx = batch_idx[ii, I[ii, jj]]
                            x["candidates"].append({
                                "id": labels[idx],
                                "text": snippets[idx],
                                "score": float(D[ii, jj])
                            })
                        fout.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
