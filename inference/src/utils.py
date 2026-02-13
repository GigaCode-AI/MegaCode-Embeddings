from typing import Tuple
import torch
import numpy as np
from scipy.sparse import csr_matrix


def topk_dense(X: torch.Tensor, k: int) -> Tuple[np.ndarray, np.ndarray]:
    top = torch.topk(X, dim=1, k=min(k, X.shape[1]))
    D = top.values.cpu().numpy()
    I = top.indices.cpu().numpy()
    return D, I

def topk_sparse(X: csr_matrix, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    D = np.zeros((n, k), dtype=np.float32) - 100
    I = np.zeros((n, k), dtype=np.int64) - 1
    for i in range(n):
        start = X.indptr[i]
        end = X.indptr[i + 1]
        if start == end:
            continue
        data = X.data[start:end]
        indices = X.indices[start:end]
        idx = np.argsort(-data)[:k]
        D[i, :idx.shape[0]] = data[idx]
        I[i, :idx.shape[0]] = indices[idx]
    return D, I