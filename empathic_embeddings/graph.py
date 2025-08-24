"""Signed graph utilities for the unsupervised mode."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_edges(
    vectors: Dict[str, np.ndarray],
    w2i: Dict[str, int],
    thr: float = 0.4,
    k_pos: int = 5,
    neg_pref: Tuple[str, ...] = ("un", "in", "im", "ir", "non", "dis", "il"),
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """Construct positive and negative edges between words.

    This mirrors the heuristic used in the original script. Positive edges are
    drawn between nearest neighbours while negative edges connect prefix-negated
    forms with their roots.
    """

    words = list(vectors)
    V = np.vstack([vectors[w] for w in words])
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    pos_edges: List[Tuple[int, int, float]] = []
    neg_edges: List[Tuple[int, int, float]] = []
    k = min(k_pos, len(words) - 1)
    for i, wi in enumerate(words):
        sims = V @ V[i]
        idxs = np.argpartition(-sims, k + 1)[: k + 1]
        for j in idxs:
            if i == j or sims[j] < thr:
                continue
            pos_edges.append((i, j, float(abs(sims[j]))))
        for pref in neg_pref:
            if wi.startswith(pref) and len(wi) > len(pref) + 2:
                root = wi[len(pref) :]
                if root in w2i:
                    j = w2i[root]
                    weight = float(abs(V[i].dot(V[j])))
                    if weight < thr:
                        weight = thr
                    neg_edges.append((i, j, weight))
    return pos_edges, neg_edges


def spectral_axis(pos, neg, n) -> np.ndarray:
    """Compute valence axis via a signed-graph Laplacian eigenproblem."""

    rows_p, cols_p, data_p = zip(*pos) if pos else ([], [], [])
    rows_m, cols_m, data_m = zip(*neg) if neg else ([], [], [])
    S_plus = sp.coo_matrix((data_p, (rows_p, cols_p)), shape=(n, n), dtype=np.float64)
    S_minus = sp.coo_matrix((data_m, (rows_m, cols_m)), shape=(n, n), dtype=np.float64)
    S_plus = S_plus + S_plus.T
    S_minus = S_minus + S_minus.T
    Dp = sp.diags(np.array(S_plus.sum(axis=1)).ravel())
    Dm = sp.diags(np.array(S_minus.sum(axis=1)).ravel())
    Lp = (Dp - S_plus).astype(np.float64)
    Lm = (Dm - S_minus).astype(np.float64)
    eps = 1e-4
    zdiag = (Lm.diagonal() == 0).astype(float)
    Lm += sp.diags(zdiag * 1.0 + eps)
    vals, vecs = spla.eigsh(Lp, k=1, M=Lm, which="SM", tol=1e-6, maxiter=500)
    axis = vecs[:, 0]
    return axis / (np.linalg.norm(axis) + 1e-12)
