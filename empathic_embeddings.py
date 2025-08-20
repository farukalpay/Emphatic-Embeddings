#!/usr/bin/env python3
"""
Empathic (Affect‑Augmented) Word Embeddings
==========================================

Supervised mode : ridge projector        (--vad <file>)
Unsupervised    : signed graph with      L⁺ z = λ L⁻ z
                  + spectral‑cap cert.   (omit --vad)

MODIFICATION NOTE:
This file exposes a small library for affect‑augmented word embeddings.
It includes a self-contained `main()` demo, while the `_cli()` function
provides the original command-line interface.
"""
import gzip, json, math, random, sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False

# ───────────────────── utilities ─────────────────────────────────────────
def read_vec(path: str,
             keep: Optional[set[str]] = None) -> Iterable[Tuple[str, np.ndarray]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().split()
        dim = int(first[1]) if len(first) == 2 and first[0].isdigit() else len(first) - 1
        if len(first) != dim + 1:
            f.seek(0)
        for ln in f:
            p = ln.rstrip().split()
            if len(p) != dim + 1:
                continue
            w, nums = p[0], p[1:]
            if keep and w not in keep:
                continue
            yield w, np.fromiter((float(x) for x in nums), dtype=np.float32)

def load_vad(path: str) -> Dict[str, List[float]]:
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            p = ln.strip().split()
            if len(p) == 4:
                out[p[0]] = [float(x) for x in p[1:]]
    return out

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

# ───────────────────── edge builder (unsupervised) ───────────────────────
def build_edges(E: Dict[str, np.ndarray],
                w2i: Dict[str, int],
                thr: float = 0.4,
                k_pos: int = 5,
                neg_pref: Tuple[str, ...] = ("un", "in", "im", "ir", "non", "dis", "il")
               ) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    words = list(E)
    V = np.vstack([E[w] for w in words])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    pos_edges: List[Tuple[int, int, float]] = []
    neg_edges: List[Tuple[int, int, float]] = []
    for i, wi in enumerate(words):
        sims = V @ V[i]
        idxs = np.argpartition(-sims, k_pos + 1)[: k_pos + 1]
        for j in idxs:
            if i == j or sims[j] < thr:
                continue
            pos_edges.append((i, j, abs(float(sims[j]))))
        for pref in neg_pref:
            if wi.startswith(pref) and len(wi) > len(pref) + 2:
                root = wi[len(pref):]
                if root in w2i:
                    j = w2i[root]
                    weight = abs(float(V[i].dot(V[j])))
                    if weight < thr:
                        weight = thr
                    neg_edges.append((i, j, weight))
    return pos_edges, neg_edges

# ───────────────────── main class ────────────────────────────────────────
class EmpathicSpace:
    def __init__(self, E: Dict[str, np.ndarray], gold: Optional[Dict[str, List[float]]] = None,
                 seed: Optional[int] = None):
        """Create an affect-augmented embedding space.

        Args:
            E: Mapping of words to their original embedding vectors.
            gold: Optional gold-standard VAD lexicon for supervised mode.
            seed: Optional random seed to make unsupervised training deterministic.
        """
        self.E = E
        self.d = len(next(iter(E.values())))
        self.words = list(E)
        self.w2i = {w: i for i, w in enumerate(self.words)}
        self.seed = seed
        if gold:
            self._train_supervised(gold)
        else:
            self._train_unsup(seed=seed)
        # Normalize and store augmented vectors for quick similarity queries.
        # A second matrix view is kept to support fast vectorised search.
        self.store = {w: self.vector(w) / np.linalg.norm(self.vector(w))
                      for w in self.words}
        self._store_matrix = np.vstack([self.store[w] for w in self.words])

        # Initialize parameters for the phrase composition tensor product.
        # This implements the logic required to repair the "monotonicity mismatch".
        self.alpha_compose = 0.5
        self.beta_compose = 0.5
        mean_vec_norm = np.mean([np.linalg.norm(v) for v in self.E.values()])
        mean_vec_norm = mean_vec_norm if mean_vec_norm > 0 else 1.0 # Avoid division by zero
        self.gamma_tensor = self.lambda_star / (2 * mean_vec_norm)

    # ─────────── vectors ────────────────────────────────────────────────
    def vector(self, w: str) -> np.ndarray:
        # Original embedding + affect dimension (scaled by alpha)
        return np.concatenate([self.E[w], [self.alpha * self.val[self.w2i[w]]]])

    def phrase_vector(self, text: str, iter: int = 0, tol: float = 1e-6) -> np.ndarray:
        """
        Computes a vector for a multi-word phrase using a compositional tensor product.

        This method implements the logic described in the introductory text to
        address the "categorical monotonicity mismatch". It combines token vectors
        using a custom tensor product `⊗` that preserves affect.

        Args:
            text (str): The input phrase (e.g., "not at all happy").
            iter (int): If > 0, repeatedly applies the self-composition functor
                        F(x) = x ⊗ x for `iter` steps or until convergence,
                        to find the fixed point for repeating phrases.
            tol (float): The tolerance for the convergence check when iter > 0.

        Returns:
            np.ndarray: The final empathic vector for the phrase.
        """
        # 1. Tokenize and filter words present in the vocabulary
        tokens = [w for w in text.split() if w in self.w2i]
        if not tokens:
            return np.zeros(self.d + 1, dtype=np.float32)

        # 2. Map each token to its initial (vector, affect) pair.
        # The initial affect `a` is `alpha * valence`, as in the single-word `vector` method.
        initial_pairs = []
        for w in tokens:
            idx = self.w2i[w]
            v = self.E[w]
            a = self.alpha * self.val[idx]
            initial_pairs.append((v, a))

        # 3. Reduce the list of pairs using a left-fold with the custom tensor product.
        # (v₁,a₁) ⊗ (v₂,a₂) = (μ(v₁,v₂), α·a₁ + β·a₂ + γ·⟨v₁,v₂⟩)
        # where μ(v₁,v₂) = v₁ + v₂.
        v_acc, a_acc = initial_pairs[0]
        if len(initial_pairs) > 1:
            for v_i, a_i in initial_pairs[1:]:
                v_next = v_acc + v_i
                a_next = (self.alpha_compose * a_acc +
                          self.beta_compose * a_i +
                          self.gamma_tensor * np.dot(v_acc, v_i))
                v_acc, a_acc = v_next, a_next

        # 4. If iter > 0, apply the endofunctor F(x) = x ⊗ x repeatedly.
        # This finds the fixed point for phrases like "very very happy",
        # modeled as F(F(happy_vec)).
        if iter > 0:
            # The prompt suggests a `while tol` loop; `iter` serves as a max_iter guard.
            for _ in range(iter):
                v_prev, a_prev = v_acc, a_acc

                # Apply F(x) = x ⊗ x, where x = (v_prev, a_prev)
                # v_new = v_prev + v_prev
                # a_new = α·a_prev + β·a_prev + γ·⟨v_prev, v_prev⟩
                v_acc = 2 * v_prev
                a_acc = (self.alpha_compose + self.beta_compose) * a_prev + \
                        self.gamma_tensor * np.dot(v_prev, v_prev)

                # Check for convergence against the previous state
                v_change = np.linalg.norm(v_acc - v_prev)
                a_change = abs(a_acc - a_prev)
                if v_change < tol and a_change < tol:
                    break
        
        # 5. Combine the final vector and affect score into a single array.
        return np.concatenate([v_acc, [a_acc]])

    def opposite(self, w: str) -> np.ndarray:
        # Same embedding, but affect dimension sign-flipped
        return np.concatenate([self.E[w], [-self.alpha * self.val[self.w2i[w]]]])
    def nearest(self, vec: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        """Return the ``n`` nearest neighbours of ``vec``.

        The previous implementation sorted the similarity against *all*
        words which becomes expensive for large vocabularies.  This version
        performs the search in a fully vectorised manner and uses
        ``numpy.argpartition`` to retrieve only the top ``n`` candidates.
        """
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            # If the query vector has zero norm (e.g., phrase with no known
            # tokens), return neutral similarities instead of NaNs.
            return [(w, 0.0) for w in self.words][:n]
        v = vec / norm
        sims = self._store_matrix @ v
        n = min(n, len(self.words))
        if n <= 0:
            return []
        idxs = np.argpartition(-sims, n - 1)[:n]
        idxs = idxs[np.argsort(-sims[idxs])]
        return [(self.words[i], float(sims[i])) for i in idxs]

    # ─────────── supervised ────────────────────────────────────────────
    def _train_supervised(self, VAD: Dict[str, List[float]], lam: float = 1e-3):
        """Train the affect projection using ridge regression.

        Uses ``np.linalg.solve`` instead of an explicit matrix inverse for
        improved numerical stability.
        """
        X = np.vstack([self.E[w] for w in VAD if w in self.E])
        Y = np.vstack([VAD[w] for w in VAD if w in self.E])
        A = X.T @ X + np.eye(self.d) * lam
        B = Y.T @ X
        W = np.linalg.solve(A.T, B.T).T
        full = np.vstack(list(self.E.values()))
        # Use only valence dimension (index 0) for augmentation
        self.val = W[0] @ full.T
        # Scale affect dimension relative to embedding norms
        self._scale()
        # For completeness, placeholders for certificate values
        n = len(self.words)
        self.L_plus = self.L_minus = sp.eye(n, dtype=np.float64)
        self.lambda_star = self.gamma = 1.0

    # ─────────── unsupervised (generalized eigen) ──────────────────────
    def _train_unsup(self, thr: float = 0.4, k: int = 5, seed: Optional[int] = None):
        # Build positive and negative edges
        pos, neg = build_edges(self.E, self.w2i, thr, k)
        n = len(self.words)
        rng = np.random.default_rng(seed)
        # Construct adjacency matrices for positive and negative edges
        rows_p, cols_p, data_p = zip(*pos) if pos else ([], [], [])
        rows_m, cols_m, data_m = zip(*neg) if neg else ([], [], [])
        S_plus = sp.coo_matrix((data_p, (rows_p, cols_p)), shape=(n, n), dtype=np.float64)
        S_minus = sp.coo_matrix((data_m, (rows_m, cols_m)), shape=(n, n), dtype=np.float64)
        # Symmetrize the edge matrices
        S_plus = S_plus + S_plus.T
        S_minus = S_minus + S_minus.T
        # Laplacians
        Dp = sp.diags(np.array(S_plus.sum(axis=1)).ravel())
        Dm = sp.diags(np.array(S_minus.sum(axis=1)).ravel())
        self.L_plus = (Dp - S_plus).astype(np.float64)
        self.L_minus = (Dm - S_minus).astype(np.float64)
        # Ensure L_minus is invertible by adding self-loop to isolated nodes
        eps = 1e-4
        zdiag = (self.L_minus.diagonal() == 0).astype(float)
        self.L_minus += sp.diags(zdiag * 1.0 + eps)
        try:
            if not HAS_SCIPY:
                raise RuntimeError("SciPy absent")
            # Compute two or three smallest generalized eigenvalues/vectors
            k_eigs = min(n - 1, 3)
            vals, vecs = spla.eigsh(self.L_plus, k=k_eigs, M=self.L_minus, which="SM", tol=1e-6, maxiter=500)
        except Exception as e:
            # Fallback: iterative generalized eigen solver for smallest two (or three) eigenpairs
            A = self.L_plus.toarray()
            B = self.L_minus.toarray()
            # First eigenpair
            x = rng.random(n)
            x = x / np.linalg.norm(x)
            lam1_est = None
            for it in range(1, 31):
                b = B.dot(x)
                y, *_ = np.linalg.lstsq(A, b, rcond=None)
                norm_y = np.linalg.norm(y)
                if norm_y < 1e-12:
                    break
                x = y / norm_y
                lam1_est = float(x.dot(A.dot(x)) / x.dot(B.dot(x)))
            lam1 = lam1_est if lam1_est is not None else 0.0
            e1 = x
            # Second eigenpair (B-orthogonal to e1)
            x2 = rng.random(n)
            e1_Bnorm = math.sqrt(e1.dot(B.dot(e1)))
            if e1_Bnorm < 1e-12:
                e1_Bnorm = 1e-12
            e1_u = e1 / e1_Bnorm
            x2 = x2 - e1_u * (e1_u.dot(B.dot(x2)))
            x2 = x2 / (np.linalg.norm(x2) + 1e-12)
            lam2_est = None
            for it in range(1, 31):
                b2 = B.dot(x2)
                y2, *_ = np.linalg.lstsq(A, b2, rcond=None)
                y2 = y2 - e1_u * (e1_u.dot(B.dot(y2)))
                norm_y2 = np.linalg.norm(y2)
                if norm_y2 < 1e-12:
                    break
                x2 = y2 / norm_y2
                x2 = x2 - e1_u * (e1_u.dot(B.dot(x2)))
                x2 = x2 / (np.linalg.norm(x2) + 1e-12)
                lam2_est = float(x2.dot(A.dot(x2)) / x2.dot(B.dot(x2)))
            lam2 = lam2_est if lam2_est is not None else lam1
            # Optionally compute third eigenpair if second found
            if lam2_est is not None:
                x3 = rng.random(n)
                e2 = x2
                e2_Bnorm = math.sqrt(e2.dot(B.dot(e2)))
                if e2_Bnorm < 1e-12:
                    e2_Bnorm = 1e-12
                e2_u = e2 / e2_Bnorm
                x3 = x3 - e1_u * (e1_u.dot(B.dot(x3))) - e2_u * (e2_u.dot(B.dot(x3)))
                x3 = x3 / (np.linalg.norm(x3) + 1e-12)
                lam3_est = None
                for it2 in range(1, 31):
                    b3 = B.dot(x3)
                    y3, *_ = np.linalg.lstsq(A, b3, rcond=None)
                    y3 = y3 - e1_u * (e1_u.dot(B.dot(y3))) - e2_u * (e2_u.dot(B.dot(y3)))
                    norm_y3 = np.linalg.norm(y3)
                    if norm_y3 < 1e-12:
                        break
                    x3 = y3 / norm_y3
                    x3 = x3 - e1_u * (e1_u.dot(B.dot(x3))) - e2_u * (e2_u.dot(B.dot(x3)))
                    x3 = x3 / (np.linalg.norm(x3) + 1e-12)
                    lam3_est = float(x3.dot(A.dot(x3)) / x3.dot(B.dot(x3)))
                lam3 = lam3_est if lam3_est is not None else lam2
                vals = np.array([lam1, lam2, lam3], dtype=float)
                vecs = np.vstack([e1, x2, x3]).T
            else:
                vals = np.array([lam1, lam2], dtype=float)
                vecs = np.vstack([e1, x2]).T
        # Select the eigenvector with maximum variance (non-trivial component)
        order = np.argsort(vals)
        best_idx = None
        best_std = -1.0
        for idx in order:
            v = vecs[:, idx].astype(float)
            stdv = np.std(v)
            if stdv > best_std:
                best_std = stdv
                best_idx = idx
        if best_idx is None:
            best_idx = order[0]
        self.lambda_star = float(vals[best_idx])
        # Compute spectral gap (difference to next eigenvalue)
        sorted_idx = sorted(order, key=lambda i: vals[i])
        pos = sorted_idx.index(best_idx)
        if pos < len(sorted_idx) - 1:
            next_idx = sorted_idx[pos + 1]
            self.gamma = float(vals[next_idx]) - self.lambda_star
        else:
            if pos > 0:
                prev_idx = sorted_idx[pos - 1]
                self.gamma = self.lambda_star - float(vals[prev_idx])
            else:
                self.gamma = 0.0
        # Center and normalize the chosen eigenvector
        vec = vecs[:, best_idx].astype(float)
        vec = vec - np.mean(vec)
        vec = vec / (np.std(vec) + 1e-12)
        self.val = vec
        # Refine antonym edges using initial val separation
        k_refine = min(n-2, 15) if n > 1 else 0
        neg_extra: List[Tuple[int, int, float]] = []
        # Build normalized embedding matrix for neighborhood search
        V = np.vstack([self.E[w] for w in self.words])
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        n_pos = np.sum(self.val > 0)
        n_neg = np.sum(self.val < 0)
        if n_pos > 0 and n_neg > 0:
            seen_pairs = set(tuple(sorted((i, j))) for i, j, _ in neg)
            for i in range(n):
                si = 1 if self.val[i] > 0 else (-1 if self.val[i] < 0 else 0)
                if si == 0:
                    continue
                sims = V.dot(V[i])
                num_nb = min(len(sims) - 1, k_refine + 1)
                idxs = np.argpartition(-sims, num_nb)[: num_nb]
                idxs = idxs[np.argsort(-sims[idxs])]
                for j in idxs:
                    if i == j:
                        continue
                    if sims[j] < thr:
                        break
                    sj = 1 if self.val[j] > 0 else (-1 if self.val[j] < 0 else 0)
                    if sj == 0:
                        continue
                    if si * sj < 0:
                        pair = (i, j) if i < j else (j, i)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            w = float(abs(sims[j]))
                            if w < thr:
                                w = thr
                            neg_extra.append((i, j, w))
                        break
        if neg_extra:
            # Recompute L_minus with added antonym edges
            neg_all = neg + neg_extra
            rows_m2, cols_m2, data_m2 = zip(*neg_all) if neg_all else ([], [], [])
            S_minus2 = sp.coo_matrix((data_m2, (rows_m2, cols_m2)), shape=(n, n), dtype=np.float64)
            S_minus2 = S_minus2 + S_minus2.T
            Dm2 = sp.diags(np.array(S_minus2.sum(axis=1)).ravel())
            L_minus2 = (Dm2 - S_minus2).astype(np.float64)
            zdiag2 = (L_minus2.diagonal() == 0).astype(float)
            L_minus2 += sp.diags(zdiag2 * 1.0 + 1e-4)
            # Solve generalized eigenproblem again with updated negative edges
            try:
                if not HAS_SCIPY:
                    raise RuntimeError("SciPy absent")
                k_eigs2 = min(n - 1, 3)
                vals2, vecs2 = spla.eigsh(self.L_plus, k=k_eigs2, M=L_minus2, which="SM", tol=1e-6, maxiter=500)
            except Exception as e:
                A2 = self.L_plus.toarray()
                B2 = L_minus2.toarray()
                x = rng.random(n)
                x = x / (np.linalg.norm(x) + 1e-12)
                lam1_est = None
                for it in range(1, 31):
                    b = B2.dot(x)
                    y, *_ = np.linalg.lstsq(A2, b, rcond=None)
                    norm_y = np.linalg.norm(y)
                    if norm_y < 1e-12:
                        break
                    x = y / norm_y
                    lam1_est = float(x.dot(A2.dot(x)) / x.dot(B2.dot(x)))
                lam1_2 = lam1_est if lam1_est is not None else 0.0
                e1_2 = x
                x2 = rng.random(n)
                e1_Bnorm2 = math.sqrt(e1_2.dot(B2.dot(e1_2)))
                if e1_Bnorm2 < 1e-12:
                    e1_Bnorm2 = 1e-12
                e1_u2 = e1_2 / e1_Bnorm2
                x2 = x2 - e1_u2 * (e1_u2.dot(B2.dot(x2)))
                x2 = x2 / (np.linalg.norm(x2) + 1e-12)
                lam2_est = None
                for it in range(1, 31):
                    b2 = B2.dot(x2)
                    y2, *_ = np.linalg.lstsq(A2, b2, rcond=None)
                    y2 = y2 - e1_u2 * (e1_u2.dot(B2.dot(y2)))
                    norm_y2 = np.linalg.norm(y2)
                    if norm_y2 < 1e-12:
                        break
                    x2 = y2 / norm_y2
                    x2 = x2 - e1_u2 * (e1_u2.dot(B2.dot(x2)))
                    x2 = x2 / (np.linalg.norm(x2) + 1e-12)
                    lam2_est = float(x2.dot(A2.dot(x2)) / x2.dot(B2.dot(x2)))
                lam2_2 = lam2_est if lam2_est is not None else lam1_2
                if lam2_est is not None:
                    x3 = rng.random(n)
                    e2 = x2
                    e2_Bnorm = math.sqrt(e2.dot(B2.dot(e2)))
                    if e2_Bnorm < 1e-12:
                        e2_Bnorm = 1e-12
                    e2_u = e2 / e2_Bnorm
                    x3 = x3 - e1_u2 * (e1_u2.dot(B2.dot(x3))) - e2_u * (e2_u.dot(B2.dot(x3)))
                    x3 = x3 / (np.linalg.norm(x3) + 1e-12)
                    lam3_est = None
                    for it2 in range(1, 31):
                        b3 = B2.dot(x3)
                        y3, *_ = np.linalg.lstsq(A2, b3, rcond=None)
                        y3 = y3 - e1_u2 * (e1_u2.dot(B2.dot(y3))) - e2_u * (e2_u.dot(B2.dot(y3)))
                        norm_y3 = np.linalg.norm(y3)
                        if norm_y3 < 1e-12:
                            break
                        x3 = y3 / norm_y3
                        x3 = x3 - e1_u2 * (e1_u2.dot(B2.dot(x3))) - e2_u * (e2_u.dot(B2.dot(x3)))
                        x3 = x3 / (np.linalg.norm(x3) + 1e-12)
                        lam3_est = float(x3.dot(A2.dot(x3)) / x3.dot(B2.dot(x3)))
                    lam3_2 = lam3_est if lam3_est is not None else lam2_2
                    vals2 = np.array([lam1_2, lam2_2, lam3_2], dtype=float)
                    vecs2 = np.vstack([e1_2, x2, x3]).T
                else:
                    vals2 = np.array([lam1_2, lam2_2], dtype=float)
                    vecs2 = np.vstack([e1_2, x2]).T
            # Select eigenvector with maximum variance from refined graph
            order2 = np.argsort(vals2)
            best_idx2 = None
            best_std2 = -1.0
            for idx in order2:
                v2 = vecs2[:, idx].astype(float)
                stdv2 = np.std(v2)
                if stdv2 > best_std2:
                    best_std2 = stdv2
                    best_idx2 = idx
            if best_idx2 is None:
                best_idx2 = order2[0]
            self.lambda_star = float(vals2[best_idx2])
            sorted_idx2 = sorted(order2, key=lambda i: vals2[i])
            pos2 = sorted_idx2.index(best_idx2)
            if pos2 < len(sorted_idx2) - 1:
                next_idx2 = sorted_idx2[pos2 + 1]
                self.gamma = float(vals2[next_idx2]) - self.lambda_star
            else:
                if pos2 > 0:
                    prev_idx2 = sorted_idx2[pos2 - 1]
                    self.gamma = self.lambda_star - float(vals2[prev_idx2])
                else:
                    self.gamma = 0.0
            vec2 = vecs2[:, best_idx2].astype(float)
            vec2 = vec2 - np.mean(vec2)
            vec2 = vec2 / (np.std(vec2) + 1e-12)
            self.val = vec2
            self.L_minus = L_minus2
        # Final scaling of valence dimension
        self._scale()

    # ─────────── certificate ───────────────────────────────────────────
    def certify(self, word: Optional[str] = None) -> Tuple[float, float]:
        z = self.val.copy()
        if word:
            z *= 0
            z[self.w2i[word]] = 1.0
        z /= (np.linalg.norm(z) + 1e-12)
        R = np.linalg.norm(self.L_plus.dot(z) - self.lambda_star * (self.L_minus.dot(z)))
        theta = math.degrees(math.asin(min(1.0, R / (self.gamma if self.gamma else 1.0))))
        return (R, theta)

    # ─────────── save / load ───────────────────────────────────────────
    def save(self, path: str):
        meta = dict(d=self.d, alpha=self.alpha,
                    val=self.val.tolist(),
                    lambda_star=self.lambda_star, gamma=self.gamma,
                    words=self.words)
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "wt", encoding="utf-8") as fh:
            json.dump(meta, fh)

    @classmethod
    def load(cls, meta_path: str, vec_path: str):
        op = gzip.open if meta_path.endswith(".gz") else open
        with op(meta_path, "rt", encoding="utf-8") as fh:
            meta = json.load(fh)
        keep = set(meta["words"])
        E = {w: v for w, v in read_vec(vec_path, keep)}
        obj = cls.__new__(cls)
        obj.E = E
        obj.d = meta["d"]
        obj.alpha = meta["alpha"]
        obj.val = np.array(meta["val"])
        obj.words = meta["words"]
        obj.w2i = {w: i for i, w in enumerate(obj.words)}
        obj.lambda_star = meta["lambda_star"]
        obj.gamma = meta["gamma"]
        n = len(obj.words)
        obj.L_plus = obj.L_minus = sp.eye(n, dtype=np.float64)
        obj.store = {w: obj.vector(w) / np.linalg.norm(obj.vector(w)) for w in obj.words}
        obj._store_matrix = np.vstack([obj.store[w] for w in obj.words])
        # Re-initialize composition parameters after loading
        mean_vec_norm = np.mean([np.linalg.norm(v) for v in obj.E.values()])
        mean_vec_norm = mean_vec_norm if mean_vec_norm > 0 else 1.0
        obj.alpha_compose = 0.5
        obj.beta_compose = 0.5
        obj.gamma_tensor = obj.lambda_star / (2 * mean_vec_norm)
        return obj

    def _scale(self):
        sem = np.mean([np.linalg.norm(v) for v in self.E.values()])
        self.alpha = sem / (np.std(self.val) + 1e-12)

###############################################################################
# ───────────────────────── script entry‑point ────────────────────────────────
###############################################################################

def _cli() -> None:
    """Original command-line interface for use with external files."""
    import argparse
    import pathlib

    p = argparse.ArgumentParser(
        description="Build empathic word‑embeddings, save the model and "
                    "run a couple of demo queries."
    )
    p.add_argument("vecs", type=pathlib.Path,
                   help="Path to .vec or .txt embeddings in word2vec format")
    p.add_argument("--vad", type=pathlib.Path, default=None,
                   help="Optional valence–arousal–dominance lexicon "
                        "(4‑column text file: word  V  A  D)")
    p.add_argument("--save", type=pathlib.Path, default="empathic.json.gz",
                   help="Where to write the trained model (default: %(default)s)")
    p.add_argument("--topk", type=int, default=5,
                   help="Words to show in the similarity demo (default: %(default)s)")
    p.add_argument("--probe", default="happy",
                   help="Probe word for demo queries (default: %(default)s)")
    args = p.parse_args()

    # ─── load embeddings and (optionally) VAD ──────────────────────────────
    E = dict(read_vec(args.vecs))
    VAD = load_vad(args.vad) if args.vad else None

    # ─── train the model ───────────────────────────────────────────────────
    print("⏳  Training space …")
    space = EmpathicSpace(E, gold=VAD)

    # ─── quick demo ────────────────────────────────────────────────────────
    if args.probe in E:
        print(f"\nNearest neighbours of “{args.probe}” in affect‑augmented space:")
        for w, s in space.nearest(space.vector(args.probe), n=args.topk):
            print(f"  {w:<15}  {s:5.3f}")
        print(f"\nOpposite‑affect neighbours of “{args.probe}”:")
        for w, s in space.nearest(space.opposite(args.probe), n=args.topk):
            print(f"  {w:<15}  {s:5.3f}")
        R, θ = space.certify(args.probe)
        print(f"\nCertificate for “{args.probe}”:  R={R:6.3e},  θ={θ:4.1f}°")
    else:
        print(f"⚠️  Probe word “{args.probe}” not in vocabulary – skipping demo.")

    # ─── persist the model ────────────────────────────────────────────────
    space.save(args.save)
    print(f"\n✔️  Model saved to {args.save}")


def main():
    """
    A self-contained main function to demonstrate the EmpathicSpace class.
    This function creates dummy data, runs both supervised and unsupervised
    training, and prints the results. It allows the script to be run
    directly without needing external files.
    """
    print("="*70)
    print(" Empathic Word Embeddings Demo")
    print("="*70)
    print("This script will now run a self-contained demonstration.")
    print("NOTE: This requires 'numpy' and 'scipy' to be installed.")
    print("You can install them by running: pip install numpy scipy\n")

    # --- 1. Create Dummy Data ---
    # We'll create some fake word vectors and VAD scores in memory.
    # In a real scenario, these would be loaded from large text files.

    # Dummy word vectors (word: vector)
    dummy_vectors_data = {
        "happy":      [0.9, 0.1, 0.1, 0.2, 0.3],
        "joyful":     [0.8, 0.2, 0.1, 0.1, 0.4],
        "sad":        [-0.8, 0.1, 0.9, 0.1, 0.2],
        "miserable":  [-0.9, 0.2, 0.8, 0.2, 0.1],
        "unhappy":    [-0.85, 0.15, 0.85, 0.15, 0.15], # Antonym of happy
        "computer":   [0.1, 0.9, 0.1, 0.8, 0.1],
        "data":       [0.2, 0.8, 0.2, 0.9, 0.2],
        "good":       [0.7, 0.3, 0.2, 0.3, 0.3],
        "bad":        [-0.7, 0.3, 0.7, 0.3, 0.3],
        "terrible":   [-0.8, 0.25, 0.75, 0.25, 0.25],
        "excellent":  [0.8, 0.25, 0.15, 0.25, 0.35],
        "neutral":    [0.0, 0.5, 0.5, 0.5, 0.5],
    }
    # Convert to numpy arrays, as required by the class
    E = {w: np.array(v, dtype=np.float32) for w, v in dummy_vectors_data.items()}

    # Dummy VAD (Valence, Arousal, Dominance) data for supervised training
    dummy_vad_data = {
        "happy":     [8.2, 6.4, 7.2],
        "joyful":    [8.5, 6.8, 6.9],
        "sad":       [2.1, 3.2, 3.0],
        "miserable": [1.8, 3.5, 2.8],
        "good":      [7.8, 5.0, 6.0],
        "bad":       [2.5, 4.0, 4.0],
    }

    # --- 2. Supervised Mode Demo ---
    print("\n--- Running SUPERVISED Mode Demo (using dummy VAD data) ---")
    try:
        # The EmpathicSpace class is initialized with the base embeddings (E)
        # and the gold-standard VAD scores.
        supervised_space = EmpathicSpace(E, gold=dummy_vad_data)
        print("✅  Supervised model trained successfully.")

        probe_word = "good"
        if probe_word in supervised_space.E:
            print(f"\nNearest neighbours of '{probe_word}' in affect-augmented space:")
            for w, s in supervised_space.nearest(supervised_space.vector(probe_word), n=5):
                print(f"  {w:<15}  {s:5.3f}")

            print(f"\nOpposite-affect neighbours of '{probe_word}':")
            for w, s in supervised_space.nearest(supervised_space.opposite(probe_word), n=5):
                print(f"  {w:<15}  {s:5.3f}")
        else:
            print(f"Probe word '{probe_word}' not in dummy vocabulary.")

    except Exception as e:
        print(f"❌  Supervised mode failed: {e}")


    # --- 3. Unsupervised Mode Demo ---
    print("\n\n--- Running UNSUPERVISED Mode Demo (no VAD data) ---")
    if not HAS_SCIPY:
        print("⚠️  SciPy not found. Unsupervised mode requires it for optimal performance.")
        print("   The script will use a slower numpy-based fallback.")
        print("   For better results, please install SciPy: pip install scipy")

    try:
        # The EmpathicSpace class is initialized with only the base embeddings (E).
        # The `gold` argument is omitted, triggering unsupervised training.
        unsupervised_space = EmpathicSpace(E, gold=None)
        print("✅  Unsupervised model trained successfully.")

        # The unsupervised mode finds a principal "affect" axis.
        # Let's see which words are at the positive and negative ends of this axis.
        # The `val` attribute stores the affect score for each word.
        sorted_by_affect = sorted(unsupervised_space.words, key=lambda w: unsupervised_space.val[unsupervised_space.w2i[w]])
        print("\nWords with most negative discovered affect:")
        for w in sorted_by_affect[:5]:
            score = unsupervised_space.val[unsupervised_space.w2i[w]]
            print(f"  {w:<15} {score:6.3f}")

        print("\nWords with most positive discovered affect:")
        for w in sorted_by_affect[-5:][::-1]:
            score = unsupervised_space.val[unsupervised_space.w2i[w]]
            print(f"  {w:<15} {score:6.3f}")

        probe_word = "good"
        if probe_word in unsupervised_space.E:
            print(f"\nNearest neighbours of '{probe_word}' in affect-augmented space:")
            for w, s in unsupervised_space.nearest(unsupervised_space.vector(probe_word), n=5):
                print(f"  {w:<15}  {s:5.3f}")

            print(f"\nOpposite-affect neighbours of '{probe_word}':")
            for w, s in unsupervised_space.nearest(unsupervised_space.opposite(probe_word), n=5):
                print(f"  {w:<15}  {s:5.3f}")

            R, theta = unsupervised_space.certify(probe_word)
            print(f"\nCertificate for '{probe_word}':  R={R:6.3e},  θ={theta:4.1f}°")
            
            # --- 4. Phrase Vector Demo ---
            print("\n--- Demonstrating new `phrase_vector` method ---")
            phrase1 = "bad happy"
            vec1 = unsupervised_space.phrase_vector(phrase1)
            print(f"\nVector for '{phrase1}': affect score = {vec1[-1]:.3f}")
            print(f"  -> Nearest neighbors for '{phrase1}':")
            for w, s in unsupervised_space.nearest(vec1, n=3):
                print(f"     {w:<15} {s:5.3f}")

            phrase2 = "happy"
            vec2_iter = unsupervised_space.phrase_vector(phrase2, iter=5)
            print(f"\nVector for 'very {phrase2}' (iter=5): affect score = {vec2_iter[-1]:.3f}")
            print(f"  -> Nearest neighbors for 'very {phrase2}':")
            for w, s in unsupervised_space.nearest(vec2_iter, n=3):
                print(f"     {w:<15} {s:5.3f}")


    except Exception as e:
        print(f"❌  Unsupervised mode failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n\n--- Demo Finished ---")
    print("The original command-line interface is still available in the `_cli()` function if you wish to use it with your own files.")


if __name__ == "__main__":
    # Invoke the command-line interface when executed as a script.
    _cli()
# ────────────────────────────── end of file ───────────────────────────────
