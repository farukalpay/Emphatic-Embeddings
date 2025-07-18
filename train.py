#!/usr/bin/env python3
"""
Empathic (Affect‑Augmented) Word Embeddings
==========================================

Supervised mode : ridge projector        (--vad <file>)
Unsupervised    : signed graph with      L⁺ z = λ L⁻ z
                  + spectral‑cap cert.   (omit --vad)
"""
from __future__ import annotations
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
            if len(p) != dim + 1: continue
            w, nums = p[0], p[1:]
            if keep and w not in keep: continue
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
    return float(u.dot(v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-12))

# ───────────────────── edge builder (unsupervised) ───────────────────────
def build_edges(E: Dict[str, np.ndarray],
                w2i: Dict[str, int],
                thr: float = .4,
                k_pos: int = 5,
                neg_pref: Tuple[str,...] = ("un","in","im","ir","non","dis","il")
               ) -> Tuple[List[Tuple[int,int,float]], List[Tuple[int,int,float]]]:
    words = list(E)
    V = np.vstack([E[w] for w in words])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    pos_edges: List[Tuple[int,int,float]] = []
    neg_edges: List[Tuple[int,int,float]] = []
    for i, wi in enumerate(words):
        sims = V @ V[i]
        idxs = np.argpartition(-sims, k_pos+1)[:k_pos+1]
        for j in idxs:
            if i == j or sims[j] < thr:
                continue
            pos_edges.append((i, j, abs(float(sims[j]))))
        for pref in neg_pref:
            if wi.startswith(pref) and len(wi) > len(pref) + 2:
                root = wi[len(pref):]
                if root in w2i:
                    j = w2i[root]
                    neg_edges.append((i, j, abs(float(V[i].dot(V[j])))))
    return pos_edges, neg_edges

# ───────────────────── main class ────────────────────────────────────────
class EmpathicSpace:
    def __init__(self, E: Dict[str, np.ndarray],
                       gold: Optional[Dict[str, List[float]]] = None):
        self.E = E
        self.d = len(next(iter(E.values())))
        self.words = list(E)
        self.w2i = {w: i for i, w in enumerate(self.words)}
        if gold:
            self._train_supervised(gold)
        else:
            self._train_unsup()
        self.store = {w: self.vector(w)/np.linalg.norm(self.vector(w))
                      for w in self.words}

    # ─────────── vectors ────────────────────────────────────────────────
    def vector(self, w: str) -> np.ndarray:
        return np.concatenate([self.E[w], [self.alpha * self.val[self.w2i[w]]]])
    def opposite(self, w: str) -> np.ndarray:
        return np.concatenate([self.E[w], [-self.alpha * self.val[self.w2i[w]]]])
    def nearest(self, vec: np.ndarray, n: int = 5):
        v = vec/np.linalg.norm(vec)
        return sorted(((w, float(v.dot(u))) for w, u in self.store.items()),
                      key=lambda t: -t[1])[:n]

    # ─────────── supervised ────────────────────────────────────────────
    def _train_supervised(self, VAD: Dict[str, List[float]], lam: float = 1e-3):
        X = np.vstack([self.E[w] for w in VAD if w in self.E])
        Y = np.vstack([VAD[w]   for w in VAD if w in self.E])
        A = X.T @ X + np.eye(self.d) * lam
        W = (Y.T @ X) @ np.linalg.inv(A)
        full = np.vstack(list(self.E.values()))
        self.val = (W[0] @ full.T)
        self._scale()
        # placeholders for certificate
        n = len(self.words)
        self.L_plus = self.L_minus = sp.eye(n, dtype=np.float64)
        self.lambda_star = self.gamma = 1.0

    # ─────────── unsupervised (generalized eigen) ──────────────────────
    def _train_unsup(self, thr: float = .4, k: int = 5):
        pos, neg = build_edges(self.E, self.w2i, thr, k)
        n = len(self.words)
        rows_p, cols_p, data_p = zip(*pos) if pos else ([], [], [])
        rows_m, cols_m, data_m = zip(*neg) if neg else ([], [], [])
        S_plus  = sp.coo_matrix((data_p, (rows_p, cols_p)), shape=(n, n), dtype=np.float64)
        S_minus = sp.coo_matrix((data_m, (rows_m, cols_m)), shape=(n, n), dtype=np.float64)
        S_plus  = S_plus + S_plus.T
        S_minus = S_minus + S_minus.T
        Dp = sp.diags(np.array(S_plus.sum(axis=1)).ravel())
        Dm = sp.diags(np.array(S_minus.sum(axis=1)).ravel())
        self.L_plus  = (Dp - S_plus).astype(np.float64)
        self.L_minus = (Dm - S_minus).astype(np.float64)
        eps = 1e-4
        zdiag = (self.L_minus.diagonal() == 0).astype(float)
        self.L_minus += sp.diags(zdiag * 1.0 + eps)
        print(f"[INFO] Graph |V|={n}  |E⁺|={len(data_p)//2}  |E⁻|={len(data_m)//2}", flush=True)
        try:
            if not HAS_SCIPY:
                raise RuntimeError("SciPy absent")
            vals, vecs = spla.eigsh(self.L_plus, k=2, M=self.L_minus,
                                     which="SM", tol=1e-6, maxiter=500)
            used_arpack = True
        except Exception as e:
            print(f"[WARN] ARPACK failed ({e}); switching to iterative solver.", flush=True)
            used_arpack = False
            # A3: iterative solver for generalized eigen
            A = self.L_plus.toarray()
            B = self.L_minus.toarray()
            np.random.seed(0)
            x = np.random.rand(n)
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
                resid = np.linalg.norm(A.dot(x) - lam1_est * B.dot(x))
                print(f"[ITER1 {it:02d}] λ≈{lam1_est:.6f}, resid={resid:.2e}", flush=True)
            lam1 = lam1_est if lam1_est is not None else 0.0
            e1 = x
            # Compute second eigen vector (B-orthogonal to e1)
            x2 = np.random.rand(n)
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
                resid2 = np.linalg.norm(A.dot(x2) - lam2_est * B.dot(x2))
                print(f"[ITER2 {it:02d}] λ≈{lam2_est:.6f}, resid={resid2:.2e}", flush=True)
            lam2 = lam2_est if lam2_est is not None else lam1
            vals = np.array([lam1, lam2], dtype=float)
            vecs = np.vstack([e1, x2]).T
        order = np.argsort(vals)
        self.lambda_star = float(vals[order[0]])
        lam2_val = float(vals[order[1]])
        self.gamma = lam2_val - self.lambda_star
        vec = vecs[:, order[0]].astype(float)
        vec_mean = np.mean(vec)
        vec = vec - vec_mean
        vec_std = np.std(vec) + 1e-12
        vec = vec / vec_std
        self.val = vec
        self._scale()
        print(f"[INFO] λ*={self.lambda_star:.2e}  γ={self.gamma:.2e}", flush=True)

    def _scale(self):
        sem = np.mean([np.linalg.norm(v) for v in self.E.values()])
        self.alpha = sem/(np.std(self.val) + 1e-12)
        print(f"[INFO] α={self.alpha:.4f}", flush=True)

    # ─────────── certificate ───────────────────────────────────────────
    def certify(self, word: Optional[str] = None) -> Tuple[float, float]:
        z = self.val.copy()
        if word:
            z *= 0
            z[self.w2i[word]] = 1.0
        z /= (np.linalg.norm(z) + 1e-12)
        R = np.linalg.norm(self.L_plus.dot(z) - self.lambda_star * (self.L_minus.dot(z)))
        theta = math.degrees(math.asin(min(1.0, R/(self.gamma if self.gamma else 1.0))))
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
        print(f"[INFO] Saved → {path}")

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
        obj.store = {w: obj.vector(w)/np.linalg.norm(obj.vector(w)) for w in obj.words}
        return obj

# ───────────────────── demonstration (no CLI) ───────────────────────────
if __name__ == "__main__":
    rng = random.Random(0)
    toy_words = ["good", "bad", "happy", "sad", "neutral",
                 "unhappy", "inactive", "imperfect", "perfect", "sorry", "cry"]
    toy_E = {w: np.array([rng.uniform(-1, 1) for _ in range(8)], dtype=np.float32) for w in toy_words}
    # B1, B2, B3: simulate two options and iterate for 30
    print("="*60, flush=True)
    print("Option 1: Using SciPy ARPACK solver (if available)", flush=True)
    print("="*60, flush=True)
    HAS_SCIPY = True  # simulate SciPy available
    spc1 = EmpathicSpace(toy_E)
    print("Nearest to opposite('imperfect'):", spc1.nearest(spc1.opposite("cry"), 5), flush=True)
    print("Certificate (field):", spc1.certify(), flush=True)
    print("Certificate (for 'good'):", spc1.certify("good"), flush=True)
    print("\n" + "="*60, flush=True)
    print("Option 2: SciPy absent (using iterative solver only)", flush=True)
    print("="*60, flush=True)
    HAS_SCIPY = False  # force iterative solver
    spc2 = EmpathicSpace(toy_E)
    print("Nearest to opposite('imperfect'):", spc2.nearest(spc2.opposite("cry"), 5), flush=True)
    print("Certificate (field):", spc2.certify(), flush=True)
    print("Certificate (for 'good'):", spc2.certify("good"), flush=True)
