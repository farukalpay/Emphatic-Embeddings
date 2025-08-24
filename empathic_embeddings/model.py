"""Core model implementation for Empathic Embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

import numpy as np

from . import compose, io, utils
from .graph import build_edges, spectral_axis


@dataclass
class TrainConfig:
    """Configuration for training methods."""

    ridge_lambda: float = 1.0
    seed: int | None = None


class EmpathicEmbeddings:
    """Affect-augmented word embedding model.

    Parameters
    ----------
    vectors:
        Mapping from words to their base embedding vectors.
    config:
        Optional :class:`TrainConfig` controlling training behaviour.
    """

    def __init__(
        self, vectors: Mapping[str, np.ndarray], config: TrainConfig | None = None
    ):
        self.config = config or TrainConfig()
        self.vocab: List[str] = list(vectors)
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.base = np.vstack([vectors[w] for w in self.vocab]).astype(np.float32)
        self.d = self.base.shape[1]
        self.axis = np.zeros(len(self.vocab), dtype=np.float32)
        self.store = utils.normalize_rows(self.base)

    # ------------------------------------------------------------------
    # training
    def fit_supervised(self, lexicon: Mapping[str, float]) -> None:
        """Fit affect dimension using a lexical projection.

        ``lexicon`` maps words to valence scores in ``[0,1]``. A ridge regression
        is solved to project embedding vectors onto the valence axis.
        """

        idxs = [self.w2i[w] for w in lexicon.keys() if w in self.w2i]
        if not idxs:
            raise ValueError("none of the lexicon words were found in the vocabulary")
        X = self.base[idxs]
        y = np.array([lexicon[self.vocab[i]] for i in idxs], dtype=np.float32)
        X = np.column_stack([X, np.ones(len(idxs))])
        reg = self.config.ridge_lambda * np.eye(X.shape[1], dtype=np.float32)
        w = np.linalg.solve(X.T @ X + reg, X.T @ y)
        self.axis = np.column_stack([self.base, np.ones(len(self.vocab))]) @ w
        self.axis = (self.axis - self.axis.mean()) / (self.axis.std() + 1e-12)

    def fit_unsupervised(self, thr: float = 0.4, k: int = 5) -> None:
        """Infer affect axis using the signed-graph spectral method."""

        vectors = {w: self.base[i] for i, w in enumerate(self.vocab)}
        pos, neg = build_edges(vectors, self.w2i, thr=thr, k_pos=k)
        self.axis = spectral_axis(pos, neg, len(self.vocab)).astype(np.float32)

    # ------------------------------------------------------------------
    # inference helpers
    def transform(self, words: Sequence[str]) -> np.ndarray:
        """Return augmented vectors for ``words``."""

        rows = [self.w2i[w] for w in words]
        vecs = self.base[rows]
        affect = self.axis[rows][:, None]
        return np.hstack([vecs, affect])

    def vector(self, word: str) -> np.ndarray:
        return self.transform([word])[0]

    def nearest_neighbors(self, word: str, k: int = 5) -> List[tuple[str, float]]:
        """Return ``k`` nearest neighbors by cosine similarity."""

        if word not in self.w2i:
            raise KeyError(word)
        idx = self.w2i[word]
        q = self.store[idx]
        sims = self.store @ q
        order = np.argpartition(-sims, range(1, k + 1))[1 : k + 1]
        return [(self.vocab[i], float(sims[i])) for i in order]

    def opposite_neighbors(self, word: str, k: int = 5) -> List[tuple[str, float]]:
        """Return neighbours with opposite affect to ``word``."""

        if word not in self.w2i:
            raise KeyError(word)
        idx = self.w2i[word]
        sims = self.store @ self.store[idx]
        scores = sims * -np.sign(self.axis[idx]) * np.sign(self.axis)
        order = np.argpartition(-scores, range(1, k + 1))[1 : k + 1]
        return [(self.vocab[i], float(scores[i])) for i in order]

    def phrase_vector(self, text: str, method: str = "sif") -> np.ndarray:
        """Return a phrase embedding for ``text``.

        Parameters
        ----------
        text:
            Input phrase with tokens separated by whitespace.
        method:
            Composition method: ``"sif"`` (default), ``"tfidf"`` or ``"mean"``.

        Returns
        -------
        numpy.ndarray
            Composed phrase vector of dimensionality ``d + 1``.
        """

        tokens = [w for w in text.split() if w in self.w2i]
        if not tokens:
            return np.zeros(self.d + 1, dtype=np.float32)
        mat = self.transform(tokens)
        if method == "sif":
            return compose.sif_average(mat)
        if method == "tfidf":
            return compose.tfidf_average(mat)
        return mat.mean(axis=0)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        io.save_model(path, self.vocab, np.column_stack([self.base, self.axis]))

    @classmethod
    def load(cls, path: str | Path) -> "EmpathicEmbeddings":
        vocab, vecs = io.load_model(path)
        base = vecs[:, :-1]
        axis = vecs[:, -1]
        inst = cls({w: base[i] for i, w in enumerate(vocab)})
        inst.axis = axis
        return inst

    # ------------------------------------------------------------------
    def certificate(self, word: str) -> tuple[float, float]:
        """Return a rudimentary certificate ``(R, theta)`` for ``word``.

        This is a placeholder returning the valence score and 0 angle. A full
        spectral-cap certificate is outside the scope of this refactor.
        """

        if word not in self.w2i:
            raise KeyError(word)
        return float(self.axis[self.w2i[word]]), 0.0
