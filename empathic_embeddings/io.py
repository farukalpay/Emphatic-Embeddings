"""Input/output utilities for Empathic Embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def read_vec(path: str | Path) -> Dict[str, np.ndarray]:
    """Read word vectors from a text ``.vec`` file.

    The file is expected to have one word per line: ``word v1 v2 ...``.
    """

    vecs: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().split()
        dim = int(first[1]) if len(first) == 2 and first[0].isdigit() else len(first) - 1
        if len(first) == dim + 1:
            word = first[0]
            vecs[word] = np.fromiter((float(x) for x in first[1:]), dtype=np.float32)
        else:
            f.seek(0)
        for ln in f:
            parts = ln.rstrip().split()
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            vec = np.fromiter((float(x) for x in parts[1:]), dtype=np.float32)
            vecs[word] = vec
    return vecs


def load_vad(path: str | Path) -> Dict[str, float]:
    """Load a simple valence lexicon from a whitespace separated file."""

    out: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            parts = ln.strip().split()
            if len(parts) == 4:
                out[parts[0]] = float(parts[1])
    return out


def save_model(path: str | Path, vocab: list[str], vectors: np.ndarray) -> None:
    """Save model vectors and vocabulary to ``path`` using ``numpy.savez``."""

    np.savez(path, vocab=vocab, vectors=vectors)


def load_model(path: str | Path) -> Tuple[list[str], np.ndarray]:
    """Load vectors saved by :func:`save_model`."""

    data = np.load(path, allow_pickle=True)
    return list(data["vocab"]), np.asarray(data["vectors"], dtype=np.float32)
