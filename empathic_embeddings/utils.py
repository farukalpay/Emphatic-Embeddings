"""Utility helpers for Empathic Embeddings.

The functions here are lightweight wrappers around common operations used
throughout the package. They are intentionally free of heavy dependencies so
that importing them has minimal side effects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration options for :class:`EmpathicEmbeddings`.

    Attributes
    ----------
    seed:
        Optional random seed to ensure deterministic behaviour.
    """

    seed: int | None = None


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a basic logging setup used by the package.

    Parameters
    ----------
    level:
        Logging level passed to :func:`logging.basicConfig`.
    """

    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Parameters
    ----------
    u, v:
        Input vectors of equal length.

    Returns
    -------
    float
        Cosine similarity ``u·v / (||u|| ||v||)``. ``0`` is returned when either
        vector has zero norm.
    """

    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)
    if un == 0 or vn == 0:
        return 0.0
    return float(u.dot(v) / (un * vn))


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Return a row-normalised copy of ``mat``.

    Zero rows are left untouched.
    """

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms
