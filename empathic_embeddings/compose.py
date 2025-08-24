"""Phrase composition utilities.

This module provides simple compositional schemes used by
``EmpathicEmbeddings.phrase_vector``.  The implementations are purposely
minimal and operate purely on ``numpy`` arrays to keep the package light.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def sif_average(
    mat: np.ndarray, freqs: Sequence[int] | None = None, a: float = 1e-3
) -> np.ndarray:
    """Smooth inverse frequency weighted average.

    Parameters
    ----------
    mat:
        Matrix of word vectors with shape ``(n_words, n_dims)``.
    freqs:
        Optional corpus frequencies for the words. If omitted, uniform
        weighting is used.
    a:
        Smoothing parameter controlling the strength of the weighting.

    Returns
    -------
    numpy.ndarray
        The weighted average vector.
    """

    if mat.size == 0:
        return mat
    if freqs is None:
        return mat.mean(axis=0)
    freqs_arr = np.asarray(freqs, dtype=np.float32)
    weights = a / (a + freqs_arr)
    weights = weights[:, None]
    return (weights * mat).sum(axis=0) / weights.sum()


def tfidf_average(mat: np.ndarray, idf: Sequence[float] | None = None) -> np.ndarray:
    """TF‑IDF weighted average.

    Parameters
    ----------
    mat:
        Matrix of word vectors with shape ``(n_words, n_dims)``.
    idf:
        Optional inverse document frequency weights. If ``None`` uniform
        averaging is performed.

    Returns
    -------
    numpy.ndarray
        The weighted average vector.
    """

    if mat.size == 0:
        return mat
    if idf is None:
        return mat.mean(axis=0)
    weights = np.asarray(idf, dtype=np.float32)[:, None]
    return (mat * weights).sum(axis=0) / weights.sum()
