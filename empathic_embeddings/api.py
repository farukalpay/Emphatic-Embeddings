"""High level API for Empathic Embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .model import EmpathicEmbeddings, TrainConfig


def load(path: str | Path) -> EmpathicEmbeddings:
    """Load a saved model.

    Parameters
    ----------
    path:
        Path to the model artifact previously written with
        :meth:`EmpathicEmbeddings.save`.

    Returns
    -------
    EmpathicEmbeddings
        Loaded model instance.
    """

    return EmpathicEmbeddings.load(path)


def train_supervised(
    vectors: Mapping[str, np.ndarray],
    lexicon: Mapping[str, float],
    config: TrainConfig | None = None,
) -> EmpathicEmbeddings:
    """Create and train a model in supervised mode.

    Parameters
    ----------
    vectors:
        Mapping of vocabulary items to their base embedding vectors.
    lexicon:
        Mapping of words to valence scores in ``[0, 1]``.
    config:
        Optional training configuration.

    Returns
    -------
    EmpathicEmbeddings
        Trained model instance.
    """

    model = EmpathicEmbeddings(vectors, config=config)
    model.fit_supervised(lexicon)
    return model


def train_unsupervised(
    vectors: Mapping[str, np.ndarray], config: TrainConfig | None = None
) -> EmpathicEmbeddings:
    """Create and train a model in unsupervised mode.

    Parameters
    ----------
    vectors:
        Mapping of vocabulary items to embedding vectors.
    config:
        Optional training configuration.

    Returns
    -------
    EmpathicEmbeddings
        Trained model instance.
    """

    model = EmpathicEmbeddings(vectors, config=config)
    model.fit_unsupervised()
    return model


def query(
    model: EmpathicEmbeddings, word: str, topk: int = 5
) -> list[tuple[str, float]]:
    """Return nearest neighbours of ``word``.

    Parameters
    ----------
    model:
        Fitted :class:`EmpathicEmbeddings` instance.
    word:
        Query word.
    topk:
        Number of neighbours to return.

    Returns
    -------
    list[tuple[str, float]]
        Top ``k`` neighbours with cosine similarity scores.
    """

    return model.nearest_neighbors(word, k=topk)
