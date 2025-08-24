import numpy as np
import pytest

from empathic_embeddings.model import EmpathicEmbeddings


@pytest.fixture
def toy_vectors() -> dict[str, np.ndarray]:
    return {
        "good": np.array([1.0, 0.0], dtype=np.float32),
        "bad": np.array([0.0, 1.0], dtype=np.float32),
        "ugly": np.array([1.0, 1.0], dtype=np.float32),
    }


@pytest.fixture
def toy_model(toy_vectors):
    lex = {"good": 1.0, "bad": 0.0}
    model = EmpathicEmbeddings(toy_vectors)
    model.fit_supervised(lex)
    return model
