import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from empathic_embeddings.utils import cosine


@given(
    st.integers(1, 5).flatmap(
        lambda n: st.tuples(
            hnp.arrays(dtype=np.float32, shape=n, elements=st.floats(-1.0, 1.0)),
            hnp.arrays(dtype=np.float32, shape=n, elements=st.floats(-1.0, 1.0)),
        )
    )
)
def test_cosine_symmetry(arrs):
    u, v = arrs
    assert np.isclose(cosine(u, v), cosine(v, u))


def test_neighbor_monotonicity(toy_model):
    res = toy_model.nearest_neighbors("good", k=2)
    assert res[0][1] >= res[1][1]
