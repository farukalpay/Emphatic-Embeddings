from empathic_embeddings import EmpathicEmbeddings
from empathic_embeddings.api import query


def test_vector_shape(toy_model):
    vec = toy_model.vector("good")
    assert vec.shape[0] == toy_model.d + 1


def test_query_helper(toy_model):
    res = query(toy_model, "good", topk=1)
    assert len(res) == 1
