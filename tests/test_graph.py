from empathic_embeddings.graph import build_edges, spectral_axis


def test_build_edges(toy_vectors):
    w2i = {w: i for i, w in enumerate(toy_vectors)}
    pos, neg = build_edges(toy_vectors, w2i, thr=0.0, k_pos=1)
    assert pos
    axis = spectral_axis(pos, neg, len(toy_vectors))
    assert axis.shape[0] == len(toy_vectors)
