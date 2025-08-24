def test_phrase_vector_methods(toy_model):
    vec_mean = toy_model.phrase_vector("good ugly", method="mean")
    vec_sif = toy_model.phrase_vector("good ugly", method="sif")
    vec_tfidf = toy_model.phrase_vector("good ugly", method="tfidf")
    assert vec_mean.shape[0] == toy_model.d + 1
    assert vec_sif.shape == vec_mean.shape
    assert vec_tfidf.shape == vec_mean.shape
