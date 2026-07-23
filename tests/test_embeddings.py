from src.search.embeddings import embed_query, embed_texts


def test_embed_texts_returns_matrix():
    vectors = embed_texts(["hello world", "goodbye world"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0


def test_embed_query_returns_vector():
    vector = embed_query("hello world")
    assert vector.ndim == 1
    assert vector.shape[0] > 0
