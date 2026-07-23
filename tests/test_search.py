import os

import pytest

from src.search.chunking import chunk_text
from src.search.hybrid_search import HybridSearchEngine

DOCUMENTS_CSV = os.path.join("input", "documents.csv")


def test_chunk_text_short_returns_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert chunks == [text]


def test_chunk_text_long_splits_with_overlap():
    words = [f"word{i}" for i in range(700)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # overlap check: last words of chunk 1 reappear at start of chunk 2
    assert chunks[0].split()[-1] in chunks[1].split()


@pytest.fixture(scope="module")
def engine():
    e = HybridSearchEngine()
    e.build_index(DOCUMENTS_CSV)
    return e


def test_build_index_creates_chunks(engine):
    assert engine.store.size > 0


def test_semantic_search_returns_relevant_doc(engine):
    results = engine.semantic_search("What is the leave policy?", top_k=3)
    assert len(results) > 0
    assert results[0]["document_id"] == "DOC001"


def test_hybrid_search_returns_relevant_doc(engine):
    results = engine.search("password change frequency", top_k=3)
    assert len(results) > 0
    assert results[0]["document_id"] == "DOC002"


def test_hybrid_search_respects_top_k(engine):
    results = engine.search("travel approval", top_k=2)
    assert len(results) <= 2
