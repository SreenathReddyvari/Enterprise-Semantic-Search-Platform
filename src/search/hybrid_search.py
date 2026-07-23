import numpy as np
from rank_bm25 import BM25Okapi

from src.search.semantic_search import SemanticSearchEngine
from src.utils.config import settings


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-10:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridSearchEngine(SemanticSearchEngine):
    def __init__(self):
        super().__init__()
        self.bm25 = None
        self.tokenized_corpus = []

    def build_index(self, documents_csv: str):
        super().build_index(documents_csv)
        self.tokenized_corpus = [c.text.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def semantic_search(self, query: str, top_k: int = 5) -> list:
        return SemanticSearchEngine.search(self, query, top_k=top_k)

    def search(self, query: str, top_k: int = 5, alpha: float = None) -> list:
        alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
        if self.bm25 is None:
            raise RuntimeError("Hybrid index not built. Call build_index() first.")

        vector_results = super().search(query, top_k=self.store.size)
        vector_scores_by_id = {r["chunk_id"]: r["score"] for r in vector_results}

        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_scores_norm = _minmax_normalize(bm25_scores)

        combined = []
        for i, chunk in enumerate(self.chunks):
            v_score = vector_scores_by_id.get(chunk.chunk_id, 0.0)
            b_score = float(bm25_scores_norm[i])
            final_score = alpha * v_score + (1 - alpha) * b_score
            combined.append(
                {
                    "score": final_score,
                    "vector_score": v_score,
                    "bm25_score": b_score,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "category": chunk.category,
                    "text": chunk.text,
                }
            )

        combined.sort(key=lambda r: r["score"], reverse=True)
        return combined[:top_k]
