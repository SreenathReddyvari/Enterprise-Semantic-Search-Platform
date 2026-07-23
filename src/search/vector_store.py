import json
import os

import faiss
import numpy as np

from src.utils.config import settings


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms


class FaissVectorStore:
    """FAISS-backed store using inner product on L2-normalized vectors (== cosine similarity)."""

    def __init__(self, dim: int, persist_dir: str = None):
        self.dim = dim
        self.persist_dir = persist_dir or settings.VECTOR_DB_DIR
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, vectors: np.ndarray, metadata: list):
        vectors = _normalize(vectors.astype("float32"))
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list:
        query_vector = _normalize(query_vector.reshape(1, -1).astype("float32"))
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"score": float(score), **self.metadata[idx]})
        return results

    def save(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.persist_dir, "index.faiss"))
        with open(os.path.join(self.persist_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def load(self) -> bool:
        index_path = os.path.join(self.persist_dir, "index.faiss")
        meta_path = os.path.join(self.persist_dir, "metadata.json")
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return False
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        return True

    @property
    def size(self) -> int:
        return self.index.ntotal


class ChromaVectorStore:
    """ChromaDB-backed alternative vector store (same interface as FaissVectorStore)."""

    def __init__(self, dim: int = None, persist_dir: str = None, collection_name: str = "documents"):
        import chromadb

        self.persist_dir = persist_dir or settings.VECTOR_DB_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, vectors: np.ndarray, metadata: list):
        ids = [m["chunk_id"] for m in metadata]
        docs = [m["text"] for m in metadata]
        metas = [{k: v for k, v in m.items() if k != "text"} for m in metadata]
        self.collection.add(
            ids=ids, embeddings=vectors.tolist(), documents=docs, metadatas=metas
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list:
        result = self.collection.query(
            query_embeddings=[query_vector.tolist()], n_results=top_k
        )
        results = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            score = 1.0 - dist  # chroma default distance is cosine distance
            results.append({"score": float(score), "text": doc, **meta})
        return results

    def save(self):
        pass  # PersistentClient writes through automatically

    def load(self) -> bool:
        return self.collection.count() > 0

    @property
    def size(self) -> int:
        return self.collection.count()


def get_vector_store(dim: int) -> "FaissVectorStore":
    if settings.VECTOR_BACKEND == "chroma":
        return ChromaVectorStore(dim=dim)
    return FaissVectorStore(dim=dim)
