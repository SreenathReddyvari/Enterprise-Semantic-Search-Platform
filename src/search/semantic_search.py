from src.data.loader import load_documents
from src.data.preprocess import preprocess_documents
from src.search.chunking import chunk_documents
from src.search.embeddings import embed_query, embed_texts
from src.search.vector_store import get_vector_store
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticSearchEngine:
    def __init__(self):
        self.store = None
        self.chunks = []

    def build_index(self, documents_csv: str):
        df = load_documents(documents_csv)
        df = preprocess_documents(df)
        self.chunks = chunk_documents(df)

        if not self.chunks:
            raise ValueError("No chunks generated from documents")

        texts = [c.text for c in self.chunks]
        vectors = embed_texts(texts)

        self.store = get_vector_store(dim=vectors.shape[1])
        metadata = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "title": c.title,
                "category": c.category,
                "text": c.text,
            }
            for c in self.chunks
        ]
        self.store.add(vectors, metadata)
        self.store.save()
        logger.info(f"Indexed {len(self.chunks)} chunks from {documents_csv}")

    def load_index(self, dim: int) -> bool:
        self.store = get_vector_store(dim=dim)
        return self.store.load()

    def search(self, query: str, top_k: int = 5) -> list:
        if self.store is None or self.store.size == 0:
            raise RuntimeError("Vector index is empty. Build the index first.")
        query_vector = embed_query(query)
        return self.store.search(query_vector, top_k=top_k)
