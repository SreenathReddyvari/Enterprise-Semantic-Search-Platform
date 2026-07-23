import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME = "Enterprise Semantic Search Platform"

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")  # "faiss" or "chroma"
    VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "vector_db")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))  # weight on vector score vs BM25

    USE_SQLITE = os.getenv("USE_SQLITE", "true").lower() == "true"
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "sqlite:///./output/metadata.db" if USE_SQLITE else "",
    )

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8070"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
