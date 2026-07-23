import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.schemas import HealthResponse, SearchRequest, SearchResponse
from src.database.db import init_db, log_search
from src.monitoring.health import get_health_status
from src.search.hybrid_search import HybridSearchEngine
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

engine = HybridSearchEngine()

DOCUMENTS_CSV = os.path.join("input", "documents.csv")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    engine.build_index(DOCUMENTS_CSV)
    logger.info("Search engine ready.")
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return get_health_status(engine)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    if request.mode == "semantic":
        raw_results = engine.semantic_search(request.query, top_k=request.top_k)
    elif request.mode == "hybrid":
        raw_results = engine.search(request.query, top_k=request.top_k)
    else:
        raise HTTPException(status_code=400, detail="mode must be 'semantic' or 'hybrid'")

    results = [
        {
            "rank": i + 1,
            "document": r["title"],
            "document_id": r["document_id"],
            "category": r["category"],
            "score": round(r["score"], 4),
            "text": r["text"],
        }
        for i, r in enumerate(raw_results)
    ]

    log_search(request.query, raw_results)

    return {"query": request.query, "mode": request.mode, "results": results}
