from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "hybrid"  # "semantic" or "hybrid"


class SearchResult(BaseModel):
    rank: int
    document: str
    document_id: str
    category: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    indexed_chunks: int
