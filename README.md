# Enterprise Semantic Search Platform

AI-powered semantic search over enterprise documents (HR policies, IT policies, SOPs, etc.). Understands the *meaning* of a query instead of relying on exact keyword matches, using sentence embeddings, a vector index, and BM25 hybrid ranking.

## Problem

Employees waste time searching HR/IT/Finance documents with keyword search that fails when their wording doesn't match the document's wording. This platform lets them ask natural-language questions and get the most relevant document sections back, ranked by similarity.

## Architecture

```
Document Upload (CSV) -> Text Cleaning -> Chunking -> Embeddings (Sentence Transformers)
        -> Vector Store (FAISS / ChromaDB) -> Semantic Search
        -> BM25 Keyword Search -> Hybrid Ranking (weighted combination)
        -> FastAPI REST API -> Streamlit Dashboard
```

## Tech Stack

- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector DB:** FAISS (default), ChromaDB (optional, `VECTOR_BACKEND=chroma`)
- **Hybrid search:** BM25 (`rank-bm25`) + vector cosine similarity
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Metadata DB:** SQLite by default (`USE_SQLITE=true`), PostgreSQL-ready via `DATABASE_URL`
- **Tests:** pytest

## Project Structure

```
Enterprise Semantic Search Platform/
├── input/                      # sample documents.csv, sample_queries.csv, synonyms.csv
├── src/
│   ├── data/                   # loader.py, preprocess.py
│   ├── search/                 # chunking.py, embeddings.py, vector_store.py,
│   │                           # semantic_search.py, hybrid_search.py
│   ├── api/                    # app.py, schemas.py
│   ├── dashboard/              # streamlit_app.py
│   ├── database/               # db.py
│   ├── utils/                  # config.py, logger.py, helper.py
│   └── monitoring/              # health.py
├── vector_db/                  # persisted FAISS index / Chroma collection
├── output/                     # sqlite metadata db
├── tests/                      # test_api.py, test_embeddings.py, test_search.py
├── requirements.txt
├── .env
└── README.md
```

## Setup

```bash
python -m venv ../ess-venv
../ess-venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

## Run the API

```bash
../ess-venv/Scripts/python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8070 --reload
```

- Swagger docs: http://localhost:8070/docs
- Health check: http://localhost:8070/health

### Example request

```bash
curl -X POST http://localhost:8070/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the leave policy?", "top_k": 3, "mode": "hybrid"}'
```

```json
{
  "query": "What is the leave policy?",
  "mode": "hybrid",
  "results": [
    {
      "rank": 1,
      "document": "Employee Handbook",
      "document_id": "DOC001",
      "category": "HR",
      "score": 0.87,
      "text": "Employees are entitled to 20 annual leave days..."
    }
  ]
}
```

## Run the Dashboard

```bash
../ess-venv/Scripts/python.exe -m streamlit run src/dashboard/streamlit_app.py --server.port 8507
```

Open http://localhost:8507, type a query, choose semantic or hybrid mode, and view ranked results with similarity scores. Results can be downloaded as CSV.

## Tests

```bash
../ess-venv/Scripts/python.exe -m pytest tests/ -v
```

## Sample Data

`input/documents.csv` contains 4 sample enterprise documents (HR, IT, Finance). `input/sample_queries.csv` has example natural-language queries, and `input/synonyms.csv` shows the kind of term/synonym pairs semantic search resolves automatically without needing an explicit synonym table.

## Notes

- Ports (8070 for API, 8507 for Streamlit) are chosen to avoid clashing with other portfolio projects running locally.
- `USE_SQLITE=true` is the default for zero-setup local dev; switch to PostgreSQL by setting `DATABASE_URL` in `.env`.
- Hybrid search blends normalized vector cosine similarity and BM25 score using `HYBRID_ALPHA` (default `0.6` = 60% vector, 40% keyword).
