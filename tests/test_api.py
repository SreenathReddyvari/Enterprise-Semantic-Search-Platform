import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["index_ready"] is True
    assert body["indexed_chunks"] > 0


def test_search_endpoint_hybrid(client):
    response = client.post("/search", json={"query": "leave policy", "top_k": 3, "mode": "hybrid"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "leave policy"
    assert len(body["results"]) > 0
    assert body["results"][0]["document_id"] == "DOC001"


def test_search_endpoint_semantic(client):
    response = client.post("/search", json={"query": "password rules", "top_k": 3, "mode": "semantic"})
    assert response.status_code == 200
    assert response.json()["results"][0]["document_id"] == "DOC002"


def test_search_endpoint_rejects_empty_query(client):
    response = client.post("/search", json={"query": "   ", "top_k": 3})
    assert response.status_code == 400


def test_search_endpoint_rejects_invalid_mode(client):
    response = client.post("/search", json={"query": "test", "mode": "invalid"})
    assert response.status_code == 400
