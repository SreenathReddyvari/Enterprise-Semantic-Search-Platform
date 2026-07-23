def get_health_status(engine) -> dict:
    index_ready = engine.store is not None and engine.store.size > 0
    return {
        "status": "healthy" if index_ready else "degraded",
        "index_ready": index_ready,
        "indexed_chunks": engine.store.size if engine.store else 0,
    }
