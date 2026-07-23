from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.utils.config import settings

Base = declarative_base()


class SearchLog(Base):
    __tablename__ = "search_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    top_document = Column(String(255))
    top_score = Column(Float)
    result_count = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def log_search(query: str, results: list):
    session = SessionLocal()
    try:
        top = results[0] if results else None
        log = SearchLog(
            query=query,
            top_document=top["title"] if top else None,
            top_score=top["score"] if top else None,
            result_count=len(results),
        )
        session.add(log)
        session.commit()
    finally:
        session.close()
