from dataclasses import dataclass

import pandas as pd

from src.utils.config import settings


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    category: str
    text: str


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(df: pd.DataFrame) -> list:
    chunks = []
    for _, row in df.iterrows():
        pieces = chunk_text(row["Content"])
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{row['Document_ID']}_C{i}",
                    document_id=row["Document_ID"],
                    title=row["Title"],
                    category=row["Category"],
                    text=piece,
                )
            )
    return chunks
