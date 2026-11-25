import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class VectorDocument(BaseModel):
    doc_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """In-memory dense vector store with metadata filtering for RAG pipelines."""

    def __init__(self) -> None:
        self._docs: Dict[str, VectorDocument] = {}

    def upsert(self, doc: VectorDocument) -> None:
        self._docs[doc.doc_id] = doc

    def search(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        scored: List[Tuple[VectorDocument, float]] = []
        for doc in self._docs.values():
            if filters and any(doc.metadata.get(k) != v for k, v in filters.items()):
                continue
            sim = _cosine_similarity(query_embedding, doc.embedding)
            scored.append((doc, round(sim, 6)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].doc_id))
        return scored[:top_k]
