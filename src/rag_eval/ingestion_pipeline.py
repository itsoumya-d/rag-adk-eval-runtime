import hashlib
import math
import re
from typing import Any, Callable, Dict, List, Set
from pydantic import BaseModel, Field
from rag_eval.retriever import VectorDocument, VectorStore


class DatasetRecord(BaseModel):
    record_id: str
    raw_text: str
    source: str = "api"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionReport(BaseModel):
    records_processed: int
    chunks_created: int
    duplicates_skipped: int
    indexed_doc_ids: List[str]


_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def clean_dataset_text(raw_text: str, *, redact_emails: bool = True) -> str:
    """Strip markup, normalize whitespace, and optionally mask email PII."""
    text = _HTML_RE.sub(" ", raw_text)
    if redact_emails:
        text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    cleaned = "\n".join(ln for ln in lines if ln)
    return _NL_RE.sub("\n\n", cleaned).strip()


def chunk_text(text: str, *, max_chars: int = 220, overlap: int = 40) -> List[str]:
    """Split cleaned text into overlapping chunks without splitting words."""
    if max_chars <= overlap:
        raise ValueError("max_chars must exceed overlap")
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        buf: List[str] = []
        idx = start
        while idx < len(words) and (sum(len(w) + 1 for w in buf) + len(words[idx]) <= max_chars or not buf):
            buf.append(words[idx])
            idx += 1
        chunks.append(" ".join(buf))
        if idx >= len(words):
            break
        # Backtrack by approximate overlap character count
        back = 0
        back_words = 0
        for w in reversed(buf):
            if back + len(w) + 1 > overlap:
                break
            back += len(w) + 1
            back_words += 1
        start = max(start + 1, idx - back_words)
    return chunks


def hash_embed(text: str, dims: int = 16) -> List[float]:
    """Deterministic lightweight bag-of-trigrams embedding for prototyping."""
    vec = [0.0] * dims
    low = text.lower()
    for i in range(max(1, len(low) - 2)):
        gram = low[i : i + 3]
        bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % dims
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [round(v / norm, 6) for v in vec] if norm else vec


class AutomationIngestionPipeline:
    """End-to-end ingestion workflow: clean -> chunk -> dedup -> embed -> upsert."""

    def __init__(
        self,
        store: VectorStore,
        *,
        embed_fn: Callable[[str], List[float]] = hash_embed,
        max_chars: int = 220,
        overlap: int = 40,
        redact_emails: bool = True,
    ) -> None:
        self.store = store
        self.embed_fn = embed_fn
        self.max_chars = max_chars
        self.overlap = overlap
        self.redact_emails = redact_emails
        self._seen_hashes: Set[str] = set()

    def ingest(self, records: List[DatasetRecord]) -> IngestionReport:
        chunks_created = 0
        duplicates_skipped = 0
        doc_ids: List[str] = []

        for rec in records:
            cleaned = clean_dataset_text(rec.raw_text, redact_emails=self.redact_emails)
            chunks = chunk_text(cleaned, max_chars=self.max_chars, overlap=self.overlap)
            for idx, chunk in enumerate(chunks):
                fp = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                if fp in self._seen_hashes:
                    duplicates_skipped += 1
                    continue
                self._seen_hashes.add(fp)
                doc_id = f"{rec.record_id}#chunk-{idx}"
                meta = {**rec.metadata, "source": rec.source, "chunk_index": idx}
                self.store.upsert(
                    VectorDocument(
                        doc_id=doc_id,
                        content=chunk,
                        embedding=self.embed_fn(chunk),
                        metadata=meta,
                    )
                )
                chunks_created += 1
                doc_ids.append(doc_id)

        return IngestionReport(
            records_processed=len(records),
            chunks_created=chunks_created,
            duplicates_skipped=duplicates_skipped,
            indexed_doc_ids=doc_ids,
        )
