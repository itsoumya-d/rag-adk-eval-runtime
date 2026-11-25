import pytest
from rag_eval.ingestion_pipeline import (
    AutomationIngestionPipeline,
    DatasetRecord,
    chunk_text,
    clean_dataset_text,
    hash_embed,
)
from rag_eval.retriever import VectorStore


def test_clean_dataset_text_strips_html_and_redacts_email():
    raw = "<div>  Hello   world!  </div>\n\n\nContact ops@example.com for access."
    cleaned = clean_dataset_text(raw, redact_emails=True)
    assert "<" not in cleaned
    assert "[REDACTED_EMAIL]" in cleaned
    assert "ops@example.com" not in cleaned


def test_chunk_text_respects_bounds_and_overlap():
    text = " ".join(f"token{i}" for i in range(40))
    chunks = chunk_text(text, max_chars=80, overlap=20)
    assert len(chunks) >= 3
    assert all(len(c) <= 80 for c in chunks)
    # Last token of chunk 0 overlaps into chunk 1
    assert chunks[0].split()[-1] in chunks[1].split()


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello world", max_chars=20, overlap=20)


def test_pipeline_deduplicates_and_retrieves_from_vector_store():
    store = VectorStore()
    pipeline = AutomationIngestionPipeline(store, max_chars=120, overlap=20)
    records = [
        DatasetRecord(
            record_id="r1",
            raw_text="<p>Automated invoice extraction using Python FastAPI webhooks.</p>",
            source="webhook",
            metadata={"tenant": "ops"},
        ),
        DatasetRecord(
            record_id="r2",
            raw_text="<p>Automated invoice extraction using Python FastAPI webhooks.</p>",
            source="webhook",
            metadata={"tenant": "ops"},
        ),
    ]
    report = pipeline.ingest(records)
    assert report.records_processed == 2
    assert report.chunks_created == 1
    assert report.duplicates_skipped == 1

    hits = store.search(hash_embed("Python FastAPI invoice webhooks"), top_k=1, filters={"tenant": "ops"})
    assert len(hits) == 1
    assert hits[0][0].doc_id == "r1#chunk-0"
