from rag_eval.agent_eval import AgentOutput, EvalCase, ProductionEvalHarness
from rag_eval.retriever import VectorDocument, VectorStore


def _build_store() -> VectorStore:
    store = VectorStore()
    store.upsert(VectorDocument(doc_id="doc-gcp", content="Vertex AI deployment guide", embedding=[0.9, 0.1, 0.0], metadata={"domain": "cloud"}))
    store.upsert(VectorDocument(doc_id="doc-rag", content="Vector index chunking policy", embedding=[0.1, 0.9, 0.0], metadata={"domain": "rag"}))
    store.upsert(VectorDocument(doc_id="doc-sec", content="VPC service controls perimeter", embedding=[0.85, 0.15, 0.0], metadata={"domain": "security"}))
    return store


def test_vector_store_cosine_ranking_and_metadata_filter():
    store = _build_store()
    hits = store.search([1.0, 0.0, 0.0], top_k=2)
    assert [doc.doc_id for doc, _ in hits] == ["doc-gcp", "doc-sec"]

    sec_only = store.search([1.0, 0.0, 0.0], top_k=2, filters={"domain": "security"})
    assert len(sec_only) == 1
    assert sec_only[0][0].doc_id == "doc-sec"


def test_eval_harness_passes_grounded_rag_agent():
    store = _build_store()
    harness = ProductionEvalHarness()

    def adk_agent(prompt: str) -> AgentOutput:
        top = store.search([0.9, 0.1, 0.0], top_k=1)
        doc, _ = top[0]
        return AgentOutput(
            answer=f"Use {doc.content} with Vertex AI pipeline monitoring.",
            cited_doc_ids=[doc.doc_id],
            tool_calls=["vector_search"],
            input_tokens=400,
            output_tokens=120,
        )

    summary = harness.evaluate_suite(
        [EvalCase(case_id="c1", prompt="How do we deploy on Vertex AI?", expected_keywords=["Vertex AI", "monitoring"], required_doc_ids=["doc-gcp"])],
        adk_agent,
    )
    assert summary.passed_cases == 1
    assert summary.pass_rate == 1.0
    assert summary.avg_grounding_score == 1.0


def test_eval_harness_flags_ungrounded_regression():
    harness = ProductionEvalHarness()

    def broken_agent(_prompt: str) -> AgentOutput:
        return AgentOutput(answer="Deploy manually.", cited_doc_ids=[], input_tokens=200, output_tokens=50)

    summary = harness.evaluate_suite(
        [EvalCase(case_id="c2", prompt="Explain chunking", expected_keywords=["chunking"], required_doc_ids=["doc-rag"])],
        broken_agent,
    )
    assert summary.passed_cases == 0
    assert any("missing citations" in r for r in summary.results[0].failure_reasons)
    assert any("keyword recall" in r for r in summary.results[0].failure_reasons)


def test_eval_harness_detects_token_cost_budget_violation():
    harness = ProductionEvalHarness(input_cost_per_1k=0.01, output_cost_per_1k=0.03)

    def expensive_agent(_prompt: str) -> AgentOutput:
        return AgentOutput(answer="Grounded answer", cited_doc_ids=["doc-gcp"], input_tokens=5000, output_tokens=2000)

    summary = harness.evaluate_suite(
        [EvalCase(case_id="c3", prompt="Summarize", required_doc_ids=["doc-gcp"], max_cost_usd=0.05)],
        expensive_agent,
    )
    assert summary.passed_cases == 0
    assert any("cost $" in r for r in summary.results[0].failure_reasons)
