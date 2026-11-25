import time
from typing import Callable, List, Optional
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    case_id: str
    prompt: str
    expected_keywords: List[str] = Field(default_factory=list)
    required_doc_ids: List[str] = Field(default_factory=list)
    max_latency_ms: float = 500.0
    max_cost_usd: float = 0.05


class AgentOutput(BaseModel):
    answer: str
    cited_doc_ids: List[str] = Field(default_factory=list)
    tool_calls: List[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class EvalCaseResult(BaseModel):
    case_id: str
    passed: bool
    grounding_score: float
    keyword_score: float
    latency_ms: float
    cost_usd: float
    failure_reasons: List[str] = Field(default_factory=list)


class EvalSuiteSummary(BaseModel):
    total_cases: int
    passed_cases: int
    pass_rate: float
    avg_grounding_score: float
    total_cost_usd: float
    results: List[EvalCaseResult]


class ProductionEvalHarness:
    """Regression evaluation runner for RAG and tool-calling agentic workflows."""

    def __init__(
        self,
        *,
        input_cost_per_1k: float = 0.0005,
        output_cost_per_1k: float = 0.0015,
        min_pass_rate: float = 1.0,
    ) -> None:
        self.input_cost_per_1k = input_cost_per_1k
        self.output_cost_per_1k = output_cost_per_1k
        self.min_pass_rate = min_pass_rate

    def _estimate_cost(self, in_tok: int, out_tok: int) -> float:
        return round((in_tok / 1000.0) * self.input_cost_per_1k + (out_tok / 1000.0) * self.output_cost_per_1k, 6)

    def evaluate_suite(
        self,
        cases: List[EvalCase],
        agent_fn: Callable[[str], AgentOutput],
    ) -> EvalSuiteSummary:
        results: List[EvalCaseResult] = []
        for case in cases:
            t0 = time.perf_counter()
            out = agent_fn(case.prompt)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            cost_usd = self._estimate_cost(out.input_tokens, out.output_tokens)

            req_docs = set(case.required_doc_ids)
            cited = set(out.cited_doc_ids)
            grounding_score = 1.0 if not req_docs else round(len(req_docs & cited) / len(req_docs), 4)

            ans_lower = out.answer.lower()
            kw_hits = sum(1 for kw in case.expected_keywords if kw.lower() in ans_lower)
            keyword_score = 1.0 if not case.expected_keywords else round(kw_hits / len(case.expected_keywords), 4)

            reasons: List[str] = []
            if grounding_score < 1.0:
                reasons.append(f"missing citations: {sorted(req_docs - cited)}")
            if keyword_score < 1.0:
                reasons.append(f"keyword recall {keyword_score:.2f} < 1.00")
            if latency_ms > case.max_latency_ms:
                reasons.append(f"latency {latency_ms}ms > {case.max_latency_ms}ms")
            if cost_usd > case.max_cost_usd:
                reasons.append(f"cost ${cost_usd} > ${case.max_cost_usd}")

            results.append(
                EvalCaseResult(
                    case_id=case.case_id,
                    passed=len(reasons) == 0,
                    grounding_score=grounding_score,
                    keyword_score=keyword_score,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    failure_reasons=reasons,
                )
            )

        passed_cnt = sum(1 for r in results if r.passed)
        total = len(results)
        return EvalSuiteSummary(
            total_cases=total,
            passed_cases=passed_cnt,
            pass_rate=round(passed_cnt / total, 4) if total else 0.0,
            avg_grounding_score=round(sum(r.grounding_score for r in results) / total, 4) if total else 0.0,
            total_cost_usd=round(sum(r.cost_usd for r in results), 6),
            results=results,
        )
