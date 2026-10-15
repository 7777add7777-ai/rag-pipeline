"""Offline evaluation: retrieval metrics and answer checks.

Golden dataset format: a list of :class:`GoldenCase` — a query, the chunk
ids that a correct answer must retrieve, and (optionally) expected answer
substrings. Everything runs offline: no API keys, no network, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .retriever import Retriever
from .vector_store import RankedChunk


@dataclass(frozen=True)
class GoldenCase:
    """One eval case: a query plus what 'correct' looks like."""

    query: str
    expected_chunk_ids: tuple[str, ...]
    expected_answer_contains: tuple[str, ...] = ()


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant chunks found in the top-k."""
    if not relevant:
        return 0.0
    hits = len(relevant & set(ranked_ids[:k]))
    return hits / len(relevant)


def mrr(ranked_ids: list[str], relevant: set[str]) -> float:
    """Mean reciprocal rank: 1 / (rank of first hit), 0 if no hit."""
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


@dataclass
class RetrievalReport:
    """Aggregated retrieval metrics over a golden dataset."""

    n_cases: int = 0
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate: float = 0.0  # fraction of cases with at least one relevant hit
    per_case: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        recall_str = " ".join(f"R@{k}={v:.2f}" for k, v in sorted(self.recall_at_k.items()))
        return (
            f"cases={self.n_cases}  {recall_str}  "
            f"MRR={self.mrr:.3f}  hit@{max(self.recall_at_k, default=0)}={self.hit_rate:.2f}"
        )


@dataclass
class AnswerReport:
    """Aggregated answer checks (substring match against golden answers)."""

    n_cases: int = 0
    exact_hits: int = 0  # cases where every expected substring was found
    hit_rate: float = 0.0
    per_case: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return f"cases={self.n_cases}  answer_hit={self.hit_rate:.2f}"


def eval_retrieval(
    retriever: Retriever,
    cases: list[GoldenCase],
    ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
) -> RetrievalReport:
    """Run every golden case through the retriever and aggregate metrics."""
    if not cases:
        raise ValueError("no golden cases")
    report = RetrievalReport(n_cases=len(cases))
    total_mrr = 0.0
    hits = 0
    recall_acc: dict[int, float] = {k: 0.0 for k in ks}

    for case in cases:
        ranked: list[RankedChunk] = retriever.search(case.query, top_k=top_k)
        ranked_ids = [rc.chunk.id for rc in ranked]
        relevant = set(case.expected_chunk_ids)
        row = {
            "query": case.query,
            "ranked_ids": ranked_ids,
            "expected": sorted(relevant),
        }
        for k in ks:
            r = recall_at_k(ranked_ids, relevant, k)
            recall_acc[k] += r
            row[f"recall@{k}"] = r
        rank = mrr(ranked_ids, relevant)
        total_mrr += rank
        row["mrr"] = rank
        if rank > 0:
            hits += 1
        report.per_case.append(row)

    report.recall_at_k = {k: v / len(cases) for k, v in recall_acc.items()}
    report.mrr = total_mrr / len(cases)
    report.hit_rate = hits / len(cases)
    return report


def eval_answers(
    answers: list[tuple[str, GoldenCase]],
) -> AnswerReport:
    """Check generated answers against golden substrings (case-insensitive)."""
    if not answers:
        raise ValueError("no answers")
    report = AnswerReport(n_cases=len(answers))
    for text, case in answers:
        if not case.expected_answer_contains:
            continue
        low = text.lower()
        missing = [s for s in case.expected_answer_contains if s.lower() not in low]
        ok = not missing
        if ok:
            report.exact_hits += 1
        report.per_case.append(
            {"query": case.query, "ok": ok, "missing": missing, "answer": text}
        )
    checked = [r for r in report.per_case if "ok" in r]
    report.hit_rate = (
        report.exact_hits / len(checked) if checked else 0.0
    )
    return report
