"""Offline demo: ingest a small policy KB, answer questions, run the eval."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .documents import Document
from .embeddings import HashEmbedder
from .evaluation import GoldenCase, eval_answers, eval_retrieval
from .generator import TemplateGenerator
from .pipeline import RAGPipeline

KB = {
    "refund": """
Acme Support Policy: Refunds.

Customers may request a full refund within 30 days of purchase for
defective products. Defects must be reported with proof of purchase.
Gold members receive refunds to their original payment method within
3-5 business days; standard members within 5-10 business days.
Refund requests older than 30 days are converted to repair or
replacement offers.
""",
    "shipping": """
Acme Support Policy: Shipping.

Standard shipping is free for orders over 50 dollars and takes 5-7
business days. Express shipping costs 9.99 dollars and takes 1-2
business days. International shipping is available to 40 countries
and takes 10-15 business days. Shipped items cannot be refunded,
only exchanged, unless the carrier damaged the package.
""",
    "warranty": """
Acme Support Policy: Warranty.

All products carry a 1 year warranty against manufacturing defects.
Warranty claims require the original receipt and a description of the
fault. Warranty service includes repair or replacement at no cost.
Consumables such as filters and batteries are excluded from warranty
coverage. Warranty does not cover cosmetic damage or misuse.
""",
    "returns": """
Acme Support Policy: Returns.

Unused items in original packaging may be returned within 60 days for
store credit. Return shipping is paid by the customer unless the item
arrived damaged. Store credit is issued within 3 business days of the
return arriving at the warehouse. Final sale items cannot be returned
under any circumstances.
""",
}

GOLDEN = [
    GoldenCase(
        query="How many days do I have to request a refund for a defective kettle?",
        expected_chunk_ids=("refund#0",),
        expected_answer_contains=("30 days",),
    ),
    GoldenCase(
        query="What does express shipping cost and how fast is it?",
        expected_chunk_ids=("shipping#0",),
        expected_answer_contains=("9.99", "1-2"),
    ),
    GoldenCase(
        query="Does the warranty cover batteries and filters?",
        expected_chunk_ids=("warranty#0",),
        expected_answer_contains=("excluded",),
    ),
    GoldenCase(
        query="Can I get store credit when returning an unused item?",
        expected_chunk_ids=("returns#0",),
        expected_answer_contains=("store credit",),
    ),
]


def build_pipeline() -> RAGPipeline:
    docs = [
        Document(id=topic, text=text.strip(), source=f"kb/{topic}")
        for topic, text in KB.items()
    ]
    pipeline = RAGPipeline(
        HashEmbedder(), TemplateGenerator(), top_k=3, max_chunk_chars=600, overlap_chars=80
    )
    pipeline.ingest(docs)
    return pipeline


def run_demo() -> int:
    console = Console(width=120, no_color=True)
    pipeline = build_pipeline()
    console.rule("[bold cyan]rag-pipeline — offline RAG demo")
    console.print(f"Ingested {pipeline.n_documents} documents -> {len(pipeline.store)} chunks")
    console.print()

    console.print(Panel.fit(f"[bold]Q: {GOLDEN[0].query}"))
    ans = pipeline.answer(GOLDEN[0].query)
    console.print(ans.answer)
    console.print()

    console.print("[bold]Retrieval eval (golden dataset, 4 cases):[/bold]")
    report = eval_retrieval(pipeline.retriever, GOLDEN, ks=(1, 3))
    table = Table(show_header=True, header_style="bold")
    table.add_column("case")
    table.add_column("ranked chunk ids", style="cyan")
    table.add_column("recall@1", justify="right")
    table.add_column("recall@3", justify="right")
    table.add_column("mrr", justify="right")
    for row in report.per_case:
        table.add_row(
            row["query"][:44] + "...",
            ", ".join(row["ranked_ids"]),
            f"{row['recall@1']:.0f}",
            f"{row['recall@3']:.0f}",
            f"{row['mrr']:.2f}",
        )
    console.print(table)
    console.print(f"[green]{report.summary()}[/green]")
    console.print()

    console.print("[bold]Answer eval (extractive generator):[/bold]")
    answers = [
        (a.answer, c)
        for a, c in zip(pipeline.ask_batch([c.query for c in GOLDEN]), GOLDEN, strict=False)
    ]
    areport = eval_answers(answers)
    console.print(f"[green]{areport.summary()}[/green]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragpy", description="rag-pipeline CLI")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="run the offline KB demo with evaluation")
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return run_demo()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
