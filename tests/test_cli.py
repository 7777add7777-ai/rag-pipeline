"""CLI tests: the offline demo must run end to end without network."""

import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def demo_output():
    proc = subprocess.run(
        [sys.executable, "-m", "rag_pipeline.cli", "demo"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc


def test_demo_exit_code(demo_output):
    assert demo_output.returncode == 0, demo_output.stderr


def test_demo_prints_metrics(demo_output):
    out = demo_output.stdout
    assert "Retrieval eval" in out
    assert "MRR=" in out
    assert "R@1=" in out
    assert "answer_hit=" in out


def test_demo_ingestion_counts(demo_output):
    assert "Ingested 4 documents" in demo_output.stdout


def test_demo_is_deterministic():
    """Two runs produce byte-identical stdout (offline core is deterministic)."""
    outs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-m", "rag_pipeline.cli", "demo"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        outs.append(proc.stdout)
    assert outs[0] == outs[1]
