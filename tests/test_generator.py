"""Generator tests: extractive template + OpenAI-compatible client."""

import httpx
import pytest

from rag_pipeline.documents import Chunk
from rag_pipeline.generator import OpenAICompatibleGenerator, TemplateGenerator
from rag_pipeline.vector_store import RankedChunk


def ranked(cid: str, text: str, score: float, source: str = "") -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(id=cid, doc_id="d", text=text, index=0, source=source), score=score
    )


def test_template_no_context():
    g = TemplateGenerator()
    assert g.generate("anything", []) == "No relevant context found."


def test_template_min_score_filters():
    g = TemplateGenerator(min_score=0.5)
    ctx = [ranked("a", "Refunds are possible within 30 days for defective items.", 0.1)]
    assert g.generate("refund days defective", ctx) == "No relevant context found."


def test_template_extractive_with_citation():
    g = TemplateGenerator()
    refund_text = "Refunds are possible within 30 days for defective items. Contact support."
    ctx = [
        ranked("a", refund_text, 0.9, source="kb/refund"),
        ranked("b", "Shipping takes five days.", 0.2, source="kb/shipping"),
    ]
    ans = g.generate("refund window for defective items", ctx)
    assert "30 days" in ans
    assert "citations:" in ans
    assert "kb/refund" in ans


def test_template_max_sentences():
    g = TemplateGenerator(max_sentences=1)
    ctx = [
        ranked(
            "a",
            "Refunds cover defects. Refunds cover shipping damage. Refunds cover warranty faults.",
            0.9,
            source="s",
        )
    ]
    ans = g.generate("refunds cover what defects damage faults", ctx)
    # only one sentence should be picked
    sentences = [s for s in ans.split(". ") if s and not s.startswith("[citations")]
    assert len(sentences) <= 1


def test_template_no_overlap_falls_back_to_top_chunk():
    g = TemplateGenerator()
    ctx = [ranked("a", "Zebra stripes are black and white.", 0.8, source="z")]
    ans = g.generate("xyzzy plugh frobnicate", ctx)
    assert "Zebra" in ans
    assert "source: z" in ans


def test_template_deterministic():
    g = TemplateGenerator()
    ctx = [ranked("a", "Refunds are within 30 days for defects. Support helps.", 0.9, source="s")]
    assert g.generate("refund defects", ctx) == g.generate("refund defects", ctx)


CHAT_OK = {
    "choices": [{"message": {"content": "The answer is 42. [42]"}}]
}


def test_openai_generator_request_payload():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=CHAT_OK)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    g = OpenAICompatibleGenerator(
        "sk-y", model="chat", base_url="https://api.test/v1", http_client=client
    )
    ctx = [ranked("a#0", "refund text", 0.9, source="kb/refund")]
    out = g.generate("refund?", ctx)
    assert out == "The answer is 42. [42]"
    assert seen["url"] == "https://api.test/v1/chat/completions"
    body = seen["body"]
    assert body["model"] == "chat"
    assert body["temperature"] == 0.0
    assert "[a#0]" in body["messages"][1]["content"]
    assert "kb/refund" in body["messages"][1]["content"]


def test_openai_generator_empty_context_marker():
    def handler(req: httpx.Request) -> httpx.Response:
        import json

        json.loads(req.content)
        return httpx.Response(200, json=CHAT_OK)

    g = OpenAICompatibleGenerator(
        "k", "m", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert g.generate("q", []) == "The answer is 42. [42]"


def test_openai_generator_error_status():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no"))
    )
    g = OpenAICompatibleGenerator("k", "m", http_client=client)
    with pytest.raises(RuntimeError, match="401"):
        g.generate("q", [])
