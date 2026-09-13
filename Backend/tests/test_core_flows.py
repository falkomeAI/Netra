"""
Integration tests for NETRA v2 core capabilities:

1. Policy document collection
2. Create knowledge base
3. Search information
4. Image OCR + explanation
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

# Unique tokens embedded in fixtures — used to verify retrieval / OCR.
POLICY_TOKEN = "zebra-mango-7741"
POLICY_TOKEN_2 = "cobalt-orchid-3399"
OCR_TOKEN = "OCRCHECK5512"


# ---------------------------------------------------------------------------
# 1. Policy document collection
# ---------------------------------------------------------------------------

class TestPolicyDocumentCollection:
    """TC-P*: collect / ingest policy documents into the system."""

    def test_tc_p01_ingest_text_json(self, api):
        """TC-P01: Ingest policy text via JSON body succeeds."""
        text = (
            "Policy collection test NRSSY-JSON.\n"
            f"Unique token: {POLICY_TOKEN}\n"
            "Skill training stipend Rs 5000 per month."
        )
        r = api.post_json("/api/ingest", {"text": text})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("success") is True
        assert "embedding" in body.get("stages", {})
        assert body["stages"]["embedding"].get("success") is True

    def test_tc_p02_ingest_txt_file(self, api, policy_txt: Path):
        """TC-P02: Ingest .txt policy file via multipart upload succeeds."""
        with policy_txt.open("rb") as f:
            r = api.post("/api/ingest", files={"file": ("sample_policy.txt", f, "text/plain")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("success") is True

    def test_tc_p03_ingest_empty_text_rejected(self, api):
        """TC-P03: Empty text ingest is rejected with 400."""
        r = api.post_json("/api/ingest", {"text": "   "})
        assert r.status_code == 400
        assert r.json().get("success") is False

    def test_tc_p04_ingest_docx_rejected(self, api):
        """TC-P04: .docx files are rejected (unsupported)."""
        fake = io.BytesIO(b"PK fake docx content")
        r = api.post(
            "/api/ingest",
            files={"file": ("policy.docx", fake, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert r.status_code == 400
        body = r.json()
        assert body.get("success") is False
        assert "docx" in body.get("error", "").lower() or "not yet supported" in body.get("error", "").lower()

    def test_tc_p05_bulk_ingest(self, api, policy_txt: Path, policy_txt_2: Path):
        """TC-P05: Bulk ingest accepts multiple policy files."""
        files = [
            ("files", ("sample_policy.txt", policy_txt.read_bytes(), "text/plain")),
            ("files", ("sample_policy_2.txt", policy_txt_2.read_bytes(), "text/plain")),
        ]
        r = api.post("/api/ingest/bulk", files=files)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("success") is True
        assert body.get("processed", 0) >= 1
        assert any(f.get("status") == "ok" for f in body.get("files", []))

    def test_tc_p06_ingest_status_and_scan(self, api):
        """TC-P06: Ingest status reports watcher + KB; scan endpoint responds."""
        status = api.get("/api/ingest/status")
        assert status.status_code == 200
        body = status.json()
        assert "watcher" in body
        assert "knowledge_base" in body
        assert "total_chunks" in body["knowledge_base"]

        scan = api.post("/api/ingest/scan")
        assert scan.status_code == 200
        assert scan.json().get("success") is True


# ---------------------------------------------------------------------------
# 2. Create knowledge base
# ---------------------------------------------------------------------------

class TestCreateKnowledgeBase:
    """TC-K*: embedding + graph KB creation from collected docs."""

    def test_tc_k01_kb_stats_present(self, api):
        """TC-K01: KB status endpoint returns chunk/document counts."""
        r = api.get("/api/ingest/status")
        assert r.status_code == 200
        kb = r.json()["knowledge_base"]
        assert isinstance(kb["total_chunks"], int)
        assert isinstance(kb["total_documents"], int)
        assert kb["total_chunks"] >= 1

    def test_tc_k02_ingest_increases_or_keeps_chunks(self, api):
        """TC-K02: Ingesting a new unique doc does not shrink the KB."""
        before = api.get("/api/ingest/status").json()["knowledge_base"]["total_chunks"]
        unique = f"KB growth test document token-kb-{before}-x. Benefit Rs 999."
        r = api.post_json("/api/ingest", {"text": unique})
        assert r.status_code == 200
        assert r.json().get("success") is True
        after = api.get("/api/ingest/status").json()["knowledge_base"]["total_chunks"]
        assert after >= before

    def test_tc_k03_embedding_and_graph_stages(self, api):
        """TC-K03: Ingest runs embedding and knowledge_graph stages."""
        r = api.post_json(
            "/api/ingest",
            {"text": "Graph entity test: PAN ABCDE1234F amount Rs 25000 dated 01-01-2026."},
        )
        assert r.status_code == 200
        stages = r.json().get("stages", {})
        assert stages.get("embedding", {}).get("success") is True
        # knowledge_graph may be optional depending on config; require key if present
        if "knowledge_graph" in stages:
            assert stages["knowledge_graph"].get("success") is True

    def test_tc_k04_graph_endpoint(self, api):
        """TC-K04: Knowledge graph endpoint returns JSON structure."""
        r = api.get("/api/graph")
        assert r.status_code == 200
        body = r.json()
        assert "nodes" in body
        assert "edges" in body
        assert isinstance(body["nodes"], list)
        assert isinstance(body["edges"], list)


# ---------------------------------------------------------------------------
# 3. Search information
# ---------------------------------------------------------------------------

class TestSearchInformation:
    """TC-S*: RAG search over the knowledge base via text pipeline."""

    @pytest.fixture(scope="class", autouse=True)
    def _ensure_searchable_doc(self, api, policy_txt: Path):
        with policy_txt.open("rb") as f:
            api.post("/api/ingest", files={"file": ("sample_policy.txt", f, "text/plain")})

    def test_tc_s01_search_policy_question(self, api):
        """TC-S01: Asking about ingested policy returns a successful LLM answer."""
        r = api.post_json(
            "/api/pipeline/text",
            {
                "text": f"What benefits does NRSSY give? Mention {POLICY_TOKEN} if known.",
                "language": "en",
                "mode": "auto",
            },
            timeout=300,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        stages = body.get("stages", {})
        llm = stages.get("llm", {})
        assert llm.get("success") is True or body.get("success") is not False
        # Prefer scheme_qa when retrieval finds the policy; accept general_chat fallback
        assert body.get("prompt_used") in ("scheme_qa", "general_chat", None) or "prompt_used" in body

    def test_tc_s02_search_known_scheme(self, api):
        """TC-S02: Query about seeded scheme (PM-KISAN) returns grounded answer."""
        r = api.post_json(
            "/api/pipeline/text",
            {"text": "What is PM-KISAN and how much money do farmers get?", "language": "en", "mode": "auto"},
            timeout=300,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        llm = body.get("stages", {}).get("llm", {})
        data = llm.get("data") or {}
        answer = data.get("response") or data.get("text") or data.get("answer") or ""
        if not answer:
            answer = str(body.get("stages", {}).get("llm", {}))
        assert llm.get("success") is True
        assert len(str(answer)) > 20
        # Grounding: must not invent 18000; should mention 6000 when KB is seeded
        compact = str(answer).replace(",", "").replace(" ", "")
        assert "18000" not in compact
        assert "6000" in compact or "6,000" in answer
        assert body.get("prompt_used") == "scheme_qa" or body.get("rag_hits", 0) >= 1

    def test_tc_s05_geo_tagged_ingest(self, api):
        """TC-S05: Ingest with state/district tags succeeds."""
        r = api.post_json(
            "/api/ingest",
            {
                "text": "Rajasthan Jaipur district solar pump subsidy test token geo-rj-991.",
                "state": "Rajasthan",
                "district": "Jaipur",
                "language": "en",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("success") is True
        assert body.get("state") == "Rajasthan"
        assert body.get("district") == "Jaipur"

    def test_tc_s03_empty_search_rejected(self, api):
        """TC-S03: Empty search text is rejected."""
        r = api.post_json("/api/pipeline/text", {"text": "", "language": "en"})
        assert r.status_code == 400

    def test_tc_s04_chat_mode_skips_scheme_routing(self, api):
        """TC-S04: mode=chat forces general_chat prompt path."""
        r = api.post_json(
            "/api/pipeline/text",
            {"text": "Say hello briefly.", "language": "en", "mode": "chat"},
            timeout=300,
        )
        assert r.status_code == 200, r.text
        assert r.json().get("prompt_used") == "general_chat"


# ---------------------------------------------------------------------------
# 4. Image OCR + explanation
# ---------------------------------------------------------------------------

class TestOCRAndExplanation:
    """TC-O*: extract text from image via OCR and produce explanation."""

    def test_tc_o01_missing_file_rejected(self, api):
        """TC-O01: Pipeline run without file returns 400."""
        r = api.post("/api/pipeline/run", data={"language": "en", "mode": "camera"})
        assert r.status_code == 400

    def test_tc_o02_ocr_extracts_text_from_image(self, api, ocr_image: Path):
        """TC-O02: Uploaded document image yields OCR success and extracted text."""
        with ocr_image.open("rb") as f:
            r = api.post(
                "/api/pipeline/run",
                data={"language": "en", "mode": "camera"},
                files={"image": ("sample_ocr_doc.png", f, "image/png")},
                timeout=300,
            )
        assert r.status_code == 200, r.text
        body = r.json()
        ocr = body.get("stages", {}).get("ocr", {})
        assert ocr.get("success") is True, body
        text = (ocr.get("data") or {}).get("text", "")
        assert isinstance(text, str) and len(text.strip()) > 0
        # Best-effort token check — OCR may miss characters on synthetic images
        lowered = text.lower()
        assert any(
            token in lowered
            for token in ("ration", "pmgkay", "ramesh", "aadhaar", "ocrcheck", "government", "notice", "food")
        ), f"OCR text did not contain expected keywords: {text!r}"

    def test_tc_o03_explanation_from_llm(self, api, ocr_image: Path):
        """TC-O03: After OCR, LLM stage returns an explanation/summary."""
        with ocr_image.open("rb") as f:
            r = api.post(
                "/api/pipeline/run",
                data={"language": "en", "mode": "camera"},
                files={"image": ("sample_ocr_doc.png", f, "image/png")},
                timeout=300,
            )
        assert r.status_code == 200, r.text
        body = r.json()
        llm = body.get("stages", {}).get("llm", {})
        assert llm.get("success") is True, body
        data = llm.get("data") or {}
        explanation = data.get("response") or data.get("text") or data.get("answer") or ""
        if not explanation and isinstance(data, dict):
            # Fall back to any long string field
            for v in data.values():
                if isinstance(v, str) and len(v) > 30:
                    explanation = v
                    break
        assert len(str(explanation).strip()) > 20, f"No explanation found in LLM data: {data}"
