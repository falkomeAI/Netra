# NETRA v2 — Core Flow Test Cases

Target: Backend `http://localhost:5000`  
Suite: `Backend/tests/test_core_flows.py`

```bash
cd profal-v2/Backend
pytest tests/test_core_flows.py -v
# or with custom base URL:
NETRA_BASE_URL=http://127.0.0.1:5000 pytest tests/test_core_flows.py -v
```

---

## 1. Policy document collection

| ID | Title | Steps | Expected |
|----|-------|-------|----------|
| TC-P01 | Ingest text JSON | `POST /api/ingest` with `{"text": "..."}` | `200`, `success=true`, embedding stage OK |
| TC-P02 | Ingest `.txt` file | Upload `sample_policy.txt` as `file` | `200`, `success=true` |
| TC-P03 | Reject empty text | `POST /api/ingest` with blank text | `400`, `success=false` |
| TC-P04 | Reject `.docx` | Upload fake `.docx` | `400`, unsupported error |
| TC-P05 | Bulk ingest | `POST /api/ingest/bulk` with 2 text files | `200`, `processed >= 1` |
| TC-P06 | Status + inbox scan | `GET /api/ingest/status`, `POST /api/ingest/scan` | Watcher + KB stats; scan `success=true` |

**Manual UI:** Inject screen → single/multi upload → Scan Inbox.

---

## 2. Create knowledge base

| ID | Title | Steps | Expected |
|----|-------|-------|----------|
| TC-K01 | KB stats | `GET /api/ingest/status` | `total_chunks >= 1`, `total_documents` present |
| TC-K02 | Ingest grows/keeps KB | Note chunks → ingest unique text → recheck | Chunks after ≥ chunks before |
| TC-K03 | Embedding + graph stages | Ingest text with entities | `embedding.success=true`; graph OK if enabled |
| TC-K04 | Graph API | `GET /api/graph` | JSON with `nodes` and `edges` arrays |

**Manual:** After ingest, Inject screen KB counters update; optional `GET /api/graph`.

---

## 3. Search information

| ID | Title | Steps | Expected |
|----|-------|-------|----------|
| TC-S01 | Search ingested policy | Ensure policy ingested → `POST /api/pipeline/text` about NRSSY | `200`, LLM stage succeeds |
| TC-S02 | Search seeded scheme | Ask about PM-KISAN benefits | `200`, grounded answer with ₹6000 (not ₹18000) |
| TC-S03 | Empty query | `POST /api/pipeline/text` with `text=""` | `400` |
| TC-S04 | Chat mode | `mode=chat` with short greeting | `prompt_used=general_chat` |
| TC-S05 | Geo-tagged ingest | Ingest with `state`/`district` | Tags stored; `success=true` |

## Offline unit / CI

| Suite | Command | Notes |
|-------|---------|-------|
| Unit | `pytest tests/test_unit_offline.py -v` | No live server |
| Integration | `pytest tests/test_core_flows.py -v` | Needs backend `:5000` |
| CI | `.github/workflows/netra-ci.yml` | lint + unit + docker build |

## Readiness

| Endpoint | Meaning |
|----------|---------|
| `GET /api/health` | Liveness + check details |
| `GET /api/ready` | `200` only if pipeline + Ollama ready |

**Manual UI:** Voice or follow-up text question about a scheme/policy.

---

## 4. Image OCR + explanation

| ID | Title | Steps | Expected |
|----|-------|-------|----------|
| TC-O01 | Missing file | `POST /api/pipeline/run` without file | `400` |
| TC-O02 | OCR extraction | Upload `sample_ocr_doc.png` as `image` | OCR `success=true`, text contains keywords |
| TC-O03 | LLM explanation | Same image upload | LLM `success=true`, explanation length > 20 |

**Manual UI:** Camera → capture document photo → read OCR + explanation result.

---

## Fixtures

| File | Purpose |
|------|---------|
| `tests/fixtures/sample_policy.txt` | Policy doc with token `zebra-mango-7741` |
| `tests/fixtures/sample_policy_2.txt` | Second policy with token `cobalt-orchid-3399` |
| `tests/fixtures/sample_ocr_doc.png` | Synthetic notice image with reference `OCRCHECK5512` |
