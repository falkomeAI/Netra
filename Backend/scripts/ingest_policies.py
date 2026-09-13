#!/usr/bin/env python3
"""
Ingest all state policy PDFs from polyic/state/ into the NETRA FAISS knowledge base.

Extracts text via pdfplumber, chunks it, encodes with the same multilingual-e5-large
model used by the EmbeddingStage, and appends to the existing FAISS index + metadata.

Usage:
    python scripts/ingest_policies.py              # Ingest all PDFs
    python scripts/ingest_policies.py --dry-run     # Preview without writing
    python scripts/ingest_policies.py --reset       # Clear KB first, then ingest
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
POLYIC_DIR = REPO_ROOT / "polyic" / "state"
KB_DIR = PROJECT_ROOT / "knowledge_base" / "vector_store"
META_PATH = KB_DIR / "metadata.json"
INDEX_PATH = KB_DIR / "index.faiss"

MODEL_ID = "intfloat/multilingual-e5-large"
PASSAGE_PREFIX = "passage: "
CHUNK_MAX_LEN = 512
CHUNK_OVERLAP = 64

OLLAMA_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b"


def _is_html_file(path: Path) -> bool:
    """Detect if a file is actually HTML despite having a .pdf extension."""
    try:
        with open(path, "rb") as f:
            header = f.read(256)
        if header.startswith(b"%PDF"):
            return False
        text = header.decode("utf-8", errors="replace").lower()
        return "<html" in text or "<!doctype" in text or "<head" in text
    except Exception:
        return False


def _extract_html_text(path: Path) -> str:
    """Extract readable text from an HTML file saved as .pdf."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        import re
        raw = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        raw = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = re.sub(r"&nbsp;", " ", raw)
        raw = re.sub(r"&amp;", "&", raw)
        raw = re.sub(r"&lt;", "<", raw)
        raw = re.sub(r"&gt;", ">", raw)
        raw = re.sub(r"&#\d+;", " ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw
    except Exception as e:
        print(f"  [WARN] HTML extraction failed for {path.name}: {e}")
        return ""


def _ocr_scanned_pdf(pdf_path: Path) -> str:
    """OCR a scanned/image-only PDF using PaddleOCR or EasyOCR as fallback."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        all_text = []
        for page_num in range(min(len(doc), 50)):
            pix = doc[page_num].get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")

            try:
                from paddleocr import PaddleOCR
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                    tmp.write(img_bytes)
                    tmp.flush()
                    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                    result = ocr.ocr(tmp.name, cls=True)
                    if result and result[0]:
                        page_text = " ".join(line[1][0] for line in result[0] if line[1])
                        all_text.append(page_text)
            except ImportError:
                try:
                    import easyocr
                    reader = easyocr.Reader(["en", "hi"], gpu=False)
                    results = reader.readtext(img_bytes)
                    page_text = " ".join(r[1] for r in results)
                    all_text.append(page_text)
                except ImportError:
                    print(f"    [WARN] No OCR library available (install paddleocr or easyocr)")
                    return ""
        doc.close()
        return "\n\n".join(all_text)
    except ImportError:
        print(f"    [WARN] PyMuPDF (fitz) not installed — cannot OCR scanned PDF")
        return ""
    except Exception as e:
        print(f"    [WARN] OCR failed for {pdf_path.name}: {e}")
        return ""


def extract_pdf_text(pdf_path: Path) -> str:
    if _is_html_file(pdf_path):
        print(f"    [INFO] Detected as HTML file — extracting text from HTML")
        return _extract_html_text(pdf_path)

    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            text = "\n\n".join(pages)
            if text and len(text.strip()) >= 50:
                return text
    except Exception as e:
        print(f"  [WARN] pdfplumber failed for {pdf_path.name}: {e}")

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(pdf_path))
        text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        if text and len(text.strip()) >= 50:
            return text
    except Exception:
        pass

    print(f"    [INFO] No text layer found — attempting OCR on scanned PDF")
    return _ocr_scanned_pdf(pdf_path)


def chunk_text(text: str, max_len: int = CHUNK_MAX_LEN, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= max_len:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_len, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_len - overlap
    return chunks


def state_display_name(folder_name: str) -> str:
    return folder_name.replace("_", " ")


def discover_docs() -> list[tuple[str, Path]]:
    """Return list of (state_folder_name, doc_path) for all PDFs and TXT files under polyic/state/."""
    results = []
    if not POLYIC_DIR.exists():
        print(f"ERROR: {POLYIC_DIR} does not exist.")
        return results
    for state_dir in sorted(POLYIC_DIR.iterdir()):
        if not state_dir.is_dir():
            continue
        for doc in sorted(list(state_dir.glob("*.pdf")) + list(state_dir.glob("*.txt"))):
            if doc.stat().st_size > 100:
                results.append((state_dir.name, doc))
    return results


def load_existing_index():
    import faiss

    KB_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else []
        return index, metadata

    return None, []


def ollama_embed_batch(texts: list[str], url: str = OLLAMA_URL, model: str = OLLAMA_EMBED_MODEL, known_dim: int = 768) -> np.ndarray:
    """Embed a list of texts via Ollama /api/embeddings (one at a time, with retry)."""
    import urllib.request

    all_vecs = []
    for i, text in enumerate(texts):
        cleaned = text.encode("utf-8", errors="replace").decode("utf-8")[:4000]
        vec = None
        for attempt in range(3):
            try:
                payload = json.dumps({"model": model, "prompt": cleaned}).encode()
                req = urllib.request.Request(
                    f"{url}/api/embeddings",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                vec = result.get("embedding", [])
                if vec:
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 + attempt * 2)
                else:
                    print(f"    [WARN] Embedding failed for chunk {i} after 3 retries: {e}")
        if not vec:
            vec = [0.0] * known_dim
        all_vecs.append(vec)
    arr = np.asarray(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    arr /= norms
    return arr


def detect_ollama_dim(url: str = OLLAMA_URL, model: str = OLLAMA_EMBED_MODEL) -> int:
    arr = ollama_embed_batch(["hello"], url, model)
    return arr.shape[1]


def main():
    args = set(sys.argv[1:])
    dry_run = "--dry-run" in args
    reset = "--reset" in args
    use_ollama = "--no-ollama" not in args

    import faiss

    pdfs = discover_docs()
    if not pdfs:
        print("No documents found in polyic/state/. Run download_policies_v2.py first.")
        return

    print(f"\nFound {len(pdfs)} policy documents across {len(set(s for s, _ in pdfs))} states")
    print("=" * 60)

    if dry_run:
        for state, pdf in pdfs:
            print(f"  [{state_display_name(state)}] {pdf.name} ({pdf.stat().st_size / 1024:.0f} KB)")
        print("\n[DRY RUN] No changes made.")
        return

    if reset:
        print("Resetting knowledge base (clearing existing index)...")
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        if META_PATH.exists():
            META_PATH.unlink()
        index = None
        metadata = []
    else:
        index, metadata = load_existing_index()

    existing_sources = {m.get("source", "") for m in metadata}

    if use_ollama:
        print(f"\nUsing Ollama embeddings: {OLLAMA_EMBED_MODEL} at {OLLAMA_URL}")
        t0 = time.time()
        dim = detect_ollama_dim()
        print(f"Ollama ready in {time.time() - t0:.1f}s (dim={dim})")
        encode_fn = lambda texts: ollama_embed_batch(texts, known_dim=dim)
    else:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer

        print(f"\nLoading embedding model: {MODEL_ID}")
        t0 = time.time()
        st_model = SentenceTransformer(MODEL_ID, device="cpu")
        dim = st_model.get_sentence_embedding_dimension()
        print(f"Model loaded in {time.time() - t0:.1f}s (dim={dim})")

        def _st_encode(texts):
            prefixed = [f"{PASSAGE_PREFIX}{t}" for t in texts]
            emb = st_model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False, batch_size=8)
            return np.asarray(emb, dtype=np.float32)
        encode_fn = _st_encode

    if index is None:
        index = faiss.IndexFlatIP(dim)
    elif index.d != dim:
        if index.ntotal > 0 and not reset:
            print(f"\n  ERROR: Existing index has dim={index.d} with {index.ntotal} vectors,")
            print(f"  but current embedding model produces dim={dim}.")
            print(f"  To rebuild the index, re-run with: python scripts/ingest_policies.py --reset")
            print(f"  To keep the existing index, use: python scripts/ingest_policies.py --no-ollama")
            return
        print(f"Dimension mismatch (index={index.d}, model={dim}). Rebuilding empty index.")
        index = faiss.IndexFlatIP(dim)
        metadata = []

    total_chunks = 0
    total_pdfs_processed = 0
    skipped = 0
    failed = 0

    for i, (state, pdf_path) in enumerate(pdfs, 1):
        source_key = str(pdf_path)

        if source_key in existing_sources:
            print(f"  [{i}/{len(pdfs)}] SKIP (already indexed): {pdf_path.name}")
            skipped += 1
            continue

        print(f"\n  [{i}/{len(pdfs)}] {state_display_name(state)} / {pdf_path.name}")

        if pdf_path.suffix.lower() == ".txt":
            text = pdf_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = extract_pdf_text(pdf_path)
        if not text or len(text.strip()) < 50:
            print(f"    [FAIL] No extractable text")
            failed += 1
            continue

        chunks = chunk_text(text)
        print(f"    Extracted {len(text):,} chars → {len(chunks)} chunks")

        embeddings = encode_fn(chunks)

        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        index.add(embeddings)
        for ci, chunk in enumerate(chunks):
            metadata.append({
                "doc_id": doc_id,
                "chunk_idx": ci,
                "text": chunk,
                "source": source_key,
                "state": state_display_name(state),
                "district": "",
            })

        total_chunks += len(chunks)
        total_pdfs_processed += 1
        print(f"    Indexed: {len(chunks)} chunks (total in index: {index.ntotal})")

        if total_pdfs_processed % 5 == 0:
            KB_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(INDEX_PATH))
            META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            print(f"    [checkpoint] Saved {index.ntotal} vectors")

    print("\n" + "=" * 60)
    print("Saving FAISS index and metadata...")
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print(f"\n{'=' * 60}")
    print(f"INGESTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Backend         : {'Ollama (' + OLLAMA_EMBED_MODEL + ')' if use_ollama else 'SentenceTransformer (' + MODEL_ID + ')'}")
    print(f"PDFs processed   : {total_pdfs_processed}")
    print(f"PDFs skipped     : {skipped}")
    print(f"PDFs failed      : {failed}")
    print(f"New chunks added : {total_chunks}")
    print(f"Total index size : {index.ntotal} vectors")
    print(f"Total metadata   : {len(metadata)} entries")
    print(f"Index saved to   : {INDEX_PATH}")
    print(f"Metadata saved to: {META_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
