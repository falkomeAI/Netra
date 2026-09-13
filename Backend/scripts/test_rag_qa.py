#!/usr/bin/env python3
"""
Test RAG Q&A over ingested policy documents.

Retrieves relevant chunks from the FAISS index, sends them as context
to Ollama LLM, and prints the answer. Validates end-to-end RAG pipeline.

Usage:
    python scripts/test_rag_qa.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "knowledge_base" / "vector_store"
META_PATH = KB_DIR / "metadata.json"
INDEX_PATH = KB_DIR / "index.faiss"

OLLAMA_URL = "http://localhost:11434"
OLLAMA_LLM = "qwen3:4b"
OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b"

TOP_K = 5
MIN_SCORE = 0.25

TEST_QUESTIONS = [
    {
        "q": "What are the benefits of PM-KISAN scheme?",
        "expected_keywords": ["6000", "farmer", "installment", "2000"],
    },
    {
        "q": "How to apply for ration card in Tamil Nadu?",
        "state": "Tamil Nadu",
        "expected_keywords": ["ration", "application", "Civil Supplies", "Tamil Nadu"],
    },
    {
        "q": "What is MGNREGA and how many days of work does it guarantee?",
        "expected_keywords": ["100", "days", "employment", "rural"],
    },
    {
        "q": "What health insurance does Ayushman Bharat provide?",
        "expected_keywords": ["5", "lakh", "health", "hospital"],
    },
    {
        "q": "What schemes are available for farmers in Rajasthan?",
        "state": "Rajasthan",
        "expected_keywords": ["farmer", "agriculture"],
    },
    {
        "q": "Tell me about Karnataka government policies",
        "state": "Karnataka",
        "expected_keywords": ["Karnataka"],
    },
    {
        "q": "What is Kisan Credit Card and what interest rate does it offer?",
        "expected_keywords": ["credit", "interest", "loan"],
    },
    {
        "q": "What housing scheme is available for rural poor?",
        "expected_keywords": ["house", "PMAY", "rural"],
    },
]


def ollama_embed(text: str) -> list[float]:
    payload = json.dumps({"model": OLLAMA_EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result.get("embedding", [])


def load_retriever():
    import faiss

    if not INDEX_PATH.exists():
        print("ERROR: No FAISS index found. Run ingest_policies.py first.")
        sys.exit(1)

    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else []

    # Verify dim matches Ollama
    test_emb = ollama_embed("test")
    ollama_dim = len(test_emb)
    if index.d != ollama_dim:
        print(f"ERROR: Index dim={index.d} but Ollama {OLLAMA_EMBED_MODEL} dim={ollama_dim}")
        sys.exit(1)

    print(f"Loaded: {index.ntotal} vectors (dim={index.d}), {len(metadata)} metadata entries")
    print(f"Retrieval via Ollama {OLLAMA_EMBED_MODEL} (dim={ollama_dim})")
    return index, metadata


def retrieve(index, metadata, question: str, state: str = "", top_k: int = TOP_K):
    emb = ollama_embed(question)
    query_vec = np.asarray([emb], dtype=np.float32)
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9

    fetch_k = min(max(top_k * 5, top_k), index.ntotal)
    scores, indices = index.search(query_vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or score < MIN_SCORE:
            continue
        meta = metadata[idx] if idx < len(metadata) else {}
        if state and meta.get("state", "").lower() != state.lower():
            continue
        results.append({
            "text": meta.get("text", ""),
            "score": float(score),
            "state": meta.get("state", ""),
            "source": Path(meta.get("source", "")).name if meta.get("source") else "",
        })
        if len(results) >= top_k:
            break

    return results


def ask_ollama(question: str, context_chunks: list[dict]) -> str:
    context_text = "\n---\n".join(c["text"][:500] for c in context_chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions about Indian government policies and schemes. "
        "Use ONLY the provided context to answer. If the context doesn't contain the answer, say so. "
        "Keep answers concise (2-4 sentences)."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"

    payload = json.dumps({
        "model": OLLAMA_LLM,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 1024},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    return result.get("message", {}).get("content", "").strip()


def check_keywords(answer: str, expected: list[str]) -> tuple[int, int, list[str]]:
    answer_lower = answer.lower()
    found = [kw for kw in expected if kw.lower() in answer_lower]
    missing = [kw for kw in expected if kw.lower() not in answer_lower]
    return len(found), len(expected), missing


def run_single(test_num: int):
    """Run a single test by number (1-based)."""
    if test_num < 1 or test_num > len(TEST_QUESTIONS):
        print(f"ERROR: test number must be 1-{len(TEST_QUESTIONS)}")
        sys.exit(1)

    test = TEST_QUESTIONS[test_num - 1]
    question = test["q"]
    state = test.get("state", "")
    expected = test.get("expected_keywords", [])

    print("=" * 70)
    print(f"  RAG Q&A TEST {test_num}/{len(TEST_QUESTIONS)}")
    print(f"  Embedding : Ollama {OLLAMA_EMBED_MODEL} (INT8 Q8_0)")
    print(f"  LLM       : Ollama {OLLAMA_LLM} (INT4 Q4_K_M)")
    print("=" * 70)

    index, metadata = load_retriever()

    print(f"\n  QUESTION: {question}")
    if state:
        print(f"  STATE FILTER: {state}")
    print(f"  EXPECTED KEYWORDS: {expected}")
    print("─" * 70)

    t0 = time.time()
    chunks = retrieve(index, metadata, question, state=state)
    retrieve_ms = (time.time() - t0) * 1000

    if not chunks:
        print(f"\n  RETRIEVAL: No relevant chunks found!")
        print(f"  RESULT: FAIL (no context)")
        return

    print(f"\n  RETRIEVAL: {len(chunks)} chunks [{retrieve_ms:.0f}ms]")
    for j, c in enumerate(chunks):
        src = c["source"][:40] if c["source"] else "seeded-scheme"
        print(f"    #{j+1} [score={c['score']:.3f}] [{c['state'] or 'general'}] {src}")
        print(f"        \"{c['text'][:120]}...\"")

    print("\n" + "─" * 70)
    print("  Sending to Ollama LLM...")
    t0 = time.time()
    answer = ask_ollama(question, chunks)
    llm_ms = (time.time() - t0) * 1000

    print(f"\n  ANSWER [{llm_ms:.0f}ms]:")
    print(f"  ┌{'─' * 66}┐")
    for line in answer.split("\n"):
        print(f"  │ {line:64s} │")
    print(f"  └{'─' * 66}┘")

    if expected:
        found, total_kw, missing = check_keywords(answer, expected)
        status = "PASS" if found >= len(expected) * 0.5 else "PARTIAL" if found > 0 else "FAIL"
        print(f"\n  KEYWORD CHECK: {found}/{total_kw} matched")
        if missing:
            print(f"  MISSING: {missing}")
    else:
        status = "PASS" if len(answer) > 20 else "FAIL"

    print(f"\n{'=' * 70}")
    print(f"  TEST {test_num}: {'✓ ' if status == 'PASS' else '~ ' if status == 'PARTIAL' else '✗ '}{status}")
    print(f"{'=' * 70}")


def main():
    print("=" * 70)
    print("  RAG Q&A TEST — Policy Documents")
    print("=" * 70)

    print("\n[1/3] Loading retrieval index + Ollama embeddings...")
    t0 = time.time()
    index, metadata = load_retriever()
    print(f"  Ready in {time.time() - t0:.1f}s\n")

    print("[2/3] Testing Ollama LLM connection...")
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            print(f"  Ollama OK. Models: {models}\n")
    except Exception as e:
        print(f"  ERROR: Ollama not reachable: {e}")
        sys.exit(1)

    print("[3/3] Running Q&A tests...")
    print("=" * 70)

    total = len(TEST_QUESTIONS)
    passed = 0
    results_summary = []

    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["q"]
        state = test.get("state", "")
        expected = test.get("expected_keywords", [])

        print(f"\n{'─' * 70}")
        print(f"  Q{i}/{total}: {question}")
        if state:
            print(f"  Filter: state={state}")
        print(f"{'─' * 70}")

        t0 = time.time()
        chunks = retrieve(index, metadata, question, state=state)
        retrieve_ms = (time.time() - t0) * 1000

        if not chunks:
            print(f"  RETRIEVAL: No relevant chunks found!")
            print(f"  RESULT: FAIL (no context)")
            results_summary.append({"q": i, "status": "FAIL", "reason": "no retrieval"})
            continue

        print(f"  RETRIEVAL: {len(chunks)} chunks (top score: {chunks[0]['score']:.3f}) [{retrieve_ms:.0f}ms]")
        for j, c in enumerate(chunks[:3]):
            src = c["source"][:30] if c["source"] else "seeded"
            print(f"    #{j+1} [score={c['score']:.3f}] [{c['state'] or 'general'}] {src}")
            print(f"        \"{c['text'][:100]}...\"")

        t0 = time.time()
        answer = ask_ollama(question, chunks)
        llm_ms = (time.time() - t0) * 1000

        print(f"\n  ANSWER [{llm_ms:.0f}ms]:")
        for line in answer.split("\n"):
            print(f"    {line}")

        if expected:
            found, total_kw, missing = check_keywords(answer, expected)
            status = "PASS" if found >= len(expected) * 0.5 else "PARTIAL" if found > 0 else "FAIL"
            print(f"\n  KEYWORDS: {found}/{total_kw} matched", end="")
            if missing:
                print(f" (missing: {missing})", end="")
            print()
        else:
            status = "PASS" if len(answer) > 20 else "FAIL"

        if status == "PASS":
            passed += 1
        print(f"  RESULT: {status}")
        results_summary.append({"q": i, "status": status})

    print(f"\n{'=' * 70}")
    print(f"  FINAL SCORE: {passed}/{total} PASSED")
    print(f"{'=' * 70}")
    for r in results_summary:
        icon = "PASS" if r["status"] == "PASS" else "PARTIAL" if r["status"] == "PARTIAL" else "FAIL"
        print(f"  Q{r['q']}: {icon}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        run_single(test_num)
    else:
        main()
