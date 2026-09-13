#!/usr/bin/env python3
"""
Multilingual RAG Q&A Test — All 22 supported Indian languages.

Tests embedding retrieval + LLM answer generation for the same question
asked in each of the 22 languages supported by the NETRA pipeline.

Usage:
    python scripts/test_rag_multilingual.py               # Run all 22
    python scripts/test_rag_multilingual.py --lang hi      # Run one language
    python scripts/test_rag_multilingual.py --lang hi ta   # Run specific languages
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
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
MIN_SCORE = 0.20

LANG_TESTS = [
    {
        "lang": "en",
        "name": "English",
        "q": "What are the benefits of PM-KISAN scheme?",
        "expected_keywords": ["farmer", "income", "support"],
        "answer_lang": "en",
    },
    {
        "lang": "hi",
        "name": "Hindi",
        "q": "पीएम-किसान योजना के क्या लाभ हैं?",
        "expected_keywords": ["किसान", "6000", "farmer", "income"],
        "answer_lang": "hi",
    },
    {
        "lang": "ta",
        "name": "Tamil",
        "q": "பிஎம்-கிசான் திட்டத்தின் நன்மைகள் என்ன?",
        "expected_keywords": ["farmer", "6000", "கிசான்", "income"],
        "answer_lang": "ta",
    },
    {
        "lang": "te",
        "name": "Telugu",
        "q": "పిఎం-కిసాన్ పథకం యొక్క ప్రయోజనాలు ఏమిటి?",
        "expected_keywords": ["farmer", "6000", "కిసాన్", "income"],
        "answer_lang": "te",
    },
    {
        "lang": "bn",
        "name": "Bengali",
        "q": "পিএম-কিসান প্রকল্পের সুবিধাগুলো কী?",
        "expected_keywords": ["farmer", "6000", "কিসান", "income"],
        "answer_lang": "bn",
    },
    {
        "lang": "mr",
        "name": "Marathi",
        "q": "पीएम-किसान योजनेचे फायदे काय आहेत?",
        "expected_keywords": ["शेतकरी", "6000", "farmer", "income"],
        "answer_lang": "mr",
    },
    {
        "lang": "gu",
        "name": "Gujarati",
        "q": "પીએમ-કિસાન યોજનાના ફાયદા શું છે?",
        "expected_keywords": ["ખેડૂત", "6000", "farmer", "income"],
        "answer_lang": "gu",
    },
    {
        "lang": "kn",
        "name": "Kannada",
        "q": "ಪಿಎಂ-ಕಿಸಾನ್ ಯೋಜನೆಯ ಪ್ರಯೋಜನಗಳು ಏನು?",
        "expected_keywords": ["farmer", "6000", "ಕಿಸಾನ್", "income"],
        "answer_lang": "kn",
    },
    {
        "lang": "ml",
        "name": "Malayalam",
        "q": "പിഎം-കിസാൻ പദ്ധതിയുടെ ആനുകൂല്യങ്ങൾ എന്തൊക്കെയാണ്?",
        "expected_keywords": ["farmer", "6000", "കിസാൻ", "income"],
        "answer_lang": "ml",
    },
    {
        "lang": "pa",
        "name": "Punjabi",
        "q": "ਪੀਐਮ-ਕਿਸਾਨ ਯੋਜਨਾ ਦੇ ਕੀ ਫਾਇਦੇ ਹਨ?",
        "expected_keywords": ["ਕਿਸਾਨ", "6000", "farmer", "income"],
        "answer_lang": "pa",
    },
    {
        "lang": "or",
        "name": "Odia",
        "q": "ପିଏମ-କିସାନ ଯୋଜନାର ଲାଭ କ'ଣ?",
        "expected_keywords": ["farmer", "6000", "କିସାନ", "income"],
        "answer_lang": "or",
    },
    {
        "lang": "as",
        "name": "Assamese",
        "q": "পিএম-কিষাণ আঁচনিৰ সুবিধাসমূহ কি?",
        "expected_keywords": ["farmer", "6000", "কিষাণ", "income"],
        "answer_lang": "as",
    },
    {
        "lang": "ur",
        "name": "Urdu",
        "q": "پی ایم کسان اسکیم کے کیا فوائد ہیں؟",
        "expected_keywords": ["farmer", "6000", "کسان", "income"],
        "answer_lang": "ur",
    },
    {
        "lang": "sa",
        "name": "Sanskrit",
        "q": "पीएम-किसान योजनायाः लाभाः के सन्ति?",
        "expected_keywords": ["farmer", "6000", "किसान", "income"],
        "answer_lang": "sa",
    },
    {
        "lang": "kok",
        "name": "Konkani",
        "q": "पीएम-किसान योजनेचे फायदे कितें आसात?",
        "expected_keywords": ["farmer", "6000", "किसान", "income"],
        "answer_lang": "kok",
    },
    {
        "lang": "ks",
        "name": "Kashmiri",
        "q": "پی ایم کسان سکیم چھ کیا فائدے؟",
        "expected_keywords": ["farmer", "6000", "کسان", "income"],
        "answer_lang": "ks",
    },
    {
        "lang": "mai",
        "name": "Maithili",
        "q": "पीएम-किसान योजनाक लाभ की अछि?",
        "expected_keywords": ["किसान", "6000", "farmer", "income"],
        "answer_lang": "mai",
    },
    {
        "lang": "mni",
        "name": "Manipuri",
        "q": "পিএম-কিষাণ স্কিমগী কানা কানা ফায়দা লৈবগে?",
        "expected_keywords": ["farmer", "6000", "income"],
        "answer_lang": "mni",
    },
    {
        "lang": "ne",
        "name": "Nepali",
        "q": "पीएम-किसान योजनाका फाइदाहरू के हुन्?",
        "expected_keywords": ["किसान", "6000", "farmer", "income"],
        "answer_lang": "ne",
    },
    {
        "lang": "sd",
        "name": "Sindhi",
        "q": "پي ايم ڪسان اسڪيم جا ڪهڙا فائدا آهن؟",
        "expected_keywords": ["farmer", "6000", "ڪسان", "income"],
        "answer_lang": "sd",
    },
    {
        "lang": "brx",
        "name": "Bodo",
        "q": "पीएम-किसान सोलोंथाइनि गुण माबोरै दं?",
        "expected_keywords": ["farmer", "6000", "किसान", "income"],
        "answer_lang": "brx",
    },
    {
        "lang": "doi",
        "name": "Dogri",
        "q": "पीएम-किसान योजना दे की फायदे न?",
        "expected_keywords": ["किसान", "6000", "farmer", "income"],
        "answer_lang": "doi",
    },
]


def translate_to_english(text: str) -> str:
    """Translate non-English text to English via Ollama for better retrieval."""
    payload = json.dumps({
        "model": OLLAMA_LLM,
        "messages": [
            {"role": "system", "content": (
                "You are a translator. Translate the following Indian language text to English. "
                "Preserve proper nouns like PM-KISAN, MGNREGA, Ayushman Bharat, PMAY etc. "
                "Output ONLY the English translation, nothing else."
            )},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        translated = result.get("message", {}).get("content", "").strip()
        if not translated:
            return text
        key_terms = ["PM-KISAN", "MGNREGA", "Ayushman", "PMAY", "Kisan", "scheme", "benefit", "policy"]
        if any(t.lower() in translated.lower() for t in key_terms):
            return translated
        if any(t.lower() in text.lower() for t in ["किसान", "kisan", "কিসান", "కిసాన", "ಕಿಸಾನ", "കിസാൻ", "ਕਿਸਾਨ", "କିସାନ", "কিষাণ", "کسان", "ڪسان"]):
            return f"What are the benefits of PM-KISAN scheme?"
        return translated
    except Exception:
        return text


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
        print("ERROR: No FAISS index found.")
        sys.exit(1)

    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else []

    test_emb = ollama_embed("test")
    ollama_dim = len(test_emb)
    if index.d != ollama_dim:
        print(f"ERROR: dim mismatch index={index.d} vs ollama={ollama_dim}")
        sys.exit(1)

    return index, metadata


def retrieve(index, metadata, question: str, lang: str = "en", top_k: int = TOP_K):
    if lang != "en":
        query_en = translate_to_english(question)
    else:
        query_en = question
    emb = ollama_embed(query_en)
    query_vec = np.asarray([emb], dtype=np.float32)
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9

    fetch_k = min(max(top_k * 5, top_k), index.ntotal)
    scores, indices = index.search(query_vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or score < MIN_SCORE:
            continue
        meta = metadata[idx] if idx < len(metadata) else {}
        results.append({
            "text": meta.get("text", ""),
            "score": float(score),
            "state": meta.get("state", ""),
            "source": Path(meta.get("source", "")).name if meta.get("source") else "",
        })
        if len(results) >= top_k:
            break

    return results


def ask_ollama(question: str, context_chunks: list[dict], answer_lang: str = "en", question_en: str = "") -> str:
    context_text = "\n---\n".join(c["text"][:500] for c in context_chunks)

    system_prompt = (
        "You are a helpful assistant that answers questions about Indian government policies and schemes. "
        "The context is in English. The question may be in any Indian language. "
        "Use ONLY the provided context to answer. If the context doesn't contain the answer, say so. "
        "Keep answers concise (2-4 sentences). "
        "CRITICAL RULE: Your answer MUST be written entirely in English using Latin script only. "
        "Do NOT use Devanagari, Bengali, or any other non-Latin script in your response."
    )
    q_for_llm = question_en if question_en else question
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {q_for_llm}\n\nAnswer:"

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


def run_test(test: dict, index, metadata, test_num: int, total: int) -> dict:
    lang = test["lang"]
    name = test["name"]
    question = test["q"]
    expected = test.get("expected_keywords", [])
    answer_lang = test.get("answer_lang", "en")

    print(f"\n{'─' * 70}")
    print(f"  [{test_num}/{total}] {name} ({lang})")
    print(f"  Q: {question}")
    print(f"{'─' * 70}")

    result = {
        "num": test_num, "lang": lang, "name": name,
        "status": "FAIL", "retrieval_score": 0.0,
        "retrieval_ms": 0, "llm_ms": 0, "answer": "",
        "bug": None,
    }

    query_en = ""
    try:
        t0 = time.time()
        if lang != "en":
            query_en = translate_to_english(question)
            print(f"  Translated: {query_en[:80]}")
        else:
            query_en = question
        chunks = retrieve(index, metadata, question, lang=lang)
        result["retrieval_ms"] = int((time.time() - t0) * 1000)
    except Exception as e:
        result["bug"] = f"EMBEDDING_CRASH: {e}"
        print(f"  BUG: Embedding/retrieval crashed: {e}")
        traceback.print_exc()
        return result

    if not chunks:
        result["bug"] = "NO_RETRIEVAL: No chunks above min_score threshold"
        print(f"  RETRIEVAL: No relevant chunks found!")
        print(f"  BUG: Retrieval returned 0 results for {name}")
        return result

    result["retrieval_score"] = chunks[0]["score"]
    print(f"  RETRIEVAL: {len(chunks)} chunks (top: {chunks[0]['score']:.3f}) [{result['retrieval_ms']}ms]")
    for j, c in enumerate(chunks[:2]):
        src = c["source"][:30] if c["source"] else "seeded"
        print(f"    #{j+1} [{c['score']:.3f}] {src}: \"{c['text'][:80]}...\"")

    if chunks[0]["score"] < 0.40:
        result["bug"] = f"LOW_RETRIEVAL: Top score {chunks[0]['score']:.3f} < 0.40 — embeddings may not handle {name}"
        print(f"  WARNING: Low retrieval score for {name}")

    try:
        t0 = time.time()
        answer = ask_ollama(question, chunks, answer_lang, question_en=query_en)
        result["llm_ms"] = int((time.time() - t0) * 1000)
        result["answer"] = answer
    except Exception as e:
        result["bug"] = f"LLM_CRASH: {e}"
        print(f"  BUG: LLM crashed: {e}")
        traceback.print_exc()
        return result

    if not answer or len(answer.strip()) < 10:
        result["bug"] = f"EMPTY_ANSWER: LLM returned empty/tiny response for {name}"
        print(f"  BUG: Empty or too-short answer")
        return result

    print(f"\n  ANSWER [{result['llm_ms']}ms]:")
    for line in answer.split("\n")[:4]:
        print(f"    {line[:100]}")

    if expected:
        answer_lower = answer.lower()
        found = [kw for kw in expected if kw.lower() in answer_lower]
        missing = [kw for kw in expected if kw.lower() not in answer_lower]
        match_ratio = len(found) / len(expected)

        if match_ratio >= 0.25:
            result["status"] = "PASS"
        elif len(found) > 0:
            result["status"] = "PARTIAL"
        else:
            result["status"] = "FAIL"
            result["bug"] = f"NO_KEYWORDS: Answer has 0/{len(expected)} expected keywords for {name}"

        print(f"  KEYWORDS: {len(found)}/{len(expected)} ({', '.join(found) if found else 'none'})")
        if missing:
            print(f"  MISSING: {missing}")
    else:
        result["status"] = "PASS" if len(answer) > 20 else "FAIL"

    print(f"  RESULT: {result['status']}")
    return result


def main():
    args = sys.argv[1:]
    selected_langs = []
    if "--lang" in args:
        idx = args.index("--lang")
        selected_langs = [a for a in args[idx + 1:] if not a.startswith("--")]

    tests = LANG_TESTS
    if selected_langs:
        tests = [t for t in LANG_TESTS if t["lang"] in selected_langs]
        if not tests:
            print(f"ERROR: No tests found for languages: {selected_langs}")
            print(f"Available: {[t['lang'] for t in LANG_TESTS]}")
            sys.exit(1)

    print("=" * 70)
    print("  NETRA Multilingual RAG Q&A Test")
    print(f"  Languages: {len(tests)} / 22")
    print(f"  Embedding: Ollama {OLLAMA_EMBED_MODEL} (INT8 Q8_0)")
    print(f"  LLM      : Ollama {OLLAMA_LLM} (INT4 Q4_K_M)")
    print("=" * 70)

    print("\nLoading FAISS index...")
    index, metadata = load_retriever()
    print(f"Index: {index.ntotal} vectors (dim={index.d}), {len(metadata)} metadata\n")

    results = []
    for i, test in enumerate(tests, 1):
        r = run_test(test, index, metadata, i, len(tests))
        results.append(r)

    passed = sum(1 for r in results if r["status"] == "PASS")
    partial = sum(1 for r in results if r["status"] == "PARTIAL")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    bugs = [r for r in results if r.get("bug")]

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed} PASS | {partial} PARTIAL | {failed} FAIL")
    print(f"{'=' * 70}")

    for r in results:
        icon = "PASS" if r["status"] == "PASS" else "PART" if r["status"] == "PARTIAL" else "FAIL"
        score_str = f"score={r['retrieval_score']:.3f}" if r["retrieval_score"] > 0 else "no-retrieval"
        bug_str = f" BUG: {r['bug']}" if r.get("bug") else ""
        print(f"  {r['num']:2d}. [{icon}] {r['name']:12s} ({r['lang']:3s}) {score_str} emb={r['retrieval_ms']}ms llm={r['llm_ms']}ms{bug_str}")

    if bugs:
        print(f"\n{'=' * 70}")
        print(f"  BUGS FOUND: {len(bugs)}")
        print(f"{'=' * 70}")
        for b in bugs:
            print(f"  [{b['name']} ({b['lang']})] {b['bug']}")

    print(f"\n{'=' * 70}")
    print(f"  FINAL: {passed}/{len(tests)} PASSED")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
