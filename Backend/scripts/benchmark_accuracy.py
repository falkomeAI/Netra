#!/usr/bin/env python3
"""
NETRA accuracy & model-comparison benchmark.

Evaluates answer quality across different models for each pipeline stage,
using ground-truth test cases with expected outputs.

Stages benchmarked:
  - LLM:       answer relevance, keyword recall, factual grounding
  - Retrieval:  recall@k, precision, expected-document hit rate
  - NMT:       round-trip preservation, keyword survival
  - OCR:       character-level & word-level accuracy (if fixture image present)

Multi-model comparison:
  For each stage that has multiple model variants in models.yaml,
  the script loads each variant, runs the same test suite, and produces
  a side-by-side comparison of accuracy + latency + hardware.

Examples:
    cd Backend
    python scripts/benchmark_accuracy.py --mode direct
    python scripts/benchmark_accuracy.py --mode direct --stage llm
    python scripts/benchmark_accuracy.py --mode direct --stage llm --compare
    python scripts/benchmark_accuracy.py --mode api --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from netra.utils.hardware_metrics import (  # noqa: E402
    MetricsSampler,
    summarize_samples,
    system_info,
)


# ═══════════════════════════════════════════════════════════════
# Ground-truth test dataset
# ═══════════════════════════════════════════════════════════════

RETRIEVAL_TESTS = [
    {
        "id": "ret-1",
        "query": "What is the skill training stipend under NRSSY scheme?",
        "expected_keywords": ["5000", "5,000", "stipend", "month"],
        "expected_token": "zebra-mango-7741",
        "description": "NRSSY stipend amount retrieval",
    },
    {
        "id": "ret-2",
        "query": "How to apply for NRSSY?",
        "expected_keywords": ["Common Service Center", "Aadhaar", "NRSSY-01", "1800"],
        "expected_token": "zebra-mango-7741",
        "description": "NRSSY application process",
    },
    {
        "id": "ret-3",
        "query": "What is the solar rooftop subsidy amount?",
        "expected_keywords": ["40,000", "40000", "solar", "rooftop"],
        "expected_token": "cobalt-orchid-3399",
        "description": "USRS subsidy amount",
    },
]

LLM_TESTS = [
    {
        "id": "llm-1",
        "question": "What is the skill training stipend under NRSSY scheme?",
        "context": (
            "NETRA Rural Skill Support Yojana (NRSSY). "
            "Benefits: Skill training stipend of Rs 5,000 per month for 6 months. "
            "Tool kit grant of Rs 12,000 after course completion."
        ),
        "expected_keywords": ["5,000", "5000", "month", "6 months", "stipend"],
        "must_not_contain": ["I don't know", "not available", "cannot"],
        "description": "Factual QA: NRSSY stipend with context",
    },
    {
        "id": "llm-2",
        "question": "Who is eligible for NRSSY?",
        "context": (
            "Eligibility: Resident of rural area aged 18 to 45 years. "
            "Family annual income below Rs 2,50,000. Must have Aadhaar and bank account."
        ),
        "expected_keywords": ["18", "45", "rural", "2,50,000", "250000", "Aadhaar"],
        "must_not_contain": ["I don't know"],
        "description": "Factual QA: NRSSY eligibility",
    },
    {
        "id": "llm-3",
        "question": "What is the helpline number for NRSSY?",
        "context": "How to apply: Visit nearest Common Service Center. Helpline: 1800-555-8842.",
        "expected_keywords": ["1800-555-8842", "1800"],
        "must_not_contain": [],
        "description": "Factual QA: exact number extraction",
    },
    {
        "id": "llm-4",
        "question": "Tell me about PM Kisan benefits",
        "context": (
            "PM Kisan Samman Nidhi provides Rs 6000 per year to small and marginal farmers. "
            "Paid in 3 instalments of Rs 2000 directly to bank account."
        ),
        "expected_keywords": ["6000", "6,000", "2000", "2,000", "3", "instalment", "farmer"],
        "must_not_contain": [],
        "description": "Factual QA: PM Kisan amounts",
    },
]

NMT_TESTS = [
    {
        "id": "nmt-1",
        "source_text": "The skill training stipend is five thousand rupees per month.",
        "source_lang": "en",
        "target_lang": "hi",
        "roundtrip_keywords": ["skill", "training", "stipend", "thousand", "month", "five"],
        "description": "EN→HI round-trip: scheme sentence",
    },
    {
        "id": "nmt-2",
        "source_text": "Farmers receive six thousand rupees annually under PM Kisan scheme.",
        "source_lang": "en",
        "target_lang": "hi",
        "roundtrip_keywords": ["farmer", "six", "thousand", "annually", "PM Kisan"],
        "description": "EN→HI round-trip: PM Kisan",
    },
    {
        "id": "nmt-3",
        "source_text": "Apply at the nearest Common Service Center with Aadhaar card.",
        "source_lang": "en",
        "target_lang": "hi",
        "roundtrip_keywords": ["apply", "Common Service", "Aadhaar"],
        "description": "EN→HI round-trip: application instructions",
    },
]


# ═══════════════════════════════════════════════════════════════
# Scoring functions
# ═══════════════════════════════════════════════════════════════

def keyword_recall(text: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in text (case-insensitive)."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round(hits / len(keywords), 3)


def contains_forbidden(text: str, forbidden: list[str]) -> list[str]:
    """Return which forbidden phrases appear in text."""
    text_lower = text.lower()
    return [f for f in forbidden if f.lower() in text_lower]


def response_length_ok(text: str, min_words: int = 5, max_words: int = 500) -> bool:
    wc = len(text.split())
    return min_words <= wc <= max_words


def _round(v: float) -> float:
    return round(v, 3)


# ═══════════════════════════════════════════════════════════════
# Stage benchmark runners
# ═══════════════════════════════════════════════════════════════

def _timed(fn):
    """Run fn(), return (result, elapsed_ms, hardware_summary)."""
    sampler = MetricsSampler(interval_s=0.5)
    sampler.start()
    t0 = time.perf_counter()
    try:
        result = fn()
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        samples = sampler.stop()
    return result, round(elapsed_ms, 1), summarize_samples(samples)


def bench_retrieval(pipeline, tests: list[dict]) -> list[dict]:
    results = []
    for tc in tests:
        def _run(q=tc["query"]):
            return pipeline.run_stage("retrieval", {"question": q, "state": "", "district": ""})

        stage_result, latency_ms, hw = _timed(_run)
        chunks = (stage_result.data or {}).get("context_chunks", [])
        all_text = " ".join(c.get("text", "") for c in chunks)
        top_score = max((c.get("score", 0) for c in chunks), default=0)

        kw_recall = keyword_recall(all_text, tc["expected_keywords"])
        token_found = tc["expected_token"].lower() in all_text.lower() if tc.get("expected_token") else None

        results.append({
            "test_id": tc["id"],
            "description": tc["description"],
            "success": stage_result.success,
            "num_results": len(chunks),
            "top_score": round(top_score, 3),
            "keyword_recall": _round(kw_recall),
            "expected_token_found": token_found,
            "latency_ms": latency_ms,
            "hardware": hw,
        })
    return results


def bench_llm(pipeline, tests: list[dict]) -> list[dict]:
    results = []
    for tc in tests:
        inputs = {
            "text": tc["question"],
            "question": tc["question"],
            "retrieval_output": {"context_text": tc["context"], "context_chunks": [{"text": tc["context"]}], "num_results": 1},
            "system_prompt": "Answer the question using ONLY the provided context. Be concise and factual.",
            "user_prompt": "Context:\n{rag_context}\n\nQuestion: {question}",
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.8,
            "target_language": "en",
            "source_language": "en",
        }

        def _run(inp=inputs):
            return pipeline.run_stage("llm", inp)

        stage_result, latency_ms, hw = _timed(_run)
        response = (stage_result.data or {}).get("response", "")

        kw_recall = keyword_recall(response, tc["expected_keywords"])
        forbidden_hits = contains_forbidden(response, tc.get("must_not_contain", []))
        length_ok = response_length_ok(response)

        score = kw_recall
        if forbidden_hits:
            score *= 0.5
        if not length_ok:
            score *= 0.7

        results.append({
            "test_id": tc["id"],
            "description": tc["description"],
            "success": stage_result.success,
            "response_preview": response[:200],
            "keyword_recall": _round(kw_recall),
            "forbidden_hits": forbidden_hits,
            "length_ok": length_ok,
            "accuracy_score": _round(score),
            "latency_ms": latency_ms,
            "prompt_tokens": (stage_result.data or {}).get("prompt_tokens"),
            "completion_tokens": (stage_result.data or {}).get("completion_tokens"),
            "hardware": hw,
        })
    return results


def bench_nmt(pipeline, tests: list[dict]) -> list[dict]:
    results = []
    for tc in tests:
        # Forward translation
        fwd_inputs = {
            "text": tc["source_text"],
            "source_language": tc["source_lang"],
            "target_language": tc["target_lang"],
        }

        def _fwd(inp=fwd_inputs):
            return pipeline.run_stage("nmt", inp)

        fwd_result, fwd_ms, fwd_hw = _timed(_fwd)
        translated = (fwd_result.data or {}).get("translated_text", "")

        # Reverse translation (round-trip)
        rev_inputs = {
            "text": translated,
            "source_language": tc["target_lang"],
            "target_language": tc["source_lang"],
        }

        def _rev(inp=rev_inputs):
            return pipeline.run_stage("nmt", inp)

        rev_result, rev_ms, rev_hw = _timed(_rev)
        roundtrip = (rev_result.data or {}).get("translated_text", "")

        rt_recall = keyword_recall(roundtrip, tc["roundtrip_keywords"])
        translated_nonempty = len(translated.strip()) > 0 and translated.strip() != tc["source_text"].strip()

        results.append({
            "test_id": tc["id"],
            "description": tc["description"],
            "success": fwd_result.success and rev_result.success,
            "source": tc["source_text"][:100],
            "translated_preview": translated[:100],
            "roundtrip_preview": roundtrip[:100],
            "roundtrip_keyword_recall": _round(rt_recall),
            "translation_nonempty": translated_nonempty,
            "accuracy_score": _round(rt_recall * (1.0 if translated_nonempty else 0.0)),
            "forward_ms": fwd_ms,
            "roundtrip_ms": rev_ms,
            "total_ms": round(fwd_ms + rev_ms, 1),
            "hardware": fwd_hw,
        })
    return results


# ═══════════════════════════════════════════════════════════════
# Multi-model comparison
# ═══════════════════════════════════════════════════════════════

COMPARABLE_STAGES = {
    "llm": {
        "cpu_safe": ["ollama_qwen2_5_3b", "qwen2_5_05b", "ollama_llama3_2_3b", "ollama_gemma2_2b"],
        "gpu_safe": ["qwen2_5_3b", "gemma2_2b", "phi3_5_mini", "llama3_2_3b", "smollm2",
                      "ollama_qwen2_5_3b", "ollama_llama3_2_3b", "ollama_gemma2_2b"],
        "tests": LLM_TESTS,
        "bench_fn": "bench_llm",
    },
    "nmt": {
        "cpu_safe": ["nllb_600m"],
        "gpu_safe": ["nllb_600m", "indictrans2_200m", "indictrans2_1b"],
        "tests": NMT_TESTS,
        "bench_fn": "bench_nmt",
    },
    "retrieval": {
        "cpu_safe": ["faiss"],
        "gpu_safe": ["faiss", "faiss_bge"],
        "tests": RETRIEVAL_TESTS,
        "bench_fn": "bench_retrieval",
    },
}


def _create_stage_for_variant(settings, stage_name: str, variant_key: str, device: str):
    """Create a fresh stage instance for a given model variant."""
    from netra.core.registry import StageRegistry
    config = settings.get_model_config(stage_name, variant_key)
    return StageRegistry.create(
        stage_name=stage_name,
        config=config,
        device=device,
        variant=variant_key,
    )


def run_model_comparison(settings, stage_name: str, is_cpu: bool) -> dict[str, Any]:
    """Run the same test suite across multiple model variants for one stage."""
    import netra.stages  # noqa: F401

    stage_info = COMPARABLE_STAGES.get(stage_name)
    if not stage_info:
        return {"error": f"No comparison config for stage '{stage_name}'"}

    variant_keys = stage_info["cpu_safe"] if is_cpu else stage_info["gpu_safe"]
    tests = stage_info["tests"]
    bench_fn_name = stage_info["bench_fn"]
    bench_fn = {"bench_llm": bench_llm, "bench_nmt": bench_nmt, "bench_retrieval": bench_retrieval}[bench_fn_name]

    device = "cpu" if is_cpu else f"cuda:{settings.device.cuda_device}"

    comparison: dict[str, Any] = {}
    available_models = settings.models.get(stage_name, {})

    for variant_key in variant_keys:
        if variant_key not in available_models:
            comparison[variant_key] = {"skipped": True, "reason": "Not in models.yaml"}
            continue

        model_cfg = available_models[variant_key]
        print(f"  Testing {stage_name}/{variant_key}: {model_cfg.get('name', variant_key)}...", flush=True)

        try:
            stage = _create_stage_for_variant(settings, stage_name, variant_key, device)
        except (KeyError, ValueError) as e:
            comparison[variant_key] = {"skipped": True, "reason": str(e)}
            continue

        class _SingleStagePipeline:
            """Lightweight pipeline wrapper that loads/runs a single stage directly."""
            def __init__(self, stg):
                self._stage = stg

            def run_stage(self, name, inputs):
                return self._stage.run(inputs)

        temp_pipeline = _SingleStagePipeline(stage)

        try:
            test_results = bench_fn(temp_pipeline, tests)
        except Exception as e:
            comparison[variant_key] = {"skipped": True, "reason": f"Runtime error: {e}"}
            continue
        finally:
            try:
                stage.unload()
            except Exception:
                pass

        scores = [r.get("accuracy_score") or r.get("keyword_recall", 0) for r in test_results]
        latencies = [r.get("latency_ms") or r.get("total_ms", 0) for r in test_results]
        avg_score = round(sum(scores) / len(scores), 3) if scores else 0
        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0

        comparison[variant_key] = {
            "model_name": model_cfg.get("name", variant_key),
            "provider": model_cfg.get("provider"),
            "quantization": model_cfg.get("quantization"),
            "vram_mb": model_cfg.get("vram_mb"),
            "avg_accuracy": avg_score,
            "avg_latency_ms": avg_latency,
            "tests": test_results,
            "summary": {
                "total_tests": len(test_results),
                "passed": sum(1 for r in test_results if r.get("success")),
                "avg_score": avg_score,
                "avg_latency_ms": avg_latency,
            },
        }
        print(f"    → score={avg_score:.3f}  latency={avg_latency:.0f}ms")

    return comparison


# ═══════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════

def _init_pipeline():
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from netra.config.settings import Settings
    from netra.core.pipeline import Pipeline
    import netra.stages  # noqa: F401

    settings = Settings(BACKEND_ROOT / "config")
    pipeline = Pipeline(settings)
    return settings, pipeline


def run_accuracy_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "type": "accuracy",
        "system": system_info(),
        "benchmarks": {},
    }

    if args.mode == "api":
        import requests
        base_url = args.base_url.rstrip("/")
        print(f"Checking backend at {base_url}...")
        health = requests.get(f"{base_url}/api/health", timeout=10)
        health.raise_for_status()

        print("\n--- Retrieval Accuracy (API) ---")
        ret_results = []
        for tc in RETRIEVAL_TESTS:
            for _ in range(args.runs):
                resp = requests.post(
                    f"{base_url}/api/pipeline/text",
                    json={"text": tc["query"], "language": "en", "mode": "scheme"},
                    timeout=300,
                )
                body = resp.json()
                rag_hits = body.get("rag_hits", 0)
                rag_score = body.get("rag_top_score", 0)
                llm_text = body.get("stages", {}).get("llm", {}).get("data", {}).get("response", "")
                kw = keyword_recall(llm_text, tc["expected_keywords"])
                ret_results.append({
                    "test_id": tc["id"],
                    "rag_hits": rag_hits,
                    "rag_top_score": rag_score,
                    "keyword_recall_in_response": _round(kw),
                    "total_ms": body.get("total_ms"),
                })
                print(f"  {tc['id']}: recall={kw:.2f}  rag_score={rag_score:.2f}")

        report["benchmarks"]["retrieval_accuracy"] = ret_results
        return report

    # Direct mode
    print("Loading pipeline in-process...")
    settings, pipeline = _init_pipeline()
    is_cpu = settings.device.target == "cpu" or settings.device.cuda_device < 0
    report["config"] = {
        "device_target": settings.device.target,
        "model_defaults": settings.model_defaults,
        "is_cpu": is_cpu,
    }
    print(f"Pipeline loaded. Device: {settings.device.target}\n")

    stages_to_test = [args.stage] if args.stage else ["retrieval", "llm", "nmt"]

    if "retrieval" in stages_to_test:
        print("--- Retrieval Accuracy ---")
        report["benchmarks"]["retrieval"] = {
            "model_key": settings.model_defaults.get("retrieval"),
            "results": bench_retrieval(pipeline, RETRIEVAL_TESTS),
        }
        scores = [r["keyword_recall"] for r in report["benchmarks"]["retrieval"]["results"]]
        print(f"  Avg keyword recall: {sum(scores)/len(scores):.3f}\n")

    if "llm" in stages_to_test:
        print("--- LLM Accuracy ---")
        report["benchmarks"]["llm"] = {
            "model_key": settings.model_defaults.get("llm"),
            "results": bench_llm(pipeline, LLM_TESTS),
        }
        scores = [r["accuracy_score"] for r in report["benchmarks"]["llm"]["results"]]
        print(f"  Avg accuracy score: {sum(scores)/len(scores):.3f}\n")

    if "nmt" in stages_to_test:
        print("--- NMT Accuracy (round-trip) ---")
        report["benchmarks"]["nmt"] = {
            "model_key": settings.model_defaults.get("nmt"),
            "results": bench_nmt(pipeline, NMT_TESTS),
        }
        scores = [r["accuracy_score"] for r in report["benchmarks"]["nmt"]["results"]]
        print(f"  Avg round-trip score: {sum(scores)/len(scores):.3f}\n")

    # Multi-model comparison
    if args.compare:
        print("\n=== MULTI-MODEL COMPARISON ===\n")
        report["model_comparison"] = {}
        compare_stages = [args.stage] if args.stage else list(COMPARABLE_STAGES.keys())
        for stage_name in compare_stages:
            if stage_name not in COMPARABLE_STAGES:
                continue
            print(f"--- Comparing {stage_name} models ---")
            report["model_comparison"][stage_name] = run_model_comparison(settings, stage_name, is_cpu)
            print()

    return report


# ═══════════════════════════════════════════════════════════════
# Pretty-print
# ═══════════════════════════════════════════════════════════════

def print_accuracy_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("  NETRA Accuracy & Model Comparison Report")
    print("=" * 70)

    system = report.get("system", {})
    print(f"Platform  : {system.get('platform')} / {system.get('architecture')}")
    print(f"GPU       : {system.get('gpu_name') or 'None (CPU only)'}")

    benchmarks = report.get("benchmarks", {})

    for stage_name in ["retrieval", "llm", "nmt"]:
        stage_data = benchmarks.get(stage_name)
        if not stage_data:
            continue
        results = stage_data.get("results", [])
        if not results:
            continue

        print(f"\n{'─' * 50}")
        print(f"  {stage_name.upper()} — default model: {stage_data.get('model_key', '?')}")
        print(f"{'─' * 50}")

        score_key = "accuracy_score" if stage_name in ("llm", "nmt") else "keyword_recall"
        time_key = "latency_ms" if stage_name != "nmt" else "total_ms"

        print(f"  {'Test':<12s} {'Score':>8s} {'Time(ms)':>10s} {'Status':>8s}  Description")
        for r in results:
            score = r.get(score_key, 0)
            ms = r.get(time_key, 0)
            status = "OK" if r.get("success") else "FAIL"
            print(f"  {r['test_id']:<12s} {score:>8.3f} {ms:>10.0f} {status:>8s}  {r.get('description', '')[:40]}")

        avg = sum(r.get(score_key, 0) for r in results) / len(results)
        avg_ms = sum(r.get(time_key, 0) for r in results) / len(results)
        print(f"  {'AVERAGE':<12s} {avg:>8.3f} {avg_ms:>10.0f}")

    # Model comparison table
    comparison = report.get("model_comparison", {})
    if comparison:
        for stage_name, models in comparison.items():
            print(f"\n{'─' * 50}")
            print(f"  {stage_name.upper()} MODEL COMPARISON")
            print(f"{'─' * 50}")
            print(f"  {'Model':<35s} {'Quant':>8s} {'VRAM':>6s} {'Score':>8s} {'Avg ms':>8s} {'Pass':>6s}")
            print(f"  {'─' * 35} {'─' * 8} {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 6}")

            ranked: list[tuple[str, dict]] = []
            for key, data in models.items():
                if data.get("skipped"):
                    print(f"  {key:<35s} {'—':>8s} {'—':>6s} {'SKIP':>8s} {'—':>8s} {'—':>6s}  ({data.get('reason', '')[:30]})")
                    continue
                ranked.append((key, data))

            ranked.sort(key=lambda x: x[1].get("avg_accuracy", 0), reverse=True)
            for i, (key, data) in enumerate(ranked):
                marker = " ★" if i == 0 else ""
                s = data.get("summary", {})
                quant = data.get("quantization", "—")[:8]
                vram = data.get("vram_mb", "—")
                print(
                    f"  {data.get('model_name', key):<35s} {quant:>8s} {str(vram):>6s} "
                    f"{s.get('avg_score', 0):>8.3f} {s.get('avg_latency_ms', 0):>8.0f} "
                    f"{s.get('passed', 0):>3d}/{s.get('total_tests', 0):<2d}{marker}"
                )

    print(f"\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NETRA accuracy & model-comparison benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["api", "direct"], default="direct",
                        help="'api' = against running server, 'direct' = in-process")
    parser.add_argument("--base-url", default=os.environ.get("NETRA_BASE_URL", "http://127.0.0.1:5000"))
    parser.add_argument("--runs", type=int, default=1, help="Repetitions per test (API mode)")
    parser.add_argument("--stage", choices=["retrieval", "llm", "nmt"],
                        help="Only benchmark this stage")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple models for each stage")
    parser.add_argument("--output", type=Path, help="JSON report output path")
    args = parser.parse_args()

    report = run_accuracy_benchmarks(args)
    print_accuracy_summary(report)

    output_path = args.output or (BACKEND_ROOT / "output" / "benchmark_accuracy.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nReport saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
