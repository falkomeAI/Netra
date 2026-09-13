#!/usr/bin/env python3
"""
NETRA application performance benchmark.

Measures:
  - CPU utilization & temperature during each operation
  - GPU utilization & temperature during each operation
  - Model-wise inference performance (every active + optional stage)
  - Data ingestion time (embedding + knowledge_graph)
  - Query search time (retrieval + LLM + NMT pipeline)

Modes:
  api    — benchmark against a running backend (start.sh / server.py first)
  direct — load pipeline in-process, measures cold-start vs warm timing

Examples:
    cd Backend
    python scripts/benchmark_performance.py --mode api --runs 3
    python scripts/benchmark_performance.py --mode direct --model-perf --runs 3
    python scripts/benchmark_performance.py --mode api --query-only --runs 5
    python scripts/benchmark_performance.py --output output/benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from netra.utils.hardware_metrics import (  # noqa: E402
    MetricsSampler,
    summarize_samples,
    system_info,
)

DEFAULT_BASE_URL = os.environ.get("NETRA_BASE_URL", "http://127.0.0.1:5000")

INGEST_TEXTS = [
    (
        "Benchmark ingest document NRSSY-BENCH.\n"
        "Skill training stipend Rs 5000 per month for 6 months.\n"
        "Tool kit grant Rs 12000 after completion.\n"
        "Unique token: benchmark-token-88421."
    ),
    (
        "PM Kisan Samman Nidhi Yojana provides Rs 6000 per year.\n"
        "Paid in 3 instalments of Rs 2000 directly to farmer bank account.\n"
        "Eligibility: small and marginal farmers.\n"
        "Token: benchmark-pmkisan-99302."
    ),
]

QUERY_TEXTS = [
    "What is the skill training stipend under NRSSY scheme?",
    "How much money does PM Kisan give per year?",
    "Tell me about rural skill development programs",
]


def _round_ms(value: float | None) -> float | None:
    return round(value, 1) if value is not None else None


def _aggregate(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "avg": None, "p50": None, "p95": None, "count": 0}
    s = sorted(values)
    n = len(s)
    return {
        "min": round(s[0], 1),
        "max": round(s[-1], 1),
        "avg": round(sum(s) / n, 1),
        "p50": round(s[n // 2], 1),
        "p95": round(s[int(n * 0.95)], 1) if n >= 2 else round(s[-1], 1),
        "count": n,
    }


def _aggregate_runs(runs: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [r[key] for r in runs if r.get(key) is not None]
    return _aggregate(values)


def _aggregate_stage_latencies(runs: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    stage_names: set[str] = set()
    for run in runs:
        stage_names.update(run.get("stages", {}).keys())
    summary: dict[str, dict[str, float | None]] = {}
    for stage in sorted(stage_names):
        values = [
            run["stages"][stage]["latency_ms"]
            for run in runs
            if stage in run.get("stages", {}) and run["stages"][stage].get("latency_ms") is not None
        ]
        summary[stage] = _aggregate(values)
    return summary


def _aggregate_hardware(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge hardware summaries across multiple runs."""
    from netra.utils.hardware_metrics import HardwareSample, _METRIC_FIELDS, _stats_for

    all_vals: dict[str, list[float]] = {f: [] for f in _METRIC_FIELDS}
    for run in runs:
        hw = run.get("hardware", {})
        for f in _METRIC_FIELDS:
            stats = hw.get(f, {})
            if stats.get("avg") is not None:
                all_vals[f].append(stats["avg"])
    return {f: _stats_for(v).to_dict() for f, v in all_vals.items()}


def _timed_with_metrics(
    fn: Callable[[], Any],
    sample_interval: float = 0.5,
) -> tuple[Any, float, dict[str, Any]]:
    """Run fn() while sampling hardware metrics in a background thread."""
    sampler = MetricsSampler(interval_s=sample_interval)
    sampler.start()
    t0 = time.perf_counter()
    try:
        result = fn()
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        samples = sampler.stop()
    return result, elapsed_ms, summarize_samples(samples)


# ═══════════════════════════════════════════════════════════════
# API mode benchmarks (against running server)
# ═══════════════════════════════════════════════════════════════

def benchmark_ingestion_api(base_url: str, text: str) -> dict[str, Any]:
    import requests

    def _call():
        resp = requests.post(
            f"{base_url}/api/ingest",
            json={"text": text},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    body, elapsed_ms, hardware = _timed_with_metrics(_call)
    stages = {
        name: {
            "success": data.get("success"),
            "latency_ms": data.get("latency_ms"),
        }
        for name, data in body.get("stages", {}).items()
    }
    return {
        "operation": "data_ingestion",
        "total_ms": _round_ms(body.get("total_ms", elapsed_ms)),
        "wall_ms": _round_ms(elapsed_ms),
        "success": body.get("success", False),
        "stages": stages,
        "hardware": hardware,
    }


def benchmark_query_api(base_url: str, query: str) -> dict[str, Any]:
    import requests

    def _call():
        resp = requests.post(
            f"{base_url}/api/pipeline/text",
            json={"text": query, "language": "hi", "mode": "auto"},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    body, elapsed_ms, hardware = _timed_with_metrics(_call)
    stages = {
        name: {
            "success": data.get("success"),
            "latency_ms": data.get("latency_ms"),
        }
        for name, data in body.get("stages", {}).items()
    }
    retrieval_ms = stages.get("retrieval", {}).get("latency_ms")
    llm_ms = stages.get("llm", {}).get("latency_ms")
    return {
        "operation": "query_search",
        "total_ms": _round_ms(body.get("total_ms", elapsed_ms)),
        "wall_ms": _round_ms(elapsed_ms),
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "prompt_used": body.get("prompt_used"),
        "rag_hits": body.get("rag_hits"),
        "rag_top_score": body.get("rag_top_score"),
        "success": all(s.get("success", False) for s in stages.values()) if stages else False,
        "stages": stages,
        "hardware": hardware,
    }


# ═══════════════════════════════════════════════════════════════
# Direct mode benchmarks (in-process pipeline)
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


def benchmark_ingestion_direct(pipeline, text: str) -> dict[str, Any]:
    inputs = {
        "text": text,
        "ocr_output": {
            "text": text,
            "lines": text.split("\n"),
            "num_lines": len(text.split("\n")),
        },
        "target_language": "hi",
        "source_language": "hi",
    }

    def _call():
        return pipeline.run(inputs, stages=["embedding", "knowledge_graph"])

    results, elapsed_ms, hardware = _timed_with_metrics(_call)
    stages = {}
    for name, result in results.items():
        stages[name] = {
            "success": result.success,
            "latency_ms": _round_ms(result.latency_ms),
            "error": result.error,
        }
    return {
        "operation": "data_ingestion",
        "total_ms": _round_ms(elapsed_ms),
        "wall_ms": _round_ms(elapsed_ms),
        "success": all(r.success for r in results.values()),
        "stages": stages,
        "hardware": hardware,
    }


def benchmark_query_direct(pipeline, query: str) -> dict[str, Any]:
    retrieval_inputs = {"question": query, "state": "", "district": ""}

    def _retrieval():
        return pipeline.run_stage("retrieval", retrieval_inputs)

    retrieval_result, retrieval_ms, retrieval_hw = _timed_with_metrics(_retrieval)

    inputs = {
        "text": query,
        "question": query,
        "target_language": "hi",
        "source_language": "hi",
        "retrieval_output": retrieval_result.data if retrieval_result.success else {},
        "system_prompt": "Answer using the provided context.",
        "user_prompt": "Question: {question}\n\nContext:\n{context}",
        "max_new_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.8,
    }

    def _pipeline():
        return pipeline.run(inputs, stages=["retrieval", "llm", "nmt"])

    results, pipeline_ms, pipeline_hw = _timed_with_metrics(_pipeline)
    stages = {}
    for name, result in results.items():
        stages[name] = {
            "success": result.success,
            "latency_ms": _round_ms(result.latency_ms),
            "error": result.error,
        }

    return {
        "operation": "query_search",
        "retrieval_only_ms": _round_ms(retrieval_ms),
        "pipeline_ms": _round_ms(pipeline_ms),
        "total_ms": _round_ms(retrieval_ms + pipeline_ms),
        "retrieval_hits": (retrieval_result.data or {}).get("num_results", 0),
        "success": retrieval_result.success and all(r.success for r in results.values()),
        "stages": stages,
        "hardware": {
            "retrieval": retrieval_hw,
            "pipeline": pipeline_hw,
        },
    }


def benchmark_model_performance(settings, pipeline) -> dict[str, Any]:
    """Run each active + optional stage once, recording model name, latency, and hardware."""
    model_defaults = settings.model_defaults
    results: dict[str, Any] = {}

    stage_inputs: dict[str, dict[str, Any]] = {
        "preprocessing": {
            "image_path": str(BACKEND_ROOT / "tests" / "fixtures" / "sample_ocr_doc.png"),
        },
        "ocr": {
            "image_path": str(BACKEND_ROOT / "tests" / "fixtures" / "sample_ocr_doc.png"),
            "target_language": "hi",
            "source_language": "hi",
        },
        "retrieval": {"question": QUERY_TEXTS[0]},
        "llm": {
            "text": QUERY_TEXTS[0],
            "question": QUERY_TEXTS[0],
            "system_prompt": "You are a helpful assistant.",
            "user_prompt": "{question}",
            "max_new_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.9,
        },
        "nmt": {
            "text": "Hello, how are you? This is a test sentence for translation benchmark.",
            "source_language": "en",
            "target_language": "hi",
        },
        "tts": {
            "text": "यह एक परीक्षण वाक्य है।",
            "target_language": "hi",
            "output_dir": str(BACKEND_ROOT / "output"),
        },
        "embedding": {
            "text": INGEST_TEXTS[0],
            "ocr_output": {"text": INGEST_TEXTS[0]},
            "target_language": "hi",
            "source_language": "hi",
        },
        "knowledge_graph": {
            "text": INGEST_TEXTS[0],
            "ocr_output": {"text": INGEST_TEXTS[0]},
        },
    }

    all_stages = list(dict.fromkeys(settings.active_stages + settings.optional_stages))

    for stage_name in all_stages:
        if stage_name not in stage_inputs:
            continue
        model_key = model_defaults.get(stage_name, "")
        try:
            model_cfg = settings.get_model_config(stage_name, model_key)
        except (KeyError, ValueError):
            continue

        inputs = stage_inputs[stage_name]

        # Check if test fixture exists for image stages
        if "image_path" in inputs and not Path(inputs["image_path"]).exists():
            results[stage_name] = {
                "model_key": model_key,
                "model_name": model_cfg.get("name", model_key),
                "provider": model_cfg.get("provider"),
                "success": False,
                "error": f"Test fixture not found: {inputs['image_path']}",
                "skipped": True,
            }
            continue

        def _run_stage(name=stage_name, inp=inputs):
            return pipeline.run_stage(name, inp)

        # Cold run (includes model load if not yet loaded)
        cold_result, cold_ms, cold_hw = _timed_with_metrics(_run_stage)

        # Warm run (model already loaded)
        warm_result, warm_ms, warm_hw = _timed_with_metrics(_run_stage)

        results[stage_name] = {
            "model_key": model_key,
            "model_name": model_cfg.get("name", model_key),
            "provider": model_cfg.get("provider"),
            "quantization": model_cfg.get("quantization"),
            "vram_mb": model_cfg.get("vram_mb"),
            "success": warm_result.success,
            "cold_start_ms": _round_ms(cold_ms),
            "warm_inference_ms": _round_ms(warm_ms),
            "latency_ms": _round_ms(warm_result.latency_ms or warm_ms),
            "error": warm_result.error,
            "hardware_cold": cold_hw,
            "hardware_warm": warm_hw,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════

def run_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "runs_per_test": args.runs,
        "system": system_info(),
        "benchmarks": {},
    }

    if args.mode == "api":
        import requests
        base_url = args.base_url.rstrip("/")
        print(f"Checking backend health at {base_url}...")
        health = requests.get(f"{base_url}/api/health", timeout=10)
        health.raise_for_status()
        print("Backend is healthy. Starting benchmarks...\n")

        if not args.query_only:
            print("--- Data Ingestion Benchmark ---")
            ingest_runs: list[dict[str, Any]] = []
            for i in range(args.runs):
                text = INGEST_TEXTS[i % len(INGEST_TEXTS)]
                print(f"  Run {i + 1}/{args.runs}...", end=" ", flush=True)
                result = benchmark_ingestion_api(base_url, text)
                ingest_runs.append(result)
                print(f"{result['wall_ms']} ms ({'OK' if result['success'] else 'FAIL'})")
            report["benchmarks"]["data_ingestion"] = {
                "runs": ingest_runs,
                "summary": {
                    "total_ms": _aggregate_runs(ingest_runs, "total_ms"),
                    "wall_ms": _aggregate_runs(ingest_runs, "wall_ms"),
                    "stages": _aggregate_stage_latencies(ingest_runs),
                    "hardware": _aggregate_hardware(ingest_runs),
                },
            }

        if not args.ingest_only:
            print("\n--- Query Search Benchmark ---")
            query_runs: list[dict[str, Any]] = []
            for i in range(args.runs):
                query = QUERY_TEXTS[i % len(QUERY_TEXTS)]
                print(f"  Run {i + 1}/{args.runs}...", end=" ", flush=True)
                result = benchmark_query_api(base_url, query)
                query_runs.append(result)
                print(f"{result['wall_ms']} ms ({'OK' if result['success'] else 'FAIL'})")
            report["benchmarks"]["query_search"] = {
                "runs": query_runs,
                "summary": {
                    "total_ms": _aggregate_runs(query_runs, "total_ms"),
                    "wall_ms": _aggregate_runs(query_runs, "wall_ms"),
                    "retrieval_ms": _aggregate_runs(query_runs, "retrieval_ms"),
                    "llm_ms": _aggregate_runs(query_runs, "llm_ms"),
                    "stages": _aggregate_stage_latencies(query_runs),
                    "hardware": _aggregate_hardware(query_runs),
                },
            }
        return report

    # Direct mode
    print("Loading pipeline in-process...")
    settings, pipeline = _init_pipeline()
    report["config"] = {
        "device_target": settings.device.target,
        "active_stages": settings.active_stages,
        "optional_stages": settings.optional_stages,
        "model_defaults": settings.model_defaults,
    }
    print(f"Pipeline loaded. Device: {settings.device.target}\n")

    if not args.query_only:
        print("--- Data Ingestion Benchmark ---")
        ingest_runs = []
        for i in range(args.runs):
            text = INGEST_TEXTS[i % len(INGEST_TEXTS)]
            print(f"  Run {i + 1}/{args.runs}...", end=" ", flush=True)
            result = benchmark_ingestion_direct(pipeline, text)
            ingest_runs.append(result)
            print(f"{result['wall_ms']} ms ({'OK' if result['success'] else 'FAIL'})")
        report["benchmarks"]["data_ingestion"] = {
            "runs": ingest_runs,
            "summary": {
                "total_ms": _aggregate_runs(ingest_runs, "total_ms"),
                "wall_ms": _aggregate_runs(ingest_runs, "wall_ms"),
                "stages": _aggregate_stage_latencies(ingest_runs),
                "hardware": _aggregate_hardware(ingest_runs),
            },
        }

    if not args.ingest_only:
        print("\n--- Query Search Benchmark ---")
        query_runs = []
        for i in range(args.runs):
            query = QUERY_TEXTS[i % len(QUERY_TEXTS)]
            print(f"  Run {i + 1}/{args.runs}...", end=" ", flush=True)
            result = benchmark_query_direct(pipeline, query)
            query_runs.append(result)
            print(f"{result['total_ms']} ms ({'OK' if result['success'] else 'FAIL'})")
        report["benchmarks"]["query_search"] = {
            "runs": query_runs,
            "summary": {
                "total_ms": _aggregate_runs(query_runs, "total_ms"),
                "retrieval_only_ms": _aggregate_runs(query_runs, "retrieval_only_ms"),
                "pipeline_ms": _aggregate_runs(query_runs, "pipeline_ms"),
                "stages": _aggregate_stage_latencies(query_runs),
                "hardware": _aggregate_hardware(query_runs),
            },
        }

    if args.model_perf:
        print("\n--- Model-wise Performance ---")
        report["benchmarks"]["model_performance"] = benchmark_model_performance(settings, pipeline)

    return report


# ═══════════════════════════════════════════════════════════════
# Pretty-print summary
# ═══════════════════════════════════════════════════════════════

def _hw_line(hw: dict[str, Any]) -> str:
    """Format one-line hardware summary from aggregated stats."""
    parts: list[str] = []
    cpu_u = hw.get("cpu_util_pct", {})
    cpu_t = hw.get("cpu_temp_c", {})
    gpu_u = hw.get("gpu_util_pct", {})
    gpu_t = hw.get("gpu_temp_c", {})
    ram = hw.get("ram_used_mb", {})

    if cpu_u.get("avg") is not None:
        parts.append(f"CPU {cpu_u['avg']}%")
    if cpu_t.get("max") is not None:
        parts.append(f"@ {cpu_t['max']}°C")
    if ram.get("avg") is not None:
        parts.append(f"RAM {ram['avg']}MB")
    if gpu_u.get("avg") is not None:
        parts.append(f"| GPU {gpu_u['avg']}%")
    if gpu_t.get("max") is not None:
        parts.append(f"@ {gpu_t['max']}°C")
    return " ".join(parts) if parts else "N/A"


def print_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("  NETRA Performance Benchmark Report")
    print("=" * 60)
    print(f"Mode      : {report.get('mode')}")
    print(f"Runs      : {report.get('runs_per_test')}")
    print(f"Generated : {report.get('generated_at')}")

    system = report.get("system", {})
    print(f"Platform  : {system.get('platform')} / {system.get('architecture')}")
    print(f"CPU cores : {system.get('cpu_count', 'N/A')}")
    print(f"RAM total : {system.get('ram_total_mb', 'N/A')} MB")
    print(f"GPU       : {system.get('gpu_name') or 'None'}")

    baseline = system.get("baseline", {})
    bl_parts: list[str] = []
    if baseline.get("cpu_util_pct") is not None:
        bl_parts.append(f"CPU {baseline['cpu_util_pct']}%")
    if baseline.get("cpu_temp_c") is not None:
        bl_parts.append(f"@ {baseline['cpu_temp_c']}°C")
    if baseline.get("gpu_util_pct") is not None:
        bl_parts.append(f"| GPU {baseline['gpu_util_pct']}%")
    if baseline.get("gpu_temp_c") is not None:
        bl_parts.append(f"@ {baseline['gpu_temp_c']}°C")
    print(f"Baseline  : {' '.join(bl_parts) if bl_parts else 'N/A'}")

    benchmarks = report.get("benchmarks", {})

    # Data Ingestion
    ingest = benchmarks.get("data_ingestion", {}).get("summary", {})
    if ingest:
        total = ingest.get("total_ms", {})
        print(f"\n{'─' * 40}")
        print("  DATA INGESTION")
        print(f"{'─' * 40}")
        print(f"  Total    : avg {total.get('avg')} ms  (min {total.get('min')}, max {total.get('max')}, p50 {total.get('p50')})")
        for stage, stats in ingest.get("stages", {}).items():
            print(f"    {stage:18s}: avg {stats.get('avg'):>8} ms  (p95 {stats.get('p95')})")
        hw = ingest.get("hardware", {})
        if hw:
            print(f"  Hardware : {_hw_line(hw)}")

    # Query Search
    query = benchmarks.get("query_search", {}).get("summary", {})
    if query:
        total = query.get("total_ms", {})
        print(f"\n{'─' * 40}")
        print("  QUERY SEARCH")
        print(f"{'─' * 40}")
        print(f"  Total    : avg {total.get('avg')} ms  (min {total.get('min')}, max {total.get('max')}, p50 {total.get('p50')})")
        ret = query.get("retrieval_ms") or query.get("retrieval_only_ms")
        if ret and ret.get("avg") is not None:
            print(f"  Retrieval: avg {ret['avg']} ms  (min {ret.get('min')}, max {ret.get('max')})")
        llm = query.get("llm_ms")
        if llm and llm.get("avg") is not None:
            print(f"  LLM      : avg {llm['avg']} ms  (min {llm.get('min')}, max {llm.get('max')})")
        for stage, stats in query.get("stages", {}).items():
            print(f"    {stage:18s}: avg {stats.get('avg'):>8} ms  (p95 {stats.get('p95')})")
        hw = query.get("hardware", {})
        if hw:
            print(f"  Hardware : {_hw_line(hw)}")

    # Model-wise Performance
    models = benchmarks.get("model_performance", {})
    if models:
        print(f"\n{'─' * 40}")
        print("  MODEL-WISE PERFORMANCE")
        print(f"{'─' * 40}")
        print(f"  {'Stage':<16s} {'Model':<35s} {'Cold(ms)':>10s} {'Warm(ms)':>10s} {'Status'}")
        print(f"  {'─' * 16} {'─' * 35} {'─' * 10} {'─' * 10} {'─' * 6}")
        for stage, data in models.items():
            if data.get("skipped"):
                print(f"  {stage:<16s} {data.get('model_name', '?'):<35s} {'—':>10s} {'—':>10s} SKIP")
                continue
            status = "OK" if data.get("success") else "FAIL"
            cold = data.get("cold_start_ms", "—")
            warm = data.get("warm_inference_ms", "—")
            print(f"  {stage:<16s} {data.get('model_name', '?'):<35s} {str(cold):>10s} {str(warm):>10s} {status}")

            hw_warm = data.get("hardware_warm", {})
            if hw_warm:
                print(f"    └─ {_hw_line(hw_warm)}")

    print(f"\n{'=' * 60}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NETRA application performance benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["api", "direct"], default="api",
                        help="'api' = against running server, 'direct' = in-process pipeline")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="Backend URL for api mode")
    parser.add_argument("--runs", type=int, default=3,
                        help="Repeat each benchmark N times (default: 3)")
    parser.add_argument("--output", type=Path,
                        help="Write JSON report to this path")
    parser.add_argument("--ingest-only", action="store_true",
                        help="Only run data ingestion benchmark")
    parser.add_argument("--query-only", action="store_true",
                        help="Only run query search benchmark")
    parser.add_argument("--model-perf", action="store_true",
                        help="Direct mode: benchmark each stage/model individually")
    args = parser.parse_args()

    if args.ingest_only and args.query_only:
        parser.error("Use only one of --ingest-only or --query-only")
    if args.model_perf and args.mode != "direct":
        parser.error("--model-perf requires --mode direct")

    report = run_benchmarks(args)
    print_summary(report)

    output_path = args.output or (BACKEND_ROOT / "output" / "benchmark_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nReport saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
