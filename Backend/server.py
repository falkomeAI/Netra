#!/usr/bin/env python3
"""
NETRA Backend — Flask ML API Server
Loads the pipeline once at startup and serves REST APIs for all ML operations.

Run:
    cd Backend
    python server.py

Endpoints:
    GET  /api/health          → Server status
    GET  /api/config          → Pipeline configuration
    POST /api/pipeline/run    → Image/Audio/PDF pipeline
    POST /api/pipeline/text   → Text pipeline
    POST /api/ingest          → Ingest data into KB
    GET  /api/graph           → Knowledge graph data
    GET  /api/output/<file>   → Download output files
"""

from __future__ import annotations

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import json
import sys
import time
import logging
from pathlib import Path
from typing import Any

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

UPLOAD_DIR = BACKEND_ROOT / "uploads"
OUTPUT_DIR = BACKEND_ROOT / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("netra.server")

pipeline = None
settings = None
doc_watcher = None


def require_pipeline(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if pipeline is None or settings is None:
            return jsonify({"error": "Pipeline not initialized"}), 503
        return f(*args, **kwargs)
    return decorated


# ── Language / Location helpers ───────────────────────────────

from netra.utils.language import detect_language  # noqa: E402
from netra.utils.location import get_state_from_pincode  # noqa: E402
from netra.utils.readiness import build_readiness  # noqa: E402


def _request_geo(data: dict | None = None) -> tuple[str, str]:
    """Read optional state/district from JSON body or multipart form."""
    data = data or {}
    state = (
        data.get("state")
        or request.form.get("state")
        or request.args.get("state")
        or ""
    )
    district = (
        data.get("district")
        or request.form.get("district")
        or request.args.get("district")
        or ""
    )
    return str(state).strip(), str(district).strip()


def _ocr_lang_for_ingest(hint: str | None = None) -> str:
    """Language for OCR during ingest — never hardcode English-only."""
    lang = (hint or request.form.get("language") or request.form.get("lang") or "").strip()
    if lang and lang != "auto":
        return lang
    return "hi"  # rural default; EasyOCR/Paddle Indic-friendly


# ── ASR (Speech to Text) ─────────────────────────────────────

_whisper_model = None


def _get_whisper():
    """Load Whisper model once, cache for all requests."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("Systran/faster-whisper-small", device="cpu", compute_type="int8")
        logger.info("Whisper ASR model loaded")
    return _whisper_model


def _is_meaningful_transcript(text: str) -> bool:
    """Reject only true garbage — empty, punctuation-only, or mojibake."""
    clean = text.strip()
    if not clean or len(clean) < 2:
        return False
    alpha_chars = sum(1 for c in clean if c.isalpha() or ord(c) > 0x0900)
    if alpha_chars == 0:
        return False
    junk = sum(1 for c in clean if ord(c) > 0x10000 or c in "\ufffd\ufffe")
    if junk > len(clean) * 0.4:
        return False
    return True


def transcribe_audio(audio_path: str, language: str = "hi") -> tuple[str, str]:
    """Transcribe audio using faster-whisper. Auto-detects language first,
    falls back to forced language if auto-detect fails.
    """
    import subprocess

    try:
        model = _get_whisper()

        # First try auto-detect to get the actual spoken language
        try:
            segments, info = model.transcribe(audio_path, beam_size=5)
            text = " ".join(seg.text for seg in segments)
            detected = getattr(info, "language", None)
            if _is_meaningful_transcript(text) and detected:
                return text.strip(), detected
        except Exception as e:
            logger.warning("ASR auto-detect failed: %s", e)

        # Fallback: force the user's preferred language
        lang = language if language and language != "auto" else "hi"
        try:
            segments, info = model.transcribe(audio_path, beam_size=5, language=lang)
            text = " ".join(seg.text for seg in segments)
            if _is_meaningful_transcript(text):
                return text.strip(), lang
        except Exception as e:
            logger.warning("ASR forced-lang failed: %s", e)

        wav_path = audio_path + ".wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, check=True, timeout=30,
            )
            segments, info = model.transcribe(wav_path, beam_size=5)
            text = " ".join(seg.text for seg in segments)
            detected = getattr(info, "language", lang)
            if _is_meaningful_transcript(text):
                return text.strip(), detected

            segments, info = model.transcribe(wav_path, beam_size=5, language=lang)
            text = " ".join(seg.text for seg in segments)
            if _is_meaningful_transcript(text):
                return text.strip(), lang
        except Exception as e:
            logger.warning("ASR ffmpeg fallback failed: %s", e)
        finally:
            if wav_path and Path(wav_path).exists():
                try:
                    Path(wav_path).unlink()
                except OSError:
                    pass

    except Exception as e:
        logger.error("ASR fatal error: %s", e)
        import traceback
        traceback.print_exc()

    return "", language


# ── Voice Command Detection ──────────────────────────────────

VOICE_COMMANDS = {
    "phir se sunao": "replay", "phir se": "replay", "fir se sunao": "replay",
    "dubara sunao": "replay", "repeat": "replay",
    "aur samjhao": "explain_more", "aur batao": "explain_more",
    "detail mein": "explain_more", "explain more": "explain_more",
    "yeh kya karna chahiye": "action_items", "kya karna hai": "action_items",
    "kya karun": "action_items", "what should i do": "action_items",
    "kiska letter hai": "identify_sender", "kisne bheja": "identify_sender",
    "kitne din hain": "deadlines", "deadline kya hai": "deadlines", "kab tak": "deadlines",
    "bhasha badlo": "change_language", "language change": "change_language",
}


def detect_voice_command(transcript: str) -> str | None:
    lower = transcript.strip().lower()
    for phrase, action in VOICE_COMMANDS.items():
        if phrase in lower:
            return action
    return None


# ── PDF Text Extraction ──────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        pass
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        pass
    return ""


# ── Pipeline Runner ───────────────────────────────────────────

def _json_safe(value):
    """Convert nested values into JSON-serializable Python types."""
    import numpy as np

    if isinstance(value, (str, bool, type(None))):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def run_pipeline(inputs: dict, skip_stages: list[str] | None = None) -> dict:
    """Run the ML pipeline and return structured results."""
    from netra.core.base import StageResult

    stages = None
    if skip_stages:
        stages = [s for s in settings.active_stages if s not in skip_stages]

    t0 = time.perf_counter()
    results = pipeline.run(inputs, stages=stages)
    total_ms = (time.perf_counter() - t0) * 1000

    if skip_stages and "ocr" in skip_stages and "ocr" not in results:
        text = inputs.get("text", "")
        results["ocr"] = StageResult(
            stage_name="ocr", success=True,
            data={"text": text, "lines": text.split("\n"), "num_lines": len(text.split("\n")), "source": "direct_input"},
            latency_ms=0.0,
        )

    output: dict = {"total_ms": round(total_ms, 1), "stages": {}}
    for stage_name, result in results.items():
        stage_data: dict = {
            "success": result.success,
            "latency_ms": round(result.latency_ms, 1),
            "error": result.error,
        }
        if result.data:
            safe = {}
            for k, v in result.data.items():
                if isinstance(v, (str, int, float, bool)):
                    safe[k] = v
                elif isinstance(v, (list, dict)):
                    safe[k] = _json_safe(v)
            stage_data["data"] = safe
        output["stages"][stage_name] = stage_data

    memory = pipeline.get_memory_status()
    output["memory"] = {
        "vram_used_mb": memory["vram_used_mb"],
        "vram_budget_mb": memory["vram_budget_mb"],
    }

    result_path = OUTPUT_DIR / "pipeline_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return output


def get_prompt(key: str) -> dict:
    try:
        return settings.get_prompt(key)
    except ValueError:
        return settings.get_prompt("document_simplification")


# ── API Routes ────────────────────────────────────────────────

@app.route("/api/health")
def health():
    """Liveness: process is up. Includes detailed checks but always 200 if Flask is up."""
    faiss_dir = BACKEND_ROOT / "knowledge_base" / "vector_store"
    ollama_url = "http://localhost:11434"
    if settings is not None:
        try:
            llm_cfg = settings.get_model_config("llm", settings.model_defaults.get("llm"))
            ollama_url = llm_cfg.get("params", {}).get("ollama_url", ollama_url)
        except Exception:
            pass
    report = build_readiness(
        pipeline_ready=pipeline is not None,
        faiss_dir=faiss_dir,
        ollama_url=ollama_url,
        require_ollama=False,
        require_faiss=False,
    )
    return jsonify({
        "status": "ok",
        "version": "2.1.0",
        "pipeline": "NETRA v2 — Suno Sutra",
        "mode": "offline",
        "pipeline_ready": pipeline is not None,
        "ready": report["ready"],
        "checks": report["checks"],
        "languages": 22,
    })


@app.route("/api/ready")
def ready():
    """Readiness: 200 only when pipeline + Ollama are available for Q&A."""
    faiss_dir = BACKEND_ROOT / "knowledge_base" / "vector_store"
    ollama_url = "http://localhost:11434"
    if settings is not None:
        try:
            llm_cfg = settings.get_model_config("llm", settings.model_defaults.get("llm"))
            ollama_url = llm_cfg.get("params", {}).get("ollama_url", ollama_url)
        except Exception:
            pass
    # Only require Ollama when the configured LLM provider is Ollama.
    llm_key = ""
    if settings is not None:
        llm_key = str(settings.model_defaults.get("llm", ""))
    require_ollama = llm_key.startswith("ollama_")
    report = build_readiness(
        pipeline_ready=pipeline is not None,
        faiss_dir=faiss_dir,
        ollama_url=ollama_url,
        require_ollama=require_ollama,
        require_faiss=False,
    )
    report["llm"] = llm_key
    report["require_ollama"] = require_ollama
    code = 200 if report["ready"] else 503
    return jsonify(report), code


@app.route("/api/config")
@require_pipeline
def config():
    return jsonify({
        "pipeline_name": settings.pipeline_name,
        "version": settings.pipeline_version,
        "active_stages": settings.active_stages,
        "device": {
            "target": settings.device.target,
            "cuda_device": settings.device.cuda_device,
            "vram_budget_mb": settings.device.vram_budget_mb,
        },
        "defaults": settings.model_defaults,
        "languages": {
            "supported": settings.languages.supported,
            "default_source": settings.languages.default_source,
            "default_target": settings.languages.default_target,
        },
    })


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """ASR-only endpoint — returns transcript for user verification."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]
    lang = request.form.get("lang") or request.form.get("language") or "hi"

    filename = secure_filename(file.filename or "audio.webm")
    filepath = str(UPLOAD_DIR / f"asr_{int(time.time())}_{filename}")
    file.save(filepath)

    try:
        transcript, detected_lang = transcribe_audio(filepath, lang)
        if not transcript.strip():
            return jsonify({"error": "Could not transcribe audio.", "transcript": ""})

        voice_cmd = detect_voice_command(transcript)
        if voice_cmd:
            return jsonify({"transcript": transcript, "voice_command": voice_cmd, "language": detected_lang})

        return jsonify({"transcript": transcript, "language": detected_lang})
    finally:
        try:
            os.unlink(filepath)
        except OSError:
            pass


@app.route("/api/pipeline/run", methods=["POST"])
@require_pipeline
def pipeline_run():
    """Unified pipeline — handles image, audio, PDF uploads."""
    if "image" not in request.files and "audio" not in request.files and "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files.get("image") or request.files.get("audio") or request.files.get("file")
    lang = request.form.get("language") or request.form.get("lang") or "hi"
    mode = request.form.get("mode", "camera")

    filename = secure_filename(file.filename or "upload")
    filepath = str(UPLOAD_DIR / f"{int(time.time())}_{filename}")
    file.save(filepath)

    try:
        ext = Path(filename).suffix.lower()

        if mode == "voice" or ext in (".wav", ".mp3", ".ogg", ".webm", ".m4a"):
            return _handle_audio(filepath, lang)
        elif ext == ".pdf":
            return _handle_pdf(filepath, lang)
        else:
            return _handle_image(filepath, lang)
    finally:
        try:
            os.unlink(filepath)
        except OSError:
            pass


def _classify_query(
    transcript: str,
    state: str = "",
    district: str = "",
) -> tuple[str, dict | None]:
    """Decide which prompt to use based on retrieval score.
    Returns (prompt_key, cached_retrieval_data):
      - 'scheme_qa' + retrieval data if KB has a good match
      - 'general_chat' + None otherwise
    """
    try:
        if pipeline is None:
            return "general_chat", None

        if pipeline.get_stage("retrieval") is None:
            return "general_chat", None

        result = pipeline.run_stage(
            "retrieval",
            {"question": transcript, "state": state, "district": district},
        )
        if not result.success or result.data is None:
            logger.info("Query routed → general_chat (retrieval failed)")
            return "general_chat", None

        num = result.data.get("num_results", 0)
        if num > 0:
            chunks = result.data.get("context_chunks", [])
            best_score = max((c.get("score", 0) for c in chunks), default=0)
            if best_score >= 0.35:
                logger.info("Query routed → scheme_qa (RAG score=%.2f, %d chunks)", best_score, num)
                return "scheme_qa", result.data

        logger.info("Query routed → general_chat (no good RAG match)")
        return "general_chat", None
    except Exception as e:
        logger.warning("Query classification error: %s — defaulting to general_chat", e)
        return "general_chat", None


def _handle_audio(filepath: str, lang: str) -> tuple:
    transcript, asr_lang = transcribe_audio(filepath, lang)
    if not transcript.strip():
        return jsonify({"error": "Could not transcribe audio."}), 500

    voice_cmd = detect_voice_command(transcript)
    if voice_cmd:
        return jsonify({"voice_command": voice_cmd, "transcript": transcript})

    detected_src = detect_language(transcript)
    target_lang = asr_lang or detected_src

    state, district = _request_geo()
    prompt_key, cached_retrieval = _classify_query(transcript, state=state, district=district)
    prompt_config = get_prompt(prompt_key)

    temperature = 0.1 if prompt_key == "scheme_qa" else 0.3
    max_new_tokens = 150

    if prompt_key == "scheme_qa" and cached_retrieval is None and pipeline is not None:
        try:
            ret = pipeline.run_stage(
                "retrieval",
                {"question": transcript, "state": state, "district": district},
            )
            if ret.success and ret.data and ret.data.get("num_results", 0) > 0:
                cached_retrieval = ret.data
        except Exception as e:
            logger.warning("Forced retrieval for voice scheme_qa failed: %s", e)

    inputs = {
        "text": transcript,
        "question": transcript,
        "target_language": target_lang,
        "source_language": detected_src,
        "system_prompt": prompt_config["system"],
        "user_prompt": prompt_config["user_template"],
        "output_dir": str(OUTPUT_DIR),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.8 if prompt_key == "scheme_qa" else 0.9,
        "is_voice_query": True,
        "prompt_key": prompt_key,
        "state": state,
        "district": district,
    }
    skip = ["preprocessing", "ocr", "embedding", "knowledge_graph", "nmt"]
    if prompt_key == "general_chat":
        skip.append("retrieval")
    elif cached_retrieval is not None:
        inputs["retrieval_output"] = cached_retrieval
        skip.append("retrieval")

    result = run_pipeline(inputs, skip_stages=skip)

    result["asr_transcript"] = transcript
    result["prompt_used"] = prompt_key
    return jsonify(result)


def _handle_image(filepath: str, lang: str) -> tuple:
    has_rag = "retrieval" in settings.active_stages
    prompt_config = get_prompt("document_rag" if has_rag else "document_simplification")
    state, district = _request_geo()
    ocr_lang = lang if lang and lang != "auto" else "hi"

    inputs = {
        "image_path": filepath,
        "target_language": lang,
        "source_language": "auto",
        "ocr_language": ocr_lang,
        "system_prompt": prompt_config["system"],
        "user_prompt": prompt_config["user_template"],
        "output_dir": str(OUTPUT_DIR),
        "max_new_tokens": 150,
        "temperature": 0.3,
        "state": state,
        "district": district,
    }
    skip = ["embedding", "knowledge_graph", "nmt"]

    result = run_pipeline(inputs, skip_stages=skip)

    ocr_text = ""
    ocr_stage = result.get("stages", {}).get("ocr", {})
    if ocr_stage.get("success") and ocr_stage.get("data"):
        ocr_text = ocr_stage["data"].get("text", "")
    if ocr_text:
        detected_src = detect_language(ocr_text)
        logger.info("Image source language detected as %s (post-OCR)", detected_src)
        result["detected_document_language"] = detected_src

    return jsonify(result)


def _handle_pdf(filepath: str, lang: str) -> tuple:
    text = extract_pdf_text(filepath)
    if not text.strip():
        return jsonify({"error": "Could not extract text from PDF."}), 500

    detected_src = detect_language(text)
    has_rag = "retrieval" in settings.active_stages
    prompt_config = get_prompt("document_rag" if has_rag else "document_simplification")
    state, district = _request_geo()

    inputs = {
        "text": text,
        "question": text[:500],
        "ocr_output": {"text": text, "lines": text.split("\n"), "num_lines": len(text.split("\n"))},
        "target_language": lang,
        "source_language": "auto",
        "system_prompt": prompt_config["system"],
        "user_prompt": prompt_config["user_template"],
        "output_dir": str(OUTPUT_DIR),
        "max_new_tokens": 150,
        "temperature": 0.3,
        "state": state,
        "district": district,
    }
    skip = ["preprocessing", "ocr", "embedding", "knowledge_graph", "nmt"]

    result = run_pipeline(inputs, skip_stages=skip)
    result["detected_document_language"] = detected_src

    return jsonify(result)


@app.route("/api/pipeline/text", methods=["POST"])
@require_pipeline
def pipeline_text():
    """Run pipeline on direct text input. Routes to scheme_qa or general_chat."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    lang = data.get("language", "hi")
    if lang == "auto":
        lang = detect_language(text)

    mode = data.get("mode", "auto")
    state, district = _request_geo(data)

    detected_src = detect_language(text)
    target_lang = lang

    retrieval_inputs = {"question": text, "state": state, "district": district}

    cached_retrieval = None
    if mode == "auto":
        prompt_key, cached_retrieval = _classify_query(text, state=state, district=district)
    elif mode == "chat":
        prompt_key = "general_chat"
    else:
        prompt_key = "scheme_qa"

    prompt_config = get_prompt(prompt_key)

    if prompt_key == "scheme_qa":
        temperature = 0.1
        max_new_tokens = 150
    else:
        temperature = 0.3
        max_new_tokens = 150

    # Ensure scheme questions always have retrieval context for grounding.
    if prompt_key == "scheme_qa" and cached_retrieval is None and pipeline is not None:
        try:
            ret = pipeline.run_stage("retrieval", retrieval_inputs)
            if ret.success and ret.data and ret.data.get("num_results", 0) > 0:
                cached_retrieval = ret.data
        except Exception as e:
            logger.warning("Forced retrieval for scheme_qa failed: %s", e)

    inputs = {
        "text": text,
        "question": text,
        "target_language": target_lang,
        "source_language": detected_src,
        "system_prompt": prompt_config["system"],
        "user_prompt": prompt_config["user_template"],
        "output_dir": str(OUTPUT_DIR),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.8 if prompt_key == "scheme_qa" else 0.9,
        "prompt_key": prompt_key,
        "state": state,
        "district": district,
    }

    skip = ["preprocessing", "ocr", "embedding", "knowledge_graph", "nmt"]
    if prompt_key == "general_chat":
        skip.append("retrieval")
    elif cached_retrieval is not None:
        inputs["retrieval_output"] = cached_retrieval
        skip.append("retrieval")

    result = run_pipeline(inputs, skip_stages=skip)

    result["prompt_used"] = prompt_key
    if cached_retrieval is not None:
        result["rag_hits"] = cached_retrieval.get("num_results", 0)
        result["rag_top_score"] = max(
            (c.get("score", 0) for c in cached_retrieval.get("context_chunks", [])),
            default=0,
        )
    return jsonify(result)


@app.route("/api/ingest", methods=["POST"])
@require_pipeline
def ingest():
    """Ingest data into the RAG knowledge base (optional state/district tags)."""
    json_data = request.get_json(silent=True) or {}
    state, district = _request_geo(json_data)
    ocr_lang = _ocr_lang_for_ingest(json_data.get("language") or json_data.get("lang"))

    if "file" in request.files:
        file = request.files["file"]
        filename = secure_filename(file.filename or "upload")
        filepath = str(UPLOAD_DIR / f"ingest_{int(time.time())}_{filename}")
        file.save(filepath)
        ext = Path(filename).suffix.lower()

        try:
            if ext == ".pdf":
                text = extract_pdf_text(filepath)
            elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                ocr_inputs = {
                    "image_path": filepath,
                    "source_language": ocr_lang,
                    "target_language": ocr_lang,
                }
                ocr_result = run_pipeline(
                    ocr_inputs,
                    skip_stages=["retrieval", "llm", "nmt", "tts", "embedding", "knowledge_graph"],
                )
                ocr_stage = ocr_result.get("stages", {}).get("ocr", {})
                text = ocr_stage.get("data", {}).get("text", "") if ocr_stage.get("success") else ""
            else:
                ext_lower = Path(filepath).suffix.lower()
                if ext_lower in (".doc", ".docx"):
                    return jsonify({
                        "error": "Word documents (.doc/.docx) not yet supported. Please convert to PDF or text.",
                        "success": False,
                    }), 400
                text = Path(filepath).read_text(encoding="utf-8")

            if not text.strip():
                return jsonify({"error": "Could not extract text", "success": False}), 400
        finally:
            try:
                os.unlink(filepath)
            except OSError:
                pass
    else:
        text = json_data.get("text", "")
        if not text.strip():
            return jsonify({"error": "No text provided", "success": False}), 400

    detected = detect_language(text)
    inputs = {
        "text": text,
        "ocr_output": {"text": text, "lines": text.split("\n"), "num_lines": len(text.split("\n"))},
        "target_language": detected,
        "source_language": detected,
        "state": state,
        "district": district,
    }

    t0 = time.perf_counter()
    results = pipeline.run(inputs, stages=["embedding", "knowledge_graph"])
    total_ms = (time.perf_counter() - t0) * 1000

    all_ok = all(r.success for r in results.values())
    output: dict = {
        "success": all_ok,
        "total_ms": round(total_ms, 1),
        "state": state,
        "district": district,
        "detected_language": detected,
        "stages": {},
    }
    for name, result in results.items():
        sd: dict = {"success": result.success, "latency_ms": round(result.latency_ms, 1), "error": result.error}
        if result.data:
            sd["data"] = {k: v for k, v in result.data.items() if isinstance(v, (str, int, float, bool, list))}
        output["stages"][name] = sd

    return jsonify(output)


@app.route("/api/ingest/status")
@require_pipeline
def ingest_status():
    """Return auto-ingest watcher status and knowledge base stats."""
    meta_path = BACKEND_ROOT / "knowledge_base" / "vector_store" / "metadata.json"
    kb_chunks = 0
    kb_docs = set()
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            kb_chunks = len(meta)
            kb_docs = {e.get("doc_id") for e in meta if e.get("doc_id")}
        except Exception:
            pass

    watcher_stats = doc_watcher.stats if doc_watcher else {"is_running": False}
    return jsonify({
        "watcher": watcher_stats,
        "knowledge_base": {
            "total_chunks": kb_chunks,
            "total_documents": len(kb_docs),
        },
    })


@app.route("/api/ingest/scan", methods=["POST"])
@require_pipeline
def ingest_scan():
    """Manually trigger scan and ingest of all files in the inbox folder."""
    if not doc_watcher:
        return jsonify({"error": "Document watcher not initialized"}), 500
    results = doc_watcher.scan_and_ingest_all()
    return jsonify({"success": True, **results})


@app.route("/api/ingest/bulk", methods=["POST"])
@require_pipeline
def ingest_bulk():
    """Accept multiple files and ingest them all into the knowledge base."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded", "success": False}), 400

    state, district = _request_geo()
    ocr_lang = _ocr_lang_for_ingest()
    results = {"success": True, "processed": 0, "failed": 0, "state": state, "district": district, "files": []}

    for file in files:
        filename = secure_filename(file.filename or "upload")
        filepath = str(UPLOAD_DIR / f"bulk_{int(time.time())}_{filename}")
        file.save(filepath)
        ext = Path(filename).suffix.lower()

        try:
            if ext == ".pdf":
                text = extract_pdf_text(filepath)
            elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                ocr_inputs = {
                    "image_path": filepath,
                    "source_language": ocr_lang,
                    "target_language": ocr_lang,
                }
                ocr_result = run_pipeline(
                    ocr_inputs,
                    skip_stages=["retrieval", "llm", "nmt", "tts", "embedding", "knowledge_graph"],
                )
                ocr_stage = ocr_result.get("stages", {}).get("ocr", {})
                text = ocr_stage.get("data", {}).get("text", "") if ocr_stage.get("success") else ""
            else:
                ext_lower = Path(filepath).suffix.lower()
                if ext_lower in (".doc", ".docx"):
                    results["failed"] += 1
                    results["files"].append({
                        "name": filename,
                        "status": "error",
                        "error": "Word documents (.doc/.docx) not yet supported",
                    })
                    continue
                text = Path(filepath).read_text(encoding="utf-8")

            if not text.strip():
                results["failed"] += 1
                results["files"].append({"name": filename, "status": "error", "error": "No text extracted"})
                continue

            detected = detect_language(text)
            inputs = {
                "text": text,
                "ocr_output": {"text": text, "lines": text.split("\n"), "num_lines": len(text.split("\n"))},
                "target_language": detected,
                "source_language": detected,
                "state": state,
                "district": district,
            }
            pipeline.run(inputs, stages=["embedding", "knowledge_graph"])
            results["processed"] += 1
            results["files"].append({"name": filename, "status": "ok"})

        except Exception as e:
            results["failed"] += 1
            results["files"].append({"name": filename, "status": "error", "error": str(e)})
        finally:
            try:
                os.unlink(filepath)
            except OSError:
                pass

    return jsonify(results)


@app.route("/api/graph")
@require_pipeline
def graph():
    gpath = BACKEND_ROOT / "knowledge_base" / "graph" / "knowledge_graph.json"
    if not gpath.exists():
        return jsonify({"nodes": [], "edges": []})
    try:
        raw = json.loads(gpath.read_text())
        nodes = [{"id": n["id"], "label": n.get("label", n["id"]), **n} for n in (raw.get("nodes") or [])]
        edges = [{"source": e["source"], "target": e["target"], "relation": e.get("relation", "")} for e in (raw.get("links") or [])]
        return jsonify({"nodes": nodes, "edges": edges, "stats": {"nodes": len(nodes), "edges": len(edges)}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/output/pdf")
@require_pipeline
def output_pdf():
    lang = request.args.get("language", "hi")
    result_path = OUTPUT_DIR / "pipeline_result.json"
    if not result_path.exists():
        return jsonify({"error": "No pipeline result found"}), 404

    with open(result_path) as f:
        result = json.load(f)

    stages = result.get("stages", {})
    translated = stages.get("nmt", {}).get("data", {}).get("translated_text", "")
    english = stages.get("llm", {}).get("data", {}).get("response", "")
    ocr_text = stages.get("ocr", {}).get("data", {}).get("text", "")

    pdf_path = OUTPUT_DIR / "summary.pdf"
    _create_pdf(str(pdf_path), translated, english, ocr_text, lang)

    if pdf_path.exists():
        return send_file(str(pdf_path))
    txt_fallback = OUTPUT_DIR / "summary.txt"
    if txt_fallback.exists():
        return send_file(str(txt_fallback), mimetype="text/plain")
    return jsonify({"error": "PDF generation failed — reportlab not installed"}), 500


def _create_pdf(pdf_path: str, translated: str, english: str, ocr: str, lang: str) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18)
        body_style = ParagraphStyle("Body2", parent=styles["Normal"], fontSize=12, leading=16)
        small_style = ParagraphStyle("Small2", parent=styles["Normal"], fontSize=10, leading=14, textColor="gray")

        story = [Paragraph("NETRA Document Summary", title_style), Spacer(1, 0.3 * inch)]
        if translated:
            story.append(Paragraph(f"<b>Translation ({lang})</b>", styles["Heading2"]))
            for line in translated.split("\n"):
                if line.strip():
                    story.append(Paragraph(line.strip(), body_style))
            story.append(Spacer(1, 0.2 * inch))
        if english:
            story.append(Paragraph("<b>AI Analysis (English)</b>", styles["Heading2"]))
            for line in english.split("\n"):
                if line.strip():
                    story.append(Paragraph(line.strip(), body_style))
            story.append(Spacer(1, 0.2 * inch))
        if ocr:
            story.append(Paragraph("<b>Original Document Text (OCR)</b>", styles["Heading2"]))
            for line in ocr.split("\n")[:30]:
                if line.strip():
                    story.append(Paragraph(line.strip(), small_style))
        doc.build(story)
    except ImportError:
        with open(pdf_path.replace(".pdf", ".txt"), "w", encoding="utf-8") as f:
            f.write(f"NETRA Document Summary\n{'='*40}\n\n")
            if translated:
                f.write(f"Translation ({lang}):\n{translated}\n\n")
            if english:
                f.write(f"AI Analysis:\n{english}\n\n")
            if ocr:
                f.write(f"Original Text:\n{ocr}\n")


# ── Location / PIN Code Lookup ─────────────────────────────────

STATE_SCHEMES = {
    "Uttar Pradesh": [
        "Kanya Sumangala Yojana", "UP Pension Yojana", "Mukhyamantri Awas Yojana",
        "UP Ration Card Yojana", "Kisan Karj Maafi", "Samajwadi Pension Yojana",
    ],
    "Bihar": [
        "Mukhyamantri Kanya Utthan Yojana", "Bihar Student Credit Card",
        "Har Ghar Bijli Yojana", "Mukhyamantri Gramin Awas Yojana",
    ],
    "Madhya Pradesh": [
        "Ladli Laxmi Yojana", "MP Kisan Karj Maafi", "Mukhyamantri Yuva Swarozgar Yojana",
        "Sambal Yojana",
    ],
    "Rajasthan": [
        "Chiranjeevi Yojana", "Indira Gandhi Free Smartphone Yojana",
        "Rajasthan Tarbandi Yojana", "Palanhar Yojana",
    ],
    "Maharashtra": [
        "Mahatma Phule Jan Arogya Yojana", "Lek Ladki Yojana",
        "Namo Shetkari Maha Sanman Nidhi", "Mukhyamantri Saur Krushi Yojana",
    ],
    "Gujarat": [
        "Kisan Sahay Yojana", "Vahli Dikri Yojana", "Mukhyamantri Mahila Utkarsh Yojana",
    ],
    "Tamil Nadu": [
        "Kalaignar Magalir Urimai Thittam", "Naan Mudhalvan Scheme",
        "Free Bus Pass for Women", "TN Housing Board Scheme",
    ],
    "Karnataka": [
        "Gruha Lakshmi", "Anna Bhagya", "Shakti Free Bus for Women",
        "Yuva Nidhi", "Gruha Jyothi",
    ],
    "Kerala": [
        "LIFE Mission Housing", "Karunya Health Scheme", "Kerala Karshaka Kshemanidhi",
    ],
    "West Bengal": [
        "Lakshmi Bhandar", "Kanyashree Prakalpa", "Swasthya Sathi",
        "Krishak Bandhu", "Rupashree Prakalpa",
    ],
    "Andhra Pradesh": [
        "Amma Vodi", "YSR Rythu Bharosa", "YSR Cheyutha", "Jagananna Vidya Deevena",
    ],
    "Telangana": [
        "Rythu Bandhu", "Kalyana Lakshmi", "KCR Kits", "Aasara Pension",
    ],
    "Odisha": [
        "KALIA Yojana", "Biju Swasthya Kalyan Yojana", "Mission Shakti",
    ],
    "Punjab": [
        "Aam Aadmi Clinic", "Punjab Ashirwad Scheme", "Ghar Ghar Rozgar",
    ],
    "Haryana": [
        "Ayushman Bharat Haryana", "Mukhyamantri Parivar Samridhi Yojana",
        "Meri Fasal Mera Byora",
    ],
    "Jharkhand": [
        "Mukhyamantri Sukanya Yojana", "Jharkhand Fasal Rahat Yojana",
    ],
    "Chhattisgarh": [
        "Rajiv Gandhi Kisan Nyay Yojana", "Godhan Nyay Yojana",
        "Mahtari Vandan Yojana",
    ],
    "Assam": [
        "Orunodoi Scheme", "Arundhati Gold Scheme", "Chief Minister's Special Scholarship",
    ],
    "Himachal Pradesh": [
        "Indira Gandhi Mahila Samman Nidhi", "Him Care Health Insurance",
    ],
}

CENTRAL_SCHEMES = [
    "PM Kisan Samman Nidhi (₹6000/year for farmers)",
    "Ayushman Bharat (₹5 lakh health insurance)",
    "PM Awas Yojana - Gramin (rural housing)",
    "MGNREGA (100 days employment guarantee)",
    "PM Ujjwala Yojana (free LPG connection)",
    "PM Fasal Bima Yojana (crop insurance)",
    "Sukanya Samriddhi Yojana (girl child savings)",
    "PM Jan Dhan Yojana (bank accounts for all)",
    "Atal Pension Yojana (pension for unorganised sector)",
    "PM Kisan Maandhan (farmer pension ₹3000/month)",
    "PM Vishwakarma Yojana (artisan/craftsperson support)",
    "PM Surya Ghar Muft Bijli (free solar rooftop)",
    "Jal Jeevan Mission (tap water to every home)",
    "PM Garib Kalyan Anna Yojana (free ration)",
    "National Rural Livelihood Mission (self-help groups)",
]


@app.route("/api/location/lookup", methods=["POST"])
def location_lookup():
    data = request.get_json(force=True)
    pincode = str(data.get("pincode", "")).strip()
    district = str(data.get("district", "")).strip()

    state_name, state_code = get_state_from_pincode(pincode)
    if not state_name:
        return jsonify({"error": "Invalid PIN code or state not found"}), 400

    state_schemes = STATE_SCHEMES.get(state_name, [])
    return jsonify({
        "pincode": pincode,
        "state": state_name,
        "state_code": state_code,
        "district": district,
        "central_schemes": CENTRAL_SCHEMES,
        "state_schemes": state_schemes,
    })


# ── News Feed (Government RSS) ────────────────────────────────

@app.route("/api/news", methods=["GET"])
def get_news():
    """Fetch government news from RSS feeds. Falls back to cached/static if offline."""
    import xml.etree.ElementTree as ET
    import urllib.request

    lang = request.args.get("lang", "hi")

    if lang == "hi":
        feeds = [
            ("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3", "PIB India"),
            ("https://www.india.gov.in/rss/cabinet-702.xml", "India.gov.in"),
        ]
    else:
        feeds = [
            ("https://indianexpress.com/section/india/feed/", "Indian Express"),
        ]

    news_items = []

    for url, source in feeds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NETRA/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.iter("item"):
                title = item.findtext("title", "")
                pub_date = item.findtext("pubDate", "")
                link = item.findtext("link", "")
                desc = item.findtext("description", "")
                if title:
                    news_items.append({
                        "title": title.strip(),
                        "date": pub_date.strip(),
                        "source": source,
                        "link": link.strip(),
                        "description": desc.strip()[:150],
                    })
        except Exception as e:
            logger.warning("Failed to fetch RSS from %s: %s", url, e)

    if not news_items:
        news_items = _get_fallback_news(lang)

    news_items = news_items[:15]

    if lang not in ("en", "hi"):
        news_items = _translate_news(news_items, lang)

    return jsonify({"news": news_items, "count": len(news_items)})


LANG_FULL_NAMES = {
    "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
    "ur": "Urdu", "sa": "Sanskrit", "kok": "Konkani", "ks": "Kashmiri",
    "mai": "Maithili", "mni": "Manipuri", "ne": "Nepali", "sd": "Sindhi",
    "doi": "Dogri", "brx": "Bodo", "sat": "Santali",
}


def _translate_news(news_items, target_lang):
    """Translate news titles to target language using Ollama."""
    import requests as rq
    import re

    lang_name = LANG_FULL_NAMES.get(target_lang, "English")
    titles = [item["title"][:60] for item in news_items[:5]]
    titles_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))

    prompt = (
        f"Translate each line to {lang_name}. "
        f"Output ONLY the numbered translations, nothing else.\n\n{titles_text}"
    )

    try:
        resp = rq.post("http://localhost:11434/api/generate", json={
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 800},
        }, timeout=60)
        if resp.status_code == 200:
            result = resp.json().get("response", "")
            lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
            translated = []
            for line in lines:
                clean = re.sub(r"^\d+[\.\)\-:]\s*", "", line).strip()
                if clean and len(clean) > 3:
                    translated.append(clean)

            for i, item in enumerate(news_items[:5]):
                if i < len(translated):
                    item["title"] = translated[i]
        else:
            logger.warning("Ollama returned status %d for news translation", resp.status_code)
    except Exception as e:
        logger.warning("News translation failed: %s", e)

    return news_items


def _warm_ollama():
    """Pre-warm all Ollama models into GPU memory so first request is fast."""
    import threading
    import requests as rq

    def _warm():
        ollama_url = "http://localhost:11434"
        models_to_warm = [
            ("qwen2.5:1.5b", "generate"),
            ("qwen3:4b", "generate"),
            ("qwen3-embedding:0.6b", "embeddings"),
        ]
        for model, api in models_to_warm:
            try:
                if api == "generate":
                    rq.post(f"{ollama_url}/api/generate", json={
                        "model": model, "prompt": "Hi", "stream": False,
                        "keep_alive": "30m", "options": {"num_predict": 1},
                    }, timeout=60)
                else:
                    rq.post(f"{ollama_url}/api/embeddings", json={
                        "model": model, "prompt": "test",
                        "keep_alive": "30m",
                    }, timeout=60)
                logger.info("Pre-warmed: %s", model)
            except Exception as e:
                logger.warning("Failed to warm %s: %s", model, e)

    threading.Thread(target=_warm, daemon=True).start()


def _get_fallback_news(lang="hi"):
    """Static fallback news for when RSS is unreachable."""
    fallback_by_lang = {
        "hi": [
            {"title": "PM किसान 17वीं किस्त जारी - स्टेटस चेक करें", "date": "2026", "source": "PIB", "link": "", "description": ""},
            {"title": "आयुष्मान भारत - 70+ आयु तक कवरेज बढ़ाया गया", "date": "2026", "source": "PIB", "link": "", "description": ""},
            {"title": "PM आवास योजना ग्रामीण - FY 2026-27 के लिए नए मकान स्वीकृत", "date": "2026", "source": "PIB", "link": "", "description": ""},
            {"title": "मनरेगा मजदूरी दरें 2026-27 के लिए बढ़ाई गईं", "date": "2026", "source": "PIB", "link": "", "description": ""},
            {"title": "PM उज्ज्वला योजना 3.0 - BPL परिवारों को मुफ्त रिफिल", "date": "2026", "source": "PIB", "link": "", "description": ""},
            {"title": "जल जीवन मिशन: 90% ग्रामीण घरों में नल का पानी पहुंचा", "date": "2026", "source": "PIB", "link": "", "description": ""},
            {"title": "PM विश्वकर्मा - कारीगर अब ₹3 लाख लोन के लिए आवेदन करें", "date": "2026", "source": "PIB", "link": "", "description": ""},
            {"title": "मुफ्त राशन योजना दिसंबर 2026 तक बढ़ाई गई", "date": "2026", "source": "PIB", "link": "", "description": ""},
        ],
        "ta": [
            {"title": "PM கிசான் 17வது தவணை வெளியிடப்பட்டது - நிலையை சரிபார்க்கவும்", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "ஆயுஷ்மான் பாரத் - 70+ வயது வரை காப்பீடு நீட்டிக்கப்பட்டது", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "PM வீட்டு வசதி திட்டம் - 2026-27க்கான புதிய வீடுகள் ஒப்புதல்", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "மகாத்மா காந்தி ஊரக வேலை உறுதி - ஊதியம் உயர்வு 2026-27", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "PM உஜ்வலா யோஜனா 3.0 - BPL குடும்பங்களுக்கு இலவச ரீஃபில்", "date": "2026", "source": "Gov", "link": "", "description": ""},
        ],
        "te": [
            {"title": "PM కిసాన్ 17వ వాయిదా విడుదల - స్టేటస్ చెక్ చేయండి", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "ఆయుష్మాన్ భారత్ - 70+ వయసు వరకు కవరేజ్ పొడిగింపు", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "PM ఆవాస్ యోజన గ్రామీణ - 2026-27 కొత్త ఇళ్ళు మంజూరు", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "మహాత్మా గాంధీ ఉపాధి హామీ - 2026-27 వేతనాలు పెంపు", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "PM ఉజ్వల యోజన 3.0 - BPL కుటుంబాలకు ఉచిత రీఫిల్", "date": "2026", "source": "Gov", "link": "", "description": ""},
        ],
        "bn": [
            {"title": "PM কিষাণ 17তম কিস্তি প্রকাশ - স্ট্যাটাস চেক করুন", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "আয়ুষ্মান ভারত - 70+ বয়স পর্যন্ত কভারেজ বৃদ্ধি", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "PM আবাস যোজনা গ্রামীণ - 2026-27 নতুন বাড়ি অনুমোদন", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "মনরেগা মজুরি বৃদ্ধি 2026-27", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "PM উজ্জ্বলা যোজনা 3.0 - BPL পরিবারগুলিকে বিনামূল্যে রিফিল", "date": "2026", "source": "Gov", "link": "", "description": ""},
        ],
        "mr": [
            {"title": "PM किसान 17वा हप्ता जारी - स्थिती तपासा", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "आयुष्मान भारत - 70+ वयापर्यंत कव्हरेज वाढवले", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "PM आवास योजना ग्रामीण - 2026-27 नवीन घरे मंजूर", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "मनरेगा मजुरी दर 2026-27 वाढवली", "date": "2026", "source": "Gov", "link": "", "description": ""},
            {"title": "PM उज्ज्वला योजना 3.0 - BPL कुटुंबांना मोफत रिफिल", "date": "2026", "source": "Gov", "link": "", "description": ""},
        ],
    }

    if lang in fallback_by_lang:
        return fallback_by_lang[lang]

    return [
        {"title": "PM Kisan 17th installment released - check status", "date": "2026", "source": "Gov", "link": "", "description": ""},
        {"title": "Ayushman Bharat coverage extended to age 70+", "date": "2026", "source": "Gov", "link": "", "description": ""},
        {"title": "PM Awas Yojana Gramin - new houses sanctioned for FY 2026-27", "date": "2026", "source": "Gov", "link": "", "description": ""},
        {"title": "MGNREGA wage rates increased for 2026-27", "date": "2026", "source": "Gov", "link": "", "description": ""},
        {"title": "PM Ujjwala Yojana 3.0 launched - free refills for BPL families", "date": "2026", "source": "Gov", "link": "", "description": ""},
        {"title": "Jal Jeevan Mission: 90% rural households now have tap water", "date": "2026", "source": "Gov", "link": "", "description": ""},
        {"title": "PM Vishwakarma - artisans can now apply for Rs 3 lakh loan", "date": "2026", "source": "Gov", "link": "", "description": ""},
        {"title": "Free ration scheme extended till December 2026", "date": "2026", "source": "Gov", "link": "", "description": ""},
    ]


# ── Weather Alerts ─────────────────────────────────────────────

@app.route("/api/weather", methods=["GET"])
def get_weather():
    """Get weather alerts based on state. Uses Open-Meteo for basic data."""
    import urllib.request

    state = request.args.get("state", "")
    if not state:
        return jsonify({"alerts": [], "error": "No state specified"}), 400

    STATE_COORDS = {
        "Uttar Pradesh": (26.85, 80.91), "Bihar": (25.60, 85.14),
        "Madhya Pradesh": (23.25, 77.41), "Rajasthan": (26.92, 75.79),
        "Maharashtra": (19.07, 72.87), "Gujarat": (23.02, 72.57),
        "Tamil Nadu": (13.08, 80.27), "Karnataka": (12.97, 77.59),
        "Kerala": (10.85, 76.27), "West Bengal": (22.57, 88.36),
        "Andhra Pradesh": (15.83, 78.04), "Telangana": (17.38, 78.49),
        "Odisha": (20.27, 85.84), "Punjab": (30.73, 76.77),
        "Haryana": (28.45, 77.02), "Jharkhand": (23.35, 85.33),
        "Chhattisgarh": (21.25, 81.63), "Assam": (26.14, 91.77),
        "Himachal Pradesh": (31.10, 77.17), "Delhi": (28.61, 77.23),
    }

    coords = STATE_COORDS.get(state, (28.61, 77.23))
    alerts = []

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords[0]}&longitude={coords[1]}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max&timezone=Asia/Kolkata&forecast_days=3"
        req = urllib.request.Request(url, headers={"User-Agent": "NETRA/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        daily = data.get("daily", {})
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        rain = daily.get("precipitation_sum", [])
        wind = daily.get("wind_speed_10m_max", [])
        dates = daily.get("time", [])

        for i in range(min(3, len(temps_max))):
            if temps_max[i] and temps_max[i] > 42:
                alerts.append({"type": "danger", "icon": "🔥", "message": f"Heat Wave Alert! {temps_max[i]}°C on {dates[i]}"})
            elif temps_max[i] and temps_max[i] > 38:
                alerts.append({"type": "warning", "icon": "☀️", "message": f"Very Hot: {temps_max[i]}°C on {dates[i]}"})

            if rain[i] and rain[i] > 50:
                alerts.append({"type": "danger", "icon": "🌊", "message": f"Heavy Rain: {rain[i]}mm on {dates[i]} - Flood risk!"})
            elif rain[i] and rain[i] > 20:
                alerts.append({"type": "warning", "icon": "🌧️", "message": f"Rain expected: {rain[i]}mm on {dates[i]}"})

            if wind[i] and wind[i] > 60:
                alerts.append({"type": "danger", "icon": "🌪️", "message": f"Storm Warning! Wind {wind[i]} km/h on {dates[i]}"})

            if not alerts and i == 0:
                alerts.append({"type": "info", "icon": "🌤️", "message": f"Normal: {temps_min[i]}°C - {temps_max[i]}°C, Rain: {rain[i] or 0}mm"})

    except Exception as e:
        logger.warning("Weather fetch failed: %s", e)
        alerts.append({"type": "info", "icon": "⚠️", "message": "Weather data unavailable - check internet"})

    return jsonify({"state": state, "alerts": alerts})


# ── Mandi Prices ───────────────────────────────────────────────

@app.route("/api/mandi", methods=["GET"])
def get_mandi():
    """Get simulated mandi (market) prices for major crops."""
    import random
    state = request.args.get("state", "")

    STATE_CROPS = {
        "Uttar Pradesh": [("Wheat", 2275, 2400), ("Rice", 2183, 2350), ("Sugarcane", 315, 340), ("Potato", 800, 1200), ("Mustard", 5450, 5800)],
        "Bihar": [("Rice", 2183, 2400), ("Wheat", 2275, 2500), ("Maize", 1962, 2100), ("Lentil", 6000, 6400), ("Potato", 700, 1100)],
        "Punjab": [("Wheat", 2275, 2500), ("Rice", 2183, 2400), ("Cotton", 6620, 7000), ("Maize", 1962, 2150), ("Mustard", 5450, 5900)],
        "Rajasthan": [("Mustard", 5450, 5800), ("Wheat", 2275, 2400), ("Bajra", 2500, 2700), ("Cumin", 30000, 35000), ("Gram", 5440, 5800)],
        "Madhya Pradesh": [("Soybean", 4600, 4900), ("Wheat", 2275, 2450), ("Gram", 5440, 5700), ("Onion", 1500, 2200), ("Garlic", 5000, 7000)],
        "Maharashtra": [("Cotton", 6620, 7200), ("Soybean", 4600, 5000), ("Onion", 1500, 2500), ("Sugarcane", 315, 350), ("Tur Dal", 7000, 7500)],
        "Gujarat": [("Cotton", 6620, 7100), ("Groundnut", 5850, 6200), ("Cumin", 30000, 34000), ("Wheat", 2275, 2400), ("Castor", 5700, 6100)],
        "West Bengal": [("Rice", 2183, 2350), ("Jute", 5050, 5400), ("Potato", 800, 1200), ("Mustard", 5450, 5700), ("Tea", 200, 350)],
        "Tamil Nadu": [("Rice", 2183, 2400), ("Coconut", 30, 35), ("Banana", 40, 55), ("Turmeric", 12000, 15000), ("Groundnut", 5850, 6200)],
        "Karnataka": [("Rice", 2183, 2350), ("Ragi", 3846, 4100), ("Coffee", 8500, 9500), ("Coconut", 30, 38), ("Arecanut", 45000, 55000)],
    }

    default_crops = [("Wheat", 2275, 2400), ("Rice", 2183, 2350), ("Onion", 1500, 2200), ("Potato", 800, 1200), ("Mustard", 5450, 5800)]
    crops = STATE_CROPS.get(state, default_crops)

    prices = []
    for crop, msp, max_price in crops:
        price = random.randint(msp, max_price)
        prices.append({
            "crop": crop,
            "price": price,
            "msp": msp,
            "unit": "₹/quintal" if price > 100 else "₹/kg",
        })

    return jsonify({"state": state, "prices": prices})


# ── Scheme Reminders ───────────────────────────────────────────

@app.route("/api/reminders", methods=["GET"])
def get_reminders():
    """Get upcoming scheme deadlines and reminders."""
    reminders = [
        {"name": "PM Kisan - 18th Installment", "date": "Aug 2026", "icon": "🌾", "urgent": False, "desc": "Check eKYC status on pmkisan.gov.in"},
        {"name": "PM Fasal Bima - Kharif Registration", "date": "31 Aug 2026", "icon": "🛡️", "urgent": True, "desc": "Last date for Kharif crop insurance"},
        {"name": "Ayushman Bharat - Card Update", "date": "Sep 2026", "icon": "🏥", "urgent": False, "desc": "Update family details on PMJAY portal"},
        {"name": "PM Awas Yojana - New Applications", "date": "30 Sep 2026", "icon": "🏠", "urgent": False, "desc": "Apply through Gram Panchayat"},
        {"name": "Ration Card - Annual Verification", "date": "Oct 2026", "icon": "🍚", "urgent": False, "desc": "Verify at nearest ration shop"},
        {"name": "Kisan Credit Card Renewal", "date": "Before harvest", "icon": "💳", "urgent": True, "desc": "Renew at bank branch for crop loan"},
        {"name": "MGNREGA - Job Card Update", "date": "Ongoing", "icon": "👷", "urgent": False, "desc": "Update at Gram Panchayat office"},
    ]
    return jsonify({"reminders": reminders})


@app.route("/api/benchmark", methods=["POST"])
@require_pipeline
def benchmark():
    """Run performance benchmarks and return results.

    JSON body (all optional):
        runs (int): number of runs per test (default 1, max 10)
        ingest_only (bool): only ingestion
        query_only (bool): only query search
    """
    from netra.utils.hardware_metrics import (
        MetricsSampler,
        summarize_samples,
        system_info,
    )

    data = request.get_json(silent=True) or {}
    runs = min(int(data.get("runs", 1)), 10)
    ingest_only = bool(data.get("ingest_only", False))
    query_only = bool(data.get("query_only", False))

    report: dict = {
        "system": system_info(),
        "benchmarks": {},
    }

    bench_ingest_text = "Benchmark document NRSSY-BENCH. Skill training stipend Rs 5000."
    bench_query_text = "What is the skill training stipend under NRSSY scheme?"

    def _bench_ingest():
        t0 = time.perf_counter()
        sampler = MetricsSampler(interval_s=0.5)
        sampler.start()
        result = run_pipeline(
            {"text": bench_ingest_text, "ocr_output": {"text": bench_ingest_text, "lines": [bench_ingest_text], "num_lines": 1},
             "target_language": "hi", "source_language": "hi"},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        hw_samples = sampler.stop()
        return {
            "total_ms": round(elapsed_ms, 1),
            "success": all(s.get("success", False) for s in result.get("stages", {}).values()),
            "stages": {k: {"latency_ms": v.get("latency_ms")} for k, v in result.get("stages", {}).items()},
            "hardware": summarize_samples(hw_samples),
        }

    def _bench_query():
        t0 = time.perf_counter()
        sampler = MetricsSampler(interval_s=0.5)
        sampler.start()
        result = run_pipeline(
            {"text": bench_query_text, "question": bench_query_text,
             "target_language": "hi", "source_language": "hi",
             "system_prompt": "Answer using the provided context.",
             "user_prompt": "Question: {question}\n\nContext:\n{context}",
             "max_new_tokens": 128, "temperature": 0.1, "top_p": 0.8},
            skip_stages=["preprocessing", "ocr", "embedding", "knowledge_graph"],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        hw_samples = sampler.stop()
        return {
            "total_ms": round(elapsed_ms, 1),
            "success": all(s.get("success", False) for s in result.get("stages", {}).values()),
            "stages": {k: {"latency_ms": v.get("latency_ms")} for k, v in result.get("stages", {}).items()},
            "hardware": summarize_samples(hw_samples),
        }

    if not query_only:
        report["benchmarks"]["data_ingestion"] = [_bench_ingest() for _ in range(runs)]
    if not ingest_only:
        report["benchmarks"]["query_search"] = [_bench_query() for _ in range(runs)]

    return jsonify(report)


@app.route("/api/benchmark/hardware")
def benchmark_hardware():
    """Return current hardware snapshot (CPU/GPU utilization and temperature)."""
    from netra.utils.hardware_metrics import snapshot_hardware, system_info as hw_system_info
    from dataclasses import asdict
    return jsonify({
        "system": hw_system_info(),
        "current": asdict(snapshot_hardware()),
    })


@app.route("/api/benchmark/accuracy", methods=["POST"])
@require_pipeline
def benchmark_accuracy():
    """Run accuracy benchmarks against ground-truth test cases.

    JSON body (all optional):
        stage (str): 'retrieval', 'llm', or 'nmt' — omit for all
        compare (bool): compare multiple model variants (default false)
    """
    from scripts.benchmark_accuracy import (
        bench_retrieval, bench_llm, bench_nmt,
        RETRIEVAL_TESTS, LLM_TESTS, NMT_TESTS,
        run_model_comparison, COMPARABLE_STAGES,
    )
    from netra.utils.hardware_metrics import system_info as hw_system_info

    data = request.get_json(silent=True) or {}
    stage_filter = data.get("stage")
    do_compare = bool(data.get("compare", False))

    is_cpu = settings.device.target == "cpu" or settings.device.cuda_device < 0

    report: dict = {
        "system": hw_system_info(),
        "benchmarks": {},
    }

    stages = [stage_filter] if stage_filter else ["retrieval", "llm", "nmt"]

    if "retrieval" in stages:
        report["benchmarks"]["retrieval"] = {
            "model_key": settings.model_defaults.get("retrieval"),
            "results": bench_retrieval(pipeline, RETRIEVAL_TESTS),
        }
    if "llm" in stages:
        report["benchmarks"]["llm"] = {
            "model_key": settings.model_defaults.get("llm"),
            "results": bench_llm(pipeline, LLM_TESTS),
        }
    if "nmt" in stages:
        report["benchmarks"]["nmt"] = {
            "model_key": settings.model_defaults.get("nmt"),
            "results": bench_nmt(pipeline, NMT_TESTS),
        }

    if do_compare:
        report["model_comparison"] = {}
        for s in stages:
            if s in COMPARABLE_STAGES:
                report["model_comparison"][s] = run_model_comparison(settings, s, is_cpu)

    return jsonify(report)


@app.route("/api/output/<filename>")
def output_file(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400
    filepath = OUTPUT_DIR / safe_name
    if not filepath.resolve().is_relative_to(OUTPUT_DIR.resolve()):
        return jsonify({"error": "Access denied"}), 403
    if filepath.exists():
        return send_file(str(filepath))
    return jsonify({"error": "File not found"}), 404


# ── Startup ───────────────────────────────────────────────────

def init_pipeline():
    global pipeline, settings, doc_watcher
    from netra.config.settings import Settings
    from netra.core.pipeline import Pipeline
    from netra.utils.log import setup_logging

    config_dir = BACKEND_ROOT / "config"
    settings = Settings(config_dir)
    setup_logging(level="WARNING")

    import netra.stages  # noqa: F401
    pipeline = Pipeline(settings)
    logger.info("Pipeline loaded — models cached in memory")

    from netra.utils.watcher import DocumentWatcher
    watch_dir = BACKEND_ROOT / "data" / "inbox"
    doc_watcher = DocumentWatcher(
        watch_dir=watch_dir,
        pipeline=pipeline,
        run_pipeline_fn=run_pipeline,
        extract_pdf_fn=extract_pdf_text,
        target_language="hi",
        poll_interval=15.0,
    )
    doc_watcher.start()
    logger.info("Document watcher started — monitoring %s", watch_dir)


def auto_ingest_policies():
    """Auto-ingest policy documents into FAISS if the index doesn't exist yet."""
    kb_index = BACKEND_ROOT / "knowledge_base" / "vector_store" / "index.faiss"
    if kb_index.exists():
        logger.info("FAISS index found (%s) — skipping auto-ingest", kb_index)
        return

    polyic_dir = BACKEND_ROOT.parent / "polyic" / "state"
    if not polyic_dir.exists():
        logger.warning("No polyic/state/ directory found — skipping auto-ingest")
        return

    logger.info("No FAISS index found — auto-ingesting policy documents from %s ...", polyic_dir)
    ingest_script = BACKEND_ROOT / "scripts" / "ingest_policies.py"
    if not ingest_script.exists():
        logger.warning("ingest_policies.py not found — skipping auto-ingest")
        return

    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(ingest_script)],
        cwd=str(BACKEND_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode == 0:
        logger.info("Auto-ingest completed successfully")
        for line in result.stdout.strip().split("\n")[-8:]:
            logger.info("  %s", line)
    else:
        logger.error("Auto-ingest failed (exit %d): %s", result.returncode, result.stderr[-500:] if result.stderr else "")


if __name__ == "__main__":
    PORT = int(os.environ.get("BACKEND_PORT", 5000))
    logger.info("Starting NETRA Backend ML Server on port %d...", PORT)
    _warm_ollama()
    auto_ingest_policies()
    init_pipeline()
    logger.info("Backend ready!")
    app.run(host="0.0.0.0", port=PORT, debug=False)
