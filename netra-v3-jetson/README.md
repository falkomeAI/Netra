# NETRA v3 — Native C++ Pipeline for Jetson Orin Nano

> **Single compiled binary. Zero Python. Zero Ollama. Pure C/C++ + CUDA.**  
> Target: **< 3 seconds** total pipeline on Jetson Orin Nano 8GB

## Why C++ Instead of Python?

| | Python (NETRA v2) | C++ (NETRA v3) |
|---|---|---|
| **LLM** | Ollama HTTP → Python → llama.cpp | **Direct llama.cpp C API** |
| **ASR** | Python faster-whisper → CTranslate2 | **Direct whisper.cpp C API** |
| **TTS** | Python piper-tts → ONNX | **Piper native binary** |
| **RAG** | Python sentence-transformers + faiss | **FAISS C++ + ONNX Runtime C++ (keyword fallback)** |
| **HTTP** | Flask (Python WSGI) | **cpp-httplib (zero-copy)** |
| **Startup** | ~5-10s (Python import) | **< 1s** (native binary) |
| **Memory** | ~500MB Python overhead | **~50MB binary** |
| **GIL** | Single-threaded bottleneck | **True multi-threading** |
| **Total** | 5-6s pipeline | **< 3s pipeline** |

## Architecture

```
┌──────────────────────────────────────────────┐
│           netra (single C++ binary)           │
│              ~15MB compiled                   │
├──────────┬──────────┬──────────┬──────────────┤
│  ASR     │   RAG    │   LLM    │    TTS       │
│ whisper  │  FAISS   │ llama    │   piper      │
│  .cpp    │  C++ +   │  .cpp    │   binary     │
│  CUDA    │  ONNX    │  CUDA    │   ONNX       │
│          │  C++     │          │              │
├──────────┴──────────┴──────────┴──────────────┤
│  cpp-httplib │ nlohmann/json │ tesseract-ocr  │
├───────────────────────────────────────────────┤
│          NVIDIA CUDA 12.x + cuBLAS            │
│       Jetson Orin Nano 8GB (JetPack 6.x)      │
└───────────────────────────────────────────────┘
```

## Tech Stack

| Component | Library | Language | Why |
|-----------|---------|----------|-----|
| **LLM Inference** | [llama.cpp](https://github.com/ggerganov/llama.cpp) | C/C++ | 18+ tok/s on Orin Nano, direct CUDA kernels |
| **Speech-to-Text** | [whisper.cpp](https://github.com/ggerganov/whisper.cpp) | C/C++ | ~1s for 5s audio, CUDA accelerated |
| **Text-to-Speech** | [Piper](https://github.com/rhasspy/piper) | C++ | ~0.3s synthesis, ONNX+espeak-ng |
| **Vector Search** | [FAISS](https://github.com/facebookresearch/faiss) | C++ | Sub-ms GPU vector search (keyword fallback when unavailable) |
| **Embedding** | [ONNX Runtime](https://onnxruntime.ai/) | C++ | GPU-accelerated BGE-small (optional) |
| **OCR** | [Tesseract](https://github.com/tesseract-ocr/tesseract) + poppler | C++ | Hindi+English OCR |
| **HTTP Server** | [cpp-httplib](https://github.com/yhirose/cpp-httplib) | C++ | Single-header, zero-copy |
| **JSON** | [nlohmann/json](https://github.com/nlohmann/json) | C++ | Single-header, industry standard |

## Models

| Stage | Model | Format | Size | Notes |
|-------|-------|--------|------|-------|
| LLM | Qwen3-4B-Instruct | GGUF Q4_K_M | ~2.5 GB | Thinking mode disabled via empty `<think>` pre-fill |
| ASR | Whisper Small | GGML | 460 MB | Optional — graceful degradation if absent |
| Embedding | BGE-small-en-v1.5 | ONNX | 130 MB | Optional — keyword search fallback if absent |
| TTS (Hindi) | Piper hi_IN-medium | ONNX | 100 MB | Optional — text-only response if absent |
| TTS (English) | Piper en_US-lessac | ONNX | 100 MB | Optional |
| OCR | Tesseract hin+eng + poppler | system | ~30 MB | pdftotext for text PDFs, tesseract for scanned |
| **Total** | | | **~3.3 GB** | Only LLM is required |

## Quick Start

```bash
# One-command: clone deps, download models, build, optimize
bash scripts/setup_jetson.sh

# Run
cd build
./netra --llm ../models/qwen2.5-3b-q4_k_m.gguf \
        --whisper ../models/ggml-small.bin \
        --port 5050
```

## Manual Build

```bash
# 1. Install dependencies
sudo apt install build-essential cmake git wget \
    libfaiss-dev tesseract-ocr tesseract-ocr-hin poppler-utils libsndfile1-dev

# 2. Clone third-party libs
cd third_party
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git
wget -O cpp-httplib/httplib.h https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h
wget -O json/single_include/nlohmann/json.hpp https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp

# 3. Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
make -j$(nproc)

# 4. Run
./netra --llm ../models/qwen2.5-3b-q4_k_m.gguf
```

## Policy Document Ingestion (RAG)

Policy documents (`.pdf`, `.txt`) are the knowledge base for RAG-powered Q&A. There are **three ways** to ingest policies:

### Method 1: Auto-Ingest on Startup (Recommended)

Place policy files in a `polyic/` directory and they will be automatically ingested when the server starts.

```bash
# Directory structure — organize by state or category
polyic/
├── state/
│   ├── Tamil_Nadu/
│   │   ├── TN_PDS_Ration_Card_Policy.txt
│   │   └── TN_Industrial_Policy_2021.pdf
│   ├── Karnataka/
│   │   └── Karnataka_Industrial_Policy_2025-30.pdf
│   └── Kerala/
│       └── Kerala_IT_Policy_2026.pdf
└── central/
    └── PM_Kisan_Scheme.pdf
```

```bash
# Start server — policies are auto-ingested
./build/netra --llm models/qwen3-4b.gguf \
              --policies ./polyic \
              --index knowledge_base/vector_store \
              --port 5050
```

The server scans the `--policies` directory **recursively** for all `.pdf` and `.txt` files. It also automatically checks `polyic/`, `../polyic/`, and `../../polyic/` relative to the working directory.

**Duplicate detection**: Already-ingested files are tracked in `knowledge_base/vector_store/metadata.txt` and skipped on subsequent starts. Moving to a new machine? Just copy the `polyic/` folder — policies will be re-ingested automatically.

### Method 2: Upload via API (Runtime)

Ingest new policies at any time without restarting:

```bash
# Ingest a single PDF
curl -X POST http://localhost:5050/api/ingest -F "file=@new_policy.pdf"

# Ingest a text file
curl -X POST http://localhost:5050/api/ingest -F "file=@scheme_details.txt"
```

Response:
```json
{
  "status": "ingested",
  "source": "new_policy.pdf",
  "chunks": 15,
  "index_size": 1092,
  "timing": {"total_ms": 234.5}
}
```

### Method 3: Copy to Policies Directory

Simply copy new files into the `polyic/` folder and restart the server — only the new files will be ingested (existing ones are skipped).

```bash
# Copy new policies
cp /path/to/new_policies/*.pdf polyic/state/Maharashtra/

# Restart — only new files are ingested
./build/netra --llm models/qwen3-4b.gguf --port 5050
```

### Supported Formats

| Format | Extraction Method | Notes |
|--------|------------------|-------|
| `.txt` | Direct read | Plain text, fastest |
| `.pdf` (text-based) | `pdftotext` (poppler) | Most government PDFs |
| `.pdf` (scanned/image) | Tesseract OCR | Requires `tesseract-ocr` installed |

### Verify Ingestion

```bash
# Check how many chunks are indexed
curl -s http://localhost:5050/api/health | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Index size: {d[\"index_size\"]} chunks')
print(f'RAG engine: {d[\"engines\"][\"rag\"]}')"

# Test a query against ingested policies
curl -s -X POST http://localhost:5050/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the ration card policy in Tamil Nadu?"}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Answer:', d['answer'][:200])
print('Sources:', [s['source'] for s in d.get('sources', [])])"
```

### Setup on a New Machine

```bash
# 1. Clone the repo
git clone https://github.com/Gaurav14cs17/NETRA-v2.git
cd NETRA-v2
git checkout netra-v3-cpp-jetson

# 2. Build (see Manual Build section below)

# 3. Copy policy documents
cp -r /path/to/polyic ./polyic

# 4. Start — policies auto-ingest on first run
./build/netra --llm models/qwen3-4b.gguf --port 5050
# Server logs: "Policy ingest: 30 new, 0 skipped, 1077 total chunks in 45000.0 ms"

# Subsequent starts skip already-ingested files:
# "Policy ingest: 0 new, 30 skipped, 1077 total chunks in 5.2 ms"
```

---

## API Endpoints

Same API as NETRA v2 — drop-in replacement:

```bash
# Text query
curl -X POST http://localhost:5050/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "प्रधानमंत्री आवास योजना में कितनी राशि मिलती है?"}'

# Audio query
curl -X POST http://localhost:5050/api/audio -F "file=@question.wav"

# PDF query
curl -X POST http://localhost:5050/api/pdf -F "file=@policy.pdf"

# Image OCR query
curl -X POST http://localhost:5050/api/image -F "file=@document.jpg"

# Ingest policy
curl -X POST http://localhost:5050/api/ingest -F "file=@new_policy.pdf"

# Health check
curl http://localhost:5050/api/health
```

## Pipeline Timing (Jetson Orin Nano 8GB)

| Stage | Engine | Time | Notes |
|-------|--------|------|-------|
| ASR | whisper.cpp CUDA | ~0.8-1.0s | Greedy decode, VAD filter |
| OCR | poppler / Tesseract | ~10-200ms | PDF text extraction instant |
| Embedding | ONNX Runtime CUDA | ~30ms | BGE-small, 384-dim |
| Search | FAISS GPU | ~0.5ms | Inner product, top-2 |
| LLM | llama.cpp CUDA | ~1.5-2.5s | 18-28 tok/s, 256 tokens max |
| TTS | Piper binary | ~0.3s | ONNX+espeak-ng |
| **Text Query** | | **~2-3s** | |
| **Audio Query** | | **~3-4s** | |

## Comparison: v2 Python vs v3 C++

| Metric | v2 (Python) | v3 (C++) | Speedup |
|--------|-------------|----------|---------|
| Binary size | ~500MB (Python + venv) | ~15MB | 33x smaller |
| Startup time | 5-10s | < 1s | 5-10x |
| LLM throughput | ~12 tok/s (Ollama) | ~18-28 tok/s | 1.5-2.3x |
| Embedding latency | ~500ms | ~30ms | 16x |
| ASR latency | ~3-5s | ~0.8-1.0s | 3-5x |
| TTS latency | ~1.5s | ~0.3s | 5x |
| Memory overhead | ~800MB | ~50MB | 16x less |
| Total pipeline | 5-6s | **< 3s** | **2x** |
| Dependencies | Python + pip + Ollama | apt + cmake | Simpler |

## Project Structure

```
netra-v3-jetson/
├── CMakeLists.txt              # Build system
├── include/                    # C++ headers
│   ├── llm_engine.h            # llama.cpp wrapper
│   ├── asr_engine.h            # whisper.cpp wrapper
│   ├── rag_engine.h            # FAISS + ONNX embedding
│   ├── tts_engine.h            # Piper TTS
│   ├── ocr_engine.h            # Tesseract + poppler
│   ├── server.h                # HTTP API server
│   └── utils.h                 # Timer, base64, logging
├── src/                        # C++ source
│   ├── main.cpp                # Entry point + CLI
│   ├── llm_engine.cpp          # LLM inference
│   ├── asr_engine.cpp          # ASR transcription
│   ├── rag_engine.cpp          # RAG search + ingest
│   ├── tts_engine.cpp          # TTS synthesis
│   ├── ocr_engine.cpp          # OCR extraction
│   ├── server.cpp              # HTTP routes
│   └── utils.cpp               # Utility functions
├── third_party/                # Git submodules
│   ├── llama.cpp/              # LLM inference engine
│   ├── whisper.cpp/            # ASR engine
│   ├── cpp-httplib/            # HTTP server (header-only)
│   └── json/                   # JSON parser (header-only)
├── models/                     # Downloaded models
│   ├── qwen2.5-3b-q4_k_m.gguf # LLM (1.8GB)
│   ├── ggml-small.bin          # Whisper (460MB)
│   └── piper/                  # TTS voices
├── knowledge_base/
│   ├── policies/               # Policy documents
│   └── vector_store/           # FAISS index
├── scripts/
│   └── setup_jetson.sh         # One-command setup
├── config/
│   └── config.yaml             # Configuration
└── README.md
```

## CLI Options

```
./netra [options]

  --llm PATH        GGUF model file              (default: models/qwen2.5-3b-q4_k_m.gguf)
  --whisper PATH    Whisper GGML model            (default: models/ggml-small.bin)
  --embedding PATH  ONNX embedding model          (default: models/bge-small-en-v1.5.onnx)
  --voices PATH     Piper voice models directory   (default: models/piper)
  --policies PATH   Policy documents directory     (default: knowledge_base/policies)
  --index PATH      FAISS vector store directory   (default: knowledge_base/vector_store)
  --host ADDR       Listen address                 (default: 0.0.0.0)
  --port PORT       Listen port                    (default: 5050)
  --max-tokens N    Max LLM output tokens          (default: 256)
  --temperature F   LLM temperature                (default: 0.3)
```

## Robustness Features

- **Graceful engine degradation** — ASR, TTS, and RAG engines are optional. If model files are missing, those engines are disabled but the server still starts and serves text queries.
- **Smart policy ingestion** — Scans multiple directories (`--policies`, `polyic/`, `../polyic/`), skips already-ingested documents using metadata tracking, handles large PDFs (400KB+) without crashes.
- **Regex-free text normalization** — OCR text processing uses manual loop-based string operations instead of `std::regex` to avoid stack overflow segfaults on large inputs.
- **LLM answer quality** — Strips `<think>` blocks and meta-commentary preamble (25+ patterns like "Hmm...", "Let me analyze...", "First document...") for clean, direct answers.
- **Keyword-overlap RAG fallback** — When FAISS/ONNX are not compiled, RAG uses word-overlap scoring to find relevant policy documents.

## Jetson Optimization

```bash
# MAX performance
sudo nvpmodel -m 0
sudo jetson_clocks
sudo sysctl -w vm.swappiness=1

# Increase CMA for CUDA unified memory
# In /boot/extlinux/extlinux.conf, add: cma=512M
```

## License

Same as NETRA v2 — see parent project LICENSE.
