#!/bin/bash
# NETRA v3 — Jetson Orin Nano C++ Build & Setup
# Usage: bash scripts/setup_jetson.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY="$PROJECT_DIR/third_party"
MODELS_DIR="$PROJECT_DIR/models"
PIPER_DIR="$MODELS_DIR/piper"

echo "============================================"
echo "  NETRA v3 — C++ Native Build (Jetson)"
echo "  No Python, No Ollama, Pure C/C++ + CUDA"
echo "============================================"

# ─────────────────────────────────────────────
# 1. System dependencies
# ─────────────────────────────────────────────
echo ""
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake git wget \
    libfaiss-dev \
    tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng \
    poppler-utils \
    libsndfile1-dev \
    2>/dev/null || echo "Some packages may already be installed"

# Install piper TTS binary (prebuilt for aarch64)
if ! command -v piper &>/dev/null; then
    echo "Installing Piper TTS binary..."
    PIPER_VERSION="2023.11.14-2"
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz"
    wget -q "$PIPER_URL" -O /tmp/piper.tar.gz 2>/dev/null || {
        echo "Piper download failed. Install manually: https://github.com/rhasspy/piper/releases"
    }
    if [ -f /tmp/piper.tar.gz ]; then
        tar -xzf /tmp/piper.tar.gz -C /usr/local/bin/ piper/piper 2>/dev/null || true
        sudo mv /usr/local/bin/piper/piper /usr/local/bin/piper 2>/dev/null || true
        rm -rf /tmp/piper.tar.gz /usr/local/bin/piper/
    fi
fi

# ─────────────────────────────────────────────
# 2. Clone third-party libraries
# ─────────────────────────────────────────────
echo ""
echo "[2/7] Fetching third-party libraries..."
mkdir -p "$THIRD_PARTY"

# llama.cpp
if [ ! -d "$THIRD_PARTY/llama.cpp" ]; then
    echo "  Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$THIRD_PARTY/llama.cpp"
else
    echo "  llama.cpp already exists"
fi

# whisper.cpp
if [ ! -d "$THIRD_PARTY/whisper.cpp" ]; then
    echo "  Cloning whisper.cpp..."
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git "$THIRD_PARTY/whisper.cpp"
else
    echo "  whisper.cpp already exists"
fi

# cpp-httplib (single header)
if [ ! -f "$THIRD_PARTY/cpp-httplib/httplib.h" ]; then
    echo "  Downloading cpp-httplib..."
    mkdir -p "$THIRD_PARTY/cpp-httplib"
    wget -q "https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h" \
         -O "$THIRD_PARTY/cpp-httplib/httplib.h"
else
    echo "  cpp-httplib already exists"
fi

# nlohmann/json (single header)
if [ ! -f "$THIRD_PARTY/json/single_include/nlohmann/json.hpp" ]; then
    echo "  Downloading nlohmann/json..."
    mkdir -p "$THIRD_PARTY/json/single_include/nlohmann"
    wget -q "https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp" \
         -O "$THIRD_PARTY/json/single_include/nlohmann/json.hpp"
else
    echo "  nlohmann/json already exists"
fi

# ─────────────────────────────────────────────
# 3. Download models
# ─────────────────────────────────────────────
echo ""
echo "[3/7] Downloading models..."
mkdir -p "$MODELS_DIR" "$PIPER_DIR"

# LLM GGUF
GGUF_FILE="$MODELS_DIR/qwen2.5-3b-q4_k_m.gguf"
if [ ! -f "$GGUF_FILE" ]; then
    echo "  Downloading Qwen2.5-3B Q4_K_M (~1.8GB)..."
    wget -q --show-progress \
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
        -O "$GGUF_FILE" || echo "  Download failed — get GGUF manually"
else
    echo "  LLM model exists: $GGUF_FILE"
fi

# Whisper GGML
WHISPER_FILE="$MODELS_DIR/ggml-small.bin"
if [ ! -f "$WHISPER_FILE" ]; then
    echo "  Downloading Whisper Small GGML (~460MB)..."
    wget -q --show-progress \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" \
        -O "$WHISPER_FILE" || echo "  Download failed — get whisper model manually"
else
    echo "  Whisper model exists: $WHISPER_FILE"
fi

# Piper Hindi voice
if [ ! -f "$PIPER_DIR/hi_IN-medium.onnx" ]; then
    echo "  Downloading Hindi TTS voice..."
    PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
    wget -q "$PIPER_BASE/hi/hi_IN/medium/hi_IN-medium.onnx" -O "$PIPER_DIR/hi_IN-medium.onnx" || true
    wget -q "$PIPER_BASE/hi/hi_IN/medium/hi_IN-medium.onnx.json" -O "$PIPER_DIR/hi_IN-medium.onnx.json" || true
else
    echo "  Hindi TTS voice exists"
fi

# Piper English voice
if [ ! -f "$PIPER_DIR/en_US-lessac-medium.onnx" ]; then
    echo "  Downloading English TTS voice..."
    PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
    wget -q "$PIPER_BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -O "$PIPER_DIR/en_US-lessac-medium.onnx" || true
    wget -q "$PIPER_BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -O "$PIPER_DIR/en_US-lessac-medium.onnx.json" || true
else
    echo "  English TTS voice exists"
fi

# ─────────────────────────────────────────────
# 4. Build
# ─────────────────────────────────────────────
echo ""
echo "[4/7] Building NETRA v3 (C++ native)..."
cd "$PROJECT_DIR"
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DNETRA_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    2>&1 | tail -5

make -j$(nproc) 2>&1 | tail -10

if [ -f ./netra ]; then
    echo "  ✓ Build successful: $(pwd)/netra"
else
    echo "  ✗ Build failed — check errors above"
    exit 1
fi

# ─────────────────────────────────────────────
# 5. Jetson optimization
# ─────────────────────────────────────────────
echo ""
echo "[5/7] Jetson optimizations..."
if [ -f /etc/nv_tegra_release ]; then
    echo "  Setting MAX performance mode..."
    sudo nvpmodel -m 0 2>/dev/null || true
    sudo jetson_clocks 2>/dev/null || true
    sudo sysctl -w vm.swappiness=1 2>/dev/null || true
    echo "  ✓ Performance mode set"
else
    echo "  Not on Jetson — skipping hardware optimizations"
fi

# ─────────────────────────────────────────────
# 6. Copy policy documents
# ─────────────────────────────────────────────
echo ""
echo "[6/7] Setting up knowledge base..."
NETRA_V2_POLICIES="$PROJECT_DIR/../polyic"
if [ -d "$NETRA_V2_POLICIES" ]; then
    echo "  Copying policy documents from NETRA v2..."
    cp -r "$NETRA_V2_POLICIES"/* "$PROJECT_DIR/knowledge_base/policies/" 2>/dev/null || true
    echo "  ✓ Policies copied"
else
    echo "  No NETRA v2 policies found at $NETRA_V2_POLICIES"
fi

# ─────────────────────────────────────────────
# 7. Done
# ─────────────────────────────────────────────
echo ""
echo "[7/7] Setup complete!"
echo ""
echo "============================================"
echo "  NETRA v3 — Ready to Run"
echo "============================================"
echo ""
echo "  Start:"
echo "    cd $PROJECT_DIR/build"
echo "    ./netra --llm ../models/qwen2.5-3b-q4_k_m.gguf \\"
echo "            --whisper ../models/ggml-small.bin \\"
echo "            --port 5050"
echo ""
echo "  API:  http://localhost:5050"
echo ""
echo "  Models:"
echo "    LLM:     $GGUF_FILE (~1.8GB)"
echo "    Whisper:  $WHISPER_FILE (~460MB)"
echo "    TTS:     $PIPER_DIR/"
echo ""
echo "  Performance target: < 3s total pipeline"
echo "============================================"
