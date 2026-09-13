#!/bin/bash
# NETRA v2 — Start both Backend and UI servers
# Usage: ./start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Add Ollama to PATH if installed locally
if [ -d "$HOME/opt/ollama/bin" ]; then
    export PATH="$HOME/opt/ollama/bin:$PATH"
fi

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║  NETRA v2 — Suno Sutra Platform               ║"
echo "  ║  Starting Backend + UI...                      ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# Step 0: Ensure Ollama server is running (for LLM inference)
if command -v ollama &>/dev/null; then
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "  [0/2] Starting Ollama server..."
        nohup ollama serve > /tmp/ollama-serve.log 2>&1 &
        OLLAMA_PID=$!
        sleep 2
        echo "        Ollama PID: $OLLAMA_PID"
    else
        echo "  [0/2] Ollama server already running."
    fi
else
    echo "  [0/2] Ollama not found — LLM will use HuggingFace fallback."
fi

# Prepare document inbox for auto-ingest
mkdir -p "$SCRIPT_DIR/Backend/data/inbox/processed"
mkdir -p "$SCRIPT_DIR/Backend/data/inbox/failed"

# Seed knowledge base if empty (first run)
KB_META="$SCRIPT_DIR/Backend/knowledge_base/vector_store/metadata.json"
if [ ! -f "$KB_META" ] || [ "$(wc -c < "$KB_META" 2>/dev/null)" -lt 100 ]; then
    echo "  [*] Seeding knowledge base with government scheme data..."
    cd "$SCRIPT_DIR/Backend"
    python3 scripts/seed_knowledge_base.py --append 2>/dev/null || echo "        (seed skipped — will seed on next restart)"
fi

# Step 1: Start Backend (Flask ML Server) on port 5000
echo "  [1/2] Starting Backend ML Server (port 5000)..."
cd "$SCRIPT_DIR/Backend"
python3 server.py &
BACKEND_PID=$!
echo "        Backend PID: $BACKEND_PID"

# Wait for Backend readiness (/api/ready = pipeline + Ollama)
echo "        Waiting for Backend readiness..."
BACKEND_READY=0
for i in $(seq 1 60); do
    if curl -sf http://localhost:5000/api/ready > /dev/null 2>&1; then
        echo "        Backend ready!"
        BACKEND_READY=1
        break
    fi
    # Fall back to liveness after pipeline boots without Ollama
    if curl -sf http://localhost:5000/api/health > /dev/null 2>&1; then
        echo "        Backend alive (waiting for /api/ready)..."
    fi
    sleep 2
done

if [ "$BACKEND_READY" -ne 1 ]; then
    echo "        ERROR: Backend did not become ready in time."
    echo "        Check Ollama (localhost:11434) and Backend logs."
    kill $BACKEND_PID ${OLLAMA_PID:-} 2>/dev/null || true
    exit 1
fi

# Step 2: Start UI (Node.js) on port 3000
echo "  [2/2] Starting UI Server (port 3000)..."
cd "$SCRIPT_DIR/UI"
npm install --silent 2>/dev/null
node server.js &
UI_PID=$!
echo "        UI PID: $UI_PID"

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║  NETRA v2 — Running!                          ║"
echo "  ║  UI:      http://localhost:3000                ║"
echo "  ║  Backend: http://localhost:5000                ║"
echo "  ║  Ready:   http://localhost:5000/api/ready      ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""
echo "  Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $UI_PID ${OLLAMA_PID:-} 2>/dev/null; exit" SIGINT SIGTERM
wait
