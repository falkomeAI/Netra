# NETRA v2 — CPU deploy image (Backend :5000 + UI :3000)
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=3000 \
    BACKEND_PORT=5000 \
    BACKEND_URL=http://127.0.0.1:5000 \
    NETRA_CONFIG=cpu.yaml \
    FLAGS_use_mkldnn=0 \
    FLAGS_use_onednn=0

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates ffmpeg nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY profal-v2/Backend/requirements.txt /app/Backend/requirements.txt

# Core API deps first (full ML stack is large — install subset for container smoke)
RUN pip install --no-cache-dir \
      flask flask-cors pyyaml numpy pillow requests \
      sentence-transformers faiss-cpu networkx \
      pytest ruff

COPY profal-v2/Backend /app/Backend
COPY profal-v2/UI /app/UI
COPY profal-v2/start.sh /app/start.sh
RUN chmod +x /app/start.sh \
    && cd /app/UI && npm install --omit=dev

EXPOSE 3000 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5000/api/ready || exit 1

CMD ["/app/start.sh"]
