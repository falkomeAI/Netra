/**
 * NETRA UI Server — Serves frontend + proxies API calls to Backend ML server.
 *
 * Run:
 *   cd UI
 *   npm install
 *   node server.js
 *
 * Expects Backend running at BACKEND_URL (default: http://localhost:5000)
 */

const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { WebSocketServer } = require("ws");
const path = require("path");
const fs = require("fs");
const http = require("http");

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000";

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

if (!fs.existsSync(path.join(__dirname, "uploads"))) {
  fs.mkdirSync(path.join(__dirname, "uploads"), { recursive: true });
}

const upload = multer({
  dest: path.join(__dirname, "uploads"),
  limits: { fileSize: 50 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const allowed = /\.(png|jpg|jpeg|tiff|bmp|pdf|wav|mp3|ogg|webm|m4a|txt|doc|docx)$/i;
    cb(null, allowed.test(path.extname(file.originalname)));
  },
});

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

function broadcast(data) {
  const msg = JSON.stringify(data);
  wss.clients.forEach((c) => {
    if (c.readyState === 1) c.send(msg);
  });
}

// ── Proxy helpers ───────────────────────────────────────────

async function proxyJSON(method, backendPath, body) {
  const url = `${BACKEND_URL}${backendPath}`;
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(url, opts);
  let data;
  try {
    data = await resp.json();
  } catch {
    data = { error: `Backend returned non-JSON response (HTTP ${resp.status})` };
  }
  return { status: resp.status, data };
}

async function proxyFormData(backendPath, filePath, formFields) {
  const { FormData, File } = await import("undici");
  const form = new FormData();

  const fieldName = formFields.fieldName || "file";
  const fileBuffer = fs.readFileSync(filePath);
  const fileName = formFields.fileName || path.basename(filePath);
  const blob = new File([fileBuffer], fileName);
  form.append(fieldName, blob);

  for (const [key, val] of Object.entries(formFields)) {
    if (key !== "fieldName" && key !== "fileName") {
      form.append(key, val);
    }
  }

  const resp = await fetch(`${BACKEND_URL}${backendPath}`, {
    method: "POST",
    body: form,
  });
  let data;
  try {
    data = await resp.json();
  } catch {
    data = { error: `Backend returned non-JSON response (HTTP ${resp.status})` };
  }
  return { status: resp.status, data };
}

// ── REST API Routes ─────────────────────────────────────────

app.get("/api/health", async (_req, res) => {
  try {
    const { status, data } = await proxyJSON("GET", "/api/health");
    res.status(status || 200).json(data);
  } catch {
    // Do not report status:ok when backend is down — breaks deploy probes.
    res.status(503).json({
      status: "error",
      pipeline_ready: false,
      ready: false,
      backend: "unavailable",
    });
  }
});

app.get("/api/ready", async (_req, res) => {
  try {
    const { status, data } = await proxyJSON("GET", "/api/ready");
    res.status(status || 200).json(data);
  } catch {
    res.status(503).json({ ready: false, status: "not_ready", backend: "unavailable" });
  }
});

app.get("/api/config", async (_req, res) => {
  try {
    const { data } = await proxyJSON("GET", "/api/config");
    res.json(data);
  } catch (e) {
    res.status(503).json({ error: "Backend unavailable" });
  }
});

app.post("/api/pipeline/run", upload.any(), async (req, res) => {
  const file = req.files && req.files[0];
  const lang = req.body.language || req.body.lang || "hi";
  const mode = req.body.mode || "camera";
  const state = req.body.state || "";
  const district = req.body.district || "";

  if (!file) return res.status(400).json({ error: "No file uploaded" });

  broadcast({ type: "pipeline:start", file: file.originalname, language: lang, timestamp: new Date().toISOString() });

  try {
    const ext = path.extname(file.originalname || "").toLowerCase();
    let fieldName;
    if (mode === "voice" || [".wav", ".mp3", ".ogg", ".webm", ".m4a"].includes(ext)) {
      fieldName = "audio";
    } else if (ext === ".pdf") {
      fieldName = "file";
    } else {
      fieldName = "image";
    }

    const { status, data } = await proxyFormData("/api/pipeline/run", file.path, {
      fieldName,
      fileName: file.originalname || "capture.jpg",
      language: lang,
      mode,
      state,
      district,
    });

    broadcast({ type: "pipeline:complete", result: data });
    res.status(status).json(data);
  } catch (e) {
    broadcast({ type: "pipeline:error", error: e.message });
    res.status(500).json({ error: "Backend error: " + e.message });
  } finally {
    try { fs.unlinkSync(file.path); } catch {}
  }
});

app.post("/api/transcribe", upload.any(), async (req, res) => {
  const file = req.files && req.files[0];
  if (!file) return res.status(400).json({ error: "No audio file" });

  const lang = req.body.lang || req.body.language || "hi";

  try {
    const { status, data } = await proxyFormData("/api/transcribe", file.path, {
      fieldName: "audio",
      fileName: file.originalname,
      lang,
    });
    res.status(status).json(data);
  } catch (e) {
    res.status(500).json({ error: "ASR error: " + e.message });
  } finally {
    try { fs.unlinkSync(file.path); } catch {}
  }
});

app.post("/api/pipeline/text", express.json(), async (req, res) => {
  const text = req.body.text;
  if (!text) return res.status(400).json({ error: "No text provided" });

  const lang = req.body.language || "hi";
  const mode = req.body.mode || "auto";
  broadcast({ type: "pipeline:start", file: "(text input)", language: lang });

  try {
    const { status, data } = await proxyJSON("POST", "/api/pipeline/text", { text, language: lang, mode });
    broadcast({ type: "pipeline:complete", result: data });
    res.status(status).json(data);
  } catch (e) {
    res.status(500).json({ error: "Backend error: " + e.message });
  }
});

app.post("/api/pipeline/run-text", express.json(), async (req, res) => {
  const text = req.body.text;
  if (!text) return res.status(400).json({ error: "No text provided" });

  const lang = req.body.language || "hi";
  broadcast({ type: "pipeline:start", file: "(text input)", language: lang });

  try {
    const { status, data } = await proxyJSON("POST", "/api/pipeline/text", { text, language: lang });
    broadcast({ type: "pipeline:complete", result: data });
    res.status(status).json(data);
  } catch (e) {
    res.status(500).json({ error: "Backend error: " + e.message });
  }
});

app.post("/api/ingest", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded", success: false });

  const lang = req.body.lang || req.body.language || "hi";
  broadcast({ type: "ingest:start", file: req.file.originalname });

  try {
    const { status, data } = await proxyFormData("/api/ingest", req.file.path, {
      fieldName: "file",
      fileName: req.file.originalname,
      language: lang,
    });
    broadcast({ type: "ingest:complete", result: data });
    res.status(status).json(data);
  } catch (e) {
    res.status(500).json({ error: e.message, success: false });
  } finally {
    try { fs.unlinkSync(req.file.path); } catch {}
  }
});

app.get("/api/ingest/status", async (_req, res) => {
  try {
    const { data } = await proxyJSON("GET", "/api/ingest/status");
    res.json(data);
  } catch (e) {
    res.json({ watcher: { is_running: false }, knowledge_base: { total_chunks: 0, total_documents: 0 } });
  }
});

app.post("/api/ingest/scan", async (_req, res) => {
  try {
    const { status, data } = await proxyJSON("POST", "/api/ingest/scan");
    res.status(status).json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/api/ingest/bulk", upload.array("files", 50), async (req, res) => {
  const files = req.files;
  if (!files || files.length === 0) return res.status(400).json({ error: "No files", success: false });

  const results = { success: true, processed: 0, failed: 0, files: [] };

  for (const file of files) {
    try {
      const { data } = await proxyFormData("/api/ingest", file.path, {
        fieldName: "file",
        fileName: file.originalname,
        language: req.body.lang || "hi",
      });
      if (data.success) {
        results.processed++;
        results.files.push({ name: file.originalname, status: "ok" });
      } else {
        results.failed++;
        results.files.push({ name: file.originalname, status: "error", error: data.error });
      }
    } catch (e) {
      results.failed++;
      results.files.push({ name: file.originalname, status: "error", error: e.message });
    } finally {
      try { fs.unlinkSync(file.path); } catch {}
    }
  }

  res.json(results);
});

app.post("/api/ingest/text", express.json(), async (req, res) => {
  const text = req.body.text;
  if (!text) return res.status(400).json({ error: "No text provided" });

  try {
    const { status, data } = await proxyJSON("POST", "/api/ingest", { text });
    res.status(status).json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/graph", async (_req, res) => {
  try {
    const { data } = await proxyJSON("GET", "/api/graph");
    res.json(data);
  } catch {
    res.json({ nodes: [], edges: [] });
  }
});

app.get("/api/news", async (req, res) => {
  try {
    const lang = req.query.lang || "hi";
    const { data } = await proxyJSON("GET", `/api/news?lang=${lang}`);
    res.json(data);
  } catch {
    res.json({ news: [], count: 0 });
  }
});

app.post("/api/location/lookup", express.json(), async (req, res) => {
  try {
    const { status, data } = await proxyJSON("POST", "/api/location/lookup", req.body);
    res.status(status || 200).json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/weather", async (req, res) => {
  try {
    const state = req.query.state || "";
    const { data } = await proxyJSON("GET", `/api/weather?state=${encodeURIComponent(state)}`);
    res.json(data);
  } catch {
    res.json({ alerts: [] });
  }
});

app.get("/api/mandi", async (req, res) => {
  try {
    const state = req.query.state || "";
    const { data } = await proxyJSON("GET", `/api/mandi?state=${encodeURIComponent(state)}`);
    res.json(data);
  } catch {
    res.json({ prices: [] });
  }
});

app.get("/api/reminders", async (_req, res) => {
  try {
    const { data } = await proxyJSON("GET", "/api/reminders");
    res.json(data);
  } catch {
    res.json({ reminders: [] });
  }
});

app.get("/api/output/pdf", async (req, res) => {
  try {
    const lang = req.query.language || "hi";
    const resp = await fetch(`${BACKEND_URL}/api/output/pdf?language=${lang}`);
    if (resp.ok) {
      res.set("Content-Type", resp.headers.get("content-type") || "application/pdf");
      const buffer = Buffer.from(await resp.arrayBuffer());
      res.send(buffer);
    } else {
      res.status(500).json({ error: "PDF generation failed" });
    }
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/output/:filename", async (req, res) => {
  try {
    const resp = await fetch(`${BACKEND_URL}/api/output/${req.params.filename}`);
    if (resp.ok) {
      res.set("Content-Type", resp.headers.get("content-type") || "application/octet-stream");
      const buffer = Buffer.from(await resp.arrayBuffer());
      res.send(buffer);
    } else {
      res.status(404).json({ error: "File not found" });
    }
  } catch {
    res.status(500).json({ error: "Backend unavailable" });
  }
});

app.get("/api/audio/:filename", async (req, res) => {
  try {
    const resp = await fetch(`${BACKEND_URL}/api/output/${req.params.filename}`);
    if (resp.ok) {
      res.set("Content-Type", "audio/wav");
      const buffer = Buffer.from(await resp.arrayBuffer());
      res.send(buffer);
    } else {
      res.status(404).json({ error: "Audio not found" });
    }
  } catch {
    res.status(500).json({ error: "Backend unavailable" });
  }
});

app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// ── Start ───────────────────────────────────────────────────

server.listen(PORT, () => {
  console.log(`
  ╔══════════════════════════════════════════════╗
  ║  NETRA v2 — UI Server                        ║
  ║  Frontend:  http://localhost:${PORT}              ║
  ║  Backend:   ${BACKEND_URL}            ║
  ╚══════════════════════════════════════════════╝
`);
});
