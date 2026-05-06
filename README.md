# ContextRevive 🧠

> **AI Context Recovery Engine** — Reconstructs missing conversation context using local LLMs (Ollama) or Groq for fast inference.

ContextRevive detects gaps in broken or truncated conversations and uses an LLM to infer what was likely discussed — so your AI assistant can keep answering coherently even when history is missing. Backed by a priority-weighted vector memory and a rolling summarizer for long sessions.

---

## ✨ What It Does

When a conversation loses turns (corrupted, truncated, context-window overflow), most chatbots break or hallucinate. ContextRevive:

1. **Detects** exactly which turns are missing
2. **Reconstructs** what was likely discussed using surrounding evidence + vector memory
3. **Scores confidence** (0–100%) and picks a recovery strategy
4. **Continues** the conversation as if the gap never happened

---

## 🏗️ Architecture

```
contextrevive/
├── core/
│   ├── ollama_client.py    ← LLM + embedding client (Ollama / Groq backends)
│   ├── tracker.py          ← Tracks turns, detects gaps (multi-range)
│   ├── memory.py           ← Priority-weighted vector memory
│   ├── summarizer.py       ← Rolling conversation summarizer
│   ├── reconstructor.py    ← Multi-gap inference engine (the heart)
│   └── agent.py            ← Orchestrator, ties everything together
├── storage/
│   └── chroma_db/          ← Persistent ChromaDB store (cosine similarity)
├── api/
│   └── server.py           ← FastAPI, 5 endpoints
├── scenarios/
│   └── conversations.json  ← Demo scripts: customer_support, student_tutor, project_planning
├── static/                 ← Frontend assets
├── contextrevive_ui.html   ← Single-page demo UI (3 scenarios + free chat)
├── ui_styles.css
├── config.py               ← All settings in one place
├── main.py                 ← Startup health check
└── requirements.txt
```

---

## ⚙️ Requirements

- **Windows** (PowerShell)
- **Python 3.11+**
- **[Ollama](https://ollama.com)** running locally with:
  ```
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```
- *(Optional)* **Groq API key** for faster inference — set `USE_GROQ=true` and `GROQ_API_KEY=...` in `.env`

---

## 🚀 Setup

```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama (in a separate terminal)
ollama serve

# 5. Verify setup
python main.py

# 6. Run the API server
uvicorn api.server:app --reload --port 8000

# 7. Open the demo UI
# Open contextrevive_ui.html in your browser (or serve via any static file server)
```

---

## 🔌 API Endpoints

All endpoints available at `http://localhost:8000`  
Interactive docs at **http://localhost:8000/docs**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Check Ollama connectivity |
| `POST` | `/sessions` | Create a new session |
| `POST` | `/sessions/{id}/chat` | Send a message |
| `POST` | `/sessions/{id}/simulate-gap` | Simulate missing turns |
| `GET`  | `/sessions/{id}/status` | Get session status |

---

## 🎮 Demo Walkthrough

### Step 1 — Health check
```bash
curl http://localhost:8000/health
# {"ollama_running": true, "model": "llama3.1:8b", "status": "ok"}
```

### Step 2 — Create session from scenario
```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d "{\"scenario\": \"student_tutor\"}"
# {"session_id": "abc-123...", "scenario": "student_tutor", "status": "ready"}
```

### Step 3 — Chat with full context
```bash
curl -X POST http://localhost:8000/sessions/abc-123.../chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Do I need a base case for fibonacci?\"}"
# {"response_text": "...", "used_reconstruction": false, "confidence": 1.0}
```

### Step 4 — Simulate a gap (turns 5–8 missing)
```bash
curl -X POST http://localhost:8000/sessions/abc-123.../simulate-gap \
  -H "Content-Type: application/json" \
  -d "{\"turn_indices\": [5, 6, 7, 8]}"
# {"gap_report": {"has_gaps": true, "severity": "medium"}, "integrity_score": 0.6667}
```

### Step 5 — Chat again — engine reconstructs the gap
```bash
curl -X POST http://localhost:8000/sessions/abc-123.../chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Do I need a base case for fibonacci?\"}"
# {"response_text": "...", "used_reconstruction": true, "confidence": 0.75, "strategy": "silent"}
```

---

## 🧠 How Reconstruction Works

```
Missing turns detected
        ↓
Get 3 turns BEFORE gap + 3 turns AFTER gap
        ↓
Retrieve top-3 relevant memories from ChromaDB
        ↓
Build structured prompt → llama3.1:8b
        ↓
Parse JSON response → confidence score
        ↓
Pick strategy:
  ≥ 0.75 → silent   (merge invisibly)
  ≥ 0.55 → soft     (flag to user)
  < 0.55 → ask_user (request clarification)
        ↓
Inject inferred context into prompt
        ↓
Generate final response
```

---

## 🎯 Memory Priority System

Memories are retrieved by a weighted priority score:

```
priority = 0.35 × recency
         + 0.30 × semantic_similarity
         + 0.20 × importance
         + 0.15 × access_frequency
```

- **Recency** — later turns in a conversation score higher
- **Semantic similarity** — cosine distance to the query (via nomic-embed-text)
- **Importance** — 1.0 if content has keywords like "refund", "order", "?", numbers; else 0.4
- **Access frequency** — starts at 0.5, increases with each retrieval (max 1.0)

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (generation) | `llama3.1:8b` via Ollama *(default)* or Groq |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector store | ChromaDB (persistent, cosine similarity) |
| Summarization | Local LLM-driven rolling summary |
| API framework | FastAPI + Uvicorn |
| Data validation | Pydantic v2 |
| Frontend | Vanilla HTML/CSS/JS (single page) |

**Default: 100% local. Optional Groq backend for fast cloud inference — no Anthropic, no OpenAI.**

---

## 📁 Demo Scenarios

Two pre-written scripts in `scenarios/conversations.json`:

| Scenario | Description |
|----------|-------------|
| `customer_support` | Damaged order → refund → replacement |
| `student_tutor` | Recursion → factorial → base case |
| `project_planning` | Multi-turn project scoping with scattered gaps |
| *Free Chat* (UI only) | Open-ended conversation, manually mark any turn(s) as missing |

The UI lets you simulate **multiple non-contiguous gap ranges** in a single session and watch the engine reconstruct each one independently.

---

## 🔧 Configuration

All settings in `config.py`:

```python
USE_GROQ = False                  # toggle Groq backend via env var USE_GROQ=true
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
MAX_TOKENS_PER_SEGMENT = 512
MAX_MEMORY_SLOTS = 50
RELEVANCE_THRESHOLD = 0.65
TOP_K_MEMORIES = 5
DB_PATH = "./storage/contextrevive.db"
CHROMA_PATH = "./storage/chroma_db"
```

`.env` overrides:
```
USE_GROQ=true
GROQ_API_KEY=gsk_...
```

---

## 🧪 Testing Individual Modules

```powershell
python core/ollama_client.py   # Ollama connectivity + embedding test
python core/tracker.py         # Gap detection + timeline
python storage/vector_store.py # ChromaDB add/search/delete
python core/memory.py          # Priority scoring + retrieval
python core/reconstructor.py   # Full reconstruction pipeline
python core/agent.py           # End-to-end: no gap vs. with gap
```

---

## ✅ What's Working

- Multi-gap reconstruction (non-contiguous ranges in a single session)
- Rolling summarizer keeps long conversations within context budget
- Groq backend for sub-second responses (Ollama remains the local default)
- 3 scripted scenarios + a free-chat mode in the demo UI
- Live transparency panel: gap ranges, integrity score, strategy, confidence

---

## 📌 Current Limitations

- No persistent sessions across server restarts (in-memory only)
- No authentication on API endpoints

---

## 🗺️ Roadmap

- [ ] Persistent sessions via SQLite
- [ ] Streaming responses
- [ ] Session export/import
- [ ] Auth + multi-user isolation

---

## 📄 License

MIT
