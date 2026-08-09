# Support Assistant — Zepto Data & AI Platform

## Overview

This module implements a **RAG (Retrieval-Augmented Generation)** based customer support assistant for Zepto. It answers customer queries by retrieving relevant content from Zepto's own policy documents rather than relying on general LLM knowledge.

**Entry point:**

```bash
cd support_assistant

# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Ingest policy documents into ChromaDB
python ingest.py

# Step 3: Start the API server (mock mode — no API key needed)
MOCK_LLM=1 uvicorn api:app --reload --port 8000

# Step 3 (production — with Gemini API key)
GOOGLE_API_KEY=your_key uvicorn api:app --reload --port 8000
```

---

## RAG Pipeline

```
Customer Question
      │
      ▼
┌─────────────────┐
│ classify_intent │  keyword-based routing
└────────┬────────┘
         │
   ┌─────┴──────┐
   │            │
"policy"    "general"
   │            │
   ▼            ▼
┌──────────────────────┐   ┌──────────────┐
│ retrieve_and_answer  │   │ direct_answer │
│                      │   │               │
│ 1. Embed query       │   │ LLM response  │
│ 2. ChromaDB search   │   │ (no retrieval)│
│ 3. Build context     │   └──────┬────────┘
│ 4. LLM synthesis     │          │
└──────────┬───────────┘          │
           └──────────┬───────────┘
                      ▼
            AssistantResponse
            { answer, sources, confidence }
```

### Components

| Component | File | Description |
|---|---|---|
| Policy documents | `docs/*.txt` | 8 Zepto policy texts |
| Ingestion | `ingest.py` | Chunk → embed → store in ChromaDB |
| Retriever | `retriever.py` | Semantic search using cosine similarity |
| Agent | `agent.py` | LangGraph StateGraph with 3 nodes |
| API | `api.py` | FastAPI with `POST /ask` |
| Tests | `test_agent.py` | Offline smoke tests (MOCK_LLM mode) |
| Docker | `Dockerfile` | Container for the full service |

---

## Policy Documents

| File | Topic |
|---|---|
| `doc_01_delivery_policy.txt` | Delivery times, scheduling, fees |
| `doc_02_refund_policy.txt` | Refund eligibility, process, timelines |
| `doc_03_membership_policy.txt` | ZeptoPass subscription benefits |
| `doc_04_privacy_policy.txt` | Data collection and protection |
| `doc_05_payment_policy.txt` | Accepted methods, COD, EMI |
| `doc_06_product_quality_policy.txt` | Freshness, sourcing standards |
| `doc_07_support_policy.txt` | Support channels and SLAs |
| `doc_08_promotions_policy.txt` | Coupons, cashback, referrals |

---

## Embedding and Vector Store

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (local, no API key)
- **Vector store:** ChromaDB (persistent, on-disk at `chroma_db/`)
- **Chunking:** 500-character chunks with 50-character overlap
- **Similarity:** Cosine distance via ChromaDB HNSW index

---

## LangGraph Architecture

**State (`AgentState`)**

```python
class AgentState(TypedDict):
    question:  str        # User question
    intent:    str        # "policy" or "general"
    chunks:    list[dict] # Retrieved document chunks
    response:  dict       # Final answer dict
```

**Nodes**

| Node | Description |
|---|---|
| `classify_intent` | Keyword-based intent detection |
| `retrieve_and_answer` | Retrieves top-3 chunks → builds prompt → calls LLM |
| `direct_answer` | Responds directly for non-policy questions |

**Routing** (conditional edge from `classify_intent`):
- `"policy"` → `retrieve_and_answer`
- `"general"` → `direct_answer`

---

## API

**Base URL:** `http://localhost:8000`

### `GET /`
Health check.
```json
{"status": "ok", "service": "Zepto Support Assistant"}
```

### `POST /ask`
Submit a question.

**Request:**
```json
{"question": "What is Zepto's refund policy for damaged items?"}
```

**Response:**
```json
{
  "answer": "Zepto accepts refund requests for damaged items...",
  "sources": ["doc_02_refund_policy.txt"],
  "confidence": 0.823
}
```

Interactive docs available at: `http://localhost:8000/docs`

---

## Running with Docker

```bash
cd support_assistant

# Build the image
docker build -t zepto-support-assistant .

# Run with mock LLM (no API key)
docker run -e MOCK_LLM=1 -p 8000:8000 zepto-support-assistant

# Run with Gemini API key
docker run -e GOOGLE_API_KEY=your_key -p 8000:8000 zepto-support-assistant
```

---

## Tests

```bash
cd support_assistant
python test_agent.py
# or
pytest test_agent.py -v
```

Tests run in `MOCK_LLM=1` mode — no API key required.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MOCK_LLM` | No | Set to any value to use mock LLM (offline mode) |
| `GOOGLE_API_KEY` | For production | Gemini API key for real LLM responses |

---

## Files

```
support_assistant/
├── docs/
│   ├── doc_01_delivery_policy.txt
│   ├── doc_02_refund_policy.txt
│   ├── doc_03_membership_policy.txt
│   ├── doc_04_privacy_policy.txt
│   ├── doc_05_payment_policy.txt
│   ├── doc_06_product_quality_policy.txt
│   ├── doc_07_support_policy.txt
│   └── doc_08_promotions_policy.txt
├── chroma_db/                  # Created by ingest.py (not committed)
├── agent.py                    # LangGraph StateGraph
├── api.py                      # FastAPI application
├── constants.py                # Shared configuration
├── ingest.py                   # Document ingestion pipeline
├── retriever.py                # ChromaDB semantic retriever
├── test_agent.py               # Smoke tests
├── requirements.txt            # Python dependencies
└── Dockerfile                  # Container configuration
```
