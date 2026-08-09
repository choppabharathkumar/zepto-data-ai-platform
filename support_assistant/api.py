"""
api.py — FastAPI application for the Zepto Support Assistant.

Endpoints:
    GET  /         — health check
    POST /ask       — submit a question, get a structured answer

Run locally:
    uvicorn api:app --reload --port 8000

With mock LLM (no API key needed):
    MOCK_LLM=1 uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import ask, AssistantResponse


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "A RAG-based customer support assistant that answers questions "
        "grounded in Zepto's policy documents."
    ),
    version="1.0.0",
)


# ── Request / response schemas ────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What is your refund policy for damaged items?"},
                {"question": "How long does delivery take?"},
                {"question": "Can I cancel my ZeptoPass subscription?"},
            ]
        }
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def health():
    """Return service status."""
    return {"status": "ok", "service": "Zepto Support Assistant"}


@app.post("/ask", response_model=AssistantResponse, summary="Ask a policy question")
def ask_question(request: QuestionRequest) -> AssistantResponse:
    """
    Submit a customer support question.

    The assistant will:
    1. Classify the intent (policy-related vs general)
    2. Retrieve relevant policy chunks if policy-related
    3. Generate a grounded answer

    Returns a JSON object with `answer`, `sources`, and `confidence`.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = ask(request.question.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    return response
