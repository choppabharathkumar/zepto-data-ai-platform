"""
agent.py — LangGraph-based support assistant for Zepto policy questions.

Architecture:
  StateGraph with three nodes:
    1. classify_intent   — decides if the question is policy-related or general
    2. retrieve_and_answer — RAG: retrieve docs + generate answer (policy questions)
    3. direct_answer     — direct response for non-policy / small-talk queries

  Conditional edge from classify_intent:
    "policy"  -> retrieve_and_answer
    "general" -> direct_answer

  Environment variable MOCK_LLM (any non-empty value) activates mock mode,
  which returns deterministic template answers without calling an external LLM.
  Mock mode is the default used by the automated test suite.

Output schema (Pydantic):
    answer:     str
    sources:    list[str]
    confidence: float  (0.0 to 1.0)
"""

import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from retriever import retrieve


# ── Output schema ─────────────────────────────────────────────────────────────

class AssistantResponse(BaseModel):
    """Structured response returned by the graph."""
    answer:     str
    sources:    list[str]
    confidence: float


# ── Graph state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:  str          # The user's question
    intent:    str          # "policy" or "general"
    chunks:    list[dict]   # Retrieved document chunks (empty for general)
    response:  dict         # Final answer dict matching AssistantResponse schema


# ── LLM helper ───────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """
    Call the LLM to generate a response.

    If MOCK_LLM is set, return a deterministic mock answer.
    Otherwise, use Google Generative AI (gemini-1.5-flash).
    """
    if os.environ.get("MOCK_LLM"):
        return "[MOCK] " + prompt[:120].replace("\n", " ")

    try:
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return "[LLM unavailable — set GOOGLE_API_KEY or MOCK_LLM=1]"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        return f"[LLM error: {exc}]"


# ── Node 1: classify_intent ───────────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """
    Decide whether the question is about Zepto policy or a general query.

    Uses keyword matching in mock/offline mode.
    Uses the LLM for classification only when MOCK_LLM is not set.
    """
    question = state["question"]

    # Keyword-based classification (fast, no LLM call needed)
    policy_keywords = [
        "delivery", "refund", "return", "payment", "cancel", "order",
        "membership", "zeptopass", "privacy", "data", "coupon", "promo",
        "offer", "discount", "fresh", "quality", "support", "charge",
        "wallet", "expire", "subscription", "cod", "cash",
    ]
    q_lower = question.lower()
    is_policy = any(kw in q_lower for kw in policy_keywords)

    intent = "policy" if is_policy else "general"
    return {**state, "intent": intent}


# ── Node 2: retrieve_and_answer ───────────────────────────────────────────────

def retrieve_and_answer(state: AgentState) -> AgentState:
    """
    Retrieve the most relevant policy chunks and synthesise an answer.
    """
    question = state["question"]
    chunks   = retrieve(question)

    if not chunks:
        response = {
            "answer":     "I could not find relevant information in the policy documents.",
            "sources":    [],
            "confidence": 0.0,
        }
        return {**state, "chunks": [], "response": response}

    # Build context from retrieved chunks
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}: {c['source']}]\n{c['document']}")
    context = "\n\n".join(context_parts)

    prompt = (
        f"You are a helpful customer support assistant for Zepto, a quick-commerce grocery delivery app.\n"
        f"Answer the customer's question using ONLY the information provided in the context below.\n"
        f"Be concise, factual, and friendly. If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Customer question: {question}\n\n"
        f"Answer:"
    )

    answer = _call_llm(prompt)

    # Confidence heuristic: based on the distance of the top result
    # ChromaDB cosine distance: 0.0 = identical, 2.0 = completely dissimilar
    top_distance = chunks[0]["distance"]
    confidence   = round(max(0.0, 1.0 - top_distance / 2.0), 3)

    sources = list(dict.fromkeys(c["source"] for c in chunks))  # deduplicated, order preserved

    response = {
        "answer":     answer,
        "sources":    sources,
        "confidence": confidence,
    }
    return {**state, "chunks": chunks, "response": response}


# ── Node 3: direct_answer ─────────────────────────────────────────────────────

def direct_answer(state: AgentState) -> AgentState:
    """
    Handle non-policy questions with a direct, friendly response.
    No documents are retrieved.
    """
    question = state["question"]

    prompt = (
        f"You are a friendly customer support assistant for Zepto grocery delivery.\n"
        f"The customer has asked a general question that is not about Zepto's policies.\n"
        f"Respond helpfully and suggest that for policy-specific questions they can ask about\n"
        f"delivery, refunds, payments, memberships, or promotions.\n\n"
        f"Customer: {question}\n\nAssistant:"
    )

    answer = _call_llm(prompt)

    response = {
        "answer":     answer,
        "sources":    [],
        "confidence": 1.0,  # full confidence — no retrieval uncertainty
    }
    return {**state, "chunks": [], "response": response}


# ── Graph construction ────────────────────────────────────────────────────────

def _route_intent(state: AgentState) -> str:
    """Conditional edge: route based on classified intent."""
    return state["intent"]


def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent",     classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer",       direct_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "policy":  "retrieve_and_answer",
            "general": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer",       END)

    return graph.compile()


# ── Public interface ──────────────────────────────────────────────────────────

_graph = None


def get_graph():
    """Return the compiled graph (singleton)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(question: str) -> AssistantResponse:
    """
    Ask the support assistant a question.

    Args:
        question: The customer's question string.

    Returns:
        AssistantResponse with answer, sources, and confidence.
    """
    graph = get_graph()
    initial_state: AgentState = {
        "question": question,
        "intent":   "",
        "chunks":   [],
        "response": {},
    }
    final_state = graph.invoke(initial_state)
    resp = final_state["response"]
    return AssistantResponse(
        answer=resp["answer"],
        sources=resp["sources"],
        confidence=resp["confidence"],
    )
