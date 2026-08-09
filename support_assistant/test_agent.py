"""
test_agent.py — Smoke tests for the support assistant.

Tests the full pipeline in mock mode (no external API key required).
MOCK_LLM=1 is set inside the test so it runs offline in any environment.

Run:
    python test_agent.py
    # or
    pytest test_agent.py -v
"""

import os
os.environ["MOCK_LLM"] = "1"   # Force mock mode — no LLM API key needed

import pytest
from agent import ask, AssistantResponse


def test_policy_question_returns_sources():
    """A delivery question should retrieve policy documents and return sources."""
    resp = ask("What is Zepto's refund policy for damaged items?")
    assert isinstance(resp, AssistantResponse)
    assert len(resp.answer) > 0
    assert len(resp.sources) > 0
    assert 0.0 <= resp.confidence <= 1.0


def test_policy_question_delivery():
    """A delivery question should be classified as policy."""
    resp = ask("How long does delivery take?")
    assert isinstance(resp, AssistantResponse)
    assert resp.answer != ""


def test_general_question_no_sources():
    """A general (non-policy) question should return empty sources list."""
    resp = ask("Hello, how are you today?")
    assert isinstance(resp, AssistantResponse)
    assert isinstance(resp.sources, list)
    # General queries get no sources (routed to direct_answer)
    assert resp.sources == []


def test_response_schema():
    """Response must always match the AssistantResponse schema."""
    resp = ask("Can I cancel my ZeptoPass membership?")
    assert hasattr(resp, "answer")
    assert hasattr(resp, "sources")
    assert hasattr(resp, "confidence")
    assert isinstance(resp.answer, str)
    assert isinstance(resp.sources, list)
    assert isinstance(resp.confidence, float)


def test_payment_question():
    """Payment-related question should be classified as policy."""
    resp = ask("What payment methods does Zepto accept?")
    assert isinstance(resp, AssistantResponse)
    assert len(resp.sources) > 0


if __name__ == "__main__":
    print("Running smoke tests in MOCK_LLM mode...\n")

    tests = [
        test_policy_question_returns_sources,
        test_policy_question_delivery,
        test_general_question_no_sources,
        test_response_schema,
        test_payment_question,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {test_fn.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")
