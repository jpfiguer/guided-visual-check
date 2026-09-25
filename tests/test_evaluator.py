"""Evaluator.evaluate() with the API client replaced by a stub. No network."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gvc import evaluator
from gvc.checkpoint import Check, Checkpoint
from gvc.evaluator import EvaluationError, Evaluator, estimate_cost

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "images"
SUBJECT = EXAMPLES / "subject.jpg"

CHECKPOINT = Checkpoint(
    id="cp1", site="site", vantage="v", reference_image=EXAMPLES / "reference.jpg",
    checks=[
        Check(id="soiling", rule="clean"),
        Check(id="cabling", rule="clipped to the rail"),
    ],
)


def finding(check, status="pass", confidence=0.95):
    return {
        "check": check, "status": status, "observed": f"{check}: observed",
        "expected": "e", "confidence": confidence, "locator": "l",
    }


def reply(findings=(), image_usable=True, stop_reason="end_turn", stop_details=None):
    """Stands in for the SDK's Message, with the fields evaluate() reads."""
    body = {
        "image_usable": image_usable,
        "unusable_reason": "" if image_usable else "too dark",
        "findings": list(findings),
        "summary": "s",
    }
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=stop_details,
        content=[SimpleNamespace(type="text", text=json.dumps(body))],
        usage=SimpleNamespace(
            input_tokens=1000, output_tokens=200,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        ),
    )


@pytest.fixture
def api(monkeypatch):
    """Replaces the API client. Set `api.reply`; requests land in `api.requests`."""
    stub = SimpleNamespace(reply=None, requests=[])

    def create(**request):
        stub.requests.append(request)
        return stub.reply

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(evaluator, "_client", lambda: client)
    return stub


def evaluate(checkpoint=CHECKPOINT, **kwargs):
    return Evaluator(model="claude-sonnet-5", **kwargs).evaluate(checkpoint, SUBJECT)


def test_a_complete_response_is_resolved(api):
    api.reply = reply([finding("soiling"), finding("cabling")])
    result = evaluate()
    assert [f.check for f in result.findings] == ["soiling", "cabling"]
    assert api.requests[0]["max_tokens"] == evaluator.MAX_TOKENS


def test_a_refusal_raises_a_clear_error(api):
    api.reply = reply(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"))
    api.reply.content = []
    with pytest.raises(EvaluationError, match="declined.*cyber"):
        evaluate()


def test_a_response_cut_at_max_tokens_is_not_parsed(api):
    api.reply = reply([finding("soiling"), finding("cabling")], stop_reason="max_tokens")
    api.reply.content[0].text = api.reply.content[0].text[:60]
    with pytest.raises(EvaluationError, match="max_tokens"):
        evaluate()


def test_an_unknown_model_fails_before_the_call(api):
    api.reply = reply([finding("soiling"), finding("cabling")])
    with pytest.raises(ValueError, match="no prices"):
        Evaluator(model="claude-unknown").evaluate(CHECKPOINT, SUBJECT)
    assert api.requests == []


def test_cost_estimates_reject_unknown_models():
    with pytest.raises(ValueError, match="no prices"):
        estimate_cost("claude-unknown", 1000, 100)
