"""The policy is the product. These tests are the specification of it."""

import pytest

from gvc.policy import Policy
from gvc.schema import Finding, Status


def finding(status: Status, confidence: float) -> Finding:
    return Finding(
        check="c1",
        status=status,
        observed="observed",
        expected="expected",
        confidence=confidence,
        locator="somewhere",
    )


def test_confident_failure_is_reported():
    r = Policy().resolve(finding(Status.fail, 0.95))
    assert r.needs_human_review is False
    assert "report" in r.action


def test_unconfident_failure_never_reaches_the_subject():
    """The single most important behaviour in this package.

    A suspected failure the model is unsure about must not be reported as a
    failure. If this test ever goes green with needs_human_review False, the
    system has started accusing people on a hunch.
    """
    r = Policy().resolve(finding(Status.fail, 0.5))
    assert r.needs_human_review is True
    assert r.status is Status.fail  # the evidence is preserved, only the action changes


def test_not_assessable_always_goes_to_a_human_however_confident():
    r = Policy().resolve(finding(Status.not_assessable, 1.0))
    assert r.needs_human_review is True


def test_unconfident_pass_is_also_reviewed():
    """Asymmetry has a cost and this is it: low-confidence passes are misses
    waiting to happen, so they are surfaced too."""
    r = Policy().resolve(finding(Status.pass_, 0.4))
    assert r.needs_human_review is True


def test_confident_pass_is_silent():
    r = Policy().resolve(finding(Status.pass_, 0.99))
    assert r.needs_human_review is False
    assert r.action == "pass"


@pytest.mark.parametrize("threshold,expected_review", [(0.3, False), (0.9, True)])
def test_threshold_is_the_tuning_knob(threshold, expected_review):
    r = Policy(confidence_threshold=threshold).resolve(finding(Status.fail, 0.6))
    assert r.needs_human_review is expected_review


def test_evidence_survives_resolution():
    """Applying policy must not destroy what the model said. Otherwise a human
    reviewing the queue has nothing to review."""
    f = finding(Status.fail, 0.5)
    r = Policy().resolve(f)
    assert (r.observed, r.expected, r.locator, r.confidence) == (
        f.observed, f.expected, f.locator, f.confidence,
    )
