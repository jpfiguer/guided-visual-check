"""The decision layer. This is the part that is deliberately not in the prompt.

A vision model is good at saying what it sees and roughly how sure it is. It is
not the right place to decide what happens as a consequence, for three reasons:

1. A prompt instruction is a request, not a guarantee. A threshold in code runs
   every time, identically.
2. Thresholds get tuned. Tuning a number in a dataclass is a code review;
   tuning a sentence inside a prompt silently changes everything else the model
   does with that prompt.
3. When the decision is questioned later — and it will be — you need to be able
   to point at the rule that produced it.

The asymmetry encoded below is domain knowledge, not a general truth: in an
inspection system a false accusation costs far more than a miss. One wrong
"you failed" destroys trust faster than ten quiet catches build it, because the
person on the receiving end stops believing the tool. So a `fail` that the model
is not confident about never reaches the subject of the inspection; it goes to a
human first. Systems where a miss is the expensive error want the opposite
asymmetry, and should say so here rather than anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Finding, ResolvedFinding, Status

#: Starting point, meant to be calibrated against labelled data for the domain.
#: It is a constant here so that changing it is a visible, reviewable diff.
DEFAULT_CONFIDENCE_THRESHOLD = 0.75


@dataclass(frozen=True)
class Policy:
    """Turns reported evidence into an operational action."""

    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD

    def resolve(self, finding: Finding) -> ResolvedFinding:
        low_confidence = finding.confidence < self.confidence_threshold

        if finding.status is Status.not_assessable:
            review, action = True, "human review: the check could not be assessed"
        elif finding.status is Status.fail and low_confidence:
            # The case the whole design exists to protect: a suspected failure
            # the model is not sure about is never reported as a failure.
            review, action = True, "human review: possible failure below confidence threshold"
        elif finding.status is Status.fail:
            review, action = False, "failure: report to the subject"
        elif low_confidence:
            review, action = True, "human review: passed, but below confidence threshold"
        else:
            review, action = False, "pass"

        return ResolvedFinding(
            check=finding.check,
            status=finding.status,
            observed=finding.observed,
            expected=finding.expected,
            confidence=finding.confidence,
            locator=finding.locator,
            needs_human_review=review,
            action=action,
        )

    def resolve_all(self, findings: list[Finding]) -> list[ResolvedFinding]:
        return [self.resolve(f) for f in findings]
