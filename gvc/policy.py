"""The decision layer: maps each finding to an action.

A fail below the confidence threshold goes to human review instead of the
subject, and so does a pass below it. README.md (decision 2) explains the
asymmetry.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Finding, ResolvedFinding, Status

#: Starting point, meant to be calibrated against labeled data for the domain.
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
            # A low-confidence failure goes to a human instead of the subject.
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
