"""What the model returns, and what the code adds afterwards.

The split in this file is the whole idea: `Evaluation` and `Finding` are
**evidence** — what the model saw, and how sure it is. Nothing in them is a
decision. `ResolvedFinding` is what comes out the other side, after the policy
in `policy.py` has turned that evidence into an action.

Keeping the two apart is what makes the system auditable. When someone asks
"why was this flagged?", the answer is a threshold in readable code, not a
sentence buried in a prompt that may or may not have been followed.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Status(str, Enum):
    """The three outcomes a single check can have.

    `not_assessable` is not a failure mode, it is a first-class answer. A model
    that cannot see the thing it was asked about should say so rather than
    guess, and the policy routes that to a human.
    """

    pass_ = "pass"
    fail = "fail"
    not_assessable = "not_assessable"


class Finding(BaseModel):
    """One check, as reported by the model. No decisions here."""

    check: str = Field(description="check id, exactly as it appears in the checkpoint")
    status: Status
    observed: str = Field(description="what is visible in the subject image, in one sentence")
    expected: str = Field(description="what the reference or the rule asks for, in one sentence")
    confidence: float = Field(description="0 to 1")
    locator: str = Field(
        description="where to look in the subject image, described in words "
        "(e.g. 'lower left panel, near the junction box'). Not coordinates."
    )

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        """Clamp instead of rejecting.

        Structured-output schemas do not enforce numeric bounds, so the range is
        imposed here. It clamps rather than raising on purpose: a 1.02 coming
        back from a model means "very sure", and throwing away the whole
        evaluation over it would be worse than reading it that way.
        """
        return max(0.0, min(1.0, float(v)))


class Evaluation(BaseModel):
    """The model's complete output for one image. Still no policy applied."""

    image_usable: bool = Field(
        description="false when the image does not allow assessment: blurred, badly "
        "framed, subject occluded, too dark"
    )
    unusable_reason: str = Field(
        description="if image_usable is false, the reason in one sentence; otherwise an empty string"
    )
    findings: list[Finding]
    summary: str = Field(description="one sentence for the person reviewing this")


# --- What the code adds, not the model --------------------------------------


class ResolvedFinding(BaseModel):
    """A finding after the confidence policy has been applied."""

    check: str
    status: Status
    observed: str
    expected: str
    confidence: float
    locator: str
    needs_human_review: bool
    action: str


class Usage(BaseModel):
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    cost_usd: float


class Result(BaseModel):
    checkpoint_id: str
    model: str
    image: str
    image_usable: bool
    unusable_reason: str
    findings: list[ResolvedFinding]
    summary: str
    measured_inputs: dict[str, str]
    usage: Usage
