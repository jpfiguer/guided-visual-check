"""What the model returns, and what the code adds afterwards.

`Evaluation` and `Finding` hold what the model reported: evidence, with no
decision in it. `ResolvedFinding` and `Result` add what the code decided.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Status(str, Enum):
    """The three outcomes a single check can have.

    `not_assessable` means the check could not be judged from the image and the
    measured inputs. The policy sends it to human review.
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
        """Clamp to [0, 1] instead of rejecting.

        Structured outputs do not support numeric bounds, so the range is
        applied here. Clamping keeps one out-of-range value, such as 1.02, from
        discarding the whole evaluation.
        """
        return max(0.0, min(1.0, float(v)))


class Evaluation(BaseModel):
    """The model's complete output for one image, before any rule or policy."""

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
