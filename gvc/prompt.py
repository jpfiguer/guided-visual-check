"""Building the request, and the block order that makes caching work.

Two things live here and they are easy to confuse. The **doctrine** is what the
model is told about how to behave. The **block order** is a cost decision.

On order: the stable material goes first — the reference image and the rules for
this checkpoint, which do not change between visits — and the volatile material
last, the image being evaluated. Cache breakpoints sit at the end of the stable
part.

The breakpoint behind the reference image, and not only after the rules, is
there for a reason that only shows up at scale: once a single image is evaluated
across several calls (one per group of checks, to keep each call small), the
rules differ between those calls. A cache that only broke after the rules would
never hit, and the reference image — the expensive block — would be paid for in
full every time. Breaking behind it means the six or eight calls for one image
read it from cache.
"""

from __future__ import annotations

from .checkpoint import Checkpoint
from .image import PreparedImage, image_block

DOCTRINE = """\
You compare an image taken in the field against a reference image that defines \
what correct looks like from the same vantage point, and you report differences.

Working rules, in order of importance:

1. YOU REPORT EVIDENCE, NOT VERDICTS. You describe what is observed and how \
confident you are. You do not decide what happens as a result; the system \
resolves that afterwards using your confidence.

2. A FALSE POSITIVE COSTS MORE THAN A FALSE NEGATIVE. Reporting a failure that \
is not there destroys trust in the system. When in doubt, use status \
"not_assessable" and lower the confidence. Do not invent failures to seem useful.

3. DO NOT ESTIMATE WHAT CANNOT BE MEASURED FROM AN IMAGE. In particular:
   - Orientation, angles, headings, which way something faces.
   - Exact counts of many small or adjacent objects.
   - Distances, dimensions, alignment in degrees.
   If a check depends on one of those and it is not given to you under MEASURED \
INPUTS, answer "not_assessable" with low confidence and say the value is \
missing. Never derive it from the image.

4. MEASURED INPUTS ARE FACTS. They come from sensors or specialised models. You \
do not question or recompute them by looking at the image; you use them to \
compare against the rule.

5. JUDGE THE IMAGE FIRST, THE SUBJECT SECOND. If the image is blurred, badly \
framed, dark, obstructed, or the subject is occluded, set image_usable to false \
and report no findings. A bad image is a capture problem, not a failure of the \
thing being inspected.

6. LOCATION IN WORDS, NOT COORDINATES. Describe where to look so that a person \
standing in front of the subject can find it.

7. EVALUATE EXACTLY THE CHECKS ON THE LIST, one by one, using the same id. No \
more, no fewer. Real differences that match no check belong in the summary, not \
as a finding."""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_usable": {
            "type": "boolean",
            "description": "false when the image does not allow assessment",
        },
        "unusable_reason": {
            "type": "string",
            "description": "reason in one sentence when image_usable is false, otherwise empty",
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "check": {"type": "string", "description": "check id, exactly as given"},
                    "status": {"type": "string", "enum": ["pass", "fail", "not_assessable"]},
                    "observed": {"type": "string", "description": "what is visible, in one sentence"},
                    "expected": {"type": "string", "description": "what the rule asks for"},
                    "confidence": {"type": "number", "description": "0 to 1"},
                    "locator": {"type": "string", "description": "where to look, in words"},
                },
                "required": ["check", "status", "observed", "expected", "confidence", "locator"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string", "description": "one sentence for the reviewer"},
    },
    "required": ["image_usable", "unusable_reason", "findings", "summary"],
    "additionalProperties": False,
}


def rules_text(checkpoint: Checkpoint) -> str:
    lines = [
        "RULES FOR THIS CHECKPOINT",
        "",
        f"Site: {checkpoint.site}",
        f"Checkpoint: {checkpoint.id}",
        f"Vantage: {checkpoint.vantage}",
    ]
    if checkpoint.notes:
        lines.append(f"Notes: {checkpoint.notes}")
    lines += ["", "CHECKS TO EVALUATE", ""]
    for check in checkpoint.checks:
        lines.append(f"- id: {check.id}")
        lines.append(f"  rule: {check.rule}")
        if check.requires_measured:
            lines.append(
                f"  DEPENDS ON MEASURED INPUT '{check.requires_measured}'. "
                "If it does not appear under MEASURED INPUTS, answer not_assessable."
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def measured_text(values: dict[str, str]) -> str:
    if not values:
        return (
            "MEASURED INPUTS\n\n"
            "(none for this evaluation — any check depending on a measured input "
            "must be reported not_assessable)"
        )
    lines = [
        "MEASURED INPUTS",
        "",
        "These come from sensors or specialised models. They are facts:",
        "",
    ]
    # Stable order on purpose: varying it between calls would break the cache
    # for no reason at all.
    for key in sorted(values):
        lines.append(f"- {key}: {values[key]}")
    return "\n".join(lines)


def build_message(
    checkpoint: Checkpoint,
    reference: PreparedImage,
    subject: PreparedImage,
    measured: dict[str, str],
) -> list[dict]:
    """The user message content, ordered so the expensive prefix can be cached."""
    return [
        {"type": "text", "text": "REFERENCE IMAGE — what correct looks like from this vantage:"},
        {**image_block(reference), "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": rules_text(checkpoint), "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": measured_text(measured)},
        {"type": "text", "text": "SUBJECT IMAGE — the one to evaluate:"},
        image_block(subject),
        {
            "type": "text",
            "text": "Evaluate the subject image against the reference and the rules. "
            "One finding per check on the list, using the same id.",
        },
    ]
