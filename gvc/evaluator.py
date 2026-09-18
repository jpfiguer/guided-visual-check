"""The call to the model, what it costs, and the guarantees applied around it.

Everything here is arranged so that the parts that must not drift — the policy,
the missing-measurement rule — are enforced by code on the way out, not only
requested in the prompt on the way in. The doctrine asks the model to report
`not_assessable` when a measured input is missing; `_enforce_measured` makes it
true whether or not the model complied. Defence in depth, because a prompt
instruction is a request and this is a guarantee.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import Checkpoint
from .image import PreparedImage, prepare
from .measured import MeasuredInputs
from .policy import Policy
from .prompt import DOCTRINE, OUTPUT_SCHEMA, build_message
from .schema import Evaluation, Finding, Result, Status, Usage

#: Read from the environment rather than hardcoded: which model to use is a
#: decision to revisit as new labelled data arrives, and revisiting it should
#: not require a deploy. Pick it by measurement on your own images, not by
#: reputation — a cheaper model that ties on your data is the right model.
DEFAULT_MODEL = os.environ.get("GVC_MODEL", "claude-sonnet-5")

#: USD per million tokens (input, output).
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, PRICES["claude-sonnet-5"])
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def actual_cost(model: str, usage) -> float:
    """Real cost from a usage object, including the two cache rates.

    Writing to cache costs about 1.25x the input rate; reading from it about
    0.10x. Ignoring the split makes cached runs look more expensive than they
    are and hides the thing you are optimising.
    """
    price_in, price_out = PRICES.get(model, PRICES["claude-sonnet-5"])
    written = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (
        usage.input_tokens * price_in
        + written * price_in * 1.25
        + read * price_in * 0.10
        + usage.output_tokens * price_out
    ) / 1_000_000


@dataclass
class Evaluator:
    """Evaluates one image against one checkpoint."""

    policy: Policy = Policy()
    measured_inputs: MeasuredInputs | None = None
    model: str = DEFAULT_MODEL
    long_edge: int | None = None

    def dry_run(self, checkpoint: Checkpoint, image_path: str | Path) -> dict:
        """Everything except the API call: what would be sent, and what it would cost.

        Worth having for its own sake. Being able to answer "what will this run
        cost" before spending anything changes how willing people are to try
        things, and it catches a malformed checkpoint without burning a call.
        """
        reference, subject, measured, missing = self._prepare(checkpoint, image_path)
        content = build_message(checkpoint, reference, subject, measured)
        text_tokens = sum(len(b.get("text", "")) for b in content if b["type"] == "text") // 4
        input_tokens = reference.vision_tokens + subject.vision_tokens + text_tokens
        output_tokens = 120 * max(1, len(checkpoint.checks))
        return {
            "checkpoint": checkpoint.id,
            "model": self.model,
            "checks": len(checkpoint.checks),
            "vision_tokens": reference.vision_tokens + subject.vision_tokens,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_cost_usd": round(estimate_cost(self.model, input_tokens, output_tokens), 6),
            "measured_inputs": measured,
            "missing_measurements": missing,
        }

    def evaluate(self, checkpoint: Checkpoint, image_path: str | Path) -> Result:
        import anthropic

        reference, subject, measured, _ = self._prepare(checkpoint, image_path)
        content = build_message(checkpoint, reference, subject, measured)

        client = _client()
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=DOCTRINE,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        )
        raw = json.loads(_first_text(response))
        evaluation = Evaluation.model_validate(raw)
        return self._resolve(checkpoint, evaluation, subject, measured, response.usage)

    # --- internals ---------------------------------------------------------

    def _prepare(self, checkpoint: Checkpoint, image_path: str | Path):
        kwargs = {"long_edge": self.long_edge} if self.long_edge else {}
        reference = prepare(checkpoint.reference_image, **kwargs)
        subject = prepare(image_path, **kwargs)
        names = checkpoint.measured_names()
        registry = self.measured_inputs or MeasuredInputs()
        measured = registry.collect(names, Path(image_path))
        missing = registry.missing(names, measured)
        return reference, subject, measured, missing

    def _resolve(self, checkpoint, evaluation, subject, measured, usage) -> Result:
        findings = _enforce_measured(checkpoint, evaluation.findings, measured)

        # Rule 5 as a guarantee, not a request: an unusable image yields no
        # findings at all. Reporting failures from an image the model itself
        # called unusable is how a system starts blaming people for bad photos.
        resolved = [] if not evaluation.image_usable else self.policy.resolve_all(findings)

        return Result(
            checkpoint_id=checkpoint.id,
            model=self.model,
            image=subject.origin,
            image_usable=evaluation.image_usable,
            unusable_reason=evaluation.unusable_reason,
            findings=resolved,
            summary=evaluation.summary,
            measured_inputs=measured,
            usage=Usage(
                input_tokens=usage.input_tokens,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                output_tokens=usage.output_tokens,
                cost_usd=actual_cost(self.model, usage),
            ),
        )


def _enforce_measured(
    checkpoint: Checkpoint, findings: list[Finding], measured: dict[str, str]
) -> list[Finding]:
    """Force `not_assessable` on checks whose measured input never arrived.

    The prompt already asks for this. This makes it so. A model that answers
    confidently about an angle it was not given has not broken any hard rule of
    the API — it has just done the thing this whole design exists to prevent.
    """
    required = {c.id: c.requires_measured for c in checkpoint.checks}
    out: list[Finding] = []
    for finding in findings:
        needed = required.get(finding.check)
        if needed and needed not in measured:
            out.append(
                finding.model_copy(
                    update={
                        "status": Status.not_assessable,
                        "confidence": 0.0,
                        "observed": f"measured input '{needed}' was not available",
                    }
                )
            )
        else:
            out.append(finding)
    return out


_shared_client = None


def _client():
    """One client per process: build the HTTP pool once, not once per image."""
    global _shared_client
    if _shared_client is None:
        import anthropic

        headers = {}
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        if workspace:
            headers["anthropic-workspace-id"] = workspace
        _shared_client = anthropic.Anthropic(default_headers=headers or None)
    return _shared_client


def _first_text(response) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("the model returned no text block")
