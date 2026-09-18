"""A checkpoint: one fixed vantage point, its reference image, and its checks.

The unit of work is not "an image". It is a **fixed place you photograph from,
with a reference image taken from that same place**. That constraint is what
separates this from asking a model whether a picture looks right:

  - With a reference from the same angle, the question becomes "what changed",
    which is a comparison.
  - Without one, the question is "does this look correct", which is an opinion,
    and the model will happily produce one.

Published work on anomaly detection puts the gap at roughly six points of
accuracy from a single reference image. The practical gap is larger, because
without a reference there is no shared definition of correct to argue with when
someone disputes a finding.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Check(BaseModel):
    """One thing to verify in the image."""

    id: str
    rule: str = Field(description="what correct looks like, in one sentence")
    requires_measured: str | None = Field(
        default=None,
        description=(
            "name of a measured input this check depends on. If the value is "
            "missing at evaluation time the check is reported not_assessable "
            "rather than guessed from the image."
        ),
    )


class Checkpoint(BaseModel):
    """A fixed vantage point and everything needed to evaluate an image from it."""

    id: str
    site: str = Field(description="which installation, store, site or unit this belongs to")
    vantage: str = Field(description="what this checkpoint looks at, so a person can stand there")
    reference_image: Path = Field(description="the image that defines correct, from this vantage")
    checks: list[Check]
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Reference images are written relative to the checkpoint file so a
        # checkpoint directory can be moved or copied without editing paths.
        ref = Path(data["reference_image"])
        if not ref.is_absolute():
            data["reference_image"] = (path.parent / ref).resolve()
        return cls.model_validate(data)

    def measured_names(self) -> list[str]:
        """Measured inputs this checkpoint needs, in declaration order."""
        seen: list[str] = []
        for check in self.checks:
            name = check.requires_measured
            if name and name not in seen:
                seen.append(name)
        return seen
