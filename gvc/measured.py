"""Measured inputs: facts the model is told, never asked for.

This is the second load-bearing idea, and the one that is easiest to skip.

Vision-language models are strong at describing a scene and weak at a specific
family of questions: metric ones. "Which way is this object facing", "what angle
is this at", "how many degrees off vertical" — published evaluations put general
VLMs near the floor on orientation tasks while humans sit near the ceiling. The
failure is quiet. The model does not refuse; it produces a confident number.

So anything with a closed form is computed by something that computes, and
handed to the model as a given, with an explicit instruction not to re-derive it
from the image. A pose estimator, an inclinometer reading, an EXIF field, a
lookup in a database — the source does not matter, only that it is not the
language model.

The part that makes it safe is the fallback: when a declared measurement is
missing, the check that depends on it is reported `not_assessable`. It is not
silently evaluated from the picture. A gap in the data has to look like a gap.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

#: A measurer takes the image being evaluated and returns a value, or None when
#: it cannot produce one. Returning None is a legitimate outcome, not an error.
Measurer = Callable[[Path], str | None]


class MeasuredInputs:
    """Registry of deterministic measurers, keyed by the name checks refer to."""

    def __init__(self) -> None:
        self._measurers: dict[str, Measurer] = {}

    def register(self, name: str, measurer: Measurer) -> None:
        self._measurers[name] = measurer

    def collect(self, names: list[str], image: Path) -> dict[str, str]:
        """Run the measurers for `names`, skipping the ones with no value.

        Names with no registered measurer are skipped too. That is intentional:
        a checkpoint may declare a measurement this deployment cannot take, and
        the right outcome is a not_assessable check, not a crash.
        """
        values: dict[str, str] = {}
        for name in names:
            measurer = self._measurers.get(name)
            if measurer is None:
                continue
            value = measurer(image)
            if value is not None:
                values[name] = value
        return values

    def missing(self, names: list[str], values: dict[str, str]) -> list[str]:
        return [n for n in names if n not in values]
