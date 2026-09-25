"""Measured inputs: values computed outside the model and given to it as facts.

A check that declares `requires_measured` depends on one of these values. When
the value is missing, the evaluator reports that check as not_assessable.
README.md (decision 3) explains why these quantities are not asked of the model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

#: A measurer takes the image being evaluated and returns a value, or None when
#: it has no reading. The value reaches the model as text, through str().
Measurer = Callable[[Path], object]


class MeasuredInputs:
    """Registry of deterministic measurers, keyed by the name checks refer to."""

    def __init__(self) -> None:
        self._measurers: dict[str, Measurer] = {}

    def register(self, name: str, measurer: Measurer) -> None:
        self._measurers[name] = measurer

    def collect(self, names: list[str], image: Path) -> dict[str, str]:
        """Run the measurers for `names`, skipping the ones with no value.

        Names with no registered measurer are skipped too, so a checkpoint can
        declare a measurement this deployment cannot take; the check that needs
        it is then reported not_assessable. Values are converted with str(), so
        a measurer can return a float.
        """
        values: dict[str, str] = {}
        for name in names:
            measurer = self._measurers.get(name)
            if measurer is None:
                continue
            value = measurer(image)
            if value is not None:
                values[name] = str(value)
        return values

    def missing(self, names: list[str], values: dict[str, str]) -> list[str]:
        return [n for n in names if n not in values]
