"""guided-visual-check — reference-guided visual inspection where the model
reports evidence and the code decides.

See README.md for the four design decisions this package exists to demonstrate.
"""

from .checkpoint import Check, Checkpoint
from .evaluator import Evaluator
from .measured import MeasuredInputs
from .policy import Policy
from .schema import Evaluation, Finding, ResolvedFinding, Result, Status

__all__ = [
    "Check",
    "Checkpoint",
    "Evaluation",
    "Evaluator",
    "Finding",
    "MeasuredInputs",
    "Policy",
    "ResolvedFinding",
    "Result",
    "Status",
]
