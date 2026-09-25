"""Reference-guided visual inspection: the model reports evidence, the code decides.

README.md describes the design.
"""

from .checkpoint import Check, Checkpoint
from .evaluator import EvaluationError, Evaluator
from .measured import MeasuredInputs
from .policy import Policy
from .schema import Evaluation, Finding, ResolvedFinding, Result, Status

__all__ = [
    "Check",
    "Checkpoint",
    "Evaluation",
    "EvaluationError",
    "Evaluator",
    "Finding",
    "MeasuredInputs",
    "Policy",
    "ResolvedFinding",
    "Result",
    "Status",
]
