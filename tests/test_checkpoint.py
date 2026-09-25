from pathlib import Path

import pytest
from pydantic import ValidationError

from gvc.checkpoint import Check, Checkpoint


def test_repeated_check_ids_are_rejected():
    """Findings are matched to checks by id, so a repeated id is ambiguous."""
    with pytest.raises(ValidationError, match="repeated: a"):
        Checkpoint(
            id="cp", site="s", vantage="v", reference_image=Path("r.jpg"),
            checks=[Check(id="a", rule="r"), Check(id="b", rule="r"), Check(id="a", rule="r")],
        )
