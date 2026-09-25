from pathlib import Path

import pytest
from pydantic import ValidationError

from gvc.checkpoint import Check, Checkpoint

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_the_example_checkpoint_resolves_its_reference_image():
    checkpoint = Checkpoint.load(EXAMPLES / "solar-array-row-3.yaml")
    assert checkpoint.reference_image == EXAMPLES / "images" / "reference.jpg"
    assert checkpoint.reference_image.is_file()


def test_a_checkpoint_without_a_reference_image_fails_validation(tmp_path):
    path = tmp_path / "checkpoint.yaml"
    path.write_text(
        "id: cp\nsite: s\nvantage: v\nchecks:\n  - id: a\n    rule: r\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError, match="reference_image"):
        Checkpoint.load(path)


def test_repeated_check_ids_are_rejected():
    """Findings are matched to checks by id, so a repeated id is ambiguous."""
    with pytest.raises(ValidationError, match="repeated: a"):
        Checkpoint(
            id="cp", site="s", vantage="v", reference_image=Path("r.jpg"),
            checks=[Check(id="a", rule="r"), Check(id="b", rule="r"), Check(id="a", rule="r")],
        )
