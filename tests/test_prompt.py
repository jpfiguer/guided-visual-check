"""Request construction: block order, cache breakpoints and prompt text."""

from pathlib import Path

from gvc.checkpoint import Check, Checkpoint
from gvc.image import prepare
from gvc.prompt import build_message, measured_text, rules_text

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "images"

CHECKPOINT = Checkpoint(
    id="cp1", site="site", vantage="v", reference_image=EXAMPLES / "reference.jpg",
    checks=[
        Check(id="tilt", rule="25 degrees", requires_measured="tilt_angle_deg"),
        Check(id="soiling", rule="clean"),
    ],
)


def message():
    ref = prepare(EXAMPLES / "reference.jpg")
    sub = prepare(EXAMPLES / "subject.jpg")
    return build_message(CHECKPOINT, ref, sub, {"tilt_angle_deg": "51.0"})


def test_reference_image_precedes_the_last_cache_breakpoint():
    """The reference image sits inside the cached prefix."""
    blocks = message()
    ref_index = next(i for i, b in enumerate(blocks) if b["type"] == "image")
    last_cached = max(i for i, b in enumerate(blocks) if "cache_control" in b)
    assert ref_index <= last_cached


def test_the_reference_image_block_carries_its_own_breakpoint():
    """The reference image has its own breakpoint, besides the one on the rules."""
    blocks = message()
    reference = next(b for b in blocks if b["type"] == "image")
    assert reference.get("cache_control") == {"type": "ephemeral"}


def test_subject_image_comes_after_every_breakpoint():
    """The volatile part must never sit inside the cached prefix."""
    blocks = message()
    subject_index = max(i for i, b in enumerate(blocks) if b["type"] == "image")
    last_cached = max(i for i, b in enumerate(blocks) if "cache_control" in b)
    assert subject_index > last_cached


def test_measured_inputs_are_listed_in_sorted_order():
    a = measured_text({"b": "2", "a": "1"})
    b = measured_text({"a": "1", "b": "2"})
    assert a == b


def test_dependency_is_announced_in_the_rules():
    text = rules_text(CHECKPOINT)
    assert "tilt_angle_deg" in text
    assert "not_assessable" in text


def test_absent_measurements_say_so_explicitly():
    assert "not_assessable" in measured_text({})
