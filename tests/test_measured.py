"""The missing-measurement guarantee, tested from both ends.

The prompt asks the model to report not_assessable when a measured input is
missing. These tests cover the case where it does not comply, which is the case
that matters.
"""

from pathlib import Path

from gvc.checkpoint import Check, Checkpoint
from gvc.evaluator import _enforce_measured
from gvc.measured import MeasuredInputs
from gvc.schema import Finding, Status

CHECKPOINT = Checkpoint(
    id="cp1",
    site="site",
    vantage="v",
    reference_image=Path("ref.jpg"),
    checks=[
        Check(id="tilt", rule="25 degrees", requires_measured="tilt_angle_deg"),
        Check(id="soiling", rule="clean"),
    ],
)


def finding(check, status=Status.pass_, confidence=0.98):
    return Finding(
        check=check, status=status, observed="looks fine", expected="e",
        confidence=confidence, locator="l",
    )


def test_a_confident_answer_about_a_missing_measurement_is_overridden():
    """The failure mode this exists to stop: the model answers about an angle it
    was never given, confidently, and nothing in the API objects."""
    out = _enforce_measured(CHECKPOINT, [finding("tilt")], measured={})
    assert out[0].status is Status.not_assessable
    assert out[0].confidence == 0.0
    assert "tilt_angle_deg" in out[0].observed


def test_checks_without_a_dependency_are_left_alone():
    out = _enforce_measured(CHECKPOINT, [finding("soiling")], measured={})
    assert out[0].status is Status.pass_
    assert out[0].confidence == 0.98


def test_when_the_measurement_arrives_the_answer_stands():
    out = _enforce_measured(CHECKPOINT, [finding("tilt")], measured={"tilt_angle_deg": "24.6"})
    assert out[0].status is Status.pass_


def test_measured_names_are_deduplicated_and_ordered():
    cp = Checkpoint(
        id="cp", site="s", vantage="v", reference_image=Path("r.jpg"),
        checks=[
            Check(id="a", rule="r", requires_measured="x"),
            Check(id="b", rule="r", requires_measured="y"),
            Check(id="c", rule="r", requires_measured="x"),
        ],
    )
    assert cp.measured_names() == ["x", "y"]


def test_a_measurer_returning_none_is_not_an_error():
    """An inclinometer that cannot get a reading is a normal Tuesday. It must
    produce a gap, which becomes not_assessable, not an exception."""
    registry = MeasuredInputs()
    registry.register("tilt_angle_deg", lambda _: None)
    values = registry.collect(["tilt_angle_deg"], Path("x.jpg"))
    assert values == {}
    assert registry.missing(["tilt_angle_deg"], values) == ["tilt_angle_deg"]


def test_an_unregistered_measurer_is_skipped_not_raised():
    registry = MeasuredInputs()
    assert registry.collect(["nothing_registered"], Path("x.jpg")) == {}
