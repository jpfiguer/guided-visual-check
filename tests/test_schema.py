from gvc.prompt import OUTPUT_SCHEMA
from gvc.schema import Finding, Status


def build(confidence):
    return Finding(
        check="c", status=Status.pass_, observed="o", expected="e",
        confidence=confidence, locator="l",
    )


def test_confidence_is_clamped_not_rejected():
    """Structured outputs do not support numeric bounds, so a value such as 1.02
    can come back. It is read as 1.0 instead of failing the evaluation."""
    assert build(1.02).confidence == 1.0
    assert build(-0.3).confidence == 0.0


def test_confidence_in_range_is_untouched():
    assert build(0.73).confidence == 0.73


def test_status_values_are_the_wire_format():
    """The same strings are the status enum in OUTPUT_SCHEMA; the two change together."""
    values = {s.value for s in Status}
    assert values == {"pass", "fail", "not_assessable"}
    finding_schema = OUTPUT_SCHEMA["properties"]["findings"]["items"]
    assert set(finding_schema["properties"]["status"]["enum"]) == values
