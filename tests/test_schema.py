from gvc.schema import Finding, Status


def build(confidence):
    return Finding(
        check="c", status=Status.pass_, observed="o", expected="e",
        confidence=confidence, locator="l",
    )


def test_confidence_is_clamped_not_rejected():
    """Structured output cannot enforce numeric bounds, so a model can and does
    return 1.02. Losing the whole evaluation over that would be worse than
    reading it as "very sure"."""
    assert build(1.02).confidence == 1.0
    assert build(-0.3).confidence == 0.0


def test_confidence_in_range_is_untouched():
    assert build(0.73).confidence == 0.73


def test_status_values_are_the_wire_format():
    """These strings appear in the output schema sent to the model. Renaming an
    enum member without updating prompt.py would break parsing silently."""
    assert {s.value for s in Status} == {"pass", "fail", "not_assessable"}
