from perception.core.timing import StageTimer


def test_measure_records_and_accumulates():
    timer = StageTimer()
    with timer.measure("detect"):
        pass
    with timer.measure("detect"):
        pass
    with timer.measure("track"):
        pass

    assert set(timer.timings_ms) == {"detect", "track"}
    assert all(ms >= 0.0 for ms in timer.timings_ms.values())
    assert timer.total_ms == sum(timer.timings_ms.values())


def test_measure_records_even_on_exception():
    timer = StageTimer()
    try:
        with timer.measure("detect"):
            raise RuntimeError
    except RuntimeError:
        pass
    assert "detect" in timer.timings_ms
