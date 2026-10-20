import pytest

from perception.core import registry


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_registry", {kind: {} for kind in registry.KINDS})


def test_register_and_build_passes_kwargs():
    @registry.register("tracker", "dummy")
    class DummyTracker:
        def __init__(self, track_buffer: int = 10):
            self.track_buffer = track_buffer

    tracker = registry.build("tracker", {"type": "dummy", "track_buffer": 30})

    assert isinstance(tracker, DummyTracker)
    assert tracker.track_buffer == 30


def test_build_does_not_mutate_config():
    @registry.register("tracker", "dummy")
    class DummyTracker:
        pass

    cfg = {"type": "dummy"}
    registry.build("tracker", cfg)
    assert cfg == {"type": "dummy"}


def test_unknown_name_lists_available():
    @registry.register("tracker", "bytetrack")
    class ByteTrack:
        pass

    with pytest.raises(KeyError, match="bytetrack"):
        registry.get("tracker", "deepsort")


def test_missing_type_raises():
    with pytest.raises(ValueError, match="missing 'type'"):
        registry.build("tracker", {"track_buffer": 30})


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown kind"):
        registry.register("teleporter", "x")


def test_duplicate_name_raises():
    @registry.register("detector", "yolo")
    class A:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @registry.register("detector", "yolo")
        class B:
            pass
