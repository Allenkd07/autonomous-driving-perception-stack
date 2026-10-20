from pathlib import Path

import pytest
from omegaconf.errors import ValidationError

from perception.core.config import PipelineConfig, load_config, to_dict

DEFAULT_YAML = Path(__file__).parents[2] / "configs" / "pipeline" / "default.yaml"


def test_defaults_without_file():
    cfg = load_config()
    assert isinstance(cfg, PipelineConfig)
    assert cfg.tracker["type"] == "bytetrack"
    assert cfg.depth is None


def test_default_yaml_loads():
    cfg = load_config(DEFAULT_YAML)
    assert cfg.detector["type"] == "yolo"
    assert cfg.depth["backend"] == "torch"


def test_cli_overrides_win():
    cfg = load_config(DEFAULT_YAML, ["tracker.type=deepsort", "depth.backend=tensorrt", "depth_every_n=3"])
    assert cfg.tracker["type"] == "deepsort"
    assert cfg.depth["backend"] == "tensorrt"
    assert cfg.depth_every_n == 3


def test_wrong_type_rejected():
    with pytest.raises(ValidationError):
        load_config(overrides=["depth_every_n=often"])


def test_stage_without_type_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("lanes:\n  smoothing: 5\n")
    with pytest.raises(ValueError, match="lanes"):
        load_config(bad)


def test_to_dict_is_plain():
    d = to_dict(load_config(DEFAULT_YAML))
    assert isinstance(d, dict)
    assert d["policy"]["ttc_threshold_s"] == 2.0
