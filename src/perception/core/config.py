"""Typed pipeline config: YAML file + command-line overrides, validated by OmegaConf.

cfg = load_config("configs/pipeline/default.yaml", ["tracker.type=deepsort"])
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class PipelineConfig:
    # Each stage is {"type": <registered name>, **constructor kwargs}. None disables the stage.
    detector: dict[str, Any] = field(default_factory=lambda: {"type": "yolo"})
    tracker: dict[str, Any] = field(default_factory=lambda: {"type": "bytetrack"})
    lanes: dict[str, Any] | None = None
    depth: dict[str, Any] | None = None
    distance: dict[str, Any] = field(default_factory=lambda: {"type": "ground_plane"})
    policy: dict[str, Any] = field(default_factory=lambda: {"type": "noop"})
    depth_every_n: int = 1  # run depth every N frames, reuse the last map in between


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> PipelineConfig:
    """Merge defaults <- YAML file <- `key=value` overrides, and validate types."""
    cfg = OmegaConf.structured(PipelineConfig)
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    _validate(cfg)
    return OmegaConf.to_object(cfg)


def to_dict(cfg: PipelineConfig) -> dict:
    """Plain dict, e.g. for `wandb.init(config=to_dict(cfg))`."""
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)


def _validate(cfg) -> None:
    for stage in ("detector", "tracker", "lanes", "depth", "distance", "policy"):
        stage_cfg = cfg[stage]
        if stage_cfg is not None and "type" not in stage_cfg:
            raise ValueError(f"Config for stage {stage!r} is missing 'type'")
    if cfg.depth_every_n < 1:
        raise ValueError("depth_every_n must be >= 1")
