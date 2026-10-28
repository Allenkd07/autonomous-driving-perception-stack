"""Fine-tune a YOLO detector with Ultralytics, logging to Weights & Biases.

    python scripts/train_detector.py --config configs/train/yolo11s_bdd100k.yaml
    python scripts/train_detector.py epochs=1 fraction=0.02 name=smoke --no-wandb
    python scripts/train_detector.py --resume runs/train/yolo11s_bdd100k/weights/last.pt

Any config key can be overridden as key=value. Each epoch's losses and val metrics go to W&B;
at the end the best weights are copied to weights/<run name>.pt and logged as a model artifact.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from omegaconf import OmegaConf
from ultralytics import YOLO


def log_epoch(run):
    """Ultralytics callback that mirrors each epoch's losses and val metrics to a W&B run."""

    def on_fit_epoch_end(trainer) -> None:
        losses = trainer.label_loss_items(trainer.tloss, prefix="train")
        run.log({**losses, **trainer.metrics, **trainer.lr, "epoch": trainer.epoch + 1})

    return on_fit_epoch_end


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train/yolo11s_bdd100k.yaml"))
    parser.add_argument("--resume", type=Path, help="last.pt of an interrupted run")
    parser.add_argument("--no-wandb", action="store_true")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    train_args = OmegaConf.to_container(cfg, resolve=True)
    weights = train_args.pop("weights")
    dataset_artifact = train_args.pop("dataset_artifact", None)
    # Relative paths get nested under runs/detect/ by Ultralytics
    train_args["project"] = str(Path(train_args["project"]).resolve())

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project="perception-stack",
            job_type="train",
            name=train_args["name"],
            config={"weights": weights, "dataset_artifact": dataset_artifact, **train_args},
            tags=["phase2", "detector"],
        )
        if dataset_artifact:
            run.use_artifact(dataset_artifact)

    model = YOLO(args.resume or weights)
    if run is not None:
        model.add_callback("on_fit_epoch_end", log_epoch(run))
    if args.resume:
        model.train(resume=True)
    else:
        model.train(**train_args)

    trainer = model.trainer
    best = Path(trainer.best)
    if best.exists():
        out = Path("weights") / f"{trainer.args.name}.pt"
        shutil.copy(best, out)
        print(f"best weights: {best} -> {out}")
        if run is not None:
            import wandb

            artifact = wandb.Artifact(trainer.args.name, type="detector", metadata=dict(trainer.metrics))
            artifact.add_file(str(best))
            run.log_artifact(artifact)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
