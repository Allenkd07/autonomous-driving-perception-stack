# CLAUDE.md

Camera perception stack (detection, tracking, lanes, depth, distance) that runs offline on recorded frames and live in CARLA through ROS2. Architecture and design trade-offs: [docs/DESIGN.md](docs/DESIGN.md).

## Environment

- CARLA 0.9.15 server runs natively on Windows; everything else (perception, ROS2 Humble, training, TensorRT) runs in WSL2 Ubuntu 22.04 and reaches CARLA on `localhost:2000` through mirrored networking.
- Keep the working copy on the Linux filesystem (`~/...`), not `/mnt/c`: git, file listing and writes there are 100× slower or worse. A Windows clone is only needed for Windows-side CARLA scripts and is refreshed with `git pull`.
- ROS2 uses Cyclone DDS (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`); the default Fast DDS hangs under WSL2 mirrored networking.
- Target GPU is an RTX 3050 with 6 GB VRAM: stick to small models (YOLO n/s, MiDaS Small), TensorRT FP16, and CARLA at `-quality-level=Low`.

## Commands (from the repo root in WSL)

```bash
source .venv/bin/activate
pytest                       # unit tests; markers: gpu (needs CUDA), carla (needs a running server)
pytest -m "not gpu and not carla"
ruff check . && ruff format --check .
pre-commit run --all-files   # hooks only check staged files on commit

# ROS2 build: ROS sourced, venv NOT active
cd ros2_ws && colcon build --symlink-install && source install/setup.bash

# ROS-side tests (message conversions): ROS and the workspace sourced, then the venv
pytest ros2_ws/src/perception_ros/test
```

The `.venv` in WSL is created with `--system-site-packages` so it sees ROS2's `rclpy`. Side effects:

- pip sees Ubuntu's `keyring` and hangs on D-Bus. After creating the venv: `pip config --site set global.keyring-provider disabled`.
- Ubuntu's old system pytest shadows the venv one unless `pytest>=8` is installed in the venv (it is pinned in `requirements.txt`).
- With ROS sourced, Humble's `launch_testing` pytest plugins (written for pytest 6) break pytest 8; `pyproject.toml` disables them.

Calling WSL from Windows: set `MSYS_NO_PATHCONV=1` in Git Bash so Linux paths aren't rewritten, and put anything with `$var` or `$(...)` in a script file, since `wsl.exe` expands it before `bash -c` sees it.

## Code conventions

- src layout: the package lives in `src/perception/` and is only importable after `pip install -e .` (done by `requirements.txt`).
- Core logic stays in `src/perception/`, independent of ROS, CARLA and W&B. ROS nodes in `ros2_ws/src/perception_ros/` are thin adapters; one `perception_node` runs all model stages in-process to share a single CUDA context.
- Stages implement the interfaces in `perception/core/interfaces.py`, register themselves in the registry, and are chosen by config. Data passed between stages uses the dataclasses in `perception/core/types.py`.
- Config is OmegaConf + dataclasses (`configs/`), overridable from the CLI (`tracker.type=deepsort`). No Hydra.
- YOLO goes through Ultralytics behind the `Detector` interface; MiDaS and lane segmentation use our own `InferenceBackend` (torch / onnx / tensorrt).
- Ruff, line length 110, target py310. LF line endings everywhere (see `.gitattributes`).
- Datasets, recordings, model weights and `wandb/` are not committed.
