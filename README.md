# Autonomous Driving Perception Stack

A camera-based perception pipeline that detects vehicles, pedestrians and lanes, tracks objects across frames, estimates their distance with monocular depth, and runs live inside the CARLA simulator through ROS2. The ego car brakes when an obstacle in its lane gets too close.

## Architecture

```text
 CARLA server (Windows, sync mode)
        │  carla Python API (localhost:2000)
        ▼
 [carla_bridge_node] ──/camera/image_raw──► [perception_node]  one process, one CUDA context
        ▲                                     PerceptionPipeline:
        │                                     YOLO + ByteTrack → lanes → MiDaS → distance / TTC
        │                                          │ /perception/objects, /perception/lanes
        │ /vehicle/control                         ▼
        └───────────────────────────────── [decision_node]  emergency brake
```

All model stages run inside one ROS2 node to save GPU memory; they stay separate in code behind common interfaces and are swapped through config.

## Tech stack

YOLO (Ultralytics) · ByteTrack / DeepSORT · OpenCV · MiDaS · CARLA 0.9.15 · ROS2 Humble · ONNX · TensorRT · Weights & Biases

## Repository layout

```text
src/perception/             core library, independent of ROS / CARLA / W&B (src layout)
    core/                     data contracts (types), stage interfaces, registry, config, timing
    backends/                 inference backends: torch | onnx | tensorrt
    detection/ tracking/ lanes/ depth/ fusion/ planning/   stage implementations
    io/ viz/ eval/            frame sources & sinks, overlays, metrics
    pipeline.py               PerceptionPipeline: runs the stages for one frame
scripts/                    CLI entry points (run, train, eval, export, benchmark)
configs/                    pipeline/, train/, scenarios/ (YAML, overridable from the CLI)
ros2_ws/src/
  perception_msgs/          custom messages (TrackedObject, LaneInfo, ControlCommand)
  perception_ros/           CARLA bridge, perception and decision nodes (thin adapters)
carla_scripts/              connection test, scenarios, data recording
optimization/               ONNX export, TensorRT engine build, benchmarks
data/                       dataset download + conversion scripts (data itself is not committed)
notebooks/                  exploration only
tests/                      unit/, contract/, integration/
docs/                       design, API/topic docs, results
```

See [docs/DESIGN.md](docs/DESIGN.md) for the architecture and design decisions.

## Setup

Tested on Windows 11 + NVIDIA RTX 3050 (6 GB). CARLA runs natively on Windows and ROS2 runs in WSL2 (Ubuntu 22.04).

### 1. Python environment (Windows)

Requires Python 3.10 (the CARLA 0.9.15 client has no wheels for newer versions). Install it from [python.org](https://www.python.org/downloads/release/python-31011/); it can sit alongside other Python versions and is selected with the `py` launcher.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True
pytest
pre-commit install   # runs ruff on every commit
```

If PowerShell refuses to run the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

`requirements.txt` installs this repo as an editable package (`pip install -e .`), so `import perception` works from scripts, notebooks and ROS nodes.

Run the pipeline with a config, overriding any key from the command line (script arrives with the first stage implementations):

```bash
python scripts/run_pipeline.py --config configs/pipeline/default.yaml tracker.type=deepsort depth.backend=onnx
```

### 2. CARLA simulator (Windows)

1. Download **CARLA 0.9.15** for Windows from the [CARLA releases page](https://github.com/carla-simulator/carla/releases) and extract it (~20 GB).
2. Start the server with low settings to save VRAM:
   ```bash
   CarlaUE4.exe -quality-level=Low
   ```
3. With the server running, check the connection:
   ```bash
   python carla_scripts/test_connection.py --frames 10
   ```
   Frames are saved to `_out/test_connection/`.

### 3. ROS2 Humble (WSL2)

```powershell
wsl --install -d Ubuntu-22.04
```

So that WSL can reach the CARLA server on `localhost`, add this to `%UserProfile%\.wslconfig` and then run `wsl --shutdown`:

```ini
[wsl2]
networkingMode=mirrored
```

Inside Ubuntu, install ROS2 Humble ([official guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)), then:

```bash
pip install carla==0.9.15
pip install -e .                                            # from the repo root
python3 carla_scripts/test_connection.py --host localhost   # same test, from WSL

cd ros2_ws && colcon build && source install/setup.bash     # build the ROS2 packages
```

### 4. Weights & Biases

```bash
wandb login
```

## Datasets

- **BDD100K**: detection boxes and lane markings (detector training and evaluation, lane evaluation)
- **CARLA synthetic frames**: tracking IDs and depth ground truth (tracking and distance evaluation)
- **KITTI** (optional): additional depth/distance evaluation

See [docs/DATA.md](docs/DATA.md) for sources, processing, the class set and what the data contains.

## Results

_Coming soon: detection mAP, tracking MOTA/IDF1, lane comparison, depth error, and PyTorch vs ONNX vs TensorRT latency._
