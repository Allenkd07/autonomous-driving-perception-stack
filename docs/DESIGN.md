# Design

## 1. Goals that drive the design

1. **Swap any component without touching the rest.** The project compares ByteTrack vs DeepSORT, classical vs segmentation lanes, MiDaS vs Depth Anything, and PyTorch vs ONNX vs TensorRT. Each comparison should be a config change, not a code change.
2. **Same pipeline everywhere.** The exact code that runs on a dataset or video offline also runs inside CARLA and behind ROS2. The numbers in the report then describe the system in the demo.
3. **Testable without the simulator.** All logic can be tested and benchmarked with no CARLA, no ROS and (for most tests) no GPU.
4. **Fit in 6 GB VRAM.** As few CUDA processes as possible, small models, FP16.

## 2. Pattern: ports and adapters around a stage pipeline

```text
                ┌──────────────────────── perception (pure Python library) ────────────────────────┐
 FrameSource ──►│  Detector ─► Tracker ─┐                                                           │
 (adapter)      │  LaneDetector ────────┼─► DistanceEstimator ─► (PerceptionOutput) ─► DecisionPolicy│──► Sink (adapter)
                │  DepthEstimator ──────┘                                                           │
                │                     each stage = interface + interchangeable implementations      │
                │                     models run on InferenceBackend: torch | onnx | tensorrt       │
                └────────────────────────────────────────────────────────────────────────────────────┘
 Sources: video file, image folder, dataset, CARLA client, ROS2 subscriber
 Sinks:   video writer, on-screen display, W&B logger, ROS2 publisher, CARLA vehicle control
```

- **Core (`src/perception/`)** knows nothing about ROS, CARLA or W&B. It only sees `Frame` in and `PerceptionOutput` out.
- **Stages** are defined by small interfaces (Python `Protocol`s). Each one has several implementations (Strategy pattern).
- **A registry + typed config (OmegaConf + dataclasses)** chooses which implementation runs (Factory pattern). Example: `tracker: {type: bytetrack}` → `tracker: {type: deepsort}`, or from the command line: `tracker.type=deepsort`.
- **Adapters** at the edges turn the outside world into core types and back. ROS2 nodes are thin adapters: message ↔ dataclass conversion plus a call into the pipeline.

## 3. Data contracts (`perception/core/types.py`)

Every stage communicates only through these types. Conventions: images are **BGR uint8 HxWx3** (OpenCV native); boxes are **xyxy in pixels (float)**; distances in **metres**; time in **seconds**; camera frame: x right, y down, z forward.

```python
@dataclass(frozen=True)
class CameraInfo:
    width: int; height: int
    K: np.ndarray            # 3x3 intrinsics
    height_m: float          # mounting height above ground (for ground-plane distance)
    pitch_rad: float = 0.0

@dataclass(frozen=True)
class Frame:
    image: np.ndarray        # BGR uint8 HxWx3
    timestamp: float         # seconds (sim time in CARLA)
    frame_id: int
    camera: CameraInfo
    depth_gt: np.ndarray | None = None   # metres; CARLA only, for evaluation

@dataclass
class Detection:
    box: np.ndarray          # xyxy
    score: float
    class_id: int
    class_name: str

@dataclass
class Track(Detection):
    track_id: int
    age: int                 # frames since first seen

@dataclass
class LaneResult:
    left: np.ndarray | None  # polynomial coeffs x = f(y) in image space
    right: np.ndarray | None
    ego_lane_mask: np.ndarray | None   # HxW bool
    confidence: float

@dataclass
class DepthMap:
    depth: np.ndarray        # HxW float32
    is_metric: bool          # False = relative (raw MiDaS), True = metres

@dataclass
class ObjectState:
    track: Track
    distance_m: float | None
    ttc_s: float | None      # time to collision, None if not approaching
    in_ego_lane: bool

@dataclass
class PerceptionOutput:
    frame: Frame
    objects: list[ObjectState]
    lanes: LaneResult | None
    depth: DepthMap | None
    timings_ms: dict[str, float]   # per stage, for benchmarking

@dataclass
class ControlCommand:
    throttle: float; brake: float; steer: float
    reason: str              # e.g. "EMERGENCY_BRAKE track 12 ttc=1.3s"
```

## 4. Stage interfaces (`perception/core/interfaces.py`)

```python
class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...

class Tracker(Protocol):
    def update(self, detections: list[Detection], frame: Frame) -> list[Track]: ...
    def reset(self) -> None: ...

class LaneDetector(Protocol):
    def detect(self, frame: Frame) -> LaneResult: ...

class DepthEstimator(Protocol):
    def estimate(self, frame: Frame) -> DepthMap: ...

class DistanceEstimator(Protocol):   # fusion: tracks + depth + lanes -> distance, TTC, in-lane
    def estimate(self, tracks: list[Track], depth: DepthMap | None,
                 lanes: LaneResult | None, frame: Frame) -> list[ObjectState]: ...

class DecisionPolicy(Protocol):
    def decide(self, output: PerceptionOutput, ego_speed_mps: float) -> ControlCommand: ...

class FrameSource(Protocol):          # adapters
    def __iter__(self) -> Iterator[Frame]: ...

class Sink(Protocol):
    def write(self, output: PerceptionOutput) -> None: ...
    def close(self) -> None: ...
```

Stateful stages (tracker, distance estimator with TTC history, lane temporal smoothing) own their state and expose `reset()`.

### Planned implementations

| Interface | Implementations |
| --- | --- |
| Detector | `YoloDetector` (weights: `.pt` / `.onnx` / `.engine`) |
| Tracker | `ByteTrackTracker`, `DeepSortTracker` |
| LaneDetector | `ClassicalLaneDetector` (OpenCV), `SegLaneDetector` (TwinLiteNet / YOLOP) |
| DepthEstimator | `MidasDepth`, `DepthAnythingDepth`, `GroundTruthDepth` (CARLA, for upper-bound experiments) |
| DistanceEstimator | `DepthScaleDistance` (calibrated MiDaS), `GroundPlaneDistance` (geometry), `FusedDistance` |
| DecisionPolicy | `EmergencyBrakePolicy` (in-lane + TTC/distance thresholds), `NoOpPolicy` |
| FrameSource | `VideoSource`, `ImageFolderSource`, `CarlaSource`, (ROS subscriber lives in the ROS adapter) |
| Sink | `VideoSink`, `DisplaySink`, `WandbSink`, (ROS publisher lives in the ROS adapter) |

## 5. Inference backends (`src/perception/backends/`)

Models are split into **pre-process → backend.run() → post-process**. Only the backend changes between PyTorch, ONNX Runtime and TensorRT. Pre- and post-processing stay identical, so latency and accuracy comparisons are apples-to-apples.

```python
class InferenceBackend(Protocol):
    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
    # TorchBackend, OnnxRuntimeBackend, TensorRTBackend
```

YOLO is the exception: Ultralytics already loads `.pt`, `.onnx` and `.engine` behind one API, so `YoloDetector` wraps Ultralytics behind our `Detector` interface. The rest of the system never imports Ultralytics. A custom TensorRT YOLO could be added as a second `Detector` implementation without changing anything else.

## 6. Pipeline, config and registry

```python
@register("tracker", "bytetrack")
class ByteTrackTracker: ...

pipeline = PerceptionPipeline.from_config("configs/pipeline/default.yaml")
for frame in source:
    out = pipeline.process(frame)        # runs stages, records timings_ms
    cmd = policy.decide(out, ego_speed)
    for s in sinks: s.write(out)
```

```yaml
# configs/pipeline/default.yaml
detector:  {type: yolo, weights: weights/yolo11s_bdd.engine, conf: 0.35, classes: [car, truck, bus, person, rider, bicycle, motorcycle]}
tracker:   {type: bytetrack, track_buffer: 30}
lanes:     {type: classical}
depth:     {type: midas, model: MiDaS_small, backend: tensorrt}
distance:  {type: fused}
policy:    {type: emergency_brake, ttc_threshold_s: 2.0, min_distance_m: 6.0}
```

- Every experiment = a config file plus optional command-line overrides (`python scripts/run_pipeline.py --config configs/pipeline/default.yaml tracker.type=deepsort depth.backend=onnx`). Configs load through OmegaConf into typed dataclasses (validated at startup); no Hydra. The resolved config is logged to W&B with each run, so every number in the report can be traced back to its config.
- Per-stage timing is built into `PerceptionPipeline` (a context-manager timer), so benchmarking needs no separate instrumentation.
- Depth runs every N frames (configurable). It is the most expensive stage, and objects' distances change slowly.

## 7. Runtime topology (CARLA + ROS2)

```text
Windows                                  WSL2 Ubuntu 22.04 (ROS2 Humble)
┌──────────────┐  carla API   ┌──────────────────┐ /camera/image_raw  ┌──────────────────────┐
│ CARLA server │◄────────────►│ carla_bridge_node │──────────────────►│ perception_node      │
│ (sync mode,  │  tick +      │ spawns ego, cams, │ /camera/depth_gt  │ runs PerceptionPipeline│
│  20 Hz)      │  control     │ scenarios, ticks  │ /ego/state        │ in-process (1 CUDA ctx)│
└──────────────┘              │ applies control   │◄──┐               └──────────┬───────────┘
                              └──────────────────┘   │ /vehicle/control          │ /perception/objects
                                                      │                           │ /perception/lanes
                                                ┌─────┴───────────┐               │ /perception/debug_image
                                                │ decision_node    │◄──────────────┘
                                                │ EmergencyBrake   │        ┌──────────────┐
                                                └─────────────────┘        │ viz / rviz2 /  │
                                                                            │ rosbag record  │
                                                                            └──────────────┘
```

- **One perception node** runs all model stages in-process. Each separate CUDA process costs about 300–500 MB of VRAM just for its context, and image hand-offs between Python nodes add copies. Stages stay separate in code, just not in processes.
- **CARLA synchronous mode at 20 Hz.** The simulation waits for the pipeline, so slow frames don't distort the physics and every scenario is reproducible run to run.
- `perception_msgs`: `TrackedObject`, `TrackedObjectArray`, `LaneInfo`, `ControlCommand` (or `ackermann_msgs`).
- `perception_ros/conversions.py` is the only place that knows both ROS messages and core types.

## 8. Repository layout

```text
src/perception/   the importable package (src layout: only importable once installed)
    core/        types.py, interfaces.py, registry.py, config.py, timing.py
    backends/    torch_backend.py, onnx_backend.py, trt_backend.py
    detection/   tracking/   lanes/   depth/   fusion/   planning/
    io/          sources.py, sinks.py
    viz/         drawing.py  (overlay boxes, IDs, lanes, distances)
    eval/        metrics for detection, tracking, lanes, depth
    pipeline.py
scripts/       CLI entry points: run_pipeline.py, train_yolo.py, eval_*.py, export_onnx.py, build_trt.py, benchmark.py
ros2_ws/src/
  perception_msgs/   msg definitions (ament_cmake)
  perception_ros/    bridge, perception, decision nodes + conversions + launch files (ament_python)
carla_scripts/ scenarios (pedestrian crossing, stopped car, cut-in), data recording
configs/       pipeline/, train/, scenarios/
notebooks/     exploration only; reusable code moves into src/perception/
tests/         unit/, contract/, integration/
docs/
pyproject.toml   `pip install -e .` makes `perception` importable from scripts, notebooks, tests and ROS nodes
```

## 9. Testing strategy

| Level | What | Needs |
| --- | --- | --- |
| Unit | lane polynomial fitting, depth scale fit, TTC maths, ground-plane distance, conversions, registry/config | CPU only, synthetic arrays |
| Contract | One parametrized test suite per interface, run against **every** implementation (output types, shapes, empty inputs, `reset()`) | CPU (GPU-only implementations skipped when no CUDA is available) |
| Golden | Pipeline on ~10 saved frames → outputs match stored results within tolerance | weights |
| Integration | Launch file in CARLA, scripted pedestrian scenario → ego stops before collision | CARLA + ROS2 (manual / pre-demo) |

## 10. Engineering conventions

- Python 3.10, type hints everywhere, `ruff` for lint + format, `pre-commit` hooks.
- Logging via the standard `logging` module (no prints in library code).
- W&B: one project; `group` = phase (`detection`, `tracking`, `lanes`, `depth`, `optim`, `carla`); the config file is logged with every run.
- Git: `main` stays working; one branch per phase/feature (`feat/detection`, `feat/lanes`, ...); conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`); squash-free merges so the history shows the work.

## 11. Design trade-offs

| Choice | Why | Alternative not taken |
| --- | --- | --- |
| One `perception_node` running all model stages in-process (plus `carla_bridge_node` and `decision_node`) | Each CUDA process costs ~300–500 MB of VRAM; one process avoids that overhead on a 6 GB GPU and needs no timestamp syncing between topics. Modularity is kept in code through interfaces + registry. | One node per stage |
| Own `InferenceBackend` layer for MiDaS and the lane model; YOLO through Ultralytics, wrapped behind `Detector` | Ultralytics already runs `.pt` / `.onnx` / `.engine`, so rewriting YOLO pre/post-processing and NMS adds work without changing the benchmark results | Custom backends for every model |
| OmegaConf + dataclasses + registry for config | Typed, validated configs with command-line overrides (`tracker.type=deepsort`), so each comparison is a one-line command | Plain YAML (no overrides); Hydra (changes the working directory per run, awkward with ROS launch) |
| Perception runs in WSL2 next to ROS2; CARLA runs natively on Windows | ROS nodes import the pipeline directly; the only cross-OS link is the CARLA client's TCP connection on `localhost`. TensorRT engines are built in WSL because they are tied to the platform they are built on. | Models on Windows with results sent across to ROS |
| src layout (`src/perception/`) | The package is only importable once installed, so tests, scripts and ROS nodes all use the installed package and packaging mistakes show up early | Package folder at the repo root |
