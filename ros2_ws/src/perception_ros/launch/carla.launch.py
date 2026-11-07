"""Closed loop in CARLA: carla_bridge_node -> perception_node -> decision_node -> back to the bridge.

    ros2 launch perception_ros carla.launch.py scenario:=configs/scenarios/stopped_car.yaml \\
        out_dir:=runs/closed_loop/stopped_car
    ros2 launch perception_ros carla.launch.py scenario:=configs/scenarios/stopped_car.yaml \\
        overrides:="policy.type=noop" out_dir:=runs/closed_loop/stopped_car_no_policy

The CARLA server must already run in the scenario's town. Run from the repo root with ROS, the workspace and
the Python environment sourced (as offline.launch.py). The launch ends when the scenario does: the bridge
exits and everything else is shut down. Topics: /camera/image_raw, /camera/camera_info, /ego/speed,
/perception/objects, /perception/lanes, /perception/debug_image, /perception/control.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CAMERA_TOPICS = [("image", "/camera/image_raw"), ("camera_info", "/camera/camera_info")]
VENV_PYTHON = "python3"  # first on PATH: the environment's, when it is active

ARGUMENTS = {
    "scenario": ("configs/scenarios/stopped_car.yaml", "scenario YAML"),
    "config": ("configs/pipeline/carla.yaml", "pipeline YAML (also holds the decision policy)"),
    "overrides": ("", "space-separated key=value pipeline overrides, e.g. policy.type=noop"),
    "out_dir": ("", "summary.json, ticks.csv and overlay frames go here"),
    "host": ("localhost", "CARLA server"),
    "port": ("2000", "CARLA RPC port"),
    "carla_timeout_s": ("60.0", "CARLA client timeout; raise it for CPU rendering"),
    "answer_timeout_s": ("30.0", "per-frame wait for the stack's command"),
}


def nodes(context):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    overrides = arg("overrides").split()
    pipeline_params = (
        {"config": arg("config"), "overrides": overrides} if overrides else {"config": arg("config")}
    )
    bridge = Node(
        package="perception_ros",
        executable="carla_bridge_node",
        prefix=VENV_PYTHON,
        parameters=[
            {
                "scenario": arg("scenario"),
                "out_dir": arg("out_dir"),
                "host": arg("host"),
                "port": int(arg("port")),
                "carla_timeout_s": float(arg("carla_timeout_s")),
                "answer_timeout_s": float(arg("answer_timeout_s")),
            }
        ],
        remappings=[
            *CAMERA_TOPICS,
            ("ego_speed", "/ego/speed"),
            ("control", "/perception/control"),
            ("debug_image", "/perception/debug_image"),
            ("objects", "/perception/objects"),
        ],
        output="screen",
    )
    return [
        bridge,
        Node(
            package="perception_ros",
            executable="perception_node",
            prefix=VENV_PYTHON,
            namespace="perception",
            # camera mount of perception.sim.carla_sim (1.6 m, pitch 0)
            parameters=[{**pipeline_params, "camera_height_m": 1.6, "camera_pitch_deg": 0.0}],
            remappings=CAMERA_TOPICS,
            output="screen",
        ),
        Node(
            package="perception_ros",
            executable="decision_node",
            prefix=VENV_PYTHON,
            namespace="perception",
            parameters=[pipeline_params],
            remappings=[("ego_speed", "/ego/speed")],
            output="screen",
        ),
        RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=[EmitEvent(event=Shutdown())])),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            # Fast DDS hangs under WSL2 mirrored networking
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
            *(
                DeclareLaunchArgument(name, default_value=d, description=h)
                for name, (d, h) in ARGUMENTS.items()
            ),
            OpaqueFunction(function=nodes),
        ]
    )
