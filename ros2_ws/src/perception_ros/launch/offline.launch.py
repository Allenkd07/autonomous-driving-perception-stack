"""Perception and decision on recorded frames, without the simulator: player -> perception -> decision.

    ros2 launch perception_ros offline.launch.py input:=data/carla/town10_noon_s0 \\
        overrides:="depth=null distance.type=ground_plane"
    ros2 launch perception_ros offline.launch.py input:=clip.mp4 \\
        config:=configs/pipeline/bdd100k.yaml fov_deg:=60

Run from the repo root (config and input paths are relative to it) with ROS, the workspace and the venv
sourced. Nodes are started as `python3 <node script>`, so they run with the venv's python (torch,
ultralytics and the perception package live there) rather than the script's /usr/bin/python3 shebang.
Topics: /camera/image_raw, /camera/camera_info, /ego/speed, /perception/objects, /perception/lanes,
/perception/debug_image, /perception/control.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CAMERA_TOPICS = [("image", "/camera/image_raw"), ("camera_info", "/camera/camera_info")]
VENV_PYTHON = "python3"  # first on PATH: the venv's, when it is active

ARGUMENTS = {
    "input": ("", "CARLA recording, image folder or video"),
    "config": ("configs/pipeline/default.yaml", "pipeline YAML (also holds the decision policy)"),
    "overrides": ("", "space-separated key=value pipeline overrides"),
    "rate": ("0.0", "playback rate in Hz (0 = recorded rate)"),
    "loop": ("false", "replay the input forever"),
    "fov_deg": ("90.0", "horizontal field of view of an image folder or video"),
    "camera_height_m": ("1.6", "camera height above the road, for ground-plane distances"),
    "pitch_deg": ("0.0", "camera pitch, positive looking down"),
}


def nodes(context):
    def arg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    overrides = arg("overrides").split()
    pipeline_params = (
        {"config": arg("config"), "overrides": overrides} if overrides else {"config": arg("config")}
    )
    camera = {"camera_height_m": float(arg("camera_height_m"))}
    return [
        Node(
            package="perception_ros",
            executable="player_node",
            prefix=VENV_PYTHON,
            parameters=[
                {
                    "input": arg("input"),
                    "rate": float(arg("rate")),
                    "loop": arg("loop").lower() == "true",
                    "fov_deg": float(arg("fov_deg")),
                    "pitch_deg": float(arg("pitch_deg")),
                    **camera,
                }
            ],
            remappings=[*CAMERA_TOPICS, ("ego_speed", "/ego/speed")],
            output="screen",
        ),
        Node(
            package="perception_ros",
            executable="perception_node",
            prefix=VENV_PYTHON,
            namespace="perception",
            parameters=[{**pipeline_params, **camera, "camera_pitch_deg": float(arg("pitch_deg"))}],
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
