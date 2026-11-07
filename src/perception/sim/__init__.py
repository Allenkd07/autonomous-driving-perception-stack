"""Closed-loop simulation: drive the ego car along a lane, stage scenarios, measure what happened.

control and metrics are plain numpy (tested without a simulator); carla_sim and scenarios talk to CARLA and
are imported only where a CARLA client is installed (carla_bridge_node, the Kaggle notebook).
"""

from perception.sim.control import LaneFollower, apply_override
from perception.sim.metrics import ScenarioLog, forward_gap

__all__ = ["LaneFollower", "ScenarioLog", "apply_override", "forward_gap"]
