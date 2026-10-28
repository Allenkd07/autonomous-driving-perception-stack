"""ROS settings for the node tests, applied before rclpy is imported (it picks the middleware at import)."""

import os
import random
import time

import pytest

# Cyclone DDS, as everywhere in this project (Fast DDS hangs under WSL2 mirrored networking); a random domain
# and localhost-only discovery keep tests from talking to any other running ROS graph
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
os.environ["ROS_DOMAIN_ID"] = str(random.randint(100, 200))
os.environ["ROS_LOCALHOST_ONLY"] = "1"


@pytest.fixture
def ros():
    import rclpy

    rclpy.init()
    yield rclpy
    rclpy.try_shutdown()


@pytest.fixture
def spin_until():
    """spin_until(executor, condition, timeout_s, publish): spin until condition() holds, calling publish()
    each round (discovery between nodes takes a moment, so messages are re-sent until they arrive)."""

    def spin(executor, condition, timeout_s=10.0, publish=None):
        deadline = time.monotonic() + timeout_s
        while not condition() and time.monotonic() < deadline:
            if publish is not None:
                publish()
            executor.spin_once(timeout_sec=0.05)
        return condition()

    return spin
