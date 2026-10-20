"""ROS2 adapters: CARLA bridge, perception and decision nodes.

Nodes stay thin: conversions.py maps ROS messages <-> perception.core.types,
and all logic lives in the perception package.
"""
