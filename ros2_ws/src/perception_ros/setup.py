from glob import glob

from setuptools import setup

package_name = "perception_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Karthik Dileep",
    maintainer_email="kumardileep1969@gmail.com",
    description="ROS2 nodes wrapping the perception pipeline and the CARLA bridge",
    license="MIT",
    entry_points={
        # Added in Phase 5: carla_bridge_node, perception_node, decision_node
        "console_scripts": [],
    },
)
