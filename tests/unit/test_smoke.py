import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "perception",
        "perception.core.types",
        "perception.core.interfaces",
        "perception.core.registry",
        "perception.core.config",
        "perception.core.timing",
        "perception.pipeline",
        "perception.backends",
        "perception.detection",
        "perception.tracking",
        "perception.lanes",
        "perception.depth",
        "perception.fusion",
        "perception.planning",
        "perception.io",
        "perception.viz",
        "perception.eval",
    ],
)
def test_package_imports(module):
    importlib.import_module(module)
