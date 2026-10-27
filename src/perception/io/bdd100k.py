"""BDD100K label reader for the per-image JSON files of the 2018 release.

Each file describes one 1280x720 image: scene attributes (weather, scene, time of day) and a
list of objects. Objects with `box2d` are detection boxes; objects with `poly2d` are lane
markings (`lane/...`) or drivable areas (`area/...`).
"""

from __future__ import annotations

import json
from pathlib import Path

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

BOX_CATEGORIES = (
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck",
)


def parse_label_file(path: str | Path) -> tuple[dict, list[dict]]:
    """Return one image record and one record per detection box."""
    with open(path, encoding="utf-8") as f:
        label = json.load(f)

    image_id = label["name"]
    attrs = label.get("attributes", {})
    objects = label["frames"][0]["objects"] if label.get("frames") else []

    boxes = []
    n_lanes = n_drivable = 0
    for obj in objects:
        category = obj["category"]
        if "box2d" in obj:
            b = obj["box2d"]
            obj_attrs = obj.get("attributes", {})
            boxes.append(
                {
                    "image_id": image_id,
                    "category": category,
                    "x1": float(b["x1"]),
                    "y1": float(b["y1"]),
                    "x2": float(b["x2"]),
                    "y2": float(b["y2"]),
                    "occluded": bool(obj_attrs.get("occluded", False)),
                    "truncated": bool(obj_attrs.get("truncated", False)),
                    "traffic_light_color": obj_attrs.get("trafficLightColor", "none"),
                }
            )
        elif category.startswith("lane/"):
            n_lanes += 1
        elif category.startswith("area/"):
            n_drivable += 1

    image = {
        "image_id": image_id,
        "weather": attrs.get("weather", "undefined"),
        "scene": attrs.get("scene", "undefined"),
        "timeofday": attrs.get("timeofday", "undefined"),
        "n_boxes": len(boxes),
        "n_lanes": n_lanes,
        "n_drivable": n_drivable,
    }
    return image, boxes
