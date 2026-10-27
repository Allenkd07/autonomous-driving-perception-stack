import json

from perception.io.bdd100k import parse_label_file


def write_label(tmp_path, objects, attributes=None):
    label = {
        "name": "img-0001",
        "attributes": (
            {"weather": "rainy", "scene": "highway", "timeofday": "night"}
            if attributes is None
            else attributes
        ),
        "frames": [{"timestamp": 10000, "objects": objects}],
    }
    path = tmp_path / "img-0001.json"
    path.write_text(json.dumps(label))
    return path


def test_parse_boxes_lanes_and_areas(tmp_path):
    path = write_label(
        tmp_path,
        [
            {
                "category": "car",
                "id": 0,
                "attributes": {"occluded": True, "truncated": False, "trafficLightColor": "none"},
                "box2d": {"x1": 10, "y1": 20, "x2": 110.5, "y2": 80},
            },
            {"category": "lane/single white", "id": 1, "attributes": {}, "poly2d": []},
            {"category": "lane/crosswalk", "id": 2, "attributes": {}, "poly2d": []},
            {"category": "area/drivable", "id": 3, "attributes": {}, "poly2d": []},
        ],
    )

    image, boxes = parse_label_file(path)

    assert image == {
        "image_id": "img-0001",
        "weather": "rainy",
        "scene": "highway",
        "timeofday": "night",
        "n_boxes": 1,
        "n_lanes": 2,
        "n_drivable": 1,
    }
    assert boxes == [
        {
            "image_id": "img-0001",
            "category": "car",
            "x1": 10.0,
            "y1": 20.0,
            "x2": 110.5,
            "y2": 80.0,
            "occluded": True,
            "truncated": False,
            "traffic_light_color": "none",
        }
    ]


def test_parse_image_without_objects(tmp_path):
    image, boxes = parse_label_file(write_label(tmp_path, [], attributes={}))

    assert boxes == []
    assert image["n_boxes"] == 0
    assert image["weather"] == "undefined"
