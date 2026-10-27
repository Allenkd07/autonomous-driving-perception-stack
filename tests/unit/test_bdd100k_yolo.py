import pandas as pd
import pytest

from perception.io.bdd100k_yolo import CLASSES, clean_boxes, select_train_subset, yolo_label_lines


def make_boxes(rows):
    return pd.DataFrame(rows, columns=["image_id", "category", "x1", "y1", "x2", "y2"])


def test_yolo_line_is_normalised_centre_and_size():
    boxes = make_boxes([("a", "bus", 320.0, 180.0, 960.0, 540.0)])

    (line,) = yolo_label_lines(boxes)

    # centre (640, 360) and size 640x360 in a 1280x720 image
    assert line == f"{CLASSES.index('bus')} 0.500000 0.500000 0.500000 0.500000"


def test_yolo_class_ids_are_ints_for_categorical_column():
    # The index stores category as a pandas categorical that also knows unused classes
    boxes = make_boxes([("a", "traffic sign", 0.0, 0.0, 64.0, 36.0)])
    boxes["category"] = pd.Categorical(boxes.category, categories=["car", "traffic sign", "train"])

    (line,) = yolo_label_lines(boxes)

    assert line.split()[0] == str(CLASSES.index("traffic sign"))


def test_yolo_lines_reject_unknown_class():
    with pytest.raises(ValueError, match="train"):
        yolo_label_lines(make_boxes([("a", "train", 0.0, 0.0, 10.0, 10.0)]))


def test_yolo_lines_empty_for_no_boxes():
    assert yolo_label_lines(make_boxes([])) == []


def test_clean_boxes_drops_other_classes_and_tiny_boxes():
    boxes = make_boxes(
        [
            ("a", "car", 0, 0, 50, 40),
            ("a", "train", 0, 0, 300, 200),  # not a training class
            ("a", "car", 10, 10, 10, 30),  # zero width
            ("a", "person", 10, 10, 11.5, 30),  # 1.5 px wide
            ("a", "traffic light", 10, 10, 12, 12),  # exactly 2 px: kept
        ]
    )

    kept = clean_boxes(boxes)

    assert list(kept.category) == ["car", "traffic light"]


def test_clean_boxes_renames_bdd_categories_to_class_names():
    boxes = make_boxes([("a", "bike", 0, 0, 50, 40), ("a", "motor", 0, 0, 50, 40)])
    boxes["category"] = pd.Categorical(boxes.category, categories=["bike", "motor", "train"])

    kept = clean_boxes(boxes)

    assert list(kept.category) == ["bicycle", "motorcycle"]
    assert set(kept.category) <= set(CLASSES)


@pytest.fixture
def pool():
    ids = pd.Series([f"img{i:03d}" for i in range(100)])
    boxes = make_boxes(
        [(i, "car", 0, 0, 10, 10) for i in ids]
        + [(i, "motorcycle", 0, 0, 10, 10) for i in ids[:5]]
        + [(i, "rider", 0, 0, 10, 10) for i in ids[3:8]]
    )
    return ids, boxes


def test_subset_contains_every_rare_image(pool):
    ids, boxes = pool

    subset = select_train_subset(ids, boxes, size=20, seed=1)

    assert len(subset) == 20 and len(set(subset)) == 20
    assert {f"img{i:03d}" for i in range(8)} <= set(subset)


def test_subset_is_reproducible(pool):
    ids, boxes = pool

    assert select_train_subset(ids, boxes, 20, seed=1) == select_train_subset(ids, boxes, 20, seed=1)
    assert select_train_subset(ids, boxes, 20, seed=1) != select_train_subset(ids, boxes, 20, seed=2)


def test_subset_smaller_than_rare_pool(pool):
    ids, boxes = pool

    subset = select_train_subset(ids, boxes, size=4, seed=0)

    assert len(subset) == 4 and set(subset) <= {f"img{i:03d}" for i in range(8)}
