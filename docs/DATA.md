# Data

How the training and evaluation data is sourced, processed and what it contains. The full exploration, with charts, is in [notebooks/01_bdd100k_eda.ipynb](../notebooks/01_bdd100k_eda.ipynb).

## Sources

| Dataset | Used for | Ground truth |
| --- | --- | --- |
| **BDD100K** (100k dashcam images, 1280×720) | Detector fine-tuning and evaluation; lane evaluation | 2D boxes (10 classes), lane-marking polylines, scene attributes |
| **CARLA 0.9.15** (recorded in the simulator) | Tracking evaluation; depth and distance evaluation; the live demo | Exact object IDs, positions and per-pixel depth from the simulator |

BDD100K provides the labelled real-world images. The simulator provides what BDD100K can't: metric depth and consistent object identities across frames.

BDD100K is used under its licence for non-commercial research and education. The images and labels are not redistributed in this repository; only the scripts that process them are.

### BDD100K files

The detection boxes and lane polylines both come from the original release's per-image label files (`bdd100k_labels.zip`, one JSON per image), alongside `bdd100k_images_100k.zip`. The Berkeley mirror at `http://dl.yf.io/bdd100k/` hosts both. Expected layout:

```text
data/bdd100k/
  images/100k/{train,val,test}/<id>.jpg      70,000 / 10,000 / 20,000 images
  labels/2018/100k/{train,val}/<id>.json     one label file per image (test has no public labels)
```

Each label file holds the image's attributes (`weather`, `scene`, `timeofday`) and a list of objects: detection boxes (`box2d` with corners in pixels, plus `occluded` and `truncated` flags), lane markings (`lane/...` polylines) and drivable areas (`area/...` polygons).

## Processing pipeline

```bash
python data/bdd100k_index.py          # label JSONs -> data/bdd100k/index/{images,boxes}.parquet   (~15 s)
python data/bdd100k_to_yolo.py        # index -> YOLO dataset in data/bdd100k/yolo/                 (~1 min)
python scripts/log_bdd100k_eda.py     # summary tables + dataset version to Weights & Biases
```

1. **Index.** Every label file is parsed once (`perception.io.bdd100k.parse_label_file`) into two tables: one row per image, one row per box. All exploration and conversion reads these tables instead of the 80,000 JSON files.
2. **YOLO conversion** (`perception.io.bdd100k_yolo`): filter the classes and bad boxes, pick the training subset, and write one label file per image with `class_id cx cy w h` normalised to 0–1. Images are symlinked from the originals (`--copy` makes real copies, e.g. for a cloud training machine).

### Class set

Nine classes, with these YOLO ids:

| id | class | id | class | id | class |
| --- | --- | --- | --- | --- | --- |
| 0 | car | 3 | person | 6 | motorcycle |
| 1 | truck | 4 | rider | 7 | traffic light |
| 2 | bus | 5 | bicycle | 8 | traffic sign |

BDD100K's `bike` and `motor` are renamed to COCO's `bicycle` and `motorcycle`, so that fine-tuning from COCO-pretrained weights keeps what the model already knows about them (Ultralytics carries pretrained classes over by name). BDD100K's tenth class, `train`, is left out: it has only 136 boxes in the train split, too few to learn from, and the CARLA scenarios contain no trains.

### Cleaning

Boxes with zero or negative size (447) or under 2 px wide or tall (437) are dropped: 0.06% of all boxes, all labelling errors. No box extends outside the image.

### Training subset

Training uses **20,000 of the 70,000 train images**, chosen as:

1. every image containing a `bicycle`, `rider` or `motorcycle` (6,394 images), then
2. a random sample of the remaining images to fill up to 20,000 (fixed seed, so the subset is reproducible).

This keeps every example of the three rarest classes, which raises their share of the training boxes about threefold, while keeping training time manageable on a single small GPU. The subset size is a script option (`--train-size 70000` uses everything).

| Split | Images | Boxes |
| --- | --- | --- |
| train (subset) | 20,000 | 399,587 |
| val (complete) | 10,000 | 185,458 |
| val, daytime only (`val_day.txt`) | 5,258 | |
| val, night only (`val_night.txt`) | 3,929 | |

The day and night lists let the detector be evaluated separately on each.

## What the data looks like

Numbers are for the train split unless stated. Per-class proportions in val match train within 0.2 percentage points, so val scores are representative.

**Classes are heavily imbalanced.** Cars are 55% of all boxes; there are 238 cars for every motorcycle. Bikes, riders and motorcycles each appear in only 3–6% of images. Per-class mAP is therefore reported alongside the overall figure, which cars would otherwise dominate.

| class | boxes | images containing it |
| --- | --- | --- |
| car | 714,121 | 98.9% |
| traffic sign | 239,961 | 81.8% |
| traffic light | 186,301 | 56.1% |
| person | 91,435 | 31.6% |
| truck | 30,012 | 27.0% |
| bus | 11,688 | 12.9% |
| bike | 7,227 | 6.2% |
| rider | 4,522 | 5.1% |
| motor | 3,002 | 3.3% |

**Most objects are small.** After resizing to the detector's 640 px input, 79% of boxes are smaller than 32×32 px (COCO's "small" threshold). All traffic lights are small, with a median height of 10 px, and a third of traffic lights and signs are under 8 px tall. Vehicles and people are larger (median height 17–35 px). Small-object accuracy is the main limit for compact models, and input resolution is the main lever.

**Night is well represented.** 53% of images are daytime, 40% night, 7% dawn or dusk; rain and snow are 7–8% each. Weather labels are less reliable: 12% are `undefined`, and 82% of night images are labelled `clear`. Night images contain fewer labelled objects (median 15 vs 19).

**Occlusion is common.** 58–89% of vehicles, people and two-wheelers are flagged as partly occluded, and 3–28% are cut off by the image border. Distance estimation reads depth inside each box, so it uses robust statistics (the median) rather than the mean.

**Objects concentrate in a horizon band** (y ≈ 300–450 px). Vehicles peak just left of centre, where the road ahead is; pedestrians and two-wheelers peak on the right, the sidewalk side in right-hand traffic.

## Known limitations

- **Domain gap to the simulator.** BDD100K frames come from phone cameras behind the windscreen, often with the dashboard or hood visible along the bottom edge. The CARLA camera has no hood in view, a different mounting height and rendered rather than real textures. Detection quality in CARLA is measured separately for this reason.
- **No public tracking labels** in the release used here, so tracking is evaluated against CARLA ground truth instead.
- **Weather attributes are coarse** (see above); results are broken down by time of day, which is reliable, rather than by weather.
