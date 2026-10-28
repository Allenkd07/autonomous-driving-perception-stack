"""Index the BDD100K 2018 lane markings and drivable areas (poly2d objects) into a Parquet table.

    python data/bdd100k_polygons.py                       # val (lane evaluation)
    python data/bdd100k_polygons.py --splits train val

Writes <root>/index/polygons.parquet: one row per polygon with split, image_id, kind (lane / area),
category, direction, style, vertices (flat x, y list) and types ("L" point, "C" Bezier control point).
Kept apart from data/bdd100k_index.py so lane work can be re-indexed without touching the detection index.
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from perception.io.bdd100k import parse_polygons


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data/bdd100k"))
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tables = []
    for split in args.splits:
        files = sorted((args.root / "labels" / "2018" / "100k" / split).glob("*.json"))
        if not files:
            raise SystemExit(f"no label files found for split '{split}' under {args.root}")
        start = time.perf_counter()
        rows = []
        with Pool(args.workers) as pool:
            for polygons in pool.imap_unordered(parse_polygons, files, chunksize=256):
                rows.extend(polygons)
        table = pd.DataFrame(rows).assign(split=split)
        print(
            f"{split}: {len(files):,} images, {len(table):,} polygons ({time.perf_counter() - start:.0f} s)"
        )
        tables.append(table)

    polygons = pd.concat(tables, ignore_index=True).sort_values(["split", "image_id"], ignore_index=True)
    for col in ("split", "kind", "category", "direction", "style"):
        polygons[col] = polygons[col].astype("category")
    out = args.root / "index" / "polygons.parquet"
    polygons.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
