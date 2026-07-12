#!/usr/bin/env python3
"""Download COCO val2017 dataset for benchmarking and calibration.

Downloads images + annotation files. Creates task-specific YAML configs.
Usage: python3 setup_coco_val.py [target_dir]
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

IMAGES_ZIP = "val2017.zip"
ANNOTATIONS_ZIP = "annotations_trainval2017.zip"

COCO_VAL_YAML = """\
path: {base}
val: images/val2017
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""

COCO_VAL_POSE_YAML = """\
path: {base}
val: images/val2017
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
"""


def download(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"[SKIP] {dst.name} already exists")
        return
    print(f"[DOWNLOAD] {url}")
    tmp = dst.with_suffix(".tmp")
    cmd = ["curl", "-L", "--progress-bar", "-o", str(tmp), url]
    subprocess.run(cmd, check=True)
    tmp.rename(dst)
    print(f"\n[DONE] {dst.name}")


def extract(zip_path: Path, target_dir: Path) -> None:
    print(f"[EXTRACT] {zip_path.name} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    zip_path.unlink()
    print(f"[DONE] extracted {zip_path.name}")


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "coco_val2017"
    target.mkdir(parents=True, exist_ok=True)

    images_zip = target / IMAGES_ZIP
    ann_zip = target / ANNOTATIONS_ZIP

    download(IMAGES_URL, images_zip)
    extract(images_zip, target)

    download(ANNOTATIONS_URL, ann_zip)
    extract(ann_zip, target)

    base = str(target)
    (target / "coco_val.yaml").write_text(COCO_VAL_YAML.format(base=base))
    (target / "coco_val_pose.yaml").write_text(COCO_VAL_POSE_YAML.format(base=base))
    print(f"[DONE] YAML configs created")

    n_images = len(list((target / "images" / "val2017").glob("*.jpg")))
    print(f"\nDataset ready at {target}")
    print(f"  Images: {n_images}")
    print(f"  Annotations:")
    for f in sorted((target / "annotations").glob("*.json")):
        print(f"    {f.name} ({f.stat().st_size // 1024} KB)")
    print(f"  YAML configs:")
    for f in sorted(target.glob("*.yaml")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
