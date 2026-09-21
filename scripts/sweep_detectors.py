"""Sweep person detectors and score thresholds on COCO-WholeBody val.

Two numbers decide the operating point, and they pull against each other:

  recall   - a person the detector misses is a person a top-down pose model
             never sees, so this caps tracking retention outright
  empty-FP - a people-free frame the detector fires on is the first half of an
             empty-room false positive, the criterion with the tightest budget

Each image is inferred once at a floor threshold and every threshold is scored
offline from the recorded scores, so the sweep costs one pass.

    .venv/bin/python scripts/sweep_detectors.py --limit 1500
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from person_detector import DETECTORS, iou  # noqa: E402

OUT = ROOT / "docs" / "tracking-eval" / "detector-sweep.json"
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# A box has to actually be on the person to count as a hit; a loose box makes
# RTMW crop the wrong region, which is worse than no detection at all.
MIN_IOU = 0.5


def load_images(limit: int):
    coco = ROOT / "data" / "external" / "coco"
    data = json.loads((coco / "coco_wholebody_val_v1.0.json").read_text())
    images = {image["id"]: image for image in data["images"]}
    by_image = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    single = sorted(
        (image_id, anns[0]) for image_id, anns in by_image.items()
        if len(anns) == 1 and anns[0].get("num_keypoints", 0) >= 8
        and anns[0].get("iscrowd", 0) == 0
    )[:limit]
    # Every people-free image in val2017, not a sample: the empty-frame budget
    # is the tightest criterion and deserves all the resolution available.
    empty = sorted(image_id for image_id in images if image_id not in by_image)
    return images, single, empty


def bucket(relative_height: float) -> str:
    if relative_height < 0.15:
        return "tiny"
    if relative_height < 0.25:
        return "small"
    if relative_height < 0.5:
        return "medium"
    return "large"


def run(detector_name: str, images, single, empty, archive, names):
    factory = DETECTORS[detector_name]
    detector = factory()
    hits: list[tuple[float, str]] = []      # (best score for a correct box, size bucket)
    misses: list[str] = []                  # size bucket with no correct box at any score
    empty_scores: list[float] = []          # best score on a people-free image

    def read(image_id):
        entry = names.get(images[image_id]["file_name"])
        if entry is None:
            return None
        return cv2.imdecode(np.frombuffer(archive.read(entry), np.uint8), cv2.IMREAD_COLOR)

    for position, (image_id, annotation) in enumerate(single):
        frame = read(image_id)
        if frame is None:
            continue
        size = bucket(annotation["bbox"][3] / frame.shape[0])
        best = 0.0
        for box in detector.detect(frame):
            if iou(box, annotation["bbox"]) >= MIN_IOU:
                best = max(best, box[4])
        if best > 0:
            hits.append((best, size))
        else:
            misses.append(size)
        if position % 200 == 0:
            print(f"  {detector_name} person {position}/{len(single)}", flush=True)

    for position, image_id in enumerate(empty):
        frame = read(image_id)
        if frame is None:
            continue
        boxes = detector.detect(frame)
        empty_scores.append(boxes[0][4] if boxes else 0.0)
        if position % 500 == 0:
            print(f"  {detector_name} empty {position}/{len(empty)}", flush=True)

    detector.close()

    total = len(hits) + len(misses)
    by_size = defaultdict(lambda: {"total": 0, "scores": []})
    for size in misses:
        by_size[size]["total"] += 1
    for score, size in hits:
        by_size[size]["total"] += 1
        by_size[size]["scores"].append(score)

    rows = []
    for threshold in THRESHOLDS:
        recall = sum(1 for score, _ in hits if score >= threshold) / max(1, total)
        false_positives = sum(1 for score in empty_scores if score >= threshold)
        rows.append({
            "threshold": threshold,
            "recall": recall,
            "emptyFalsePositives": false_positives,
            "emptyImages": len(empty_scores),
            "emptyFalsePositiveRate": false_positives / max(1, len(empty_scores)),
            "recallBySize": {
                size: sum(1 for score in stats["scores"] if score >= threshold) / max(1, stats["total"])
                for size, stats in by_size.items()
            },
        })
    return {"detector": detector_name, "personImages": total, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--detectors", nargs="*", default=list(DETECTORS))
    args = parser.parse_args()

    images, single, empty = load_images(args.limit)
    print(f"{len(single)} single-person, {len(empty)} people-free images", flush=True)
    results = []
    with zipfile.ZipFile(ROOT / "data" / "external" / "coco" / "val2017.zip") as archive:
        names = {Path(name).name: name for name in archive.namelist()}
        for detector_name in args.detectors:
            results.append(run(detector_name, images, single, empty, archive, names))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"minIou": MIN_IOU, "results": results}, indent=2) + "\n")

    print(f"\n{'detector':20s} {'thr':>5s} {'recall':>7s} {'tiny':>7s} {'small':>7s} "
          f"{'medium':>7s} {'large':>7s} {'emptyFP':>8s}")
    for result in results:
        for row in result["rows"]:
            sizes = row["recallBySize"]
            print(f"{result['detector']:20s} {row['threshold']:5.2f} {row['recall']:7.1%} "
                  f"{sizes.get('tiny', 0):7.1%} {sizes.get('small', 0):7.1%} "
                  f"{sizes.get('medium', 0):7.1%} {sizes.get('large', 0):7.1%} "
                  f"{row['emptyFalsePositives']:4d}/{row['emptyImages']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
