"""Fit RTMW's per-keypoint visibility calibration on COCO-WholeBody.

RTMW is a SimCC model: its output is an unbounded classification peak, not a
visibility probability. Feeding that peak straight into a tracker that reasons
about `observed` / `inferred` / `missing` would mark every occluded joint as
confidently seen - the exact thing the study must not do.

So we fit one: a 1-D logistic (Platt) calibration from the raw SimCC peak to
P(keypoint is annotated visible), trained on one split of COCO-WholeBody val
and scored on a disjoint split. Two parameters, held-out evaluation, coefficients
written to config/ with their metrics.

    .venv/bin/python scripts/fit_rtmw_visibility.py --model rtmw-l-384x288 --limit 1200
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtmw_backend import RTMWBackend, RTMW_TO_MEDIAPIPE  # noqa: E402

OUT = ROOT / "config" / "rtmw-visibility-calibration.json"
# The tracker treats anything below this as not seen (MIN_JOINT_VISIBILITY in
# app/global-person-track.ts). The calibration must be judged at that point.
TRACKER_VISIBILITY_THRESHOLD = 0.25


def collect(model_dir: Path, limit: int):
    backend = RTMWBackend(model_dir)
    coco = ROOT / "data" / "external" / "coco"
    annotations = json.loads((coco / "coco_wholebody_val_v1.0.json").read_text())
    images = {image["id"]: image for image in annotations["images"]}
    by_image = defaultdict(list)
    for annotation in annotations["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    single = sorted(
        (image_id, anns[0]) for image_id, anns in by_image.items()
        if len(anns) == 1 and anns[0].get("num_keypoints", 0) >= 8
        and anns[0].get("iscrowd", 0) == 0
    )[:limit]

    scores: list[float] = []
    labels: list[int] = []
    groups: list[int] = []
    with zipfile.ZipFile(coco / "val2017.zip") as archive:
        names = {Path(name).name: name for name in archive.namelist()}
        for position, (image_id, annotation) in enumerate(single):
            entry = names.get(images[image_id]["file_name"])
            if entry is None:
                continue
            frame = cv2.imdecode(
                np.frombuffer(archive.read(entry), np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                continue
            raw = backend.raw_scores(frame, tuple(annotation["bbox"]))
            if raw is None:
                continue
            keypoints = annotation["keypoints"]
            for rtmw_index in RTMW_TO_MEDIAPIPE:
                if rtmw_index >= 17:
                    # Only the 17 COCO body points carry a per-point visibility
                    # flag we can supervise against.
                    continue
                scores.append(float(raw[rtmw_index]))
                labels.append(1 if keypoints[rtmw_index * 3 + 2] > 0 else 0)
                groups.append(image_id)
            if position % 200 == 0:
                print(f"  {position}/{len(single)}", flush=True)
    return np.array(scores), np.array(labels), np.array(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rtmw-l-384x288")
    parser.add_argument("--limit", type=int, default=1200)
    args = parser.parse_args()

    model_dir = ROOT / "models" / "rtmw" / args.model
    scores, labels, groups = collect(model_dir, args.limit)
    if len(set(labels)) < 2:
        sys.exit("need both visible and occluded keypoints to fit a calibration")

    # Split by image, not by keypoint: keypoints from one photo are correlated,
    # so a random keypoint split would leak and flatter the held-out score.
    unique = np.unique(groups)
    cut = unique[: int(len(unique) * 0.7)]
    train = np.isin(groups, cut)
    test = ~train
    print(f"keypoints: {len(scores)}  train={train.sum()}  test={test.sum()}  "
          f"visible={labels.mean():.1%}")

    model = LogisticRegression(max_iter=1000)
    model.fit(scores[train].reshape(-1, 1), labels[train])
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])

    probability = model.predict_proba(scores[test].reshape(-1, 1))[:, 1]
    auc = float(roc_auc_score(labels[test], probability))
    kept = probability >= TRACKER_VISIBILITY_THRESHOLD
    precision = float(labels[test][kept].mean()) if kept.any() else 0.0
    recall = float(kept[labels[test] == 1].mean())
    # What the uncalibrated model would have done: every peak clamps to 1.0,
    # so nothing is ever below the tracker's threshold.
    baseline_recall = 1.0
    baseline_precision = float(labels[test].mean())

    entry = {
        "kind": "platt-1d",
        "note": (
            "visibility = 1 / (1 + exp(-(slope * simcc_peak + intercept))). "
            "Fitted on COCO-WholeBody val body keypoints, split by image."
        ),
        "slope": slope,
        "intercept": intercept,
        "trackerThreshold": TRACKER_VISIBILITY_THRESHOLD,
        "fit": {
            "keypoints": int(len(scores)),
            "trainKeypoints": int(train.sum()),
            "testKeypoints": int(test.sum()),
            "heldOutAuc": auc,
            "precisionAtThreshold": precision,
            "recallAtThreshold": recall,
            "uncalibratedPrecision": baseline_precision,
            "uncalibratedRecall": baseline_recall,
        },
        "source": "COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    document = json.loads(OUT.read_text()) if OUT.exists() else {"models": {}}
    document.setdefault("models", {})[args.model] = entry
    OUT.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(entry["fit"], indent=2))
    print(f"wrote {OUT} [{args.model}]")


if __name__ == "__main__":
    main()
