"""Turn public datasets into EvalScenario JSON for scripts/run-tracking-eval.ts.

Runs the *same* MediaPipe Pose model the app uses over dataset images, pairs each
result with the dataset's ground truth, and writes the observation stream the
Node harness replays. That keeps one scoring implementation for synthetic and
real data alike.

    .venv/bin/python scripts/build_eval_scenarios.py coco-wholebody --limit 400

Only public datasets and developer pre-validation clips belong here. Study
participant footage must never be passed to this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from person_detector import DETECTORS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POSE_MODEL = ROOT / "public" / "models" / "pose_landmarker_heavy.task"
DETECTOR_MODEL = ROOT / "public" / "models" / "efficientdet_lite0.tflite"
OUT_DIR = ROOT / "data" / "eval-scenarios"
RTMW_DIRS = {
    "rtmw-l-384x288": ROOT / "models" / "rtmw" / "rtmw-l-384x288",
    "rtmw-l-256x192": ROOT / "models" / "rtmw" / "rtmw-l-256x192",
    "rtmw-m-256x192": ROOT / "models" / "rtmw" / "rtmw-m-256x192",
}

# COCO body keypoint index -> MediaPipe Pose landmark index.
COCO_TO_MEDIAPIPE = {5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
                     11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}
# COCO head keypoints, used only to decide whether the head is visible.
COCO_HEAD = (0, 1, 2, 3, 4)
TICK_MS = 100
# Each still image is replayed as a short steady burst so the presence gate can
# actually run. One frame could never satisfy GLOBAL_CONFIRMATION_FRAMES, which
# would measure the harness rather than the tracker.
BURST = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(keypoints: list[float], bbox: list[float], image_h: int) -> str:
    """Condition label from ground truth alone - never from our own prediction."""
    visible = {index for index in range(17) if keypoints[index * 3 + 2] > 0}
    body_visible = sum(1 for index in COCO_TO_MEDIAPIPE if index in visible)
    if bbox[3] / image_h < 0.25:
        return "distant"
    if body_visible < 8:
        return "partial_body"
    if not (set(COCO_HEAD) & visible):
        return "head_occluded"
    if 0 not in visible:
        return "back"
    if (5 in visible) != (6 in visible) or (11 in visible) != (12 in visible):
        return "side"
    return "front"


def landmarks_payload(result):
    """MediaPipe result -> the NormalizedLandmarkLike[] the TS harness expects."""
    if not result.pose_landmarks:
        return None
    pose = result.pose_landmarks[0]
    return [
        {"x": point.x, "y": point.y, "z": point.z,
         "visibility": round(float(point.visibility), 4)}
        for point in pose
    ]


def truth_payload(keypoints: list[float], width: int, height: int):
    joints = {}
    for coco_index, mp_index in COCO_TO_MEDIAPIPE.items():
        x, y, v = keypoints[coco_index * 3: coco_index * 3 + 3]
        if v > 0:
            joints[str(mp_index)] = {"x": x / width, "y": y / height}
    return joints


def build_coco_wholebody(args) -> dict:
    coco_dir = ROOT / "data" / "external" / "coco"
    ann_path = coco_dir / "coco_wholebody_val_v1.0.json"
    images_zip = coco_dir / "val2017.zip"
    for path in (ann_path, images_zip, POSE_MODEL, DETECTOR_MODEL):
        if not path.exists():
            sys.exit(f"missing required file: {path}")

    print(f"reading {ann_path.name} ...", flush=True)
    data = json.loads(ann_path.read_text())
    images = {image["id"]: image for image in data["images"]}
    by_image: dict[int, list] = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    # The system tracks at most one person, so a crowd scene would measure a
    # capability it does not claim. Single-person images only, plus genuine
    # people-free images as the false-positive set.
    single = [
        (image_id, anns[0]) for image_id, anns in by_image.items()
        if len(anns) == 1 and anns[0].get("num_keypoints", 0) >= 8
        and anns[0].get("iscrowd", 0) == 0
    ]
    empty = [image_id for image_id in images if image_id not in by_image]
    single.sort()
    empty.sort()
    single = single[: args.limit]
    empty = empty[: args.empty_limit or max(1, args.limit // 4)]
    print(f"selected {len(single)} single-person and {len(empty)} people-free images",
          flush=True)

    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            # The Metal delegate crashes inside MediaPipe's Python build on
            # macOS. CPU is slower but this is an offline batch job.
            base_options=mp_python.BaseOptions(
                model_asset_path=str(POSE_MODEL),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
    )

    # The app never acquires a target on pose output alone: person detection
    # gates acquisition. Modelling that here is the difference between measuring
    # the tracker and measuring a pose model with no gate in front. Detector and
    # threshold are swept - see scripts/sweep_detectors.py.
    detector = DETECTORS[args.detector]()

    # Optional second backend. Same images, same detector, same ground truth,
    # same scoring code - only the pose model differs, which is what makes the
    # comparison an A/B rather than two unrelated numbers.
    rtmw = None
    if args.pose_model != "mediapipe-heavy":
        from rtmw_backend import RTMWBackend
        rtmw = RTMWBackend(RTMW_DIRS[args.pose_model])

    ticks = []
    timestamp = 0
    with zipfile.ZipFile(images_zip) as archive:
        names = {Path(name).name: name for name in archive.namelist()
                 if name.endswith(".jpg")}

        def run(image_id: int, annotation=None):
            nonlocal timestamp
            meta = images[image_id]
            entry = names.get(meta["file_name"])
            if entry is None:
                return
            buffer = archive.read(entry)
            frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return
            height, width = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = [box for box in detector.detect(frame)
                          if box[4] >= args.detector_threshold]
            person_confidence = detections[0][4] if detections else 0.0
            if rtmw is None:
                landmarks = landmarks_payload(landmarker.detect(image))
            elif detections:
                # RTMW is top-down, so it only runs where the detector - the
                # same gate the app uses - found a person.
                landmarks = rtmw.detect(frame, detections[0][:4])
            else:
                landmarks = None
            present = annotation is not None
            condition = (classify(annotation["keypoints"], annotation["bbox"], height)
                         if present else "empty")
            truth = ({"camera": 1, "joints": truth_payload(annotation["keypoints"], width, height)}
                     if present else None)
            for index in range(BURST):
                tick = {
                    "timestamp": timestamp,
                    "personPresent": present,
                    "condition": condition,
                    # Each image is independent: the previous image's person is
                    # not this image's person, so continuity must not carry over.
                    "resetBefore": index == 0,
                    "observations": [{
                        "cameraSlot": 1,
                        "timestamp": timestamp,
                        "landmarks": landmarks,
                        "personConfidence": person_confidence,
                        # A still image has no background model, so there is no
                        # foreground signal to report. Acquisition therefore
                        # rests on the detector alone, which is the stricter of
                        # the two paths the app allows.
                        "foregroundRatio": 0.0,
                        "confirmed": landmarks is not None and person_confidence > 0,
                    }],
                }
                if truth:
                    tick["truth"] = truth
                ticks.append(tick)
                timestamp += TICK_MS

        for position, (image_id, annotation) in enumerate(single):
            run(image_id, annotation)
            if position % 100 == 0:
                print(f"  {position}/{len(single)}", flush=True)
        for image_id in empty:
            run(image_id, None)

    landmarker.close()
    detector.close()
    suffix = "" if args.pose_model == "mediapipe-heavy" else f"-{args.pose_model}"
    if args.label:
        suffix = f"{suffix}-{args.label}"
    return {
        "name": f"coco-wholebody-val{suffix}",
        "description": (
            f"{len(single)} single-person and {len(empty)} people-free COCO val2017 "
            f"images, posed with {args.pose_model}, gated by "
            f"{args.detector} at score >= {args.detector_threshold}."
        ),
        "source": (
            "COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, "
            f"annotations sha256={sha256(ann_path)[:16]}...) over COCO val2017 images "
            "(http://images.cocodataset.org/zips/val2017.zip)"
        ),
        "ticks": ticks,
    }


BUILDERS = {"coco-wholebody": build_coco_wholebody}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(BUILDERS))
    parser.add_argument("--detector", default="efficientdet-lite0", choices=list(DETECTORS))
    parser.add_argument("--detector-threshold", type=float, default=0.5)
    parser.add_argument("--label", default="", help="suffix for the scenario name")
    parser.add_argument("--empty-limit", type=int, default=0,
                        help="people-free images; 0 means limit/4")
    parser.add_argument("--pose-model", default="mediapipe-heavy",
                        choices=["mediapipe-heavy", *RTMW_DIRS])
    parser.add_argument("--limit", type=int, default=400,
                        help="max single-person images (people-free adds limit/4)")
    args = parser.parse_args()

    scenario = BUILDERS[args.dataset](args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{scenario['name']}.json"
    out.write_text(json.dumps(scenario))
    print(f"wrote {out} ({len(scenario['ticks'])} ticks, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
