#!/usr/bin/env python3
"""Train an app-compatible café-work primitive bootstrap on LARa OMoCap."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/lara/OMoCap data.zip"
MODEL_PATH = ROOT / "models/lara-primitives-v1.json"
REPORT_PATH = ROOT / "docs/lara-training-report.md"
CACHE_PATH = ROOT / "data/external/lara/lara-app-features-v2.npz"
SEED = 20260916
SOURCE_HZ = 200
TARGET_HZ = 15
WINDOW_SAMPLES = 32
WINDOW_STRIDE = 16
MAX_WINDOWS_PER_SUBJECT_CLASS = 120
DISPLAY_ASPECT_RATIO = 16 / 9

TRAIN_SUBJECTS = {1, 2, 3, 4, 7, 8, 9, 10}
VALIDATION_SUBJECTS = {5, 11, 12}
TEST_SUBJECTS = {6, 13, 14}

CLASS_NAMES = {
    0: "STAND",
    1: "WALK",
    2: "PUSH_PULL",
    3: "LIFT",
    4: "TRANSFER",
    5: "BEND_DOWN",
}
JOINTS = [
    "L_humerus", "R_humerus", "L_elbow", "R_elbow", "L_wrist", "R_wrist",
    "L_femur", "R_femur", "L_tibia", "R_tibia", "L_foot", "R_foot",
]


def subject_from_member(name: str) -> int | None:
    match = re.search(r"(?:^|[/_])S(\d{2})(?:[/_])", name)
    return int(match.group(1)) if match else None


def resample(sequence: np.ndarray, count: int = 28) -> np.ndarray:
    positions = np.linspace(0, len(sequence) - 1, count)
    before = np.floor(positions).astype(int)
    after = np.minimum(len(sequence) - 1, before + 1)
    amount = (positions - before).reshape(-1, 1, 1)
    return sequence[before] + (sequence[after] - sequence[before]) * amount


def app_feature(sequence: np.ndarray, horizontal_axis: int) -> np.ndarray | None:
    points = resample(sequence)
    points_2d = points[:, :, [horizontal_axis, 2]].copy()
    points_2d[:, :, 0] *= DISPLAY_ASPECT_RATIO
    shoulders = points_2d[:, :2]
    centers = shoulders.mean(axis=1)
    spans = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    valid = spans[np.isfinite(spans) & (spans > 1e-6)]
    if not len(valid):
        return None
    scales = np.maximum(np.median(valid) * 0.2, spans / 0.64)
    normalized = (points_2d - centers[:, None, :]) / scales[:, None, None]
    travel = (centers - centers[0]) / scales[0]
    pose = normalized.reshape(28, -1)
    motion = pose - pose[0]
    velocity = np.diff(motion, axis=0)
    pose_bins = np.stack([chunk.mean(axis=0) for chunk in np.array_split(pose, 4)])
    pose_features = np.concatenate([
        pose.mean(axis=0), pose.std(axis=0), pose.min(axis=0), pose.max(axis=0),
        motion[-1], np.ptp(motion, axis=0), np.abs(velocity).mean(axis=0),
        np.abs(velocity).max(axis=0), pose_bins.reshape(-1),
    ])
    travel_velocity = np.diff(travel, axis=0)
    travel_features = np.concatenate([
        travel[-1], np.ptp(travel, axis=0), np.abs(travel_velocity).sum(axis=0),
        np.abs(travel_velocity).mean(axis=0), np.abs(travel_velocity).max(axis=0),
    ])
    return np.concatenate([pose_features, travel_features]).astype(np.float32)


def add_reservoir(store, seen, key, window, rng):
    seen[key] += 1
    values = store[key]
    if len(values) < MAX_WINDOWS_PER_SUBJECT_CLASS:
        values.append(window)
        return
    replacement = int(rng.integers(0, seen[key]))
    if replacement < MAX_WINDOWS_PER_SUBJECT_CLASS:
        values[replacement] = window


def flush_segment(store, seen, subject, label, samples, rng):
    if label not in CLASS_NAMES or len(samples) < 12:
        return
    array = np.asarray(samples, dtype=np.float32)
    starts = list(range(0, max(1, len(array) - WINDOW_SAMPLES + 1), WINDOW_STRIDE))
    if len(array) >= WINDOW_SAMPLES and starts[-1] != len(array) - WINDOW_SAMPLES:
        starts.append(len(array) - WINDOW_SAMPLES)
    for start in starts:
        window = array[start:start + WINDOW_SAMPLES]
        if len(window) < 12:
            continue
        add_reservoir(store, seen, (subject, label), window, rng)


def load_windows():
    rng = np.random.default_rng(SEED)
    store = defaultdict(list)
    seen = defaultdict(int)
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = [
            name for name in archive.namelist()
            if name.endswith("_norm_data.csv")
            and not Path(name).name.startswith("._")
            and subject_from_member(name) in TRAIN_SUBJECTS | VALIDATION_SUBJECTS | TEST_SUBJECTS
        ]
        print(f"Reading {len(members)} annotated LARa recordings from the archive")
        for member_index, member in enumerate(members, start=1):
            subject = subject_from_member(member)
            with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                reader = csv.reader(text)
                header = [value.strip().replace(" ", "_") for value in next(reader)]
                indexes = []
                for joint in JOINTS:
                    indexes.append([header.index(f"{joint}_T{axis}") for axis in "XYZ"])
                current_label = None
                segment = []
                downsample_counter = 0
                for row in reader:
                    try:
                        label = int(float(row[1]))
                    except (ValueError, IndexError):
                        continue
                    if current_label is None:
                        current_label = label
                    if label != current_label:
                        flush_segment(store, seen, subject, current_label, segment, rng)
                        current_label = label
                        segment = []
                        downsample_counter = 0
                    if downsample_counter % round(SOURCE_HZ / TARGET_HZ) == 0:
                        try:
                            segment.append([[float(row[index]) for index in joint] for joint in indexes])
                        except (ValueError, IndexError):
                            pass
                    downsample_counter += 1
                flush_segment(store, seen, subject, current_label, segment, rng)
            if member_index % 25 == 0 or member_index == len(members):
                print(f"Parsed {member_index}/{len(members)} recordings")
    return store, seen


def make_split(store, subjects):
    features, labels = [], []
    for (subject, label), windows in sorted(store.items()):
        if subject not in subjects:
            continue
        for window in windows:
            for horizontal_axis in (0, 1):
                feature = app_feature(window, horizontal_axis)
                if feature is not None and np.all(np.isfinite(feature)):
                    features.append(feature)
                    labels.append(label)
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def loss_and_accuracy(x, y, weights, bias, l2=0.0):
    probabilities = softmax(x @ weights + bias)
    loss = -np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-12)).mean()
    loss += l2 * float(np.square(weights).sum())
    return float(loss), float((probabilities.argmax(axis=1) == y).mean())


def train_mlp(train_x, train_y, validation_x, validation_y):
    rng = np.random.default_rng(SEED)
    class_count = len(CLASS_NAMES)
    hidden_count = 64
    parameters = {
        "inputHidden": (rng.standard_normal((train_x.shape[1], hidden_count)) * math.sqrt(2 / train_x.shape[1])).astype(np.float32),
        "hiddenBias": np.zeros(hidden_count, dtype=np.float32),
        "hiddenOutput": (rng.standard_normal((hidden_count, class_count)) * math.sqrt(2 / hidden_count)).astype(np.float32),
        "outputBias": np.zeros(class_count, dtype=np.float32),
    }
    moments = {key: np.zeros_like(value) for key, value in parameters.items()}
    velocities = {key: np.zeros_like(value) for key, value in parameters.items()}
    best = None
    best_loss = math.inf
    patience = 0
    step = 0
    for epoch in range(1, 121):
        order = rng.permutation(len(train_y))
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            x, y = train_x[batch], train_y[batch]
            hidden_linear = x @ parameters["inputHidden"] + parameters["hiddenBias"]
            hidden = np.maximum(hidden_linear, 0)
            probabilities = softmax(hidden @ parameters["hiddenOutput"] + parameters["outputBias"])
            probabilities[np.arange(len(y)), y] -= 1
            probabilities /= len(y)
            hidden_gradient = probabilities @ parameters["hiddenOutput"].T
            hidden_gradient[hidden_linear <= 0] = 0
            gradients = {
                "hiddenOutput": hidden.T @ probabilities + 2e-4 * parameters["hiddenOutput"],
                "outputBias": probabilities.sum(axis=0),
                "inputHidden": x.T @ hidden_gradient + 2e-4 * parameters["inputHidden"],
                "hiddenBias": hidden_gradient.sum(axis=0),
            }
            step += 1
            correction_1 = 1 - 0.9 ** step
            correction_2 = 1 - 0.999 ** step
            for key in parameters:
                moments[key] = 0.9 * moments[key] + 0.1 * gradients[key]
                velocities[key] = 0.999 * velocities[key] + 0.001 * np.square(gradients[key])
                parameters[key] -= 0.0008 * (moments[key] / correction_1) / (np.sqrt(velocities[key] / correction_2) + 1e-8)
        validation_logits = np.maximum(
            validation_x @ parameters["inputHidden"] + parameters["hiddenBias"], 0,
        ) @ parameters["hiddenOutput"] + parameters["outputBias"]
        validation_probabilities = softmax(validation_logits)
        validation_loss = float(-np.log(np.maximum(validation_probabilities[np.arange(len(validation_y)), validation_y], 1e-12)).mean())
        validation_accuracy = float((validation_probabilities.argmax(axis=1) == validation_y).mean())
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.3f}")
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best = ({key: value.copy() for key, value in parameters.items()}, epoch)
            patience = 0
        else:
            patience += 1
            if patience >= 18:
                break
    return best


def evaluate(x, y, parameters):
    hidden = np.maximum(x @ parameters["inputHidden"] + parameters["hiddenBias"], 0)
    predicted = (hidden @ parameters["hiddenOutput"] + parameters["outputBias"]).argmax(axis=1)
    class_count = len(CLASS_NAMES)
    confusion = np.zeros((class_count, class_count), dtype=int)
    for actual, prediction in zip(y, predicted):
        confusion[actual, prediction] += 1
    per_class = {
        CLASS_NAMES[index]: float(confusion[index, index] / max(1, confusion[index].sum()))
        for index in range(class_count)
    }
    return {
        "accuracy": float((predicted == y).mean()),
        "macroRecall": float(np.mean(list(per_class.values()))),
        "perClassRecall": per_class,
        "confusion": confusion.tolist(),
        "samples": int(len(y)),
    }


def rounded_list(array):
    return np.round(array, 7).tolist()


def main():
    if not ARCHIVE.exists():
        raise SystemExit(f"Missing {ARCHIVE}; run npm run download:lara first")
    if CACHE_PATH.exists():
        cached = np.load(CACHE_PATH)
        train_x, train_y = cached["train_x"], cached["train_y"]
        validation_x, validation_y = cached["validation_x"], cached["validation_y"]
        test_x, test_y = cached["test_x"], cached["test_y"]
        seen = {}
        print(f"Loaded cached features from {CACHE_PATH}")
    else:
        store, seen = load_windows()
        train_x, train_y = make_split(store, TRAIN_SUBJECTS)
        validation_x, validation_y = make_split(store, VALIDATION_SUBJECTS)
        test_x, test_y = make_split(store, TEST_SUBJECTS)
        np.savez_compressed(
            CACHE_PATH,
            train_x=train_x,
            train_y=train_y,
            validation_x=validation_x,
            validation_y=validation_y,
            test_x=test_x,
            test_y=test_y,
        )
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-5] = 1
    train_x = np.clip((train_x - mean) / std, -8, 8)
    validation_x = np.clip((validation_x - mean) / std, -8, 8)
    test_x = np.clip((test_x - mean) / std, -8, 8)
    print(f"windows train={len(train_y)} validation={len(validation_y)} test={len(test_y)}")
    parameters, best_epoch = train_mlp(train_x, train_y, validation_x, validation_y)
    train_metrics = evaluate(train_x, train_y, parameters)
    validation_metrics = evaluate(validation_x, validation_y, parameters)
    test_metrics = evaluate(test_x, test_y, parameters)
    model = {
        "schemaVersion": 1,
        "name": "LARa café-work primitive bootstrap",
        "source": "LARa Version 01 OMoCap",
        "sourceUrl": "https://zenodo.org/records/3862782",
        "featureLayout": "28-frame app-compatible 12-joint pose, motion, velocity, temporal-bin and travel statistics",
        "classes": [CLASS_NAMES[index] for index in range(len(CLASS_NAMES))],
        "officialSplit": {
            "trainSubjects": sorted(TRAIN_SUBJECTS),
            "validationSubjects": sorted(VALIDATION_SUBJECTS),
            "testSubjects": sorted(TEST_SUBJECTS),
        },
        "bestEpoch": best_epoch,
        "normalization": {"mean": rounded_list(mean), "std": rounded_list(std)},
        "architecture": {"type": "mlp", "hiddenUnits": 64, "activation": "relu"},
        "parameters": {key: rounded_list(value) for key, value in parameters.items()},
        "metrics": {"train": train_metrics, "validation": validation_metrics, "test": test_metrics},
        "splitClassCounts": {
            split_name: {
                CLASS_NAMES[label]: int(np.count_nonzero(labels == label))
                for label in range(len(CLASS_NAMES))
            }
            for split_name, labels in {
                "train": train_y,
                "validation": validation_y,
                "test": test_y,
            }.items()
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, separators=(",", ":")) + "\n", encoding="utf-8")
    rows = "\n".join(
        f"| {label} | {test_metrics['perClassRecall'][label] * 100:.1f}% |"
        for label in model["classes"]
    )
    REPORT_PATH.write_text(
        f"""# LARa 1차 사전학습 결과

- 공식 사람 분리: 학습 8명 / 검증 3명 / 평가 3명
- 앱 호환 입력: 공통 전신 12관절, 28프레임, 2D 가상 시점
- 평가 표본: {test_metrics['samples']:,}개
- 최적 epoch: {best_epoch}

## 결과

- 평가 정확도: **{test_metrics['accuracy'] * 100:.1f}%**
- macro recall: **{test_metrics['macroRecall'] * 100:.1f}%**

| 카페 원시동작군 | 평가 recall |
|---|---:|
{rows}

## 주의

LARa 물류 환경 내부의 사람 분리 결과이며 두 카페 참여자의 현장 성능이 아니다. 이 모델은 공개 데이터 사전학습 초기값으로 사용하고, 두 참여자의 다른 날짜 영상으로 개인화·잠금 평가해야 한다.
""",
        encoding="utf-8",
    )
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
