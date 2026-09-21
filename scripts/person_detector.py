"""Person detectors behind one interface, so they can be swept against each other.

The app currently gates target acquisition with EfficientDet-Lite0. That gate is
a hard ceiling on a top-down pose model like RTMW: a person the detector misses
is a person RTMW never sees. Measured on COCO-WholeBody, EfficientDet-Lite0
recalls only 53% of subjects under 15% of frame height - which is exactly the
"distant person" condition the study has to handle.

Every detector here returns the same thing: a list of (x, y, w, h, score) in
image pixels, sorted by score, unfiltered. Thresholding is the caller's job so
one inference pass can serve a whole threshold sweep.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class EfficientDetLite0:
    """The detector the app ships with today, via MediaPipe Tasks."""

    name = "efficientdet-lite0"

    def __init__(self, score_threshold: float = 0.05):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.detector = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(ROOT / "public" / "models" / "efficientdet_lite0.tflite"),
                    delegate=mp_python.BaseOptions.Delegate.CPU,
                ),
                running_mode=vision.RunningMode.IMAGE,
                max_results=5,
                # Swept offline, so the model runs once at its floor.
                score_threshold=score_threshold,
                category_allowlist=["person"],
            )
        )

    def detect(self, frame_bgr: np.ndarray):
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        )
        boxes = []
        for detection in self.detector.detect(image).detections:
            box = detection.bounding_box
            boxes.append((float(box.origin_x), float(box.origin_y),
                          float(box.width), float(box.height),
                          float(detection.categories[0].score)))
        return sorted(boxes, key=lambda item: -item[4])

    def close(self):
        self.detector.close()


class YoloxPerson:
    """YOLOX person detector from the MMPose ONNX SDK (HumanArt training set).

    NMS is baked into the exported graph, so the outputs are already final
    boxes. Preprocessing is letterbox resize with 114 padding, BGR, no
    normalisation - matching the exported pipeline, not the stale pipeline.json
    size field.
    """

    def __init__(self, model_dir: Path, name: str):
        import onnxruntime as ort

        self.name = name
        options = ort.SessionOptions()
        options.log_severity_level = 3
        onnx = next(Path(model_dir).rglob("end2end.onnx"))
        self.session = ort.InferenceSession(
            str(onnx), options, providers=["CPUExecutionProvider"]
        )
        shape = self.session.get_inputs()[0].shape
        self.input_h, self.input_w = int(shape[2]), int(shape[3])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame_bgr: np.ndarray):
        height, width = frame_bgr.shape[:2]
        ratio = min(self.input_h / height, self.input_w / width)
        resized = cv2.resize(
            frame_bgr,
            (int(round(width * ratio)), int(round(height * ratio))),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32)

        dets, labels = self.session.run(None, {self.input_name: tensor})
        boxes = []
        for (x1, y1, x2, y2, score), label in zip(dets[0], labels[0]):
            # COCO class 0 is person; the export keeps the full COCO head.
            if int(label) != 0 or score <= 0:
                continue
            x1, y1, x2, y2 = (value / ratio for value in (x1, y1, x2, y2))
            boxes.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(score)))
        return sorted(boxes, key=lambda item: -item[4])

    def close(self):
        pass


DETECTORS = {
    "efficientdet-lite0": lambda: EfficientDetLite0(),
    "yolox-nano": lambda: YoloxPerson(ROOT / "models" / "yolox" / "yolox_nano", "yolox-nano"),
    "yolox-tiny": lambda: YoloxPerson(ROOT / "models" / "yolox" / "yolox_tiny", "yolox-tiny"),
    "yolox-m": lambda: YoloxPerson(ROOT / "models" / "yolox" / "yolox_m", "yolox-m"),
}


def iou(a, b) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    overlap = (x2 - x1) * (y2 - y1)
    return overlap / (aw * ah + bw * bh - overlap)
