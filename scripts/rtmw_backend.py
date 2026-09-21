"""RTMW WholeBody ONNX inference, mapped into the MediaPipe Pose 33-point layout.

Top-down model: it needs a person box, which comes from the same EfficientDet
detector the app already gates acquisition with. Keypoints RTMW does not
provide come back with visibility 0 - never a fabricated coordinate.

Used by scripts/build_eval_scenarios.py to produce an RTMW observation stream
that the Node harness scores with exactly the same code as the MediaPipe one.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# RTMW emits the 133-point COCO-WholeBody layout. Only the points that exist in
# the MediaPipe Pose layout the rest of the app is written against are carried
# over; the face and hand points are handled by the separate hand pipeline.
RTMW_TO_MEDIAPIPE = {
    # COCO body
    5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
    11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
    # feet: big toe -> foot index, heel -> heel
    17: 31, 19: 29, 20: 32, 22: 30,
    # head points MediaPipe also carries, used for head direction only
    0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
}
MEDIAPIPE_LANDMARK_COUNT = 33
# Padding MMPose uses around the detection box for top-down whole-body models.
BBOX_PADDING = 1.25


class RTMWBackend:
    def __init__(self, model_dir: Path):
        onnx_path = Path(model_dir) / "end2end.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"no end2end.onnx under {model_dir}")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        shape = self.session.get_inputs()[0].shape
        # The zipped pipeline.json is stale in some SDK builds, so the graph's
        # own input shape is the authoritative model resolution.
        self.input_h, self.input_w = int(shape[2]), int(shape[3])
        self.input_name = self.session.get_inputs()[0].name
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        # SimCC emits an unbounded classification peak, not a probability: on
        # COCO-WholeBody the peak for an *occluded* keypoint has a median of
        # 2.60 against 4.04 for a visible one, so clamping it to [0,1] would
        # mark every occlusion as confidently seen. The calibration maps the
        # peak to P(visible); without it, visibility is left unreported.
        self.name = Path(model_dir).name
        self.calibration = None
        calibration_path = (
            Path(model_dir).parent.parent.parent / "config" / "rtmw-visibility-calibration.json"
        )
        if calibration_path.exists():
            # Keyed per model: the SimCC peak scale differs between variants,
            # so one variant's calibration is not valid for another.
            fitted = json.loads(calibration_path.read_text()).get("models", {}).get(self.name)
            if fitted:
                self.calibration = (float(fitted["slope"]), float(fitted["intercept"]))

    def _affine(self, bbox: tuple[float, float, float, float]):
        """Box -> model input warp, preserving aspect ratio like MMPose does."""
        x, y, w, h = bbox
        center = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        aspect = self.input_w / self.input_h
        if w > h * aspect:
            h = w / aspect
        else:
            w = h * aspect
        scale = np.array([w, h], dtype=np.float32) * BBOX_PADDING
        source = np.array([
            center,
            center + [0, -scale[1] / 2],
            center + [scale[0] / 2, 0],
        ], dtype=np.float32)
        target = np.array([
            [self.input_w / 2, self.input_h / 2],
            [self.input_w / 2, 0],
            [self.input_w, self.input_h / 2],
        ], dtype=np.float32)
        return cv2.getAffineTransform(source, target)

    def raw_scores(self, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]):
        """Unbounded SimCC peak per keypoint - the calibration's input."""
        result = self._infer(frame_bgr, bbox)
        return None if result is None else result[1]

    def _infer(self, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]):
        if bbox[2] <= 1 or bbox[3] <= 1:
            return None
        warp = self._affine(bbox)
        patch = cv2.warpAffine(frame_bgr, warp, (self.input_w, self.input_h),
                               flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = ((rgb - self.mean) / self.std).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self.session.run(None, {self.input_name: tensor})
        simcc_x, simcc_y = simcc_x[0], simcc_y[0]
        ratio_x = simcc_x.shape[1] / self.input_w
        ratio_y = simcc_y.shape[1] / self.input_h
        points = np.stack([
            simcc_x.argmax(axis=1) / ratio_x,
            simcc_y.argmax(axis=1) / ratio_y,
        ], axis=1).astype(np.float32)
        scores = np.minimum(simcc_x.max(axis=1), simcc_y.max(axis=1))
        inverse = cv2.invertAffineTransform(warp)
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        return np.concatenate([points, ones], axis=1) @ inverse.T, scores

    def detect(self, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]):
        """Returns MediaPipe-layout landmarks, or None when the box is unusable."""
        height, width = frame_bgr.shape[:2]
        result = self._infer(frame_bgr, bbox)
        if result is None:
            return None
        image_points, scores = result

        landmarks = [
            {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
            for _ in range(MEDIAPIPE_LANDMARK_COUNT)
        ]
        found = False
        for rtmw_index, mp_index in RTMW_TO_MEDIAPIPE.items():
            if rtmw_index >= image_points.shape[0]:
                continue
            score = float(scores[rtmw_index])
            if self.calibration is None:
                # No fitted calibration: refuse to invent a visibility number.
                visibility = 0.0
            else:
                slope, intercept = self.calibration
                visibility = 1.0 / (1.0 + np.exp(-(slope * score + intercept)))
            if visibility <= 0:
                continue
            px, py = image_points[rtmw_index]
            landmarks[mp_index] = {
                "x": float(px / width),
                "y": float(py / height),
                # RTMW is a 2D model; it has no depth to report and must not
                # pretend otherwise.
                "z": 0.0,
                "visibility": round(float(visibility), 4),
            }
            found = True
        return landmarks if found else None
