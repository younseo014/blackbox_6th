import assert from "node:assert/strict";
import test from "node:test";
import { generateEventMotion } from "../app/demo-motion.ts";
import { compareSkeletonMotions } from "../app/motion-classifier.ts";
import { analyzeCameraContinuity, isExternalCamera, mergeCameraFrameStreams, selectExternalCameraSlots } from "../app/multi-camera.ts";
import { BODY_LANDMARK_COUNT, HAND_LANDMARK_COUNT, MOTION_FRAME_STRIDE } from "../app/pose-store.ts";

function frame(timeMs: number, { body = true, full = true, hands = 0, centerX = 0.5, facing = "front" } = {}) {
  const values = new Array(MOTION_FRAME_STRIDE).fill(Number.NaN);
  values[0] = timeMs;
  values[1] = Number(body);
  values[2] = Number(full);
  values[6] = Number(hands > 0);
  values[7] = Number(hands > 1);
  for (let index = 0; index < BODY_LANDMARK_COUNT; index += 1) {
    const offset = 10 + index * 4;
    values[offset] = centerX;
    values[offset + 1] = 0.5;
    values[offset + 2] = 0;
    values[offset + 3] = body ? 0.9 : 0;
  }
  values[10] = centerX + (facing === "front" ? 0.1 : -0.1);
  values[14] = centerX + (facing === "front" ? -0.1 : 0.1);
  return values;
}

test("two cameras become one owner timeline without duplicate frames", () => {
  const merged = mergeCameraFrameStreams([
    { cameraSlot: 1, frames: [frame(0), frame(100, { body: false }), frame(200, { full: false })] },
    { cameraSlot: 2, frames: [frame(100, { hands: 2, centerX: 0.8 }), frame(200, { centerX: 0.82 })] },
  ]);
  assert.deepEqual(merged.map((item) => item[0]), [0, 100, 200]);
  assert.equal(merged[1][7], 1, "the clearer second-camera observation should win");
  assert.equal(merged[2][2], 1, "full-body observation should beat a partial duplicate");
  assert.equal(merged[1][10 + 12 * 4], 0.5, "camera handoff should remove the image-coordinate jump");
});

test("camera handoff canonicalizes front/back shoulder orientation", () => {
  const merged = mergeCameraFrameStreams([
    { cameraSlot: 1, frames: [frame(0, { facing: "front" })] },
    { cameraSlot: 2, frames: [frame(100, { facing: "back" })] },
  ]);
  assert.ok(merged[0][10] > merged[0][14]);
  assert.ok(merged[1][10] > merged[1][14]);
});

test("developer continuity check reports a timely camera handoff", () => {
  const report = analyzeCameraContinuity([
    { cameraSlot: 1, frames: [frame(0), frame(100)] },
    { cameraSlot: 2, frames: [frame(200), frame(300)] },
  ]);
  assert.equal(report.status, "pass");
  assert.deepEqual(report.handoffs[0], { from: 1, to: 2, atMs: 200, gapMs: 100 });
});

test("developer continuity check warns on an eight-second blind gap", () => {
  const report = analyzeCameraContinuity([
    { cameraSlot: 1, frames: [frame(0), frame(100)] },
    { cameraSlot: 2, frames: [frame(9_000)] },
  ]);
  assert.equal(report.status, "warning");
  assert.equal(report.longestBlindGapMs, 8_800);
});

test("front/back horizontal reversal keeps the same motion distance", () => {
  const original = generateEventMotion("normal_task", 7);
  const mirrored = original.map((source) => {
    const next = [...source];
    next[3] = -next[3];
    for (let index = 0; index < BODY_LANDMARK_COUNT; index += 1) {
      next[10 + index * 4] = 1 - next[10 + index * 4];
    }
    const handsStart = 10 + BODY_LANDMARK_COUNT * 4;
    for (let index = 0; index < HAND_LANDMARK_COUNT * 2; index += 1) {
      next[handsStart + index * 3] = 1 - next[handsStart + index * 3];
    }
    return next;
  });
  assert.ok((compareSkeletonMotions(mirrored, original) ?? 1) < 1e-6);
});

test("external camera slots exclude the laptop camera and restore saved order", () => {
  const device = (deviceId: string, label: string, groupId = "") => ({ deviceId, label, groupId });
  const cameras = [
    device("laptop", "FaceTime HD Camera"),
    device("new-c", "Webcam C", "port-c"),
    device("new-a", "Webcam A", "port-a"),
    device("new-b", "Webcam B", "port-b"),
  ];
  const saved = [
    device("old-b", "Webcam B", "port-b"),
    device("old-a", "Webcam A", "port-a"),
    device("old-c", "Webcam C", "port-c"),
  ];
  assert.deepEqual(selectExternalCameraSlots(cameras, saved).map((camera) => camera.deviceId), ["new-b", "new-a", "new-c"]);
});

test("identical webcams keep their slots across app restarts", () => {
  const cameras = ["webcam-3", "webcam-1", "webcam-2"].map((deviceId) => ({
    deviceId,
    groupId: deviceId,
    label: "USB Camera",
  }));
  const saved = [cameras[2], cameras[0], cameras[1]];
  assert.deepEqual(selectExternalCameraSlots(cameras, saved).map((camera) => camera.deviceId), ["webcam-2", "webcam-3", "webcam-1"]);
});

test("a newly selected camera overrides the saved slot", () => {
  const cameras = ["3D5F", "camera-b", "camera-c"].map((deviceId) => ({ deviceId, groupId: deviceId, label: deviceId }));
  const saved = [cameras[0], cameras[1], cameras[2]];
  assert.deepEqual(
    selectExternalCameraSlots(cameras, saved, ["camera-c", "camera-b", "3D5F"]).map((camera) => camera.deviceId),
    ["camera-c", "camera-b", "3D5F"],
  );
});

test("built-in and Continuity cameras stay out of external-camera slots", () => {
  const camera = (label: string) => ({ deviceId: label, groupId: label, label });
  assert.equal(isExternalCamera(camera("‘서연’ 카메라")), false);
  assert.equal(isExternalCamera(camera("‘서연’ 데스크뵰 카메라")), false);
  assert.equal(isExternalCamera(camera("‘서연’ 데스크뷰 카메라")), false);
  assert.equal(isExternalCamera(camera("SNAP U2")), true);
});
