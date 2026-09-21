import assert from "node:assert/strict";
import test from "node:test";
import {
  DETECTOR_INTERVAL_MS,
  EMPTY_SCHEDULER_STATE,
  POSE_INTERVAL_MS,
  cameraTier,
  detectorIntervalFor,
  poseIntervalFor,
  selectCameraForInference,
  type CameraActivity,
  type SchedulerState,
} from "../app/pose-scheduler.ts";
import { GLOBAL_CONFIRMATION_FRAMES } from "../app/global-person-track.ts";
import type { CameraSlot } from "../app/multi-camera.ts";

function activity(cameraSlot: CameraSlot, overrides: Partial<CameraActivity> = {}): CameraActivity {
  return {
    cameraSlot,
    lastPersonSeenAt: -Infinity,
    lastForegroundAt: -Infinity,
    lastInferenceAt: -Infinity,
    isPrimary: false,
    ready: true,
    ...overrides,
  };
}

test("an idle camera watching an empty room runs least often", () => {
  const now = 10_000;
  assert.equal(cameraTier(activity(1), now), "idle");
  assert.equal(poseIntervalFor(activity(1), now), POSE_INTERVAL_MS.idle);
});

test("foreground change immediately raises a quiet camera's cadence", () => {
  const now = 10_000;
  const alerted = activity(1, { lastForegroundAt: now - 200 });
  assert.equal(cameraTier(alerted, now), "alerted");
  assert.ok(poseIntervalFor(alerted, now) < POSE_INTERVAL_MS.idle);
});

test("the camera holding the global track gets the highest frame rate", () => {
  const now = 10_000;
  const primary = activity(2, { isPrimary: true, lastPersonSeenAt: now });
  const secondary = activity(3, { lastPersonSeenAt: now });
  assert.equal(poseIntervalFor(primary, now), POSE_INTERVAL_MS.primary);
  assert.ok(poseIntervalFor(secondary, now) > poseIntervalFor(primary, now));
  assert.ok(1000 / POSE_INTERVAL_MS.primary <= 10, "body inference must stay within the 5-10 FPS budget");
  assert.ok(1000 / POSE_INTERVAL_MS.secondary >= 5);
});

test("at most one camera infers per tick", () => {
  const now = 10_000;
  const result = selectCameraForInference(
    EMPTY_SCHEDULER_STATE,
    [activity(1, { isPrimary: true }), activity(2), activity(3)],
    now,
  );
  assert.equal(typeof result.slot, "number");
  assert.equal([1, 2, 3].includes(result.slot!), true);
});

test("no camera runs before its own interval has elapsed", () => {
  const now = 10_000;
  const result = selectCameraForInference(
    EMPTY_SCHEDULER_STATE,
    [
      activity(1, { isPrimary: true, lastInferenceAt: now - 10 }),
      activity(2, { lastInferenceAt: now - 10 }),
      activity(3, { lastInferenceAt: now - 10 }),
    ],
    now,
  );
  assert.equal(result.slot, null);
});

test("a disconnected camera is never scheduled", () => {
  const result = selectCameraForInference(
    EMPTY_SCHEDULER_STATE,
    [activity(1, { ready: false }), activity(2, { ready: false }), activity(3, { ready: false })],
    10_000,
  );
  assert.equal(result.slot, null);
});

test("equal claims rotate instead of starving the higher slots", () => {
  let state: SchedulerState = EMPTY_SCHEDULER_STATE;
  const picked: CameraSlot[] = [];
  for (let tick = 0; tick < 3; tick += 1) {
    const now = 10_000 + tick;
    const result = selectCameraForInference(
      state,
      [activity(1), activity(2), activity(3)].map((item) => ({
        ...item,
        lastInferenceAt: picked.includes(item.cameraSlot) ? now : -Infinity,
      })),
      now,
    );
    state = result.state;
    if (result.slot) picked.push(result.slot);
  }
  assert.deepEqual([...picked].sort(), [1, 2, 3]);
});

test("the person detector backs off once somebody is tracked", () => {
  assert.equal(detectorIntervalFor(false), DETECTOR_INTERVAL_MS.searching);
  assert.ok(detectorIntervalFor(true) > detectorIntervalFor(false));
});

test("an idle camera can still recognise an entry inside the 1s budget", () => {
  // Worst case: entry lands just after an idle poll, so we pay one idle
  // interval to notice the foreground change, then two alerted intervals to
  // reach GLOBAL_CONFIRMATION_FRAMES.
  const worstCaseMs = POSE_INTERVAL_MS.idle + POSE_INTERVAL_MS.alerted * (GLOBAL_CONFIRMATION_FRAMES - 1);
  assert.ok(worstCaseMs <= 1_000, `worst-case recognition would take ${worstCaseMs}ms`);
});
