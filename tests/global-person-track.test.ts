import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_TUNING,
  EMPTY_GLOBAL_TRACK,
  GLOBAL_CONFIRMATION_FRAMES,
  GLOBAL_RELEASE_AFTER_MS,
  anatomyIsValid,
  jointAnalysisWeight,
  torsoFrame,
  updateGlobalPersonTrack,
  type CameraObservation,
  type GlobalPersonTrack,
} from "../app/global-person-track.ts";
import type { CameraSlot } from "../app/multi-camera.ts";
import type { NormalizedLandmarkLike } from "../app/motion-analysis.ts";

type Overrides = Partial<Record<number, Partial<NormalizedLandmarkLike>>>;

/** Upright person centred at `centerX`, torso length ~0.2 of the frame. */
function person(centerX = 0.5, overrides: Overrides = {}): NormalizedLandmarkLike[] {
  const base = (x: number, y: number, visible = 0.9) => ({ x, y, z: 0, visibility: visible });
  const points: NormalizedLandmarkLike[] = Array.from({ length: 33 }, () => base(centerX, 0.5, 0));
  // Left shoulder sits to the right of the right shoulder in a mirrored front view.
  points[11] = base(centerX + 0.1, 0.35);
  points[12] = base(centerX - 0.1, 0.35);
  points[13] = base(centerX + 0.14, 0.47);
  points[14] = base(centerX - 0.14, 0.47);
  points[15] = base(centerX + 0.16, 0.58);
  points[16] = base(centerX - 0.16, 0.58);
  points[23] = base(centerX + 0.07, 0.55);
  points[24] = base(centerX - 0.07, 0.55);
  points[25] = base(centerX + 0.07, 0.72);
  points[26] = base(centerX - 0.07, 0.72);
  points[27] = base(centerX + 0.07, 0.9);
  points[28] = base(centerX - 0.07, 0.9);
  for (const [index, patch] of Object.entries(overrides)) {
    points[Number(index)] = { ...points[Number(index)], ...patch };
  }
  return points;
}

function observation(
  cameraSlot: CameraSlot,
  timestamp: number,
  landmarks: NormalizedLandmarkLike[] | null,
  extra: Partial<CameraObservation> = {},
): CameraObservation {
  return {
    cameraSlot,
    timestamp,
    landmarks,
    personConfidence: landmarks ? 0.8 : 0,
    foregroundRatio: landmarks ? 0.1 : 0,
    confirmed: Boolean(landmarks),
    ...extra,
  };
}

function advance(
  track: GlobalPersonTrack,
  frames: CameraObservation[][],
  startMs = 0,
  stepMs = 100,
) {
  let current = track;
  frames.forEach((observations, index) => {
    const timestamp = startMs + index * stepMs;
    current = updateGlobalPersonTrack(
      current,
      observations.map((item) => ({ ...item, timestamp })),
      timestamp,
      () => "person-test",
    );
  });
  return current;
}

test("three cameras seeing the same person produce exactly one global track", () => {
  const track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [
    observation(1, 0, person(0.5)),
    observation(2, 0, person(0.3)),
    observation(3, 0, person(0.7)),
  ]));
  assert.equal(track.presence, "present");
  assert.equal(track.id, "person-test");
  assert.deepEqual(track.observingCameras, [1, 2, 3]);
  assert.equal(track.coordinateFrame !== null, true);
});

test("three-view capture reports a person only while all cameras have fresh confirmed skeletons", () => {
  const policy = { requiredCameraSlots: [1, 2, 3] as const, maxObservationAgeMs: 500 };
  const one = updateGlobalPersonTrack(EMPTY_GLOBAL_TRACK, [
    observation(1, 100, person()),
    observation(2, 100, null),
    observation(3, 100, null),
  ], 100, () => "person-test", DEFAULT_TUNING, policy);
  assert.equal(one.presence, "provisional");
  assert.deepEqual(one.observingCameras, [1]);

  const all = updateGlobalPersonTrack(one, [
    observation(1, 100, person()),
    observation(2, 300, person(0.3)),
    observation(3, 400, person(0.7)),
  ], 400, () => "person-test", DEFAULT_TUNING, policy);
  assert.equal(all.presence, "present");
  assert.deepEqual(all.observingCameras, [1, 2, 3]);

  const stale = updateGlobalPersonTrack(all, [
    observation(1, 100, person()),
    observation(2, 550, person(0.3)),
    observation(3, 600, person(0.7)),
  ], 650, () => "person-test", DEFAULT_TUNING, policy);
  assert.equal(stale.presence, "provisional", "a camera older than the sync window cannot keep person present");
  assert.deepEqual(stale.observingCameras, [2, 3]);
});

test("presence needs consecutive frames before it is reported as present", () => {
  let track = EMPTY_GLOBAL_TRACK;
  for (let frame = 1; frame < GLOBAL_CONFIRMATION_FRAMES; frame += 1) {
    track = advance(track, [[observation(1, 0, person())]], frame * 100);
    assert.equal(track.presence, "provisional", `frame ${frame} must stay provisional`);
  }
  track = advance(track, [[observation(1, 0, person())]], GLOBAL_CONFIRMATION_FRAMES * 100);
  assert.equal(track.presence, "present");
});

test("an empty room produces no track at all", () => {
  const track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 5 }, () => [
    observation(1, 0, null),
    observation(2, 0, null),
    observation(3, 0, null),
  ]));
  assert.equal(track.presence, "absent");
  assert.equal(track.joints.length, 0);
});

test("a joint hidden in the primary view is filled from another camera as inferred", () => {
  const occluded = person(0.5, { 15: { visibility: 0.02 }, 13: { visibility: 0.02 } });
  const track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [
    observation(1, 0, occluded),
    observation(2, 0, person(0.3), { personConfidence: 0.4 }),
  ]));
  const wrist = track.joints.find((joint) => joint.index === 15)!;
  assert.equal(wrist.state, "inferred");
  assert.equal(wrist.sourceCamera, 2);
  assert.ok(wrist.confidence < 0.9, "inferred confidence must be discounted");
  assert.ok(jointAnalysisWeight(wrist) < wrist.confidence);
});

test("a joint no camera sees is marked missing, never invented", () => {
  const noAnkles = person(0.5, { 27: { visibility: 0 }, 28: { visibility: 0 } });
  const track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [
    observation(1, 0, noAnkles),
    observation(2, 0, noAnkles),
  ]));
  const ankle = track.joints.find((joint) => joint.index === 27)!;
  assert.equal(ankle.state, "missing");
  assert.equal(Number.isNaN(ankle.x), true);
  assert.equal(jointAnalysisWeight(ankle), 0);
});

test("a person with no visible head is still tracked", () => {
  const headless = person(0.5);
  for (let index = 0; index <= 10; index += 1) headless[index] = { x: 0.5, y: 0, z: 0, visibility: 0 };
  assert.equal(anatomyIsValid(headless), true);
  const track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [observation(1, 0, headless)]));
  assert.equal(track.presence, "present");
});

test("shoulders and one arm alone are enough to hold the track", () => {
  const upperOnly = person(0.5);
  for (const index of [23, 24, 25, 26, 27, 28, 14, 16]) {
    upperOnly[index] = { ...upperOnly[index], visibility: 0 };
  }
  assert.ok(torsoFrame(upperOnly), "shoulder-only torso frame must resolve");
  const track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [observation(1, 0, upperOnly)]));
  assert.equal(track.presence, "present");
});

test("a grossly asymmetric skeleton is rejected as not-a-body", () => {
  const hanger = person(0.5, { 13: { x: 0.505, y: 0.36 }, 15: { x: 0.505, y: 0.365 } });
  assert.equal(anatomyIsValid(hanger), false);
  const track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [observation(1, 0, hanger)]));
  assert.equal(track.presence, "absent");
});

test("a brief dropout coasts instead of releasing the person", () => {
  let track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [observation(1, 0, person())]));
  const id = track.id;
  track = advance(track, [[observation(1, 0, null)]], 400);
  assert.equal(track.presence, "coasting");
  assert.equal(track.id, id, "a short dropout must not create a new person");
});

test("a long absence releases the track before a new one can be acquired", () => {
  let track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [observation(1, 0, person())]));
  track = updateGlobalPersonTrack(track, [observation(1, 0, null)], GLOBAL_RELEASE_AFTER_MS + 1_000);
  assert.equal(track.presence, "absent");
  assert.equal(track.id, "");
});

test("camera handoff is reported with its blind gap", () => {
  let track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [
    observation(1, 0, person(0.5)),
  ]));
  assert.equal(track.coordinateFrame, 1);
  // Camera 1 loses the person; camera 2 picks it up 200ms later.
  track = updateGlobalPersonTrack(track, [observation(2, 0, person(0.5))], 500);
  assert.equal(track.coordinateFrame, 2);
  assert.equal(track.handoffFromCamera, 1);
  assert.equal(track.handoffGapMs, 200);
  assert.equal(track.id, "person-test", "handoff must keep the same global person");
});

test("a teleporting body is rejected as a different person, not adopted", () => {
  let track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [
    observation(1, 0, person(0.2)),
  ]));
  // 0.6 normalized units in 33ms is far beyond a human walk at this torso scale.
  track = updateGlobalPersonTrack(track, [observation(1, 0, person(0.8))], 333);
  assert.equal(track.presence, "coasting", "the jump must not be accepted as the tracked person");
});

test("a camera with no person evidence at all cannot acquire a track", () => {
  const track = advance(EMPTY_GLOBAL_TRACK, Array.from({ length: 4 }, () => [
    observation(1, 0, person(), { personConfidence: 0, foregroundRatio: 0, confirmed: false }),
  ]));
  assert.equal(track.presence, "absent");
});

test("the anatomy floor is the value fitted on COCO-WholeBody, not a guess", () => {
  // Swept end to end in scripts/fit-tracking-thresholds.ts: 0.25 costs ~4.9
  // points of retention on real images, 0.05 lets empty-frame false positives
  // through. Changing this without re-running that sweep is a regression.
  assert.equal(DEFAULT_TUNING.limbAsymmetryFloor, 0.1);
});

test("a decoy with a collapsed limb is still rejected at the fitted floor", () => {
  const hanger = person(0.5, { 13: { x: 0.505, y: 0.36 }, 15: { x: 0.505, y: 0.365 } });
  assert.equal(anatomyIsValid(hanger), false);
});

test("a foreshortened but real limb survives the fitted floor", () => {
  // Arm pointed at the camera: the near limb projects to a third of the far
  // one. The old 0.25 floor threw this away.
  const foreshortened = person(0.5, { 13: { x: 0.56, y: 0.39 }, 15: { x: 0.575, y: 0.42 } });
  assert.equal(anatomyIsValid(foreshortened), true);
});

test("tuning is injectable so a backend can carry its own thresholds", () => {
  const hanger = person(0.5, { 13: { x: 0.505, y: 0.36 }, 15: { x: 0.505, y: 0.365 } });
  assert.equal(anatomyIsValid(hanger, { ...DEFAULT_TUNING, limbAsymmetryFloor: 0 }), true);
  const strict = { ...DEFAULT_TUNING, confirmationFrames: 5 };
  let track = EMPTY_GLOBAL_TRACK;
  for (let frame = 0; frame < 4; frame += 1) {
    track = updateGlobalPersonTrack(
      track, [observation(1, frame * 100, person())], frame * 100, () => "person-test", strict,
    );
  }
  assert.equal(track.presence, "provisional", "a higher confirmation count must delay presence");
});
