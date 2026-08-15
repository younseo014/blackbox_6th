import assert from "node:assert/strict";
import test from "node:test";
import {
  BODY_LANDMARK_COUNT,
  HAND_LANDMARK_COUNT,
  MOTION_FRAME_STRIDE,
  deriveBodyProportionProfile,
  parseSessionFrame,
} from "../app/pose-store.ts";
import { generateEventMotion } from "../app/demo-motion.ts";

// Builds a synthetic raw frame matching the stored layout (see
// MOTION_FRAME_STRIDE / downloadMotionSession's frame_layout schema), so
// parseSessionFrame - the translator the session-replay viewer depends on -
// can be checked without needing IndexedDB or a browser.
function buildRawFrame(overrides: {
  relativeTimeMs?: number;
  bodyDetected?: 0 | 1;
  fullBodyVisible?: 0 | 1;
  headYaw?: number;
  headPitch?: number;
  headRoll?: number;
  leftHandDetected?: 0 | 1;
  rightHandDetected?: 0 | 1;
  leftHandConfidence?: number;
  rightHandConfidence?: number;
} = {}): number[] {
  const frame = new Array(MOTION_FRAME_STRIDE).fill(0);
  frame[0] = overrides.relativeTimeMs ?? 1200;
  frame[1] = overrides.bodyDetected ?? 1;
  frame[2] = overrides.fullBodyVisible ?? 1;
  frame[3] = overrides.headYaw ?? 0.1;
  frame[4] = overrides.headPitch ?? -0.05;
  frame[5] = overrides.headRoll ?? 0.02;
  frame[6] = overrides.leftHandDetected ?? 1;
  frame[7] = overrides.rightHandDetected ?? 0;
  frame[8] = overrides.leftHandConfidence ?? 0.9;
  frame[9] = overrides.rightHandConfidence ?? 0;

  // Mark the body block with a recognizable value so slicing can be checked.
  const bodyStart = 10;
  for (let i = 0; i < BODY_LANDMARK_COUNT * 4; i += 1) frame[bodyStart + i] = 100 + i;

  const leftHandStart = bodyStart + BODY_LANDMARK_COUNT * 4;
  for (let i = 0; i < HAND_LANDMARK_COUNT * 3; i += 1) frame[leftHandStart + i] = 200 + i;

  const rightHandStart = leftHandStart + HAND_LANDMARK_COUNT * 3;
  for (let i = 0; i < HAND_LANDMARK_COUNT * 3; i += 1) frame[rightHandStart + i] = 300 + i;

  return frame;
}

test("deriveBodyProportionProfile: extracts anonymous ratios from full-body coordinate frames", () => {
  const frames = generateEventMotion("normal_task", 4);
  const profile = deriveBodyProportionProfile(frames, "recent-local-session", 1234);
  assert.ok(profile);
  assert.equal(profile?.sourceSessionId, "recent-local-session");
  assert.ok((profile?.usableFrames ?? 0) >= 3);
  assert.ok((profile?.shoulderToTorso ?? 0) > 0.45);
  assert.ok((profile?.thighToTorso ?? 0) > 0.65);
  assert.ok((profile?.handToForearm ?? 0) >= 0.25);
});

test("parseSessionFrame: reads header fields (time, detection flags, head direction)", () => {
  const parsed = parseSessionFrame(buildRawFrame({ relativeTimeMs: 4400, headYaw: 0.3, headPitch: -0.2, headRoll: 0.05 }));
  assert.equal(parsed.relativeTimeMs, 4400);
  assert.equal(parsed.bodyDetected, true);
  assert.equal(parsed.fullBodyVisible, true);
  assert.deepEqual(parsed.head, { yaw: 0.3, pitch: -0.2, roll: 0.05 });
});

test("parseSessionFrame: head is null when the body wasn't detected that frame", () => {
  const parsed = parseSessionFrame(buildRawFrame({ bodyDetected: 0 }));
  assert.equal(parsed.head, null);
});

test("parseSessionFrame: extracts the correct body slice (22 landmarks * 4)", () => {
  const parsed = parseSessionFrame(buildRawFrame());
  assert.equal(parsed.body.length, BODY_LANDMARK_COUNT * 4);
  assert.equal(parsed.body[0], 100);
  assert.equal(parsed.body[parsed.body.length - 1], 100 + BODY_LANDMARK_COUNT * 4 - 1);
});

test("parseSessionFrame: left hand present, right hand null when undetected", () => {
  const parsed = parseSessionFrame(buildRawFrame({ leftHandDetected: 1, rightHandDetected: 0 }));
  assert.ok(parsed.leftHand);
  assert.equal(parsed.leftHand!.length, HAND_LANDMARK_COUNT * 3);
  assert.equal(parsed.leftHand![0], 200);
  assert.equal(parsed.rightHand, null);
});

test("parseSessionFrame: both hands present when both detected", () => {
  const parsed = parseSessionFrame(buildRawFrame({ leftHandDetected: 1, rightHandDetected: 1 }));
  assert.ok(parsed.leftHand);
  assert.ok(parsed.rightHand);
  assert.equal(parsed.rightHand![0], 300);
});
