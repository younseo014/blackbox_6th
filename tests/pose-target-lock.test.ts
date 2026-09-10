import assert from "node:assert/strict";
import test from "node:test";
import {
  handBelongsToPose,
  matchHandToPose,
  selectLockedPose,
} from "../app/pose-target-lock.ts";
import type { NormalizedLandmarkLike } from "../app/motion-analysis.ts";

function point(x: number, y: number, z = 0, visibility = 0.95): NormalizedLandmarkLike {
  return { x, y, z, visibility };
}

function standingPose(centerX = 0.5, scale = 1): NormalizedLandmarkLike[] {
  const transform = (x: number, y: number) => point(
    centerX + (x - 0.5) * scale,
    0.52 + (y - 0.52) * scale,
  );
  const pose = Array.from({ length: 33 }, () => transform(0.5, 0.5));
  pose[0] = transform(0.5, 0.1);
  pose[7] = transform(0.46, 0.1);
  pose[8] = transform(0.54, 0.1);
  pose[11] = transform(0.41, 0.23);
  pose[12] = transform(0.59, 0.23);
  pose[13] = transform(0.38, 0.37);
  pose[14] = transform(0.62, 0.37);
  pose[15] = transform(0.36, 0.5);
  pose[16] = transform(0.64, 0.5);
  pose[23] = transform(0.44, 0.53);
  pose[24] = transform(0.56, 0.53);
  pose[25] = transform(0.44, 0.74);
  pose[26] = transform(0.56, 0.74);
  pose[27] = transform(0.44, 0.95);
  pose[28] = transform(0.56, 0.95);
  return pose;
}

function handAt(x: number, y: number): NormalizedLandmarkLike[] {
  return Array.from({ length: 21 }, (_, index) => point(x + index * 0.001, y));
}

test("target lock acquires the strongest centered full-body candidate", () => {
  const background = standingPose(0.86, 0.55);
  const owner = standingPose(0.5, 0.9);
  const selection = selectLockedPose([background, owner], null, 1000);
  assert.equal(selection.state, "acquired");
  assert.equal(selection.landmarks, owner);
  assert.ok(selection.lock);
});

test("target lock follows the same moving person instead of candidate array order", () => {
  const owner = standingPose(0.48, 0.9);
  const acquired = selectLockedPose([owner], null, 1000);
  const movedOwner = standingPose(0.54, 0.91);
  const background = standingPose(0.82, 0.65);
  const tracked = selectLockedPose([background, movedOwner], acquired.lock, 1066);
  assert.equal(tracked.state, "tracking");
  assert.equal(tracked.landmarks, movedOwner);
});

test("target lock reports missing instead of switching to a distant pose-like object", () => {
  const owner = standingPose(0.45, 0.9);
  const acquired = selectLockedPose([owner], null, 1000);
  const distantObject = standingPose(0.88, 0.42);
  const missing = selectLockedPose([distantObject], acquired.lock, 1066);
  assert.equal(missing.state, "missing");
  assert.equal(missing.landmarks, null);
  assert.equal(missing.lock, acquired.lock);
});

test("the single visible person is reacquired after a sustained tracking loss", () => {
  const owner = standingPose(0.42, 0.9);
  const acquired = selectLockedPose([owner], null, 1000);
  const repositionedOwner = standingPose(0.88, 0.42);
  const recovered = selectLockedPose([repositionedOwner], acquired.lock, 1700);
  assert.equal(recovered.state, "acquired");
  assert.equal(recovered.landmarks, repositionedOwner);
  assert.ok(recovered.lock);
});

test("target lock stays missing instead of merging with a similarly-placed but differently-built stranger", () => {
  const owner = standingPose(0.45, 0.9);
  const acquired = selectLockedPose([owner], null, 1000);
  // Same general position and scale as the owner (small center step, small
  // scale change) - the exact situation where slightly reframing the camera
  // lets a bystander land close to where the owner was standing. Only the
  // hip width is drastically different, simulating a different person's
  // build rather than the owner shifting stance.
  const stranger = standingPose(0.5, 0.88);
  stranger[23] = point(0.15, 0.53);
  stranger[24] = point(0.85, 0.53);
  const result = selectLockedPose([stranger], acquired.lock, 1066);
  assert.equal(result.state, "missing");
  assert.equal(result.landmarks, null);
  assert.equal(result.lock, acquired.lock);
});

test("cropped upper body is locked immediately before the user moves backward", () => {
  const cropped = standingPose();
  cropped[27] = point(0.44, 0.95, 0, 0.1);
  cropped[28] = point(0.56, 0.95, 0, 0.1);
  const selection = selectLockedPose([cropped], null, 1000);
  assert.equal(selection.state, "acquired");
  assert.equal(selection.landmarks, cropped);
  assert.ok(selection.lock);
});

test("shoulders and arms are enough to lock a true upper-body-only pose", () => {
  const upperBody = standingPose();
  for (let index = 23; index < 33; index += 1) {
    upperBody[index] = point(0.5, 1.2, 0, 0.05);
  }
  const selection = selectLockedPose([upperBody], null, 1000);
  assert.equal(selection.state, "acquired");
  assert.equal(selection.landmarks, upperBody);
  assert.ok(selection.lock);
});

test("one visible arm is enough to recognize and track a partial person", () => {
  const owner = standingPose();
  const acquired = selectLockedPose([owner], null, 1000);
  const armOnly = standingPose();
  for (const index of [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
    armOnly[index] = { ...armOnly[index], visibility: 0.05 };
  }
  for (const index of [11, 13, 15]) armOnly[index] = { ...owner[index], visibility: 0.8 };

  assert.equal(selectLockedPose([armOnly], null, 1066).landmarks, armOnly);
  assert.equal(selectLockedPose([armOnly], acquired.lock, 1066).landmarks, armOnly);
});

test("target lock follows the same person from close upper-body to farther full-body framing", () => {
  const close = standingPose(0.5, 1.05);
  close[27] = point(0.44, 0.95, 0, 0.1);
  close[28] = point(0.56, 0.95, 0, 0.1);
  const acquired = selectLockedPose([close], null, 1000);
  const farther = standingPose(0.51, 0.58);
  const background = standingPose(0.86, 0.48);
  const tracked = selectLockedPose([background, farther], acquired.lock, 1120);
  assert.equal(tracked.state, "tracking");
  assert.equal(tracked.landmarks, farther);
  assert.ok(tracked.lock);
});

test("head movement and missing limbs do not release the body-based target lock", () => {
  const owner = standingPose(0.5, 0.9);
  const acquired = selectLockedPose([owner], null, 1000);
  const cropped = standingPose(0.53, 0.9);
  for (let index = 0; index <= 10; index += 1) {
    cropped[index] = point(0.95, 0.05, 0, 0.05);
  }
  for (const index of [14, 16, 26, 28]) {
    cropped[index] = point(1.2, 0.5, 0, 0.05);
  }
  const tracked = selectLockedPose([cropped], acquired.lock, 1066);
  assert.equal(tracked.state, "tracking");
  assert.equal(tracked.landmarks, cropped);
});

test("upper-body rotation during a head-direction test keeps the same target", () => {
  const owner = standingPose(0.5, 0.9);
  const acquired = selectLockedPose([owner], null, 1000);
  const turned = standingPose(0.51, 0.9);
  turned[11] = point(0.49, 0.259);
  turned[12] = point(0.53, 0.259);
  for (let index = 0; index <= 10; index += 1) {
    turned[index] = point(0.68, 0.18, 0, 0.8);
  }
  const tracked = selectLockedPose([turned], acquired.lock, 1066);
  assert.equal(tracked.state, "tracking");
  assert.equal(tracked.landmarks, turned);
  assert.ok(tracked.lock);
});

test("hand filtering only attaches hands near the locked person's wrists", () => {
  const owner = standingPose();
  assert.equal(handBelongsToPose(handAt(0.36, 0.5), owner), true);
  assert.equal(handBelongsToPose(handAt(0.9, 0.15), owner), false);
});

test("hand association ignores incompatible z scales from separate vision models", () => {
  const owner = standingPose();
  const leftHand = handAt(0.36, 0.5).map((landmark) => ({ ...landmark, z: 2.5 }));
  const rightHand = handAt(0.64, 0.5).map((landmark) => ({ ...landmark, z: -1.8 }));
  assert.equal(matchHandToPose(leftHand, owner), "left");
  assert.equal(matchHandToPose(rightHand, owner), "right");
});
