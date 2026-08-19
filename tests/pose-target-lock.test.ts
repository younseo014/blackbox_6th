import assert from "node:assert/strict";
import test from "node:test";
import {
  handBelongsToPose,
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

test("target lock is not acquired from a cropped partial-body candidate", () => {
  const cropped = standingPose();
  cropped[27] = point(0.44, 0.95, 0, 0.1);
  cropped[28] = point(0.56, 0.95, 0, 0.1);
  const selection = selectLockedPose([cropped], null, 1000);
  assert.equal(selection.state, "searching");
  assert.equal(selection.lock, null);
});

test("hand filtering only attaches hands near the locked person's wrists", () => {
  const owner = standingPose();
  assert.equal(handBelongsToPose(handAt(0.36, 0.5), owner), true);
  assert.equal(handBelongsToPose(handAt(0.9, 0.15), owner), false);
});
