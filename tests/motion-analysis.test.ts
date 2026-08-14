import assert from "node:assert/strict";
import test from "node:test";
import {
  isFullBodyVisible,
  getHeadDirection,
  describeHeadDirection,
  computeHandMotionVariability,
  computeMovementSmoothness,
  approximateHeadCenterFromShoulders,
  type NormalizedLandmarkLike,
} from "../app/motion-analysis.ts";

function landmark(x: number, y: number, z = 0, visibility = 1): NormalizedLandmarkLike {
  return { x, y, z, visibility };
}

// A minimal but complete 33-point body pose, standing centered in frame,
// fully visible head-to-toe. Indices follow MediaPipe Pose's layout.
function standingPose(): NormalizedLandmarkLike[] {
  const points: NormalizedLandmarkLike[] = new Array(33).fill(null).map(() =>
    landmark(0.5, 0.5, 0, 1),
  );
  points[0] = landmark(0.5, 0.125); // nose, level with the ears (facing forward)
  points[7] = landmark(0.46, 0.12); // left ear
  points[8] = landmark(0.54, 0.12); // right ear
  points[11] = landmark(0.42, 0.22); // left shoulder
  points[12] = landmark(0.58, 0.22); // right shoulder
  points[23] = landmark(0.44, 0.55); // left hip
  points[24] = landmark(0.56, 0.55); // right hip
  points[25] = landmark(0.44, 0.75); // left knee
  points[26] = landmark(0.56, 0.75); // right knee
  points[27] = landmark(0.44, 0.95); // left ankle
  points[28] = landmark(0.56, 0.95); // right ankle
  return points;
}

// -- Persona 3: 풀바디가 카메라에 잡히지 않는 매장 환경 -----------------

test("isFullBodyVisible: true for a fully framed standing pose", () => {
  assert.equal(isFullBodyVisible(standingPose()), true);
});

test("isFullBodyVisible: false when the ankles are cropped out of frame", () => {
  const pose = standingPose();
  pose[27] = landmark(0.44, 0.95, 0, 0.1); // low visibility ankle (cropped)
  pose[28] = landmark(0.56, 0.95, 0, 0.1);
  assert.equal(isFullBodyVisible(pose), false);
});

test("isFullBodyVisible: false when the subject is too close to the camera (short apparent height)", () => {
  const pose = standingPose();
  // Head and ankle too close together vertically -> fails the height check.
  pose[7] = landmark(0.46, 0.4);
  pose[8] = landmark(0.54, 0.4);
  pose[27] = landmark(0.44, 0.6);
  pose[28] = landmark(0.56, 0.6);
  assert.equal(isFullBodyVisible(pose), false);
});

test("isFullBodyVisible: false when a required landmark is missing/undefined", () => {
  const pose = standingPose();
  // @ts-expect-error intentionally simulating a dropped landmark
  pose[26] = undefined;
  assert.equal(isFullBodyVisible(pose), false);
});

// -- Head direction -------------------------------------------------------

test("getHeadDirection: returns null when ears aren't confidently visible", () => {
  const pose = standingPose();
  pose[7] = landmark(0.46, 0.12, 0, 0.1);
  pose[8] = landmark(0.54, 0.12, 0, 0.1);
  assert.equal(getHeadDirection(pose), null);
});

test("getHeadDirection + describeHeadDirection: facing forward", () => {
  const head = getHeadDirection(standingPose());
  assert.ok(head);
  assert.equal(describeHeadDirection(head), "정면을 보는 중");
});

test("describeHeadDirection: looking down when pitch is strongly positive", () => {
  assert.equal(
    describeHeadDirection({ yaw: 0, pitch: 0.5, roll: 0, centerX: 0.5, centerY: 0.1 }),
    "아래를 보는 중",
  );
});

test("describeHeadDirection: null head direction is reported as unconfirmed", () => {
  assert.equal(describeHeadDirection(null), "머리 방향 미확인");
});

// -- Hand motion signals (explicitly NOT clinical tremor detection) -------

test("computeHandMotionVariability: null when there isn't enough data", () => {
  assert.equal(computeHandMotionVariability([landmark(0.5, 0.5, 0)]), null);
});

test("computeHandMotionVariability: near zero for a nearly still hand", () => {
  const frames = Array.from({ length: 10 }, (_, i) =>
    landmark(0.5 + i * 0.0001, 0.5, 0),
  );
  const variability = computeHandMotionVariability(frames);
  assert.ok(variability !== null);
  assert.ok(variability! < 0.5, `expected low variability, got ${variability}`);
});

test("computeHandMotionVariability: higher for an erratic hand than a steady one", () => {
  const steady = Array.from({ length: 12 }, (_, i) => landmark(0.5 + i * 0.01, 0.5, 0));
  const erratic = Array.from({ length: 12 }, (_, i) =>
    landmark(0.5 + (i % 2 === 0 ? 0.02 : -0.02) * (i + 1), 0.5, 0),
  );
  const steadyScore = computeHandMotionVariability(steady)!;
  const erraticScore = computeHandMotionVariability(erratic)!;
  assert.ok(
    erraticScore > steadyScore,
    `expected erratic (${erraticScore}) > steady (${steadyScore})`,
  );
});

test("computeHandMotionVariability: skips null (undetected) frames rather than throwing", () => {
  const frames = [landmark(0.5, 0.5, 0), null, landmark(0.51, 0.5, 0), null, landmark(0.52, 0.5, 0), landmark(0.53, 0.5, 0)];
  assert.doesNotThrow(() => computeHandMotionVariability(frames));
});

test("computeMovementSmoothness: near zero for constant-velocity motion", () => {
  const frames = Array.from({ length: 10 }, (_, i) => landmark(0.5 + i * 0.01, 0.5, 0));
  const smoothness = computeMovementSmoothness(frames);
  assert.ok(smoothness !== null);
  assert.ok(smoothness! < 0.001, `expected near-zero jerk, got ${smoothness}`);
});

test("computeMovementSmoothness: higher for jerky/inconsistent motion", () => {
  const smooth = Array.from({ length: 10 }, (_, i) => landmark(0.5 + i * 0.01, 0.5, 0));
  const jerky = [0.5, 0.52, 0.5, 0.55, 0.49, 0.6, 0.48, 0.62, 0.47, 0.63].map((x) =>
    landmark(x, 0.5, 0),
  );
  const smoothScore = computeMovementSmoothness(smooth)!;
  const jerkyScore = computeMovementSmoothness(jerky)!;
  assert.ok(jerkyScore > smoothScore, `expected jerky (${jerkyScore}) > smooth (${smoothScore})`);
});

// -- approximateHeadCenterFromShoulders (session-replay head placement) --

// flatBody layout: [x,y,z,visibility] per landmark, index 0 = left shoulder,
// index 1 = right shoulder (see BODY_LANDMARK_NAMES in pose-store.ts).
function flatBodyWithShoulders(
  left: { x: number; y: number; visibility: number },
  right: { x: number; y: number; visibility: number },
): number[] {
  const flat = new Array(22 * 4).fill(0);
  flat[0] = left.x;
  flat[1] = left.y;
  flat[3] = left.visibility;
  flat[4] = right.x;
  flat[5] = right.y;
  flat[7] = right.visibility;
  return flat;
}

test("approximateHeadCenterFromShoulders: places the head above the shoulder midpoint when both shoulders are visible", () => {
  const flat = flatBodyWithShoulders(
    { x: 0.42, y: 0.3, visibility: 1 },
    { x: 0.58, y: 0.3, visibility: 1 },
  );
  const center = approximateHeadCenterFromShoulders(flat);
  assert.ok(center);
  assert.ok(Math.abs(center!.x - 0.5) < 1e-6, `expected x centered between shoulders, got ${center!.x}`);
  assert.ok(center!.y < 0.3, "expected the approximated head center to sit above the shoulder line");
});

test("approximateHeadCenterFromShoulders: null when a shoulder isn't confidently visible", () => {
  const flat = flatBodyWithShoulders(
    { x: 0.42, y: 0.3, visibility: 0.1 },
    { x: 0.58, y: 0.3, visibility: 1 },
  );
  assert.equal(approximateHeadCenterFromShoulders(flat), null);
});
