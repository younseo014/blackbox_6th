import assert from "node:assert/strict";
import test from "node:test";
import { generateEventMotion, DEMO_MOTION_LABELS, DEMO_SAMPLE_RATE, type DemoMotionType } from "../app/demo-motion.ts";
import { parseSessionFrame } from "../app/pose-store.ts";
import { computeHandMotionVariability, computeMovementSmoothness, type Point3D } from "../app/motion-analysis.ts";

// These synthetic clips are what a developer sees when replaying a demo
// persona's flagged event ("마감 반복 확인", "미세 지연", ...). The point of
// these tests isn't pixel-perfect choreography - it's confirming each
// motion type actually carries the qualitative signature it's supposed to
// represent, using the *same* real analysis functions the app uses.

const ALL_TYPES: DemoMotionType[] = [
  "double_check",
  "micro_delay",
  "safety_alert",
  "normal_task",
  "fine_hand_task",
  "register_tap",
  "queue_shift",
  "high_reach",
  "low_bend",
];

function rightWristPoints(rawFrames: number[][]): Array<Point3D | null> {
  return rawFrames.map(parseSessionFrame).map((frame) =>
    frame.rightHand ? { x: frame.rightHand[0], y: frame.rightHand[1], z: frame.rightHand[2] } : null,
  );
}

function rollingVariabilityAt(points: Array<Point3D | null>, index: number, window = 6): number {
  const start = Math.max(0, index - window + 1);
  return computeHandMotionVariability(points.slice(start, index + 1)) ?? 0;
}

test("generateEventMotion: every type produces a non-empty, well-formed clip", () => {
  for (const type of ALL_TYPES) {
    const frames = generateEventMotion(type, 0);
    assert.ok(frames.length > 10, `${type} should generate more than a handful of frames`);
    for (const frame of frames) {
      assert.ok(frame.every((value) => Number.isFinite(value)), `${type} frame has a non-finite value`);
    }
  }
});

test("generateEventMotion: relative_time_ms advances monotonically at the sample interval", () => {
  const frames = generateEventMotion("normal_task", 0);
  const stepMs = 1000 / DEMO_SAMPLE_RATE;
  for (let i = 1; i < frames.length; i += 1) {
    assert.ok(
      Math.abs(frames[i][0] - frames[i - 1][0] - stepMs) < 1e-6,
      `expected a constant ${stepMs}ms step at frame ${i}`,
    );
  }
});

test("generateEventMotion: parses cleanly and both hands are marked detected", () => {
  const frames = generateEventMotion("double_check", 0);
  const parsed = frames.map(parseSessionFrame);
  for (const frame of parsed) {
    assert.equal(frame.bodyDetected, true);
    assert.ok(frame.leftHand);
    assert.ok(frame.rightHand);
  }
});

test("generateEventMotion: deterministic - same type+seed always returns the same clip", () => {
  const a = generateEventMotion("safety_alert", 3);
  const b = generateEventMotion("safety_alert", 3);
  assert.deepEqual(a, b);
});

test("generateEventMotion: different seeds vary the clip slightly (seed actually has an effect)", () => {
  const a = generateEventMotion("double_check", 1);
  const b = generateEventMotion("double_check", 5);
  assert.notDeepEqual(a, b);
});

// -- double_check: two separated approach-and-retreat events --------------

test("double_check motion: quiet in the middle, active during the two check moments", () => {
  const frames = generateEventMotion("double_check", 0);
  const points = rightWristPoints(frames);
  const midIndex = Math.round(frames.length * 0.55);
  const firstCheckIndex = Math.round(frames.length * 0.2);
  const secondCheckIndex = Math.round(frames.length * 0.8);

  const midVariability = rollingVariabilityAt(points, midIndex);
  const firstCheckVariability = rollingVariabilityAt(points, firstCheckIndex);
  const secondCheckVariability = rollingVariabilityAt(points, secondCheckIndex);

  assert.ok(
    midVariability < firstCheckVariability && midVariability < secondCheckVariability,
    `expected a lull between the two checks (mid=${midVariability}, checks=${firstCheckVariability}/${secondCheckVariability})`,
  );
});

// -- micro_delay: a frozen (near-zero-velocity) middle segment ------------

test("micro_delay motion: freezes in the middle, moves before and after", () => {
  const frames = generateEventMotion("micro_delay", 0);
  const points = rightWristPoints(frames);
  const frozenIndex = Math.round(frames.length * 0.6);
  // The ramp itself only spans t=0..0.12 (a handful of frames at this
  // clip's short duration), too few samples for computeHandMotionVariability
  // (needs >=4 speed samples) to say anything on their own. Its ROLLING
  // window looks backward, so an index placed just after each boundary
  // still has the ramp frames in view and picks up the speed-change they
  // produced - that's what actually distinguishes "there was motion here"
  // from the flat freeze, not a literal in-ramp/in-freeze split.
  const rampUpIndex = Math.round(frames.length * 0.2);
  const rampDownIndex = Math.round(frames.length * 0.9);

  const frozenVariability = rollingVariabilityAt(points, frozenIndex);
  const rampUpVariability = rollingVariabilityAt(points, rampUpIndex);
  const rampDownVariability = rollingVariabilityAt(points, rampDownIndex);

  assert.ok(frozenVariability < 1e-6, `expected ~0 variability during the freeze, got ${frozenVariability}`);
  assert.ok(rampUpVariability > 0, "expected motion before the freeze");
  assert.ok(rampDownVariability > 0, "expected motion after the freeze resumes");
});

// -- safety_alert: fast/jerky vs a steady baseline task -------------------

test("safety_alert motion: jerkier (less smooth) than a routine task", () => {
  const alertPoints = rightWristPoints(generateEventMotion("safety_alert", 0));
  const taskPoints = rightWristPoints(generateEventMotion("normal_task", 0));
  const alertSmoothness = computeMovementSmoothness(alertPoints)!;
  const taskSmoothness = computeMovementSmoothness(taskPoints)!;
  assert.ok(
    alertSmoothness > taskSmoothness,
    `expected a reactive safety-alert motion (${alertSmoothness}) to be jerkier than a routine task (${taskSmoothness})`,
  );
});

// -- register_tap: a repeated-tap burst that stalls mid-sequence ----------

test("register_tap motion: taps repeatedly, then stalls mid-tap (not back at rest)", () => {
  const frames = generateEventMotion("register_tap", 0);
  const points = rightWristPoints(frames);
  const tapPhaseIndex = Math.round(frames.length * 0.2);
  const stallIndex = Math.round(frames.length * 0.8);

  const tapVariability = rollingVariabilityAt(points, tapPhaseIndex);
  const stallVariability = rollingVariabilityAt(points, stallIndex);

  assert.ok(tapVariability > 0, "expected visible motion during the tapping burst");
  assert.ok(
    stallVariability < 1e-6,
    `expected the stall to be a true hold (no motion), got ${stallVariability}`,
  );
});

// -- high_reach: both hands move well away from resting position ----------

test("high_reach motion: raises both hands well above their resting height", () => {
  const frames = generateEventMotion("high_reach", 0).map(parseSessionFrame);
  const peak = frames[Math.round(frames.length * 0.32)];
  const rest = frames[0];
  assert.ok(peak.rightHand && rest.rightHand, "expected both hands detected");
  assert.ok(peak.leftHand && rest.leftHand, "expected both hands detected");
  // Smaller y = higher on screen; the reach should move the hands well up.
  assert.ok(
    peak.rightHand![1] < rest.rightHand![1] - 0.15,
    "expected the right hand to reach noticeably higher than rest",
  );
  assert.ok(
    peak.leftHand![1] < rest.leftHand![1] - 0.15,
    "expected the left hand to reach noticeably higher than rest",
  );
});

// -- low_bend: the whole body crouches, not just the arm ------------------

test("low_bend motion: crouches (lowers the lower body) at the peak of the bend", () => {
  const frames = generateEventMotion("low_bend", 0).map(parseSessionFrame);
  const peak = frames[Math.round(frames.length * 0.5)];
  const rest = frames[0];
  // storedIndex 13 (right_hip) is body slice index 13 -> flat offset 13*4.
  const hipY = (frame: (typeof frames)[number]) => frame.body[13 * 4 + 1];
  assert.ok(
    hipY(peak) > hipY(rest) + 0.02,
    `expected the hips to visibly lower during the bend (rest=${hipY(rest)}, peak=${hipY(peak)})`,
  );
});

test("DEMO_MOTION_LABELS: every motion type has a non-empty Korean label", () => {
  for (const type of ALL_TYPES) {
    assert.ok(DEMO_MOTION_LABELS[type] && DEMO_MOTION_LABELS[type].length > 0);
  }
});
