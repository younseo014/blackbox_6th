import assert from "node:assert/strict";
import test from "node:test";
import {
  detectMotionEvents,
  motionSamplesFromRawFrames,
  FREEZE_MIN_DURATION_MS,
  AWAY_DISTANCE_THRESHOLD,
  type MotionSample,
} from "../app/motion-detection.ts";
import { generateEventMotion, type DemoMotionType } from "../app/demo-motion.ts";

// This is the piece the user specifically asked for: the app should decide
// "was this abnormal?" FROM the skeleton motion itself, not from a
// hand-typed label. These tests build small synthetic trajectories by hand
// (no dependency on demo-motion's generators) so the detector's own logic
// is under test in isolation, then separately confirm it actually agrees
// with what each demo-motion generator was designed to represent.

const STEP_MS = 200; // matches MOTION_SAMPLE_RATE (5Hz)

function samplesFrom(points: Array<{ x: number; y: number } | null>): MotionSample[] {
  return points.map((point, index) => ({
    point: point ? { x: point.x, y: point.y, z: 0 } : null,
    relativeTimeMs: index * STEP_MS,
  }));
}

test("detectMotionEvents: a hand that never moves produces no events", () => {
  const points = Array.from({ length: 20 }, () => ({ x: 0.5, y: 0.5 }));
  assert.deepEqual(detectMotionEvents(samplesFrom(points)), []);
});

test("detectMotionEvents: micro_delay - moves, freezes past the duration threshold, resumes", () => {
  const points: Array<{ x: number; y: number }> = [];
  // Ramp away from rest (real motion).
  for (let i = 0; i <= 5; i += 1) points.push({ x: 0.5 + i * 0.02, y: 0.5 });
  // Frozen for well over FREEZE_MIN_DURATION_MS.
  const freezeSteps = Math.ceil(FREEZE_MIN_DURATION_MS / STEP_MS) + 3;
  for (let i = 0; i < freezeSteps; i += 1) points.push({ x: points[points.length - 1].x, y: 0.5 });
  // Ramp again (real motion resumes).
  for (let i = 1; i <= 5; i += 1) points.push({ x: points[points.length - 1].x + i * 0.02, y: 0.5 });

  const events = detectMotionEvents(samplesFrom(points));
  assert.equal(events.length, 1);
  assert.equal(events[0].type, "micro_delay");
  assert.match(events[0].evidence, /\d/, "evidence should cite an actual measured number");
});

test("detectMotionEvents: a brief pause under the freeze threshold is NOT flagged", () => {
  const points: Array<{ x: number; y: number }> = [];
  for (let i = 0; i <= 5; i += 1) points.push({ x: 0.5 + i * 0.02, y: 0.5 });
  // Only 2 frozen steps (400ms) - well short of FREEZE_MIN_DURATION_MS.
  points.push({ x: points[points.length - 1].x, y: 0.5 });
  points.push({ x: points[points.length - 1].x, y: 0.5 });
  for (let i = 1; i <= 5; i += 1) points.push({ x: points[points.length - 1].x + i * 0.02, y: 0.5 });

  const events = detectMotionEvents(samplesFrom(points));
  assert.equal(events.filter((e) => e.type === "micro_delay").length, 0);
});

test("detectMotionEvents: standing still the whole time is NOT a hesitation (no motion to hesitate from)", () => {
  const points: Array<{ x: number; y: number }> = [];
  const freezeSteps = Math.ceil(FREEZE_MIN_DURATION_MS / STEP_MS) + 5;
  for (let i = 0; i < freezeSteps; i += 1) points.push({ x: 0.5, y: 0.5 });
  const events = detectMotionEvents(samplesFrom(points));
  assert.equal(events.filter((e) => e.type === "micro_delay").length, 0);
});

test("detectMotionEvents: double_check - goes away from start, fully returns, goes away again", () => {
  const start = { x: 0.4, y: 0.5 };
  const away = { x: 0.4 + AWAY_DISTANCE_THRESHOLD * 3, y: 0.5 };
  const points: Array<{ x: number; y: number }> = [];
  const hold = (point: { x: number; y: number }, steps: number) => {
    for (let i = 0; i < steps; i += 1) points.push(point);
  };
  hold(start, 3);
  hold(away, Math.ceil(600 / STEP_MS) + 2); // sustained "away" episode 1
  hold(start, Math.ceil(600 / STEP_MS) + 2); // sustained "near" return
  hold(away, Math.ceil(600 / STEP_MS) + 2); // sustained "away" episode 2
  hold(start, 3);

  const events = detectMotionEvents(samplesFrom(points));
  const doubleChecks = events.filter((e) => e.type === "double_check");
  assert.equal(doubleChecks.length, 1);
  assert.match(doubleChecks[0].evidence, /2번/);
});

test("detectMotionEvents: going away once and staying there is NOT a double_check", () => {
  const start = { x: 0.4, y: 0.5 };
  const away = { x: 0.4 + AWAY_DISTANCE_THRESHOLD * 3, y: 0.5 };
  const points: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < 3; i += 1) points.push(start);
  for (let i = 0; i < 15; i += 1) points.push(away);
  const events = detectMotionEvents(samplesFrom(points));
  assert.equal(events.filter((e) => e.type === "double_check").length, 0);
});

test("detectMotionEvents: a sharp jerk above the safety threshold is flagged", () => {
  // A slow drift, then one huge single-step jump, then slow drift again -
  // a large, sudden change in speed (jerk), not a gradual acceleration.
  const points: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < 5; i += 1) points.push({ x: 0.5 + i * 0.001, y: 0.5 });
  points.push({ x: 1.3, y: 1.3 }); // sudden large jump
  for (let i = 0; i < 5; i += 1) points.push({ x: 1.3 + i * 0.001, y: 1.3 });
  const events = detectMotionEvents(samplesFrom(points));
  assert.ok(events.some((e) => e.type === "safety_alert"));
});

test("detectMotionEvents: gentle continuous motion stays under the safety jerk threshold", () => {
  const points: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < 30; i += 1) {
    points.push({ x: 0.5 + Math.sin(i * 0.2) * 0.02, y: 0.5 });
  }
  const events = detectMotionEvents(samplesFrom(points));
  assert.equal(events.filter((e) => e.type === "safety_alert").length, 0);
});

test("motionSamplesFromRawFrames + detectMotionEvents: null (undetected) frames don't crash the detector", () => {
  // Sparse detection shouldn't throw - just skip the gaps.
  const raw: number[][] = [];
  for (let i = 0; i < 10; i += 1) {
    const frame = new Array(10 + 22 * 4 + 21 * 3 + 21 * 3).fill(0);
    frame[0] = i * STEP_MS;
    frame[1] = 1;
    frame[7] = i % 3 === 0 ? 0 : 1; // right hand detected 2/3 of the time
    raw.push(frame);
  }
  assert.doesNotThrow(() => detectMotionEvents(motionSamplesFromRawFrames(raw)));
});

// -- Closed-loop check against every demo persona motion generator --------
//
// This is the property the user actually asked for: replaying a demo
// persona's event should show what the motion detector genuinely found in
// that clip's coordinates, and that finding should agree with what the
// generator was designed to represent (see app/demo-motion.ts). If a future
// change to either the generator or the detector breaks this agreement,
// this test - not a UI glance - is what should catch it.

const EXPECTED_TYPES: Record<DemoMotionType, string[]> = {
  double_check: ["double_check"],
  micro_delay: ["micro_delay"],
  safety_alert: ["safety_alert"],
  register_tap: ["micro_delay"],
  normal_task: [],
  fine_hand_task: [],
  queue_shift: [],
  high_reach: [],
  low_bend: [],
};

for (const [type, expected] of Object.entries(EXPECTED_TYPES) as Array<[DemoMotionType, string[]]>) {
  test(`detectMotionEvents agrees with demo-motion's "${type}" generator across seeds`, () => {
    for (let seed = 0; seed < 6; seed += 1) {
      const frames = generateEventMotion(type, seed);
      const detected = detectMotionEvents(motionSamplesFromRawFrames(frames))
        .map((event) => event.type)
        .sort();
      assert.deepEqual(
        detected,
        [...expected].sort(),
        `seed ${seed}: expected [${expected.join(",")}], got [${detected.join(",")}]`,
      );
    }
  });
}
