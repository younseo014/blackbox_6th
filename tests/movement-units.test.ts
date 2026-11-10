import assert from "node:assert/strict";
import test from "node:test";
import { generateEventMotion } from "../app/demo-motion.ts";
import { detectMovementCycles } from "../app/movement-units.ts";
import { detectWorkInterruptions, summarizeInterruptions, normalizeInterruptionSummary } from "../app/work-interruptions.ts";

test("legacy interruption summaries retain unexplained pauses without double-counting overlap", () => {
  const upgraded = normalizeInterruptionSummary({ intervals: [
    { kind: "stillness", startMs: 0, endMs: 5000 },
    { kind: "tracking_gap", startMs: 3000, endMs: 6000 },
  ], suspectedSeconds: 5, unobservedSeconds: 3, remainingSeconds: 4 });
  assert.equal(upgraded.remainingSeconds, 7);
  assert.equal(upgraded.unexplainedPauseSeconds, 5);
  assert.equal(upgraded.suspectedSeconds, 0);
  assert.deepEqual(normalizeInterruptionSummary(upgraded), upgraded);
});

function repeatedWristMotion(cycleCount: number, periodMs = 1200) {
  const base = generateEventMotion("normal_task", 1)[0];
  return Array.from({ length: cycleCount * periodMs / 100 + 1 }, (_, index) => {
    const frame = [...base];
    frame[0] = index * 100;
    const shoulderX = frame[10 + 1 * 4];
    frame[10 + 5 * 4] = shoulderX + 0.12 * Math.cos(2 * Math.PI * frame[0] / periodMs);
    frame[10 + 5 * 4 + 3] = 0.95;
    frame[10 + 1 * 4 + 3] = 0.95;
    return frame;
  });
}

test("detectMovementCycles records full wrist returns with timestamps", () => {
  const cycles = detectMovementCycles(repeatedWristMotion(4));
  assert.ok(cycles.length >= 3);
  assert.ok(cycles.every((cycle) => cycle.hand === "right" && cycle.axis === "x"));
  assert.ok(cycles.every((cycle) => Math.abs(cycle.durationSeconds - 1.2) <= 0.2));
  assert.ok(cycles.every((cycle, index) => index === 0 || cycles[index - 1].endMs <= cycle.startMs));
});

test("detectMovementCycles does not turn a tracking gap into a long cycle", () => {
  const frames = repeatedWristMotion(4).filter((frame) => frame[0] < 1200 || frame[0] > 2600);
  const cycles = detectMovementCycles(frames);
  assert.ok(cycles.every((cycle) => cycle.endMs <= 1200 || cycle.startMs >= 2600));
});

test("small stationary wrist jitter is not counted as repeated work", () => {
  const frames = repeatedWristMotion(4).map((raw, index) => {
    const frame = [...raw];
    frame[10 + 5 * 4] = frame[10 + 1 * 4] + 0.002 * Math.sin(index * 2);
    return frame;
  });
  assert.equal(detectMovementCycles(frames).length, 0);
});

function interruptionFrames() {
  const base = generateEventMotion("normal_task", 1)[0];
  return Array.from({ length: 121 }, (_, index) => {
    const frame = [...base];
    frame[0] = index * 100;
    for (const joint of [0, 1, 4, 5]) {
      frame[10 + joint * 4] = joint % 2 ? 0.6 : 0.4;
      frame[10 + joint * 4 + 1] = joint < 2 ? 0.4 : 0.65;
      frame[10 + joint * 4 + 3] = 0.95;
    }
    const holding = index >= 30 && index <= 80;
    if (holding) {
      frame[10 + 4 * 4] = 0.42;
      frame[10 + 4 * 4 + 1] = 0.3;
    } else {
      frame[10 + 4 * 4] += 0.1 * Math.sin(index * 0.7);
      frame[10 + 5 * 4] += 0.1 * Math.cos(index * 0.7);
    }
    return frame;
  });
}

test("detects a sustained raised-hand interruption inside work without calling it a confirmed phone call", () => {
  const intervals = detectWorkInterruptions(interruptionFrames());
  const hold = intervals.find((item) => item.kind === "raised_hand_hold");
  assert.ok(hold);
  assert.equal(hold.startMs, 3000);
  assert.equal(hold.endMs, 8100);
  const summary = summarizeInterruptions(intervals, 0, 12000);
  assert.ok(summary.suspectedSeconds >= 5 && summary.suspectedSeconds < 6);
  assert.equal(summary.unobservedSeconds, 0);
  assert.ok(summary.remainingSeconds > 6);
});

test("brief raised hand and moving work are not sustained interruptions", () => {
  const frames = interruptionFrames().filter((frame) => frame[0] < 4000).map((frame) => [...frame]);
  assert.equal(detectWorkInterruptions(frames).length, 0);
});

test("missing skeletons produce one tracking gap and never count as a pause or hand hold", () => {
  const frames = interruptionFrames().map((frame) => {
    const next = [...frame];
    if (next[0] >= 3000 && next[0] <= 8000) next[1] = 0;
    return next;
  });
  const intervals = detectWorkInterruptions(frames);
  assert.equal(intervals.length, 1);
  assert.equal(intervals[0].kind, "tracking_gap");
  assert.ok(summarizeInterruptions(intervals, 0, 12000).unobservedSeconds >= 5);
});

test("overlapping cues are counted once and clipped to the reviewed task", () => {
  const summary = summarizeInterruptions([
    { kind: "raised_hand_hold", startMs: 1000, endMs: 6000 },
    { kind: "stillness", startMs: 2000, endMs: 7000 },
    { kind: "tracking_gap", startMs: 7000, endMs: 9000 },
  ], 3000, 8000);
  assert.equal(summary.suspectedSeconds, 3);
  assert.equal(summary.unexplainedPauseSeconds, 4);
  assert.equal(summary.unobservedSeconds, 1);
  assert.equal(summary.remainingSeconds, 1);
});

test("a recording with no usable skeleton remains unknown instead of interruption-free", () => {
  const frames = interruptionFrames().map((frame) => { const next = [...frame]; next[1] = 0; return next; });
  const summary = summarizeInterruptions(detectWorkInterruptions(frames), 0, 12000);
  assert.equal(summary.unobservedSeconds, 12);
  assert.equal(summary.remainingSeconds, 0);
  assert.equal(summary.suspectedSeconds, 0);
});

test("no stored coordinates over a known recording window are reported as unobserved", () => {
  const intervals = detectWorkInterruptions([], { startMs: 1500, endMs: 11500 });
  assert.equal(summarizeInterruptions(intervals, 1500, 11500).unobservedSeconds, 10);
});

test("one occluded wrist does not discard the visible phone-posture cue or invent complete stillness", () => {
  const frames = interruptionFrames().map((raw) => { const frame = [...raw]; frame[10 + 5 * 4 + 3] = 0; return frame; });
  const intervals = detectWorkInterruptions(frames);
  assert.ok(intervals.some((item) => item.kind === "raised_hand_hold"));
  assert.ok(!intervals.some((item) => item.kind === "stillness" || item.kind === "tracking_gap"));
});
