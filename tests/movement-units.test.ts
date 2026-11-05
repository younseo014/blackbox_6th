import assert from "node:assert/strict";
import test from "node:test";
import { generateEventMotion } from "../app/demo-motion.ts";
import { detectMovementCycles } from "../app/movement-units.ts";

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
