import test from "node:test";
import assert from "node:assert/strict";
import { generateEventMotion } from "../app/demo-motion.ts";
import { sliceTargetMotions } from "../app/motion-segmentation.ts";

function shifted(frames: number[][], offsetMs: number) {
  return frames.map((frame) => {
    const next = [...frame];
    next[0] += offsetMs;
    return next;
  });
}

function upperBodyOnly(frames: number[][]) {
  return frames.map((frame) => {
    const next = [...frame];
    next[2] = 0;
    for (let joint = 12; joint < 22; joint += 1) {
      next[10 + joint * 4 + 3] = 0;
    }
    return next;
  });
}

function translatedX(frame: number[], offsetX: number) {
  const next = [...frame];
  for (let joint = 0; joint < 22; joint += 1) {
    next[10 + joint * 4] += offsetX;
  }
  const handsStart = 10 + 22 * 4;
  for (let joint = 0; joint < 42; joint += 1) {
    const xIndex = handsStart + joint * 3;
    if (Number.isFinite(next[xIndex])) next[xIndex] += offsetX;
  }
  return next;
}

test("a continuous session is split into separate actions across a quiet interval", () => {
  const first = generateEventMotion("high_reach", 1);
  const restingFrame = first[first.length - 1];
  const quiet = Array.from({ length: 40 }, (_, index) => {
    const frame = [...restingFrame];
    frame[0] = restingFrame[0] + (index + 1) * 100;
    return frame;
  });
  const secondStart = quiet[quiet.length - 1][0] + 100;
  const second = shifted(generateEventMotion("low_bend", 2), secondStart);

  const slices = sliceTargetMotions([...first, ...quiet, ...second]);

  assert.equal(slices.length, 2);
  assert.ok(slices[0].endMs < slices[1].startMs);
  assert.ok(slices.every((slice) => slice.durationSeconds < 3));
  assert.ok(slices.every((slice) => slice.originalDurationSeconds > 5));
});

test("a tracking gap creates separate action intervals instead of one long episode", () => {
  const first = generateEventMotion("register_tap", 4);
  const second = shifted(generateEventMotion("high_reach", 5), first[first.length - 1][0] + 10_000);

  const slices = sliceTargetMotions([...first, ...second]);

  assert.equal(slices.length, 2);
  assert.ok(slices[1].startMs - slices[0].endMs > 8_000);
});

test("business-day timestamps are preserved for actions many hours apart", () => {
  const morningOffset = 15 * 60 * 1000;
  const eveningOffset = 11 * 60 * 60 * 1000;
  const morning = shifted(generateEventMotion("register_tap", 7), morningOffset);
  const evening = shifted(generateEventMotion("low_bend", 8), eveningOffset);

  const slices = sliceTargetMotions([...morning, ...evening]);

  assert.equal(slices.length, 2);
  assert.ok(slices[0].startMs >= morningOffset);
  assert.ok(slices[1].startMs >= eveningOffset);
  assert.ok(slices[1].originalDurationSeconds > 10 * 60 * 60);
});

test("upper-body tracking can create motion intervals without the full-body flag", () => {
  const first = upperBodyOnly(generateEventMotion("high_reach", 11));
  const restingFrame = first[first.length - 1];
  const quiet = Array.from({ length: 35 }, (_, index) => {
    const frame = [...restingFrame];
    frame[0] = restingFrame[0] + (index + 1) * 100;
    return frame;
  });
  const secondStart = quiet[quiet.length - 1][0] + 100;
  const second = shifted(upperBodyOnly(generateEventMotion("register_tap", 12)), secondStart);

  const slices = sliceTargetMotions([...first, ...quiet, ...second]);

  assert.equal(slices.length, 2);
  assert.ok(slices.every((slice) => slice.frames.every((frame) => frame[2] === 0)));
});

test("walking between two tasks becomes its own semantic boundary without a quiet pause", () => {
  const first = generateEventMotion("high_reach", 21);
  const firstEnd = first[first.length - 1];
  const travel = Array.from({ length: 18 }, (_, index) => {
    const frame = translatedX(firstEnd, (index + 1) * 0.012);
    frame[0] = firstEnd[0] + (index + 1) * 100;
    return frame;
  });
  const travelEnd = travel[travel.length - 1];
  const second = generateEventMotion("register_tap", 22).map((frame) => {
    const moved = translatedX(frame, 18 * 0.012);
    moved[0] += travelEnd[0] + 100;
    return moved;
  });

  const slices = sliceTargetMotions([...first, ...travel, ...second]);

  assert.ok(slices.length >= 3);
  assert.ok(slices.every((slice) => slice.durationSeconds < 3));
  assert.ok(slices.every((slice, index) => index === 0 || slices[index - 1].endMs <= slice.startMs));
});

test("continuous movement is not allowed to become one unbounded episode", () => {
  const source = generateEventMotion("register_tap", 30);
  const frames = Array.from({ length: 32 }, (_, repetition) =>
    source.map((frame, index) => {
      const next = [...frame];
      next[0] = (repetition * source.length + index) * 100;
      return next;
    }),
  ).flat();

  const slices = sliceTargetMotions(frames);

  assert.ok(slices.length >= 2);
  assert.ok(slices.every((slice) => slice.durationSeconds <= 21));
});
