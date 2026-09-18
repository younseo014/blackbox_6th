import assert from "node:assert/strict";
import test from "node:test";
import { addObservationMotionSample, classifyLearnedMotion } from "../app/custom-motion-training.ts";
import { generateEventMotion } from "../app/demo-motion.ts";

function upperBodyOnly(frames: number[][]) {
  return frames.map((frame) => {
    const next = [...frame];
    for (let joint = 12; joint < 22; joint += 1) {
      next[10 + joint * 4 + 3] = 0;
    }
    return next;
  });
}

function unevenTiming(frames: number[][]) {
  const middleStart = Math.floor(frames.length * 0.35);
  const middleEnd = Math.floor(frames.length * 0.65);
  return [
    ...Array.from({ length: 8 }, () => frames[0]),
    ...frames.flatMap((frame, index) => index >= middleStart && index <= middleEnd
      ? [frame, frame, frame]
      : [frame]),
    ...Array.from({ length: 8 }, () => frames[frames.length - 1]),
  ];
}

test("classifyLearnedMotion recognizes a newly observed motion from repeated examples", () => {
  const result = classifyLearnedMotion(generateEventMotion("high_reach", 7), [
    {
      id: "serving",
      label: "서빙",
      samples: [1, 2, 3, 4, 5].map((seed) => generateEventMotion("normal_task", seed)),
    },
    {
      id: "washing",
      label: "설거지",
      samples: [1, 2, 3, 4, 5].map((seed) => generateEventMotion("high_reach", seed)),
    },
  ]);

  assert.equal(result.status, "matched");
  assert.equal(result.label, "설거지");
  assert.ok(result.confidence >= 0.52);
  assert.equal(result.candidates[0].label, "설거지");
});

test("classifyLearnedMotion refuses clips without enough skeleton frames", () => {
  const result = classifyLearnedMotion(generateEventMotion("normal_task", 0).slice(0, 3), [
    { id: "serving", label: "서빙", samples: [generateEventMotion("normal_task", 1)] },
  ]);

  assert.equal(result.status, "insufficient");
  assert.equal(result.label, null);
});

test("classifyLearnedMotion works when only the upper body is visible", () => {
  const result = classifyLearnedMotion(upperBodyOnly(generateEventMotion("high_reach", 8)), [
    {
      id: "serving",
      label: "서빙",
      samples: [1, 2, 3, 4, 5].map((seed) => upperBodyOnly(generateEventMotion("normal_task", seed))),
    },
    {
      id: "reach",
      label: "선반 정리",
      samples: [1, 2, 3, 4, 5].map((seed) => upperBodyOnly(generateEventMotion("high_reach", seed))),
    },
  ]);

  assert.equal(result.status, "matched");
  assert.equal(result.label, "선반 정리");
});

test("classifyLearnedMotion tolerates pauses and uneven execution speed", () => {
  const result = classifyLearnedMotion(unevenTiming(generateEventMotion("high_reach", 9)), [
    { id: "serving", label: "서빙", samples: [1, 2].map((seed) => generateEventMotion("normal_task", seed)) },
    { id: "reach", label: "선반 정리", samples: [1, 2].map((seed) => generateEventMotion("high_reach", seed)) },
  ]);

  assert.equal(result.status, "matched");
  assert.equal(result.label, "선반 정리");
});

test("classifyLearnedMotion never exposes NaN confidence when some coordinates are invalid", () => {
  const corrupted = generateEventMotion("high_reach", 3).map((frame, frameIndex) => {
    const next = [...frame];
    if (frameIndex % 2 === 0) {
      next[10] = Number.NaN;
      next[10 + 22 * 4 + 8 * 3] = Number.NaN;
    }
    return next;
  });
  const result = classifyLearnedMotion(corrupted, [
    { id: "reach", label: "선반 정리", samples: [generateEventMotion("high_reach", 1)] },
  ]);

  assert.equal(Number.isFinite(result.confidence), true);
  assert.equal(result.candidates.every((candidate) => Number.isFinite(candidate.distance)), true);
});

test("observation labels accumulate separate motion slices from the same long session", () => {
  const first = { sessionId: "business-day", startMs: 1_000, endMs: 3_000 };
  const second = { sessionId: "business-day", startMs: 8_000, endMs: 10_000 };
  const initial = [{ id: "observed-wash", label: "설거지", samples: [first], createdAt: 1 }];
  const updated = addObservationMotionSample(initial, "설거지", second, 2);

  assert.equal(updated[0].samples.length, 2);
  assert.deepEqual(updated[0].samples, [first, second]);
});

test("observation labels replace only the exact same slice", () => {
  const sample = { sessionId: "business-day", startMs: 1_000, endMs: 3_000 };
  const initial = [{ id: "observed-wash", label: "설거지", samples: [sample], createdAt: 1 }];
  const updated = addObservationMotionSample(initial, " 설거지 ", sample, 2);

  assert.equal(updated[0].samples.length, 1);
});
