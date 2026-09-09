import assert from "node:assert/strict";
import test from "node:test";
import { classifyLearnedMotion } from "../app/custom-motion-training.ts";
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
