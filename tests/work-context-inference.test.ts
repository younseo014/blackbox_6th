import assert from "node:assert/strict";
import test from "node:test";
import {
  buildWorkContextCandidates,
  listWorkContextLabels,
  workContextTaskType,
} from "../app/work-context-inference.ts";

const config = {
  routines: [{
    days: [4],
    schedule: [
      { id: "prep", time: "10:00", label: "음료 제조" },
      { id: "wash", time: "18:00", label: "설거지" },
    ],
  }],
  closedDays: [],
};

test("work-context time is a weak prior for the current weekday", () => {
  const candidates = buildWorkContextCandidates({
    config,
    recordedAt: new Date(2026, 8, 10, 10, 20).getTime(),
  });
  assert.equal(candidates.find((item) => item.taskLabel === "음료 제조")?.confidence, 0.48);
  assert.equal(candidates.find((item) => item.taskLabel === "설거지")?.confidence, 0.32);
});

test("a learned skeleton match can outweigh the schedule prior", () => {
  const candidates = buildWorkContextCandidates({
    config,
    recordedAt: new Date(2026, 8, 10, 10, 20).getTime(),
    motionCandidates: [{ label: "설거지", confidence: 0.81 }],
  });
  assert.equal(
    Math.max(...candidates.filter((item) => item.taskType === workContextTaskType("설거지")).map((item) => item.confidence)),
    0.81,
  );
});

test("manual label choices are de-duplicated by normalized task label", () => {
  const duplicated = {
    routines: [
      ...config.routines,
      { days: [5], schedule: [{ id: "prep-2", time: "11:00", label: "  음료   제조 " }] },
    ],
    closedDays: [],
  };
  assert.deepEqual(listWorkContextLabels(duplicated), [
    { taskType: workContextTaskType("음료 제조"), taskLabel: "음료 제조" },
    { taskType: workContextTaskType("설거지"), taskLabel: "설거지" },
  ]);
});

test("no registered work context produces no semantic task candidates", () => {
  const candidates = buildWorkContextCandidates({
    config: { routines: [{ days: [4], schedule: [] }], closedDays: [] },
    recordedAt: new Date(2026, 8, 10, 14, 42).getTime(),
    motionCandidates: [
      { label: "주문·결제", confidence: 0.9 },
      { label: "음료 제조", confidence: 0.8 },
    ],
  });
  assert.deepEqual(candidates, []);
});
