import test from "node:test";
import assert from "node:assert/strict";
import { contextualMotionSlices, isSupportedVisualSample, rankVisualContext, visualWorkAreaCoverage, visualActionIntervals, type VisualActionSample } from "../app/local-action-context.ts";
import { generateEventMotion } from "../app/demo-motion.ts";
import { summarizeInterruptions } from "../app/work-interruptions.ts";
import { SELF_TEST_MOTION_SEGMENTATION, sliceTargetMotions } from "../app/motion-segmentation.ts";

const sample = (timeMs: number, overrides: Partial<VisualActionSample> = {}): VisualActionSample => ({
  timeMs, label: "음료 재료 혼합", kind: "task", similarity: 0.34, margin: 0.07, bodyObserved: true, ...overrides,
});

test("one image or inconsistent context cannot establish an automatic task", () => {
  assert.deepEqual(visualActionIntervals([sample(1000)], { startMs: 0, endMs: 5000 }), []);
  assert.deepEqual(visualActionIntervals([sample(0), sample(1000, { margin: 0.01 }), sample(2000)], { startMs: 0, endMs: 5000 }), []);
  assert.equal(isSupportedVisualSample(sample(0, { bodyObserved: false })), false);
  assert.equal(isSupportedVisualSample(sample(0, { similarity: Number.NaN })), false);
});

test("sustained context bounds a task without manual start and end markers", () => {
  const intervals = visualActionIntervals([sample(1000), sample(2000), sample(3000)], { startMs: 800, endMs: 3200 });
  assert.deepEqual(intervals.map(i => [i.startMs, i.endMs, i.sampleCount]), [[800, 3200, 3]]);
  assert.equal(intervals[0].label, "음료 재료 혼합");
});

test("missing observations and out-of-session samples cannot be bridged into a stable prediction", () => {
  const samples = [sample(-2000), sample(-1000), sample(0), sample(5000), sample(6000)];
  assert.deepEqual(visualActionIntervals(samples, { startMs: 0, endMs: 7000 }), []);
});

test("an interrupted task keeps one interval while retaining phone and unexplained pause time", () => {
  const base = generateEventMotion("normal_task", 1)[0];
  const frames = Array.from({ length: 101 }, (_, i) => { const f = [...base]; f[0] = i * 100; f[10 + 5 * 4] += Math.sin(i * 0.25) * 0.1; return f; });
  const intervals = visualActionIntervals([
    ...[1000, 2000, 3000].map(t => sample(t)),
    ...[4000, 5000, 6000].map(t => sample(t, { label: "전화 사용 의심", kind: "phone" })),
    ...[7000, 8000, 9000].map(t => sample(t)),
  ], { startMs: 0, endMs: 10000 });
  const slices = contextualMotionSlices(frames, sliceTargetMotions(frames, SELF_TEST_MOTION_SEGMENTATION), intervals);
  assert.equal(slices.length, 1);
  assert.equal(slices[0].startMs, 500);
  assert.equal(slices[0].endMs, 9500);
  const summary = summarizeInterruptions([
    { kind: "raised_hand_hold", startMs: 4000, endMs: 6000 },
    { kind: "visual_phone_candidate", startMs: 3500, endMs: 6500 },
    { kind: "stillness", startMs: 1000, endMs: 4000 },
  ], 500, 9500);
  assert.equal(summary.suspectedSeconds, 3);
  assert.equal(summary.unexplainedPauseSeconds, 3);
  assert.equal(summary.remainingSeconds, 6);
});

test("a strong image match cannot invent a task in a motionless scene", () => {
  const base = generateEventMotion("normal_task", 1)[0];
  const frames = Array.from({ length: 41 }, (_, i) => { const f = [...base]; f[0] = i * 100; return f; });
  const intervals = visualActionIntervals([sample(1000), sample(2000), sample(3000)], { startMs: 0, endMs: 4000 });
  assert.deepEqual(contextualMotionSlices(frames, sliceTargetMotions(frames), intervals), []);
});

test("known walking or phone context is not relabelled as an uncovered work candidate", () => {
  const base = generateEventMotion("normal_task", 1)[0];
  const frames = Array.from({ length: 101 }, (_, i) => { const f = [...base]; f[0] = i * 100; return f; });
  const fallback = [{ frames, startMs: 0, endMs: 10000, durationSeconds: 10, originalDurationSeconds: 10, excludedFrameCount: 0, reason: "test" }];
  const intervals = visualActionIntervals([sample(1000), sample(2000), sample(3000),
    ...[5000, 6000, 7000].map(t => sample(t, { label: "이동", kind: "other" }))], { startMs: 0, endMs: 10000 });
  const slices = contextualMotionSlices(frames, fallback, intervals);
  assert.equal(slices.filter(s => s.visualEvidence).length, 1);
  assert.ok(slices.every(s => !s.frames.some(f => f[0] >= 4500 && f[0] <= 7500)));
  assert.ok(slices.every((s, i) => i === 0 || slices[i - 1].endMs < s.startMs));
});

test("ranking uses cosine similarity with alternatives, rejecting invalid embeddings", () => {
  const a = Array(512).fill(0), b = [...a]; a[0] = 1; b[1] = 1;
  const result = rankVisualContext(a, [{ label: "컵 보충", kind: "task", embedding: a }, { label: "기타", kind: "other", embedding: b }]);
  assert.equal(result.label, "컵 보충"); assert.equal(result.similarity, 1); assert.equal(result.margin, 1);
  assert.throws(() => rankVisualContext(Array(512).fill(0), []));
});

test("image context cannot widen observed work entry and return boundaries", () => {
  const base = generateEventMotion("normal_task", 1)[0];
  const frames = Array.from({ length: 101 }, (_, i) => { const f = [...base]; f[0] = i * 100; return f; });
  const fallback = [{ frames: frames.filter(f => f[0] >= 2000 && f[0] <= 8000), startMs: 2000, endMs: 8000,
    durationSeconds: 6, originalDurationSeconds: 10, excludedFrameCount: 40, reason: "test",
    boundaryEvidence: { method: "arm_lifecycle" as const, startObserved: true, endObserved: true } }];
  const intervals = visualActionIntervals(Array.from({ length: 11 }, (_, i) => sample(i * 1000)), { startMs: 0, endMs: 10000 });
  const slices = contextualMotionSlices(frames, fallback, intervals);
  assert.equal(slices.length, 1);
  assert.equal(slices[0].startMs, 2000); assert.equal(slices[0].endMs, 8000);
  assert.equal(slices[0].visualEvidence?.label, "음료 재료 혼합");
});

test("strong setup context vetoes automatic learning without erasing the observed movement", () => {
  const base = generateEventMotion("normal_task", 1)[0];
  const frames = Array.from({ length: 41 }, (_, i) => { const f = [...base]; f[0] = i * 100; return f; });
  const fallback = [{ frames, startMs: 0, endMs: 4000, durationSeconds: 4, originalDurationSeconds: 4, excludedFrameCount: 0, reason: "test",
    boundaryEvidence: { method: "arm_lifecycle" as const, startObserved: true, endObserved: true } }];
  const intervals = visualActionIntervals([sample(1000, { label: "설정·기타", kind: "other" }), sample(2000, { label: "설정·기타", kind: "other" }), sample(3000, { label: "설정·기타", kind: "other" })], { startMs: 0, endMs: 4000 });
  const slices = contextualMotionSlices(frames, fallback, intervals);
  assert.equal(slices.length, 1); assert.equal(slices[0].contextNonWork, true);
  assert.equal(slices[0].frames.length, frames.length);
});


test("visual diagnostics retain alternatives and identify body parts outside the model crop", () => {
  const a=Array(512).fill(0),b=Array(512).fill(0);a[0]=1;b[1]=1;
  const result=rankVisualContext(a,[{label:"A",kind:"task",embedding:a},{label:"Other",kind:"other",embedding:b}]);
  assert.equal(result.candidates.length,2);
  const points=Array.from({length:33},()=>({x:.5,y:.5,visibility:.95}));
  const geometry={width:1920,height:1080,left:420,top:0,size:1080,transform:"center_crop" as const};
  assert.equal(visualWorkAreaCoverage(points,geometry),1);
  points[15].x=.05;points[16].x=.95;
  assert.equal(visualWorkAreaCoverage(points,geometry),4/6);
  assert.equal(visualWorkAreaCoverage(points.map(p=>({...p,visibility:0})),geometry),null);
});
