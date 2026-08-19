import test from "node:test";
import assert from "node:assert/strict";

import { buildBaseline } from "../app/observation-engine.ts";
import { OCCUPATION_TEMPLATES } from "../app/occupation-templates.ts";
import {
  createSyntheticTrainingDataset,
  getSyntheticTrainingClips,
} from "../app/synthetic-training.ts";
import { classifySkeletonMotion } from "../app/motion-classifier.ts";
import { sliceTargetMotion } from "../app/motion-segmentation.ts";
import { SYNTHETIC_REPLAY_FPS } from "../app/synthetic-training.ts";

const FIXED_NOW = new Date("2026-08-15T12:00:00.000Z").getTime();

test("every occupation produces a complete fourteen-day synthetic baseline", () => {
  for (const occupation of OCCUPATION_TEMPLATES) {
    const dataset = createSyntheticTrainingDataset(occupation.id, 7, FIXED_NOW);
    const baseline = buildBaseline(dataset.episodes, 7);
    assert.equal(baseline.confidence, 100, occupation.label);
    assert.equal(baseline.eligibleDays, 14, occupation.label);
    assert.ok(baseline.acceptedSamples >= 20, occupation.label);
    assert.ok(dataset.episodes.every((episode) => episode.source === "synthetic_training"));
    assert.ok(dataset.episodes.every((episode) => episode.disposition === "accepted"));
  }
});

test("every non-rest task has a deterministic replay clip and four baseline samples", () => {
  for (const occupation of OCCUPATION_TEMPLATES) {
    const dataset = createSyntheticTrainingDataset(occupation.id, 3, FIXED_NOW);
    const clips = getSyntheticTrainingClips(occupation.id);
    const expectedTasks = occupation.tasks.filter((task) => task.id !== "REST");
    assert.equal(clips.length, expectedTasks.length, occupation.label);
    for (const task of expectedTasks) {
      const clip = clips.find((item) => item.taskType === task.id);
      const samples = dataset.episodes.filter((episode) => episode.taskType === task.id);
      assert.ok(clip, `${occupation.label} ${task.label}`);
      // demo-motion.ts's clips got noticeably faster (shorter) so the
      // synthetic skeleton motion reads as human-paced rather than
      // slow-motion - the shortest type (safety_alert) is now ~12 frames,
      // so this just checks for "a real clip", not a specific old length.
      assert.ok((clip?.frames.length ?? 0) > 8);
      assert.ok(samples.length >= 4);
      assert.ok(samples.every((sample) => sample.primitiveLabels.length > 0));
    }
  }
});

test("synthetic generation is stable for the same occupation", () => {
  const first = getSyntheticTrainingClips("cafe");
  const second = getSyntheticTrainingClips("cafe");
  assert.deepEqual(first, second);
});

test("synthetic previews use human-paced duration and smooth 30fps coordinates", () => {
  const clips = getSyntheticTrainingClips("cafe");
  for (const clip of clips) {
    const durationSeconds = clip.frames.at(-1)![0] / 1000;
    assert.ok(durationSeconds >= 4.8 && durationSeconds <= 6.2, `${clip.taskLabel}: ${durationSeconds}`);
    assert.ok(
      Math.abs(clip.frames.length - (durationSeconds * SYNTHETIC_REPLAY_FPS + 1)) <= 1,
      `${clip.taskLabel} should replay near ${SYNTHETIC_REPLAY_FPS}fps`,
    );
  }
});

test("every labeled task has a distinct visible motion trajectory", () => {
  const clips = OCCUPATION_TEMPLATES.flatMap((occupation) => getSyntheticTrainingClips(occupation.id));
  const signatures = clips.map((clip) => {
    const sampleIndexes = Array.from({ length: 10 }, (_, index) =>
      Math.min(clip.frames.length - 1, Math.round((clip.frames.length - 1) * index / 9)),
    );
    return sampleIndexes.map((frameIndex) => {
      const frame = clip.frames[frameIndex];
      const bodyStart = 10;
      const values = [
        frame[bodyStart + 4 * 4], frame[bodyStart + 4 * 4 + 1],
        frame[bodyStart + 5 * 4], frame[bodyStart + 5 * 4 + 1],
        frame[bodyStart + 12 * 4], frame[bodyStart + 13 * 4],
      ];
      return values.map((value) => value.toFixed(4)).join(",");
    }).join("|");
  });
  assert.equal(new Set(signatures).size, clips.length);
});

test("default skeleton uses narrower shoulders and longer human-like limbs", () => {
  const clip = getSyntheticTrainingClips("cafe")[0];
  const body = clip.frames[0].slice(10, 98);
  const metric = (from: number, to: number) => Math.hypot(
    (body[from * 4] - body[to * 4]) * (16 / 9),
    body[from * 4 + 1] - body[to * 4 + 1],
  );
  const shoulderMid = {
    x: (body[0] + body[4]) / 2,
    y: (body[1] + body[5]) / 2,
  };
  const hipMid = {
    x: (body[12 * 4] + body[13 * 4]) / 2,
    y: (body[12 * 4 + 1] + body[13 * 4 + 1]) / 2,
  };
  const torso = Math.hypot((shoulderMid.x - hipMid.x) * (16 / 9), shoulderMid.y - hipMid.y);
  assert.ok(metric(0, 1) / torso < 0.95);
  assert.ok((metric(0, 2) + metric(2, 4)) / torso > 1.25);
});

test("skeleton classifier identifies each followed cafe motion without using the selected label", () => {
  const clips = getSyntheticTrainingClips("cafe");
  for (const clip of clips) {
    const result = classifySkeletonMotion(clip.frames, clips);
    assert.equal(result.status, "matched", clip.taskLabel);
    assert.equal(result.predictedTaskType, clip.taskType, clip.taskLabel);
    assert.ok(result.confidence >= 0.9, `${clip.taskLabel}: ${result.confidence}`);
    assert.equal(result.candidates[0]?.taskType, clip.taskType);
  }
});

test("skeleton classifier is duration-independent and reports alternatives", () => {
  const clips = getSyntheticTrainingClips("convenience_store");
  const target = clips.find((clip) => clip.taskType === "SHELF_RESTOCK")!;
  const slowerFrames = target.frames.map((frame) => {
    const next = [...frame];
    next[0] *= 2.5;
    return next;
  });
  const result = classifySkeletonMotion(slowerFrames, clips);
  assert.equal(result.predictedTaskType, target.taskType);
  assert.ok(result.candidates.length >= 2);
  assert.ok(result.evidence.some((item) => item.includes("관절 궤적")));
});

test("skeleton classifier defers when there are too few usable frames", () => {
  const clips = getSyntheticTrainingClips("cafe");
  const result = classifySkeletonMotion(clips[0].frames.slice(0, 3), clips);
  assert.equal(result.status, "insufficient");
  assert.equal(result.predictedTaskType, null);
});

test("target slicer removes close-up camera setup and stop handling around a full-body motion", () => {
  const clip = getSyntheticTrainingClips("cafe").find((item) => item.taskType === "DRINK_PREP_TASK")!;
  const frameInterval = 1000 / 5;
  const partial = (source: number[], timeMs: number) => {
    const frame = [...source];
    frame[0] = timeMs;
    frame[2] = 0;
    return frame;
  };
  const prefix = Array.from({ length: 25 }, (_, index) => partial(clip.frames[0], index * frameInterval));
  const targetStart = prefix.length * frameInterval;
  const target = clip.frames
    .filter((_, index) => index % 6 === 0)
    .map((source, index) => {
      const frame = [...source];
      frame[0] = targetStart + index * frameInterval;
      return frame;
    });
  const suffixStart = target.at(-1)![0] + frameInterval;
  const suffix = Array.from({ length: 25 }, (_, index) => partial(clip.frames.at(-1)!, suffixStart + index * frameInterval));
  const whole = [...prefix, ...target, ...suffix];
  const sliced = sliceTargetMotion(whole, clip.primitiveLabels);
  assert.ok(sliced.frames.length < whole.length / 2);
  assert.ok(sliced.durationSeconds >= 2 && sliced.durationSeconds <= 7);
  assert.ok(sliced.originalDurationSeconds > sliced.durationSeconds * 2);
  assert.ok(sliced.reason.includes("목표 동작"));
});
