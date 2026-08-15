import test from "node:test";
import assert from "node:assert/strict";

import { buildBaseline } from "../app/observation-engine.ts";
import { OCCUPATION_TEMPLATES } from "../app/occupation-templates.ts";
import {
  createSyntheticTrainingDataset,
  getSyntheticTrainingClips,
} from "../app/synthetic-training.ts";

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
      assert.ok((clip?.frames.length ?? 0) > 20);
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
