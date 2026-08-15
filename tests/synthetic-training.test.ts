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

