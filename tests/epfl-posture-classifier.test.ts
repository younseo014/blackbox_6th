import assert from "node:assert/strict";
import test from "node:test";
import { buildEpflPostureFeatures, classifyEpflPosture } from "../app/epfl-posture-classifier.ts";
import { generateEventMotion } from "../app/demo-motion.ts";

test("EPFL posture feature adapter produces the trained 298-value layout", () => {
  const features = buildEpflPostureFeatures(generateEventMotion("normal_task", 4));
  assert.ok(features);
  assert.equal(features.length, 298);
  assert.ok(features.every(Number.isFinite));
});

test("EPFL posture head returns normalized class probabilities", () => {
  const result = classifyEpflPosture(generateEventMotion("normal_task", 7));
  assert.notEqual(result.status, "insufficient");
  assert.ok(["STAND", "WALK", "BEND_DOWN"].includes(result.label ?? ""));
  assert.ok(Math.abs(result.scores.reduce((sum, score) => sum + score.confidence, 0) - 1) < 1e-9);
  assert.equal(result.model, "epfl-posture-v1");
});

test("EPFL posture head refuses cropped clips because it was validated on full bodies", () => {
  const cropped = generateEventMotion("normal_task", 2).map((frame) => {
    const next = [...frame];
    next[2] = 0;
    return next;
  });
  const result = classifyEpflPosture(cropped);
  assert.equal(result.status, "insufficient");
  assert.equal(result.label, null);
});
