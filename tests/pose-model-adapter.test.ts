import assert from "node:assert/strict";
import test from "node:test";
import {
  BACKEND_PREFERENCE,
  loadPoseBackend,
  orderFactories,
  type PoseBackend,
  type PoseBackendFactory,
  type PoseBackendId,
} from "../app/pose-model-adapter.ts";

function backend(id: PoseBackendId): PoseBackend {
  return {
    id,
    label: id,
    capabilities: { execution: "worker", delegate: "webgpu", wholeBody: true },
    source: { url: `https://example.invalid/${id}`, sha256: "0".repeat(64), license: "Apache-2.0" },
    detect: (_frame, timestamp) => ({ landmarks: [], timestamp }),
    close: () => {},
  };
}

function working(id: PoseBackendId): PoseBackendFactory {
  return { id, label: id, create: async () => backend(id) };
}

function broken(id: PoseBackendId, message: string): PoseBackendFactory {
  return { id, label: id, create: async () => { throw new Error(message); } };
}

test("the preferred backend is used when it loads", async () => {
  const outcome = await loadPoseBackend([working("rtmw-l-384x288"), working("mediapipe-heavy")]);
  assert.equal(outcome.backend?.id, "rtmw-l-384x288");
  assert.equal(outcome.usedFallback, false);
});

test("a failed model load falls back to MediaPipe and says so", async () => {
  const outcome = await loadPoseBackend([
    broken("rtmw-l-384x288", "WebGPU adapter unavailable"),
    broken("rtmw-l-256x192", "model fetch failed"),
    working("mediapipe-heavy"),
  ]);
  assert.equal(outcome.backend?.id, "mediapipe-heavy");
  assert.equal(outcome.usedFallback, true);
  assert.equal(outcome.attempts.length, 3);
  assert.match(outcome.attempts[0].error!, /WebGPU/);
  assert.equal(outcome.attempts.filter((attempt) => attempt.ok).length, 1);
});

test("every backend failing returns null rather than a broken backend", async () => {
  const outcome = await loadPoseBackend([
    broken("rtmw-l-384x288", "no"),
    broken("mediapipe-heavy", "also no"),
  ]);
  assert.equal(outcome.backend, null);
  assert.equal(outcome.attempts.every((attempt) => !attempt.ok), true);
});

test("MediaPipe Heavy is always the last resort, never skipped past", () => {
  assert.equal(BACKEND_PREFERENCE[BACKEND_PREFERENCE.length - 1], "mediapipe-heavy");
  const ordered = orderFactories([working("mediapipe-heavy"), working("rtmw-m-256x192"), working("rtmw-l-384x288")]);
  assert.deepEqual(ordered.map((factory) => factory.id), [
    "rtmw-l-384x288",
    "rtmw-m-256x192",
    "mediapipe-heavy",
  ]);
});

test("a backend must declare its provenance and checksum", async () => {
  const outcome = await loadPoseBackend([working("rtmw-l-256x192")]);
  assert.ok(outcome.backend!.source.url.length > 0);
  assert.equal(outcome.backend!.source.sha256.length, 64);
  assert.ok(outcome.backend!.source.license.length > 0);
});
