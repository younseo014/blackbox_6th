import assert from "node:assert/strict";
import test from "node:test";
import {
  applyManualActionLabel,
  assessActionAmbiguity,
  createActionReview,
  listPendingActionReviews,
  markActionReviewUnresolved,
  normalizeActionCandidates,
} from "../app/action-review.ts";
import { buildBaseline, type ObservationEpisode } from "../app/observation-engine.ts";

function episode(overrides: Partial<ObservationEpisode> = {}): ObservationEpisode {
  return {
    id: "episode-1",
    sessionId: "session-1",
    recordedAt: 100,
    date: "2026-09-10",
    occupation: "cafe",
    mode: "learning",
    phase: "business",
    taskType: "DRINK_PREP",
    taskLabel: "음료 제조",
    taskConfidence: 0.5,
    primitiveLabels: [],
    features: {
      durationSeconds: 10,
      activeRatio: 0.5,
      pauseCount: 0,
      longestPauseSeconds: 0,
      pathLength: 0.8,
      routeComplexity: 1,
      repetitionCount: 2,
      dominantZone: "DRINK_PREP",
      zoneTransitions: 0,
    },
    disposition: "quarantined",
    dispositionReason: "ambiguous",
    durationZScore: null,
    pauseZScore: null,
    contextWeight: 1,
    baselineVersion: 1,
    ...overrides,
  };
}

test("assessActionAmbiguity flags low confidence, close candidates and tracking issues", () => {
  const result = assessActionAmbiguity({
    confidence: 0.55,
    candidates: [
      { taskType: "CLEAN", taskLabel: "세척", confidence: 0.55 },
      { taskType: "PREP", taskLabel: "제조", confidence: 0.5 },
    ],
    qualityReasons: ["poor_tracking"],
  });
  assert.equal(result.ambiguous, true);
  assert.deepEqual(result.reasons, ["low_confidence", "close_candidates", "poor_tracking"]);
  assert.ok(Math.abs(result.topCandidateMargin! - 0.05) < 1e-9);
});

test("assessActionAmbiguity accepts a clear, high-confidence winner", () => {
  const result = assessActionAmbiguity({
    confidence: 0.82,
    candidates: [
      { taskType: "PREP", taskLabel: "제조", confidence: 0.82 },
      { taskType: "CLEAN", taskLabel: "세척", confidence: 0.4 },
    ],
  });
  assert.equal(result.ambiguous, false);
  assert.deepEqual(result.reasons, []);
});

test("ambiguity boundaries accept exactly 60% confidence and a 12 point lead", () => {
  const result = assessActionAmbiguity({
    confidence: 0.6,
    candidates: [
      { taskType: "PREP", taskLabel: "제조", confidence: 0.6 },
      { taskType: "CLEAN", taskLabel: "세척", confidence: 0.48 },
    ],
  });
  assert.equal(result.ambiguous, false);
  assert.deepEqual(result.reasons, []);
});

test("normalizeActionCandidates clamps, de-duplicates and sorts candidates", () => {
  assert.deepEqual(normalizeActionCandidates([
    { taskType: "A", taskLabel: "에이", confidence: -1 },
    { taskType: "B", taskLabel: "비", confidence: 2 },
    { taskType: "A", taskLabel: "에이", confidence: 0.4 },
  ]), [
    { taskType: "B", taskLabel: "비", confidence: 1 },
    { taskType: "A", taskLabel: "에이", confidence: 0.4 },
  ]);
});

test("createActionReview only creates metadata for ambiguous episodes", () => {
  const pending = createActionReview(episode(), { createdAt: 200 });
  assert.equal(pending?.status, "pending");
  assert.deepEqual(pending?.reasons, ["low_confidence"]);
  assert.equal(createActionReview(episode({ taskConfidence: 0.9 })), null);
});

test("an observation without registered candidates stays unclassified", () => {
  const unclassified = episode({
    taskType: "UNCLASSIFIED",
    taskLabel: "미분류",
    taskConfidence: 0,
  });
  const review = createActionReview(unclassified, {
    candidates: [],
    qualityReasons: ["missing_zone", "missing_time_context", "unknown_motion"],
  });
  assert.deepEqual(review?.candidates, []);
  assert.deepEqual(review?.reasons, [
    "no_candidates",
    "low_confidence",
    "missing_zone",
    "missing_time_context",
    "unknown_motion",
  ]);
});

test("applyManualActionLabel confirms the label without mutating the source episode", () => {
  const source = { ...episode(), actionReview: createActionReview(episode(), { createdAt: 200 })! };
  const result = applyManualActionLabel(source, {
    taskType: "CLEAN",
    taskLabel: "세척",
    reviewer: "developer",
    reviewedAt: 300,
  });
  assert.notEqual(result, source);
  assert.equal(source.taskType, "DRINK_PREP");
  assert.equal(source.actionReview.status, "pending");
  assert.equal(result.taskType, "CLEAN");
  assert.equal(result.taskConfidence, 1);
  assert.equal(result.disposition, "accepted");
  assert.equal(result.actionReview?.status, "confirmed");
  assert.equal(result.actionReview?.manualLabel?.reviewer, "developer");
  const baseline = buildBaseline([result], 1);
  assert.equal(baseline.acceptedSamples, 1);
  assert.equal(baseline.tasks[0]?.taskLabel, "세척");
});

test("unresolved reviews are excluded and pending queue is chronological", () => {
  const first = { ...episode({ id: "first", recordedAt: 100 }), actionReview: createActionReview(episode(), { createdAt: 100 })! };
  const second = { ...episode({ id: "second", recordedAt: 200 }), actionReview: createActionReview(episode(), { createdAt: 200 })! };
  const unresolved = markActionReviewUnresolved(first, 300);
  assert.equal(unresolved.disposition, "excluded");
  assert.equal(unresolved.actionReview?.status, "unresolved");
  assert.deepEqual(listPendingActionReviews([second, unresolved]).map((item) => item.id), ["second"]);
});
