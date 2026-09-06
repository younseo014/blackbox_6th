import assert from "node:assert/strict";
import test from "node:test";
import {
  computeEventChangeEvidence,
  computeDailyChangeBurden,
  computeDailyFlowSimilarity,
  computeWeeklyFlowSimilarity,
  weeklyAlertLevel,
  sampleGrade,
  isTaskBaselineScorable,
} from "../app/dfs-score.ts";
import type { ObservationEpisode, TaskBaseline } from "../app/observation-engine.ts";

function episode(overrides: Partial<ObservationEpisode> = {}): ObservationEpisode {
  return {
    id: "episode-1",
    sessionId: "session-1",
    recordedAt: Date.parse("2026-08-14T10:00:00Z"),
    date: "2026-08-14",
    occupation: "cafe",
    mode: "analysis",
    phase: "business",
    taskType: "DRINK_PREP",
    taskLabel: "음료 제조",
    taskConfidence: 0.9,
    primitiveLabels: [],
    features: {
      durationSeconds: 0,
      activeRatio: 0,
      pauseCount: 0,
      longestPauseSeconds: 0,
      pathLength: 0,
      routeComplexity: 0,
      repetitionCount: 0,
      dominantZone: null,
      zoneTransitions: 0,
    },
    disposition: "analysis_only",
    dispositionReason: "",
    durationZScore: null,
    pauseZScore: null,
    contextWeight: 1,
    baselineVersion: 1,
    ...overrides,
  };
}

// -- computeEventChangeEvidence ---------------------------------------------

test("ECE matches the spec section 4.5 worked example", () => {
  const ece = computeEventChangeEvidence(
    episode({ durationZScore: 2.4, pauseZScore: 2.1, contextWeight: 3 }),
  );
  assert.ok(ece !== null);
  assert.ok(Math.abs(ece! - 1.56) < 0.01);
});

test("ECE is null below the task-confidence gate", () => {
  const ece = computeEventChangeEvidence(episode({ taskConfidence: 0.4, durationZScore: 3 }));
  assert.equal(ece, null);
});

test("ECE is null for rest context (contextWeight 0)", () => {
  const ece = computeEventChangeEvidence(episode({ contextWeight: 0, durationZScore: 3 }));
  assert.equal(ece, null);
});

test("ECE is null when neither z-score is available", () => {
  const ece = computeEventChangeEvidence(episode({ durationZScore: null, pauseZScore: null }));
  assert.equal(ece, null);
});

test("ECE uses the single available z-score at full weight", () => {
  const ece = computeEventChangeEvidence(episode({ durationZScore: 3, pauseZScore: null, contextWeight: 1 }));
  assert.ok(ece !== null);
  assert.ok(Math.abs(ece! - 1) < 0.001);
});

// -- computeDailyChangeBurden -------------------------------------------------

test("DCB deduplicates same task within a 2-hour window, keeping the higher ECE", () => {
  const base = Date.parse("2026-08-14T10:00:00Z");
  const episodes = [
    episode({ id: "a", recordedAt: base, durationZScore: 2.0, contextWeight: 1 }),
    episode({ id: "b", recordedAt: base + 30 * 60 * 1000, durationZScore: 3.0, contextWeight: 1 }),
  ];
  const [day] = computeDailyChangeBurden(episodes);
  assert.equal(day.contributingEpisodeIds.length, 1);
  assert.equal(day.contributingEpisodeIds[0], "b");
});

test("DCB caps a single task's daily contribution at 3.0 and the day at 9.0", () => {
  const base = Date.parse("2026-08-14T08:00:00Z");
  const episodes = [
    episode({ id: "t1", taskType: "A", recordedAt: base, durationZScore: 3, contextWeight: 3 }),
    episode({ id: "t2", taskType: "A", recordedAt: base + 3 * 3600_000, durationZScore: 3, contextWeight: 3 }),
    episode({ id: "t3", taskType: "B", recordedAt: base + 1 * 3600_000, durationZScore: 3, contextWeight: 3 }),
    episode({ id: "t4", taskType: "C", recordedAt: base + 2 * 3600_000, durationZScore: 3, contextWeight: 3 }),
  ];
  const [day] = computeDailyChangeBurden(episodes);
  assert.ok(day.dcb <= 9);
  assert.equal(day.dcb, 9);
});

// -- computeDailyFlowSimilarity / weekly -------------------------------------

test("DFS is 100 at DCB=0 and floored at 40 for large DCB", () => {
  assert.equal(computeDailyFlowSimilarity(0), 100);
  assert.equal(computeDailyFlowSimilarity(100), 40);
});

test("weekly flow similarity requires at least 3 eligible days", () => {
  assert.equal(computeWeeklyFlowSimilarity([90, 90]), null);
  assert.equal(computeWeeklyFlowSimilarity([90, 90, 90]), 90);
});

test("weeklyAlertLevel escalates with repeated high-DCB days", () => {
  assert.equal(weeklyAlertLevel([0, 0, 0]), "none");
  assert.equal(weeklyAlertLevel([3, 0, 0]), "reference");
  assert.equal(weeklyAlertLevel([3, 3, 0]), "watch");
  assert.equal(weeklyAlertLevel([4, 4, 4]), "notable");
});

// -- sample grading -----------------------------------------------------------

test("sampleGrade follows spec section 11.2 thresholds", () => {
  assert.equal(sampleGrade(7), "insufficient");
  assert.equal(sampleGrade(8), "provisional");
  assert.equal(sampleGrade(19), "provisional");
  assert.equal(sampleGrade(20), "eligible");
  assert.equal(sampleGrade(29), "eligible");
  assert.equal(sampleGrade(30), "stable");
});

test("isTaskBaselineScorable requires n>=8 and both SDs >= 0.05", () => {
  const baseline: TaskBaseline = {
    taskType: "DRINK_PREP",
    taskLabel: "음료 제조",
    sampleCount: 10,
    meanDuration: 30,
    durationSD: 4,
    meanLongestPause: 2,
    pauseSD: 1,
    meanRouteComplexity: 1,
  };
  assert.equal(isTaskBaselineScorable(baseline), true);
  assert.equal(isTaskBaselineScorable({ ...baseline, sampleCount: 5 }), false);
  assert.equal(isTaskBaselineScorable({ ...baseline, durationSD: 0.01 }), false);
});
