import assert from "node:assert/strict";
import test from "node:test";
import { generateEventMotion } from "../app/demo-motion.ts";
import {
  DEFAULT_PROFILE,
  buildBaseline,
  createObservationEpisode,
  extractObservationFeatures,
  safeZScore,
  type ObservationEpisode,
} from "../app/observation-engine.ts";
import { OCCUPATION_TEMPLATES, getOccupationTemplate } from "../app/occupation-templates.ts";

test("occupation templates: all six requested occupations have zones, tasks and sequences", () => {
  assert.deepEqual(
    OCCUPATION_TEMPLATES.map((item) => item.id).sort(),
    ["cafe", "clothing_store", "convenience_store", "hair_salon", "restaurant", "workshop"].sort(),
  );
  for (const template of OCCUPATION_TEMPLATES) {
    assert.ok(template.zones.length >= 8, `${template.id} should have practical zones`);
    assert.ok(template.tasks.some((task) => task.phase === "business"));
    assert.ok(template.sequences.length >= 1);
  }
});

test("safeZScore: uses personal mean and SD and defers when SD is too small", () => {
  assert.equal(safeZScore(13, 10, 2), 1.5);
  assert.equal(safeZScore(11, 10, 0), null);
});

test("extractObservationFeatures: finds a pause in a synthetic micro-delay clip", () => {
  const frames = generateEventMotion("micro_delay", 2);
  const features = extractObservationFeatures(frames, Array(9).fill("DRINK_PREP"));
  assert.ok(features.durationSeconds > 1);
  assert.ok(features.pauseCount >= 1);
  assert.ok(features.longestPauseSeconds >= 1);
  assert.equal(features.dominantZone, "DRINK_PREP");
});

test("learning mode: a long unexplained pause is quarantined, not learned", () => {
  const profile = {
    ...DEFAULT_PROFILE,
    learningStartedAt: Date.now(),
    updatedAt: Date.now(),
    zoneGrid: Array(9).fill("DRINK_PREP"),
  };
  const episode = createObservationEpisode({
    sessionId: "session-long-pause",
    recordedAt: Date.now(),
    profile,
    phase: "business",
    features: {
      durationSeconds: 30,
      activeRatio: 0.4,
      pauseCount: 2,
      longestPauseSeconds: 12,
      pathLength: 1,
      routeComplexity: 2,
      repetitionCount: 3,
      dominantZone: "DRINK_PREP",
      zoneTransitions: 0,
    },
    baseline: buildBaseline([], 1),
  });
  assert.equal(episode.disposition, "quarantined");
});

test("analysis mode: episodes never become baseline learning samples", () => {
  const profile = {
    ...DEFAULT_PROFILE,
    mode: "analysis" as const,
    learningStartedAt: Date.now(),
    updatedAt: Date.now(),
    zoneGrid: Array(9).fill("POS"),
  };
  const episode = createObservationEpisode({
    sessionId: "session-analysis",
    recordedAt: Date.now(),
    profile,
    phase: "business",
    features: {
      durationSeconds: 20,
      activeRatio: 0.5,
      pauseCount: 0,
      longestPauseSeconds: 0,
      pathLength: 0.8,
      routeComplexity: 1,
      repetitionCount: 2,
      dominantZone: "POS",
      zoneTransitions: 0,
    },
    baseline: buildBaseline([], 1),
  });
  assert.equal(episode.disposition, "analysis_only");
  assert.equal(buildBaseline([episode], 1).acceptedSamples, 0);
});

test("buildBaseline: separates task baselines and counts only accepted episodes", () => {
  const base = {
    id: "episode",
    sessionId: "session",
    recordedAt: Date.now(),
    date: "2026-08-15",
    occupation: "cafe" as const,
    mode: "learning" as const,
    phase: "business" as const,
    taskType: "ORDER_ZONE",
    taskLabel: "주문·결제",
    taskConfidence: 0.8,
    primitiveLabels: [],
    disposition: "accepted" as const,
    dispositionReason: "accepted",
    durationZScore: null,
    pauseZScore: null,
    contextWeight: 1 as const,
    baselineVersion: 1,
    features: {
      durationSeconds: 20,
      activeRatio: 0.4,
      pauseCount: 0,
      longestPauseSeconds: 0,
      pathLength: 0.5,
      routeComplexity: 1,
      repetitionCount: 1,
      dominantZone: "POS",
      zoneTransitions: 0,
    },
  } satisfies ObservationEpisode;
  const excluded = { ...base, id: "excluded", disposition: "excluded" as const };
  const prep = {
    ...base,
    id: "prep",
    taskType: "DRINK_PREP_TASK",
    taskLabel: getOccupationTemplate("cafe").tasks.find((item) => item.id === "DRINK_PREP_TASK")!.label,
  };
  const baseline = buildBaseline([base, excluded, prep], 1);
  assert.equal(baseline.acceptedSamples, 2);
  assert.equal(baseline.tasks.length, 2);
});
