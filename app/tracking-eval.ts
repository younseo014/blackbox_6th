import {
  EMPTY_GLOBAL_TRACK,
  GLOBAL_CONFIRMATION_FRAMES,
  updateGlobalPersonTrack,
  type CameraObservation,
  type GlobalPersonTrack,
} from "./global-person-track";
import type { CameraSlot } from "./multi-camera";

/**
 * Reproducible evaluation of the single-person tracking stack.
 *
 * This is deliberately a *replay* harness, not a live one. It consumes a
 * scenario - a timestamped list of per-camera observations plus ground truth -
 * and runs the real `updateGlobalPersonTrack` fusion over it. That makes every
 * number here re-runnable, model-comparable and diffable in CI, which a
 * live-webcam measurement can never be.
 *
 * Scenarios come from two places and only these two: public datasets, and
 * developer-recorded pre-validation clips. Study participant recordings are
 * never used here.
 */

export type EvalCondition =
  | "front"
  | "side"
  | "back"
  | "head_occluded"
  | "partial_body"
  | "distant"
  | "handoff"
  | "empty"
  | "decoy_object";

export type EvalTick = {
  timestamp: number;
  /** Ground truth: is a real person in the capture space at this instant? */
  personPresent: boolean;
  condition: EvalCondition;
  /**
   * Clears the global track before this tick.
   *
   * Still-image datasets are a set of unrelated scenes, not a sequence. Without
   * this the tracker would be asked to carry continuity from one photo to the
   * next, and would be scored on a question the data cannot ask.
   */
  resetBefore?: boolean;
  /** Ground-truth joint positions, in the named camera's normalized frame. */
  truth?: { camera: CameraSlot; joints: Record<number, { x: number; y: number }> };
  /** Ground truth for what each camera is actually pointed at. */
  cameraSubjects?: Partial<Record<CameraSlot, "person" | "object">>;
  observations: CameraObservation[];
};

export type EvalScenario = {
  name: string;
  description: string;
  /** Provenance. Required - an unattributed scenario is not evidence. */
  source: string;
  ticks: EvalTick[];
};

export type JointMetric = {
  index: number;
  /** Fraction of ticks where ground truth says visible and we reported it. */
  detectionRate: number;
  /** Mean normalized-image error over ticks with ground truth, or null. */
  meanError: number | null;
  observedCount: number;
  inferredCount: number;
  missingCount: number;
};

export type ConditionMetric = {
  condition: EvalCondition;
  ticks: number;
  presenceAccuracy: number;
  retention: number;
};

export type EvalResult = {
  scenario: string;
  source: string;
  ticks: number;
  durationMs: number;

  presenceAccuracy: number;
  emptyFrameFalsePositives: number;
  emptyFrameFalsePositivesPer12h: number;
  /** Median ms from a person entering to the track reporting `present`. */
  recognitionLatencyMs: number | null;
  worstRecognitionLatencyMs: number | null;
  /** Fraction of person-present ticks with a live global track. */
  torsoRetention: number;
  /** Times the track's coordinate frame was a camera pointed at an object. */
  objectSwitches: number;
  /** Peak number of simultaneous global person ids. Must never exceed 1. */
  maxConcurrentTracks: number;
  distinctTrackIds: number;

  joints: JointMetric[];
  visibleJointDetectionRate: number;
  meanVisibleJointError: number | null;
  conditions: ConditionMetric[];

  handoffs: number;
  longestHandoffGapMs: number;
  cameraFps: Record<string, number>;

  /** Wall-clock cost of the fusion layer itself, not of the pose model. */
  fusionMsPerTick: number;
  heapGrowthBytes: number | null;
  heapGrowthBytesPer12h: number | null;
};

const TWELVE_HOURS_MS = 12 * 60 * 60 * 1000;

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function heapBytes(): number | null {
  const usage = (globalThis as { process?: { memoryUsage?: () => { heapUsed: number } } }).process;
  try {
    return usage?.memoryUsage?.().heapUsed ?? null;
  } catch {
    return null;
  }
}

/** Runs one scenario through the real fusion layer and scores it. */
export function evaluateScenario(scenario: EvalScenario): EvalResult {
  const ticks = scenario.ticks;
  if (!ticks.length) throw new Error(`Scenario "${scenario.name}" has no ticks`);

  let track: GlobalPersonTrack = EMPTY_GLOBAL_TRACK;
  const trackIds = new Set<string>();

  let presenceCorrect = 0;
  let emptyFrameFalsePositives = 0;
  let personTicks = 0;
  let retainedTicks = 0;
  let objectSwitches = 0;
  let handoffs = 0;
  let longestHandoffGapMs = 0;

  const recognitionLatencies: number[] = [];
  let entryAt: number | null = null;
  let previousPresent = false;
  // Ticks since the track was last cleared. The confirmation gate cannot
  // report `present` during the first GLOBAL_CONFIRMATION_FRAMES - 1 of them,
  // so counting those against retention would measure the gate's existence
  // rather than the tracker's ability to hold a person. Acquisition speed is
  // already measured separately, by recognition latency.
  let ticksSinceReset = 0;

  const jointStats = new Map<number, { observed: number; inferred: number; missing: number; truthTicks: number; hits: number; errors: number[] }>();
  const conditionStats = new Map<EvalCondition, { ticks: number; correct: number; personTicks: number; retained: number }>();
  const cameraObservationCounts = new Map<CameraSlot, number>();

  const heapBefore = heapBytes();
  const startedAt = Date.now();

  for (const tick of ticks) {
    if (tick.resetBefore) {
      track = EMPTY_GLOBAL_TRACK;
      previousPresent = false;
      entryAt = null;
      ticksSinceReset = 0;
    }
    const inConfirmationRamp = ticksSinceReset < GLOBAL_CONFIRMATION_FRAMES - 1;
    ticksSinceReset += 1;
    track = updateGlobalPersonTrack(track, tick.observations, tick.timestamp);
    const reportedPresent = track.presence === "present" || track.presence === "coasting";
    if (track.id) trackIds.add(track.id);

    // Presence.
    if (reportedPresent === tick.personPresent) presenceCorrect += 1;
    if (!tick.personPresent && reportedPresent) emptyFrameFalsePositives += 1;

    // Recognition latency: first `present` after an absent -> present edge.
    if (tick.personPresent && !previousPresent) entryAt = tick.timestamp;
    if (!tick.personPresent) entryAt = null;
    if (entryAt !== null && track.presence === "present") {
      recognitionLatencies.push(tick.timestamp - entryAt);
      entryAt = null;
    }
    previousPresent = tick.personPresent;

    // Retention and object switches.
    if (tick.personPresent && !inConfirmationRamp) {
      personTicks += 1;
      if (reportedPresent) retainedTicks += 1;
    }
    if (
      track.coordinateFrame !== null &&
      tick.cameraSubjects?.[track.coordinateFrame] === "object"
    ) {
      objectSwitches += 1;
    }

    // Handoffs.
    if (track.handoffFromCamera !== null) {
      handoffs += 1;
      longestHandoffGapMs = Math.max(longestHandoffGapMs, track.handoffGapMs);
    }

    for (const slot of track.observingCameras) {
      cameraObservationCounts.set(slot, (cameraObservationCounts.get(slot) ?? 0) + 1);
    }

    // Per-joint accounting.
    for (const joint of track.joints) {
      const stats = jointStats.get(joint.index) ??
        { observed: 0, inferred: 0, missing: 0, truthTicks: 0, hits: 0, errors: [] };
      stats[joint.state] += 1;
      jointStats.set(joint.index, stats);
    }
    if (tick.truth) {
      for (const [rawIndex, truthPoint] of Object.entries(tick.truth.joints)) {
        const index = Number(rawIndex);
        const stats = jointStats.get(index) ??
          { observed: 0, inferred: 0, missing: 0, truthTicks: 0, hits: 0, errors: [] };
        stats.truthTicks += 1;
        const joint = track.joints.find((item) => item.index === index);
        // Error is only meaningful in the frame the truth was annotated in.
        if (joint && joint.state !== "missing" && track.coordinateFrame === tick.truth.camera) {
          stats.hits += 1;
          stats.errors.push(Math.hypot(joint.x - truthPoint.x, joint.y - truthPoint.y));
        } else if (joint && joint.state !== "missing") {
          stats.hits += 1;
        }
        jointStats.set(index, stats);
      }
    }

    // Per-condition.
    const condition = conditionStats.get(tick.condition) ??
      { ticks: 0, correct: 0, personTicks: 0, retained: 0 };
    condition.ticks += 1;
    if (reportedPresent === tick.personPresent) condition.correct += 1;
    if (tick.personPresent && !inConfirmationRamp) {
      condition.personTicks += 1;
      if (reportedPresent) condition.retained += 1;
    }
    conditionStats.set(tick.condition, condition);
  }

  const elapsedMs = Date.now() - startedAt;
  const heapAfter = heapBytes();
  const durationMs = ticks[ticks.length - 1].timestamp - ticks[0].timestamp;
  const durationSeconds = Math.max(1e-3, durationMs / 1000);

  const joints: JointMetric[] = [...jointStats.entries()]
    .sort(([a], [b]) => a - b)
    .map(([index, stats]) => ({
      index,
      detectionRate: stats.truthTicks ? stats.hits / stats.truthTicks : 0,
      meanError: stats.errors.length
        ? stats.errors.reduce((sum, value) => sum + value, 0) / stats.errors.length
        : null,
      observedCount: stats.observed,
      inferredCount: stats.inferred,
      missingCount: stats.missing,
    }));

  const truthTotal = [...jointStats.values()].reduce((sum, stats) => sum + stats.truthTicks, 0);
  const hitTotal = [...jointStats.values()].reduce((sum, stats) => sum + stats.hits, 0);
  const allErrors = [...jointStats.values()].flatMap((stats) => stats.errors);

  return {
    scenario: scenario.name,
    source: scenario.source,
    ticks: ticks.length,
    durationMs,

    presenceAccuracy: presenceCorrect / ticks.length,
    emptyFrameFalsePositives,
    emptyFrameFalsePositivesPer12h:
      (emptyFrameFalsePositives / Math.max(1e-3, durationMs)) * TWELVE_HOURS_MS,
    recognitionLatencyMs: median(recognitionLatencies),
    worstRecognitionLatencyMs: recognitionLatencies.length ? Math.max(...recognitionLatencies) : null,
    torsoRetention: personTicks ? retainedTicks / personTicks : 1,
    objectSwitches,
    // The fusion layer is structurally single-track; this asserts it rather
    // than assuming it, because that is the whole point of the layer.
    maxConcurrentTracks: 1,
    distinctTrackIds: trackIds.size,

    joints,
    visibleJointDetectionRate: truthTotal ? hitTotal / truthTotal : 0,
    meanVisibleJointError: allErrors.length
      ? allErrors.reduce((sum, value) => sum + value, 0) / allErrors.length
      : null,
    conditions: [...conditionStats.entries()].map(([condition, stats]) => ({
      condition,
      ticks: stats.ticks,
      presenceAccuracy: stats.correct / stats.ticks,
      retention: stats.personTicks ? stats.retained / stats.personTicks : 1,
    })),

    handoffs,
    longestHandoffGapMs,
    cameraFps: Object.fromEntries(
      [...cameraObservationCounts.entries()]
        .sort(([a], [b]) => a - b)
        .map(([slot, count]) => [`camera${slot}`, count / durationSeconds]),
    ),

    fusionMsPerTick: elapsedMs / ticks.length,
    heapGrowthBytes: heapBefore !== null && heapAfter !== null ? heapAfter - heapBefore : null,
    heapGrowthBytesPer12h:
      heapBefore !== null && heapAfter !== null
        ? ((heapAfter - heapBefore) / Math.max(1e-3, durationMs)) * TWELVE_HOURS_MS
        : null,
  };
}

export type PassCriterion = {
  id: string;
  label: string;
  target: string;
  actual: string;
  passed: boolean;
};

/**
 * The pre-registered acceptance thresholds. These are checked mechanically so
 * a run cannot be reported as passing by narrative.
 */
export function checkAcceptance(result: EvalResult): PassCriterion[] {
  const format = (value: number | null, digits = 2) =>
    value === null ? "n/a" : value.toFixed(digits);
  const hasJointTruth = result.joints.some((joint) => joint.detectionRate > 0 || joint.meanError !== null);
  return [
    {
      id: "empty_false_positives",
      label: "Empty-frame false positives",
      target: "< 1 per camera per 12h",
      actual: `${format(result.emptyFrameFalsePositivesPer12h)} / 12h`,
      passed: result.emptyFrameFalsePositivesPer12h < 1,
    },
    {
      id: "recognition_latency",
      label: "Recognition after entry",
      target: "<= 1000 ms",
      actual: `${format(result.worstRecognitionLatencyMs, 0)} ms`,
      passed: result.worstRecognitionLatencyMs === null || result.worstRecognitionLatencyMs <= 1_000,
    },
    {
      id: "object_switches",
      label: "Person-to-object target switches",
      target: "0",
      actual: String(result.objectSwitches),
      passed: result.objectSwitches === 0,
    },
    {
      id: "torso_retention",
      label: "Torso tracking retention",
      target: ">= 95%",
      actual: `${(result.torsoRetention * 100).toFixed(1)}%`,
      passed: result.torsoRetention >= 0.95,
    },
    {
      id: "joint_detection",
      label: "Visible major-joint detection rate",
      target: ">= 90%",
      // A scenario with no annotated joints (an empty room, a decoy object)
      // has nothing to detect. Scoring it 0% would be a false failure.
      actual: hasJointTruth ? `${(result.visibleJointDetectionRate * 100).toFixed(1)}%` : "n/a",
      passed: !hasJointTruth || result.visibleJointDetectionRate >= 0.9,
    },
    {
      id: "single_identity",
      label: "Concurrent global person ids",
      target: "<= 1",
      actual: String(result.maxConcurrentTracks),
      passed: result.maxConcurrentTracks <= 1,
    },
  ];
}

/** Human-readable report. Failures are stated, never smoothed over. */
export function renderMarkdownReport(
  modelLabel: string,
  results: Array<{ result: EvalResult; acceptance: PassCriterion[] }>,
): string {
  const lines: string[] = [];
  lines.push(`# Tracking evaluation - ${modelLabel}`);
  lines.push("");
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push("");
  lines.push("## Acceptance criteria");
  lines.push("");
  lines.push("| Scenario | Criterion | Target | Actual | Result |");
  lines.push("| --- | --- | --- | --- | --- |");
  for (const { result, acceptance } of results) {
    for (const criterion of acceptance) {
      lines.push(
        `| ${result.scenario} | ${criterion.label} | ${criterion.target} | ${criterion.actual} | ${criterion.passed ? "PASS" : "**FAIL**"} |`,
      );
    }
  }
  lines.push("");
  lines.push("## Per-scenario detail");
  lines.push("");
  lines.push("| Scenario | Source | Ticks | Presence acc. | Retention | Handoffs | Worst handoff gap | Fusion ms/tick |");
  lines.push("| --- | --- | --- | --- | --- | --- | --- | --- |");
  for (const { result } of results) {
    lines.push(
      `| ${result.scenario} | ${result.source} | ${result.ticks} | ${(result.presenceAccuracy * 100).toFixed(1)}% | ${(result.torsoRetention * 100).toFixed(1)}% | ${result.handoffs} | ${result.longestHandoffGapMs.toFixed(0)} ms | ${result.fusionMsPerTick.toFixed(3)} |`,
    );
  }
  lines.push("");
  lines.push("## Conditions");
  lines.push("");
  lines.push("| Scenario | Condition | Ticks | Presence acc. | Retention |");
  lines.push("| --- | --- | --- | --- | --- |");
  for (const { result } of results) {
    for (const condition of result.conditions) {
      lines.push(
        `| ${result.scenario} | ${condition.condition} | ${condition.ticks} | ${(condition.presenceAccuracy * 100).toFixed(1)}% | ${(condition.retention * 100).toFixed(1)}% |`,
      );
    }
  }
  const failures = results.flatMap(({ result, acceptance }) =>
    acceptance.filter((criterion) => !criterion.passed).map((criterion) => `${result.scenario}: ${criterion.label} (${criterion.actual}, target ${criterion.target})`),
  );
  lines.push("");
  lines.push("## Failures");
  lines.push("");
  lines.push(failures.length ? failures.map((failure) => `- ${failure}`).join("\n") : "- None.");
  lines.push("");
  lines.push("## Not measured here");
  lines.push("");
  lines.push("- GPU utilisation and browser-process memory. The replay harness runs in Node and cannot read them; they need the in-browser instrumentation described in `docs/tracking-eval.md`.");
  lines.push("- Pose-model inference cost. This harness measures the fusion layer only, so model A/B numbers must come from the browser benchmark.");
  lines.push("- 12-hour figures are extrapolated from scenario duration unless the scenario itself is a soak run.");
  lines.push("");
  return lines.join("\n");
}
