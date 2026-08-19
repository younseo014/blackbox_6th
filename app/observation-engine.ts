import { parseSessionFrame, type BodyProportionProfile } from "./pose-store";
import {
  getOccupationTemplate,
  type OccupationId,
  type PrimitiveMotionLabel,
  type TaskTemplate,
  type WorkPhase,
} from "./occupation-templates";
import type { MotionClassification } from "./motion-classifier";

export type ObservationMode = "learning" | "analysis";
export type LearningDisposition = "accepted" | "quarantined" | "excluded" | "analysis_only";

export type ZoneGrid = Array<string | null>;

export type ObservationProfile = {
  id: "primary";
  occupation: OccupationId;
  mode: ObservationMode;
  learningStartedAt: number;
  baselineVersion: number;
  baselineSource?: "real" | "synthetic";
  syntheticDatasetId?: string | null;
  activeTestTaskId?: string | null;
  bodyProportionProfile?: BodyProportionProfile | null;
  zoneGrid: ZoneGrid;
  updatedAt: number;
};

export type ObservationFeatures = {
  durationSeconds: number;
  activeRatio: number;
  pauseCount: number;
  longestPauseSeconds: number;
  pathLength: number;
  routeComplexity: number;
  repetitionCount: number;
  dominantZone: string | null;
  zoneTransitions: number;
};

export type ObservationEpisode = {
  id: string;
  sessionId: string;
  recordedAt: number;
  date: string;
  occupation: OccupationId;
  mode: ObservationMode;
  phase: WorkPhase;
  taskType: string;
  taskLabel: string;
  taskConfidence: number;
  primitiveLabels: PrimitiveMotionLabel[];
  features: ObservationFeatures;
  disposition: LearningDisposition;
  dispositionReason: string;
  durationZScore: number | null;
  pauseZScore: number | null;
  contextWeight: 0 | 1 | 2 | 3;
  baselineVersion: number;
  source?: "real_learning" | "synthetic_training" | "real_analysis" | "performance_test";
  syntheticDatasetId?: string;
  testTargetTaskType?: string;
  motionClassification?: MotionClassification;
  motionSlice?: {
    startMs: number;
    endMs: number;
    durationSeconds: number;
    originalDurationSeconds: number;
    excludedFrameCount: number;
    reason: string;
  };
};

export type TaskBaseline = {
  taskType: string;
  taskLabel: string;
  sampleCount: number;
  meanDuration: number;
  durationSD: number;
  meanLongestPause: number;
  pauseSD: number;
  meanRouteComplexity: number;
};

export type BaselineSnapshot = {
  version: number;
  createdAt: number;
  eligibleDays: number;
  acceptedSamples: number;
  confidence: number;
  tasks: TaskBaseline[];
};

export type AnalysisFeedback = {
  id: string;
  eventId: string;
  mode: ObservationMode;
  verdict: "accurate" | "false_positive";
  reason?: "interaction" | "rest" | "wrong_task" | "camera_error" | "other";
  baselineVersion: number;
  createdAt: number;
};

export const DEFAULT_PROFILE: ObservationProfile = {
  id: "primary",
  occupation: "cafe",
  mode: "learning",
  learningStartedAt: 0,
  baselineVersion: 1,
  baselineSource: "real",
  syntheticDatasetId: null,
  activeTestTaskId: null,
  bodyProportionProfile: null,
  zoneGrid: Array(9).fill(null),
  updatedAt: 0,
};

function average(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function standardDeviation(values: number[], mean: number) {
  if (values.length < 2) return 0;
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
}

export function safeZScore(value: number, mean: number, sd: number): number | null {
  if (!Number.isFinite(value) || !Number.isFinite(mean) || sd < 0.05) return null;
  return (value - mean) / sd;
}

export function buildBaseline(
  episodes: ObservationEpisode[],
  version = 1,
): BaselineSnapshot {
  const accepted = episodes.filter((episode) => episode.disposition === "accepted");
  const taskIds = [...new Set(accepted.map((episode) => episode.taskType))];
  const tasks = taskIds.map((taskType) => {
    const items = accepted.filter((episode) => episode.taskType === taskType);
    const durations = items.map((episode) => episode.features.durationSeconds);
    const pauses = items.map((episode) => episode.features.longestPauseSeconds);
    const complexities = items.map((episode) => episode.features.routeComplexity);
    const meanDuration = average(durations);
    const meanLongestPause = average(pauses);
    return {
      taskType,
      taskLabel: items[0]?.taskLabel ?? taskType,
      sampleCount: items.length,
      meanDuration,
      durationSD: standardDeviation(durations, meanDuration),
      meanLongestPause,
      pauseSD: standardDeviation(pauses, meanLongestPause),
      meanRouteComplexity: average(complexities),
    };
  });
  const eligibleDays = new Set(accepted.map((episode) => episode.date)).size;
  const sampleProgress = Math.min(1, accepted.length / 20);
  const dayProgress = Math.min(1, eligibleDays / 14);
  return {
    version,
    createdAt: Date.now(),
    eligibleDays,
    acceptedSamples: accepted.length,
    confidence: Math.round((sampleProgress * 0.55 + dayProgress * 0.45) * 100),
    tasks,
  };
}

type FlatPoint = { x: number; y: number; z: number };

function primaryWrist(frame: ReturnType<typeof parseSessionFrame>): FlatPoint | null {
  // The current detector and demo generators treat the right hand as the
  // primary work hand. Fall back to the left only when the right is missing.
  const hand = frame.rightHand ?? frame.leftHand;
  if (hand) return { x: hand[0], y: hand[1], z: hand[2] };
  const body = frame.body;
  const leftVisibility = body[4 * 4 + 3] ?? 0;
  const rightVisibility = body[5 * 4 + 3] ?? 0;
  const wristIndex = leftVisibility >= rightVisibility ? 4 : 5;
  if ((body[wristIndex * 4 + 3] ?? 0) < 0.25) return null;
  return {
    x: body[wristIndex * 4],
    y: body[wristIndex * 4 + 1],
    z: body[wristIndex * 4 + 2],
  };
}

function bodyCenter(frame: ReturnType<typeof parseSessionFrame>): { x: number; y: number } | null {
  const body = frame.body;
  const leftHip = 12;
  const rightHip = 13;
  if ((body[leftHip * 4 + 3] ?? 0) < 0.25 || (body[rightHip * 4 + 3] ?? 0) < 0.25) {
    return null;
  }
  return {
    x: (body[leftHip * 4] + body[rightHip * 4]) / 2,
    y: (body[leftHip * 4 + 1] + body[rightHip * 4 + 1]) / 2,
  };
}

function distance(a: FlatPoint, b: FlatPoint) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function gridIndex(point: { x: number; y: number }) {
  const column = Math.min(2, Math.max(0, Math.floor(point.x * 3)));
  const row = Math.min(2, Math.max(0, Math.floor(point.y * 3)));
  return row * 3 + column;
}

export function extractObservationFeatures(rawFrames: number[][], zoneGrid: ZoneGrid): ObservationFeatures {
  if (rawFrames.length === 0) {
    return {
      durationSeconds: 0,
      activeRatio: 0,
      pauseCount: 0,
      longestPauseSeconds: 0,
      pathLength: 0,
      routeComplexity: 0,
      repetitionCount: 0,
      dominantZone: null,
      zoneTransitions: 0,
    };
  }
  const frames = rawFrames.map(parseSessionFrame);
  const points = frames.map(primaryWrist);
  const zoneTrail = frames.map((frame) => {
    const center = bodyCenter(frame);
    return center ? zoneGrid[gridIndex(center)] ?? null : null;
  });
  let pathLength = 0;
  let activeSteps = 0;
  let pauseCount = 0;
  let pauseStartedAt: number | null = null;
  let longestPauseSeconds = 0;
  let directionChanges = 0;
  let previousDx: number | null = null;
  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1];
    const current = points[index];
    if (!previous || !current) continue;
    const dtSeconds = Math.max(0.001, (frames[index].relativeTimeMs - frames[index - 1].relativeTimeMs) / 1000);
    const step = distance(previous, current);
    pathLength += step;
    const speed = step / dtSeconds;
    if (speed >= 0.02) {
      activeSteps += 1;
      if (pauseStartedAt !== null) {
        const duration = (frames[index].relativeTimeMs - pauseStartedAt) / 1000;
        if (duration >= 1) pauseCount += 1;
        longestPauseSeconds = Math.max(longestPauseSeconds, duration);
        pauseStartedAt = null;
      }
    } else if (pauseStartedAt === null) {
      pauseStartedAt = frames[index - 1].relativeTimeMs;
    }
    const dx = current.x - previous.x;
    if (previousDx !== null && Math.abs(dx) > 0.002 && Math.sign(dx) !== Math.sign(previousDx)) {
      directionChanges += 1;
    }
    if (Math.abs(dx) > 0.002) previousDx = dx;
  }
  if (pauseStartedAt !== null) {
    longestPauseSeconds = Math.max(
      longestPauseSeconds,
      (frames[frames.length - 1].relativeTimeMs - pauseStartedAt) / 1000,
    );
  }
  const compactTrail = zoneTrail.filter((zone, index) => zone && zone !== zoneTrail[index - 1]);
  const counts = new Map<string, number>();
  zoneTrail.forEach((zone) => {
    if (zone) counts.set(zone, (counts.get(zone) ?? 0) + 1);
  });
  const dominantZone = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
  return {
    durationSeconds: Math.max(0, (frames[frames.length - 1].relativeTimeMs - frames[0].relativeTimeMs) / 1000),
    activeRatio: points.length > 1 ? activeSteps / (points.length - 1) : 0,
    pauseCount,
    longestPauseSeconds,
    pathLength,
    routeComplexity: compactTrail.length + directionChanges / 3,
    repetitionCount: Math.floor(directionChanges / 2),
    dominantZone,
    zoneTransitions: Math.max(0, compactTrail.length - 1),
  };
}

export function inferTask(
  occupation: OccupationId,
  phase: WorkPhase,
  features: ObservationFeatures,
): { task: TaskTemplate; confidence: number; primitiveLabels: PrimitiveMotionLabel[] } {
  const template = getOccupationTemplate(occupation);
  const candidates = template.tasks.filter((candidate) => candidate.phase === phase);
  const pool = candidates.length ? candidates : template.tasks.filter((candidate) => candidate.phase === "business");
  const scored = pool.map((candidate) => {
    let score = candidate.priority === 1 ? 0.15 : 0;
    if (features.dominantZone && candidate.zones.includes(features.dominantZone)) score += 0.62;
    if (features.pathLength > 0.5 && candidate.motions.includes("WALK")) score += 0.12;
    if (features.repetitionCount >= 2 && candidate.motions.includes("REPETITIVE_ARM")) score += 0.12;
    if (features.longestPauseSeconds >= 5 && candidate.motions.includes("STATIC_PAUSE")) score += 0.08;
    return { candidate, score };
  }).sort((a, b) => b.score - a.score);
  const winner = scored[0] ?? { candidate: template.tasks[0], score: 0.2 };
  const primitiveLabels = new Set<PrimitiveMotionLabel>();
  if (features.pathLength > 0.4) primitiveLabels.add("WALK");
  if (features.zoneTransitions > 0) primitiveLabels.add("ZONE_TRANSITION");
  if (features.longestPauseSeconds >= 1) primitiveLabels.add("STATIC_PAUSE");
  if (features.repetitionCount >= 2) primitiveLabels.add("REPETITIVE_ARM");
  if (features.activeRatio < 0.15) primitiveLabels.add("STAND");
  return {
    task: winner.candidate,
    confidence: Math.min(0.94, Math.max(0.28, winner.score)),
    primitiveLabels: [...primitiveLabels],
  };
}

function contextWeight(features: ObservationFeatures, task: TaskTemplate): 0 | 1 | 2 | 3 {
  if (task.id === "REST") return 0;
  if (features.longestPauseSeconds >= 5 && features.activeRatio >= 0.1) return 3;
  if (features.zoneTransitions > 0 && features.longestPauseSeconds >= 2) return 2;
  return 1;
}

export function createObservationEpisode(args: {
  sessionId: string;
  recordedAt: number;
  profile: ObservationProfile;
  phase: WorkPhase;
  features: ObservationFeatures;
  baseline: BaselineSnapshot;
  motionClassification?: MotionClassification;
  motionSlice?: ObservationEpisode["motionSlice"];
  testTargetTask?: TaskTemplate;
  /** @deprecated Kept for older callers; use motionClassification + testTargetTask. */
  taskOverride?: TaskTemplate;
}): ObservationEpisode {
  const inferred = inferTask(args.profile.occupation, args.phase, args.features);
  const template = getOccupationTemplate(args.profile.occupation);
  const predictedTask = args.motionClassification?.predictedTaskType
    ? template.tasks.find((task) => task.id === args.motionClassification?.predictedTaskType)
    : undefined;
  const legacyOverride = args.taskOverride;
  const inferredTask = predictedTask ?? legacyOverride ?? inferred.task;
  const confidence = args.motionClassification
    ? args.motionClassification.confidence
    : legacyOverride
      ? 0.94
      : inferred.confidence;
  const primitiveLabels = args.motionClassification?.primitiveLabels.length
    ? args.motionClassification.primitiveLabels
    : legacyOverride
      ? legacyOverride.motions
      : inferred.primitiveLabels;
  const taskBaseline = args.baseline.tasks.find((item) => item.taskType === inferredTask.id);
  const durationZScore = taskBaseline
    ? safeZScore(args.features.durationSeconds, taskBaseline.meanDuration, taskBaseline.durationSD)
    : null;
  const pauseZScore = taskBaseline
    ? safeZScore(args.features.longestPauseSeconds, taskBaseline.meanLongestPause, taskBaseline.pauseSD)
    : null;
  let disposition: LearningDisposition = args.profile.mode === "analysis" ? "analysis_only" : "accepted";
  let dispositionReason = args.profile.mode === "analysis"
    ? "분석 모드 기록은 개인 기준선에 자동으로 합치지 않아요."
    : "평소 업무 후보로 개인 기준선에 포함했어요.";
  if (confidence < 0.4) {
    disposition = "excluded";
    dispositionReason = "업무 추정 신뢰도가 낮아 학습과 분석에서 제외했어요.";
  } else if (
    args.profile.mode === "learning" &&
    ((durationZScore !== null && durationZScore >= 1.5) ||
      (pauseZScore !== null && pauseZScore >= 1.5) ||
      args.features.longestPauseSeconds >= 10)
  ) {
    disposition = "quarantined";
    dispositionReason = "평소 흐름으로 확정하기 어려워 기준선 학습에서 잠시 보류했어요.";
  }
  const useFallbackLabel = !args.motionClassification && confidence < 0.55 && inferredTask.fallbackLabel;
  return {
    id: `episode-${crypto.randomUUID()}`,
    sessionId: args.sessionId,
    recordedAt: args.recordedAt,
    date: new Date(args.recordedAt).toISOString().slice(0, 10),
    occupation: args.profile.occupation,
    mode: args.profile.mode,
    phase: args.phase,
    taskType: useFallbackLabel ? `${inferredTask.id}_GENERAL` : inferredTask.id,
    taskLabel: useFallbackLabel ? inferredTask.fallbackLabel! : inferredTask.label,
    taskConfidence: confidence,
    primitiveLabels,
    features: args.features,
    disposition,
    dispositionReason,
    durationZScore,
    pauseZScore,
    contextWeight: contextWeight(args.features, inferredTask),
    baselineVersion: args.profile.baselineVersion,
    source: args.testTargetTask || legacyOverride
      ? "performance_test"
      : args.profile.mode === "analysis"
        ? "real_analysis"
        : "real_learning",
    testTargetTaskType: args.testTargetTask?.id ?? legacyOverride?.id,
    motionClassification: args.motionClassification,
    motionSlice: args.motionSlice,
  };
}

export function qualitativeMotionTags(features: ObservationFeatures, baseline?: TaskBaseline) {
  const durationZ = baseline
    ? safeZScore(features.durationSeconds, baseline.meanDuration, baseline.durationSD)
    : null;
  return {
    speed: durationZ !== null && durationZ >= 1.5 ? "평소보다 느림" : features.activeRatio > 0.55 ? "빠름" : "보통",
    range: features.pathLength > 1.8 ? "큼" : features.pathLength > 0.45 ? "보통" : "작음",
    flow: features.longestPauseSeconds >= 5 ? "중단 후 재개" : features.repetitionCount >= 2 ? "반복됨" : "자연스러움",
  };
}
