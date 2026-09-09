import { compareSkeletonMotions } from "./motion-classifier";

export type LearnedMotionSample = {
  sessionId: string;
  globalSessionId?: string;
  startMs: number;
  endMs: number;
};

export type LearnedMotionAction = {
  id: string;
  label: string;
  samples: LearnedMotionSample[];
  createdAt: number;
};

export type LoadedLearnedMotionAction = {
  id: string;
  label: string;
  samples: number[][][];
};

export type LearnedMotionResult = {
  status: "matched" | "uncertain" | "insufficient";
  actionId: string | null;
  label: string | null;
  confidence: number;
  candidates: Array<{ actionId: string; label: string; distance: number }>;
};

const STORAGE_KEY = "memory-guard-custom-motions-v1";
const MATCH_CONFIDENCE = 0.52;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function isLearnedMotionAction(value: unknown): value is LearnedMotionAction {
  if (!value || typeof value !== "object") return false;
  const action = value as LearnedMotionAction;
  return typeof action.id === "string"
    && typeof action.label === "string"
    && typeof action.createdAt === "number"
    && Array.isArray(action.samples)
    && action.samples.every((sample) =>
      typeof sample?.sessionId === "string"
      && typeof sample.startMs === "number"
      && typeof sample.endMs === "number",
    );
}

export function loadLearnedMotionActions() {
  if (typeof window === "undefined") return [];
  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "[]") as unknown;
    return Array.isArray(parsed) ? parsed.filter(isLearnedMotionAction) : [];
  } catch {
    return [];
  }
}

export function saveLearnedMotionActions(actions: LearnedMotionAction[]) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(actions));
}

export function clearLearnedMotionActions() {
  if (typeof window !== "undefined") window.localStorage.removeItem(STORAGE_KEY);
}

function closestExamplesDistance(values: number[]) {
  const closest = [...values].sort((a, b) => a - b).slice(0, 2);
  return closest.reduce((sum, value) => sum + value, 0) / closest.length;
}

export function classifyLearnedMotion(
  observedFrames: number[][],
  actions: LoadedLearnedMotionAction[],
): LearnedMotionResult {
  const candidates = actions
    .map((action) => {
      const distances = action.samples
        .map((sample) => compareSkeletonMotions(observedFrames, sample))
        .filter((distance): distance is number => distance !== null);
      return distances.length > 0
        ? { actionId: action.id, label: action.label, distance: closestExamplesDistance(distances) }
        : null;
    })
    .filter((candidate): candidate is NonNullable<typeof candidate> => Boolean(candidate))
    .sort((a, b) => a.distance - b.distance);

  if (candidates.length === 0) {
    return { status: "insufficient", actionId: null, label: null, confidence: 0, candidates: [] };
  }

  const best = candidates[0];
  const runnerUpDistance = candidates[1]?.distance;
  const shapeScore = Math.exp(-best.distance * 4.2);
  const margin = runnerUpDistance
    ? clamp((runnerUpDistance - best.distance) / runnerUpDistance, 0, 1)
    : 0.5;
  const confidence = clamp(0.18 + shapeScore * 0.58 + margin * 0.24, 0.05, 0.99);

  return {
    status: confidence >= MATCH_CONFIDENCE ? "matched" : "uncertain",
    actionId: best.actionId,
    label: best.label,
    confidence,
    candidates: candidates.slice(0, 3),
  };
}
