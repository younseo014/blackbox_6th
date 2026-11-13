import { compareSkeletonMotions } from "./motion-classifier";
import { compareWorkSequences, representativeWorkSamples, resampleWorkSequence, workPoseQuality, workSequence } from "./work-motion-features";

export type LearnedMotionSample = {
  episodeId?: string;
  labelSource?: "manual" | "visual_model" | "sequence_model";
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
  boundarySamples?: number[][][];
  sampleSources?: Array<LearnedMotionSample["labelSource"]>;
};

export function workReferenceSamples(action: LoadedLearnedMotionAction) {
  if (action.boundarySamples?.length) return {
    samples: [...action.boundarySamples, ...action.samples.filter((_, i) => action.sampleSources?.[i] === "visual_model")],
    independent: true, preferredIndexes: action.boundarySamples.map((_, i) => i),
  };
  const samples = action.samples.filter((_, i) => action.sampleSources?.[i] !== "sequence_model");
  const independent = action.samples.map((frames, i) => ({ frames, source: action.sampleSources?.[i] }))
    .filter(s => s.source === "manual" || s.source === "visual_model");
  return independent.length ? { samples: independent.map(s => s.frames), independent: true,
    preferredIndexes: independent.flatMap((s, i) => s.source === "manual" ? [i] : []) }
    : { samples, independent: false, preferredIndexes: [] };
}

const referenceCache = new WeakMap<LoadedLearnedMotionAction, ReturnType<typeof workReferenceSamples> & { indexes: number[] }>();
export function selectedWorkReferences(action: LoadedLearnedMotionAction) {
  let selected = referenceCache.get(action);
  if (!selected) {
    const pool = workReferenceSamples(action);
    selected = { ...pool, indexes: representativeWorkSamples(pool.samples, pool.preferredIndexes) };
    referenceCache.set(action, selected);
  }
  return selected;
}

export type WorkMotionResult = LearnedMotionResult & {
  independentSupport: boolean;
  candidateMargin: number | null;
  referenceCounts: Array<{ label: string; available: number; selected: number }>;
  observedQuality: number;
};

/** Re-score every registered label on the same completed interval, independent of its proposer. */
export function classifyWorkMotion(observedFrames: number[][], actions: LoadedLearnedMotionAction[], expectedLabels = actions.map(a => a.label)): WorkMotionResult {
  const observed = resampleWorkSequence(workSequence(observedFrames));
  const referenceCounts: WorkMotionResult["referenceCounts"] = [];
  const support = new Map<string, boolean>();
  const candidates = actions.flatMap(action => {
    const pool = selectedWorkReferences(action), indexes = pool.indexes;
    referenceCounts.push({ label: action.label, available: pool.samples.length, selected: indexes.length });
    const distances = indexes.map(i => compareWorkSequences(observed, resampleWorkSequence(workSequence(pool.samples[i])))).filter(Number.isFinite);
    support.set(action.label, pool.independent);
    return distances.length ? [{ actionId: action.id, label: action.label, distance: closestExamplesDistance(distances) }] : [];
  }).sort((a, b) => a.distance - b.distance || a.label.localeCompare(b.label));
  const best = candidates[0], runner = candidates[1];
  const candidateMargin = best && runner ? Math.exp(-best.distance * 4.2) - Math.exp(-runner.distance * 4.2) : null;
  const covered = expectedLabels.every(label => candidates.some(c => c.label === label));
  const confidence = best ? Math.exp(-best.distance * 4.2) * workPoseQuality(observed) : 0;
  return {
    status: !best || observed.length < 6 ? "insufficient"
      : covered && confidence >= 0.55 && candidateMargin !== null && candidateMargin >= 0.04 ? "matched" : "uncertain",
    actionId: best?.actionId ?? null, label: best?.label ?? null, confidence,
    candidates, independentSupport: best ? support.get(best.label) === true : false, candidateMargin, referenceCounts, observedQuality: workPoseQuality(observed),
  };
}

export type LearnedMotionResult = {
  status: "matched" | "uncertain" | "insufficient";
  actionId: string | null;
  label: string | null;
  confidence: number;
  candidates: Array<{ actionId: string; label: string; distance: number }>;
};

const STORAGE_KEY = "memory-guard-custom-motions-v1";
const OBSERVATION_STORAGE_KEY = "memory-guard-observation-motion-labels-v1";
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

function loadActions(storageKey: string) {
  if (typeof window === "undefined") return [];
  try {
    const parsed = JSON.parse(window.localStorage.getItem(storageKey) ?? "[]") as unknown;
    return Array.isArray(parsed) ? parsed.filter(isLearnedMotionAction) : [];
  } catch {
    return [];
  }
}

export function loadLearnedMotionActions() {
  return loadActions(STORAGE_KEY);
}

export function saveLearnedMotionActions(actions: LearnedMotionAction[]) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(actions));
}

export function clearLearnedMotionActions() {
  if (typeof window !== "undefined") window.localStorage.removeItem(STORAGE_KEY);
}

export function loadObservationMotionActions() {
  return loadActions(OBSERVATION_STORAGE_KEY);
}

export function saveObservationMotionActions(actions: LearnedMotionAction[]) {
  window.localStorage.setItem(OBSERVATION_STORAGE_KEY, JSON.stringify(actions));
}

export function clearObservationMotionActions() {
  if (typeof window !== "undefined") window.localStorage.removeItem(OBSERVATION_STORAGE_KEY);
}

export function addObservationMotionSample(
  actions: LearnedMotionAction[],
  label: string,
  sample: LearnedMotionSample,
  createdAt = Date.now(),
) {
  actions = removeObservationMotionSample(actions, sample);
  const normalizedLabel = label.trim().toLocaleLowerCase();
  const matching = actions.find((action) => action.label.trim().toLocaleLowerCase() === normalizedLabel);
  if (!matching) {
    return [...actions, {
      id: `observed-${crypto.randomUUID()}`,
      label: label.trim(),
      samples: [sample],
      createdAt,
    }];
  }
  const isSameSlice = (item: LearnedMotionSample) =>
    item.sessionId === sample.sessionId &&
    item.startMs === sample.startMs &&
    item.endMs === sample.endMs;
  return actions.map((action) => action.id === matching.id
    ? { ...action, samples: [...action.samples.filter((item) => !isSameSlice(item)), sample] }
    : action);
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
        .filter((distance): distance is number => distance !== null && Number.isFinite(distance));
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
  const margin = runnerUpDistance !== undefined
    ? runnerUpDistance > 0 ? clamp((runnerUpDistance - best.distance) / runnerUpDistance, 0, 1) : 0
    : 0.5;
  const confidence = clamp(0.18 + shapeScore * 0.58 + margin * 0.24, 0.05, 0.99);

  return {
    status: confidence >= MATCH_CONFIDENCE && (runnerUpDistance === undefined || runnerUpDistance - best.distance > 1e-6) ? "matched" : "uncertain",
    actionId: best.actionId,
    label: best.label,
    confidence,
    candidates: candidates.slice(0, 3),
  };
}

export function removeObservationMotionSample(actions: LearnedMotionAction[], sample: LearnedMotionSample) {
  return actions.map((action) => ({ ...action, samples: action.samples.filter((item) =>
    !(sample.episodeId && item.episodeId === sample.episodeId)
    && !(item.sessionId === sample.sessionId && item.startMs === sample.startMs && item.endMs === sample.endMs),
  ) })).filter((action) => action.samples.length > 0);
}
