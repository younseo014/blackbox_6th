import { parseSessionFrame } from "./pose-store";
import type { LoadedLearnedMotionAction } from "./custom-motion-training";
import { sliceTargetMotions, SELF_TEST_MOTION_SEGMENTATION, type TargetMotionSlice } from "./motion-segmentation";
import { segmentArmWorkEpisodes } from "./work-episode-segmentation";

type Pose = { time: number; values: number[]; mask: boolean[] };
type Match = { startMs: number; endMs: number; distance: number; label: string };
const SAMPLE_MS = 200;
const MAX_DISTANCE = 0.18;

function sequence(frames: number[][]): Pose[] {
  const result: Pose[] = [];
  for (const raw of frames) {
    const parsed = parseSessionFrame(raw), body = parsed.body, time = parsed.relativeTimeMs;
    if (!parsed.bodyDetected || !Number.isFinite(time) || (result.length && time - result.at(-1)!.time < SAMPLE_MS)) continue;
    const visible = (j: number) => Number.isFinite(body[j * 4]) && Number.isFinite(body[j * 4 + 1]) && body[j * 4 + 3] >= 0.35;
    if (![0, 1].every(visible)) continue;
    const cx = (body[0] + body[4]) / 2, cy = (body[1] + body[5]) / 2;
    const scale = Math.max(0.08, Math.hypot((body[0] - body[4]) * 16 / 9, body[1] - body[5]));
    const values: number[] = [], mask: boolean[] = [];
    for (const joint of [2, 3, 4, 5]) {
      const valid = visible(joint);
      values.push(valid ? (body[joint * 4] - cx) * 16 / 9 / scale : 0, valid ? (body[joint * 4 + 1] - cy) / scale : 0);
      mask.push(valid, valid);
    }
    if (mask.filter(Boolean).length >= 4) result.push({ time, values, mask });
  }
  return result;
}

function cost(a: Pose, b: Pose) {
  let squared = 0, observed = 0;
  for (let i = 0; i < a.values.length; i++) if (a.mask[i] && b.mask[i]) {
    squared += (a.values[i] - b.values[i]) ** 2; observed++;
  }
  return observed >= 4 ? squared / observed : Infinity;
}

/** Open-begin/open-end DTW: locate the whole learned sequence inside a longer recording. */
function subsequences(reference: Pose[], observed: Pose[], label: string): Match[] {
  if (reference.length < 6 || observed.length < 6) return [];
  const duration = reference.at(-1)!.time - reference[0].time;
  if (duration < 700) return [];
  if (![reference[0], reference.at(-1)!].every(p => p.mask.slice(4).every(Boolean))) return [];
  const ranges = reference[0].values.map((_, dimension) => {
    const values = reference.filter(p => p.mask[dimension]).map(p => p.values[dimension]);
    return values.length ? Math.max(...values) - Math.min(...values) : 0;
  });
  const activity = Math.sqrt(ranges.reduce((sum, value) => sum + value ** 2, 0) / ranges.length);
  if (activity < 0.05) return [];
  const maximumDistance = Math.min(MAX_DISTANCE, activity * 0.35);
  const width = observed.length + 1;
  let previous = new Float64Array(width);
  let previousStarts = Int32Array.from({ length: width }, (_, i) => i);
  let previousLengths = new Uint32Array(width);
  for (let i = 1; i <= reference.length; i++) {
    const current = new Float64Array(width).fill(Infinity);
    const starts = new Int32Array(width), lengths = new Uint32Array(width);
    for (let j = 1; j < width; j++) {
      const crossedGap = j > 1 && observed[j - 1].time - observed[j - 2].time > 1200;
      const local = cost(reference[i - 1], observed[j - 1]);
      let best = !crossedGap || i === 1 ? previous[j - 1] : Infinity;
      let begin = previousStarts[j - 1], length = previousLengths[j - 1];
      if (previous[j] + 0.0025 < best) { best = previous[j] + 0.0025; begin = previousStarts[j]; length = previousLengths[j]; }
      if (!crossedGap && current[j - 1] + 0.0025 < best) { best = current[j - 1] + 0.0025; begin = starts[j - 1]; length = lengths[j - 1]; }
      current[j] = local + best;
      starts[j] = begin; lengths[j] = length + 1;
    }
    previous = current; previousStarts = starts; previousLengths = lengths;
  }
  const candidates: Match[] = [];
  for (let j = 1; j < width; j++) {
    const start = previousStarts[j], end = j - 1;
    if (start < 0 || start >= end) continue;
    const span = observed[end].time - observed[start].time;
    const distance = Math.sqrt(previous[j] / previousLengths[j]);
    if (span < duration * 0.5 || span > duration * 3 || distance > maximumDistance) continue;
    if (![observed[start], observed[end]].every(p => p.mask.slice(4).every(Boolean))) continue;
    if (cost(reference[0], observed[start]) > maximumDistance ** 2 || cost(reference.at(-1)!, observed[end]) > maximumDistance ** 2) continue;
    candidates.push({ label, startMs: observed[start].time, endMs: observed[end].time, distance });
  }
  return candidates;
}

const overlapRatio = (a: Match, b: Match) => Math.max(0, Math.min(a.endMs, b.endMs) - Math.max(a.startMs, b.startMs))
  / Math.max(1, Math.min(a.endMs - a.startMs, b.endMs - b.startMs));

export function matchLearnedWorkSequences(rawFrames: number[][], actions: LoadedLearnedMotionAction[]): TargetMotionSlice[] {
  if (!actions.length || rawFrames.length < 6) return [];
  const observed = sequence(rawFrames);
  const matches: Match[] = [];
  const indexAt = (time: number) => {
    let low = 0, high = observed.length;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (observed[middle].time < time) low = middle + 1;
      else high = middle;
    }
    return low;
  };
  // Bound search memory and retain overlap so a task can cross a processing window.
  for (const action of actions) {
    const samples = action.boundarySamples ?? action.samples;
    const indexes = [...new Set([0, Math.floor(samples.length / 2), samples.length - 1])].filter(i => i >= 0);
    for (const index of indexes) {
      let reference = sequence(samples[index]);
      if (reference.length > 150) {
        const source = reference;
        reference = Array.from({ length: 150 }, (_, i) => source[Math.round(i * (source.length - 1) / 149)]);
      }
      if (reference.length < 6) continue;
      const windowMs = Math.max(60_000, (reference.at(-1)!.time - reference[0].time) * 4);
      const hopMs = windowMs / 2;
      for (let from = observed[0]?.time ?? 0; from <= (observed.at(-1)?.time ?? -1); from += hopMs) {
        const window = observed.slice(indexAt(from), indexAt(from + windowMs + 1));
        matches.push(...subsequences(reference, window, action.label));
      }
    }
  }
  const selected: Match[] = [];
  for (const match of matches.sort((a, b) => a.distance - b.distance)) {
    if (selected.some(other => overlapRatio(match, other) > 0.4)) continue;
    selected.push(match);
  }
  return selected.sort((a, b) => a.startMs - b.startMs).map(match => {
    const frames = rawFrames.filter(f => f[0] >= match.startMs && f[0] <= match.endMs);
    const runner = matches.filter(other => other.label !== match.label && overlapRatio(match, other) > 0.4).sort((a, b) => a.distance - b.distance)[0];
    return {
      frames, startMs: match.startMs, endMs: match.endMs, durationSeconds: (match.endMs - match.startMs) / 1000,
      originalDurationSeconds: ((rawFrames.at(-1)?.[0] ?? 0) - (rawFrames[0]?.[0] ?? 0)) / 1000,
      excludedFrameCount: rawFrames.length - frames.length,
      boundaryEvidence: { method: "learned_sequence" as const, startObserved: true, endObserved: true,
        label: match.label, distance: match.distance, margin: runner ? runner.distance - match.distance : 1 },
      reason: "개인 표본의 전체 동작 순서를 연속 기록 안에서 찾아 시작·완료 시각을 정렬했어요.",
    };
  });
}

export function segmentWorkEpisodes(rawFrames: number[][], actions: LoadedLearnedMotionAction[] = []): TargetMotionSlice[] {
  const matched = matchLearnedWorkSequences(rawFrames, actions);
  const lifecycle = segmentArmWorkEpisodes(rawFrames);
  // Do not silently turn movement-only fragments into completed tasks when wrists are hidden.
  const candidates = lifecycle.length ? lifecycle : sliceTargetMotions(rawFrames, SELF_TEST_MOTION_SEGMENTATION).map(slice => ({
    ...slice, boundaryEvidence: { method: "arm_lifecycle" as const, startObserved: false, endObserved: false },
  }));
  return [...matched, ...candidates.filter(slice => !matched.some(match => match.startMs <= slice.endMs && match.endMs >= slice.startMs))]
    .sort((a, b) => a.startMs - b.startMs);
}
