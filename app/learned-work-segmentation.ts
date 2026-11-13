import { parseSessionFrame } from "./pose-store";
import type { LoadedLearnedMotionAction } from "./custom-motion-training";
import { sliceTargetMotions, SELF_TEST_MOTION_SEGMENTATION, type TargetMotionSlice } from "./motion-segmentation";
import { segmentArmWorkEpisodes } from "./work-episode-segmentation";

import { selectedWorkReferences } from "./custom-motion-training";
import { workSequence as sequence, workPoseCost as cost, WORK_FEATURE_VERSION, WORK_SAMPLE_MS, type WorkPose as Pose } from "./work-motion-features";

type Match = { startMs: number; endMs: number; distance: number; label: string; reference?: Pose[] };
const MAX_DISTANCE = 0.18;
export const WORK_RECOGNITION_VERSION = "work-recognition-v3";
export type WorkCandidateTrace = {
  id: string; stage: "generated" | "selected" | "alternative" | "refined" | "residual" | "incomplete";
  startMs: number; endMs: number; label?: string; distance?: number; reason: string;
};
export type WorkSegmentationReport = {
  version: string; featureVersion: string;
  tracking?: { storedFrames: number; bodyFrames: number; handFrames: number; gaps: Array<{ startMs: number; endMs: number }> };
  references: Array<{ label: string; available: number; selected: number; independent: boolean }>;
  candidates: WorkCandidateTrace[];
};

/** Open-begin/open-end DTW: locate the whole learned sequence inside a longer recording. */
function subsequences(reference: Pose[], observed: Pose[], label: string): Match[] {
  if (reference.length < 6 || observed.length < 6) return [];
  const duration = reference.at(-1)!.time - reference[0].time;
  if (duration < 700) return [];
  if (![reference[0], reference.at(-1)!].every(p => p.wrists.every(Boolean))) return [];
  const activity = Math.max(...[[0, 8], [10, 30]].map(([from, to]) => {
    const ranges = reference[0].values.slice(from, to).map((_, offset) => {
      const values = reference.filter(p => p.weights[from + offset]).map(p => p.values[from + offset]);
      return values.length ? Math.max(...values) - Math.min(...values) : 0;
    });
    return Math.sqrt(ranges.reduce((sum, value) => sum + value ** 2, 0) / ranges.length);
  }));
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
    if (span < 700 || distance > maximumDistance) continue;
    if (![observed[start], observed[end]].every(p => p.wrists.every(Boolean))) continue;
    if (cost(reference[0], observed[start]) > maximumDistance ** 2 || cost(reference.at(-1)!, observed[end]) > maximumDistance ** 2) continue;
    candidates.push({ label, startMs: observed[start].time, endMs: observed[end].time, distance, reference });
  }
  return candidates;
}

const overlaps = (a: Match, b: Match) => Math.min(a.endMs, b.endMs) > Math.max(a.startMs, b.startMs);

/** One person cannot complete two competing work intervals at the same time. */
export function selectWorkCandidates(matches: Match[]): Match[] {
  const sorted = [...matches].sort((a, b) => a.endMs - b.endMs || a.startMs - b.startMs || a.distance - b.distance);
  const previous = sorted.map((match, i) => {
    let low = 0, high = i;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (sorted[middle].endMs <= match.startMs) low = middle + 1;
      else high = middle;
    }
    return low - 1;
  });
  const scores = new Float64Array(sorted.length + 1), take = new Uint8Array(sorted.length);
  for (let i = 0; i < sorted.length; i++) {
    const withMatch = Math.exp(-sorted[i].distance * 4.2) + scores[previous[i] + 1];
    if (withMatch > scores[i]) { scores[i + 1] = withMatch; take[i] = 1; }
    else scores[i + 1] = scores[i];
  }
  const selected: Match[] = [];
  for (let i = sorted.length - 1; i >= 0;) {
    if (take[i]) { selected.push(sorted[i]); i = previous[i]; }
    else i--;
  }
  return selected.reverse();
}

function rawIndexAt(frames: number[][], time: number) {
  let low = 0, high = frames.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (frames[middle][0] < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function refineEndpoints(match: Match, rawFrames: number[][], lower: number, upper: number): Match {
  if (!match.reference?.length) return match;
  const nearest = (time: number, endpoint: Pose) => sequence(rawFrames.slice(
    rawIndexAt(rawFrames, Math.max(lower, time - WORK_SAMPLE_MS)), rawIndexAt(rawFrames, Math.min(upper, time + WORK_SAMPLE_MS) + 0.001)), 0)
    .filter(p => p.wrists.every(Boolean))
    .sort((a, b) => cost(endpoint, a) - cost(endpoint, b) || Math.abs(a.time - time) - Math.abs(b.time - time))[0]?.time ?? time;
  const startMs = nearest(match.startMs, match.reference[0]);
  const endMs = nearest(match.endMs, match.reference.at(-1)!);
  return endMs - startMs >= 700 ? { ...match, startMs, endMs } : match;
}

function findLearnedWork(rawFrames: number[][], actions: LoadedLearnedMotionAction[], report: WorkSegmentationReport): TargetMotionSlice[] {
  if (!actions.length || rawFrames.length < 6) return [];
  const observed = sequence(rawFrames), matches: Match[] = [];
  const indexAt = (time: number) => {
    let low = 0, high = observed.length;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (observed[middle].time < time) low = middle + 1;
      else high = middle;
    }
    return low;
  };
  const lifecycle = segmentArmWorkEpisodes(rawFrames);
  for (const action of actions) {
    const pool = selectedWorkReferences(action);
    report.references.push({ label: action.label, available: pool.samples.length, selected: pool.indexes.length, independent: pool.independent });
    for (const index of pool.indexes) {
      let reference = sequence(pool.samples[index]);
      if (reference.length > 150) {
        const source = reference;
        reference = Array.from({ length: 150 }, (_, i) => source[Math.round(i * (source.length - 1) / 149)]);
      }
      if (reference.length < 6) continue;
      const windowMs = Math.max(60_000, (reference.at(-1)!.time - reference[0].time) * 4);
      for (let from = observed[0]?.time ?? 0; from <= (observed.at(-1)?.time ?? -1); from += windowMs / 2) {
        matches.push(...subsequences(reference, observed.slice(indexAt(from), indexAt(from + windowMs + 1)), action.label));
      }
      // Long, continuously observed work must not disappear because a search window is shorter.
      for (const interval of lifecycle.filter(s => s.endMs - s.startMs > windowMs / 2)) {
        matches.push(...subsequences(reference, observed.slice(indexAt(interval.startMs), indexAt(interval.endMs + 1)), action.label));
      }
    }
  }
  const uniqueByTime = new Map<string, Match>();
  for (const match of matches) {
    const key = `${match.label}:${match.startMs}:${match.endMs}`, previous = uniqueByTime.get(key);
    if (!previous || match.distance < previous.distance) uniqueByTime.set(key, match);
  }
  const unique = [...uniqueByTime.values()], selected = selectWorkCandidates(unique);
  unique.forEach((m, i) => report.candidates.push({ id: `sequence:${i}`, startMs: m.startMs, endMs: m.endMs, label: m.label, distance: m.distance,
    stage: selected.includes(m) ? "selected" : "alternative",
    reason: selected.includes(m) ? "전체 시간축의 비중첩 후보 선택" : "같은 시간대의 경쟁 후보로 보존" }));
  return selected.map((original, i) => {
    const lower = i ? (selected[i - 1].endMs + original.startMs) / 2 : rawFrames[0][0];
    const upper = i + 1 < selected.length ? (original.endMs + selected[i + 1].startMs) / 2 : rawFrames.at(-1)![0];
    const match = refineEndpoints(original, rawFrames, lower, upper);
    if (match.startMs !== original.startMs || match.endMs !== original.endMs) report.candidates.push({ id: `refined:${i}`, stage: "refined",
      startMs: match.startMs, endMs: match.endMs, label: match.label, reason: "원본 관측 프레임으로 끝점 재정렬" });
    const frames = rawFrames.slice(rawIndexAt(rawFrames, match.startMs), rawIndexAt(rawFrames, match.endMs + 0.001));
    const runner = unique.filter(other => other.label !== match.label && overlaps(match, other)).sort((a, b) => a.distance - b.distance)[0];
    return {
      frames, startMs: match.startMs, endMs: match.endMs, durationSeconds: (match.endMs - match.startMs) / 1000,
      originalDurationSeconds: ((rawFrames.at(-1)?.[0] ?? 0) - (rawFrames[0]?.[0] ?? 0)) / 1000,
      excludedFrameCount: rawFrames.length - frames.length,
      boundaryEvidence: { method: "learned_sequence" as const, startObserved: true, endObserved: true,
        label: match.label, distance: match.distance, margin: runner ? runner.distance - match.distance : null,
        searchStartMs: original.startMs, searchEndMs: original.endMs },
      reason: "전체 동작 순서로 구간을 찾고 원본 관측으로 끝점을 확인했어요. 업무명은 모든 등록 후보와 다시 비교합니다.",
    };
  });
}

const newReport = (): WorkSegmentationReport => ({ version: WORK_RECOGNITION_VERSION, featureVersion: WORK_FEATURE_VERSION, references: [], candidates: [] });
export function matchLearnedWorkSequences(rawFrames: number[][], actions: LoadedLearnedMotionAction[]): TargetMotionSlice[] {
  return findLearnedWork(rawFrames, actions, newReport());
}

export function preserveUncoveredWork(candidates: TargetMotionSlice[], matched: TargetMotionSlice[], rawFrames: number[][], report: WorkSegmentationReport) {
  const residual: TargetMotionSlice[] = [];
  for (const [index, slice] of candidates.entries()) {
    report.candidates.push({ id: `motion:${index}`, stage: "generated", startMs: slice.startMs, endMs: slice.endMs, reason: slice.reason });
    let fragments = [{ startMs: slice.startMs, endMs: slice.endMs }];
    for (const match of matched) fragments = fragments.flatMap(f => {
      if (match.endMs < f.startMs || match.startMs > f.endMs) return [f];
      return [ ...(match.startMs > f.startMs ? [{ startMs: f.startMs, endMs: match.startMs - 0.001 }] : []),
        ...(match.endMs < f.endMs ? [{ startMs: match.endMs + 0.001, endMs: f.endMs }] : []) ];
    });
    for (const fragment of fragments) {
      const frames = slice.frames.filter(f => f[0] >= fragment.startMs && f[0] <= fragment.endMs);
      if (frames.length < 6 || frames.at(-1)![0] - frames[0][0] < 700) {
        report.candidates.push({ id: `short:${index}:${fragment.startMs}`, stage: "incomplete", ...fragment, reason: "잔여 관측은 보존했지만 독립 업무를 구성할 프레임 부족" });
        continue;
      }
      const startMs = frames[0][0], endMs = frames.at(-1)![0];
      const unchanged = startMs === slice.startMs && endMs === slice.endMs;
      residual.push({ ...slice, frames, startMs, endMs, durationSeconds: (endMs - startMs) / 1000,
        excludedFrameCount: rawFrames.length - frames.length,
        boundaryEvidence: unchanged ? slice.boundaryEvidence : { method: "arm_lifecycle", startObserved: false, endObserved: false },
        reason: unchanged ? slice.reason : "다른 후보로 설명되지 않은 잔여 움직임입니다. 시작·완료 확인이 필요합니다." });
      report.candidates.push({ id: `residual:${index}:${startMs}`, stage: "residual", startMs, endMs, reason: unchanged ? "독립 움직임 후보 유지" : "겹친 부분만 제외하고 잔여 관측 보존" });
    }
  }
  return residual;
}

export function segmentWorkEpisodesDetailed(rawFrames: number[][], actions: LoadedLearnedMotionAction[] = []) {
  const report = newReport(), matched = findLearnedWork(rawFrames, actions, report);
  report.tracking = { storedFrames: rawFrames.length, bodyFrames: 0, handFrames: 0, gaps: [] };
  rawFrames.forEach((raw, i) => {
    const frame = parseSessionFrame(raw);
    if (frame.bodyDetected) report.tracking!.bodyFrames++;
    if (frame.leftHandDetected || frame.rightHandDetected) report.tracking!.handFrames++;
    if (i && raw[0] - rawFrames[i - 1][0] > 1200) report.tracking!.gaps.push({ startMs: rawFrames[i - 1][0], endMs: raw[0] });
  });
  const lifecycle = segmentArmWorkEpisodes(rawFrames);
  const candidates = lifecycle.length ? lifecycle : sliceTargetMotions(rawFrames, SELF_TEST_MOTION_SEGMENTATION).map(slice => ({
    ...slice, boundaryEvidence: { method: "arm_lifecycle" as const, startObserved: false, endObserved: false },
  }));
  const slices = [...matched, ...preserveUncoveredWork(candidates, matched, rawFrames, report)].sort((a, b) => a.startMs - b.startMs);
  return { slices, report };
}

export function segmentWorkEpisodes(rawFrames: number[][], actions: LoadedLearnedMotionAction[] = []): TargetMotionSlice[] {
  return segmentWorkEpisodesDetailed(rawFrames, actions).slices;
}
