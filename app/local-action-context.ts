import type { TargetMotionSlice } from "./motion-segmentation";

export type VisualActionKind = "task" | "phone" | "interaction" | "other";
export type VisualActionSample = {
  timeMs: number;
  label: string;
  kind: VisualActionKind;
  similarity: number;
  margin: number;
  bodyObserved: boolean;
  candidates?: Array<{ label: string; kind: VisualActionKind; similarity: number }>;
  inputGeometry?: { width: number; height: number; left: number; top: number; size: number; transform: "center_crop" };
  workAreaCoverage?: number | null;
};
export type VisualActionInterval = {
  startMs: number;
  endMs: number;
  label: string;
  kind: VisualActionKind;
  similarity: number;
  margin: number;
  sampleCount: number;
};
export type ContextMotionSlice = TargetMotionSlice & { visualEvidence?: VisualActionInterval; contextNonWork?: boolean };

export function visualWorkAreaCoverage(points: Array<{ x: number; y: number; visibility?: number }>, geometry: NonNullable<VisualActionSample["inputGeometry"]>) {
  const visible = [11, 12, 13, 14, 15, 16].map(i => points[i]).filter(p => p && (p.visibility ?? 0) >= 0.35 && Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!visible.length || !(geometry.width > 0 && geometry.height > 0 && geometry.size > 0)) return null;
  return visible.filter(p => p.x * geometry.width >= geometry.left && p.x * geometry.width <= geometry.left + geometry.size
    && p.y * geometry.height >= geometry.top && p.y * geometry.height <= geometry.top + geometry.size).length / visible.length;
}

// Conservative operating points, pending calibration on independently labelled cafe data.
export function isSupportedVisualSample(sample: VisualActionSample) {
  return sample.bodyObserved && Number.isFinite(sample.timeMs) && Number.isFinite(sample.similarity)
    && Number.isFinite(sample.margin) && sample.similarity >= 0.28 && sample.margin >= 0.04;
}

/** Require sustained agreement; isolated frames never establish an action. */
export function visualActionIntervals(samples: VisualActionSample[], window: { startMs: number; endMs: number }): VisualActionInterval[] {
  const runs: VisualActionSample[][] = [];
  let current: VisualActionSample[] = [];
  for (const sample of [...samples].sort((a, b) => a.timeMs - b.timeMs)) {
    if (sample.timeMs < window.startMs || sample.timeMs > window.endMs) continue;
    const previous = current.at(-1);
    if (previous && sample.timeMs <= previous.timeMs) continue;
    if (!isSupportedVisualSample(sample) || (previous && (previous.label !== sample.label || previous.kind !== sample.kind || sample.timeMs - previous.timeMs > 3500))) {
      if (current.length) runs.push(current);
      current = [];
    }
    if (isSupportedVisualSample(sample)) current.push(sample);
  }
  if (current.length) runs.push(current);
  return runs.filter(run => run.length >= 3 && run.at(-1)!.timeMs - run[0].timeMs >= 2000).map(run => ({
    startMs: Math.max(window.startMs, run[0].timeMs - 500),
    endMs: Math.min(window.endMs, run.at(-1)!.timeMs + 500),
    label: run[0].label, kind: run[0].kind,
    similarity: Math.min(...run.map(s => s.similarity)), margin: Math.min(...run.map(s => s.margin)), sampleCount: run.length,
  }));
}

/** Bound task candidates by observed context, retaining unknown movement for review. */
export function contextualMotionSlices(rawFrames: number[][], fallback: TargetMotionSlice[], intervals: VisualActionInterval[]): ContextMotionSlice[] {
  if (!intervals.length) return fallback;
  if (fallback.some(slice => slice.boundaryEvidence)) {
    // Image similarity supplies a label; it must not expand an observed start/return boundary.
    return fallback.map(slice => {
      const nonWorkMs = intervals.filter(i => i.kind === "other" && ["설정·기타", "사람 없는 화면"].includes(i.label))
        .reduce((sum, i) => sum + Math.max(0, Math.min(i.endMs, slice.endMs) - Math.max(i.startMs, slice.startMs)), 0);
      if (nonWorkMs >= (slice.endMs - slice.startMs) * 0.6) return { ...slice, contextNonWork: true };
      const support = intervals.filter(i => i.kind === "task" && Math.min(i.endMs, slice.endMs) - Math.max(i.startMs, slice.startMs) >= 500);
      const labels = [...new Set(support.map(i => i.label))];
      if (labels.length !== 1) return slice;
      const matching = support.filter(i => i.label === labels[0]);
      const covered = matching.reduce((sum, i) => sum + Math.min(i.endMs, slice.endMs) - Math.max(i.startMs, slice.startMs), 0);
      if (covered < Math.max(1000, (slice.endMs - slice.startMs) * 0.35)) return slice;
      return { ...slice, visualEvidence: { ...matching[0], startMs: slice.startMs, endMs: slice.endMs,
        similarity: Math.min(...matching.map(i => i.similarity)), margin: Math.min(...matching.map(i => i.margin)),
        sampleCount: matching.reduce((sum, i) => sum + i.sampleCount, 0),
      } };
    });
  }
  const tasks: VisualActionInterval[] = [];
  for (const interval of intervals) {
    if (interval.kind !== "task") continue;
    const previous = tasks.at(-1);
    const between = previous && intervals.filter(i => i.startMs >= previous.endMs && i.endMs <= interval.startMs);
    if (previous && previous.label === interval.label && interval.startMs - previous.endMs <= 30_000
      && between && between.length > 0 && between.every(i => i.kind === "phone" || i.kind === "interaction" || i.label === "정지·휴식 원인 미확인")) {
      previous.endMs = interval.endMs;
      previous.sampleCount += interval.sampleCount;
      previous.similarity = Math.min(previous.similarity, interval.similarity);
      previous.margin = Math.min(previous.margin, interval.margin);
    } else tasks.push({ ...interval });
  }
  const slices: ContextMotionSlice[] = [];
  const originalDurationSeconds = Math.max(0, ((rawFrames.at(-1)?.[0] ?? 0) - (rawFrames[0]?.[0] ?? 0)) / 1000);
  const add = (startMs: number, endMs: number, visualEvidence?: VisualActionInterval) => {
    const frames = rawFrames.filter(f => f[0] >= startMs && f[0] <= endMs);
    if (frames.length < 6) return;
    const first = frames[0][0], last = frames.at(-1)![0];
    slices.push({ frames, startMs: first, endMs: last, durationSeconds: (last - first) / 1000,
      originalDurationSeconds, excludedFrameCount: rawFrames.length - frames.length,
      reason: visualEvidence ? "연속된 로컬 영상 맥락 후보와 스켈레톤으로 구간을 구성했어요. 업무 정답으로 검증된 경계는 아닙니다." : "영상 맥락이 불확실한 움직임 구간으로 검토가 필요해요.",
      ...(visualEvidence ? { visualEvidence } : {}),
    });
  };
  // Objects in a stationary scene are not evidence that the person performed a task.
  const observedTasks = tasks.flatMap(task => {
    const overlaps = fallback.map(slice => ({ startMs: Math.max(task.startMs, slice.startMs), endMs: Math.min(task.endMs, slice.endMs) }))
      .filter(range => range.endMs - range.startMs >= 500);
    return overlaps.length ? [{ ...task, startMs: Math.min(...overlaps.map(i => i.startMs)), endMs: Math.max(...overlaps.map(i => i.endMs)) }] : [];
  });
  observedTasks.forEach(task => add(task.startMs, task.endMs, task));
  const reserved = [...observedTasks, ...intervals.filter(i => i.kind !== "task")].sort((a, b) => a.startMs - b.startMs);
  for (const slice of fallback) {
    let cursor = slice.startMs;
    for (const block of reserved) {
      if (block.endMs < cursor || block.startMs > slice.endMs) continue;
      if (block.startMs > cursor) add(cursor, Math.min(slice.endMs, block.startMs - 1));
      cursor = Math.max(cursor, block.endMs + 1);
    }
    if (cursor <= slice.endMs) add(cursor, slice.endMs);
  }
  return slices.sort((a, b) => a.startMs - b.startMs);
}

export function rankVisualContext(embedding: ArrayLike<number>, prototypes: Array<{ label: string; kind: VisualActionKind; embedding: number[] }>, taskLabels?: string[]) {
  if (embedding.length !== 512 || !Array.from(embedding).every(Number.isFinite)) throw new Error("Invalid CLIP image embedding");
  const norm = Math.hypot(...Array.from(embedding));
  if (!norm) throw new Error("Empty CLIP image embedding");
  const scores = prototypes.filter(p => p.kind !== "task" || !taskLabels || taskLabels.includes(p.label)).map(p => {
    if (p.embedding.length !== 512 || !p.embedding.every(Number.isFinite)) throw new Error("Invalid CLIP text prototype");
    const similarity = p.embedding.reduce((sum, value, i) => sum + value * embedding[i] / norm, 0);
    return { label: p.label, kind: p.kind, similarity };
  }).sort((a, b) => b.similarity - a.similarity);
  if (scores.length < 2) throw new Error("CLIP requires task and alternative prototypes");
  return { ...scores[0], margin: scores[0].similarity - scores[1].similarity, candidates: scores };
}
