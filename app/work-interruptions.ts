import { parseSessionFrame } from "./pose-store";

export type WorkInterruption = {
  kind: "raised_hand_hold" | "stillness" | "tracking_gap" | "visual_phone_candidate" | "visual_interaction_candidate";
  startMs: number;
  endMs: number;
};

export const INTERRUPTION_LABELS: Record<WorkInterruption["kind"], string> = {
  raised_hand_hold: "손을 머리 쪽에 유지 · 전화 등 중단 의심",
  stillness: "원인 미확인 정지 · 분석 신호로 유지",
  tracking_gap: "좌표 누락 · 업무 중단 여부 미확인",
  visual_phone_candidate: "로컬 영상 모델의 전화 사용 후보 · 통화 확정 아님",
  visual_interaction_candidate: "로컬 영상 모델의 고객 응대 후보 · 원인 확정 아님",
};

export type InterruptionSummary = {
  intervals: WorkInterruption[];
  suspectedSeconds: number;
  unobservedSeconds: number;
  unexplainedPauseSeconds?: number;
  remainingSeconds: number;
};

export function hasInterruptionConfound(summary?: InterruptionSummary) {
  return summary?.intervals.some((item) => item.kind !== "stillness") ?? false;
}

/** Observable posture cues only: skeletons do not establish a phone call or customer interaction. */
export function detectWorkInterruptions(rawFrames: number[][], recordingWindow?: { startMs: number; endMs: number }): WorkInterruption[] {
  const frames = rawFrames.map(parseSessionFrame).sort((a, b) => a.relativeTimeMs - b.relativeTimeMs);
  const results: WorkInterruption[] = [];
  let holdStart: number | null = null;
  let quietStart: number | null = null;
  let previous: typeof frames[number] | null = null;
  let firstUsableTime: number | null = null;
  function flush(endMs: number) {
    if (holdStart !== null && endMs - holdStart >= 2000) results.push({ kind: "raised_hand_hold", startMs: holdStart, endMs });
    if (quietStart !== null && endMs - quietStart >= 3000) results.push({ kind: "stillness", startMs: quietStart, endMs });
    holdStart = quietStart = null;
  }
  const visible = (body: number[], joint: number) => (body[joint * 4 + 3] ?? 0) >= 0.5;
  const point = (body: number[], joint: number) => [body[joint * 4], body[joint * 4 + 1]];
  const distance = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  for (const frame of frames) {
    const time = frame.relativeTimeMs;
    const observed = (joint: number) => visible(frame.body, joint) && point(frame.body, joint).every(Number.isFinite);
    const usable = frame.bodyDetected && [0, 1].every(observed) && [4, 5].some(observed);
    if (!usable) continue;
    firstUsableTime ??= time;
    const gap = previous ? time - previous.relativeTimeMs : 0;
    if (gap > 1200) {
      if (previous) {
        flush(previous.relativeTimeMs);
        if (gap > 1200) results.push({ kind: "tracking_gap", startMs: previous.relativeTimeMs, endMs: time });
      }
      previous = null;
    }
    const width = Math.max(0.06, distance(point(frame.body, 0), point(frame.body, 1)));
    const raised = [0, 1].some((side) => {
      if (!observed(side + 4)) return false;
      const shoulder = point(frame.body, side);
      const wrist = point(frame.body, side + 4);
      return wrist[1] < shoulder[1] - width * 0.2 && distance(wrist, shoulder) < width * 0.95;
    });
    if (raised) holdStart ??= time;
    else {
      if (holdStart !== null && time - holdStart >= 2000) results.push({ kind: "raised_hand_hold", startMs: holdStart, endMs: time });
      holdStart = null;
    }
    if (previous) {
      const dt = Math.max(0.01, (time - previous.relativeTimeMs) / 1000);
      const bothWristsObserved = [4, 5].every((joint) => observed(joint) && visible(previous!.body, joint));
      const speed = bothWristsObserved ? Math.max(...[0, 1, 4, 5].map((joint) => distance(point(frame.body, joint), point(previous!.body, joint)))) / width / dt : Infinity;
      if (speed < 0.12) quietStart ??= previous.relativeTimeMs;
      else {
        if (quietStart !== null && previous.relativeTimeMs - quietStart >= 3000) results.push({ kind: "stillness", startMs: quietStart, endMs: previous.relativeTimeMs });
        quietStart = null;
      }
    }
    previous = frame;
  }
  if (previous) flush(previous.relativeTimeMs);
  const lastTime = recordingWindow?.endMs ?? frames.at(-1)?.relativeTimeMs;
  const firstTime = recordingWindow?.startMs ?? frames[0]?.relativeTimeMs;
  if (firstTime !== undefined && lastTime !== undefined && (firstUsableTime ?? lastTime) - firstTime > 1200) {
    results.push({ kind: "tracking_gap", startMs: firstTime, endMs: firstUsableTime ?? lastTime });
  }
  if (previous && lastTime !== undefined && lastTime - previous.relativeTimeMs > 1200) results.push({ kind: "tracking_gap", startMs: previous.relativeTimeMs, endMs: lastTime });
  return results.sort((a, b) => a.startMs - b.startMs);
}

function unionMs(intervals: WorkInterruption[]) {
  let total = 0;
  let end = -Infinity;
  for (const item of [...intervals].sort((a, b) => a.startMs - b.startMs)) {
    total += Math.max(0, item.endMs - Math.max(end, item.startMs));
    end = Math.max(end, item.endMs);
  }
  return total;
}

/** Older summaries subtracted unexplained pauses along with external confounds. */
export function normalizeInterruptionSummary(summary: InterruptionSummary): InterruptionSummary {
  if (summary.unexplainedPauseSeconds !== undefined) return summary;
  const confounds = summary.intervals.filter((item) => item.kind !== "stillness");
  return {
    ...summary,
    suspectedSeconds: unionMs(summary.intervals.filter((item) => item.kind !== "stillness" && item.kind !== "tracking_gap")) / 1000,
    unobservedSeconds: unionMs(summary.intervals.filter((item) => item.kind === "tracking_gap")) / 1000,
    unexplainedPauseSeconds: unionMs(summary.intervals.filter((item) => item.kind === "stillness")) / 1000,
    remainingSeconds: summary.remainingSeconds + (unionMs(summary.intervals) - unionMs(confounds)) / 1000,
  };
}

export function summarizeInterruptions(intervals: WorkInterruption[], startMs: number, endMs: number): InterruptionSummary {
  const clipped = intervals.map((item) => ({ ...item, startMs: Math.max(startMs, item.startMs), endMs: Math.min(endMs, item.endMs) }))
    .filter((item) => item.endMs > item.startMs);
  return {
    intervals: clipped,
    suspectedSeconds: unionMs(clipped.filter((item) => item.kind !== "stillness" && item.kind !== "tracking_gap")) / 1000,
    unobservedSeconds: unionMs(clipped.filter((item) => item.kind === "tracking_gap")) / 1000,
    unexplainedPauseSeconds: unionMs(clipped.filter((item) => item.kind === "stillness")) / 1000,
    remainingSeconds: Math.max(0, endMs - startMs - unionMs(clipped.filter((item) => item.kind !== "stillness"))) / 1000,
  };
}
