import { parseSessionFrame } from "./pose-store";
import type { TargetMotionSlice } from "./motion-segmentation";

type ArmObservation = { time: number; working: boolean; returned: boolean };

function armObservation(raw: number[]): ArmObservation | null {
  const frame = parseSessionFrame(raw), body = frame.body;
  if (!frame.bodyDetected || !Number.isFinite(frame.relativeTimeMs)) return null;
  const angles = [0, 1].map(side => {
    const elbow = side + 2, wrist = side + 4;
    if ([elbow, wrist, side].some(j => !Number.isFinite(body[j * 4]) || !Number.isFinite(body[j * 4 + 1])
      || !Number.isFinite(body[j * 4 + 3]) || body[j * 4 + 3] < 0.35)) return null;
    const dx = (body[wrist * 4] - body[elbow * 4]) * 16 / 9;
    const dy = body[wrist * 4 + 1] - body[elbow * 4 + 1];
    const length = Math.hypot(dx, dy);
    return Number.isFinite(length) && length > 0.02 ? Math.acos(Math.max(-1, Math.min(1, dy / length))) * 180 / Math.PI : null;
  });
  if (angles.every(a => a === null)) return null;
  return {
    time: frame.relativeTimeMs,
    working: angles.some(a => a !== null && a >= 70),
    // A hidden hand cannot prove that both hands returned from the work.
    returned: angles.every(a => a !== null && a < 60),
  };
}

/** Observe entry, sustained handling, and return rather than splitting at every quiet gap. */
export function segmentArmWorkEpisodes(rawFrames: number[][]): TargetMotionSlice[] {
  if (!rawFrames.length) return [];
  const originalDurationSeconds = (rawFrames.at(-1)![0] - rawFrames[0][0]) / 1000;
  const result: TargetMotionSlice[] = [];
  let previousTime: number | null = null;
  let neutralSince: number | null = null;
  let ready = false;
  let onset: number | null = null;
  let start: number | null = null;
  let hasStart = false;
  let returnSince: number | null = null;
  let trackingGapMs = 0;

  const emit = (end: number, endObserved: boolean) => {
    if (start === null || end - start < 700) return;
    const frames = rawFrames.filter(f => f[0] >= start! && f[0] <= end);
    if (frames.length < 6) return;
    result.push({ frames, startMs: frames[0][0], endMs: frames.at(-1)![0],
      durationSeconds: (frames.at(-1)![0] - frames[0][0]) / 1000, originalDurationSeconds,
      excludedFrameCount: rawFrames.length - frames.length,
      boundaryEvidence: { method: "arm_lifecycle", startObserved: hasStart, endObserved, trackingGapMs },
      reason: !hasStart || !endObserved ? "관측 시작 또는 종료가 업무 중간에 걸려 완전한 1회로 세지 않는 구간입니다."
        : trackingGapMs > 1200 ? "시작·복귀 사이에 좌표 누락이 있어 작업 순서 전체를 확인하지 못한 구간입니다."
          : "팔의 작업 진입·유지·복귀를 관측해 후보 구간을 구성했어요. 업무 종류의 정답은 별도 확인이 필요합니다.",
    });
  };
  for (const raw of rawFrames) {
    const observation = armObservation(raw);
    if (!observation) {
      neutralSince = returnSince = onset = null;
      if (start === null) ready = false;
      continue;
    }
    const time = observation.time;
    if (previousTime !== null && time <= previousTime) continue;
    if (previousTime !== null && time - previousTime > 1200) {
      if (start !== null) trackingGapMs += time - previousTime;
      neutralSince = returnSince = onset = null;
      if (time - previousTime > 3500) {
        emit(previousTime, false); start = null; ready = false;
      }
    }
    previousTime = time;
    if (start === null) {
      if (observation.returned) {
        neutralSince ??= time;
        if (time - neutralSince >= 600) ready = true;
        onset = null;
      } else if (observation.working) {
        onset ??= time;
        if (time - onset >= 400) {
          hasStart = ready;
          trackingGapMs = 0;
          start = Math.max(rawFrames[0][0], onset - (ready ? 500 : 0));
          neutralSince = returnSince = null;
          ready = false;
        }
      } else onset = null;
    } else if (observation.returned) {
      returnSince ??= time;
      if (time - returnSince >= 600) {
        emit(returnSince, true);
        start = onset = returnSince = null;
        neutralSince = time; ready = true;
      }
    } else returnSince = null;
  }
  if (start !== null && previousTime !== null) emit(previousTime, false);
  return result;
}

export function hasCompleteWorkBoundaries(slice: Pick<TargetMotionSlice, "boundaryEvidence">) {
  const evidence = slice.boundaryEvidence;
  return Boolean(evidence?.startObserved && evidence.endObserved && !(evidence.trackingGapMs && evidence.trackingGapMs > 1200));
}
