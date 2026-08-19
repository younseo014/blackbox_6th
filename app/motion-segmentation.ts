import { parseSessionFrame } from "./pose-store";
import type { PrimitiveMotionLabel } from "./occupation-templates";

export type TargetMotionSlice = {
  frames: number[][];
  startMs: number;
  endMs: number;
  durationSeconds: number;
  originalDurationSeconds: number;
  excludedFrameCount: number;
  reason: string;
};

const MIN_USABLE_FRAMES = 6;
const MAX_ACTIVE_GAP_MS = 1100;
const SLICE_PADDING_MS = 450;
const ASPECT_RATIO = 16 / 9;

type MotionFrame = {
  raw: number[];
  originalIndex: number;
  timeMs: number;
  centerX: number;
  centerY: number;
  torsoScale: number;
  pose: number[];
};

function bodyPoint(body: number[], index: number) {
  return {
    x: body[index * 4] * ASPECT_RATIO,
    y: body[index * 4 + 1],
    visibility: body[index * 4 + 3] ?? 0,
  };
}

function toMotionFrame(raw: number[], originalIndex: number): MotionFrame | null {
  if ((raw[2] ?? 0) < 0.5) return null;
  const parsed = parseSessionFrame(raw);
  const leftShoulder = bodyPoint(parsed.body, 0);
  const rightShoulder = bodyPoint(parsed.body, 1);
  const leftHip = bodyPoint(parsed.body, 12);
  const rightHip = bodyPoint(parsed.body, 13);
  const required = [leftShoulder, rightShoulder, leftHip, rightHip];
  if (required.some((point) => point.visibility < 0.25)) return null;
  const shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
  const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
  const centerX = (leftHip.x + rightHip.x) / 2;
  const centerY = (leftHip.y + rightHip.y) / 2;
  const torsoScale = Math.max(0.08, Math.hypot(shoulderX - centerX, shoulderY - centerY));
  const pose = [2, 3, 4, 5].flatMap((index) => {
    const point = bodyPoint(parsed.body, index);
    return [(point.x - centerX) / torsoScale, (point.y - centerY) / torsoScale];
  });
  return {
    raw,
    originalIndex,
    timeMs: parsed.relativeTimeMs,
    centerX,
    centerY,
    torsoScale,
    pose,
  };
}

function percentile(values: number[], ratio: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * ratio))];
}

function splitFullBodyRuns(rawFrames: number[][]) {
  const runs: MotionFrame[][] = [];
  let current: MotionFrame[] = [];
  rawFrames.forEach((raw, originalIndex) => {
    const frame = toMotionFrame(raw, originalIndex);
    if (frame && (!current.length || frame.originalIndex === current[current.length - 1].originalIndex + 1)) {
      current.push(frame);
      return;
    }
    if (current.length >= MIN_USABLE_FRAMES) runs.push(current);
    current = frame ? [frame] : [];
  });
  if (current.length >= MIN_USABLE_FRAMES) runs.push(current);
  return runs;
}

function frameMotionSpeed(previous: MotionFrame, current: MotionFrame) {
  const dtSeconds = Math.max(0.04, (current.timeMs - previous.timeMs) / 1000);
  const squared = current.pose.reduce((sum, value, index) => sum + (value - previous.pose[index]) ** 2, 0);
  return Math.sqrt(squared / current.pose.length) / dtSeconds;
}

function boundaryHandlingIndex(
  run: MotionFrame[],
  side: "start" | "end",
  allowHorizontalTravel: boolean,
) {
  if (run.length < 10) return null;
  const start = side === "start" ? 1 : Math.max(1, Math.floor(run.length * 0.72));
  const end = side === "start" ? Math.ceil(run.length * 0.28) : run.length;
  const flagged: number[] = [];
  for (let index = start; index < end; index += 1) {
    const previous = run[index - 1];
    const current = run[index];
    const dtSeconds = Math.max(0.04, (current.timeMs - previous.timeMs) / 1000);
    const scaleChange = Math.abs(Math.log(current.torsoScale / previous.torsoScale)) / dtSeconds;
    const centerSpeed = Math.hypot(
      current.centerX - previous.centerX,
      current.centerY - previous.centerY,
    ) / previous.torsoScale / dtSeconds;
    if (scaleChange >= 0.32 || (!allowHorizontalTravel && centerSpeed >= 0.62)) {
      flagged.push(index);
    }
  }
  if (!flagged.length) return null;
  return side === "start" ? flagged[flagged.length - 1] : flagged[0];
}

function activeBout(run: MotionFrame[]) {
  const speeds = run.slice(1).map((frame, index) => frameMotionSpeed(run[index], frame));
  if (!speeds.length) return null;
  const threshold = Math.max(0.11, percentile(speeds, 0.75) * 0.28);
  const activeIndexes = speeds
    .map((speed, index) => ({ speed, index: index + 1 }))
    .filter((item) => item.speed >= threshold);
  if (!activeIndexes.length) return null;

  const bouts: Array<{ start: number; end: number; energy: number }> = [];
  let current = {
    start: Math.max(0, activeIndexes[0].index - 1),
    end: activeIndexes[0].index,
    energy: activeIndexes[0].speed,
  };
  for (const item of activeIndexes.slice(1)) {
    if (run[item.index].timeMs - run[current.end].timeMs <= MAX_ACTIVE_GAP_MS) {
      current.end = item.index;
      current.energy += item.speed;
    } else {
      bouts.push(current);
      current = { start: Math.max(0, item.index - 1), end: item.index, energy: item.speed };
    }
  }
  bouts.push(current);
  return bouts.sort((a, b) => b.energy - a.energy)[0];
}

/**
 * Removes camera setup/stop handling and keeps the strongest full-body work
 * bout. This intentionally uses the full-body flag first: a person walking
 * close to a laptop to press the camera button is usually only partially in
 * frame, while the target test is performed after their whole body is ready.
 */
export function sliceTargetMotion(
  rawFrames: number[][],
  targetPrimitiveLabels: PrimitiveMotionLabel[] = [],
): TargetMotionSlice {
  const originalDurationSeconds = rawFrames.length > 1
    ? Math.max(0, (rawFrames[rawFrames.length - 1][0] - rawFrames[0][0]) / 1000)
    : 0;
  const runs = splitFullBodyRuns(rawFrames);
  if (!runs.length) {
    return {
      frames: rawFrames,
      startMs: rawFrames[0]?.[0] ?? 0,
      endMs: rawFrames[rawFrames.length - 1]?.[0] ?? 0,
      durationSeconds: originalDurationSeconds,
      originalDurationSeconds,
      excludedFrameCount: 0,
      reason: "전신이 연속으로 인식된 구간이 부족해 전체 기록을 사용했어요.",
    };
  }

  const allowsTravel = targetPrimitiveLabels.some((label) =>
    label === "WALK" || label === "ZONE_TRANSITION" || label === "CARRY",
  );
  const candidates = runs.map((sourceRun) => {
    let run = sourceRun;
    const startHandling = boundaryHandlingIndex(run, "start", allowsTravel);
    if (startHandling !== null && startHandling < run.length - MIN_USABLE_FRAMES) {
      run = run.slice(startHandling);
    }
    const endHandling = boundaryHandlingIndex(run, "end", allowsTravel);
    if (endHandling !== null && endHandling >= MIN_USABLE_FRAMES) {
      run = run.slice(0, endHandling);
    }
    const bout = activeBout(run);
    if (!bout) return { run, energy: 0 };
    const startTime = run[bout.start].timeMs - SLICE_PADDING_MS;
    const endTime = run[bout.end].timeMs + SLICE_PADDING_MS;
    const sliced = run.filter((frame) => frame.timeMs >= startTime && frame.timeMs <= endTime);
    return { run: sliced.length >= MIN_USABLE_FRAMES ? sliced : run, energy: bout.energy };
  });
  const selected = candidates.sort((a, b) => b.energy - a.energy || b.run.length - a.run.length)[0].run;
  const frames = selected.map((frame) => frame.raw);
  const startMs = frames[0]?.[0] ?? 0;
  const endMs = frames[frames.length - 1]?.[0] ?? startMs;
  const durationSeconds = Math.max(0, (endMs - startMs) / 1000);
  return {
    frames,
    startMs,
    endMs,
    durationSeconds,
    originalDurationSeconds,
    excludedFrameCount: Math.max(0, rawFrames.length - frames.length),
    reason: `전체 ${originalDurationSeconds.toFixed(1)}초 중 전신이 안정된 목표 동작 ${durationSeconds.toFixed(1)}초만 분석했어요.`,
  };
}
