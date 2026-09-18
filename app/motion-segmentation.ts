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
const MAX_TRACKING_GAP_MS = 1200;
const MAX_ACTIVE_GAP_MS = 1400;
const MIN_ACTIVE_BOUT_MS = 300;
const MIN_SEMANTIC_SEGMENT_MS = 700;
const MAX_SEMANTIC_SEGMENT_MS = 20_000;
const MOTION_MODE_WINDOW_FRAMES = 4;
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
  const parsed = parseSessionFrame(raw);
  if (!parsed.bodyDetected) return null;
  const leftShoulder = bodyPoint(parsed.body, 0);
  const rightShoulder = bodyPoint(parsed.body, 1);
  const leftHip = bodyPoint(parsed.body, 12);
  const rightHip = bodyPoint(parsed.body, 13);
  if (leftShoulder.visibility < 0.25 || rightShoulder.visibility < 0.25) return null;
  const armPoints = [2, 3, 4, 5].map((index) => bodyPoint(parsed.body, index));
  if (armPoints.filter((point) => point.visibility >= 0.2).length < 2) return null;
  const shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
  const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
  const hipsVisible = leftHip.visibility >= 0.25 && rightHip.visibility >= 0.25;
  const centerX = hipsVisible ? (leftHip.x + rightHip.x) / 2 : shoulderX;
  const centerY = hipsVisible ? (leftHip.y + rightHip.y) / 2 : shoulderY;
  const shoulderWidth = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
  const torsoScale = Math.max(
    0.08,
    hipsVisible ? Math.hypot(shoulderX - centerX, shoulderY - centerY) : shoulderWidth,
  );
  const pose = armPoints.flatMap((point, index) => {
    const fallback = index % 2 === 0 ? leftShoulder : rightShoulder;
    const trackedPoint = point.visibility >= 0.2 ? point : fallback;
    return [(trackedPoint.x - centerX) / torsoScale, (trackedPoint.y - centerY) / torsoScale];
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

function splitTrackedRuns(rawFrames: number[][]) {
  const runs: MotionFrame[][] = [];
  let current: MotionFrame[] = [];
  rawFrames.forEach((raw, originalIndex) => {
    const frame = toMotionFrame(raw, originalIndex);
    if (frame && (
      !current.length || (
        frame.originalIndex === current[current.length - 1].originalIndex + 1 &&
        frame.timeMs - current[current.length - 1].timeMs <= MAX_TRACKING_GAP_MS
      )
    )) {
      current.push(frame);
      return;
    }
    if (current.length >= MIN_USABLE_FRAMES) runs.push(current);
    current = frame ? [frame] : [];
  });
  if (current.length >= MIN_USABLE_FRAMES) runs.push(current);
  return runs;
}

function frameMotionComponents(previous: MotionFrame, current: MotionFrame) {
  const dtSeconds = Math.max(0.04, (current.timeMs - previous.timeMs) / 1000);
  const squared = current.pose.reduce((sum, value, index) => sum + (value - previous.pose[index]) ** 2, 0);
  const poseSpeed = Math.sqrt(squared / current.pose.length) / dtSeconds;
  const travelSpeed = Math.hypot(
    current.centerX - previous.centerX,
    current.centerY - previous.centerY,
  ) / previous.torsoScale / dtSeconds;
  return {
    poseSpeed,
    travelSpeed,
    totalSpeed: Math.max(poseSpeed, Math.min(3, travelSpeed) * 0.35),
  };
}

function frameMotionSpeed(previous: MotionFrame, current: MotionFrame) {
  return frameMotionComponents(previous, current).totalSpeed;
}

function activeBouts(run: MotionFrame[]) {
  const speeds = run.slice(1).map((frame, index) => frameMotionSpeed(run[index], frame));
  if (!speeds.length) return [];
  const threshold = Math.max(0.11, percentile(speeds, 0.75) * 0.28);
  const activeIndexes = speeds
    .map((speed, index) => ({ speed, index: index + 1 }))
    .filter((item) => item.speed >= threshold);
  if (!activeIndexes.length) return [];

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
  return bouts.filter((bout) => run[bout.end].timeMs - run[bout.start].timeMs >= MIN_ACTIVE_BOUT_MS);
}

type MotionBout = { start: number; end: number; energy: number };

function isTravelDominant(run: MotionFrame[], index: number) {
  if (index <= 0) return false;
  const motion = frameMotionComponents(run[index - 1], run[index]);
  return motion.travelSpeed >= 0.45 && motion.travelSpeed >= motion.poseSpeed * 0.75;
}

function travelRatio(run: MotionFrame[], start: number, end: number) {
  let travelFrames = 0;
  let frameCount = 0;
  for (let index = Math.max(1, start); index < Math.min(run.length, end); index += 1) {
    frameCount += 1;
    if (isTravelDominant(run, index)) travelFrames += 1;
  }
  return frameCount > 0 ? travelFrames / frameCount : 0;
}

function weakestMotionIndex(run: MotionFrame[], start: number, end: number) {
  let weakestIndex = start;
  let weakestSpeed = Number.POSITIVE_INFINITY;
  for (let index = Math.max(1, start); index <= Math.min(end, run.length - 1); index += 1) {
    const speed = frameMotionSpeed(run[index - 1], run[index]);
    if (speed < weakestSpeed) {
      weakestSpeed = speed;
      weakestIndex = index;
    }
  }
  return weakestIndex;
}

function frameIndexAtOrAfter(run: MotionFrame[], timeMs: number, start: number, end: number) {
  let low = Math.max(0, start);
  let high = Math.min(run.length, end);
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (run[middle].timeMs < timeMs) low = middle + 1;
    else high = middle;
  }
  return low < Math.min(run.length, end) ? low : -1;
}

function splitContinuousBout(run: MotionFrame[], bout: MotionBout): MotionBout[] {
  const boundaries: number[] = [];
  let lastBoundaryTime = run[bout.start].timeMs;
  for (
    let index = bout.start + MOTION_MODE_WINDOW_FRAMES;
    index <= bout.end - MOTION_MODE_WINDOW_FRAMES;
    index += 1
  ) {
    const beforeTravel = travelRatio(run, index - MOTION_MODE_WINDOW_FRAMES, index);
    const afterTravel = travelRatio(run, index, index + MOTION_MODE_WINDOW_FRAMES);
    const changedMode = (beforeTravel <= 0.25 && afterTravel >= 0.75) ||
      (beforeTravel >= 0.75 && afterTravel <= 0.25);
    const candidateTime = run[index].timeMs;
    if (
      changedMode &&
      candidateTime - lastBoundaryTime >= MIN_SEMANTIC_SEGMENT_MS &&
      run[bout.end].timeMs - candidateTime >= MIN_SEMANTIC_SEGMENT_MS
    ) {
      boundaries.push(index);
      lastBoundaryTime = candidateTime;
      index += MOTION_MODE_WINDOW_FRAMES - 1;
    }
  }

  const withMaximumLength: number[] = [];
  let segmentStart = bout.start;
  for (const nextBoundary of [...boundaries, bout.end + 1]) {
    while (run[Math.min(nextBoundary - 1, bout.end)].timeMs - run[segmentStart].timeMs > MAX_SEMANTIC_SEGMENT_MS) {
      const earliestCutTime = run[segmentStart].timeMs + MAX_SEMANTIC_SEGMENT_MS * 0.55;
      const latestCutTime = run[segmentStart].timeMs + MAX_SEMANTIC_SEGMENT_MS;
      const earliestCut = frameIndexAtOrAfter(run, earliestCutTime, segmentStart + 1, nextBoundary);
      const latestCut = frameIndexAtOrAfter(run, latestCutTime, segmentStart + 1, nextBoundary);
      const searchStart = earliestCut >= 0 ? earliestCut : segmentStart + 1;
      const searchEnd = latestCut >= 0 ? Math.min(latestCut, nextBoundary - 1) : nextBoundary - 1;
      const cut = weakestMotionIndex(run, searchStart, searchEnd);
      if (cut <= segmentStart || cut >= nextBoundary) break;
      withMaximumLength.push(cut);
      segmentStart = cut;
    }
    if (nextBoundary <= bout.end) {
      withMaximumLength.push(nextBoundary);
      segmentStart = nextBoundary;
    }
  }

  const cuts = [...new Set(withMaximumLength)].sort((a, b) => a - b);
  const segments: MotionBout[] = [];
  let start = bout.start;
  for (const cut of [...cuts, bout.end + 1]) {
    const end = Math.min(bout.end, cut - 1);
    if (end >= start) {
      segments.push({
        start,
        end,
        energy: run.slice(start + 1, end + 1).reduce(
          (sum, frame, offset) => sum + frameMotionSpeed(run[start + offset], frame),
          0,
        ),
      });
    }
    start = cut;
  }
  return segments;
}

function emptyFallback(rawFrames: number[][]): TargetMotionSlice {
  const originalDurationSeconds = rawFrames.length > 1
    ? Math.max(0, (rawFrames[rawFrames.length - 1][0] - rawFrames[0][0]) / 1000)
    : 0;
  return {
    frames: rawFrames,
    startMs: rawFrames[0]?.[0] ?? 0,
    endMs: rawFrames[rawFrames.length - 1]?.[0] ?? 0,
    durationSeconds: originalDurationSeconds,
    originalDurationSeconds,
    excludedFrameCount: 0,
    reason: "상반신 관절이 연속으로 인식된 구간이 부족해 전체 기록을 사용했어요.",
  };
}

/**
 * Divides a long camera session into independent movement episodes. Tracking
 * gaps split runs first, then sustained quiet gaps split separate actions.
 */
export function sliceTargetMotions(
  rawFrames: number[][],
): TargetMotionSlice[] {
  if (rawFrames.length < MIN_USABLE_FRAMES) return [];
  const originalDurationSeconds = rawFrames.length > 1
    ? Math.max(0, (rawFrames[rawFrames.length - 1][0] - rawFrames[0][0]) / 1000)
    : 0;
  const slices: TargetMotionSlice[] = [];

  for (const sourceRun of splitTrackedRuns(rawFrames)) {
    const run = sourceRun;
    const bouts = activeBouts(run).flatMap((bout) => splitContinuousBout(run, bout));

    for (const [boutIndex, bout] of bouts.entries()) {
      const previousBout = bouts[boutIndex - 1];
      const nextBout = bouts[boutIndex + 1];
      const previousBoundary = previousBout
        ? (run[previousBout.end].timeMs + run[bout.start].timeMs) / 2
        : Number.NEGATIVE_INFINITY;
      const nextBoundary = nextBout
        ? (run[bout.end].timeMs + run[nextBout.start].timeMs) / 2
        : Number.POSITIVE_INFINITY;
      const startTime = Math.max(run[bout.start].timeMs - SLICE_PADDING_MS, previousBoundary);
      const endTime = Math.min(run[bout.end].timeMs + SLICE_PADDING_MS, nextBoundary);
      const frames = run
        .filter((frame) => frame.timeMs >= startTime && frame.timeMs <= endTime)
        .map((frame) => frame.raw);
      if (frames.length < MIN_USABLE_FRAMES) continue;
      const startMs = frames[0][0];
      const endMs = frames[frames.length - 1][0];
      const durationSeconds = Math.max(0, (endMs - startMs) / 1000);
      slices.push({
        frames,
        startMs,
        endMs,
        durationSeconds,
        originalDurationSeconds,
        excludedFrameCount: Math.max(0, rawFrames.length - frames.length),
        reason: `전체 ${originalDurationSeconds.toFixed(1)}초 기록에서 ${
          (startMs / 1000).toFixed(1)
        }–${(endMs / 1000).toFixed(1)}초 동작 구간을 분리했어요.`,
      });
    }
  }

  return slices.sort((a, b) => a.startMs - b.startMs);
}

/**
 * Compatibility helper for callers that need the single longest detected
 * movement from a session.
 */
export function sliceTargetMotion(
  rawFrames: number[][],
  targetPrimitiveLabels: PrimitiveMotionLabel[] = [],
): TargetMotionSlice {
  void targetPrimitiveLabels;
  const slices = sliceTargetMotions(rawFrames);
  return slices.sort((a, b) => b.frames.length - a.frames.length)[0] ?? emptyFallback(rawFrames);
}
