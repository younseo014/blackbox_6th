// The actual motion-signal-based detector: given a hand trajectory (a
// sequence of positions over time, from either a real recorded camera
// session or a procedurally generated demo clip), decides whether it
// contains a "마감 반복 확인" (went somewhere and came back, twice), a
// "미세 지연" (a stall in the middle of an otherwise-moving task), and/or a
// "안전 알림" (a sharp, reactive burst) - purely from the coordinates
// themselves, no label or intended category given as input.
//
// This is the piece that was missing before: safetyAlerts/doubleChecks/
// microDelaySeconds used to only come from UI button clicks and timers
// (recordDoubleCheck/recordSafetyAlert/task timers in page.tsx), with the
// camera's variability/smoothness numbers shown as a disconnected FYI stat.
// This module makes the skeleton motion itself a real input to those counts
// - see recordMotionDetections() in page.tsx, which runs this on a finished
// session's trajectory and adds ANY detected events to the day's counts
// alongside (not instead of) the existing button/timer signals.
//
// Pure and DOM-free so it's directly unit-testable (tests/motion-detection.test.ts).

import { parseSessionFrame } from "./pose-store";

export type Point3D = { x: number; y: number; z: number };

export type MotionSample = {
  point: Point3D | null;
  relativeTimeMs: number;
};

export type DetectedMotionEvent = {
  type: "double_check" | "micro_delay" | "safety_alert";
  startMs: number;
  endMs: number;
  /** Plain-language, numbers-included explanation of what was actually measured. */
  evidence: string;
};

// -- Tunable thresholds -----------------------------------------------------
// Distances are in normalized image-coordinate units (same space pose-store.ts
// stores), speeds are that distance per second. These are deliberately
// documented and exported so tests (and the "감지 근거" UI) can cite the
// exact same numbers instead of a hardcoded copy that could drift.

/** Below this speed, a frame counts as "still". */
export const FREEZE_SPEED_THRESHOLD = 0.01; // units/sec
/**
 * The bar for "there was SOME real motion here" - used both to confirm a
 * freeze was bracketed by actual activity (not just standing still) and as
 * the low end of the double_check "went away from start" band. Kept low so
 * small motions (a repeated small tap, a modest reach) still count.
 */
export const MOTION_PRESENCE_THRESHOLD = 0.02; // units/sec (speed) / units (distance)
/** A still run shorter than this isn't a meaningful pause - could just be noise. */
export const FREEZE_MIN_DURATION_MS = 1000;
/**
 * Peak jerk (rate of change of speed, i.e. acceleration - units/sec^2) at/
 * above this = a sharp reactive motion. Deliberately normalized by time
 * (not just "speed delta between two consecutive samples") so this stays
 * correct no matter how finely spaced the samples are - a real camera
 * session and a synthetic demo clip can be sampled at different rates
 * without one becoming artificially harder or easier to flag.
 */
export const SAFETY_JERK_THRESHOLD = 18;

/**
 * How far (in the same normalized-coordinate units) the hand has to get
 * from where the clip started to count as "went to do something", for
 * double_check detection. Distance-based rather than instantaneous-speed
 * based, because a smooth (raised-cosine) approach naturally has near-zero
 * *speed* right at its own peak (a momentary "hold"), which would otherwise
 * look like two separate bursts inside a single approach.
 */
export const AWAY_DISTANCE_THRESHOLD = 0.05;
/** Below this distance from the start, the hand counts as "back near rest". */
export const NEAR_DISTANCE_THRESHOLD = 0.02;
/** An away/near episode shorter than this isn't a deliberate episode - could be noise. */
export const EPISODE_MIN_DURATION_MS = 500;

function distance(a: Point3D, b: Point3D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

type TimedValue = { value: number; startMs: number; endMs: number };

function runDurationMs(run: TimedValue[]): number {
  if (run.length === 0) return 0;
  return run[run.length - 1].endMs - run[0].startMs;
}

/** Maximal contiguous runs of samples whose value all satisfies `predicate`. */
function findRuns(series: TimedValue[], predicate: (value: number) => boolean): TimedValue[][] {
  const runs: TimedValue[][] = [];
  let current: TimedValue[] = [];
  for (const item of series) {
    if (predicate(item.value)) {
      current.push(item);
    } else if (current.length > 0) {
      runs.push(current);
      current = [];
    }
  }
  if (current.length > 0) runs.push(current);
  return runs;
}

type Step = { speed: number; startMs: number; endMs: number };

/** Per-step speed (distance/time), skipping steps where either side is undetected. */
function computeSteps(samples: MotionSample[]): Step[] {
  const steps: Step[] = [];
  for (let i = 1; i < samples.length; i += 1) {
    const previous = samples[i - 1];
    const current = samples[i];
    if (!previous.point || !current.point) continue;
    const dtMs = current.relativeTimeMs - previous.relativeTimeMs;
    if (dtMs <= 0) continue;
    const speed = distance(previous.point, current.point) / (dtMs / 1000);
    steps.push({ speed, startMs: previous.relativeTimeMs, endMs: current.relativeTimeMs });
  }
  return steps;
}

function detectMicroDelay(steps: Step[]): DetectedMotionEvent[] {
  const series: TimedValue[] = steps.map((s) => ({ value: s.speed, startMs: s.startMs, endMs: s.endMs }));
  const stillRuns = findRuns(series, (speed) => speed <= FREEZE_SPEED_THRESHOLD);
  const events: DetectedMotionEvent[] = [];
  for (const run of stillRuns) {
    const durationMs = runDurationMs(run);
    if (durationMs < FREEZE_MIN_DURATION_MS) continue;
    const runStart = series.indexOf(run[0]);
    const runEnd = series.indexOf(run[run.length - 1]);
    // Only counts as a "hesitation" if there was real motion right before
    // and right after - otherwise this is just someone standing still, not
    // a pause mid-task.
    const movedBefore = series
      .slice(Math.max(0, runStart - 3), runStart)
      .some((s) => s.value >= MOTION_PRESENCE_THRESHOLD);
    const movedAfter = series
      .slice(runEnd + 1, runEnd + 4)
      .some((s) => s.value >= MOTION_PRESENCE_THRESHOLD);
    if (!movedBefore || !movedAfter) continue;
    events.push({
      type: "micro_delay",
      startMs: run[0].startMs,
      endMs: run[run.length - 1].endMs,
      evidence: `손 움직임이 거의 멈춘(초당 이동량 ${FREEZE_SPEED_THRESHOLD} 미만) 구간이 ${(durationMs / 1000).toFixed(1)}초 지속됐어요 (기준: ${(FREEZE_MIN_DURATION_MS / 1000).toFixed(1)}초 이상, 앞뒤로는 실제 움직임이 있었어요).`,
    });
  }
  return events;
}

/**
 * Went away from where the clip started, came back near it, went away
 * again - twice or more. Uses sustained DISTANCE from the start position
 * rather than instantaneous speed (see AWAY_DISTANCE_THRESHOLD doc), so a
 * momentary near-zero-speed "hold" at the peak of one approach doesn't get
 * mistaken for a return to rest.
 */
function detectDoubleCheck(samples: MotionSample[]): DetectedMotionEvent[] {
  const firstDetected = samples.find((s) => s.point);
  if (!firstDetected?.point) return [];
  const reference = firstDetected.point;

  const series: TimedValue[] = [];
  for (const sample of samples) {
    if (!sample.point) continue;
    series.push({ value: distance(sample.point, reference), startMs: sample.relativeTimeMs, endMs: sample.relativeTimeMs });
  }
  if (series.length === 0) return [];

  const awayEpisodes = findRuns(series, (value) => value >= AWAY_DISTANCE_THRESHOLD).filter(
    (run) => runDurationMs(run) >= EPISODE_MIN_DURATION_MS,
  );
  if (awayEpisodes.length < 2) return [];

  const nearEpisodes = findRuns(series, (value) => value <= NEAR_DISTANCE_THRESHOLD).filter(
    (run) => runDurationMs(run) >= EPISODE_MIN_DURATION_MS,
  );

  // Count away-episodes that are genuinely separate approaches: only when a
  // "back near start" episode of its own falls between one approach ending
  // and the next one starting.
  let separatedApproaches = 1;
  for (let i = 1; i < awayEpisodes.length; i += 1) {
    const previousEnd = awayEpisodes[i - 1][awayEpisodes[i - 1].length - 1].endMs;
    const nextStart = awayEpisodes[i][0].startMs;
    const hasReturnBetween = nearEpisodes.some(
      (near) => near[0].startMs >= previousEnd && near[near.length - 1].endMs <= nextStart,
    );
    if (hasReturnBetween) separatedApproaches += 1;
  }
  if (separatedApproaches < 2) return [];

  return [
    {
      type: "double_check",
      startMs: awayEpisodes[0][0].startMs,
      endMs: awayEpisodes[awayEpisodes.length - 1][awayEpisodes[awayEpisodes.length - 1].length - 1].endMs,
      evidence: `시작 위치로 완전히 돌아왔다가 다시 멀어지는 접근 동작이 ${separatedApproaches}번 있었어요 (기준: ${AWAY_DISTANCE_THRESHOLD} 이상 멀어졌다가 ${NEAR_DISTANCE_THRESHOLD} 이하로 돌아오는 것을 한 번의 접근으로 계산).`,
    },
  ];
}

function detectSafetyAlert(steps: Step[]): DetectedMotionEvent[] {
  let peakJerk = 0;
  let peakIndex = -1;
  for (let i = 1; i < steps.length; i += 1) {
    // dt between the MIDPOINTS of two consecutive steps - true acceleration
    // (speed change per second), not a raw per-sample delta, so this reads
    // the same whether samples are 200ms apart (real camera) or much finer
    // (a smooth-playback synthetic clip).
    const dtSec = ((steps[i].startMs + steps[i].endMs) / 2 - (steps[i - 1].startMs + steps[i - 1].endMs) / 2) / 1000;
    if (dtSec <= 0) continue;
    const jerk = Math.abs(steps[i].speed - steps[i - 1].speed) / dtSec;
    if (jerk > peakJerk) {
      peakJerk = jerk;
      peakIndex = i;
    }
  }
  if (peakIndex < 0 || peakJerk < SAFETY_JERK_THRESHOLD) return [];
  return [
    {
      type: "safety_alert",
      startMs: steps[Math.max(0, peakIndex - 1)].startMs,
      endMs: steps[peakIndex].endMs,
      evidence: `속도가 급격히 변하는 반응성 동작(저크 ${peakJerk.toFixed(2)})이 감지됐어요 (기준: ${SAFETY_JERK_THRESHOLD.toFixed(2)} 이상).`,
    },
  ];
}

/**
 * Analyzes a hand trajectory and returns every motion pattern it actually
 * contains - a clip can contain more than one (or none). Detection is
 * purely from the coordinates: no label, seed, or intended-type is used as
 * an input here, so this is a genuine measurement, not a lookup.
 */
export function detectMotionEvents(samples: MotionSample[]): DetectedMotionEvent[] {
  const steps = computeSteps(samples);
  if (steps.length === 0) return [];

  const doubleChecks = detectDoubleCheck(samples);
  const microDelays = detectMicroDelay(steps);

  // A double_check's "return to rest between approaches" is, structurally,
  // also a still period - don't ALSO report it as an independent hesitation
  // when it's fully explained by the repeated-approach pattern.
  const microDelaysNotExplainedByDoubleCheck = microDelays.filter(
    (delay) =>
      !doubleChecks.some((check) => delay.startMs >= check.startMs && delay.endMs <= check.endMs),
  );

  return [...detectSafetyAlert(steps), ...doubleChecks, ...microDelaysNotExplainedByDoubleCheck];
}

/**
 * Converts the raw stored-frame layout (see MOTION_FRAME_STRIDE in
 * pose-store.ts) into the right-hand trajectory `detectMotionEvents` needs.
 * Shared by real recorded sessions (page.tsx, after a camera session ends)
 * and synthetic demo clips (demo-personas.ts) - same parser, same detector,
 * so a demo clip's "감지 근거" is a genuine measurement of that clip, not a
 * hand-typed label, and a real session is analyzed the exact same way.
 */
export function motionSamplesFromRawFrames(frames: number[][]): MotionSample[] {
  return frames.map(parseSessionFrame).map((frame) => ({
    point: frame.rightHand ? { x: frame.rightHand[0], y: frame.rightHand[1], z: frame.rightHand[2] } : null,
    relativeTimeMs: frame.relativeTimeMs,
  }));
}
