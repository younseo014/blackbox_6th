import type { CameraSlot } from "./multi-camera";

/**
 * Inference budget across three cameras.
 *
 * Running Pose Heavy on three webcams every animation frame is what makes the
 * UI stutter and what makes a 12-hour session drift ever further behind real
 * time. Two rules fix both: only one camera may run a body inference per tick
 * (round-robin, so no frame ever pays for three heavy graphs), and each camera
 * earns its own cadence from what it is currently looking at.
 *
 * Every number here is a target interval in milliseconds. They are starting
 * points, tuned against the evaluation harness rather than guessed once and
 * frozen - `scripts/run-tracking-eval.ts` reports the resulting FPS and the
 * recognition latency they produce.
 */
export const POSE_INTERVAL_MS = {
  /** The camera currently giving the global track its coordinate frame. */
  primary: 100,
  /** Saw the person recently but is not the clearest view. */
  secondary: 150,
  /** No person, but the background gate reports something moved. */
  alerted: 200,
  /**
   * Quiet camera watching an empty room.
   *
   * This is bounded by the 1s recognition budget, not by taste. Worst case a
   * person enters just after an idle poll: 300ms to the next poll, which
   * promotes the camera to `alerted`, then two more frames at 200ms to satisfy
   * GLOBAL_CONFIRMATION_FRAMES = 700ms total. At 500ms idle the same path
   * costs 900ms and leaves no margin for a pose that does not lock on the
   * first frame.
   */
  idle: 300,
} as const;

/** Person detector cadence. It only has to gate acquisition, not track. */
export const DETECTOR_INTERVAL_MS = {
  /** Nobody is being tracked - the detector is the thing that finds them. */
  searching: 400,
  /** Somebody is tracked, so the detector is only a sanity check. */
  tracking: 1_200,
} as const;

/** A camera is considered to have "recently" seen the person within this window. */
const RECENT_PERSON_MS = 1_500;
/** Foreground motion counts as a wake-up signal for this long. */
const RECENT_FOREGROUND_MS = 1_200;

export type CameraActivity = {
  cameraSlot: CameraSlot;
  /** Last time this camera's own tracker produced a confirmed person. */
  lastPersonSeenAt: number;
  /** Last time the background gate saw foreground change on this camera. */
  lastForegroundAt: number;
  /** Last time a body inference actually ran for this camera. */
  lastInferenceAt: number;
  /** True for the camera supplying the global track's coordinate frame. */
  isPrimary: boolean;
  /** False when the stream is missing or has no fresh frame to read. */
  ready: boolean;
};

export type SchedulerState = {
  /** Slot that ran last, so the next tick starts after it. */
  lastSlot: CameraSlot;
};

export const EMPTY_SCHEDULER_STATE: SchedulerState = { lastSlot: 3 };

export type CameraTier = keyof typeof POSE_INTERVAL_MS;

export function cameraTier(activity: CameraActivity, timestamp: number): CameraTier {
  if (activity.isPrimary) return "primary";
  if (timestamp - activity.lastPersonSeenAt <= RECENT_PERSON_MS) return "secondary";
  if (timestamp - activity.lastForegroundAt <= RECENT_FOREGROUND_MS) return "alerted";
  return "idle";
}

export function poseIntervalFor(activity: CameraActivity, timestamp: number): number {
  return POSE_INTERVAL_MS[cameraTier(activity, timestamp)];
}

export function detectorIntervalFor(trackIsPresent: boolean): number {
  return trackIsPresent ? DETECTOR_INTERVAL_MS.tracking : DETECTOR_INTERVAL_MS.searching;
}

/**
 * Picks at most one camera to infer on this tick.
 *
 * Cameras that are overdue by the widest margin relative to their own target
 * interval win, so a quiet camera cannot be starved forever by a busy one, but
 * the tracked person's clearest view still gets the most frames.
 */
export function selectCameraForInference(
  state: SchedulerState,
  activities: CameraActivity[],
  timestamp: number,
): { slot: CameraSlot | null; state: SchedulerState } {
  const due = activities
    .filter((activity) => activity.ready)
    .map((activity) => ({
      activity,
      overdue: (timestamp - activity.lastInferenceAt) / poseIntervalFor(activity, timestamp),
    }))
    .filter((candidate) => candidate.overdue >= 1);

  if (!due.length) return { slot: null, state };

  // Round-robin tiebreak: start scanning just past whoever ran last, so equal
  // claims rotate instead of always resolving to the lowest slot number.
  const rotation = (slot: CameraSlot) => (slot - state.lastSlot + 3) % 3;
  due.sort((a, b) =>
    b.overdue - a.overdue ||
    rotation(a.activity.cameraSlot) - rotation(b.activity.cameraSlot),
  );
  const slot = due[0].activity.cameraSlot;
  return { slot, state: { lastSlot: slot } };
}
