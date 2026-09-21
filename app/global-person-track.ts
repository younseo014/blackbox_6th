import type { NormalizedLandmarkLike } from "./motion-analysis";
import type { CameraSlot } from "./multi-camera";

/**
 * Single global person layer.
 *
 * Each camera runs its own locked pose tracker (see `pose-target-lock.ts`).
 * This module sits on top and collapses those independent per-camera
 * observations into exactly one owner track, so three cameras watching the
 * same participant never become three people and one action is never counted
 * three times.
 *
 * Without camera calibration there is no metric space to fuse into, so the
 * fused skeleton lives in the *primary* (currently clearest) camera's
 * normalized image frame. Joints borrowed from another camera are re-projected
 * with a torso-anchored similarity transform and are always marked `inferred`,
 * never `observed`, because that transform is an approximation - not geometry.
 * `camera-calibration.ts` can later replace that step; the joint-state contract
 * does not change.
 */

/** Body landmark indices carried into the stored frame format. */
export const GLOBAL_JOINT_INDICES = Array.from({ length: 22 }, (_, offset) => 11 + offset);

const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;

/**
 * Tunable decision thresholds.
 *
 * These are fitted against real data, not chosen by taste - see
 * `scripts/fit-tracking-thresholds.ts` and the provenance table in
 * `docs/tracking-eval.md`. They are a parameter rather than a constant because
 * different pose backends report visibility on different scales: MediaPipe has
 * a trained visibility head, RTMW has a calibrated SimCC peak, and one floor
 * cannot be right for both.
 */
export type TrackingTuning = {
  /** Below this the landmark is not treated as seen by that camera at all. */
  minJointVisibility: number;
  /** Paired limbs shorter than this ratio of each other are not a human body. */
  limbAsymmetryFloor: number;
  /** Consecutive frames before a provisional track becomes present. */
  confirmationFrames: number;
};

export const DEFAULT_TUNING: TrackingTuning = {
  minJointVisibility: 0.25,
  // Swept on COCO-WholeBody val: 0.25 rejected real foreshortened bodies and
  // cost 4.9 points of retention, while 0.05 let empty-frame false positives
  // through. 0.1 is the knee - see docs/tracking-eval.md.
  limbAsymmetryFloor: 0.1,
  confirmationFrames: 3,
};

/** A camera must be at least this good relative to the primary to donate joints. */
const MIN_DONOR_QUALITY = 0.25;
/** Consecutive confirmed frames before a provisional track becomes present. */
export const GLOBAL_CONFIRMATION_FRAMES = 3;
/** How long the track survives with no observation at all before release. */
export const GLOBAL_RELEASE_AFTER_MS = 2_500;
/** How long a present track is reported as coasting before it drops to absent. */
export const GLOBAL_COAST_MS = 700;

export type JointState = "observed" | "inferred" | "missing";

export type GlobalJoint = {
  index: number;
  x: number;
  y: number;
  z: number;
  confidence: number;
  state: JointState;
  sourceCamera: CameraSlot | null;
};

export type CameraObservation = {
  cameraSlot: CameraSlot;
  timestamp: number;
  /** Locked, confirmation-gated landmarks for this camera, or null. */
  landmarks: NormalizedLandmarkLike[] | null;
  /** 0..1 person-detector confidence for this camera, 0 when it did not run. */
  personConfidence: number;
  /** Background-gate foreground ratio, 0..1. */
  foregroundRatio: number;
  /** True when this camera's own confirmation gate has accepted the target. */
  confirmed: boolean;
};

export type ObservationSummary = {
  cameraSlot: CameraSlot;
  timestamp: number;
  personConfidence: number;
  foregroundRatio: number;
  torsoCenter: { x: number; y: number } | null;
  torsoScale: number;
  visibleJoints: number;
  quality: number;
  anatomyValid: boolean;
  temporallyContinuous: boolean;
  /** False when the camera contributed nothing to the fused skeleton. */
  used: boolean;
};

export type GlobalPresence = "absent" | "provisional" | "present" | "coasting";

export type GlobalPersonTrack = {
  id: string;
  createdAt: number;
  updatedAt: number;
  /** Last time any camera actually observed the person. */
  lastObservedAt: number;
  presence: GlobalPresence;
  consecutiveFrames: number;
  primaryCamera: CameraSlot | null;
  /** Frame the fused joints live in. Always equals `primaryCamera` when set. */
  coordinateFrame: CameraSlot | null;
  /** True only when a validated calibration produced these coordinates. */
  calibrated: false;
  joints: GlobalJoint[];
  observations: ObservationSummary[];
  /** Cameras that saw the person at this instant. Duplicate views, one person. */
  observingCameras: CameraSlot[];
  handoffFromCamera: CameraSlot | null;
  handoffGapMs: number;
};

function visibility(point: NormalizedLandmarkLike | undefined): number {
  if (!point) return 0;
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return 0;
  return Math.max(0, Math.min(1, point.visibility ?? 0));
}

function midpoint(a: NormalizedLandmarkLike | undefined, b: NormalizedLandmarkLike | undefined) {
  if (!a || !b) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

export type TorsoFrame = {
  center: { x: number; y: number };
  scale: number;
  /** null when the view gives no reliable left/right ordering (pure side view). */
  facingFront: boolean | null;
};

/**
 * Torso anchor for an observation. Falls back to the shoulder line alone when
 * the hips are cropped out (CCTV angles and desk-height cameras), to the hips
 * alone when the upper body is occluded, and to a single shoulder-hip pair for
 * a pure side view.
 */
export function torsoFrame(landmarks: NormalizedLandmarkLike[]): TorsoFrame | null {
  const shouldersVisible =
    visibility(landmarks[LEFT_SHOULDER]) >= 0.15 && visibility(landmarks[RIGHT_SHOULDER]) >= 0.15;
  const hipsVisible =
    visibility(landmarks[LEFT_HIP]) >= 0.15 && visibility(landmarks[RIGHT_HIP]) >= 0.15;
  const shoulderCenter = shouldersVisible
    ? midpoint(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    : null;
  const hipCenter = hipsVisible ? midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP]) : null;

  if (shoulderCenter && hipCenter) {
    const scale = Math.hypot(shoulderCenter.x - hipCenter.x, shoulderCenter.y - hipCenter.y);
    if (scale < 0.02) return null;
    return {
      center: { x: (shoulderCenter.x + hipCenter.x) / 2, y: (shoulderCenter.y + hipCenter.y) / 2 },
      scale,
      // Shoulder x ordering is the only cheap facing cue a single 2D view has.
      facingFront: landmarks[LEFT_SHOULDER].x > landmarks[RIGHT_SHOULDER].x,
    };
  }
  if (shoulderCenter) {
    const width = Math.hypot(
      landmarks[LEFT_SHOULDER].x - landmarks[RIGHT_SHOULDER].x,
      landmarks[LEFT_SHOULDER].y - landmarks[RIGHT_SHOULDER].y,
    );
    if (width < 0.02) return null;
    return {
      center: shoulderCenter,
      scale: width / 0.6,
      facingFront: landmarks[LEFT_SHOULDER].x > landmarks[RIGHT_SHOULDER].x,
    };
  }
  if (hipCenter) {
    const width = Math.hypot(
      landmarks[LEFT_HIP].x - landmarks[RIGHT_HIP].x,
      landmarks[LEFT_HIP].y - landmarks[RIGHT_HIP].y,
    );
    if (width < 0.02) return null;
    return { center: hipCenter, scale: width / 0.45, facingFront: landmarks[LEFT_HIP].x > landmarks[RIGHT_HIP].x };
  }

  // Pure side view: the far shoulder and far hip are occluded by the body, so
  // no left/right pair resolves. One shoulder plus the hip below it still gives
  // a real torso axis - and facing is genuinely unknowable from it, so say so
  // rather than guessing a mirror that would flip the skeleton.
  for (const [shoulder, hip] of [[LEFT_SHOULDER, LEFT_HIP], [RIGHT_SHOULDER, RIGHT_HIP]] as const) {
    if (visibility(landmarks[shoulder]) < 0.15 || visibility(landmarks[hip]) < 0.15) continue;
    const scale = Math.hypot(
      landmarks[shoulder].x - landmarks[hip].x,
      landmarks[shoulder].y - landmarks[hip].y,
    );
    if (scale < 0.02) continue;
    return {
      center: midpoint(landmarks[shoulder], landmarks[hip])!,
      scale,
      facingFront: null,
    };
  }
  return null;
}

/**
 * Structural sanity check. A pose model at a low threshold will happily drape a
 * skeleton over a chair or a coat on a hanger; those hallucinations usually
 * fail on limb-length symmetry or collapse to a near-zero torso.
 */
export function anatomyIsValid(
  landmarks: NormalizedLandmarkLike[],
  tuning: TrackingTuning = DEFAULT_TUNING,
): boolean {
  const frame = torsoFrame(landmarks);
  if (!frame) return false;
  const limbs: Array<[number, number, number, number]> = [
    [11, 13, 12, 14],
    [13, 15, 14, 16],
    [23, 25, 24, 26],
    [25, 27, 26, 28],
  ];
  for (const [leftA, leftB, rightA, rightB] of limbs) {
    const leftSeen = visibility(landmarks[leftA]) >= tuning.minJointVisibility &&
      visibility(landmarks[leftB]) >= tuning.minJointVisibility;
    const rightSeen = visibility(landmarks[rightA]) >= tuning.minJointVisibility &&
      visibility(landmarks[rightB]) >= tuning.minJointVisibility;
    if (!leftSeen || !rightSeen) continue;
    const left = Math.hypot(landmarks[leftA].x - landmarks[leftB].x, landmarks[leftA].y - landmarks[leftB].y);
    const right = Math.hypot(landmarks[rightA].x - landmarks[rightB].x, landmarks[rightA].y - landmarks[rightB].y);
    const longer = Math.max(left, right);
    const shorter = Math.min(left, right);
    // Perspective and foreshortening are real and routinely halve a limb, so
    // this only rejects gross asymmetry - a 10x difference between matched
    // limbs is not a body. The floor is fitted, not guessed.
    if (longer > 0.01 && shorter / longer < tuning.limbAsymmetryFloor) return false;
  }
  return true;
}

function observationQuality(
  landmarks: NormalizedLandmarkLike[],
  frame: TorsoFrame,
  observation: CameraObservation,
  tuning: TrackingTuning,
): number {
  const visibilities = GLOBAL_JOINT_INDICES.map((index) => visibility(landmarks[index]));
  const visible = visibilities.filter((value) => value >= tuning.minJointVisibility);
  const meanVisibility = visible.length
    ? visible.reduce((sum, value) => sum + value, 0) / visible.length
    : 0;
  const coverage = visible.length / GLOBAL_JOINT_INDICES.length;
  // A person filling more of the frame is both easier to pose and easier to
  // attribute correctly, but the term saturates so a close-up does not always
  // outrank a clean full-body view.
  const scaleTerm = Math.min(1, frame.scale / 0.25);
  return (
    meanVisibility * 0.35 +
    coverage * 0.3 +
    scaleTerm * 0.15 +
    Math.min(1, observation.personConfidence) * 0.15 +
    (observation.confirmed ? 0.05 : 0)
  );
}

function summarize(observation: CameraObservation, tuning: TrackingTuning): ObservationSummary {
  const base: ObservationSummary = {
    cameraSlot: observation.cameraSlot,
    timestamp: observation.timestamp,
    personConfidence: observation.personConfidence,
    foregroundRatio: observation.foregroundRatio,
    torsoCenter: null,
    torsoScale: 0,
    visibleJoints: 0,
    quality: 0,
    anatomyValid: false,
    temporallyContinuous: false,
    used: false,
  };
  const landmarks = observation.landmarks;
  if (!landmarks) return base;
  const frame = torsoFrame(landmarks);
  if (!frame) return base;
  const anatomyValid = anatomyIsValid(landmarks, tuning);
  return {
    ...base,
    torsoCenter: frame.center,
    torsoScale: frame.scale,
    visibleJoints: GLOBAL_JOINT_INDICES.filter(
      (index) => visibility(landmarks[index]) >= tuning.minJointVisibility,
    ).length,
    quality: anatomyValid ? observationQuality(landmarks, frame, observation, tuning) : 0,
    anatomyValid,
  };
}

/**
 * Re-projects a donor camera's joint into the primary camera's image frame by
 * aligning torso centre, torso scale and facing. Approximate by construction -
 * callers must mark the result `inferred`.
 */
function reproject(
  point: NormalizedLandmarkLike,
  donor: TorsoFrame,
  primary: TorsoFrame,
) {
  const ratio = primary.scale / Math.max(1e-4, donor.scale);
  // Only mirror when both views actually resolved a facing and they disagree.
  // An unknown facing must not flip the skeleton left-for-right.
  const mirror =
    donor.facingFront !== null && primary.facingFront !== null && donor.facingFront !== primary.facingFront
      ? -1
      : 1;
  return {
    x: primary.center.x + (point.x - donor.center.x) * ratio * mirror,
    y: primary.center.y + (point.y - donor.center.y) * ratio,
    z: point.z * ratio,
  };
}

function missingJoint(index: number): GlobalJoint {
  return { index, x: Number.NaN, y: Number.NaN, z: Number.NaN, confidence: 0, state: "missing", sourceCamera: null };
}

export const EMPTY_GLOBAL_TRACK: GlobalPersonTrack = {
  id: "",
  createdAt: 0,
  updatedAt: 0,
  lastObservedAt: 0,
  presence: "absent",
  consecutiveFrames: 0,
  primaryCamera: null,
  coordinateFrame: null,
  calibrated: false,
  joints: [],
  observations: [],
  observingCameras: [],
  handoffFromCamera: null,
  handoffGapMs: 0,
};

/**
 * Advances the single global person track by one fused frame.
 *
 * Order of operations follows the required gate sequence: detector/foreground
 * evidence, anatomical validity, temporal continuity with the existing global
 * person, then per-joint selection of the most confident camera.
 */
export function updateGlobalPersonTrack(
  previous: GlobalPersonTrack,
  observations: CameraObservation[],
  timestamp: number,
  newId: () => string = () => `person-${Math.round(timestamp)}`,
  tuning: TrackingTuning = DEFAULT_TUNING,
): GlobalPersonTrack {
  const hadTrack = previous.presence !== "absent";
  const summaries = observations.map((observation) => summarize(observation, tuning));

  // 1-2. Detector / foreground evidence plus anatomical validity.
  const eligible = summaries
    .map((summary, position) => ({ summary, observation: observations[position] }))
    .filter(({ summary, observation }) =>
      summary.anatomyValid &&
      summary.quality > 0 &&
      observation.landmarks !== null &&
      (observation.personConfidence > 0 || observation.foregroundRatio > 0 || observation.confirmed),
    );

  // 3. Temporal continuity with the existing global person. A camera whose
  //    torso jumped implausibly since the last fused frame may still be a real
  //    person, but it is not *this* person in this camera's coordinate frame.
  for (const candidate of eligible) {
    const wasPrimary = previous.coordinateFrame === candidate.observation.cameraSlot;
    const previousCenter = wasPrimary ? previousTorsoCenter(previous) : null;
    if (!hadTrack || !previousCenter || !candidate.summary.torsoCenter) {
      candidate.summary.temporallyContinuous = true;
      continue;
    }
    const elapsed = Math.max(1, timestamp - previous.updatedAt);
    const step = Math.hypot(
      candidate.summary.torsoCenter.x - previousCenter.x,
      candidate.summary.torsoCenter.y - previousCenter.y,
    ) / Math.max(0.05, candidate.summary.torsoScale);
    // Torso lengths per second. Generous enough for a fast walk close to the
    // lens, tight enough that a jump to a different body is rejected.
    candidate.summary.temporallyContinuous = step / (elapsed / 1000) <= 12;
  }
  const continuous = eligible.filter(({ summary }) => summary.temporallyContinuous);
  const usable = continuous.length ? continuous : hadTrack ? [] : eligible;

  if (!usable.length) {
    const sinceObserved = timestamp - previous.lastObservedAt;
    if (!hadTrack || sinceObserved >= GLOBAL_RELEASE_AFTER_MS) {
      return { ...EMPTY_GLOBAL_TRACK, updatedAt: timestamp, observations: summaries };
    }
    return {
      ...previous,
      updatedAt: timestamp,
      observations: summaries,
      observingCameras: [],
      presence: previous.presence === "present" || previous.presence === "coasting"
        ? sinceObserved <= GLOBAL_COAST_MS ? "coasting" : "present"
        : "provisional",
      handoffFromCamera: null,
      handoffGapMs: 0,
    };
  }

  // 4-5. Pick the clearest view as the coordinate frame, then fill each joint
  //      from whichever camera sees it best.
  usable.sort((a, b) => b.summary.quality - a.summary.quality);
  const primary = usable[0];
  const primaryLandmarks = primary.observation.landmarks!;
  const primaryFrame = torsoFrame(primaryLandmarks)!;
  primary.summary.used = true;

  const donors = usable.slice(1).filter(({ summary }) => summary.quality >= MIN_DONOR_QUALITY);
  const donorFrames = donors.map((donor) => ({
    donor,
    frame: torsoFrame(donor.observation.landmarks!)!,
  }));

  const joints = GLOBAL_JOINT_INDICES.map((index) => {
    const own = primaryLandmarks[index];
    const ownVisibility = visibility(own);
    if (ownVisibility >= tuning.minJointVisibility) {
      return {
        index,
        x: own.x,
        y: own.y,
        z: own.z ?? 0,
        confidence: ownVisibility,
        state: "observed" as JointState,
        sourceCamera: primary.observation.cameraSlot,
      };
    }
    let best: { joint: GlobalJoint; score: number } | null = null;
    for (const { donor, frame } of donorFrames) {
      const point = donor.observation.landmarks![index];
      const donorVisibility = visibility(point);
      if (donorVisibility < tuning.minJointVisibility) continue;
      // Donor confidence is discounted: it is a re-projection, not a view of
      // this joint from the frame we are reporting in.
      const score = donorVisibility * donor.summary.quality;
      if (best && score <= best.score) continue;
      const projected = reproject(point, frame, primaryFrame);
      donor.summary.used = true;
      best = {
        score,
        joint: {
          index,
          ...projected,
          confidence: donorVisibility * 0.6,
          state: "inferred" as JointState,
          sourceCamera: donor.observation.cameraSlot,
        },
      };
    }
    return best?.joint ?? missingJoint(index);
  });

  const observingCameras = usable.map(({ observation }) => observation.cameraSlot).sort();
  const sameTrack = hadTrack;
  const consecutiveFrames = sameTrack ? previous.consecutiveFrames + 1 : 1;
  const presence: GlobalPresence =
    previous.presence === "present" || previous.presence === "coasting" ||
    consecutiveFrames >= tuning.confirmationFrames
      ? "present"
      : "provisional";
  const handedOff = sameTrack &&
    previous.coordinateFrame !== null &&
    previous.coordinateFrame !== primary.observation.cameraSlot;

  return {
    id: sameTrack && previous.id ? previous.id : newId(),
    createdAt: sameTrack && previous.createdAt ? previous.createdAt : timestamp,
    updatedAt: timestamp,
    lastObservedAt: timestamp,
    presence,
    consecutiveFrames,
    primaryCamera: primary.observation.cameraSlot,
    coordinateFrame: primary.observation.cameraSlot,
    calibrated: false,
    joints,
    observations: summaries,
    observingCameras,
    handoffFromCamera: handedOff ? previous.coordinateFrame : null,
    handoffGapMs: handedOff ? Math.max(0, timestamp - previous.lastObservedAt) : 0,
  };
}

function previousTorsoCenter(track: GlobalPersonTrack) {
  const shoulders = track.joints.filter((joint) => joint.index === LEFT_SHOULDER || joint.index === RIGHT_SHOULDER);
  const hips = track.joints.filter((joint) => joint.index === LEFT_HIP || joint.index === RIGHT_HIP);
  const points = [...shoulders, ...hips].filter((joint) => joint.state !== "missing");
  if (!points.length) return null;
  return {
    x: points.reduce((sum, joint) => sum + joint.x, 0) / points.length,
    y: points.reduce((sum, joint) => sum + joint.y, 0) / points.length,
  };
}

/**
 * Analysis weight for a fused joint. Inferred joints are real information but
 * must never carry the same weight as a joint the reporting camera actually saw.
 */
export function jointAnalysisWeight(joint: GlobalJoint): number {
  if (joint.state === "missing") return 0;
  return joint.state === "observed" ? joint.confidence : joint.confidence * 0.5;
}
