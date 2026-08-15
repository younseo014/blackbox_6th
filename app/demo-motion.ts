// Synthetic (procedurally generated, deterministic) skeleton motion clips
// for the demo persona events in app/demo-personas.ts.
//
// These are NOT recordings and are not derived from any real user's
// movement. They exist so a developer can preview - via the same
// session-replay viewer used for real camera sessions - what a plausible
// physical movement might look like for a flagged event type ("마감 반복
// 확인", "미세 지연", "안전 알림", ...), since a virtual persona obviously
// has no real recording to show. No Math.random()/Date.now() is used, so
// the same (type, seed) always produces the exact same clip - both for
// reproducible unit tests and so replaying an event twice looks identical.
//
// The output is the same raw per-frame layout pose-store.ts stores
// (see MOTION_FRAME_STRIDE), so it flows through parseSessionFrame() and
// the SessionReplayPanel renderer completely unchanged from a real session.
//
// Beyond just moving a hand, each clip can also lean the torso, bend the
// knees ("crouch"), shift weight side to side, and move BOTH arms - so
// different event types read as visibly different body language, not just
// the same arm swing played at a different speed.

import {
  BODY_LANDMARK_COUNT,
  HAND_LANDMARK_COUNT,
  MOTION_FRAME_STRIDE,
  MOTION_SAMPLE_RATE,
} from "./pose-store";

export type DemoMotionType =
  | "double_check"
  | "micro_delay"
  | "safety_alert"
  | "normal_task"
  | "fine_hand_task"
  | "register_tap"
  | "queue_shift"
  | "high_reach"
  | "low_bend";

type Point2D = { x: number; y: number };

const FRAME_INTERVAL_MS = 1000 / MOTION_SAMPLE_RATE;

// Baseline standing pose, one point per storedIndex 0..21 - see
// BODY_LANDMARK_NAMES in pose-store.ts (starts at the shoulders, in the
// same order: shoulders, elbows, wrists, hand anchors, hips, knees,
// ankles, heels, foot index). Normalized image coordinates.
const BASE_BODY: Point2D[] = [
  { x: 0.42, y: 0.3 }, // left_shoulder
  { x: 0.58, y: 0.3 }, // right_shoulder
  { x: 0.4, y: 0.42 }, // left_elbow
  { x: 0.6, y: 0.42 }, // right_elbow
  { x: 0.39, y: 0.53 }, // left_wrist
  { x: 0.61, y: 0.53 }, // right_wrist
  { x: 0.385, y: 0.55 }, // left_pinky_anchor
  { x: 0.615, y: 0.55 }, // right_pinky_anchor
  { x: 0.385, y: 0.545 }, // left_index_anchor
  { x: 0.615, y: 0.545 }, // right_index_anchor
  { x: 0.395, y: 0.535 }, // left_thumb_anchor
  { x: 0.605, y: 0.535 }, // right_thumb_anchor
  { x: 0.44, y: 0.55 }, // left_hip
  { x: 0.56, y: 0.55 }, // right_hip
  { x: 0.44, y: 0.75 }, // left_knee
  { x: 0.56, y: 0.75 }, // right_knee
  { x: 0.44, y: 0.95 }, // left_ankle
  { x: 0.56, y: 0.95 }, // right_ankle
  { x: 0.435, y: 0.97 }, // left_heel
  { x: 0.565, y: 0.97 }, // right_heel
  { x: 0.45, y: 0.98 }, // left_foot_index
  { x: 0.55, y: 0.98 }, // right_foot_index
];

const LEFT_SHOULDER_INDEX = 0;
const RIGHT_SHOULDER_INDEX = 1;
const LEFT_ELBOW_INDEX = 2;
const RIGHT_ELBOW_INDEX = 3;
const LEFT_WRIST_INDEX = 4;
const RIGHT_WRIST_INDEX = 5;
// Hips/knees/ankles/heels/foot-index - the whole lower body, which moves
// together for a crouch or a side-to-side weight shift.
const LOWER_BODY_INDICES = new Set([12, 13, 14, 15, 16, 17, 18, 19, 20, 21]);

const REST_RIGHT: Point2D = { x: 0.61, y: 0.53 };
const REST_LEFT: Point2D = { x: 0.4, y: 0.58 };
const REST_RIGHT_ANGLE = 0.15;
const REST_LEFT_ANGLE = Math.PI - 0.15;

// Rough finger geometry (angle from "straight up" at the wrist, plus 4
// joint segment lengths) - not anatomically exact, just plausible enough
// for a debug skeleton view.
const FINGER_DEFS: Array<{ angle: number; lengths: number[] }> = [
  { angle: -0.9, lengths: [0.022, 0.018, 0.016, 0.014] }, // thumb
  { angle: -0.24, lengths: [0.046, 0.03, 0.022, 0.02] }, // index
  { angle: -0.05, lengths: [0.05, 0.032, 0.024, 0.02] }, // middle
  { angle: 0.15, lengths: [0.046, 0.03, 0.022, 0.02] }, // ring
  { angle: 0.34, lengths: [0.04, 0.025, 0.018, 0.016] }, // pinky
];

function buildSyntheticHand(center: Point2D, handAngleRad: number, openness: number): Point2D[] {
  const points: Point2D[] = [{ x: center.x, y: center.y }];
  for (const finger of FINGER_DEFS) {
    const dirAngle = handAngleRad + finger.angle;
    const dx = Math.sin(dirAngle);
    const dy = -Math.cos(dirAngle);
    let cumulative = 0;
    for (const segmentLength of finger.lengths) {
      cumulative += segmentLength * (0.45 + 0.55 * openness);
      points.push({ x: center.x + dx * cumulative, y: center.y + dy * cumulative });
    }
  }
  return points;
}

function idleSway(frameIndex: number, seed: number) {
  return {
    x: Math.sin((frameIndex + seed * 3) * 0.18) * 0.003,
    y: Math.cos((frameIndex + seed * 5) * 0.13) * 0.002,
  };
}

function idlePoint(base: Point2D, frameIndex: number, seed: number): Point2D {
  const sway = idleSway(frameIndex, seed);
  return { x: base.x + sway.x, y: base.y + sway.y };
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function addPoints(a: Point2D, b: Point2D): Point2D {
  return { x: a.x + b.x, y: a.y + b.y };
}

// A smooth (raised-cosine) bump: 0 far from `center`, eases up to 1 at
// `center`, eases back to 0 - used to shape approach/hold/retreat motions
// without hard velocity discontinuities.
function bump(t: number, center: number, halfWidth: number): number {
  const d = Math.abs(t - center) / halfWidth;
  if (d >= 1) return 0;
  return 0.5 * (1 + Math.cos(Math.PI * d));
}

function lerpPoint(a: Point2D, b: Point2D, t: number): Point2D {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

// Blends a base value toward one or more weighted targets. Weights are
// expected not to overlap (e.g. two non-overlapping bump() windows), so at
// most one entry is ever meaningfully non-zero at a time; when all weights
// are 0 this returns `base` unchanged. Used so a clip can visit two
// different target points (e.g. two different check-spots) with a single
// easing scalar per target, without special-casing which one is "active".
function blendPoint(base: Point2D, entries: Array<{ point: Point2D; weight: number }>): Point2D {
  let usedWeight = 0;
  let x = 0;
  let y = 0;
  for (const entry of entries) {
    x += entry.point.x * entry.weight;
    y += entry.point.y * entry.weight;
    usedWeight += entry.weight;
  }
  const baseWeight = Math.max(0, 1 - usedWeight);
  return { x: x + base.x * baseWeight, y: y + base.y * baseWeight };
}

type FrameSpec = {
  headYaw: number;
  headPitch: number;
  headRoll: number;
  /** Whole-upper-body offset (leaning forward/back/sideways toward something). */
  torsoLean?: Point2D;
  /** 0..1 knee bend - lowers the whole lower body, for reaching/checking something low. */
  crouch?: number;
  /** Side-to-side weight shift, independent of torsoLean (e.g. pacing/waiting). */
  weightShiftX?: number;
  rightHandCenter: Point2D;
  rightHandAngle: number;
  rightOpenness: number;
  leftHandCenter: Point2D;
  leftHandAngle: number;
  leftOpenness: number;
};

function buildRawFrame(frameIndex: number, spec: FrameSpec): number[] {
  const frame = new Array(MOTION_FRAME_STRIDE).fill(0);
  frame[0] = frameIndex * FRAME_INTERVAL_MS;
  frame[1] = 1; // body detected
  frame[2] = 1; // full body visible
  frame[3] = spec.headYaw;
  frame[4] = spec.headPitch;
  frame[5] = spec.headRoll;
  frame[6] = 1; // left hand detected
  frame[7] = 1; // right hand detected
  frame[8] = 0.85; // left hand confidence
  frame[9] = 0.85; // right hand confidence

  const bodyStart = 10;
  const torsoLean = spec.torsoLean ?? { x: 0, y: 0 };
  const crouch = spec.crouch ?? 0;
  const weightShiftX = spec.weightShiftX ?? 0;
  // The lower body follows the torso lean at reduced strength (a lean
  // isn't a full hinge at the hips) plus its own crouch/weight-shift.
  const lowerBodyOffset: Point2D = {
    x: torsoLean.x * 0.3 + weightShiftX,
    y: torsoLean.y * 0.3 + crouch * 0.07,
  };
  const rightShoulder = addPoints(BASE_BODY[RIGHT_SHOULDER_INDEX], torsoLean);
  const leftShoulder = addPoints(BASE_BODY[LEFT_SHOULDER_INDEX], torsoLean);

  const bodyPoints = BASE_BODY.map((point, index) => {
    if (index === RIGHT_SHOULDER_INDEX) return rightShoulder;
    if (index === LEFT_SHOULDER_INDEX) return leftShoulder;
    if (index === RIGHT_WRIST_INDEX) return spec.rightHandCenter;
    if (index === LEFT_WRIST_INDEX) return spec.leftHandCenter;
    if (index === RIGHT_ELBOW_INDEX) {
      // Rough single-joint "elbow follows halfway between shoulder and
      // wrist, bowed slightly outward" - not real IK, just visually coherent.
      const mid = lerpPoint(rightShoulder, spec.rightHandCenter, 0.5);
      return { x: mid.x + 0.02, y: mid.y };
    }
    if (index === LEFT_ELBOW_INDEX) {
      const mid = lerpPoint(leftShoulder, spec.leftHandCenter, 0.5);
      return { x: mid.x - 0.02, y: mid.y };
    }
    if (LOWER_BODY_INDICES.has(index)) return addPoints(point, lowerBodyOffset);
    // Pinky/index/thumb anchor points (6-11) - not individually driven,
    // but still drift with the torso lean so they stay visually attached.
    if (index >= 6 && index <= 11) return addPoints(point, torsoLean);
    return point;
  });
  bodyPoints.forEach((point, index) => {
    frame[bodyStart + index * 4] = point.x;
    frame[bodyStart + index * 4 + 1] = point.y;
    frame[bodyStart + index * 4 + 2] = 0;
    frame[bodyStart + index * 4 + 3] = 1;
  });

  const leftHandStart = bodyStart + BODY_LANDMARK_COUNT * 4;
  buildSyntheticHand(spec.leftHandCenter, spec.leftHandAngle, spec.leftOpenness).forEach(
    (point, index) => {
      frame[leftHandStart + index * 3] = point.x;
      frame[leftHandStart + index * 3 + 1] = point.y;
      frame[leftHandStart + index * 3 + 2] = 0;
    },
  );

  const rightHandStart = leftHandStart + HAND_LANDMARK_COUNT * 3;
  buildSyntheticHand(spec.rightHandCenter, spec.rightHandAngle, spec.rightOpenness).forEach(
    (point, index) => {
      frame[rightHandStart + index * 3] = point.x;
      frame[rightHandStart + index * 3 + 1] = point.y;
      frame[rightHandStart + index * 3 + 2] = 0;
    },
  );

  return frame;
}

const CLIP_SECONDS: Record<DemoMotionType, number> = {
  double_check: 8,
  micro_delay: 7,
  safety_alert: 4.5,
  normal_task: 6,
  fine_hand_task: 6,
  register_tap: 6.5,
  queue_shift: 7,
  high_reach: 5.5,
  low_bend: 6,
};

function frameCountFor(type: DemoMotionType): number {
  return Math.round(CLIP_SECONDS[type] * MOTION_SAMPLE_RATE);
}

// A small deterministic offset derived from the seed, so events of the same
// type don't play back pixel-identical when several appear back to back.
function seedJitter(seed: number, scale: number): number {
  return (((seed * 37) % 11) / 10 - 0.5) * scale;
}

/**
 * "마감 반복 확인" - reaches toward two DIFFERENT check points (e.g. door
 * handle, then a latch higher up) with a clear full stop between them, plus
 * a forward torso lean and a hand-twist as if checking a lock. Two
 * separated peaks in hand displacement, with a wide flat rest in between,
 * is the signature this leaves in computeHandMotionVariability.
 */
function generateDoubleCheckMotion(seed: number): number[][] {
  const frameCount = frameCountFor("double_check");
  const firstTarget: Point2D = { x: 0.67 + seedJitter(seed, 0.02), y: 0.42 };
  const secondTarget: Point2D = { x: 0.73 + seedJitter(seed + 1, 0.02), y: 0.26 };
  const frames: number[][] = [];
  for (let i = 0; i < frameCount; i += 1) {
    const t = i / (frameCount - 1);
    const firstProgress = bump(t, 0.27, 0.1);
    const secondProgress = bump(t, 0.73, 0.1);
    const combined = clamp01(firstProgress + secondProgress);
    frames.push(
      buildRawFrame(i, {
        headYaw: 0.22 * combined,
        headPitch: -0.16 * combined,
        headRoll: 0.05 * secondProgress,
        torsoLean: { x: 0.02 * combined, y: -0.015 * combined },
        rightHandCenter: blendPoint(REST_RIGHT, [
          { point: firstTarget, weight: firstProgress },
          { point: secondTarget, weight: secondProgress },
        ]),
        rightHandAngle: REST_RIGHT_ANGLE - 0.7 * firstProgress - 0.95 * secondProgress,
        rightOpenness: 0.5 + 0.35 * combined,
        leftHandCenter: idlePoint(REST_LEFT, i, seed),
        leftHandAngle: REST_LEFT_ANGLE,
        leftOpenness: 0.4,
      }),
    );
  }
  return frames;
}

/**
 * "미세 지연" - starts reaching for one thing (e.g. an item at chest
 * height), freezes mid-motion for an extended pause (hand suspended, head
 * dipping as if hesitating), then resumes toward a SECOND, lower point to
 * actually finish the task. The frozen middle segment is an exact
 * near-zero-velocity plateau, distinct from the continuous ramps before
 * and after it.
 */
function generateMicroDelayMotion(seed: number): number[][] {
  const frameCount = frameCountFor("micro_delay");
  const reachTarget: Point2D = { x: 0.58 + seedJitter(seed, 0.02), y: 0.46 };
  const finishTarget: Point2D = { x: 0.5 + seedJitter(seed, 0.03), y: 0.67 };
  const frames: number[][] = [];
  for (let i = 0; i < frameCount; i += 1) {
    const t = i / (frameCount - 1);
    let rightHandCenter: Point2D;
    let headPitch: number;
    let handAngle: number;
    if (t < 0.28) {
      const local = clamp01(t / 0.28);
      rightHandCenter = lerpPoint(REST_RIGHT, reachTarget, local);
      headPitch = 0.05 + 0.12 * local;
      handAngle = REST_RIGHT_ANGLE - 0.35 * local;
    } else if (t < 0.72) {
      // the hesitation / freeze - held exactly at reachTarget
      rightHandCenter = reachTarget;
      headPitch = 0.17;
      handAngle = REST_RIGHT_ANGLE - 0.35;
    } else {
      const local = clamp01((t - 0.72) / 0.28);
      rightHandCenter = lerpPoint(reachTarget, finishTarget, local);
      headPitch = 0.17 - 0.12 * local;
      handAngle = REST_RIGHT_ANGLE - 0.35 + 0.2 * local;
    }
    frames.push(
      buildRawFrame(i, {
        headYaw: -0.06,
        headPitch,
        headRoll: 0,
        torsoLean: { x: 0.012, y: 0.018 },
        rightHandCenter,
        rightHandAngle: handAngle,
        rightOpenness: 0.5,
        leftHandCenter: idlePoint(REST_LEFT, i, seed),
        leftHandAngle: REST_LEFT_ANGLE,
        leftOpenness: 0.4,
      }),
    );
  }
  return frames;
}

/**
 * "안전 알림" - a fast, reactive full-body motion: crouches down toward a
 * floor-level target (e.g. slapping a plug/switch off), then straightens
 * back up quickly. The crouch plus a sharp narrow approach leaves a high
 * peak speed / jerk signature in computeMovementSmoothness.
 */
function generateSafetyAlertMotion(seed: number): number[][] {
  const frameCount = frameCountFor("safety_alert");
  const target: Point2D = { x: 0.63 + seedJitter(seed, 0.02), y: 0.78 };
  const frames: number[][] = [];
  for (let i = 0; i < frameCount; i += 1) {
    const t = i / (frameCount - 1);
    const progress = clamp01(Math.max(bump(t, 0.3, 0.09), bump(t, 0.58, 0.12) * 0.35));
    const crouch = clamp01(bump(t, 0.3, 0.16));
    frames.push(
      buildRawFrame(i, {
        headYaw: 0.3 * progress,
        headPitch: 0.28 * crouch,
        headRoll: 0.04 * progress,
        torsoLean: { x: 0.02 * progress, y: 0.05 * crouch },
        crouch,
        rightHandCenter: lerpPoint(REST_RIGHT, target, progress),
        rightHandAngle: REST_RIGHT_ANGLE - 0.9 * progress,
        rightOpenness: 0.3 + 0.55 * progress,
        leftHandCenter: idlePoint(REST_LEFT, i, seed),
        leftHandAngle: REST_LEFT_ANGLE,
        leftOpenness: 0.4,
      }),
    );
  }
  return frames;
}

/**
 * Baseline routine task - continuous side-to-side motion at chest height
 * (e.g. wiping a counter) with a matching mild torso sway and weight
 * shift, so the whole body reads as "working", not just one arm twitching.
 */
function generateNormalTaskMotion(seed: number): number[][] {
  const frameCount = frameCountFor("normal_task");
  const frames: number[][] = [];
  for (let i = 0; i < frameCount; i += 1) {
    const t = i / (frameCount - 1);
    const sway = Math.sin(t * Math.PI * 2.2 + seed * 0.6);
    const rightHandCenter: Point2D = { x: 0.58 + sway * 0.09, y: 0.5 + Math.abs(sway) * 0.03 };
    frames.push(
      buildRawFrame(i, {
        headYaw: 0.05 * sway,
        headPitch: 0.02,
        headRoll: 0,
        torsoLean: { x: sway * 0.012, y: 0 },
        weightShiftX: sway * 0.01,
        rightHandCenter,
        rightHandAngle: REST_RIGHT_ANGLE + sway * 0.15,
        rightOpenness: 0.5,
        leftHandCenter: idlePoint(REST_LEFT, i, seed),
        leftHandAngle: REST_LEFT_ANGLE,
        leftOpenness: 0.4,
      }),
    );
  }
  return frames;
}

/** Normal small-joint work such as counting cash or tapping a POS screen. */
function generateFineHandTaskMotion(seed: number): number[][] {
  const frameCount = frameCountFor("fine_hand_task");
  const target: Point2D = { x: 0.57 + seedJitter(seed, 0.012), y: 0.48 };
  const frames: number[][] = [];
  for (let i = 0; i < frameCount; i += 1) {
    const t = i / (frameCount - 1);
    const rhythm = Math.sin(t * Math.PI * 2 * 4 + seed * 0.25);
    const progress = 0.72 + rhythm * 0.08;
    frames.push(
      buildRawFrame(i, {
        headYaw: -0.04,
        headPitch: 0.08,
        headRoll: 0,
        torsoLean: { x: 0.006, y: 0.008 },
        rightHandCenter: lerpPoint(REST_RIGHT, target, progress),
        rightHandAngle: REST_RIGHT_ANGLE - 0.18,
        rightOpenness: 0.48 + rhythm * 0.28,
        leftHandCenter: { x: 0.47, y: 0.5 + rhythm * 0.006 },
        leftHandAngle: REST_LEFT_ANGLE + 0.1,
        leftOpenness: 0.45 - rhythm * 0.12,
      }),
    );
  }
  return frames;
}

/**
 * "결제·입력 재시도" - a burst of quick small taps (e.g. re-entering a card
 * or order), which then gets INTERRUPTED mid-sequence (holds mid-tap, not
 * back at rest), before one final decisive tap finishes it. Distinct from
 * micro_delay's single smooth hesitation: this one is rhythmic/repetitive
 * before it stalls, matching "재입력" style events rather than a one-off
 * reach.
 */
function generateRegisterTapMotion(seed: number): number[][] {
  const frameCount = frameCountFor("register_tap");
  const target: Point2D = { x: 0.56 + seedJitter(seed, 0.015), y: 0.47 };
  const frames: number[][] = [];
  for (let i = 0; i < frameCount; i += 1) {
    const t = i / (frameCount - 1);
    let progress: number;
    if (t < 0.55) {
      const envelope = clamp01(t / 0.18);
      const tapOscillation = 0.5 + 0.5 * Math.sin(t * Math.PI * 2 * 7 + seed);
      progress = envelope * (0.55 + 0.45 * tapOscillation);
    } else if (t < 0.78) {
      progress = 0.55; // interrupted - holds mid-tap, not back at rest
    } else {
      const local = clamp01((t - 0.78) / 0.22);
      progress = 0.55 + 0.45 * local;
    }
    frames.push(
      buildRawFrame(i, {
        headYaw: -0.03,
        headPitch: 0.06,
        headRoll: 0,
        torsoLean: { x: 0.01, y: 0.01 },
        rightHandCenter: lerpPoint(REST_RIGHT, target, progress),
        rightHandAngle: REST_RIGHT_ANGLE - 0.2 * progress,
        rightOpenness: 0.45,
        leftHandCenter: idlePoint(REST_LEFT, i, seed),
        leftHandAngle: REST_LEFT_ANGLE,
        leftOpenness: 0.4,
      }),
    );
  }
  return frames;
}

/**
 * "대기·서성임" - standing and waiting: weight shifts side to side, head
 * turns to check the line/door, hands stay mostly relaxed. Represents a
 * queue or a busy moment rather than a specific hand task.
 */
function generateQueueShiftMotion(seed: number): number[][] {
  const frameCount = frameCountFor("queue_shift");
  const frames: number[][] = [];
  for (let i = 0; i < frameCount; i += 1) {
    const t = i / (frameCount - 1);
    const shift = Math.sin(t * Math.PI * 2 * 1.3 + seed * 0.4);
    frames.push(
      buildRawFrame(i, {
        headYaw: shift * 0.4,
        headPitch: 0.03,
        headRoll: shift * 0.05,
        torsoLean: { x: shift * 0.025, y: 0 },
        weightShiftX: shift * 0.035,
        rightHandCenter: idlePoint(REST_RIGHT, i, seed),
        rightHandAngle: REST_RIGHT_ANGLE,
        rightOpenness: 0.4,
        leftHandCenter: idlePoint(REST_LEFT, i, seed),
        leftHandAngle: REST_LEFT_ANGLE,
        leftOpenness: 0.4,
      }),
    );
  }
  return frames;
}

/**
 * "높은 선반 정리" - both arms raise together toward a high shelf (two
 * reaches, e.g. shelving two items), a genuinely two-handed motion rather
 * than one arm moving while the other idles.
 */
function generateHighReachMotion(seed: number): number[][] {
  const frameCount = frameCountFor("high_reach");
  const rightTarget: Point2D = { x: 0.64 + seedJitter(seed, 0.02), y: 0.12 };
  const leftTarget: Point2D = { x: 0.36 + seedJitter(seed, 0.02), y: 0.14 };
  const frames: number[][] = [];
  for (let i = 0; i < frameCount; i += 1) {
    const t = i / (frameCount - 1);
    const progress = clamp01(Math.max(bump(t, 0.32, 0.22), bump(t, 0.68, 0.22)));
    frames.push(
      buildRawFrame(i, {
        headYaw: 0,
        headPitch: -0.25 * progress,
        headRoll: 0,
        torsoLean: { x: 0, y: -0.02 * progress },
        rightHandCenter: lerpPoint(REST_RIGHT, rightTarget, progress),
        rightHandAngle: REST_RIGHT_ANGLE - 0.5 * progress,
        rightOpenness: 0.55,
        leftHandCenter: lerpPoint(REST_LEFT, leftTarget, progress),
        leftHandAngle: REST_LEFT_ANGLE + 0.5 * progress,
        leftOpenness: 0.55,
      }),
    );
  }
  return frames;
}

/**
 * "허리 숙여 확인" - bends the torso and knees down toward a floor-level
 * target (e.g. checking a low shelf or the bottom of a door), holds, then
 * straightens back up. A genuine whole-body bend, not just an arm drop.
 */
function generateLowBendMotion(seed: number): number[][] {
  const frameCount = frameCountFor("low_bend");
  const target: Point2D = { x: 0.58 + seedJitter(seed, 0.02), y: 0.88 };
  const frames: number[][] = [];
  for (let i = 0; i < frameCount; i += 1) {
    const t = i / (frameCount - 1);
    const progress = clamp01(bump(t, 0.5, 0.38));
    frames.push(
      buildRawFrame(i, {
        headYaw: 0,
        headPitch: 0.35 * progress,
        headRoll: 0,
        torsoLean: { x: 0.02 * progress, y: 0.14 * progress },
        crouch: progress * 0.6,
        rightHandCenter: lerpPoint(REST_RIGHT, target, progress),
        rightHandAngle: REST_RIGHT_ANGLE - 0.4 * progress,
        rightOpenness: 0.4,
        leftHandCenter: idlePoint(REST_LEFT, i, seed),
        leftHandAngle: REST_LEFT_ANGLE,
        leftOpenness: 0.4,
      }),
    );
  }
  return frames;
}

/**
 * Generates a deterministic synthetic skeleton clip for a demo persona
 * event. Same (type, seed) always returns the same frames.
 */
export function generateEventMotion(type: DemoMotionType, seed = 0): number[][] {
  switch (type) {
    case "double_check":
      return generateDoubleCheckMotion(seed);
    case "micro_delay":
      return generateMicroDelayMotion(seed);
    case "safety_alert":
      return generateSafetyAlertMotion(seed);
    case "register_tap":
      return generateRegisterTapMotion(seed);
    case "fine_hand_task":
      return generateFineHandTaskMotion(seed);
    case "queue_shift":
      return generateQueueShiftMotion(seed);
    case "high_reach":
      return generateHighReachMotion(seed);
    case "low_bend":
      return generateLowBendMotion(seed);
    case "normal_task":
    default:
      return generateNormalTaskMotion(seed);
  }
}

export const DEMO_MOTION_LABELS: Record<DemoMotionType, string> = {
  double_check: "마감 반복 확인 동작",
  micro_delay: "미세 지연(머뭇거림) 동작",
  safety_alert: "안전 알림 반응 동작",
  normal_task: "평소 업무 동작",
  fine_hand_task: "손가락 중심의 정상 업무 동작",
  register_tap: "결제·입력 재시도 동작",
  queue_shift: "대기·서성임 동작",
  high_reach: "높은 선반 정리 동작",
  low_bend: "허리 숙여 확인 동작",
};
