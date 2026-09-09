import type { NormalizedLandmarkLike } from "./motion-analysis";

const CORE_LANDMARKS = [11, 12] as const;
// Head and face landmarks are intentionally excluded from target identity.
// Tracking is based on torso, arm, and leg continuity only.
const BODY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] as const;
const MAX_CENTER_STEP = 0.55;
const MAX_SCALE_LOG_CHANGE = 0.65;
// Shape (shoulder/hip/leg ratios) is the only real identity signal this
// single 2D camera has - position and scale alone can't tell the locked
// person apart from a similarly-sized stranger who steps into roughly the
// same spot. Weighted and gated harder than before for that reason (a body
// that appears in the general area with a clearly different build must not
// be picked up as a continuity match).
const MAX_SHAPE_DISTANCE = 0.5;
const MAX_CONTINUITY_SCORE = 0.4;
const REACQUIRE_AFTER_MS = 650;

type TargetDescriptor = {
  centerX: number;
  centerY: number;
  height: number;
  width: number;
  torsoLength: number;
  shoulderRatio: number;
  hipRatio: number;
  legRatio: number;
  quality: number;
};

export type PoseTargetLock = {
  acquiredAt: number;
  lastSeenAt: number;
  initial: TargetDescriptor;
  latest: TargetDescriptor;
};

export type PoseTargetSelection<T extends NormalizedLandmarkLike> = {
  landmarks: T[] | null;
  lock: PoseTargetLock | null;
  state: "searching" | "acquired" | "tracking" | "missing";
};

function imageDistance(a: NormalizedLandmarkLike, b: NormalizedLandmarkLike): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function landmarkVisibility(point: NormalizedLandmarkLike): number {
  return Math.max(0, Math.min(1, point.visibility ?? 0));
}

function visibilityWeightedCenter(
  first: NormalizedLandmarkLike,
  second: NormalizedLandmarkLike,
) {
  const firstWeight = Math.max(0.05, landmarkVisibility(first));
  const secondWeight = Math.max(0.05, landmarkVisibility(second));
  const weight = firstWeight + secondWeight;
  return {
    x: (first.x * firstWeight + second.x * secondWeight) / weight,
    y: (first.y * firstWeight + second.y * secondWeight) / weight,
    z: (first.z * firstWeight + second.z * secondWeight) / weight,
  };
}

function pairIsVisible(
  first: NormalizedLandmarkLike,
  second: NormalizedLandmarkLike,
  threshold = 0.25,
): boolean {
  return landmarkVisibility(first) >= threshold && landmarkVisibility(second) >= threshold;
}

function descriptorFor(landmarks: NormalizedLandmarkLike[]): TargetDescriptor | null {
  if (landmarks.length < 29) return null;
  if (CORE_LANDMARKS.some((index) => !landmarks[index])) return null;

  const visiblePoints = BODY_LANDMARKS
    .map((index) => landmarks[index])
    .filter((point) => point && landmarkVisibility(point) >= 0.25);
  if (visiblePoints.length < 4) return null;

  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];
  const shoulderCenter = visibilityWeightedCenter(leftShoulder, rightShoulder);
  const shoulderWidth = pairIsVisible(leftShoulder, rightShoulder, 0.15)
    ? imageDistance(leftShoulder, rightShoulder)
    : 0;
  const hipsVisible = pairIsVisible(leftHip, rightHip);
  const hipCenter = hipsVisible
    ? visibilityWeightedCenter(leftHip, rightHip)
    : { x: shoulderCenter.x, y: shoulderCenter.y + shoulderWidth / 0.6, z: shoulderCenter.z };
  const torsoLength = hipsVisible
    ? imageDistance(shoulderCenter, hipCenter)
    : shoulderWidth / 0.6;
  if (torsoLength < 0.045) return null;

  const xs = visiblePoints.map((point) => point.x);
  const ys = visiblePoints.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  // Torso scale stays comparable when legs or arms move outside the frame.
  const height = Math.max(0.12, torsoLength * 2.8, Math.min(torsoLength * 3.6, maxY - minY));
  const width = Math.max(0.08, maxX - minX);
  const hipWidth = hipsVisible
    ? imageDistance(leftHip, rightHip)
    : Number.NaN;
  const leftLeg = pairIsVisible(landmarks[25], landmarks[27])
    ? imageDistance(leftHip, landmarks[25]) + imageDistance(landmarks[25], landmarks[27])
    : Number.NaN;
  const rightLeg = pairIsVisible(landmarks[26], landmarks[28])
    ? imageDistance(rightHip, landmarks[26]) + imageDistance(landmarks[26], landmarks[28])
    : Number.NaN;
  const visibleLegs = [leftLeg, rightLeg].filter(Number.isFinite);

  return {
    centerX: (shoulderCenter.x + hipCenter.x) / 2,
    centerY: (shoulderCenter.y + hipCenter.y) / 2,
    height,
    width,
    torsoLength,
    shoulderRatio: shoulderWidth / torsoLength,
    hipRatio: hipWidth / torsoLength,
    legRatio: visibleLegs.length
      ? visibleLegs.reduce((sum, value) => sum + value, 0) / (visibleLegs.length * torsoLength)
      : Number.NaN,
    quality:
      visiblePoints.reduce((sum, point) => sum + (point.visibility ?? 0), 0) /
      visiblePoints.length,
  };
}

function shapeDistance(a: TargetDescriptor, b: TargetDescriptor): number {
  const ratios = ([
    [a.shoulderRatio, b.shoulderRatio],
    [a.hipRatio, b.hipRatio],
    [a.legRatio, b.legRatio],
  ] as Array<[number, number]>).filter(([left, right]) =>
    Number.isFinite(left) && Number.isFinite(right),
  );
  if (ratios.length === 0) return 0;
  return ratios.reduce(
    (sum, [left, right]) => sum + Math.abs(Math.log(Math.max(0.01, left) / Math.max(0.01, right))),
    0,
  ) / ratios.length;
}

function acquisitionScore(descriptor: TargetDescriptor): number {
  const centerDistance = Math.hypot(descriptor.centerX - 0.5, descriptor.centerY - 0.5);
  return descriptor.quality * 0.5 + Math.min(1, descriptor.height) * 0.35 - centerDistance * 0.15;
}

function continuityScore(lock: PoseTargetLock, candidate: TargetDescriptor): number {
  const referenceHeight = Math.max(0.16, lock.latest.height, candidate.height);
  const centerStep = Math.hypot(
    candidate.centerX - lock.latest.centerX,
    candidate.centerY - lock.latest.centerY,
  ) / referenceHeight;
  const scaleChange = Math.abs(Math.log(candidate.height / lock.latest.height));
  const currentShapeDistance = shapeDistance(lock.initial, candidate);
  if (
    centerStep > MAX_CENTER_STEP ||
    scaleChange > MAX_SCALE_LOG_CHANGE ||
    currentShapeDistance > MAX_SHAPE_DISTANCE
  ) {
    return Number.POSITIVE_INFINITY;
  }
  return centerStep * 0.4 + scaleChange * 0.2 + currentShapeDistance * 0.4;
}

/**
 * Locks onto the strongest usable upper-body or full-body pose immediately,
 * then accepts only spatially/anatomically continuous body candidates. Head
 * landmarks are never part of the identity descriptor. A rejected frame is
 * reported as missing without releasing the lock, so temporary cropping does
 * not silently switch tracking to another person or a pose-like object.
 */
export function selectLockedPose<T extends NormalizedLandmarkLike>(
  candidates: T[][],
  currentLock: PoseTargetLock | null,
  timestamp: number,
): PoseTargetSelection<T> {
  const described = candidates
    .map((landmarks) => ({ landmarks, descriptor: descriptorFor(landmarks) }))
    .filter((candidate): candidate is { landmarks: T[]; descriptor: TargetDescriptor } =>
      candidate.descriptor !== null,
    );

  if (!currentLock) {
    const rankedCandidates = described
      .sort((a, b) => acquisitionScore(b.descriptor) - acquisitionScore(a.descriptor));
    const selected = rankedCandidates[0];
    if (!selected) {
      return { landmarks: null, lock: null, state: "searching" };
    }
    return {
      landmarks: selected.landmarks,
      lock: {
        acquiredAt: timestamp,
        lastSeenAt: timestamp,
        initial: selected.descriptor,
        latest: selected.descriptor,
      },
      state: "acquired",
    };
  }

  const matches = described
    .map((candidate) => ({ ...candidate, score: continuityScore(currentLock, candidate.descriptor) }))
    .filter((candidate) => candidate.score <= MAX_CONTINUITY_SCORE)
    .sort((a, b) => a.score - b.score);
  const selected = matches[0];
  if (!selected) {
    const reacquired = timestamp - currentLock.lastSeenAt >= REACQUIRE_AFTER_MS
      ? described.sort((a, b) => acquisitionScore(b.descriptor) - acquisitionScore(a.descriptor))[0]
      : null;
    if (reacquired) {
      return {
        landmarks: reacquired.landmarks,
        lock: {
          acquiredAt: timestamp,
          lastSeenAt: timestamp,
          initial: reacquired.descriptor,
          latest: reacquired.descriptor,
        },
        state: "acquired",
      };
    }
    return { landmarks: null, lock: currentLock, state: "missing" };
  }

  return {
    landmarks: selected.landmarks,
    lock: {
      ...currentLock,
      lastSeenAt: timestamp,
      latest: selected.descriptor,
    },
    state: "tracking",
  };
}

/**
 * Associates a separately detected hand with the closest pose wrist.
 * PoseLandmarker and HandLandmarker do not share the same z-coordinate scale,
 * so cross-model association must use normalized image x/y coordinates only.
 */
export function matchHandToPose(
  hand: NormalizedLandmarkLike[],
  pose: NormalizedLandmarkLike[] | null,
): "left" | "right" | null {
  if (!pose || !hand[0] || !pose[15] || !pose[16]) return null;
  const descriptor = descriptorFor(pose);
  if (!descriptor) return null;
  const wrist = hand[0];
  const leftDistance = imageDistance(wrist, pose[15]);
  const rightDistance = imageDistance(wrist, pose[16]);
  const threshold = Math.max(0.14, descriptor.height * 0.32);
  if (Math.min(leftDistance, rightDistance) > threshold) return null;
  return leftDistance <= rightDistance ? "left" : "right";
}

/** Only attach hand landmarks that are spatially close to the tracked body. */
export function handBelongsToPose(
  hand: NormalizedLandmarkLike[],
  pose: NormalizedLandmarkLike[] | null,
): boolean {
  return matchHandToPose(hand, pose) !== null;
}
