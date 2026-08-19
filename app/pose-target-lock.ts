import {
  isFullBodyVisible,
  type NormalizedLandmarkLike,
} from "./motion-analysis";

const CORE_LANDMARKS = [11, 12, 23, 24] as const;
const BODY_LANDMARKS = [7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] as const;
const MAX_CENTER_STEP = 0.72;
const MAX_SCALE_LOG_CHANGE = 0.52;
const MAX_SHAPE_DISTANCE = 0.5;
const MAX_CONTINUITY_SCORE = 0.68;

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

function distance(a: NormalizedLandmarkLike, b: NormalizedLandmarkLike): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function descriptorFor(landmarks: NormalizedLandmarkLike[]): TargetDescriptor | null {
  if (landmarks.length < 29) return null;
  if (CORE_LANDMARKS.some((index) => !landmarks[index])) return null;

  const visiblePoints = BODY_LANDMARKS
    .map((index) => landmarks[index])
    .filter((point) => point && (point.visibility ?? 0) >= 0.35);
  if (visiblePoints.length < 8) return null;

  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];
  const shoulderCenter = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
    z: (leftShoulder.z + rightShoulder.z) / 2,
  };
  const hipCenter = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
    z: (leftHip.z + rightHip.z) / 2,
  };
  const torsoLength = distance(shoulderCenter, hipCenter);
  if (torsoLength < 0.08) return null;

  const xs = visiblePoints.map((point) => point.x);
  const ys = visiblePoints.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const height = Math.max(0.12, maxY - minY);
  const width = Math.max(0.08, maxX - minX);
  const shoulderWidth = distance(leftShoulder, rightShoulder);
  const hipWidth = distance(leftHip, rightHip);
  const leftLeg = landmarks[25] && landmarks[27]
    ? distance(leftHip, landmarks[25]) + distance(landmarks[25], landmarks[27])
    : torsoLength * 2;
  const rightLeg = landmarks[26] && landmarks[28]
    ? distance(rightHip, landmarks[26]) + distance(landmarks[26], landmarks[28])
    : torsoLength * 2;

  return {
    centerX: (shoulderCenter.x + hipCenter.x) / 2,
    centerY: (shoulderCenter.y + hipCenter.y) / 2,
    height,
    width,
    torsoLength,
    shoulderRatio: shoulderWidth / torsoLength,
    hipRatio: hipWidth / torsoLength,
    legRatio: (leftLeg + rightLeg) / (2 * torsoLength),
    quality:
      visiblePoints.reduce((sum, point) => sum + (point.visibility ?? 0), 0) /
      visiblePoints.length,
  };
}

function shapeDistance(a: TargetDescriptor, b: TargetDescriptor): number {
  const ratios: Array<[number, number]> = [
    [a.shoulderRatio, b.shoulderRatio],
    [a.hipRatio, b.hipRatio],
    [a.legRatio, b.legRatio],
  ];
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
  return centerStep * 0.58 + scaleChange * 0.24 + currentShapeDistance * 0.18;
}

/**
 * Locks onto one full-body pose and only accepts spatially/anatomically
 * continuous candidates afterwards. A rejected frame is reported as missing
 * instead of silently switching to another person or a person-shaped object.
 * The lock intentionally lives until the camera session ends.
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
    const acquisitionCandidates = described
      .filter((candidate) => isFullBodyVisible(candidate.landmarks))
      .sort((a, b) => acquisitionScore(b.descriptor) - acquisitionScore(a.descriptor));
    const selected = acquisitionCandidates[0];
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

/** Only attach hand landmarks that are spatially close to the locked body. */
export function handBelongsToPose(
  hand: NormalizedLandmarkLike[],
  pose: NormalizedLandmarkLike[] | null,
): boolean {
  if (!pose || !hand[0] || !pose[15] || !pose[16]) return false;
  const descriptor = descriptorFor(pose);
  if (!descriptor) return false;
  const wrist = hand[0];
  const wristDistance = Math.min(distance(wrist, pose[15]), distance(wrist, pose[16]));
  return wristDistance <= Math.max(0.11, descriptor.height * 0.28);
}
