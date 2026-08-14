// Pure, DOM-free analysis helpers extracted from app/page.tsx so they can be
// unit-tested without a browser (see tests/motion-analysis.test.mjs).
//
// Honesty note: the on-device pose/hand sampling rate is MOTION_SAMPLE_RATE
// (5 Hz, see pose-store.ts). By the Nyquist limit that means signals faster
// than ~2.5 Hz cannot be resolved. Clinical hand tremor (e.g. essential or
// parkinsonian tremor) typically sits in the 4-12 Hz band, so the
// "variability"/"smoothness" numbers below are deliberately NOT labeled as
// tremor detection. They are coarse, unvalidated wellness signals only -
// keep that framing in any UI copy that surfaces them.

export type Point3D = { x: number; y: number; z: number };
export type NormalizedLandmarkLike = Point3D & { visibility?: number };

export const FULL_BODY_LANDMARKS = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28];

export function isFullBodyVisible(landmarks: NormalizedLandmarkLike[]): boolean {
  const allRequiredVisible = FULL_BODY_LANDMARKS.every((index) => {
    const point = landmarks[index];
    return (
      point &&
      (point.visibility ?? 0) >= 0.55 &&
      point.x >= 0.015 &&
      point.x <= 0.985 &&
      point.y >= 0.015 &&
      point.y <= 0.985
    );
  });
  if (!allRequiredVisible) return false;
  const ankleY = (landmarks[27].y + landmarks[28].y) / 2;
  const headY = (landmarks[7].y + landmarks[8].y) / 2;
  return ankleY - headY >= 0.5;
}

export type HeadDirection = {
  yaw: number;
  pitch: number;
  roll: number;
  centerX: number;
  centerY: number;
};

export function getHeadDirection(
  landmarks: NormalizedLandmarkLike[],
): HeadDirection | null {
  const nose = landmarks[0];
  const leftEar = landmarks[7];
  const rightEar = landmarks[8];
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  if (
    !nose ||
    !leftEar ||
    !rightEar ||
    (leftEar.visibility ?? 0) < 0.35 ||
    (rightEar.visibility ?? 0) < 0.35
  ) {
    return null;
  }
  const centerX = (leftEar.x + rightEar.x) / 2;
  const centerY = (leftEar.y + rightEar.y) / 2;
  const shoulderWidth = Math.max(
    0.08,
    Math.abs(leftShoulder.x - rightShoulder.x),
  );
  const clamp = (value: number) => Math.max(-1, Math.min(1, value));
  return {
    yaw: clamp((nose.x - centerX) / (shoulderWidth * 0.32)),
    pitch: clamp((nose.y - centerY) / (shoulderWidth * 0.32)),
    roll: Math.atan2(rightEar.y - leftEar.y, rightEar.x - leftEar.x),
    centerX,
    centerY,
  };
}

export function describeHeadDirection(head: HeadDirection | null): string {
  if (!head) return "머리 방향 미확인";
  if (head.pitch < -0.32) return "위를 보는 중";
  if (head.pitch > 0.32) return "아래를 보는 중";
  if (head.yaw < -0.3) return "왼쪽을 보는 중";
  if (head.yaw > 0.3) return "오른쪽을 보는 중";
  return "정면을 보는 중";
}

/**
 * Coefficient of variation of frame-to-frame hand speed. Higher = jerkier /
 * more erratic hand motion between frames. This is NOT a tremor-frequency
 * measurement (see file header). Returns null when there isn't enough data.
 */
export function computeHandMotionVariability(
  handFrames: Array<Point3D | null>,
): number | null {
  const speeds: number[] = [];
  for (let i = 1; i < handFrames.length; i += 1) {
    const previous = handFrames[i - 1];
    const current = handFrames[i];
    if (!previous || !current) continue;
    const dx = current.x - previous.x;
    const dy = current.y - previous.y;
    const dz = current.z - previous.z;
    speeds.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  if (speeds.length < 4) return null;
  const mean = speeds.reduce((sum, value) => sum + value, 0) / speeds.length;
  if (mean === 0) return 0;
  const variance =
    speeds.reduce((sum, value) => sum + (value - mean) ** 2, 0) / speeds.length;
  const stdDev = Math.sqrt(variance);
  return stdDev / mean;
}

/**
 * Approximates where to draw a head marker when only shoulder landmarks are
 * available. Stored session frames don't include ear positions (see
 * pose-store.ts BODY_LANDMARK_NAMES - storage starts at the shoulders), so
 * the session-replay viewer can't recover the real head center used by the
 * live overlay. This is a rough visual placement for replay/debug display
 * only, not a real head-position measurement.
 *
 * `flatBody` is the stored [x,y,z,visibility] * BODY_LANDMARK_COUNT layout,
 * where index 0 is the left shoulder and index 1 is the right shoulder.
 */
export function approximateHeadCenterFromShoulders(
  flatBody: number[],
): { x: number; y: number } | null {
  const leftShoulderX = flatBody[0];
  const leftShoulderY = flatBody[1];
  const leftShoulderVis = flatBody[3];
  const rightShoulderX = flatBody[4];
  const rightShoulderY = flatBody[5];
  const rightShoulderVis = flatBody[7];
  if (
    leftShoulderVis === undefined ||
    leftShoulderVis < 0.35 ||
    rightShoulderVis === undefined ||
    rightShoulderVis < 0.35
  ) {
    return null;
  }
  const shoulderWidth = Math.max(0.08, Math.abs(leftShoulderX - rightShoulderX));
  return {
    x: (leftShoulderX + rightShoulderX) / 2,
    y: Math.min(leftShoulderY, rightShoulderY) - shoulderWidth * 0.9,
  };
}

/**
 * Normalized-jerk-style smoothness metric over a body/hand landmark
 * trajectory. Lower = smoother motion. Unvalidated, exploratory signal only.
 */
export function computeMovementSmoothness(
  points: Array<Point3D | null>,
): number | null {
  const velocities: Point3D[] = [];
  for (let i = 1; i < points.length; i += 1) {
    const previous = points[i - 1];
    const current = points[i];
    if (!previous || !current) continue;
    velocities.push({
      x: current.x - previous.x,
      y: current.y - previous.y,
      z: current.z - previous.z,
    });
  }
  if (velocities.length < 4) return null;
  let jerkSum = 0;
  let count = 0;
  for (let i = 1; i < velocities.length; i += 1) {
    const dvx = velocities[i].x - velocities[i - 1].x;
    const dvy = velocities[i].y - velocities[i - 1].y;
    const dvz = velocities[i].z - velocities[i - 1].z;
    jerkSum += Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
    count += 1;
  }
  return count > 0 ? jerkSum / count : null;
}
