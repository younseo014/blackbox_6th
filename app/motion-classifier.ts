import { parseSessionFrame } from "./pose-store";

const SAMPLE_COUNT = 28;
const DISPLAY_ASPECT_RATIO = 16 / 9;
const BODY_JOINTS = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17];
const FINGER_TIPS = [4, 8, 12, 16, 20];

type SignatureFrame = {
  values: number[];
  mask: number[];
};

type MotionSignature = {
  frames: SignatureFrame[];
  usableFrames: number;
  handCoverage: number;
};

function point(body: number[], index: number) {
  return {
    x: body[index * 4] * DISPLAY_ASPECT_RATIO,
    y: body[index * 4 + 1],
    visibility: body[index * 4 + 3] ?? 0,
  };
}

function midpoint(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function appendHandShape(
  values: number[],
  mask: number[],
  hand: number[] | null,
  scale: number,
) {
  if (!hand) {
    FINGER_TIPS.forEach(() => {
      values.push(0, 0);
      mask.push(0, 0);
    });
    return false;
  }
  const wristX = hand[0] * DISPLAY_ASPECT_RATIO;
  const wristY = hand[1];
  FINGER_TIPS.forEach((index) => {
    values.push(
      (hand[index * 3] * DISPLAY_ASPECT_RATIO - wristX) / scale,
      (hand[index * 3 + 1] - wristY) / scale,
    );
    mask.push(0.42, 0.42);
  });
  return true;
}

function frameSignature(rawFrame: number[]): { frame: SignatureFrame; center: { x: number; y: number }; scale: number; hasHand: boolean } | null {
  const parsed = parseSessionFrame(rawFrame);
  if (!parsed.bodyDetected) return null;
  const leftShoulder = point(parsed.body, 0);
  const rightShoulder = point(parsed.body, 1);
  if ([leftShoulder, rightShoulder].some((item) => item.visibility < 0.15)) {
    return null;
  }
  const shoulderCenter = midpoint(leftShoulder, rightShoulder);
  const shoulderSpan = Math.hypot(
    leftShoulder.x - rightShoulder.x,
    leftShoulder.y - rightShoulder.y,
  );
  const upperBodyScale = Math.max(0.08, shoulderSpan / 0.64);
  const values: number[] = [];
  const mask: number[] = [];
  BODY_JOINTS.forEach((index) => {
    const joint = point(parsed.body, index);
    const visible = joint.visibility >= 0.25 ? 1 : 0;
    values.push(
      (joint.x - shoulderCenter.x) / upperBodyScale,
      (joint.y - shoulderCenter.y) / upperBodyScale,
    );
    mask.push(visible, visible);
  });
  values.push(parsed.head?.yaw ?? 0, parsed.head?.pitch ?? 0, parsed.head?.roll ?? 0);
  mask.push(parsed.head ? 0.35 : 0, parsed.head ? 0.35 : 0, parsed.head ? 0.25 : 0);
  const leftHand = appendHandShape(values, mask, parsed.leftHand, upperBodyScale);
  const rightHand = appendHandShape(values, mask, parsed.rightHand, upperBodyScale);
  return {
    frame: { values, mask },
    center: shoulderCenter,
    scale: upperBodyScale,
    hasHand: leftHand || rightHand,
  };
}

function interpolateFrame(a: SignatureFrame, b: SignatureFrame, amount: number): SignatureFrame {
  return {
    values: a.values.map((value, index) => value + (b.values[index] - value) * amount),
    mask: a.mask.map((value, index) => Math.min(value, b.mask[index])),
  };
}

function resample(frames: SignatureFrame[], count: number): SignatureFrame[] {
  if (frames.length === 1) return Array.from({ length: count }, () => frames[0]);
  return Array.from({ length: count }, (_, index) => {
    const position = (index / (count - 1)) * (frames.length - 1);
    const before = Math.floor(position);
    const after = Math.min(frames.length - 1, before + 1);
    return interpolateFrame(frames[before], frames[after], position - before);
  });
}

/**
 * Builds a translation-, scale-, and duration-normalized skeleton signature.
 * The body is expressed relative to the shoulders so a cropped upper body is
 * enough for short developer tests. Lower-body joints still contribute when
 * visible, and missing joints are masked out instead of rejecting the frame.
 */
export function buildMotionSignature(rawFrames: number[][]): MotionSignature | null {
  const usable = rawFrames.map(frameSignature).filter((item): item is NonNullable<typeof item> => Boolean(item));
  if (usable.length < 6) return null;
  const firstCenter = usable[0].center;
  const firstFrame = usable[0].frame;
  const travelScale = usable[0].scale;
  const withTravel = usable.map((item) => {
    const values = item.frame.values.map((value, index) => value - firstFrame.values[index]);
    values.push((item.center.x - firstCenter.x) / travelScale, (item.center.y - firstCenter.y) / travelScale);
    return { values, mask: [...item.frame.mask, 0.8, 0.8] };
  });
  return {
    frames: resample(withTravel, SAMPLE_COUNT),
    usableFrames: usable.length,
    handCoverage: usable.filter((item) => item.hasHand).length / usable.length,
  };
}

function frameDistance(a: SignatureFrame, b: SignatureFrame) {
  let weightedError = 0;
  let totalWeight = 0;
  for (let index = 0; index < a.values.length; index += 1) {
    const weight = Math.min(a.mask[index] ?? 0, b.mask[index] ?? 0);
    if (weight <= 0) continue;
    weightedError += (a.values[index] - b.values[index]) ** 2 * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? Math.sqrt(weightedError / totalWeight) : 10;
}

function signatureDistance(a: MotionSignature, b: MotionSignature) {
  let previous = Array(b.frames.length + 1).fill(Number.POSITIVE_INFINITY);
  previous[0] = 0;
  for (const observed of a.frames) {
    const current = Array(b.frames.length + 1).fill(Number.POSITIVE_INFINITY);
    for (let index = 1; index <= b.frames.length; index += 1) {
      current[index] = frameDistance(observed, b.frames[index - 1]) + Math.min(
        previous[index],
        current[index - 1],
        previous[index - 1],
      );
    }
    previous = current;
  }
  return previous[b.frames.length] / Math.max(a.frames.length, b.frames.length);
}

function mirrorSignature(signature: MotionSignature): MotionSignature {
  return {
    ...signature,
    frames: signature.frames.map((frame) => ({
      mask: frame.mask,
      values: frame.values.map((value, index) => {
        const bodyX = index < 24 && index % 2 === 0;
        const headYaw = index === 24;
        const handX = index >= 27 && index < 47 && (index - 27) % 2 === 0;
        const travelX = index === 47;
        return bodyX || headYaw || handX || travelX ? -value : value;
      }),
    })),
  };
}

function viewInvariantDistance(observed: MotionSignature, reference: MotionSignature) {
  return Math.min(
    signatureDistance(observed, reference),
    signatureDistance(mirrorSignature(observed), reference),
  );
}

export function compareSkeletonMotions(observedFrames: number[][], referenceFrames: number[][]) {
  const observed = buildMotionSignature(observedFrames);
  const reference = buildMotionSignature(referenceFrames);
  return observed && reference ? viewInvariantDistance(observed, reference) : null;
}
