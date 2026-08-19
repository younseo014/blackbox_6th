import { parseSessionFrame } from "./pose-store";
import type { PrimitiveMotionLabel } from "./occupation-templates";
import type { SyntheticTrainingClip } from "./synthetic-training";

export type MotionClassificationCandidate = {
  taskType: string;
  taskLabel: string;
  confidence: number;
  distance: number;
  primitiveLabels: PrimitiveMotionLabel[];
};

export type MotionClassification = {
  status: "matched" | "uncertain" | "insufficient";
  predictedTaskType: string | null;
  predictedTaskLabel: string | null;
  confidence: number;
  candidates: MotionClassificationCandidate[];
  primitiveLabels: PrimitiveMotionLabel[];
  evidence: string[];
};

const SAMPLE_COUNT = 28;
const DISPLAY_ASPECT_RATIO = 16 / 9;
const BODY_JOINTS = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17];
const FINGER_TIPS = [4, 8, 12, 16, 20];
const MATCH_CONFIDENCE = 0.52;

type SignatureFrame = {
  values: number[];
  mask: number[];
};

type MotionSignature = {
  frames: SignatureFrame[];
  usableFrames: number;
  handCoverage: number;
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

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
  const leftHip = point(parsed.body, 12);
  const rightHip = point(parsed.body, 13);
  if ([leftShoulder, rightShoulder, leftHip, rightHip].some((item) => item.visibility < 0.25)) {
    return null;
  }
  const shoulderCenter = midpoint(leftShoulder, rightShoulder);
  const hipCenter = midpoint(leftHip, rightHip);
  const torso = Math.max(0.08, Math.hypot(shoulderCenter.x - hipCenter.x, shoulderCenter.y - hipCenter.y));
  const values: number[] = [];
  const mask: number[] = [];
  BODY_JOINTS.forEach((index) => {
    const joint = point(parsed.body, index);
    const visible = joint.visibility >= 0.25 ? 1 : 0;
    values.push((joint.x - hipCenter.x) / torso, (joint.y - hipCenter.y) / torso);
    mask.push(visible, visible);
  });
  values.push(parsed.head?.yaw ?? 0, parsed.head?.pitch ?? 0, parsed.head?.roll ?? 0);
  mask.push(parsed.head ? 0.35 : 0, parsed.head ? 0.35 : 0, parsed.head ? 0.25 : 0);
  const leftHand = appendHandShape(values, mask, parsed.leftHand, torso);
  const rightHand = appendHandShape(values, mask, parsed.rightHand, torso);
  return {
    frame: { values, mask },
    center: hipCenter,
    scale: torso,
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
 * The body is expressed relative to the hips and torso length, while the hip
 * travel from the first frame is retained so walking/zone-transition motions
 * do not collapse into a stationary pose. Finger tips are compared relative
 * to their wrist and simply masked out when the real camera did not see them.
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
  return a.frames.reduce((sum, frame, index) => sum + frameDistance(frame, b.frames[index]), 0) / a.frames.length;
}

function confidenceFor(distance: number, runnerUpDistance: number) {
  const shapeScore = Math.exp(-distance * 4.2);
  const margin = runnerUpDistance > 0
    ? clamp((runnerUpDistance - distance) / runnerUpDistance, 0, 1)
    : 1;
  return clamp(0.18 + shapeScore * 0.58 + margin * 0.24, 0.05, 0.99);
}

export function classifySkeletonMotion(
  rawFrames: number[][],
  clips: SyntheticTrainingClip[],
): MotionClassification {
  const observed = buildMotionSignature(rawFrames);
  if (!observed || clips.length === 0) {
    return {
      status: "insufficient",
      predictedTaskType: null,
      predictedTaskLabel: null,
      confidence: 0,
      candidates: [],
      primitiveLabels: [],
      evidence: ["전신 좌표가 충분하지 않아 동작을 확정하지 못했어요."],
    };
  }

  const ranked = clips
    .map((clip) => {
      const reference = buildMotionSignature(clip.frames);
      return {
        clip,
        distance: reference ? signatureDistance(observed, reference) : 10,
      };
    })
    .sort((a, b) => a.distance - b.distance);
  const best = ranked[0];
  const runnerUpDistance = ranked[1]?.distance ?? best.distance * 2;
  const confidence = confidenceFor(best.distance, runnerUpDistance);
  const candidates = ranked.slice(0, 3).map((item, index) => ({
    taskType: item.clip.taskType,
    taskLabel: item.clip.taskLabel,
    confidence: index === 0
      ? confidence
      : clamp(confidence * Math.exp(-(item.distance - best.distance) * 5.5), 0.02, confidence - 0.01),
    distance: item.distance,
    primitiveLabels: item.clip.primitiveLabels,
  }));
  const matched = confidence >= MATCH_CONFIDENCE;
  return {
    status: matched ? "matched" : "uncertain",
    predictedTaskType: best.clip.taskType,
    predictedTaskLabel: best.clip.taskLabel,
    confidence,
    candidates,
    primitiveLabels: best.clip.primitiveLabels,
    evidence: [
      `${observed.usableFrames}개 전신 좌표 프레임의 관절 궤적을 비교했어요.`,
      observed.handCoverage >= 0.5
        ? "손가락 좌표가 충분해 손 모양 변화도 함께 비교했어요."
        : "손가락 인식이 적어 전신·팔 궤적을 중심으로 비교했어요.",
      matched
        ? `${best.clip.taskLabel} 예시와 움직임의 방향·범위·순서가 가장 가까웠어요.`
        : "후보 간 차이가 작아 동작을 확정하지 않고 가능성이 높은 순서로 보여드려요.",
    ],
  };
}
