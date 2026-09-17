import postureModelJson from "../models/epfl-posture-v1.json";
import { parseSessionFrame } from "./pose-store";

const SAMPLE_COUNT = 28;
const DISPLAY_ASPECT_RATIO = 16 / 9;
const BODY_JOINTS = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17] as const;
const MINIMUM_USABLE_FRAMES = 6;

export type EpflPostureLabel = "STAND" | "WALK" | "BEND_DOWN";

export type EpflPostureResult = {
  status: "matched" | "uncertain" | "insufficient";
  label: EpflPostureLabel | null;
  displayLabel: string | null;
  confidence: number;
  scores: Array<{ label: EpflPostureLabel; displayLabel: string; confidence: number }>;
  model: "epfl-posture-v1";
};

type Model = {
  classes: EpflPostureLabel[];
  normalization: { mean: number[]; std: number[] };
  parameters: {
    inputHidden: number[][];
    hiddenBias: number[];
    hiddenOutput: number[][];
    outputBias: number[];
  };
};

type PoseFrame = { points: number[]; center: [number, number]; span: number };

const model = postureModelJson as unknown as Model;
const DISPLAY_LABELS: Record<EpflPostureLabel, string> = {
  STAND: "서 있기",
  WALK: "걷기",
  BEND_DOWN: "숙이기",
};

function interpolate(a: number, b: number, amount: number) {
  return a + (b - a) * amount;
}

function resample(frames: PoseFrame[]) {
  if (frames.length === 1) return Array.from({ length: SAMPLE_COUNT }, () => frames[0]);
  return Array.from({ length: SAMPLE_COUNT }, (_, index) => {
    const position = (index / (SAMPLE_COUNT - 1)) * (frames.length - 1);
    const before = Math.floor(position);
    const after = Math.min(frames.length - 1, before + 1);
    const amount = position - before;
    return {
      points: frames[before].points.map((value, pointIndex) =>
        interpolate(value, frames[after].points[pointIndex], amount)),
      center: [
        interpolate(frames[before].center[0], frames[after].center[0], amount),
        interpolate(frames[before].center[1], frames[after].center[1], amount),
      ] as [number, number],
      span: interpolate(frames[before].span, frames[after].span, amount),
    };
  });
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function columnStatistic(rows: number[][], statistic: (values: number[]) => number) {
  return rows[0].map((_, column) => statistic(rows.map((row) => row[column])));
}

function frameFromRaw(rawFrame: number[]): PoseFrame | null {
  const parsed = parseSessionFrame(rawFrame);
  if (!parsed.bodyDetected || !parsed.fullBodyVisible) return null;
  const joints = BODY_JOINTS.map((joint) => ({
    x: parsed.body[joint * 4] * DISPLAY_ASPECT_RATIO,
    // EPFL's vertical axis grows upward; MediaPipe image y grows downward.
    y: -parsed.body[joint * 4 + 1],
    visibility: parsed.body[joint * 4 + 3] ?? 0,
  }));
  if (joints.some((joint) => joint.visibility < 0.25 || !Number.isFinite(joint.x) || !Number.isFinite(joint.y))) {
    return null;
  }
  const center: [number, number] = [
    (joints[0].x + joints[1].x) / 2,
    (joints[0].y + joints[1].y) / 2,
  ];
  const span = Math.hypot(joints[0].x - joints[1].x, joints[0].y - joints[1].y);
  if (!Number.isFinite(span) || span <= 1e-6) return null;
  return { points: joints.flatMap((joint) => [joint.x, joint.y]), center, span };
}

/** Reproduces the 298-value feature layout used by train-epfl-posture.py. */
export function buildEpflPostureFeatures(rawFrames: number[][]): number[] | null {
  const usable = rawFrames.map(frameFromRaw).filter((frame): frame is PoseFrame => frame !== null);
  if (usable.length < MINIMUM_USABLE_FRAMES) return null;
  const frames = resample(usable);
  const referenceScale = Math.max(median(frames.map((frame) => frame.span)) * 0.2, 1e-6);
  const scales = frames.map((frame) => Math.max(referenceScale, frame.span / 0.64));
  const pose = frames.map((frame, frameIndex) => frame.points.map((value, index) =>
    (value - frame.center[index % 2]) / scales[frameIndex]));
  const motion = pose.map((row) => row.map((value, index) => value - pose[0][index]));
  const velocity = motion.slice(1).map((row, index) => row.map((value, column) => value - motion[index][column]));
  const binSize = SAMPLE_COUNT / 4;
  const poseBins = Array.from({ length: 4 }, (_, bin) =>
    columnStatistic(pose.slice(bin * binSize, (bin + 1) * binSize), (values) => values.reduce((sum, value) => sum + value, 0) / values.length));
  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const poseFeatures = [
    ...columnStatistic(pose, mean),
    ...columnStatistic(pose, (values) => {
      const average = mean(values);
      return Math.sqrt(mean(values.map((value) => (value - average) ** 2)));
    }),
    ...columnStatistic(pose, (values) => Math.min(...values)),
    ...columnStatistic(pose, (values) => Math.max(...values)),
    ...motion[motion.length - 1],
    ...columnStatistic(motion, (values) => Math.max(...values) - Math.min(...values)),
    ...columnStatistic(velocity, (values) => mean(values.map(Math.abs))),
    ...columnStatistic(velocity, (values) => Math.max(...values.map(Math.abs))),
    ...poseBins.flat(),
  ];
  const travel = frames.map((frame) => [
    (frame.center[0] - frames[0].center[0]) / scales[0],
    (frame.center[1] - frames[0].center[1]) / scales[0],
  ]);
  const travelVelocity = travel.slice(1).map((row, index) => row.map((value, column) => value - travel[index][column]));
  const travelFeatures = [
    ...travel[travel.length - 1],
    ...columnStatistic(travel, (values) => Math.max(...values) - Math.min(...values)),
    ...columnStatistic(travelVelocity, (values) => values.reduce((sum, value) => sum + Math.abs(value), 0)),
    ...columnStatistic(travelVelocity, (values) => mean(values.map(Math.abs))),
    ...columnStatistic(travelVelocity, (values) => Math.max(...values.map(Math.abs))),
  ];
  const features = [...poseFeatures, ...travelFeatures];
  return features.length === 298 && features.every(Number.isFinite) ? features : null;
}

export function classifyEpflPosture(rawFrames: number[][]): EpflPostureResult {
  const features = buildEpflPostureFeatures(rawFrames);
  if (!features) {
    return { status: "insufficient", label: null, displayLabel: null, confidence: 0, scores: [], model: "epfl-posture-v1" };
  }
  const normalized = features.map((value, index) =>
    Math.max(-8, Math.min(8, (value - model.normalization.mean[index]) / model.normalization.std[index])));
  const hidden = model.parameters.hiddenBias.map((bias, hiddenIndex) => Math.max(0,
    bias + normalized.reduce((sum, value, inputIndex) => sum + value * model.parameters.inputHidden[inputIndex][hiddenIndex], 0)));
  const logits = model.parameters.outputBias.map((bias, outputIndex) =>
    bias + hidden.reduce((sum, value, hiddenIndex) => sum + value * model.parameters.hiddenOutput[hiddenIndex][outputIndex], 0));
  const maximum = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  const scores = model.classes.map((label, index) => ({
    label,
    displayLabel: DISPLAY_LABELS[label],
    confidence: exponentials[index] / total,
  })).sort((a, b) => b.confidence - a.confidence);
  const best = scores[0];
  return {
    status: best.confidence >= 0.6 ? "matched" : "uncertain",
    label: best.label,
    displayLabel: best.displayLabel,
    confidence: best.confidence,
    scores,
    model: "epfl-posture-v1",
  };
}
