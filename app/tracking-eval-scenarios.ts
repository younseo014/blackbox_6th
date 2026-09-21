import type { CameraObservation } from "./global-person-track";
import type { CameraSlot } from "./multi-camera";
import type { EvalCondition, EvalScenario, EvalTick } from "./tracking-eval";
import type { NormalizedLandmarkLike } from "./motion-analysis";

/**
 * Synthetic pre-validation scenarios.
 *
 * These are *not* a substitute for real footage. They are a deterministic
 * regression floor: they encode the failure modes the acceptance criteria name
 * (empty room, decoy furniture, headless and partial bodies, camera handoff)
 * so a refactor that reintroduces one fails in CI rather than in the study.
 *
 * Real scenarios - public datasets and developer-recorded clips - are loaded
 * from `data/eval-scenarios/*.json` in the same shape and score identically.
 * Study participant footage is never used.
 */

const TICK_MS = 100;

type PersonOptions = {
  centerX?: number;
  scale?: number;
  hidden?: number[];
  /** Back view: shoulder x ordering flips. */
  back?: boolean;
  /** Side view: the far-side limbs drop out and the torso narrows. */
  side?: boolean;
};

const BODY: Array<[number, number, number]> = [
  // index, x offset (torso widths), y offset (torso lengths)
  [11, 0.5, -0.5], [12, -0.5, -0.5],
  [13, 0.7, 0.1], [14, -0.7, 0.1],
  [15, 0.8, 0.65], [16, -0.8, 0.65],
  [23, 0.35, 0.5], [24, -0.35, 0.5],
  [25, 0.35, 1.35], [26, -0.35, 1.35],
  [27, 0.35, 2.2], [28, -0.35, 2.2],
];

export function syntheticPerson(options: PersonOptions = {}): NormalizedLandmarkLike[] {
  const { centerX = 0.5, scale = 0.2, hidden = [], back = false, side = false } = options;
  const points: NormalizedLandmarkLike[] = Array.from({ length: 33 }, () => ({
    x: centerX, y: 0.5, z: 0, visibility: 0,
  }));
  const mirror = back ? -1 : 1;
  const narrow = side ? 0.25 : 1;
  for (const [index, xOffset, yOffset] of BODY) {
    points[index] = {
      x: centerX + xOffset * scale * mirror * narrow,
      y: 0.45 + yOffset * scale,
      z: 0,
      visibility: hidden.includes(index) ? 0 : 0.9,
    };
  }
  // A side view loses the far arm and leg entirely.
  if (side) for (const index of [12, 14, 16, 24, 26, 28]) {
    points[index] = { ...points[index], visibility: 0.05 };
  }
  return points;
}

/** A chair-like decoy: two near-collinear "limbs" with no valid torso symmetry. */
export function syntheticDecoy(centerX = 0.4): NormalizedLandmarkLike[] {
  const points = syntheticPerson({ centerX, scale: 0.12 });
  // One arm collapses to a point - the asymmetry a real body cannot show.
  points[13] = { x: centerX + 0.061, y: 0.4, z: 0, visibility: 0.55 };
  points[15] = { x: centerX + 0.062, y: 0.402, z: 0, visibility: 0.5 };
  return points;
}

function observation(
  cameraSlot: CameraSlot,
  timestamp: number,
  landmarks: NormalizedLandmarkLike[] | null,
  extra: Partial<CameraObservation> = {},
): CameraObservation {
  return {
    cameraSlot,
    timestamp,
    landmarks,
    personConfidence: landmarks ? 0.75 : 0,
    foregroundRatio: landmarks ? 0.08 : 0.001,
    confirmed: Boolean(landmarks),
    ...extra,
  };
}

function truthFor(camera: CameraSlot, landmarks: NormalizedLandmarkLike[]) {
  const joints: Record<number, { x: number; y: number }> = {};
  for (const [index] of BODY) {
    if ((landmarks[index].visibility ?? 0) >= 0.25) {
      joints[index] = { x: landmarks[index].x, y: landmarks[index].y };
    }
  }
  return { camera, joints };
}

function steady(
  name: string,
  description: string,
  condition: EvalCondition,
  build: (tickIndex: number) => NormalizedLandmarkLike[] | null,
  tickCount = 60,
  cameras: CameraSlot[] = [1],
): EvalScenario {
  const ticks: EvalTick[] = [];
  for (let index = 0; index < tickCount; index += 1) {
    const timestamp = index * TICK_MS;
    const landmarks = build(index);
    ticks.push({
      timestamp,
      personPresent: landmarks !== null,
      condition,
      truth: landmarks ? truthFor(cameras[0], landmarks) : undefined,
      observations: cameras.map((slot) =>
        observation(slot, timestamp, slot === cameras[0] ? landmarks : null),
      ),
    });
  }
  return { name, description, source: "synthetic:developer-pre-validation", ticks };
}

export function builtInScenarios(): EvalScenario[] {
  const scenarios: EvalScenario[] = [];

  scenarios.push(steady(
    "empty-room",
    "No person, no foreground motion, for 30 simulated seconds.",
    "empty",
    () => null,
    300,
  ));

  scenarios.push({
    name: "decoy-furniture",
    description: "A chair-shaped pose hallucination in an otherwise empty room.",
    source: "synthetic:developer-pre-validation",
    ticks: Array.from({ length: 200 }, (_, index) => {
      const timestamp = index * TICK_MS;
      return {
        timestamp,
        personPresent: false,
        condition: "decoy_object" as EvalCondition,
        cameraSubjects: { 1: "object" as const },
        observations: [observation(1, timestamp, syntheticDecoy(), {
          personConfidence: 0,
          foregroundRatio: 0.002,
          confirmed: false,
        })],
      };
    }),
  });

  scenarios.push({
    name: "entry-latency",
    description: "Empty room for 2s, then a person walks in and stays.",
    source: "synthetic:developer-pre-validation",
    ticks: Array.from({ length: 120 }, (_, index) => {
      const timestamp = index * TICK_MS;
      const present = index >= 20;
      const landmarks = present ? syntheticPerson({ centerX: 0.5 }) : null;
      return {
        timestamp,
        personPresent: present,
        condition: present ? ("front" as EvalCondition) : ("empty" as EvalCondition),
        truth: landmarks ? truthFor(1, landmarks) : undefined,
        observations: [observation(1, timestamp, landmarks)],
      };
    }),
  });

  scenarios.push(steady(
    "side-view",
    "Person seen from the side; the far arm and leg are not visible.",
    "side",
    () => syntheticPerson({ side: true }),
  ));

  scenarios.push(steady(
    "back-view",
    "Person facing away from the camera.",
    "back",
    () => syntheticPerson({ back: true }),
  ));

  scenarios.push(steady(
    "headless",
    "CCTV-height camera: the head is above the frame, body fully visible.",
    "head_occluded",
    () => syntheticPerson(),
  ));

  scenarios.push(steady(
    "upper-body-only",
    "Desk-height camera: only shoulders, arms and one hip are in frame.",
    "partial_body",
    () => syntheticPerson({ hidden: [25, 26, 27, 28, 24] }),
  ));

  scenarios.push(steady(
    "distant-small",
    "Person far from the camera, torso roughly 6% of frame height.",
    "distant",
    () => syntheticPerson({ scale: 0.06 }),
  ));

  scenarios.push({
    name: "camera-handoff",
    description: "Person walks out of camera 1 and into camera 2, then camera 3.",
    source: "synthetic:developer-pre-validation",
    ticks: Array.from({ length: 150 }, (_, index) => {
      const timestamp = index * TICK_MS;
      const slot: CameraSlot = index < 50 ? 1 : index < 100 ? 2 : 3;
      const landmarks = syntheticPerson({ centerX: 0.5 });
      return {
        timestamp,
        personPresent: true,
        condition: "handoff" as EvalCondition,
        truth: truthFor(slot, landmarks),
        observations: ([1, 2, 3] as CameraSlot[]).map((candidate) =>
          observation(candidate, timestamp, candidate === slot ? landmarks : null),
        ),
      };
    }),
  });

  scenarios.push({
    name: "three-camera-overlap",
    description: "All three cameras see the same person; exactly one id must exist.",
    source: "synthetic:developer-pre-validation",
    ticks: Array.from({ length: 100 }, (_, index) => {
      const timestamp = index * TICK_MS;
      const primary = syntheticPerson({ centerX: 0.5 });
      return {
        timestamp,
        personPresent: true,
        condition: "front" as EvalCondition,
        truth: truthFor(1, primary),
        observations: [
          observation(1, timestamp, primary),
          observation(2, timestamp, syntheticPerson({ centerX: 0.35, side: true })),
          observation(3, timestamp, syntheticPerson({ centerX: 0.62, back: true })),
        ],
      };
    }),
  });

  return scenarios;
}
