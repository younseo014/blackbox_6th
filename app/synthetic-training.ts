import { generateEventMotion, type DemoMotionType } from "./demo-motion";
import {
  BODY_LANDMARK_COUNT,
  HAND_LANDMARK_COUNT,
  type BodyProportionProfile,
} from "./pose-store";
import {
  extractObservationFeatures,
  type ObservationEpisode,
  type ObservationFeatures,
  type ZoneGrid,
} from "./observation-engine";
import {
  getOccupationTemplate,
  type OccupationId,
  type PrimitiveMotionLabel,
  type TaskTemplate,
  type WorkPhase,
} from "./occupation-templates";

export type SyntheticTrainingClip = {
  id: string;
  occupation: OccupationId;
  taskType: string;
  taskLabel: string;
  phase: WorkPhase;
  phaseLabel: string;
  zoneId: string;
  zoneLabel: string;
  primitiveLabels: PrimitiveMotionLabel[];
  motionType: DemoMotionType;
  motionLabel: string;
  instruction: string;
  expectedMinSeconds: number;
  expectedMaxSeconds: number;
  frames: number[][];
};

export type SyntheticTrainingDataset = {
  id: string;
  occupation: OccupationId;
  baselineVersion: number;
  generatedAt: number;
  zoneGrid: ZoneGrid;
  episodes: ObservationEpisode[];
  clips: SyntheticTrainingClip[];
};

const DAY_MS = 24 * 60 * 60 * 1000;
const VARIATIONS = [0.92, 0.97, 1.04, 1.09];
const BODY_START = 10;
const LEFT_HAND_START = BODY_START + BODY_LANDMARK_COUNT * 4;
const RIGHT_HAND_START = LEFT_HAND_START + HAND_LANDMARK_COUNT * 3;
const DISPLAY_ASPECT_RATIO = 16 / 9;
export const SYNTHETIC_REPLAY_FPS = 30;
const OBSERVED_TEST_MOTION_SECONDS = 5.2;

export const DEFAULT_BODY_PROPORTIONS: Omit<BodyProportionProfile, "sourceSessionId" | "sampledAt" | "usableFrames"> = {
  shoulderToTorso: 0.86,
  hipToTorso: 0.58,
  upperArmToTorso: 0.74,
  forearmToTorso: 0.66,
  thighToTorso: 1.05,
  shinToTorso: 1,
  handToForearm: 0.42,
};

const PHASE_LABELS: Record<WorkPhase, string> = {
  open: "오픈",
  business: "영업 중",
  close: "마감",
  break: "휴식",
  unknown: "기타",
};

function stableHash(value: string) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash;
}

type MetricPoint = { x: number; y: number };

function bump(t: number) {
  return Math.sin(Math.PI * Math.max(0, Math.min(1, t)));
}

function addTaskChoreography(frames: number[][], task: TaskTemplate, signatureKey: string): number[][] {
  const hash = stableHash(signatureKey);
  const style = hash % 6;
  const frequency = 1 + ((hash >>> 4) % 3);
  const amplitude = 0.026 + ((hash >>> 7) % 4) * 0.008;
  const direction = (hash & 1) === 0 ? 1 : -1;

  const moveHand = (frame: number[], side: "left" | "right", dx: number, dy: number) => {
    const wrist = side === "left" ? 4 : 5;
    const elbow = side === "left" ? 2 : 3;
    const anchors = side === "left" ? [6, 8, 10] : [7, 9, 11];
    frame[BODY_START + wrist * 4] += dx;
    frame[BODY_START + wrist * 4 + 1] += dy;
    frame[BODY_START + elbow * 4] += dx * 0.52;
    frame[BODY_START + elbow * 4 + 1] += dy * 0.52;
    anchors.forEach((index) => {
      frame[BODY_START + index * 4] += dx;
      frame[BODY_START + index * 4 + 1] += dy;
    });
    const handStart = side === "left" ? LEFT_HAND_START : RIGHT_HAND_START;
    for (let index = 0; index < HAND_LANDMARK_COUNT; index += 1) {
      frame[handStart + index * 3] += dx;
      frame[handStart + index * 3 + 1] += dy;
    }
  };

  const translateBody = (frame: number[], dx: number, dy: number) => {
    for (let index = 0; index < BODY_LANDMARK_COUNT; index += 1) {
      frame[BODY_START + index * 4] += dx;
      frame[BODY_START + index * 4 + 1] += dy;
    }
    for (const handStart of [LEFT_HAND_START, RIGHT_HAND_START]) {
      for (let index = 0; index < HAND_LANDMARK_COUNT; index += 1) {
        frame[handStart + index * 3] += dx;
        frame[handStart + index * 3 + 1] += dy;
      }
    }
  };

  const animateFingers = (frame: number[], handStart: number, amount: number) => {
    const wristX = frame[handStart];
    const wristY = frame[handStart + 1];
    for (let index = 1; index < HAND_LANDMARK_COUNT; index += 1) {
      const finger = Math.floor((index - 1) / 4);
      const perFinger = 1 + amount * (finger % 2 === 0 ? 1 : -0.65);
      frame[handStart + index * 3] = wristX + (frame[handStart + index * 3] - wristX) * perFinger;
      frame[handStart + index * 3 + 1] = wristY + (frame[handStart + index * 3 + 1] - wristY) * perFinger;
    }
  };

  return frames.map((source, index) => {
    const frame = [...source];
    const t = frames.length > 1 ? index / (frames.length - 1) : 0;
    const wave = Math.sin(t * Math.PI * 2 * frequency + (hash % 17) * 0.13);
    const phase = bump(t);
    let rightDx = 0;
    let rightDy = 0;
    let leftDx = 0;
    let leftDy = 0;
    if (style === 0) {
      rightDx = wave * amplitude;
      leftDx = -wave * amplitude * 0.65;
    } else if (style === 1) {
      rightDy = wave * amplitude;
      leftDy = -wave * amplitude * 0.45;
    } else if (style === 2) {
      rightDx = wave * amplitude;
      rightDy = -wave * amplitude * 0.75;
      leftDx = wave * amplitude * 0.35;
      leftDy = wave * amplitude * 0.5;
    } else if (style === 3) {
      rightDx = Math.cos(t * Math.PI * 2 * frequency) * amplitude;
      rightDy = Math.sin(t * Math.PI * 2 * frequency) * amplitude * 0.7;
      leftDx = -rightDx * 0.55;
      leftDy = rightDy * 0.45;
    } else if (style === 4) {
      rightDx = phase * amplitude * direction;
      leftDx = -phase * amplitude * direction;
      rightDy = leftDy = -phase * amplitude * 0.45;
    } else {
      rightDx = phase * amplitude * 1.25 * direction;
      rightDy = -phase * amplitude * 0.8;
      leftDx = wave * amplitude * 0.25;
    }

    if (task.motions.includes("REACH_UP") || task.motions.includes("ARM_ELEVATED")) {
      rightDy -= phase * 0.075;
      leftDy -= phase * 0.055;
    }
    if (task.motions.includes("REACH_DOWN") || task.motions.includes("BEND_DOWN")) {
      rightDy += phase * 0.055;
    }
    if (task.motions.includes("PUSH_PULL")) {
      const push = Math.sin(t * Math.PI * 2 * (frequency + 1)) * 0.045;
      rightDx += push;
      leftDx += push;
    }
    moveHand(frame, "right", rightDx, rightDy);
    moveHand(frame, "left", leftDx, leftDy);

    if (task.motions.includes("WALK") || task.motions.includes("ZONE_TRANSITION")) {
      const travel = direction * (t - 0.5) * (task.motions.includes("ZONE_TRANSITION") ? 0.2 : 0.13);
      translateBody(frame, travel, Math.sin(t * Math.PI * 2 * frequency) * 0.006);
    }

    const fingerAmount = Math.sin(t * Math.PI * 2 * (frequency + 2)) * (task.motions.includes("REPETITIVE_ARM") ? 0.16 : 0.08);
    animateFingers(frame, RIGHT_HAND_START, fingerAmount);
    animateFingers(frame, LEFT_HAND_START, -fingerAmount * 0.7);
    frame[3] += wave * (0.08 + style * 0.015);
    frame[4] += (phase - 0.5) * ((hash % 5) * 0.015);
    return frame;
  });
}

function applyBodyProportions(
  frames: number[][],
  suppliedProfile?: BodyProportionProfile | null,
): number[][] {
  const profile = suppliedProfile ?? DEFAULT_BODY_PROPORTIONS;
  const read = (frame: number[], index: number): MetricPoint => ({
    x: frame[BODY_START + index * 4] * DISPLAY_ASPECT_RATIO,
    y: frame[BODY_START + index * 4 + 1],
  });
  const write = (frame: number[], index: number, point: MetricPoint) => {
    frame[BODY_START + index * 4] = point.x / DISPLAY_ASPECT_RATIO;
    frame[BODY_START + index * 4 + 1] = point.y;
  };
  const middle = (a: MetricPoint, b: MetricPoint): MetricPoint => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  const direction = (from: MetricPoint, to: MetricPoint, fallback: MetricPoint): MetricPoint => {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const length = Math.hypot(dx, dy);
    return length > 0.001 ? { x: dx / length, y: dy / length } : fallback;
  };
  const along = (origin: MetricPoint, unit: MetricPoint, length: number): MetricPoint => ({
    x: origin.x + unit.x * length,
    y: origin.y + unit.y * length,
  });
  const shoulderToAnkle = 0.66;
  const targetTorso = shoulderToAnkle / (1 + profile.thighToTorso + profile.shinToTorso);
  const targets = {
    shoulder: targetTorso * profile.shoulderToTorso,
    hip: targetTorso * profile.hipToTorso,
    upperArm: targetTorso * profile.upperArmToTorso,
    forearm: targetTorso * profile.forearmToTorso,
    thigh: targetTorso * profile.thighToTorso,
    shin: targetTorso * profile.shinToTorso,
  };

  return frames.map((source) => {
    const frame = [...source];
    const original = Array.from({ length: BODY_LANDMARK_COUNT }, (_, index) => read(frame, index));
    const shoulderCenter = middle(original[0], original[1]);
    const hipCenterOriginal = middle(original[12], original[13]);
    const torsoDirection = direction(shoulderCenter, hipCenterOriginal, { x: 0, y: 1 });
    const shoulderDirection = direction(original[0], original[1], { x: 1, y: 0 });
    const hipDirection = direction(original[12], original[13], { x: 1, y: 0 });
    const hipCenter = along(shoulderCenter, torsoDirection, targetTorso);
    const next = [...original];
    next[0] = along(shoulderCenter, shoulderDirection, -targets.shoulder / 2);
    next[1] = along(shoulderCenter, shoulderDirection, targets.shoulder / 2);
    next[12] = along(hipCenter, hipDirection, -targets.hip / 2);
    next[13] = along(hipCenter, hipDirection, targets.hip / 2);

    const oldWrists = { left: original[4], right: original[5] };
    for (const side of [
      { shoulder: 0, elbow: 2, wrist: 4 },
      { shoulder: 1, elbow: 3, wrist: 5 },
    ]) {
      const upperDirection = direction(original[side.shoulder], original[side.elbow], { x: 0, y: 1 });
      next[side.elbow] = along(next[side.shoulder], upperDirection, targets.upperArm);
      const forearmDirection = direction(original[side.elbow], original[side.wrist], { x: 0, y: 1 });
      next[side.wrist] = along(next[side.elbow], forearmDirection, targets.forearm);
    }
    for (const side of [
      { hip: 12, knee: 14, ankle: 16, heel: 18, foot: 20 },
      { hip: 13, knee: 15, ankle: 17, heel: 19, foot: 21 },
    ]) {
      const thighDirection = direction(original[side.hip], original[side.knee], { x: 0, y: 1 });
      next[side.knee] = along(next[side.hip], thighDirection, targets.thigh);
      const shinDirection = direction(original[side.knee], original[side.ankle], { x: 0, y: 1 });
      next[side.ankle] = along(next[side.knee], shinDirection, targets.shin);
      const footScale = targets.shin / Math.max(0.05, Math.hypot(original[side.ankle].x - original[side.knee].x, original[side.ankle].y - original[side.knee].y));
      next[side.heel] = {
        x: next[side.ankle].x + (original[side.heel].x - original[side.ankle].x) * footScale,
        y: next[side.ankle].y + (original[side.heel].y - original[side.ankle].y) * footScale,
      };
      next[side.foot] = {
        x: next[side.ankle].x + (original[side.foot].x - original[side.ankle].x) * footScale,
        y: next[side.ankle].y + (original[side.foot].y - original[side.ankle].y) * footScale,
      };
    }

    [6, 8, 10].forEach((index) => {
      next[index] = { x: original[index].x + next[4].x - oldWrists.left.x, y: original[index].y + next[4].y - oldWrists.left.y };
    });
    [7, 9, 11].forEach((index) => {
      next[index] = { x: original[index].x + next[5].x - oldWrists.right.x, y: original[index].y + next[5].y - oldWrists.right.y };
    });
    next.forEach((point, index) => write(frame, index, point));

    for (const hand of [
      { start: LEFT_HAND_START, wrist: next[4] },
      { start: RIGHT_HAND_START, wrist: next[5] },
    ]) {
      const oldWrist: MetricPoint = { x: frame[hand.start] * DISPLAY_ASPECT_RATIO, y: frame[hand.start + 1] };
      const middleTip: MetricPoint = { x: frame[hand.start + 12 * 3] * DISPLAY_ASPECT_RATIO, y: frame[hand.start + 12 * 3 + 1] };
      const currentHandLength = Math.max(0.015, Math.hypot(middleTip.x - oldWrist.x, middleTip.y - oldWrist.y));
      const scale = Math.min(1.8, Math.max(0.55, (targets.forearm * profile.handToForearm) / currentHandLength));
      for (let index = 0; index < HAND_LANDMARK_COUNT; index += 1) {
        const x = frame[hand.start + index * 3] * DISPLAY_ASPECT_RATIO;
        const y = frame[hand.start + index * 3 + 1];
        frame[hand.start + index * 3] = (hand.wrist.x + (x - oldWrist.x) * scale) / DISPLAY_ASPECT_RATIO;
        frame[hand.start + index * 3 + 1] = hand.wrist.y + (y - oldWrist.y) * scale;
      }
    }
    return frame;
  });
}

function generateTaskMotion(
  task: TaskTemplate,
  seed: number,
  profile: BodyProportionProfile | null | undefined,
  signatureKey: string,
) {
  const base = generateEventMotion(motionTypeForTask(task), seed);
  return applyBodyProportions(addTaskChoreography(base, task, signatureKey), profile);
}

function motionTypeForTask(task: TaskTemplate): DemoMotionType {
  if (task.motions.includes("REACH_UP") || task.motions.includes("ARM_ELEVATED")) return "high_reach";
  if (task.motions.includes("BEND_DOWN") || task.motions.includes("BEND_FORWARD") || task.motions.includes("REACH_DOWN")) return "low_bend";
  if (task.motions.includes("WALK") || task.motions.includes("ZONE_TRANSITION") || task.motions.includes("CARRY")) return "queue_shift";
  if (task.id.includes("POS") || task.id.includes("CASH") || task.id.includes("ADMIN") || task.id.includes("ORDER")) return "fine_hand_task";
  return "normal_task";
}

function instructionForTask(task: TaskTemplate) {
  const labels: string[] = [];
  if (task.motions.includes("WALK") || task.motions.includes("ZONE_TRANSITION")) labels.push("화면 안에서 짧게 이동해 주세요");
  if (task.motions.includes("REPETITIVE_ARM")) labels.push("팔이나 손을 일정한 리듬으로 반복해 주세요");
  if (task.motions.includes("REACH_UP")) labels.push("양팔을 높은 위치로 뻗어 주세요");
  if (task.motions.includes("REACH_FORWARD")) labels.push("앞쪽 물체를 향해 손을 뻗어 주세요");
  if (task.motions.includes("BEND_DOWN") || task.motions.includes("BEND_FORWARD")) labels.push("상체를 숙였다가 천천히 돌아와 주세요");
  if (task.motions.includes("CARRY")) labels.push("물건을 든 것처럼 양손을 유지해 주세요");
  if (task.motions.includes("PUSH_PULL")) labels.push("양손으로 밀고 당기는 동작을 해주세요");
  return labels.slice(0, 2).join(" · ") || "화면의 스켈레톤과 비슷한 속도로 동작해 주세요";
}

function baseDurationForTask(task: TaskTemplate) {
  const hash = stableHash(task.id);
  let duration = 12 + (hash % 12);
  if (task.motions.includes("ZONE_TRANSITION") || task.id.includes("CLOSING")) duration += 12;
  if (task.motions.includes("WALK") || task.motions.includes("CARRY")) duration += 5;
  if (task.phase === "open" || task.phase === "close") duration += 4;
  return duration;
}

function retimeFrames(frames: number[][], durationSeconds: number) {
  const lastTime = frames[frames.length - 1]?.[0] ?? 1;
  const scale = lastTime > 0 ? (durationSeconds * 1000) / lastTime : 1;
  return frames.map((frame) => {
    const next = [...frame];
    next[0] = frame[0] * scale;
    return next;
  });
}

function previewDurationForTask(task: TaskTemplate) {
  if (task.motions.includes("WALK") || task.motions.includes("ZONE_TRANSITION")) {
    return OBSERVED_TEST_MOTION_SECONDS + 0.8;
  }
  if (task.motions.includes("REPETITIVE_ARM")) return OBSERVED_TEST_MOTION_SECONDS;
  return OBSERVED_TEST_MOTION_SECONDS - 0.4;
}

/**
 * The camera sample used for calibration contained about five seconds of
 * actual target motion inside a much longer setup/stop session. Replay clips
 * use that human-paced duration, then interpolate coordinates to 30fps so a
 * short gesture remains readable instead of jumping between sparse poses.
 */
function smoothReplayFrames(frames: number[][], durationSeconds: number) {
  if (frames.length < 2) return retimeFrames(frames, durationSeconds);
  const outputCount = Math.max(2, Math.round(durationSeconds * SYNTHETIC_REPLAY_FPS) + 1);
  return Array.from({ length: outputCount }, (_, outputIndex) => {
    const progress = outputIndex / (outputCount - 1);
    const sourcePosition = progress * (frames.length - 1);
    const beforeIndex = Math.floor(sourcePosition);
    const afterIndex = Math.min(frames.length - 1, beforeIndex + 1);
    const amount = sourcePosition - beforeIndex;
    const before = frames[beforeIndex];
    const after = frames[afterIndex];
    return before.map((value, valueIndex) =>
      valueIndex === 0
        ? progress * durationSeconds * 1000
        : value + (after[valueIndex] - value) * amount,
    );
  });
}

function featuresForTask(task: TaskTemplate, frames: number[][], durationSeconds: number): ObservationFeatures {
  const zoneId = task.zones[0] ?? null;
  const grid = Array(9).fill(zoneId) as ZoneGrid;
  const features = extractObservationFeatures(retimeFrames(frames, durationSeconds), grid);
  return {
    ...features,
    durationSeconds,
    dominantZone: zoneId,
    zoneTransitions: task.motions.includes("ZONE_TRANSITION") ? Math.max(1, task.zones.length - 1) : features.zoneTransitions,
    routeComplexity: task.motions.includes("ZONE_TRANSITION") ? Math.max(2, task.zones.length) : features.routeComplexity,
    repetitionCount: task.motions.includes("REPETITIVE_ARM") ? Math.max(2, features.repetitionCount) : features.repetitionCount,
    longestPauseSeconds: Math.min(1.2, features.longestPauseSeconds),
  };
}

function autoZoneGrid(occupation: OccupationId): ZoneGrid {
  const template = getOccupationTemplate(occupation);
  const priorityZones = template.tasks
    .filter((task) => task.id !== "REST")
    .flatMap((task) => task.zones);
  const unique = [...new Set(priorityZones)];
  return Array.from({ length: 9 }, (_, index) => unique[index] ?? null);
}

function recordedAtForDay(generatedAt: number, dayIndex: number, phase: WorkPhase, offsetMinutes: number) {
  const date = new Date(generatedAt - (13 - dayIndex) * DAY_MS);
  const hour = phase === "open" ? 8 : phase === "close" ? 20 : phase === "break" ? 15 : 12;
  date.setHours(hour, offsetMinutes % 60, 0, 0);
  return date.getTime();
}

function localDateKey(timestamp: number) {
  const date = new Date(timestamp);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function getSyntheticTrainingClips(
  occupation: OccupationId,
  bodyProfile?: BodyProportionProfile | null,
): SyntheticTrainingClip[] {
  const template = getOccupationTemplate(occupation);
  return template.tasks
    .filter((task) => task.id !== "REST")
    .map((task, index) => {
      const motionType = motionTypeForTask(task);
      const signatureKey = `${occupation}-${task.id}`;
      const frames = generateTaskMotion(task, stableHash(signatureKey) % 97, bodyProfile, signatureKey);
      const baseDuration = baseDurationForTask(task);
      const zoneId = task.zones[0] ?? template.zones[0]?.id ?? "UNKNOWN";
      return {
        id: `synthetic-clip-${occupation}-${task.id}`,
        occupation,
        taskType: task.id,
        taskLabel: task.label,
        phase: task.phase,
        phaseLabel: PHASE_LABELS[task.phase],
        zoneId,
        zoneLabel: template.zones.find((zone) => zone.id === zoneId)?.label ?? zoneId,
        primitiveLabels: task.motions,
        motionType,
        motionLabel: `${task.label} 전용 전신·손 궤적`,
        instruction: instructionForTask(task),
        expectedMinSeconds: Math.round(baseDuration * VARIATIONS[0] * 10) / 10,
        expectedMaxSeconds: Math.round(baseDuration * VARIATIONS[VARIATIONS.length - 1] * 10) / 10,
        frames: smoothReplayFrames(frames, previewDurationForTask(task) + (index % 3) * 0.08),
      };
    });
}

export function createSyntheticTrainingDataset(
  occupation: OccupationId,
  baselineVersion: number,
  generatedAt = Date.now(),
  bodyProfile?: BodyProportionProfile | null,
): SyntheticTrainingDataset {
  const template = getOccupationTemplate(occupation);
  const tasks = template.tasks.filter((task) => task.id !== "REST");
  const clips = getSyntheticTrainingClips(occupation, bodyProfile);
  const datasetId = `synthetic-${occupation}-v${baselineVersion}-${generatedAt}`;
  const episodes: ObservationEpisode[] = [];
  let globalIndex = 0;

  tasks.forEach((task, taskIndex) => {
    const signatureKey = `${occupation}-${task.id}`;
    const baseFrames = generateTaskMotion(task, (stableHash(signatureKey) + taskIndex) % 101, bodyProfile, signatureKey);
    const baseDuration = baseDurationForTask(task);
    VARIATIONS.forEach((variation, sampleIndex) => {
      const durationSeconds = Math.round(baseDuration * variation * 10) / 10;
      const dayIndex = globalIndex % 14;
      const recordedAt = recordedAtForDay(generatedAt, dayIndex, task.phase, taskIndex * 7 + sampleIndex);
      const features = featuresForTask(task, baseFrames, durationSeconds);
      episodes.push({
        id: `episode-${datasetId}-${task.id}-${sampleIndex}`,
        sessionId: `synthetic-session-${datasetId}-${task.id}-${sampleIndex}`,
        recordedAt,
        date: localDateKey(recordedAt),
        occupation,
        mode: "learning",
        phase: task.phase,
        taskType: task.id,
        taskLabel: task.label,
        taskConfidence: 0.94,
        primitiveLabels: task.motions,
        features,
        disposition: "accepted",
        dispositionReason: "성능 테스트를 위해 만든 정상 업무 표본이에요.",
        durationZScore: null,
        pauseZScore: null,
        contextWeight: task.motions.includes("ZONE_TRANSITION") ? 2 : 1,
        baselineVersion,
        source: "synthetic_training",
        syntheticDatasetId: datasetId,
      });
      globalIndex += 1;
    });
  });

  // Guarantee all fourteen dates participate even for future small templates.
  const usedDays = new Set(episodes.map((episode) => episode.date));
  for (let dayIndex = 0; usedDays.size < 14 && dayIndex < 14; dayIndex += 1) {
    const source = episodes[dayIndex % episodes.length];
    const recordedAt = recordedAtForDay(generatedAt, dayIndex, source.phase, dayIndex * 3);
    const date = localDateKey(recordedAt);
    if (usedDays.has(date)) continue;
    episodes.push({
      ...source,
      id: `${source.id}-day-${dayIndex}`,
      sessionId: `${source.sessionId}-day-${dayIndex}`,
      recordedAt,
      date,
    });
    usedDays.add(date);
  }

  return {
    id: datasetId,
    occupation,
    baselineVersion,
    generatedAt,
    zoneGrid: autoZoneGrid(occupation),
    episodes,
    clips,
  };
}
