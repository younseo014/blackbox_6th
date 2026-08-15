import { generateEventMotion, type DemoMotionType } from "./demo-motion";
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

function stretchFrames(frames: number[][], durationSeconds: number) {
  const lastTime = frames[frames.length - 1]?.[0] ?? 1;
  const scale = lastTime > 0 ? (durationSeconds * 1000) / lastTime : 1;
  return frames.map((frame) => {
    const next = [...frame];
    next[0] = frame[0] * scale;
    return next;
  });
}

function featuresForTask(task: TaskTemplate, frames: number[][], durationSeconds: number): ObservationFeatures {
  const zoneId = task.zones[0] ?? null;
  const grid = Array(9).fill(zoneId) as ZoneGrid;
  const features = extractObservationFeatures(stretchFrames(frames, durationSeconds), grid);
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

export function getSyntheticTrainingClips(occupation: OccupationId): SyntheticTrainingClip[] {
  const template = getOccupationTemplate(occupation);
  return template.tasks
    .filter((task) => task.id !== "REST")
    .map((task, index) => {
      const motionType = motionTypeForTask(task);
      const frames = generateEventMotion(motionType, stableHash(`${occupation}-${task.id}`) % 97);
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
        motionLabel: task.motions.slice(0, 3).join(" · "),
        instruction: instructionForTask(task),
        expectedMinSeconds: Math.round(baseDuration * VARIATIONS[0] * 10) / 10,
        expectedMaxSeconds: Math.round(baseDuration * VARIATIONS[VARIATIONS.length - 1] * 10) / 10,
        frames: stretchFrames(frames, Math.min(14, Math.max(7, baseDuration * 0.45 + index * 0.1))),
      };
    });
}

export function createSyntheticTrainingDataset(
  occupation: OccupationId,
  baselineVersion: number,
  generatedAt = Date.now(),
): SyntheticTrainingDataset {
  const template = getOccupationTemplate(occupation);
  const tasks = template.tasks.filter((task) => task.id !== "REST");
  const clips = getSyntheticTrainingClips(occupation);
  const datasetId = `synthetic-${occupation}-v${baselineVersion}-${generatedAt}`;
  const episodes: ObservationEpisode[] = [];
  let globalIndex = 0;

  tasks.forEach((task, taskIndex) => {
    const motionType = motionTypeForTask(task);
    const baseFrames = generateEventMotion(motionType, (stableHash(task.id) + taskIndex) % 101);
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
