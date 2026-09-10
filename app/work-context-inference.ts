import type { ActionReviewCandidate } from "./action-review";

export type WorkContextScheduleLike = {
  id: string;
  time: string;
  label: string;
};

export type WorkContextRoutineLike = {
  days: number[];
  schedule: WorkContextScheduleLike[];
};

export type WorkContextLike = {
  routines: WorkContextRoutineLike[];
  closedDays: number[];
};

export type MotionLabelCandidate = {
  label: string;
  confidence: number;
};

function normalizeLabel(label: string) {
  return label.trim().replace(/\s+/g, " ").toLocaleLowerCase();
}

export function workContextTaskType(label: string) {
  return `WORK_CONTEXT:${normalizeLabel(label)}`;
}

export function listWorkContextLabels(config: WorkContextLike) {
  const labels = new Map<string, { taskType: string; taskLabel: string }>();
  config.routines.flatMap((routine) => routine.schedule).forEach((item) => {
    const taskLabel = item.label.trim();
    if (!taskLabel) return;
    const taskType = workContextTaskType(taskLabel);
    if (!labels.has(taskType)) labels.set(taskType, { taskType, taskLabel });
  });
  return [...labels.values()];
}

function minuteOfDay(date: Date) {
  return date.getHours() * 60 + date.getMinutes();
}

function minutesFromClock(value: string) {
  const [hour, minute] = value.split(":").map(Number);
  return Number.isFinite(hour) && Number.isFinite(minute) ? hour * 60 + minute : null;
}

/**
 * Builds weak-label candidates. Registered time is deliberately a prior, not
 * an answer: it can narrow the queue but cannot teach a skeleton by itself.
 */
export function buildWorkContextCandidates(args: {
  config: WorkContextLike;
  recordedAt: number;
  motionCandidates?: MotionLabelCandidate[];
}) {
  const recordedDate = new Date(args.recordedAt);
  const weekday = recordedDate.getDay();
  const currentMinutes = minuteOfDay(recordedDate);
  const candidates: ActionReviewCandidate[] = [];
  const isClosed = args.config.closedDays.includes(weekday);

  args.config.routines.forEach((routine) => {
    const isTodaysRoutine = !isClosed && routine.days.includes(weekday);
    routine.schedule.forEach((item) => {
      const scheduledMinutes = minutesFromClock(item.time);
      const distance = scheduledMinutes === null ? Number.POSITIVE_INFINITY : Math.abs(currentMinutes - scheduledMinutes);
      const confidence = isTodaysRoutine
        ? distance <= 60 ? 0.48 : distance <= 180 ? 0.4 : 0.32
        : 0.22;
      candidates.push({
        taskType: workContextTaskType(item.label),
        taskLabel: item.label.trim(),
        confidence,
      });
    });
  });

  args.motionCandidates?.forEach((candidate) => {
    const matchingLabel = listWorkContextLabels(args.config).find(
      (item) => normalizeLabel(item.taskLabel) === normalizeLabel(candidate.label),
    );
    if (matchingLabel) candidates.push({ ...matchingLabel, confidence: candidate.confidence });
  });

  return candidates;
}
