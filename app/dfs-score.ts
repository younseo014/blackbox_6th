// Pure calculation helpers for the "평소 흐름 일치도" (Daily Flow Similarity)
// scoring design in memory-guard-quantitative-scoring-spec.md, sections 4-7 and 11.
// Kept separate from care-metrics.ts per spec section 14 item 4: run in parallel,
// don't replace the existing care report metrics yet.
//
// Internal names (never shown to the user): ECE, DCB, DFS, WFS.
// User-facing name for DFS/WFS: "평소 흐름 일치도".
//
// Explicitly out of scope here (spec sections 12-13, section 14 items 6-7):
// view-signature/VCS and multi-camera CCAC tracklet linking - both require
// multi-camera capture, which this prototype doesn't have.

import type { ObservationEpisode, TaskBaseline } from "./observation-engine";

export type SampleGrade = "insufficient" | "provisional" | "eligible" | "stable";

/** Spec section 11.2: n<8 insufficient, 8-19 provisional, 20-29 eligible, 30+ stable. */
export function sampleGrade(sampleCount: number): SampleGrade {
  if (sampleCount < 8) return "insufficient";
  if (sampleCount < 20) return "provisional";
  if (sampleCount < 30) return "eligible";
  return "stable";
}

const ECE_Z_THRESHOLD = 1.5;
const ECE_Z_SPAN = 1.5;
const ECE_DURATION_WEIGHT = 0.6;
const ECE_PAUSE_WEIGHT = 0.4;
const TASK_CONFIDENCE_MIN = 0.55;
const TASK_DAILY_CAP = 3.0;
const DAY_DCB_CAP = 9.0;
const DUPLICATE_WINDOW_MS = 2 * 60 * 60 * 1000;

/** Spec section 4.1: normalized excess over the 1.5 SD alert threshold, capped at 1. */
function excessNormalized(z: number): number {
  return Math.min(1, Math.max(0, (Math.abs(z) - ECE_Z_THRESHOLD) / ECE_Z_SPAN));
}

/** Spec section 4.3: context weight by contextWeight bucket (0 rest / 1 general / 2 transit / 3 stalled-task). */
function contextMultiplier(contextWeight: ObservationEpisode["contextWeight"]): number {
  return contextWeight;
}

/**
 * Spec sections 3.2 and 4: Event Change Evidence for one episode, or null when
 * the episode doesn't qualify for scoring (spec section 3.2 eligibility gate).
 */
export function computeEventChangeEvidence(episode: ObservationEpisode): number | null {
  if (episode.taskConfidence < TASK_CONFIDENCE_MIN) return null;
  if (episode.contextWeight === 0) return null;
  const durationZ = episode.durationZScore;
  const pauseZ = episode.pauseZScore;
  if (durationZ === null && pauseZ === null) return null;

  const durationE = durationZ !== null ? excessNormalized(durationZ) : null;
  const pauseE = pauseZ !== null ? excessNormalized(pauseZ) : null;
  const combined =
    durationE !== null && pauseE !== null
      ? ECE_DURATION_WEIGHT * durationE + ECE_PAUSE_WEIGHT * pauseE
      : (durationE ?? pauseE)!;

  return contextMultiplier(episode.contextWeight) * combined;
}

export type DailyChangeBurden = {
  date: string;
  dcb: number;
  contributingEpisodeIds: string[];
};

/**
 * Spec section 5: per-day burden. Duplicate suppression keeps only the
 * highest-ECE episode per taskType within any 2-hour window, caps a single
 * task's daily contribution at 3.0, and caps the day at 9.0.
 */
export function computeDailyChangeBurden(episodes: ObservationEpisode[]): DailyChangeBurden[] {
  const byDate = new Map<string, ObservationEpisode[]>();
  for (const episode of episodes) {
    if (!byDate.has(episode.date)) byDate.set(episode.date, []);
    byDate.get(episode.date)!.push(episode);
  }

  const results: DailyChangeBurden[] = [];
  for (const [date, dayEpisodes] of byDate) {
    const scored = dayEpisodes
      .map((episode) => ({ episode, ece: computeEventChangeEvidence(episode) }))
      .filter((entry): entry is { episode: ObservationEpisode; ece: number } => entry.ece !== null)
      .sort((a, b) => a.episode.recordedAt - b.episode.recordedAt);

    const keptByTask = new Map<string, { recordedAt: number; ece: number; episodeId: string }[]>();

    for (const entry of scored) {
      const kept = keptByTask.get(entry.episode.taskType) ?? [];
      const withinWindow = kept.find(
        (k) => Math.abs(k.recordedAt - entry.episode.recordedAt) < DUPLICATE_WINDOW_MS,
      );
      if (withinWindow) {
        if (entry.ece <= withinWindow.ece) continue;
        withinWindow.ece = entry.ece;
        withinWindow.recordedAt = entry.episode.recordedAt;
        withinWindow.episodeId = entry.episode.id;
      } else {
        kept.push({ recordedAt: entry.episode.recordedAt, ece: entry.ece, episodeId: entry.episode.id });
        keptByTask.set(entry.episode.taskType, kept);
      }
    }

    let dcb = 0;
    const contributingEpisodeIds: string[] = [];
    for (const [, windows] of keptByTask) {
      const taskTotal = Math.min(TASK_DAILY_CAP, windows.reduce((sum, w) => sum + w.ece, 0));
      dcb += taskTotal;
      windows.forEach((w) => contributingEpisodeIds.push(w.episodeId));
    }

    results.push({ date, dcb: Math.min(DAY_DCB_CAP, dcb), contributingEpisodeIds });
  }

  return results.sort((a, b) => a.date.localeCompare(b.date));
}

/** Spec section 6: 0-100 user-facing score, floored at 40. */
export function computeDailyFlowSimilarity(dcb: number): number {
  return Math.max(40, Math.round(100 * Math.exp(-0.1 * dcb)));
}

export function dailyFlowSimilarityLabel(dfs: number): string {
  if (dfs >= 85) return "평소와 비슷";
  if (dfs >= 70) return "조금 더 살펴보기";
  return "변화 관찰";
}

/**
 * Spec section 7.1: weekly average, or null when fewer than 3 eligible days
 * ("기록을 더 모으는 중").
 */
export function computeWeeklyFlowSimilarity(dailyScores: number[]): number | null {
  if (dailyScores.length < 3) return null;
  return Math.round(dailyScores.reduce((sum, v) => sum + v, 0) / dailyScores.length);
}

export type WeeklyAlertLevel = "reference" | "watch" | "notable" | "none";

/** Spec section 7.2: weekly alert trigger over the last 7 eligible days' DCB values. */
export function weeklyAlertLevel(recentDcb: number[]): WeeklyAlertLevel {
  const daysAtOrAbove3 = recentDcb.filter((v) => v >= 3).length;
  const daysAtOrAbove4 = recentDcb.filter((v) => v >= 4).length;
  const wfs = computeWeeklyFlowSimilarity(recentDcb.map(computeDailyFlowSimilarity));
  if (daysAtOrAbove4 >= 3 || (wfs !== null && wfs < 70)) return "notable";
  if (daysAtOrAbove3 >= 2) return "watch";
  if (daysAtOrAbove3 >= 1) return "reference";
  return "none";
}

/** Spec section 3.1: whether a TaskBaseline entry may be used for scoring at all (n>=8, SD>=0.05). */
export function isTaskBaselineScorable(task: TaskBaseline): boolean {
  return task.sampleCount >= 8 && task.durationSD >= 0.05 && task.pauseSD >= 0.05;
}
