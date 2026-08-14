// Pure calculation helpers for the "케어 기록" (care report) feature.
// Kept dependency-free and DOM-free so they can be unit tested directly
// (see tests/care-metrics.test.mjs) and reused from both the client store
// (metrics-store.ts) and, optionally, a server route.
//
// Important scope note: this file computes signals FROM real interactions
// the user already takes in the app (closing re-checks, unfinished tasks,
// gaps between logged events). It does not diagnose anything - see
// `detectChangeSignal`, which deliberately returns a "watch" / "none" style
// signal plus the plain-language reasons behind it, never a diagnostic label.

export type BusyLevel = "quiet" | "normal" | "busy";

export type DailyLog = {
  date: string; // YYYY-MM-DD, local date
  safetyAlerts: number;
  doubleChecks: number;
  tasksStarted: number;
  tasksCompleted: number;
  microDelaySeconds: number[];
  busyLevel: BusyLevel;
};

export function emptyDailyLog(date: string): DailyLog {
  return {
    date,
    safetyAlerts: 0,
    doubleChecks: 0,
    tasksStarted: 0,
    tasksCompleted: 0,
    microDelaySeconds: [],
    busyLevel: "normal",
  };
}

export type MetricSummary = {
  safetyAlerts: number;
  doubleChecks: number;
  dropRate: number; // 0..1, share of started tasks never completed
  microDelayRate: number; // 0..1, share of delays over the slow threshold
  droppedTasks: number;
  sampleCount: number; // number of days the summary is built from
};

// Exported so callers that explain *why* something was counted (e.g. the
// demo persona replay's "감지 근거" panel) can cite the exact same number
// instead of a hardcoded copy that could drift out of sync.
export const SLOW_DELAY_SECONDS = 120;

export function summarizeLog(log: DailyLog): MetricSummary {
  const droppedTasks = Math.max(0, log.tasksStarted - log.tasksCompleted);
  const dropRate = log.tasksStarted > 0 ? droppedTasks / log.tasksStarted : 0;
  const slowDelays = log.microDelaySeconds.filter(
    (seconds) => seconds >= SLOW_DELAY_SECONDS,
  ).length;
  const microDelayRate =
    log.microDelaySeconds.length > 0
      ? slowDelays / log.microDelaySeconds.length
      : 0;
  return {
    safetyAlerts: log.safetyAlerts,
    doubleChecks: log.doubleChecks,
    dropRate,
    microDelayRate,
    droppedTasks,
    sampleCount: 1,
  };
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/**
 * A personal baseline built from the user's own quieter days only, so a busy
 * Saturday doesn't get compared against itself. Excludes "busy" days by
 * default because traffic-driven slowdowns are a known confound (see the
 * product review notes on 금/토 성수기 patterns).
 */
export function computeBaseline(
  logs: DailyLog[],
  options: { excludeBusy?: boolean; lookbackDays?: number } = {},
): (MetricSummary & { dayCount: number }) | null {
  const { excludeBusy = true, lookbackDays = 28 } = options;
  const sorted = [...logs].sort((a, b) => (a.date < b.date ? 1 : -1));
  const windowed = sorted.slice(0, lookbackDays);
  const eligible = excludeBusy
    ? windowed.filter((log) => log.busyLevel !== "busy")
    : windowed;
  if (eligible.length < 3) return null; // not enough history for a fair baseline

  const summaries = eligible.map(summarizeLog);
  return {
    safetyAlerts: average(summaries.map((s) => s.safetyAlerts)),
    doubleChecks: average(summaries.map((s) => s.doubleChecks)),
    dropRate: average(summaries.map((s) => s.dropRate)),
    microDelayRate: average(summaries.map((s) => s.microDelayRate)),
    droppedTasks: average(summaries.map((s) => s.droppedTasks)),
    sampleCount: eligible.length,
    dayCount: eligible.length,
  };
}

export type ChangeSignal = {
  level: "none" | "watch" | "notable";
  reasons: string[];
  confoundNote: string | null;
};

/**
 * Compares the most recent days against the personal baseline. Never
 * returns a diagnosis - only a soft "watch" / "notable" level plus the
 * plain-language reasons, matching the app's existing "케어, not 진단" tone.
 */
export function detectChangeSignal(
  recentLogs: DailyLog[],
  baseline: ReturnType<typeof computeBaseline>,
): ChangeSignal {
  if (!baseline || recentLogs.length === 0) {
    return {
      level: "none",
      reasons: ["아직 비교할 만큼의 평소 기록이 쌓이지 않았어요."],
      confoundNote: null,
    };
  }

  const busyDays = recentLogs.filter((log) => log.busyLevel === "busy").length;
  const recentSummaries = recentLogs.map(summarizeLog);
  const recent: MetricSummary = {
    safetyAlerts: average(recentSummaries.map((s) => s.safetyAlerts)),
    doubleChecks: average(recentSummaries.map((s) => s.doubleChecks)),
    dropRate: average(recentSummaries.map((s) => s.dropRate)),
    microDelayRate: average(recentSummaries.map((s) => s.microDelayRate)),
    droppedTasks: average(recentSummaries.map((s) => s.droppedTasks)),
    sampleCount: recentLogs.length,
  };

  const reasons: string[] = [];
  let signalCount = 0;

  if (recent.safetyAlerts >= baseline.safetyAlerts + 1) {
    reasons.push("최근 안전 알림 빈도가 평소보다 늘었어요.");
    signalCount += 1;
  }
  if (recent.doubleChecks >= baseline.doubleChecks + 1) {
    reasons.push("마감 반복 확인 횟수가 평소보다 늘었어요.");
    signalCount += 1;
  }
  if (recent.dropRate >= baseline.dropRate + 0.15) {
    reasons.push("시작한 업무가 완료되지 않고 남는 비율이 늘었어요.");
    signalCount += 1;
  }
  if (recent.microDelayRate >= baseline.microDelayRate + 0.15) {
    reasons.push("반복 업무 처리 시간이 평소보다 늘어난 날이 많았어요.");
    signalCount += 1;
  }

  const level: ChangeSignal["level"] =
    signalCount >= 3 ? "notable" : signalCount >= 1 ? "watch" : "none";

  const confoundNote =
    busyDays >= Math.ceil(recentLogs.length / 2)
      ? "이 기간 중 바쁜 날이 많았어요. 손님이 많아 생긴 변화일 수 있으니 참고만 해주세요."
      : null;

  return {
    level,
    reasons: reasons.length > 0 ? reasons : ["평소와 비슷한 흐름이 이어지고 있어요."],
    confoundNote,
  };
}
