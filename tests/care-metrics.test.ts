import assert from "node:assert/strict";
import test from "node:test";
import {
  emptyDailyLog,
  summarizeLog,
  computeBaseline,
  detectChangeSignal,
  type DailyLog,
} from "../app/care-metrics.ts";

function log(date: string, overrides: Partial<DailyLog> = {}): DailyLog {
  return { ...emptyDailyLog(date), ...overrides };
}

function dateNDaysAgo(n: number): string {
  const d = new Date("2026-08-14T00:00:00");
  d.setDate(d.getDate() - n);
  return d.toISOString().slice(0, 10);
}

// -- summarizeLog -----------------------------------------------------------

test("summarizeLog: drop rate is 0 when there are no started tasks", () => {
  const summary = summarizeLog(log("2026-08-01"));
  assert.equal(summary.dropRate, 0);
});

test("summarizeLog: drop rate reflects unfinished started tasks", () => {
  const summary = summarizeLog(
    log("2026-08-01", { tasksStarted: 4, tasksCompleted: 1 }),
  );
  assert.equal(summary.droppedTasks, 3);
  assert.equal(summary.dropRate, 0.75);
});

test("summarizeLog: micro-delay rate only counts delays over the slow threshold", () => {
  const summary = summarizeLog(
    log("2026-08-01", { microDelaySeconds: [10, 30, 150, 200] }),
  );
  assert.equal(summary.microDelayRate, 0.5); // 2 of 4 are >= 120s
});

// -- computeBaseline ----------------------------------------------------

test("computeBaseline: null until there are at least 3 days of history", () => {
  const logs = [log(dateNDaysAgo(1)), log(dateNDaysAgo(2))];
  assert.equal(computeBaseline(logs), null);
});

test("computeBaseline: averages quiet/normal days once there are 3+", () => {
  const logs = [
    log(dateNDaysAgo(1), { safetyAlerts: 0 }),
    log(dateNDaysAgo(2), { safetyAlerts: 2 }),
    log(dateNDaysAgo(3), { safetyAlerts: 1 }),
  ];
  const baseline = computeBaseline(logs);
  assert.ok(baseline);
  assert.equal(baseline!.safetyAlerts, 1);
});

// -- Persona 4: busy-day confound ------------------------------------------

test("computeBaseline: excludes 'busy' days by default so a hectic Saturday doesn't skew the baseline", () => {
  const logs = [
    log(dateNDaysAgo(1), { safetyAlerts: 0, busyLevel: "normal" }),
    log(dateNDaysAgo(2), { safetyAlerts: 0, busyLevel: "normal" }),
    log(dateNDaysAgo(3), { safetyAlerts: 0, busyLevel: "normal" }),
    log(dateNDaysAgo(4), { safetyAlerts: 5, busyLevel: "busy" }), // outlier, busy day
  ];
  const baseline = computeBaseline(logs);
  assert.ok(baseline);
  assert.equal(baseline!.safetyAlerts, 0, "busy day should be excluded from the baseline");
});

test("detectChangeSignal: flags a confound note when most recent days were busy", () => {
  const baselineLogs = [
    log(dateNDaysAgo(10), { safetyAlerts: 0 }),
    log(dateNDaysAgo(11), { safetyAlerts: 0 }),
    log(dateNDaysAgo(12), { safetyAlerts: 0 }),
  ];
  const baseline = computeBaseline(baselineLogs);
  const recent = [
    log(dateNDaysAgo(1), { safetyAlerts: 2, doubleChecks: 2, busyLevel: "busy" }),
    log(dateNDaysAgo(2), { safetyAlerts: 2, doubleChecks: 2, busyLevel: "busy" }),
  ];
  const signal = detectChangeSignal(recent, baseline);
  assert.ok(signal.confoundNote, "expected a confound note for a mostly-busy recent window");
});

test("detectChangeSignal: no confound note when recent days were normal", () => {
  const baselineLogs = [
    log(dateNDaysAgo(10), { safetyAlerts: 0 }),
    log(dateNDaysAgo(11), { safetyAlerts: 0 }),
    log(dateNDaysAgo(12), { safetyAlerts: 0 }),
  ];
  const baseline = computeBaseline(baselineLogs);
  const recent = [log(dateNDaysAgo(1), { safetyAlerts: 0, busyLevel: "normal" })];
  const signal = detectChangeSignal(recent, baseline);
  assert.equal(signal.confoundNote, null);
});

// -- Persona 6: gradually increasing pattern over weeks ----------------

test("detectChangeSignal: 'none' when recent days look like the personal baseline", () => {
  const baselineLogs = Array.from({ length: 14 }, (_, i) =>
    log(dateNDaysAgo(i + 4), { safetyAlerts: 0, doubleChecks: 1, tasksStarted: 4, tasksCompleted: 4 }),
  );
  const baseline = computeBaseline(baselineLogs);
  const recent = [
    log(dateNDaysAgo(1), { safetyAlerts: 0, doubleChecks: 1, tasksStarted: 4, tasksCompleted: 4 }),
    log(dateNDaysAgo(2), { safetyAlerts: 0, doubleChecks: 1, tasksStarted: 4, tasksCompleted: 4 }),
  ];
  const signal = detectChangeSignal(recent, baseline);
  assert.equal(signal.level, "none");
});

test("detectChangeSignal: 'notable' when several indicators rise together (synthetic decline scenario)", () => {
  const baselineLogs = Array.from({ length: 14 }, (_, i) =>
    log(dateNDaysAgo(i + 4), {
      safetyAlerts: 0,
      doubleChecks: 0,
      tasksStarted: 4,
      tasksCompleted: 4,
      microDelaySeconds: [30, 40, 20],
      busyLevel: "normal",
    }),
  );
  const baseline = computeBaseline(baselineLogs);
  const recent = [
    log(dateNDaysAgo(1), {
      safetyAlerts: 2,
      doubleChecks: 2,
      tasksStarted: 4,
      tasksCompleted: 1,
      microDelaySeconds: [150, 180, 200],
      busyLevel: "normal",
    }),
    log(dateNDaysAgo(2), {
      safetyAlerts: 1,
      doubleChecks: 2,
      tasksStarted: 4,
      tasksCompleted: 2,
      microDelaySeconds: [160, 170],
      busyLevel: "normal",
    }),
  ];
  const signal = detectChangeSignal(recent, baseline);
  assert.equal(signal.level, "notable");
  assert.ok(signal.reasons.length >= 3);
  assert.equal(signal.confoundNote, null, "these days were not busy, so no confound note is expected");
});

test("detectChangeSignal: never returns a diagnostic label, only soft levels", () => {
  const baseline = computeBaseline(
    Array.from({ length: 5 }, (_, i) => log(dateNDaysAgo(i + 1), { safetyAlerts: 5 })),
  );
  const recent = [log(dateNDaysAgo(1), { safetyAlerts: 9 })];
  const signal = detectChangeSignal(recent, baseline);
  assert.ok(["none", "watch", "notable"].includes(signal.level));
  for (const reason of signal.reasons) {
    assert.doesNotMatch(reason, /치매|진단|질환/);
  }
});
