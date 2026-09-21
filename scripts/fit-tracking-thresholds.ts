/**
 * Fits the tracking layer's own decision thresholds against real data.
 *
 *   npx tsx scripts/fit-tracking-thresholds.ts coco-wholebody-val-rtmw-l-384x288
 *
 * Sweeps `TrackingTuning` over a scenario and reports the retention /
 * empty-frame-false-positive trade-off at each point. The scenario is split by
 * image so the reported winner is chosen on one half and scored on the other -
 * a threshold picked on the data it is then judged by is not evidence.
 *
 * These thresholds are cheap: none of them changes runtime cost. They are also
 * bounded - the diagnosis in docs/tracking-eval.md shows the anatomy gate only
 * costs ~2 points - so this exists to confirm the defaults are not leaving
 * something free on the table, not to rescue the numbers.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  DEFAULT_TUNING,
  EMPTY_GLOBAL_TRACK,
  updateGlobalPersonTrack,
  type GlobalPersonTrack,
  type TrackingTuning,
} from "../app/global-person-track.ts";
import type { EvalScenario, EvalTick } from "../app/tracking-eval.ts";

const name = process.argv[2] ?? "coco-wholebody-val-rtmw-l-384x288";
const scenario = JSON.parse(
  readFileSync(join(process.cwd(), "data", "eval-scenarios", `${name}.json`), "utf8"),
) as EvalScenario;

/** Groups ticks into the independent images they came from. */
function segments(ticks: EvalTick[]): EvalTick[][] {
  const groups: EvalTick[][] = [];
  for (const tick of ticks) {
    if (tick.resetBefore || !groups.length) groups.push([]);
    groups[groups.length - 1].push(tick);
  }
  return groups;
}

function score(groups: EvalTick[][], tuning: TrackingTuning) {
  let personTicks = 0;
  let retained = 0;
  let emptyFalsePositives = 0;
  for (const group of groups) {
    let track: GlobalPersonTrack = EMPTY_GLOBAL_TRACK;
    group.forEach((tick, index) => {
      track = updateGlobalPersonTrack(
        track, tick.observations, tick.timestamp, undefined, tuning,
      );
      const present = track.presence === "present" || track.presence === "coasting";
      // The gate cannot report present during its own ramp; counting those
      // would score the gate's existence rather than the thresholds.
      if (index < tuning.confirmationFrames - 1) return;
      if (tick.personPresent) {
        personTicks += 1;
        if (present) retained += 1;
      } else if (present) {
        emptyFalsePositives += 1;
      }
    });
  }
  return {
    retention: personTicks ? retained / personTicks : 0,
    emptyFalsePositives,
  };
}

const all = segments(scenario.ticks);
// Split by image, interleaved, so both halves carry the same condition mix.
const fitSet = all.filter((_, index) => index % 2 === 0);
const testSet = all.filter((_, index) => index % 2 === 1);

const grid: TrackingTuning[] = [];
for (const minJointVisibility of [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]) {
  for (const limbAsymmetryFloor of [0, 0.1, 0.2, 0.25, 0.3, 0.4]) {
    for (const confirmationFrames of [2, 3, 4]) {
      grid.push({ minJointVisibility, limbAsymmetryFloor, confirmationFrames });
    }
  }
}

const scored = grid.map((tuning) => ({ tuning, fit: score(fitSet, tuning) }));

// Retention is the objective; empty-frame false positives are the constraint,
// and the baseline is what the current defaults already achieve.
const baseline = score(fitSet, DEFAULT_TUNING);
const feasible = scored.filter(
  (entry) => entry.fit.emptyFalsePositives <= baseline.emptyFalsePositives,
);
feasible.sort((a, b) =>
  b.fit.retention - a.fit.retention ||
  a.fit.emptyFalsePositives - b.fit.emptyFalsePositives ||
  a.tuning.confirmationFrames - b.tuning.confirmationFrames,
);

const best = feasible[0];
const describe = (tuning: TrackingTuning) =>
  `vis=${tuning.minJointVisibility} asym=${tuning.limbAsymmetryFloor} frames=${tuning.confirmationFrames}`;

console.log(`scenario: ${scenario.name}`);
console.log(`images:   ${all.length} (fit ${fitSet.length} / test ${testSet.length})\n`);
console.log(`${"tuning".padEnd(36)} ${"fit reten".padStart(10)} ${"fit FP".padStart(7)} ${"TEST reten".padStart(11)} ${"TEST FP".padStart(8)}`);

const defaultTest = score(testSet, DEFAULT_TUNING);
console.log(
  `${(describe(DEFAULT_TUNING) + "  [default]").padEnd(36)} ${(baseline.retention * 100).toFixed(1).padStart(9)}% ${String(baseline.emptyFalsePositives).padStart(7)} ${(defaultTest.retention * 100).toFixed(1).padStart(10)}% ${String(defaultTest.emptyFalsePositives).padStart(8)}`,
);

for (const entry of feasible.slice(0, 8)) {
  const test = score(testSet, entry.tuning);
  console.log(
    `${describe(entry.tuning).padEnd(36)} ${(entry.fit.retention * 100).toFixed(1).padStart(9)}% ${String(entry.fit.emptyFalsePositives).padStart(7)} ${(test.retention * 100).toFixed(1).padStart(10)}% ${String(test.emptyFalsePositives).padStart(8)}`,
  );
}

if (best) {
  const bestTest = score(testSet, best.tuning);
  const delta = (bestTest.retention - defaultTest.retention) * 100;
  console.log(
    `\nbest on fit set: ${describe(best.tuning)}\n` +
    `held-out delta vs default: ${delta >= 0 ? "+" : ""}${delta.toFixed(2)} pp retention, ` +
    `${bestTest.emptyFalsePositives - defaultTest.emptyFalsePositives >= 0 ? "+" : ""}` +
    `${bestTest.emptyFalsePositives - defaultTest.emptyFalsePositives} empty false positives`,
  );
  if (delta < 0.5) {
    console.log("\nThe defaults are within half a point of the best grid entry on held-out data.\nChanging them is not supported by this evidence.");
  }
}
