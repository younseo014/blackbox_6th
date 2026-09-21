/**
 * Attributes every lost person-present tick in a scenario to the gate that
 * dropped it, so tuning targets the actual bottleneck instead of the first
 * threshold that comes to mind.
 *
 *   npx tsx scripts/diagnose-eval.ts coco-wholebody-val
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  EMPTY_GLOBAL_TRACK,
  anatomyIsValid,
  torsoFrame,
  updateGlobalPersonTrack,
  type GlobalPersonTrack,
} from "../app/global-person-track.ts";
import type { EvalScenario } from "../app/tracking-eval.ts";

const name = process.argv[2] ?? "coco-wholebody-val";
const scenario = JSON.parse(
  readFileSync(join(process.cwd(), "data", "eval-scenarios", `${name}.json`), "utf8"),
) as EvalScenario;

type Reason =
  | "ok"
  | "pose_found_nothing"
  | "no_detector_evidence"
  | "no_torso_frame"
  | "anatomy_rejected"
  | "not_continuous"
  | "awaiting_confirmation";

const byReason = new Map<Reason, number>();
const byConditionReason = new Map<string, Map<Reason, number>>();

let track: GlobalPersonTrack = EMPTY_GLOBAL_TRACK;
for (const tick of scenario.ticks) {
  if (tick.resetBefore) track = EMPTY_GLOBAL_TRACK;
  track = updateGlobalPersonTrack(track, tick.observations, tick.timestamp);
  if (!tick.personPresent) continue;

  const present = track.presence === "present" || track.presence === "coasting";
  let reason: Reason = "ok";
  if (!present) {
    const observation = tick.observations[0];
    const landmarks = observation.landmarks;
    if (!landmarks) reason = "pose_found_nothing";
    else if (observation.personConfidence <= 0 && observation.foregroundRatio <= 0 && !observation.confirmed) {
      reason = "no_detector_evidence";
    } else if (!torsoFrame(landmarks)) reason = "no_torso_frame";
    else if (!anatomyIsValid(landmarks)) reason = "anatomy_rejected";
    else if (track.presence === "provisional") reason = "awaiting_confirmation";
    else reason = "not_continuous";
  }
  byReason.set(reason, (byReason.get(reason) ?? 0) + 1);
  const perCondition = byConditionReason.get(tick.condition) ?? new Map<Reason, number>();
  perCondition.set(reason, (perCondition.get(reason) ?? 0) + 1);
  byConditionReason.set(tick.condition, perCondition);
}

const total = [...byReason.values()].reduce((sum, value) => sum + value, 0);
console.log(`scenario: ${scenario.name}  person-present ticks: ${total}\n`);
console.log("overall:");
for (const [reason, count] of [...byReason.entries()].sort((a, b) => b[1] - a[1])) {
  console.log(`  ${reason.padEnd(22)} ${String(count).padStart(6)}  ${((count / total) * 100).toFixed(1)}%`);
}
console.log("\nby condition:");
for (const [condition, reasons] of byConditionReason) {
  const conditionTotal = [...reasons.values()].reduce((sum, value) => sum + value, 0);
  const parts = [...reasons.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([reason, count]) => `${reason}=${((count / conditionTotal) * 100).toFixed(0)}%`);
  console.log(`  ${condition.padEnd(14)} n=${String(conditionTotal).padStart(5)}  ${parts.join("  ")}`);
}
