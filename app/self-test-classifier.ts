import type { ObservationFeatures } from "./observation-engine";
import { getOccupationTemplate } from "./occupation-templates";
import { parseSessionFrame } from "./pose-store";
import { workContextTaskType } from "./work-context-inference";

const TEST_LABELS = new Set(["컵 보충", "냉장 재료 꺼내기", "음료 재료 혼합", "테이블 닦기", "식기 세척"]);

function hasRepeatedWristMotion(rawFrames: number[][]) {
  return [0, 1].some((side) => {
    let previousRelativeX: number | null = null;
    let previousDirection = 0;
    let reversals = 0;
    for (const raw of rawFrames) {
      const body = parseSessionFrame(raw).body;
      const shoulder = side;
      const wrist = side + 4;
      if ((body[shoulder * 4 + 3] ?? 0) < 0.35 || (body[wrist * 4 + 3] ?? 0) < 0.35) continue;
      const relativeX = body[wrist * 4] - body[shoulder * 4];
      if (previousRelativeX !== null) {
        const delta = relativeX - previousRelativeX;
        if (Math.abs(delta) > 0.008) {
          const direction = Math.sign(delta);
          if (previousDirection && direction !== previousDirection) reversals += 1;
          previousDirection = direction;
        }
      }
      previousRelativeX = relativeX;
    }
    return reversals >= 2;
  });
}

/** Scores the five cafe tasks using the observable parts of their context graph. */
export function classifyCafeSelfTest(features: ObservationFeatures, rawFrames: number[][]) {
  const observed = new Set<string>();
  if (features.pathLength > 0.45) {
    observed.add("WALK");
    observed.add("CARRY");
  }
  if (features.zoneTransitions > 0) observed.add("ZONE_TRANSITION");
  if (hasRepeatedWristMotion(rawFrames)) {
    // Repeated arm motion is a shared visible cue for these graph primitives.
    ["MIX", "STIR", "WIPE", "SCRUB", "WASH"].forEach((motion) => observed.add(motion));
  }
  if (features.longestPauseSeconds >= 2) observed.add("STATIC_PAUSE");

  return getOccupationTemplate("cafe").tasks
    .filter((task) => TEST_LABELS.has(task.label))
    .map((task) => {
      const zoneMatch = Boolean(features.dominantZone && task.zones.includes(features.dominantZone));
      const motionMatches = task.motions.filter((motion) => observed.has(motion)).length;
      const confidence = Math.min(0.78, (zoneMatch ? 0.64 : 0.12) + Math.min(2, motionMatches) * 0.07);
      return {
        taskType: workContextTaskType(task.label),
        taskLabel: task.label,
        confidence,
      };
    })
    .sort((a, b) => b.confidence - a.confidence);
}
