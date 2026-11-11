import type { ObservationFeatures } from "./observation-engine";

export const SELF_TEST_PROTOCOL = "workbench-v2";
export const LEGACY_SELF_TEST_ACTION_LABELS = ["컵 보충", "냉장 재료 꺼내기", "음료 재료 혼합", "테이블 닦기", "식기 세척"];
export const SELF_TEST_ACTION_LABELS = ["컵 보충", "컵에 얼음 담기", "음료 재료 혼합", "컵 뚜껑 닫기", "음료 포장"];
export const SELF_TEST_ZONE_GRID: Array<string | null> = Array(9).fill("DRINK_PREP");

export function selfTestTrainingKey(participant: string, protocol?: string) {
  // Preserve legacy banks under their original keys; new protocols never reuse them.
  return protocol === SELF_TEST_PROTOCOL ? JSON.stringify([SELF_TEST_PROTOCOL, participant]) : participant;
}

export function selfTestLabels(protocol?: string) {
  return protocol === SELF_TEST_PROTOCOL ? SELF_TEST_ACTION_LABELS : LEGACY_SELF_TEST_ACTION_LABELS;
}

export function selfTestFeatures(features: ObservationFeatures, protocol?: string): ObservationFeatures {
  return protocol === SELF_TEST_PROTOCOL
    ? { ...features, dominantZone: "DRINK_PREP", zoneTransitions: 0 }
    : features;
}
