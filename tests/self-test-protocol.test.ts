import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SELF_TEST_PROTOCOL, SELF_TEST_ACTION_LABELS, LEGACY_SELF_TEST_ACTION_LABELS, SELF_TEST_ZONE_GRID, selfTestTrainingKey, selfTestLabels, selfTestFeatures } from "../app/self-test-protocol.ts";
import { extractObservationFeatures } from "../app/observation-engine.ts";
import { classifyCafeSelfTest } from "../app/self-test-classifier.ts";
import { rankVisualContext } from "../app/local-action-context.ts";

test("workbench tasks match the actual five classifier and model labels", () => {
  assert.deepEqual(SELF_TEST_ACTION_LABELS, ["컵 보충", "컵에 얼음 담기", "음료 재료 혼합", "컵 뚜껑 닫기", "음료 포장"]);
  const features = selfTestFeatures(extractObservationFeatures([], []), SELF_TEST_PROTOCOL);
  assert.deepEqual(classifyCafeSelfTest(features, []).map(c => c.taskLabel).sort(), [...SELF_TEST_ACTION_LABELS].sort());
  const prototypes = JSON.parse(readFileSync(new URL("../public/models/action-context/prototypes.json", import.meta.url), "utf8")).prototypes;
  for (const label of [...SELF_TEST_ACTION_LABELS, ...LEGACY_SELF_TEST_ACTION_LABELS]) {
    const p = prototypes.find((p: {label:string}) => p.label === label);
    assert.equal(p?.embedding.length, 512);
  }
});

test("fixed workbench context does not require a visible hip or turn lost tracking into movement", () => {
  const measured = extractObservationFeatures([], []);
  assert.equal(measured.dominantZone, null);
  const fixed = selfTestFeatures(measured, SELF_TEST_PROTOCOL);
  assert.equal(fixed.dominantZone, "DRINK_PREP");
  assert.equal(fixed.zoneTransitions, 0);
  assert.equal(fixed.pathLength, measured.pathLength);
  assert.equal(fixed.activeRatio, measured.activeRatio);
  assert.equal(selfTestFeatures(measured), measured);
  assert.deepEqual(SELF_TEST_ZONE_GRID, Array(9).fill("DRINK_PREP"));
});

test("old and new participant banks and review options remain separate", () => {
  const oldKey = selfTestTrainingKey("A");
  const newKey = selfTestTrainingKey("A", SELF_TEST_PROTOCOL);
  assert.notEqual(oldKey, newKey);
  assert.equal(oldKey, "A");
  const banks = { [oldKey]: ["테이블 닦기"], [newKey]: ["컵 뚜껑 닫기"] };
  assert.deepEqual(banks[newKey], ["컵 뚜껑 닫기"]);
  assert.deepEqual(selfTestLabels(), LEGACY_SELF_TEST_ACTION_LABELS);
  assert.deepEqual(selfTestLabels(SELF_TEST_PROTOCOL), SELF_TEST_ACTION_LABELS);
});

test("task filtering happens before scoring and retains interruption alternatives", () => {
  const vector = (first:number, second:number) => [first,second,...Array(510).fill(0)];
  const candidates = [
    {label:"테이블 닦기",kind:"task" as const,embedding:vector(1,0)},
    {label:"컵 뚜껑 닫기",kind:"task" as const,embedding:vector(.9,.1)},
    {label:"전화 사용 의심",kind:"phone" as const,embedding:vector(.95,.05)},
  ];
  assert.equal(rankVisualContext(vector(1,0), candidates).label, "테이블 닦기");
  assert.equal(rankVisualContext(vector(1,0), candidates, SELF_TEST_ACTION_LABELS).label, "전화 사용 의심");
  assert.equal(rankVisualContext(vector(1,0), candidates, LEGACY_SELF_TEST_ACTION_LABELS).label, "테이블 닦기");
});
