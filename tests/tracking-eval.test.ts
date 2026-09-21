import assert from "node:assert/strict";
import test from "node:test";
import {
  checkAcceptance,
  evaluateScenario,
  renderMarkdownReport,
  type EvalScenario,
} from "../app/tracking-eval.ts";
import { builtInScenarios, syntheticPerson } from "../app/tracking-eval-scenarios.ts";

function scenarioNamed(name: string): EvalScenario {
  const scenario = builtInScenarios().find((item) => item.name === name);
  assert.ok(scenario, `missing built-in scenario ${name}`);
  return scenario;
}

test("every built-in scenario declares its provenance", () => {
  for (const scenario of builtInScenarios()) {
    assert.ok(scenario.source.length > 0, `${scenario.name} has no source`);
    assert.ok(scenario.ticks.length > 0, `${scenario.name} has no ticks`);
  }
});

test("an empty room produces no false detections", () => {
  const result = evaluateScenario(scenarioNamed("empty-room"));
  assert.equal(result.emptyFrameFalsePositives, 0);
  assert.equal(result.presenceAccuracy, 1);
  assert.ok(result.emptyFrameFalsePositivesPer12h < 1);
});

test("a chair-shaped decoy never becomes the tracked target", () => {
  const result = evaluateScenario(scenarioNamed("decoy-furniture"));
  assert.equal(result.objectSwitches, 0);
  assert.equal(result.emptyFrameFalsePositives, 0);
});

test("a person entering is recognised within one second", () => {
  const result = evaluateScenario(scenarioNamed("entry-latency"));
  assert.ok(result.worstRecognitionLatencyMs !== null);
  assert.ok(
    result.worstRecognitionLatencyMs! <= 1_000,
    `recognition took ${result.worstRecognitionLatencyMs}ms`,
  );
});

test("three overlapping cameras never produce more than one person id", () => {
  const result = evaluateScenario(scenarioNamed("three-camera-overlap"));
  assert.equal(result.maxConcurrentTracks, 1);
  assert.equal(result.distinctTrackIds, 1);
  assert.equal(Object.keys(result.cameraFps).length, 3, "all three cameras must be reported");
});

test("camera handoff keeps one identity and reports the blind gap", () => {
  const result = evaluateScenario(scenarioNamed("camera-handoff"));
  assert.equal(result.distinctTrackIds, 1);
  assert.equal(result.handoffs, 2);
  assert.ok(result.longestHandoffGapMs <= 500, `gap was ${result.longestHandoffGapMs}ms`);
});

test("side, back, headless and partial views all keep the person", () => {
  for (const name of ["side-view", "back-view", "headless", "upper-body-only", "distant-small"]) {
    const result = evaluateScenario(scenarioNamed(name));
    assert.ok(result.torsoRetention >= 0.95, `${name} retention was ${result.torsoRetention}`);
  }
});

test("joint states are reported per joint and never silently invented", () => {
  const result = evaluateScenario(scenarioNamed("upper-body-only"));
  const ankle = result.joints.find((joint) => joint.index === 27)!;
  assert.equal(ankle.observedCount, 0, "an out-of-frame ankle must not be reported as observed");
  const shoulder = result.joints.find((joint) => joint.index === 11)!;
  assert.ok(shoulder.observedCount > 0);
});

test("acceptance check reports a failure instead of smoothing it over", () => {
  const broken: EvalScenario = {
    name: "regression",
    description: "A person the tracker never picks up.",
    source: "synthetic:unit-test",
    ticks: Array.from({ length: 30 }, (_, index) => ({
      timestamp: index * 100,
      personPresent: true,
      condition: "front" as const,
      truth: { camera: 1 as const, joints: { 11: { x: 0.5, y: 0.4 } } },
      observations: [{
        cameraSlot: 1 as const,
        timestamp: index * 100,
        landmarks: null,
        personConfidence: 0,
        foregroundRatio: 0,
        confirmed: false,
      }],
    })),
  };
  const result = evaluateScenario(broken);
  const acceptance = checkAcceptance(result);
  const retention = acceptance.find((criterion) => criterion.id === "torso_retention")!;
  assert.equal(retention.passed, false);
  const report = renderMarkdownReport("unit-test", [{ result, acceptance }]);
  assert.match(report, /\*\*FAIL\*\*/);
  assert.match(report, /## Failures/);
  assert.ok(!report.includes("- None."), "a failing run must not print an empty failure list");
});

test("scenarios without annotated joints are not scored as 0% detection", () => {
  const result = evaluateScenario(scenarioNamed("empty-room"));
  const joints = checkAcceptance(result).find((criterion) => criterion.id === "joint_detection")!;
  assert.equal(joints.actual, "n/a");
  assert.equal(joints.passed, true);
});

test("the synthetic person generator produces an anatomically usable body", () => {
  const person = syntheticPerson();
  assert.ok((person[11].visibility ?? 0) > 0.5);
  assert.ok(person[11].x > person[12].x, "front view keeps the mirrored shoulder ordering");
  assert.ok(syntheticPerson({ back: true })[11].x < syntheticPerson({ back: true })[12].x);
});
