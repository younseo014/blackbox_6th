import test from "node:test";
import assert from "node:assert/strict";
import { generateEventMotion } from "../app/demo-motion.ts";
import { workSequence, workPoseCost, representativeWorkSamples } from "../app/work-motion-features.ts";
import { classifyWorkMotion, selectedWorkReferences, type LoadedLearnedMotionAction } from "../app/custom-motion-training.ts";
import { decideWorkLabel, canAutomaticallyLearnWork } from "../app/work-action-decision.ts";
import { preserveUncoveredWork, selectWorkCandidates, segmentWorkEpisodesDetailed, type WorkSegmentationReport } from "../app/learned-work-segmentation.ts";
import type { TargetMotionSlice } from "../app/motion-segmentation.ts";

function gesture(kind: number, offset = 0, speed = 1) {
  const base = generateEventMotion("normal_task", 1)[0];
  return Array.from({ length: 81 }, (_, i) => {
    const f = [...base]; f[0] = offset + i * 100 * speed; f[6] = f[7] = 0;
    const progress = i / 80, amount = Math.sin(progress * Math.PI);
    for (const [joint, x, y] of [[0, .4, .3], [1, .6, .3], [2, .4, .5], [3, .6, .5],
      [4, .4 + (kind % 2 === 0 ? .2 : -.2) * amount, .7 - .15 * amount],
      [5, .6 + (kind < 2 ? .06 : -.15) * amount, .7 - (kind === 3 ? .32 : .08) * amount]]) {
      f[10 + joint * 4] = x; f[10 + joint * 4 + 1] = y; f[10 + joint * 4 + 3] = .95;
    }
    return f;
  });
}
const action = (id: string, kind: number, source: "manual" | "sequence_model" = "manual"): LoadedLearnedMotionAction => ({
  id, label: id, samples: [gesture(kind)], sampleSources: [source],
});
const report = (): WorkSegmentationReport => ({ version: "test", featureVersion: "test", references: [], candidates: [] });
const slice = (frames: number[][]): TargetMotionSlice => ({ frames, startMs: frames[0][0], endMs: frames.at(-1)![0],
  durationSeconds: (frames.at(-1)![0] - frames[0][0]) / 1000, originalDurationSeconds: 8, excludedFrameCount: 0, reason: "fixture",
  boundaryEvidence: { method: "arm_lifecycle", startObserved: true, endObserved: true } });

test("competing intervals cannot both survive regardless of their overlap fraction", () => {
  for (const scale of [1, 3, 11]) for (const fraction of Array.from({length:20},(_,i)=>(i+1)/20)) {
    const a = { startMs: 100 * scale, endMs: 1100 * scale, label: "A", distance: .01 };
    const b = { startMs: (1100 - 1000 * fraction) * scale, endMs: (2100 - 1000 * fraction) * scale, label: "A", distance: .1 };
    assert.equal(selectWorkCandidates([b, a]).length, 1);
  }
  const separate = [{ startMs: 0, endMs: 1000, label: "A", distance: .01 }, { startMs: 1000, endMs: 2000, label: "A", distance: .01 }];
  assert.equal(selectWorkCandidates(separate).length, 2);
});

test("a small overlapping match preserves unexplained movement on both sides", () => {
  const frames = gesture(0), original = slice(frames), middle = slice(frames.filter(f => f[0] >= 2500 && f[0] <= 4500));
  const audit = report();
  const residual = preserveUncoveredWork([original], [middle], frames, audit);
  assert.equal(residual.length, 2);
  assert.ok(residual[0].endMs < middle.startMs && residual[1].startMs > middle.endMs);
  assert.ok(residual.every(s => !s.boundaryEvidence?.startObserved && !s.boundaryEvidence?.endObserved));
  const retained = new Set([...middle.frames, ...residual.flatMap(s => s.frames)].map(f => f[0]));
  assert.equal(retained.size, frames.length);
  assert.equal(audit.candidates.filter(c => c.stage === "residual").length, 2);
});

test("short residual observations remain in the decision trace", () => {
  const frames = gesture(0), audit = report();
  preserveUncoveredWork([slice(frames)], [slice(frames.filter(f => f[0] >= 200 && f[0] <= 7800))], frames, audit);
  assert.equal(audit.candidates.filter(c => c.stage === "incomplete").length, 2);
});

test("reference selection finds diversity missed by storage-order sampling", () => {
  const common = gesture(0), different = gesture(3);
  const samples = Array.from({ length: 20 }, () => common); samples.splice(2, 0, different);
  assert.ok(representativeWorkSamples(samples).includes(2));
  const reversed = [...samples].reverse();
  assert.ok(representativeWorkSamples(reversed).some(i => reversed[i] === different));
  assert.equal(representativeWorkSamples(Array(30).fill(common)).length, 1);
});

test("completed work is compared against all labels independently of its proposal", () => {
  const actions = [action("A", 0), action("B", 3), action("C", 1), action("D", 2)];
  const ranked = classifyWorkMotion(gesture(3, 11000), actions);
  assert.equal(ranked.label, "B");
  assert.equal(ranked.candidates.length, 4);
  assert.equal(ranked.independentSupport, true);
  assert.deepEqual(decideWorkLabel(ranked, actions.map(a => a.label)).candidates.map(c => c.taskLabel), ranked.candidates.map(c => c.label));
});

test("equal labels and missing alternatives cannot become confident classifications", () => {
  const shared = gesture(0);
  const equal = classifyWorkMotion(shared, [{id:"A",label:"A",samples:[shared]}, {id:"B",label:"B",samples:[shared]}]);
  assert.equal(equal.status, "uncertain");
  assert.equal(equal.candidateMargin, 0);
  assert.equal(decideWorkLabel(equal, ["A", "B"]).autoLearningEligible, false);
  const onlyOne = classifyWorkMotion(shared, [action("A", 0)], ["A", "B"]);
  assert.equal(onlyOne.candidateMargin, null);
  assert.equal(onlyOne.status, "uncertain");
});

test("recursive automatic examples cannot serve as independent references", () => {
  const automatic = action("A", 0, "sequence_model");
  assert.equal(selectedWorkReferences(automatic).samples.length, 0);
  assert.equal(classifyWorkMotion(gesture(0), [automatic]).status, "insufficient");
  const legacy = classifyWorkMotion(gesture(0), [{id:"A",label:"A",samples:[gesture(0)]}, {id:"B",label:"B",samples:[gesture(3)]}]);
  assert.equal(legacy.independentSupport, false);
  assert.equal(decideWorkLabel(legacy, ["A","B"]).autoLearningEligible, false);
});

test("weak location and motion hints are not emitted as an initial work label", () => {
  const motion = classifyWorkMotion(gesture(0), []);
  const decision = decideWorkLabel(motion, ["A", "B"]);
  assert.deepEqual(decision.candidates, []);
  assert.equal(decision.source, "insufficient");
  assert.equal(decision.autoLearningEligible, false);
});

test("observed visual context can bootstrap a task but conflict remains reviewable", () => {
  const visual = {startMs:0,endMs:8000,label:"A",kind:"task" as const,similarity:.4,margin:.08,sampleCount:5};
  const cold = decideWorkLabel(classifyWorkMotion(gesture(0), []), ["A","B"], visual);
  assert.equal(cold.source, "visual_context"); assert.equal(cold.autoLearningEligible, true);
  const conflict = decideWorkLabel(classifyWorkMotion(gesture(3), [action("A",0),action("B",3)]), ["A","B"], visual);
  assert.ok(conflict.qualityReasons.includes("unknown_motion")); assert.equal(conflict.autoLearningEligible, false);
});

test("missing hands are masked instead of inserted as a stationary hand shape", () => {
  const observed = workSequence(gesture(0));
  assert.ok(observed.every(p => p.weights.slice(10,30).every(w => w === 0)));
  assert.equal(workPoseCost(observed[4], observed[4]), 0);
});

test("global execution speed does not define a different work label", () => {
  const actions = [action("A",0),action("B",3)];
  for (const speed of [.6, 1, 2, 4]) assert.equal(classifyWorkMotion(gesture(0, 19000, speed), actions).label, "A");
});

test("segmentation retains auditable references and never trains from the recording", () => {
  const actions = [action("A",0),action("B",3)], before = JSON.stringify(actions);
  const result = segmentWorkEpisodesDetailed(gesture(0), actions);
  assert.equal(JSON.stringify(actions), before);
  assert.equal(result.report.references.length, 2);
  assert.equal(result.report.featureVersion, "shoulder-arms-hands-v1");
  assert.ok(result.report.candidates.length > 0);
});


test("analysis never updates learning, including confidently matched work", () => {
  const matched = decideWorkLabel(classifyWorkMotion(gesture(0),[action("A",0),action("B",3)]),["A","B"],
    {startMs:0,endMs:8000,label:"A",kind:"task",similarity:.4,margin:.08,sampleCount:5});
  assert.equal(matched.autoLearningEligible,true);
  assert.equal(canAutomaticallyLearnWork(matched,"training",false),true);
  assert.equal(canAutomaticallyLearnWork(matched,"learning",false),true);
  assert.equal(canAutomaticallyLearnWork(matched,"evaluation",false),false);
  assert.equal(canAutomaticallyLearnWork(matched,"analysis",false),false);
  assert.equal(canAutomaticallyLearnWork(matched,"training",true),false);
});

test("independent image evidence may enrol a new class without a manual bootstrap", () => {
  const partial = classifyWorkMotion(gesture(3),[action("A",0)],["A","B"]);
  const decision = decideWorkLabel(partial,["A","B"],{startMs:0,endMs:8000,label:"B",kind:"task",similarity:.4,margin:.08,sampleCount:5});
  assert.equal(decision.candidates[0].taskLabel,"B");
  assert.equal(decision.source,"visual_context");
  assert.equal(canAutomaticallyLearnWork(decision,"training",false),true);
});

test("confirmed representatives and independently observed visual variants can coexist", () => {
  const manual = gesture(0), visualVariant = gesture(3);
  const pool = selectedWorkReferences({id:"A",label:"A",samples:[manual,visualVariant,gesture(1)],sampleSources:["manual","visual_model","sequence_model"],boundarySamples:[manual]});
  assert.equal(pool.samples.length,2);
  assert.ok(pool.indexes.includes(0) && pool.indexes.includes(1));
});


test("a confident pose-only classification is retained without feeding itself back into training", () => {
  const motion = classifyWorkMotion(gesture(0),[action("A",0),action("B",3)]);
  const decision = decideWorkLabel(motion,["A","B"]);
  assert.equal(motion.status,"matched");
  assert.equal(decision.candidates[0].taskLabel,"A");
  assert.deepEqual(decision.qualityReasons,[]);
  assert.equal(decision.autoLearningEligible,false);
});
