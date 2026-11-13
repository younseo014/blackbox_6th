import { readFileSync, writeFileSync, realpathSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve, dirname } from "node:path";
import { segmentWorkEpisodesDetailed, WORK_RECOGNITION_VERSION } from "../app/learned-work-segmentation.ts";
import { classifyWorkMotion, type LoadedLearnedMotionAction } from "../app/custom-motion-training.ts";
import { decideWorkLabel } from "../app/work-action-decision.ts";
import { contextualMotionSlices, visualActionIntervals } from "../app/local-action-context.ts";
import { hasCompleteWorkBoundaries } from "../app/work-episode-segmentation.ts";
import { detectWorkInterruptions, hasInterruptionConfound, summarizeInterruptions } from "../app/work-interruptions.ts";
import { selfTestLabels } from "../app/self-test-protocol.ts";

type Interval = { startMs: number; endMs: number; label: string };
type Prediction = Interval & { reviewRequired: boolean; candidates?: unknown };
type Reference = { label: string; recording: string; startMs: number; endMs: number; labelSource?: "manual" | "visual_model" | "sequence_model" };
type TestCase = { name: string; split: "development" | "heldout" | "regression" | "unlabelled"; recording: string;
  references?: Reference[]; referenceRecordings?: Record<string,string>; allowColdStart?: boolean; expectedLabels?: string[]; expected: Interval[] | null; baselinePredictions?: string };
const args = process.argv.slice(2), manifest = args[args.indexOf("--cases") + 1];
if (!args.includes("--cases") || !manifest) throw Error("Use --cases external-cases.json --output report.json");
const casesPath = resolve(manifest), base = dirname(casesPath);
const output = args.includes("--output") ? args[args.indexOf("--output") + 1] : "work-recognition-evaluation.json";
const cases = JSON.parse(readFileSync(casesPath, "utf8")) as TestCase[];
const hash = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");
const validate = (i: Interval) => {
  if (!Number.isFinite(i.startMs) || !Number.isFinite(i.endMs) || i.startMs < 0 || i.endMs <= i.startMs || typeof i.label !== "string") throw Error("Invalid labelled interval");
};
const iou = (a: Interval, b: Interval) => Math.max(0, Math.min(a.endMs, b.endMs) - Math.max(a.startMs, b.startMs)) /
  Math.max(1, Math.max(a.endMs, b.endMs) - Math.min(a.startMs, b.startMs));
const quantile = (values: number[], fraction: number) => values.length ? [...values].sort((a,b) => a-b)[Math.ceil((values.length - 1) * fraction)] : null;
function score(expected: Interval[] | null, predicted: Prediction[]) {
  if (expected === null) return null;
  expected.forEach(validate);
  const pairs = expected.flatMap((truth, t) => predicted.map((p, n) => ({ t, n, overlap: iou(truth, p) }))).sort((a,b) => b.overlap-a.overlap);
  const usedTruth = new Set<number>(), usedPred = new Set<number>(), errors: Array<{ start: number; end: number; duration: number }> = [];
  let labelCorrect = 0, automaticCorrect = 0;
  const confusion: Record<string, Record<string, number>> = {};
  const record = (actual: string, predicted: string) => { confusion[actual] ??= {}; confusion[actual][predicted] = (confusion[actual][predicted] ?? 0) + 1; };
  for (const pair of pairs) {
    if (pair.overlap < .5 || usedTruth.has(pair.t) || usedPred.has(pair.n)) continue;
    usedTruth.add(pair.t); usedPred.add(pair.n);
    const truth = expected[pair.t], p = predicted[pair.n];
    record(truth.label, p.label);
    if (truth.label === p.label) { labelCorrect++; if (!p.reviewRequired) automaticCorrect++; }
    errors.push({ start: Math.abs(truth.startMs - p.startMs) / 1000, end: Math.abs(truth.endMs - p.endMs) / 1000,
      duration: Math.abs((truth.endMs-truth.startMs)-(p.endMs-p.startMs))/1000 });
  }
  expected.forEach((truth, i) => { if (!usedTruth.has(i)) record(truth.label, "MISSED"); });
  return { actualCount: expected.length, predictedCount: predicted.length, oneToOneMatched: usedTruth.size,
    labelCorrect, automaticCorrect, missed: expected.length-usedTruth.size, unmatchedPredictions: predicted.length-usedPred.size,
    reviewRequired: predicted.filter(p => p.reviewRequired).length,
    overlappingExtraCandidates: predicted.filter((p,i) => !usedPred.has(i) && expected.some(t => iou(t,p)>0)).length,
    possibleMergedCandidates: predicted.filter(p => expected.filter(t => iou(t,p)>0).length>1).length,
    labelSuccessPerActual: expected.length ? labelCorrect/expected.length : null,
    automaticSuccessPerActual: expected.length ? automaticCorrect/expected.length : null,
    boundarySeconds: { startMedian: quantile(errors.map(e=>e.start),.5), startP90: quantile(errors.map(e=>e.start),.9),
      endMedian: quantile(errors.map(e=>e.end),.5), endP90: quantile(errors.map(e=>e.end),.9), durationMedian: quantile(errors.map(e=>e.duration),.5) }, confusion };
}
const results = cases.map(testCase => {
  if (!["development","heldout","regression","unlabelled"].includes(testCase.split)) throw Error("Explicit evaluation split required");
  const path = realpathSync(resolve(base,testCase.recording)), data = JSON.parse(readFileSync(path,"utf8"));
  const frames = data.frames as number[][], capture = data.capture_manifest;
  if (!Array.isArray(frames)) throw Error("Recording contains no skeleton frames");
  const frozenBank = capture?.recognitionContext?.learnedActions;
  const references: Reference[] | undefined = testCase.references ?? frozenBank?.flatMap((action: {label:string;samples:Array<{sessionId:string;startMs:number;endMs:number;labelSource?:Reference["labelSource"]}>}) =>
    action.samples.map(sample => {
      const recording = testCase.referenceRecordings?.[sample.sessionId];
      if (!recording) throw Error("A reference recording is missing from the frozen bank; supply referenceRecordings by session id");
      return {...sample,label:action.label,recording};
    }));
  if (!references && !testCase.allowColdStart) throw Error("Explicit reference recordings or allowColdStart required; never reconstruct training from evaluation labels");
  const recordingHashes = new Set([hash(path)]), loaded: LoadedLearnedMotionAction[] = [];
  for (const reference of references ?? []) {
    validate(reference);
    const refPath = realpathSync(resolve(base,reference.recording)), refData = JSON.parse(readFileSync(refPath,"utf8"));
    if (refPath === path || (data.session?.id && data.session.id === refData.session?.id) || recordingHashes.has(hash(refPath))) {
      throw Error("Training/evaluation recording overlap: use separate recording sessions");
    }
    const refFrames = refData.frames.filter((f:number[]) => f[0]>=reference.startMs && f[0]<=reference.endMs);
    let action = loaded.find(a => a.label === reference.label);
    if (!action) { action = { id:reference.label, label:reference.label, samples:[],sampleSources:[] }; loaded.push(action); }
    action.samples.push(refFrames); action.sampleSources!.push(reference.labelSource);
  }
  const labels = testCase.expectedLabels ?? selfTestLabels(capture?.selfTest?.protocol);
  const startMs = capture?.selfTest?.recordingStartMs ?? frames[0]?.[0] ?? 0, endMs = frames.at(-1)?.[0] ?? 0;
  const intervals = visualActionIntervals(capture?.visualContext?.samples ?? [], {startMs,endMs});
  const segmented = segmentWorkEpisodesDetailed(frames, loaded), slices = contextualMotionSlices(frames,segmented.slices,intervals);
  const interruptions = [...detectWorkInterruptions(frames, {startMs,endMs}), ...intervals.filter(i=>i.kind === "phone" || i.kind === "interaction").map(i=>({
    kind:i.kind === "phone" ? "visual_phone_candidate" as const : "visual_interaction_candidate" as const,startMs:i.startMs,endMs:i.endMs }))];
  const predictions = slices.map(slice => {
    const motion = classifyWorkMotion(slice.frames,loaded,labels), decision = decideWorkLabel(motion,labels,slice.visualEvidence);
    return { startMs:slice.startMs,endMs:slice.endMs,label:decision.candidates[0]?.taskLabel ?? "UNCLASSIFIED",
      reviewRequired: Boolean(decision.qualityReasons.length || !hasCompleteWorkBoundaries(slice) || slice.contextNonWork
        || hasInterruptionConfound(summarizeInterruptions(interruptions,slice.startMs,slice.endMs))),
      candidates:decision.candidates, boundaryEvidence:slice.boundaryEvidence };
  });
  const baseline = testCase.baselinePredictions ? JSON.parse(readFileSync(resolve(base,testCase.baselinePredictions),"utf8")) as Prediction[] : null;
  return { name:testCase.name,split:testCase.split,recordingSha256:hash(path),
    references:(references ?? []).map(r=>({ label:r.label,recordingSha256:hash(resolve(base,r.recording)),startMs:r.startMs,endMs:r.endMs,source:r.labelSource ?? "legacy" })),
    predictions, metrics:score(testCase.expected,predictions), recordedBaseline:baseline ? { predictions:baseline,metrics:score(testCase.expected,baseline) } : null,
    trace:segmented.report };
});
const report = { version:WORK_RECOGNITION_VERSION, generatedAt:new Date().toISOString(),
  note:"Expected intervals are used only for scoring. Regression/development results are not independent accuracy estimates. No parameters are fitted by this script.",
  implementation:Object.fromEntries(["app/learned-work-segmentation.ts","app/work-motion-features.ts","app/custom-motion-training.ts","app/work-action-decision.ts"].map(path=>[path,hash(path)])),results };
writeFileSync(output,JSON.stringify(report,null,2)+"\n");
console.log(JSON.stringify({output,cases:results.length,scored:results.filter(r=>r.metrics!==null).length}));
