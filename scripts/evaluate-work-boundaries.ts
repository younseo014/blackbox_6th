import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { sliceTargetMotions, SELF_TEST_MOTION_SEGMENTATION, type TargetMotionSlice } from "../app/motion-segmentation.ts";
import { segmentWorkEpisodes } from "../app/learned-work-segmentation.ts";
import { hasCompleteWorkBoundaries } from "../app/work-episode-segmentation.ts";

type Interval = { startMs: number; endMs: number };
type Case = { name: string; recording: string; split: "development" | "heldout" | "unlabelled"; expected: Interval[] | null };
const args = process.argv.slice(2);
const manifest = args[args.indexOf("--manifest") + 1];
if (!args.includes("--manifest") || !manifest) throw new Error("Use --manifest cases.json [--output report-prefix]");
const output = args.includes("--output") ? args[args.indexOf("--output") + 1] : "docs/work-boundary-evaluation";
const cases = JSON.parse(readFileSync(manifest, "utf8")) as Case[];
const iou = (a: Interval, b: Interval) => Math.max(0, Math.min(a.endMs, b.endMs) - Math.max(a.startMs, b.startMs))
  / Math.max(1, Math.max(a.endMs, b.endMs) - Math.min(a.startMs, b.startMs));
function metrics(expected: Interval[] | null, candidates: TargetMotionSlice[]) {
  if (expected === null) return null;
  const pairs = expected.flatMap((truth, truthIndex) => candidates.map((candidate, candidateIndex) => ({ truthIndex, candidateIndex, iou: iou(truth, candidate) })))
    .sort((a, b) => b.iou - a.iou);
  const usedTruth = new Set<number>(), usedPred = new Set<number>();
  const matched: Array<{ iou: number; startErrorSeconds: number; endErrorSeconds: number }> = [];
  for (const pair of pairs) {
    if (pair.iou < 0.5 || usedTruth.has(pair.truthIndex) || usedPred.has(pair.candidateIndex)) continue;
    usedTruth.add(pair.truthIndex); usedPred.add(pair.candidateIndex);
    const truth = expected[pair.truthIndex], prediction = candidates[pair.candidateIndex];
    matched.push({ iou: pair.iou, startErrorSeconds: Math.abs(truth.startMs - prediction.startMs) / 1000,
      endErrorSeconds: Math.abs(truth.endMs - prediction.endMs) / 1000 });
  }
  const truePositive = matched.length, falsePositive = candidates.length - truePositive, falseNegative = expected.length - truePositive;
  return { truePositive, falsePositive, falseNegative,
    precision: candidates.length ? truePositive / candidates.length : 0,
    recall: expected.length ? truePositive / expected.length : null,
    f1: (2 * truePositive + falsePositive + falseNegative) ? 2 * truePositive / (2 * truePositive + falsePositive + falseNegative) : 1,
    matched };
}
const results = cases.map(testCase => {
  const dataset = JSON.parse(readFileSync(testCase.recording, "utf8"));
  const frames = dataset.frames as number[][];
  // Ground truth is used only below for scoring, never passed to either segmenter.
  const baseline = sliceTargetMotions(frames, SELF_TEST_MOTION_SEGMENTATION);
  const proposed = segmentWorkEpisodes(frames);
  const complete = proposed.filter(hasCompleteWorkBoundaries);
  const brief = (s: TargetMotionSlice) => ({ startMs: s.startMs, endMs: s.endMs, boundaryEvidence: s.boundaryEvidence });
  return { name: testCase.name, split: testCase.split, frameCount: frames.length, expected: testCase.expected,
    baseline: { candidates: baseline.map(brief), metrics: metrics(testCase.expected, baseline) },
    proposed: { candidates: proposed.map(brief), allCandidateMetrics: metrics(testCase.expected, proposed),
      completeMetrics: metrics(testCase.expected, complete), completeCount: complete.length, incompleteCount: proposed.length - complete.length } };
});
const hash = (file: string) => createHash("sha256").update(readFileSync(file)).digest("hex");
const report = { generatedAt: new Date().toISOString(), implementations: {
  baseline: hash("app/motion-segmentation.ts"), lifecycle: hash("app/work-episode-segmentation.ts"), sequence: hash("app/learned-work-segmentation.ts"),
}, note: "Development-case regression, not population accuracy. No reference taken from ground truth. All candidates and incomplete observations are reported separately.", results };
writeFileSync(`${output}.json`, JSON.stringify(report, null, 2) + "\n");
const rows = results.map(r => {
  const oldMatch = r.baseline.metrics?.matched[0], nextMatch = r.proposed.completeMetrics?.matched[0];
  return `| ${r.name} | ${r.split} | ${r.baseline.candidates.length} | ${r.proposed.candidates.length} | ${r.proposed.completeCount} | ${r.proposed.incompleteCount} | ${oldMatch ? (oldMatch.iou * 100).toFixed(2) : "n/a"} | ${nextMatch ? (nextMatch.iou * 100).toFixed(2) : "n/a"} |`;
});
writeFileSync(`${output}.md`, `# Work boundary evaluation\n\n${report.note}\n\n| Case | Split | Old candidates | New all candidates | New complete | New incomplete | Old matched IoU % | New complete IoU % |\n|---|---|---:|---:|---:|---:|---:|---:|\n${rows.join("\n")}\n\nSee JSON for false positives, misses and boundary errors. Unlabelled recordings are not scored.\n`);
console.log(JSON.stringify(report, null, 2));
