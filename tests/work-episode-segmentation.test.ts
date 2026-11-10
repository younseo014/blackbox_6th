import assert from "node:assert/strict";
import test from "node:test";
import { generateEventMotion } from "../app/demo-motion.ts";
import { segmentArmWorkEpisodes, hasCompleteWorkBoundaries } from "../app/work-episode-segmentation.ts";
import { matchLearnedWorkSequences, segmentWorkEpisodes } from "../app/learned-work-segmentation.ts";

const rest = generateEventMotion("normal_task", 1)[0];
function frame(timeMs: number, left: [number, number], right: [number, number]) {
  const f = [...rest]; f[0] = timeMs;
  for (const [joint, x, y] of [[0, .4, .3], [1, .6, .3], [2, .4, .5], [3, .6, .5], [4, ...left], [5, ...right]]) {
    f[10 + joint * 4] = x; f[10 + joint * 4 + 1] = y; f[10 + joint * 4 + 3] = .95;
  }
  return f;
}
const mix = (a: [number, number], b: [number, number], amount: number): [number, number] => [a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount];
function task(offset = 0, speed = 1, mirror = false) {
  const result: number[][] = [];
  for (let i = 0; i <= Math.round(90 * speed); i++) {
    const t = i / 10 / speed;
    let left: [number, number] = [.4, .7], right: [number, number] = [.6, .7];
    if (t >= 1 && t < 2) left = mix(left, [.52, .49], t - 1);
    else if (t >= 2 && t < 7) left = [.52 + .02 * Math.sin((t - 2) * 5), .49];
    else if (t >= 7 && t < 8) left = mix([.52, .49], left, t - 7);
    if (t >= 2 && t < 3) right = mix(right, [.65, .44], t - 2);
    else if (t >= 3 && t < 6) right = [.65, .44];
    else if (t >= 6 && t < 7) right = mix([.65, .44], right, t - 6);
    if (mirror) { left[0] = .8 - left[0]; right[0] = 1.2 - right[0]; }
    result.push(frame(offset + i * 100, left, right));
  }
  return result;
}

test("work entry and return bound the task rather than the full moving recording", () => {
  const complete = segmentArmWorkEpisodes(task()).filter(hasCompleteWorkBoundaries);
  assert.equal(complete.length, 1);
  assert.ok(Math.abs(complete[0].startMs - 1000) <= 600);
  assert.ok(Math.abs(complete[0].endMs - 8000) <= 600);
});

test("an internal quiet hold is retained inside a task", () => {
  const frames = task();
  const held = frames.find(f => f[0] === 4000)!;
  const withHold = [...frames.filter(f => f[0] <= 4000), ...Array.from({ length: 50 }, (_, i) => { const f = [...held]; f[0] += (i + 1) * 100; return f; }),
    ...frames.filter(f => f[0] > 4000).map(f => { const next = [...f]; next[0] += 5000; return next; })];
  const complete = segmentArmWorkEpisodes(withHold).filter(hasCompleteWorkBoundaries);
  assert.equal(complete.length, 1);
  assert.ok(complete[0].startMs < 2000 && complete[0].endMs > 12000);
});

test("a task already under way when tracking begins is not counted as fully observed", () => {
  const slices = segmentArmWorkEpisodes(task().filter(f => f[0] >= 3000));
  assert.equal(slices.filter(hasCompleteWorkBoundaries).length, 0);
  assert.ok(slices.some(s => !s.boundaryEvidence?.startObserved));
});

test("a recording stopped during handling has no invented completion", () => {
  const slices = segmentArmWorkEpisodes(task().filter(f => f[0] <= 5000));
  assert.equal(slices.filter(hasCompleteWorkBoundaries).length, 0);
  assert.ok(slices.some(s => !s.boundaryEvidence?.endObserved));
});

test("tracking absence cannot supply a completion or a pause", () => {
  const input = task().filter(f => f[0] < 3500 || f[0] > 5200);
  const slices = segmentArmWorkEpisodes(input);
  assert.equal(slices.length, 1);
  assert.equal(slices.filter(hasCompleteWorkBoundaries).length, 0);
  assert.ok((slices[0].boundaryEvidence?.trackingGapMs ?? 0) > 1200);
});

test("two reach-handle-return cycles are two tasks even without a long still interval", () => {
  const input = [...task(), ...task(9100)];
  assert.equal(segmentArmWorkEpisodes(input).filter(hasCompleteWorkBoundaries).length, 2);
});

test("sequence matching locates a learned task inside unrelated preparation and tail movement", () => {
  const reference = task().filter(f => f[0] >= 1000 && f[0] <= 8000);
  const input = [...task(0, 1, true), ...task(15000), ...task(28000, 1, true)];
  const matches = matchLearnedWorkSequences(input, [{ id: "mix", label: "혼합", samples: [reference] }]);
  assert.equal(matches.length, 1);
  assert.ok(Math.abs(matches[0].startMs - 16000) <= 600);
  assert.ok(Math.abs(matches[0].endMs - 23000) <= 600);
});

test("sequence completion tolerates a different execution speed", () => {
  const reference = task().filter(f => f[0] >= 1000 && f[0] <= 8000);
  const matches = matchLearnedWorkSequences(task(5000, 1.5), [{ id: "mix", label: "혼합", samples: [reference] }]);
  assert.equal(matches.length, 1);
  assert.ok(Math.abs(matches[0].startMs - 6500) <= 800);
  assert.ok(Math.abs(matches[0].endMs - 17000) <= 800);
});

test("a partial action and idle posture cannot satisfy the complete learned sequence", () => {
  const reference = task().filter(f => f[0] >= 1000 && f[0] <= 8000);
  const actions = [{ id: "mix", label: "혼합", samples: [reference] }];
  assert.equal(matchLearnedWorkSequences(task().filter(f => f[0] <= 4500), actions).length, 0);
  const idle = Array.from({ length: 200 }, (_, i) => frame(i * 100, [.4, .7], [.6, .7]));
  assert.equal(matchLearnedWorkSequences(idle, actions).length, 0);
});

test("identical references with different names remain label-ambiguous", () => {
  const reference = task().filter(f => f[0] >= 1000 && f[0] <= 8000);
  const matches = matchLearnedWorkSequences(task(), [{ id: "a", label: "혼합", samples: [reference] }, { id: "b", label: "닦기", samples: [reference] }]);
  assert.ok(matches.length > 0);
  assert.equal(matches[0].boundaryEvidence?.margin, 0);
});

test("a stationary reference cannot label arbitrary idle time as completed work", () => {
  const idle = Array.from({ length: 200 }, (_, i) => frame(i * 100, [.4, .7], [.6, .7]));
  assert.equal(matchLearnedWorkSequences(idle, [{ id: "bad", label: "혼합", samples: [idle.slice(0, 50)] }]).length, 0);
});

test("adding a learned sequence replaces an overlapping broad movement candidate", () => {
  const reference = task().filter(f => f[0] >= 1000 && f[0] <= 8000);
  const slices = segmentWorkEpisodes(task(), [{ id: "a", label: "혼합", samples: [reference] }]);
  assert.equal(slices.length, 1);
  assert.equal(slices[0].boundaryEvidence?.method, "learned_sequence");
});

test("sequence alignment preserves an interruption instead of cutting it into two tasks", () => {
  const reference = task().filter(f => f[0] >= 1000 && f[0] <= 8000);
  const input = task(), hold = input.find(f => f[0] === 4000)!;
  const observed = [...input.filter(f => f[0] <= 4000), ...Array.from({ length: 40 }, (_, i) => { const f = [...hold]; f[0] += (i + 1) * 100; return f; }),
    ...input.filter(f => f[0] > 4000).map(f => { const next = [...f]; next[0] += 4000; return next; })];
  const matches = matchLearnedWorkSequences(observed, [{ id: "a", label: "혼합", samples: [reference] }]);
  assert.equal(matches.length, 1);
  assert.ok(matches[0].startMs < 2000 && matches[0].endMs > 11000);
});

test("normalised sequence boundaries survive camera translation and scale", () => {
  const reference = task().filter(f => f[0] >= 1000 && f[0] <= 8000);
  const observed = task(17000).map(raw => {
    const f = [...raw];
    for (let joint = 0; joint < 22; joint++) {
      f[10 + joint * 4] = (f[10 + joint * 4] - .5) * 1.25 + .62;
      f[10 + joint * 4 + 1] = (f[10 + joint * 4 + 1] - .5) * 1.25 + .55;
    }
    return f;
  });
  const matches = matchLearnedWorkSequences(observed, [{ id: "a", label: "혼합", samples: [reference] }]);
  assert.equal(matches.length, 1);
  assert.ok(Math.abs(matches[0].startMs - 18000) <= 600 && Math.abs(matches[0].endMs - 25000) <= 600);
});

test("multiple learned task completions remain separate across background time", () => {
  const reference = task().filter(f => f[0] >= 1000 && f[0] <= 8000);
  const observed = [...task(), ...task(22000)];
  const matches = matchLearnedWorkSequences(observed, [{ id: "a", label: "혼합", samples: [reference] }]);
  assert.equal(matches.length, 2);
  assert.ok(matches[0].endMs < matches[1].startMs);
});

test("hidden wrists cannot prove a learned task's start and completion", () => {
  const reference = task().filter(f => f[0] >= 1000 && f[0] <= 8000);
  const observed = task().map(raw => { const f = [...raw]; f[10 + 4 * 4 + 3] = 0; f[10 + 5 * 4 + 3] = 0; return f; });
  assert.equal(matchLearnedWorkSequences(observed, [{ id: "a", label: "혼합", samples: [reference] }]).length, 0);
});

test("a task crossing a search window is located once with its full completion", () => {
  const reference = task().filter(f => f[0] >= 1000 && f[0] <= 8000);
  const prefix = Array.from({ length: 590 }, (_, i) => frame(i * 100, [.4, .7], [.6, .7]));
  const matches = matchLearnedWorkSequences([...prefix, ...task(59000)], [{ id: "a", label: "혼합", samples: [reference] }]);
  assert.equal(matches.length, 1);
  assert.ok(Math.abs(matches[0].startMs - 60000) <= 600 && Math.abs(matches[0].endMs - 67000) <= 600);
});
