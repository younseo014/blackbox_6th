import { parseSessionFrame } from "./pose-store";

export type MovementCycle = {
  startMs: number;
  endMs: number;
  durationSeconds: number;
  range: number;
  hand: "left" | "right";
  axis: "x" | "y";
};

type Sample = { timeMs: number; x: number; y: number };

function handSamples(frames: number[][], hand: "left" | "right"): Sample[] {
  const wristIndex = hand === "left" ? 4 : 5;
  const shoulderIndex = hand === "left" ? 0 : 1;
  return frames.flatMap((raw) => {
    const frame = parseSessionFrame(raw);
    const body = frame.body;
    if ((body[wristIndex * 4 + 3] ?? 0) < 0.5 || (body[shoulderIndex * 4 + 3] ?? 0) < 0.5) return [];
    const x = body[wristIndex * 4] - body[shoulderIndex * 4];
    const y = body[wristIndex * 4 + 1] - body[shoulderIndex * 4 + 1];
    return Number.isFinite(x) && Number.isFinite(y)
      ? [{ timeMs: frame.relativeTimeMs, x, y }]
      : [];
  });
}

function percentileRange(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length * 0.9)] - sorted[Math.floor(sorted.length * 0.1)];
}

function cyclesInRun(samples: Sample[], hand: "left" | "right"): MovementCycle[] {
  if (samples.length < 7) return [];
  const axis: "x" | "y" = percentileRange(samples.map((item) => item.x)) >= percentileRange(samples.map((item) => item.y)) ? "x" : "y";
  const values = samples.map((_, index) => {
    const from = Math.max(0, index - 1);
    const to = Math.min(samples.length - 1, index + 1);
    let sum = 0;
    for (let cursor = from; cursor <= to; cursor += 1) sum += samples[cursor][axis];
    return sum / (to - from + 1);
  });
  const turns: Array<{ index: number; kind: -1 | 1 }> = [];
  for (let index = 1; index < values.length - 1; index += 1) {
    const previous = values[index] - values[index - 1];
    const next = values[index + 1] - values[index];
    if (previous > 0 && next <= 0) turns.push({ index, kind: 1 });
    if (previous < 0 && next >= 0) turns.push({ index, kind: -1 });
  }
  const cycles: MovementCycle[] = [];
  for (let index = 0; index + 2 < turns.length;) {
    const first = turns[index];
    const middle = turns[index + 1];
    const last = turns[index + 2];
    const firstRange = Math.abs(values[first.index] - values[middle.index]);
    const secondRange = Math.abs(values[middle.index] - values[last.index]);
    const durationSeconds = (samples[last.index].timeMs - samples[first.index].timeMs) / 1000;
    if (first.kind === last.kind && firstRange >= 0.035 && secondRange >= 0.035
      && durationSeconds >= 0.5 && durationSeconds <= 5) {
      cycles.push({
        startMs: samples[first.index].timeMs,
        endMs: samples[last.index].timeMs,
        durationSeconds,
        range: Math.min(firstRange, secondRange),
        hand,
        axis,
      });
      index += 2;
    } else {
      index += 1;
    }
  }
  return cycles;
}

/** Conservative full out-and-back wrist cycles within one already identified work episode. */
export function detectMovementCycles(frames: number[][]): MovementCycle[] {
  const byHand = (["left", "right"] as const).map((hand) => {
    const samples = handSamples(frames, hand);
    const runs: Sample[][] = [];
    for (const sample of samples) {
      const run = runs[runs.length - 1];
      if (!run || sample.timeMs - run[run.length - 1].timeMs > 300) runs.push([sample]);
      else run.push(sample);
    }
    return runs.flatMap((run) => cyclesInRun(run, hand));
  });
  return byHand[1].length >= byHand[0].length ? byHand[1] : byHand[0];
}
