import { parseSessionFrame } from "./pose-store";

export type WorkPose = { time: number; values: number[]; weights: number[]; wrists: boolean[] };
export const WORK_FEATURE_VERSION = "shoulder-arms-hands-v1";
export const WORK_SAMPLE_MS = 200;

/** Identical geometry for searching boundaries and comparing the completed work. */
export function workSequence(frames: number[][], sampleMs = WORK_SAMPLE_MS): WorkPose[] {
  const result: WorkPose[] = [];
  for (const raw of frames) {
    const { body, bodyDetected, relativeTimeMs: time, leftHand, rightHand } = parseSessionFrame(raw);
    if (!bodyDetected || !Number.isFinite(time) || (result.length && (time <= result.at(-1)!.time || time - result.at(-1)!.time < sampleMs))) continue;
    const visible = (j: number) => Number.isFinite(body[j * 4]) && Number.isFinite(body[j * 4 + 1])
      && Number.isFinite(body[j * 4 + 3]) && body[j * 4 + 3] >= 0.35;
    if (![0, 1].every(visible)) continue;
    const cx = (body[0] + body[4]) / 2, cy = (body[1] + body[5]) / 2;
    const scale = Math.max(0.08, Math.hypot((body[0] - body[4]) * 16 / 9, body[1] - body[5]));
    const values: number[] = [], weights: number[] = [];
    for (const j of [2, 3, 4, 5]) {
      const valid = visible(j);
      values.push(valid ? (body[j * 4] - cx) * 16 / 9 / scale : 0, valid ? (body[j * 4 + 1] - cy) / scale : 0);
      weights.push(valid ? 1 : 0, valid ? 1 : 0);
    }
    if (weights.filter(Boolean).length < 4) continue;
    const wrists = [visible(4), visible(5)];
    values.push(values[6] - values[4], values[7] - values[5]);
    weights.push(wrists.every(Boolean) ? 0.5 : 0, wrists.every(Boolean) ? 0.5 : 0);
    for (const [side, hand] of [leftHand, rightHand].entries()) {
      const valid = wrists[side] && hand && hand.every(Number.isFinite);
      // Use the already observed fingertip geometry, never fill missing hands with a posture.
      for (const tip of [4, 8, 12, 16, 20]) {
        values.push(valid ? (hand[tip * 3] - hand[0]) * 16 / 9 / scale : 0,
          valid ? (hand[tip * 3 + 1] - hand[1]) / scale : 0);
        weights.push(valid ? 0.42 : 0, valid ? 0.42 : 0);
      }
    }
    result.push({ time, values, weights, wrists });
  }
  const velocities = result.map((pose, i) => {
    const previous = result[i - 1];
    return [4, 6].map(j => {
      if (!previous || pose.time - previous.time > 1200 || !pose.weights[j] || !previous.weights[j]) return null;
      const dt = (pose.time - previous.time) / 1000;
      return dt > 0 ? [(pose.values[j] - previous.values[j]) / dt, (pose.values[j + 1] - previous.values[j + 1]) / dt] : null;
    });
  });
  // Relative speed retains the motion pattern while allowing globally slower work.
  const speeds = velocities.flatMap(sides => sides.flatMap(v => v && Math.hypot(...v) > 0 ? [Math.hypot(...v)] : [])).sort((a, b) => a - b);
  const speedScale = speeds[Math.floor(speeds.length / 2)] ?? 1;
  result.forEach((pose, i) => {
    for (const v of velocities[i]) {
      const speed = v ? Math.hypot(...v) : 0;
      pose.values.push(v && speed ? v[0] / speed : 0, v && speed ? v[1] / speed : 0,
        v ? speed / (speed + speedScale) : 0);
      pose.weights.push(v ? 0.25 : 0, v ? 0.25 : 0, v ? 0.25 : 0);
    }
  });
  return result;
}

export function workPoseCost(a: WorkPose, b: WorkPose) {
  let observedArms = 0, weight = 0, squared = 0;
  for (let i = 0; i < a.values.length; i++) {
    const w = Math.min(a.weights[i], b.weights[i]);
    if (!(w > 0)) continue;
    if (i < 8) observedArms++;
    squared += (a.values[i] - b.values[i]) ** 2 * w; weight += w;
  }
  return observedArms >= 4 && weight > 0 ? squared / weight : Infinity;
}

export function workPoseQuality(sequence: WorkPose[]) {
  return sequence.length ? sequence.reduce((sum, p) => sum + p.weights.slice(0, 8).filter(Boolean).length / 8, 0) / sequence.length : 0;
}

export function resampleWorkSequence(sequence: WorkPose[], count = 28): WorkPose[] {
  if (!sequence.length) return [];
  const length = Math.min(count, sequence.length);
  return Array.from({ length }, (_, i) => sequence[Math.round(i * (sequence.length - 1) / Math.max(1, length - 1))]);
}

/** Complete-to-complete alignment: no duration cutoff, no filling gaps or stationary time. */
export function compareWorkSequences(a: WorkPose[], b: WorkPose[]) {
  if (a.length < 6 || b.length < 6) return Infinity;
  let previous = new Float64Array(b.length + 1).fill(Infinity); previous[0] = 0;
  let lengths = new Uint32Array(b.length + 1);
  for (let i = 1; i <= a.length; i++) {
    const row = new Float64Array(b.length + 1).fill(Infinity), nextLengths = new Uint32Array(b.length + 1);
    for (let j = 1; j <= b.length; j++) {
      let best = previous[j - 1], length = lengths[j - 1];
      if (previous[j] + 0.0025 < best) { best = previous[j] + 0.0025; length = lengths[j]; }
      if (row[j - 1] + 0.0025 < best) { best = row[j - 1] + 0.0025; length = nextLengths[j - 1]; }
      row[j] = best + workPoseCost(a[i - 1], b[j - 1]); nextLengths[j] = length + 1;
    }
    previous = row; lengths = nextLengths;
  }
  return Math.sqrt(previous[b.length] / Math.max(1, lengths[b.length]));
}

/** Quality-weighted diversity, independent of storage order and evaluation recordings. */
export function representativeWorkSamples(samples: number[][][], preferredIndexes: number[] = []): number[] {
  const usable = samples.map((frames, index) => ({ index, poses: resampleWorkSequence(workSequence(frames)) }))
    .filter(s => s.poses.length >= 6);
  if (!usable.length) return [];
  const distances = usable.map(a => usable.map(b => compareWorkSequences(a.poses, b.poses)));
  const means = distances.map(row => row.reduce((sum, d) => sum + d, 0) / row.length);
  const preferred = usable.flatMap((s, i) => preferredIndexes.includes(s.index) ? [i] : []);
  const starters = preferred.length ? preferred : usable.map((_, i) => i);
  let first = starters[0];
  for (const i of starters) if (means[i] < means[first]) first = i;
  const selected = [first];
  // Computational budget, not a fitted recognition threshold. Calibrate on separate development data.
  const budget = Math.ceil(Math.sqrt(usable.length));
  while (selected.length < budget) {
    let best = -1, bestDiversity = 0;
    for (let i = 0; i < usable.length; i++) {
      if (selected.includes(i)) continue;
      const diversity = Math.min(...selected.map(j => distances[i][j])) * workPoseQuality(usable[i].poses);
      if (Number.isFinite(diversity) && diversity > bestDiversity) { best = i; bestDiversity = diversity; }
    }
    if (best < 0) break;
    selected.push(best);
  }
  return selected.map(i => usable[i].index);
}
