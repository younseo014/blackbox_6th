import fs from "node:fs";
import path from "node:path";
import { classifyLearnedMotion } from "../app/custom-motion-training.ts";

type DatasetAction =
  | "walk"
  | "sitDown"
  | "standUp"
  | "pickUp"
  | "carry"
  | "throw"
  | "push"
  | "pull"
  | "waveHands"
  | "clapHands";

type PrimitiveLabel =
  | "WALK"
  | "SIT"
  | "STAND"
  | "LIFT"
  | "CARRY"
  | "TRANSFER"
  | "PUSH"
  | "PULL"
  | "REPETITIVE_ARM";

type LabelRange = { action: DatasetAction; start: number; end: number };
type Clip = {
  id: string;
  subject: number;
  exercise: number;
  sourceAction: DatasetAction;
  label: PrimitiveLabel;
  frames: number[][];
};

const ROOT = process.cwd();
const DATASET_DIR = path.join(ROOT, "data/external/utkinect");
const LABEL_PATH = path.join(DATASET_DIR, "actionLabel.txt");
const MODEL_PATH = path.join(ROOT, "models/utkinect-primitives-v1.json");
const REPORT_PATH = path.join(ROOT, "docs/utkinect-training-report.md");
const ACTION_TO_PRIMITIVE: Record<DatasetAction, PrimitiveLabel> = {
  walk: "WALK",
  sitDown: "SIT",
  standUp: "STAND",
  pickUp: "LIFT",
  carry: "CARRY",
  throw: "TRANSFER",
  push: "PUSH",
  pull: "PULL",
  waveHands: "REPETITIVE_ARM",
  clapHands: "REPETITIVE_ARM",
};
const KINECT_TO_APP = [4, 8, 5, 9, 6, 10, 7, 11, 7, 11, 7, 11, 12, 16, 13, 17, 14, 18, 14, 18, 15, 19];
const FRAME_STRIDE = 10 + 22 * 4 + 21 * 2 * 3;
const FIVE_SHOT_SAMPLES = 5;

function parseLabels(text: string) {
  const labels = new Map<string, LabelRange[]>();
  let sequence = "";
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    if (/^s\d+_e\d+$/.test(line)) {
      sequence = line;
      labels.set(sequence, []);
      continue;
    }
    const match = line.match(/^(\w+):\s+(\S+)\s+(\S+)$/);
    if (!match || !sequence || match[2] === "NaN" || match[3] === "NaN") continue;
    labels.get(sequence)!.push({
      action: match[1] as DatasetAction,
      start: Number(match[2]),
      end: Number(match[3]),
    });
  }
  return labels;
}

function readSkeleton(sequence: string) {
  const filePath = path.join(DATASET_DIR, "joints", `joints_${sequence}.txt`);
  return fs.readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => line.trim().split(/\s+/).map(Number))
    .filter((row) => row.length >= 61 && row.every(Number.isFinite));
}

function toAppFrame(row: number[], frameIndex: number) {
  const frame = new Array(FRAME_STRIDE).fill(0);
  frame[0] = frameIndex * (1000 / 15);
  frame[1] = 1;
  frame[2] = 1;
  for (let appIndex = 0; appIndex < KINECT_TO_APP.length; appIndex += 1) {
    const sourceIndex = KINECT_TO_APP[appIndex];
    const sourceOffset = 1 + sourceIndex * 3;
    const targetOffset = 10 + appIndex * 4;
    const depth = Math.max(0.1, row[sourceOffset + 2]);
    // Approximate the RGB camera's perspective view from Kinect world-space
    // coordinates. This is much closer to MediaPipe's normalized image input
    // than treating metric X/Y as if they already were image coordinates.
    frame[targetOffset] = row[sourceOffset] / depth;
    frame[targetOffset + 1] = -row[sourceOffset + 1] / depth;
    frame[targetOffset + 2] = row[sourceOffset + 2];
    frame[targetOffset + 3] = 1;
  }
  return frame;
}

function loadClips() {
  const labels = parseLabels(fs.readFileSync(LABEL_PATH, "utf8"));
  const clips: Clip[] = [];
  for (const [sequence, ranges] of labels) {
    const match = sequence.match(/^s(\d+)_e(\d+)$/)!;
    const subject = Number(match[1]);
    const exercise = Number(match[2]);
    const rows = readSkeleton(sequence);
    for (const range of ranges) {
      const selected = rows.filter((row) => row[0] >= range.start && row[0] <= range.end);
      if (selected.length < 6) continue;
      clips.push({
        id: `${sequence}_${range.action}`,
        subject,
        exercise,
        sourceAction: range.action,
        label: ACTION_TO_PRIMITIVE[range.action],
        frames: selected.map(toAppFrame),
      });
    }
  }
  return clips;
}

function groupTrainingSamples(clips: Clip[], limitPerClass?: number) {
  const groups = new Map<PrimitiveLabel, Clip[]>();
  for (const clip of clips) {
    const group = groups.get(clip.label) ?? [];
    if (limitPerClass === undefined || group.length < limitPerClass) group.push(clip);
    groups.set(clip.label, group);
  }
  return [...groups.entries()].map(([label, samples]) => ({
    id: label,
    label,
    samples: samples.map((sample) => sample.frames),
  }));
}

function evaluate(test: Clip[], training: Clip[], limitPerClass?: number) {
  const actions = groupTrainingSamples(training, limitPerClass);
  const labels = [...new Set(test.map((clip) => clip.label))].sort();
  const confusion = Object.fromEntries(labels.map((actual) => [actual, Object.fromEntries(labels.map((predicted) => [predicted, 0]))]));
  let correct = 0;
  let matched = 0;
  for (const clip of test) {
    const result = classifyLearnedMotion(clip.frames, actions);
    const predicted = result.label as PrimitiveLabel;
    if (predicted === clip.label) correct += 1;
    if (result.status === "matched") matched += 1;
    confusion[clip.label][predicted] += 1;
  }
  const perClass = Object.fromEntries(labels.map((label) => {
    const total = test.filter((clip) => clip.label === label).length;
    return [label, confusion[label][label] / total];
  }));
  return {
    accuracy: correct / test.length,
    matchedRate: matched / test.length,
    correct,
    total: test.length,
    perClass,
    confusion,
  };
}

function rounded(value: number) {
  return Math.round(value * 1000) / 1000;
}

function main() {
  const clips = loadClips();
  const training = clips.filter((clip) => clip.subject % 2 === 1);
  const test = clips.filter((clip) => clip.subject % 2 === 0);
  const oneShot = evaluate(test, training, 1);
  const fiveShot = evaluate(test, training, FIVE_SHOT_SAMPLES);
  const allSamples = evaluate(test, training);
  const labels = Object.keys(allSamples.perClass).sort();
  const selected = new Map<PrimitiveLabel, Clip[]>();
  for (const clip of training) {
    const group = selected.get(clip.label) ?? [];
    group.push(clip);
    selected.set(clip.label, group);
  }
  const model = {
    schemaVersion: 1,
    name: "UTKinect café-work primitive bootstrap",
    source: "UTKinect-Action3D",
    sourceUrl: "https://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html",
    trainedAt: new Date().toISOString(),
    split: { trainingSubjects: [1, 3, 5, 7, 9], testSubjects: [2, 4, 6, 8, 10] },
    projection: "Kinect 3D skeleton projected to app-compatible x/y skeleton; shoulder-normalized DTW",
    trainingSamplesPerClass: Object.fromEntries([...selected.entries()].map(([label, samples]) => [label, samples.length])),
    classes: [...selected.entries()].map(([label, samples]) => ({
      label,
      samples: samples.map((sample) => ({
        id: sample.id,
        sourceAction: sample.sourceAction,
        frames: sample.frames,
      })),
    })),
    evaluation: {
      oneShot: { accuracy: rounded(oneShot.accuracy), matchedRate: rounded(oneShot.matchedRate) },
      fiveShot: { accuracy: rounded(fiveShot.accuracy), matchedRate: rounded(fiveShot.matchedRate) },
      allSamples: { accuracy: rounded(allSamples.accuracy), matchedRate: rounded(allSamples.matchedRate) },
      perClass: Object.fromEntries(Object.entries(allSamples.perClass).map(([key, value]) => [key, rounded(value)])),
      confusion: allSamples.confusion,
    },
  };
  fs.mkdirSync(path.dirname(MODEL_PATH), { recursive: true });
  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  fs.writeFileSync(MODEL_PATH, `${JSON.stringify(model)}\n`);
  const classRows = labels.map((label) => `| ${label} | ${(oneShot.perClass[label] * 100).toFixed(1)}% | ${(allSamples.perClass[label] * 100).toFixed(1)}% |`).join("\n");
  const report = `# UTKinect 1차 스켈레톤 학습 결과\n\n- 데이터: UTKinect-Action3D, 유효 클립 ${clips.length}개\n- 분할: 홀수 피험자 5명 학습 / 짝수 피험자 5명 평가 (사람 중복 없음)\n- 입력: 앱과 동일한 12개 전신 관절, 28프레임 정규화, 좌우 반전 허용 DTW\n- 클래스: ${labels.length}개 업무 원시동작\n- 저장 모델: 학습 피험자의 유효 표본 전체 사용\n\n## 성능\n\n| 설정 | 정확도 | 신뢰도 기준 통과율 |\n|---|---:|---:|\n| 1-shot 기준선 | ${(oneShot.accuracy * 100).toFixed(1)}% (${oneShot.correct}/${oneShot.total}) | ${(oneShot.matchedRate * 100).toFixed(1)}% |\n| 5-shot 중간 확인 | ${(fiveShot.accuracy * 100).toFixed(1)}% (${fiveShot.correct}/${fiveShot.total}) | ${(fiveShot.matchedRate * 100).toFixed(1)}% |\n| 전체 표본 1차 모델 | ${(allSamples.accuracy * 100).toFixed(1)}% (${allSamples.correct}/${allSamples.total}) | ${(allSamples.matchedRate * 100).toFixed(1)}% |\n\n정확도 변화: **${(oneShot.accuracy * 100).toFixed(1)}% → ${(allSamples.accuracy * 100).toFixed(1)}% (${((allSamples.accuracy - oneShot.accuracy) * 100).toFixed(1)}%p)**\n\n## 클래스별 정확도\n\n| 원시동작 | 1-shot | 전체 표본 모델 |\n|---|---:|---:|\n${classRows}\n\n전체 표본 모델의 혼동행렬은 모델 JSON의 \`evaluation\`에 저장했다.\n\n## 해석 제한\n\n이 수치는 UTKinect 내부의 피험자 독립 평가다. 카페 현장의 원거리 RGB 카메라, 가림, 물체 상호작용에 대한 실성능을 뜻하지 않는다. 외부 3D 스켈레톤을 앱의 2D 공통 관절로 투영했으므로, 로컬 카메라 검증셋으로 재보정해야 한다. 특히 CARRY와 LIFT는 현장 적용 전에 추가 데이터가 필요하다.\n`;
  fs.writeFileSync(REPORT_PATH, report);
  console.log(JSON.stringify({ clips: clips.length, training: training.length, test: test.length, oneShot, fiveShot, allSamples }, null, 2));
}

main();
