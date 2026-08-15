"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  approximateHeadCenterFromShoulders,
  computeHandMotionVariability,
  computeMovementSmoothness,
  type Point3D,
} from "./motion-analysis";
import {
  MOTION_SAMPLE_RATE,
  getSessionFrames,
  parseSessionFrame,
  type ParsedMotionFrame,
} from "./pose-store";
import { drawMotionSkeleton } from "./skeleton-draw";
import type { DemoDetectionExplanation } from "./demo-personas";
import { detectMotionEvents, type DetectedMotionEvent } from "./motion-detection";
import type { ObservationMode } from "./observation-engine";
import { saveAnalysisFeedback } from "./observation-store";

const ROLLING_WINDOW = 10;
const CANVAS_WIDTH = 640;
const CANVAS_HEIGHT = 360;
const SPARKLINE_WIDTH = 640;
const SPARKLINE_HEIGHT = 40;

function wristPoint(frame: ParsedMotionFrame, hand: "left" | "right"): Point3D | null {
  const source = hand === "left" ? frame.leftHand : frame.rightHand;
  return source ? { x: source[0], y: source[1], z: source[2] } : null;
}

function primaryHandPoint(frame: ParsedMotionFrame): Point3D | null {
  return wristPoint(frame, "right") ?? wristPoint(frame, "left");
}

function formatRelativeTime(ms: number) {
  return `${(Math.max(0, ms) / 1000).toFixed(1)}초`;
}

function humanizeDetection(event: DetectedMotionEvent) {
  const duration = Math.max(0, (event.endMs - event.startMs) / 1000);
  if (event.type === "double_check") {
    return {
      badge: "반복 확인 2회",
      title: "같은 동작을 다시 확인했어요",
      description: "손을 뻗었다가 제자리로 돌아온 뒤 다시 손을 뻗는 동작이 2회 감지됐어요.",
      tone: "warning",
      timeline: ["손 뻗기", "제자리로 복귀", "다시 손 뻗기"],
    };
  }
  if (event.type === "micro_delay") {
    return {
      badge: `움직임 멈춤 ${duration.toFixed(1)}초`,
      title: "업무 흐름이 잠시 멈췄어요",
      description: `움직이던 손이 ${duration.toFixed(1)}초 동안 멈춘 뒤 다시 이어졌어요. 한 번의 장면만으로 판단하지 않고 반복되는지 살펴봐요.`,
      tone: "notice",
      timeline: ["업무 진행", `잠시 멈춤 ${duration.toFixed(1)}초`, "업무 재개"],
    };
  }
  return {
    badge: "급한 반응 동작",
    title: "속도가 갑자기 달라졌어요",
    description: "손 움직임의 속도가 짧은 순간 크게 달라졌어요. 주변 상황과 안전 이벤트를 함께 확인해요.",
    tone: "notice",
    timeline: ["평소 움직임", "갑작스러운 변화", "이후 동작 확인"],
  };
}

export type SessionReplaySource =
  | { kind: "recorded"; sessionId: string }
  | { kind: "synthetic"; frames: number[][] };

export function SessionReplayPanel({
  source,
  sessionLabel,
  detectionExplanation,
  feedbackEventId,
  observationMode = "analysis",
  baselineVersion = 1,
  onClose,
}: {
  source: SessionReplaySource;
  sessionLabel?: string;
  detectionExplanation?: DemoDetectionExplanation;
  feedbackEventId?: string;
  observationMode?: ObservationMode;
  baselineVersion?: number;
  onClose: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sparklineRef = useRef<HTMLCanvasElement>(null);
  const [frames, setFrames] = useState<ParsedMotionFrame[] | null>(() =>
    source.kind === "synthetic" ? source.frames.map(parseSessionFrame) : null,
  );
  const [loadFailed, setLoadFailed] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [feedback, setFeedback] = useState<"accurate" | "false_positive" | null>(null);
  const [showFeedbackReasons, setShowFeedbackReasons] = useState(false);

  useEffect(() => {
    if (source.kind !== "recorded") return;
    let cancelled = false;
    getSessionFrames(source.sessionId)
      .then((rawFrames) => { if (!cancelled) setFrames(rawFrames.map(parseSessionFrame)); })
      .catch(() => { if (!cancelled) setLoadFailed(true); });
    return () => { cancelled = true; };
  }, [source]);

  const primaryPoints = useMemo(() => (frames ? frames.map(primaryHandPoint) : []), [frames]);
  const variabilitySeries = useMemo(() => primaryPoints.map((_, index) => {
    const start = Math.max(0, index - ROLLING_WINDOW + 1);
    return computeHandMotionVariability(primaryPoints.slice(start, index + 1)) ?? 0;
  }), [primaryPoints]);
  const smoothnessSeries = useMemo(() => primaryPoints.map((_, index) => {
    const start = Math.max(0, index - ROLLING_WINDOW + 1);
    return computeMovementSmoothness(primaryPoints.slice(start, index + 1)) ?? 0;
  }), [primaryPoints]);

  const detectedEvents = useMemo(() => {
    if (detectionExplanation) return detectionExplanation.detected;
    if (!frames) return [];
    return detectMotionEvents(frames.map((frame) => ({
      point: primaryHandPoint(frame),
      relativeTimeMs: frame.relativeTimeMs,
    })));
  }, [detectionExplanation, frames]);

  const analysis = useMemo(() => {
    const event = detectedEvents[0];
    return event ? humanizeDetection(event) : {
      badge: "평소 흐름",
      title: "특별한 변화가 보이지 않아요",
      description: "이 장면에서는 반복 확인이나 긴 멈춤에 해당하는 동작이 감지되지 않았어요.",
      tone: "stable",
      timeline: ["업무 시작", "자연스러운 움직임", "업무 이어짐"],
    };
  }, [detectedEvents]);

  const motionTags = useMemo(() => {
    let totalDistance = 0;
    let active = 0;
    let directionChanges = 0;
    let previousDx: number | null = null;
    for (let index = 1; index < primaryPoints.length; index += 1) {
      const previous = primaryPoints[index - 1];
      const current = primaryPoints[index];
      if (!previous || !current) continue;
      const dx = current.x - previous.x;
      const step = Math.hypot(dx, current.y - previous.y, current.z - previous.z);
      totalDistance += step;
      if (step >= 0.004) active += 1;
      if (previousDx !== null && Math.abs(dx) > 0.002 && Math.sign(dx) !== Math.sign(previousDx)) directionChanges += 1;
      if (Math.abs(dx) > 0.002) previousDx = dx;
    }
    const activeRatio = primaryPoints.length > 1 ? active / (primaryPoints.length - 1) : 0;
    return {
      speed: activeRatio > 0.6 ? "빠름" : activeRatio < 0.2 ? "느림" : "보통",
      range: totalDistance > 1.5 ? "큼" : totalDistance > 0.4 ? "보통" : "작음",
      rhythm: directionChanges >= 4 ? "반복됨" : detectedEvents.some((event) => event.type === "micro_delay") ? "중단 후 재개" : "자연스러움",
    };
  }, [primaryPoints, detectedEvents]);

  useEffect(() => {
    if (!playing || !frames || frames.length < 2) return;
    let cancelled = false;
    let timeoutId: number;
    const scheduleNext = (delayMs: number) => { timeoutId = window.setTimeout(tick, Math.max(20, delayMs / playbackRate)); };
    function tick() {
      if (cancelled) return;
      setFrameIndex((current) => {
        const next = current + 1;
        if (next >= frames!.length) { setPlaying(false); return current; }
        const delta = frames![next].relativeTimeMs - frames![current].relativeTimeMs;
        scheduleNext(delta > 0 ? delta : 1000 / MOTION_SAMPLE_RATE);
        return next;
      });
    }
    scheduleNext(1000 / MOTION_SAMPLE_RATE);
    return () => { cancelled = true; window.clearTimeout(timeoutId); };
  }, [playing, frames, playbackRate]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const frame = frames?.[frameIndex];
    if (!canvas || !frame) return;
    const center = approximateHeadCenterFromShoulders(frame.body);
    const head = frame.head && center ? { ...frame.head, centerX: center.x, centerY: center.y } : null;
    drawMotionSkeleton(canvas, frame.body, frame.leftHand, frame.rightHand, head, frame.fullBodyVisible);
  }, [frames, frameIndex]);

  useEffect(() => {
    const canvas = sparklineRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context || variabilitySeries.length === 0) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const max = Math.max(...variabilitySeries, 0.0001);
    const width = canvas.width / variabilitySeries.length;
    variabilitySeries.forEach((value, index) => {
      const height = Math.max(1, (value / max) * canvas.height);
      context.fillStyle = index === frameIndex ? "#2f6e4f" : "#c7ddc9";
      context.fillRect(index * width, canvas.height - height, Math.max(1, width - 0.5), height);
    });
  }, [variabilitySeries, frameIndex]);

  const currentFrame = frames?.[frameIndex] ?? null;
  const currentVariability = variabilitySeries[frameIndex];
  const currentSmoothness = smoothnessSeries[frameIndex];

  async function submitFeedback(
    verdict: "accurate" | "false_positive",
    reason?: "interaction" | "rest" | "wrong_task" | "camera_error" | "other",
  ) {
    const eventId = feedbackEventId ?? (source.kind === "recorded" ? source.sessionId : sessionLabel ?? "synthetic-demo");
    setFeedback(verdict);
    setShowFeedbackReasons(false);
    await saveAnalysisFeedback({
      id: `feedback-${crypto.randomUUID()}`,
      eventId,
      mode: observationMode,
      verdict,
      reason,
      baselineVersion,
      createdAt: Date.now(),
    });
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="replay-modal replay-modal-premium" role="dialog" aria-modal="true" aria-labelledby="replay-title" onMouseDown={(event) => event.stopPropagation()}>
        <button className="modal-close" type="button" onClick={onClose} aria-label="닫기">×</button>
        <header className="replay-header">
          <span className={`replay-source-chip ${source.kind}`}>{source.kind === "synthetic" ? "가상 시나리오" : "좌표 기록"}</span>
          <h2 id="replay-title">{sessionLabel ?? "기록된 동작"}</h2>
          <p>{source.kind === "synthetic" ? "설명을 위해 만든 예시 동작입니다." : "영상·음성 없이 저장된 몸과 손 좌표를 재생합니다."}</p>
        </header>

        {loadFailed && <p className="replay-empty">이 세션의 좌표 기록을 불러오지 못했어요.</p>}
        {!loadFailed && frames && frames.length === 0 && <p className="replay-empty">이 세션에는 저장된 좌표 프레임이 없어요.</p>}
        {!loadFailed && frames === null && <p className="replay-empty">좌표 기록을 불러오는 중...</p>}

        {frames && frames.length > 0 && currentFrame && (
          <>
            <div className="replay-main-grid">
              <div className="replay-visual-column">
                <div className="replay-canvas-wrap">
                  <canvas ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} aria-label="리플레이 스켈레톤" />
                  <div className="replay-flags">
                    <span className={currentFrame.bodyDetected ? "on" : ""}>몸 인식</span>
                    <span className={currentFrame.fullBodyVisible ? "on" : ""}>전신</span>
                    <span className={currentFrame.leftHandDetected || currentFrame.rightHandDetected ? "on" : ""}>손 관절</span>
                  </div>
                </div>
                <input type="range" className="replay-scrubber" min={0} max={frames.length - 1} value={frameIndex} onChange={(event) => { setPlaying(false); setFrameIndex(Number(event.target.value)); }} aria-label="재생 위치" />
                <div className="replay-controls">
                  <button type="button" onClick={() => { if (!playing && frameIndex >= frames.length - 1) setFrameIndex(0); setPlaying((value) => !value); }}>
                    {playing ? "일시정지" : frameIndex >= frames.length - 1 ? "처음부터 재생" : "재생"}
                  </button>
                  <span className="replay-time">{formatRelativeTime(currentFrame.relativeTimeMs)}</span>
                  <div className="replay-speed" role="group" aria-label="재생 속도">
                    {[0.5, 1, 2].map((rate) => <button key={rate} type="button" className={playbackRate === rate ? "active" : ""} onClick={() => { if (frameIndex >= frames.length - 1) setFrameIndex(0); setPlaybackRate(rate); }}>{rate}x</button>)}
                  </div>
                </div>
              </div>

              <aside className="analysis-panel" aria-label="AI 동작 분석">
                <span className="analysis-eyebrow">AI 동작 분석</span>
                <div className="analysis-badges">
                  <span className={`analysis-badge ${analysis.tone}`}>{analysis.badge}</span>
                  {detectionExplanation?.metricLine && <span className="analysis-badge info">개인 기준 비교</span>}
                </div>
                <div className="analysis-message">
                  <span aria-hidden="true">✦</span>
                  <div>
                    <h3>{analysis.title}</h3>
                    <p>{analysis.description}</p>
                    {detectionExplanation?.metricLine && <small>{detectionExplanation.metricLine}</small>}
                  </div>
                </div>
                <ol className="motion-timeline" aria-label="동작 흐름">
                  {analysis.timeline.map((label, index) => <li key={label}><span>{index + 1}</span><strong>{label}</strong></li>)}
                </ol>
                <div className="qualitative-tags">
                  <span>동작 속도 · <strong>{motionTags.speed}</strong></span>
                  <span>움직임 범위 · <strong>{motionTags.range}</strong></span>
                  <span>업무 흐름 · <strong>{motionTags.rhythm}</strong></span>
                </div>
                {detectionExplanation?.contributesToSignal && <p className="care-included">이번 주 케어 흐름에 반영된 항목이에요.</p>}
                <div className="analysis-feedback">
                  <strong>이 분석이 맞았나요?</strong>
                  {feedback ? <p>{feedback === "accurate" ? "확인해 주셔서 감사해요. 분석 결과를 유지할게요." : "이 장면은 분석 집계에서 제외했어요."}</p> : (
                    <div>
                      <button type="button" onClick={() => void submitFeedback("accurate")}>👍 정확한 분석이에요</button>
                      <button type="button" onClick={() => setShowFeedbackReasons(true)}>👎 잘못된 감지예요</button>
                    </div>
                  )}
                  {showFeedbackReasons && <div className="feedback-reasons" role="group" aria-label="오탐 이유">
                    <button type="button" onClick={() => void submitFeedback("false_positive", "interaction")}>손님 응대</button>
                    <button type="button" onClick={() => void submitFeedback("false_positive", "rest")}>휴식 중</button>
                    <button type="button" onClick={() => void submitFeedback("false_positive", "wrong_task")}>다른 업무</button>
                    <button type="button" onClick={() => void submitFeedback("false_positive", "camera_error")}>카메라 오류</button>
                  </div>}
                </div>
              </aside>
            </div>

            <details className="analysis-details">
              <summary>분석 상세 정보 보기</summary>
              <p>이 수치는 분석 검토용이며 건강 상태를 판단하는 값이 아닙니다.</p>
              <div className="replay-stats">
                <article><span>구간 동작 변동성</span><strong>{currentVariability !== undefined ? currentVariability.toFixed(2) : "–"}</strong></article>
                <article><span>구간 매끄러움</span><strong>{currentSmoothness !== undefined ? currentSmoothness.toFixed(4) : "–"}</strong></article>
                <article><span>프레임</span><strong>{frameIndex + 1} / {frames.length}</strong></article>
                <article><span>경과 시간</span><strong>{formatRelativeTime(currentFrame.relativeTimeMs)}</strong></article>
              </div>
              <canvas ref={sparklineRef} className="replay-sparkline" width={SPARKLINE_WIDTH} height={SPARKLINE_HEIGHT} aria-label="구간별 동작 변동성 그래프" />
              {detectedEvents.map((event, index) => <p className="technical-evidence" key={`${event.type}-${index}`}>{event.evidence}</p>)}
              {detectionExplanation?.rule && <p className="technical-evidence">{detectionExplanation.rule}</p>}
            </details>
          </>
        )}
      </section>
    </div>
  );
}
