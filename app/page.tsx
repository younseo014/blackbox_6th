"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { NormalizedLandmark, PoseLandmarker } from "@mediapipe/tasks-vision";
import {
  POSE_FRAME_STRIDE,
  POSE_LANDMARK_COUNT,
  POSE_SAMPLE_RATE,
  appendPoseChunk,
  createPoseSession,
  downloadPoseSession,
  finishPoseSession,
  listPoseSessions,
  type PoseSessionRecord,
} from "./pose-store";

type View = "today" | "timeline" | "closing" | "care";
type CameraStatus = "idle" | "requesting" | "connected" | "error";
type PoseStatus = "idle" | "loading" | "searching" | "partial" | "full" | "error";
type ClosingStatus = "idle" | "checking" | "attention" | "done";
type EventKind = "payment" | "door" | "safety" | "booking";

type TimelineEvent = {
  id: string;
  time: string;
  title: string;
  detail: string;
  kind: EventKind;
  poseSessionId?: string;
  poseSnapshot?: number[];
};

type PoseStats = {
  frames: number;
  detectedFrames: number;
  fullBodyFrames: number;
  storageBytes: number;
  startedAt: number | null;
};

const POSE_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19],
  [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20],
  [16, 22], [18, 20], [11, 23], [12, 24], [23, 24], [23, 25],
  [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31],
  [30, 32], [27, 31], [28, 32],
];

const FULL_BODY_LANDMARKS = [0, 11, 12, 23, 24, 25, 26, 27, 28];
const CHUNK_FRAME_COUNT = POSE_SAMPLE_RATE * 30;

const initialEvents: TimelineEvent[] = [
  {
    id: "sample-1",
    time: "16:30",
    title: "손님이 나가셨어요",
    detail: "출입문 움직임 감지",
    kind: "door",
  },
  {
    id: "sample-2",
    time: "14:05",
    title: "결제가 완료됐어요",
    detail: "카드 결제 · 45,000원",
    kind: "payment",
  },
  {
    id: "sample-3",
    time: "13:52",
    title: "예약을 등록했어요",
    detail: "김하나 고객 · 커트",
    kind: "booking",
  },
  {
    id: "sample-4",
    time: "09:42",
    title: "매장 문을 열었어요",
    detail: "출입문 센서 감지",
    kind: "door",
  },
];

const navItems: Array<{ id: View; label: string; icon: string }> = [
  { id: "today", label: "오늘", icon: "⌂" },
  { id: "timeline", label: "타임라인", icon: "≡" },
  { id: "closing", label: "스마트 마감", icon: "✓" },
  { id: "care", label: "케어 기록", icon: "♡" },
];

const eventPresets: Array<{
  title: string;
  detail: string;
  kind: EventKind;
  button: string;
}> = [
  {
    title: "결제가 완료됐어요",
    detail: "로컬 카메라 테스트 · 결제 이벤트",
    kind: "payment",
    button: "결제 완료",
  },
  {
    title: "손님이 나가셨어요",
    detail: "로컬 카메라 테스트 · 퇴장 이벤트",
    kind: "door",
    button: "손님 퇴장",
  },
  {
    title: "출입문이 열렸어요",
    detail: "로컬 카메라 테스트 · 출입 이벤트",
    kind: "door",
    button: "출입문 열림",
  },
];

const kindLabel: Record<EventKind, string> = {
  payment: "결제",
  door: "출입",
  safety: "안전",
  booking: "예약",
};

function currentTime() {
  return new Intl.DateTimeFormat("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date());
}

function currentMonotonicTime() {
  return performance.now();
}

function currentEpochTime() {
  return Math.round(performance.timeOrigin + performance.now());
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDuration(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
  const seconds = (totalSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function drawSkeleton(
  canvas: HTMLCanvasElement,
  landmarks: NormalizedLandmark[] | number[],
  fullBody: boolean,
) {
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);

  const point = (index: number) => {
    if (typeof landmarks[0] === "number") {
      const flat = landmarks as number[];
      return {
        x: flat[index * 4],
        y: flat[index * 4 + 1],
        z: flat[index * 4 + 2],
        visibility: flat[index * 4 + 3],
      };
    }
    return (landmarks as NormalizedLandmark[])[index];
  };

  context.lineWidth = Math.max(3, canvas.width / 180);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = fullBody ? "#6ff0b7" : "#ffc36e";
  context.shadowColor = "rgba(8, 38, 32, 0.75)";
  context.shadowBlur = 8;

  for (const [from, to] of POSE_CONNECTIONS) {
    const start = point(from);
    const end = point(to);
    if (!start || !end || start.visibility < 0.35 || end.visibility < 0.35) continue;
    context.beginPath();
    context.moveTo(start.x * canvas.width, start.y * canvas.height);
    context.lineTo(end.x * canvas.width, end.y * canvas.height);
    context.stroke();
  }

  context.shadowBlur = 5;
  for (let index = 0; index < POSE_LANDMARK_COUNT; index += 1) {
    const landmark = point(index);
    if (!landmark || landmark.visibility < 0.35) continue;
    context.beginPath();
    context.arc(
      landmark.x * canvas.width,
      landmark.y * canvas.height,
      Math.max(3, canvas.width / 150),
      0,
      Math.PI * 2,
    );
    context.fillStyle = index === 0 ? "#ffffff" : fullBody ? "#b9ffdc" : "#ffe0aa";
    context.fill();
  }
}

function PoseSnapshot({ points }: { points: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    drawSkeleton(canvasRef.current, points, true);
  }, [points]);

  return <canvas ref={canvasRef} width={640} height={360} aria-label="기록된 스켈레톤 좌표" />;
}

function TimelineList({
  events,
  onSelect,
  compact = false,
}: {
  events: TimelineEvent[];
  onSelect: (event: TimelineEvent) => void;
  compact?: boolean;
}) {
  const visibleEvents = compact ? events.slice(0, 4) : events;

  return (
    <div className="timeline-list">
      {visibleEvents.map((event, index) => (
        <button
          className="timeline-event"
          key={event.id}
          onClick={() => onSelect(event)}
          type="button"
        >
          <span className="event-time">{event.time}</span>
          <span className={`event-node ${event.kind}`} aria-hidden="true">
            {event.kind === "safety" ? "✓" : ""}
          </span>
          {index < visibleEvents.length - 1 && (
            <span className="event-line" aria-hidden="true" />
          )}
          <span className="event-copy">
            <span className="event-title-row">
              <strong>{event.title}</strong>
              {event.poseSessionId && <span className="video-tag">좌표</span>}
            </span>
            <span>{event.detail}</span>
          </span>
          <span className="event-chevron" aria-hidden="true">
            ›
          </span>
        </button>
      ))}
    </div>
  );
}

export default function Home() {
  const [view, setView] = useState<View>("today");
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("idle");
  const [cameraMessage, setCameraMessage] = useState(
    "카메라를 연결하면 오늘의 장면을 확인할 수 있어요.",
  );
  const [poseStatus, setPoseStatus] = useState<PoseStatus>("idle");
  const [poseStats, setPoseStats] = useState<PoseStats>({
    frames: 0,
    detectedFrames: 0,
    fullBodyFrames: 0,
    storageBytes: 0,
    startedAt: null,
  });
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [sessionCount, setSessionCount] = useState(0);
  const [latestSession, setLatestSession] = useState<PoseSessionRecord | null>(null);
  const [events, setEvents] = useState<TimelineEvent[]>(initialEvents);
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const [toast, setToast] = useState("");
  const [heaterOn, setHeaterOn] = useState(true);
  const [closingStatus, setClosingStatus] = useState<ClosingStatus>("idle");
  const [closingStep, setClosingStep] = useState(0);
  const [savepointOpen, setSavepointOpen] = useState(true);
  const [bookingOpen, setBookingOpen] = useState(false);
  const [bookingName, setBookingName] = useState("김하나");
  const [bookingService, setBookingService] = useState("커트");

  const videoRef = useRef<HTMLVideoElement>(null);
  const processingVideoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const poseAnimationRef = useRef<number | null>(null);
  const trackingActiveRef = useRef(false);
  const lastDetectionTimeRef = useRef(0);
  const lastSampleTimeRef = useRef(0);
  const lastPoseRef = useRef<NormalizedLandmark[] | null>(null);
  const poseBufferRef = useRef<number[]>([]);
  const chunkDetectedFramesRef = useRef(0);
  const chunkFullBodyFramesRef = useRef(0);
  const poseSessionRef = useRef<PoseSessionRecord | null>(null);
  const poseWriteQueueRef = useRef<Promise<void>>(Promise.resolve());
  const sessionPerformanceStartRef = useRef(0);
  const lastStatsUpdateRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const poseStatusRef = useRef<PoseStatus>("idle");
  const poseStatsRef = useRef<PoseStats>({
    frames: 0,
    detectedFrames: 0,
    fullBodyFrames: 0,
    storageBytes: 0,
    startedAt: null,
  });

  const todayEvents = useMemo(() => events, [events]);

  useEffect(() => {
    if (cameraStatus === "connected" && streamRef.current) {
      if (videoRef.current) {
        videoRef.current.srcObject = streamRef.current;
        videoRef.current.play().catch(() => undefined);
      }
      if (processingVideoRef.current) {
        processingVideoRef.current.srcObject = streamRef.current;
        processingVideoRef.current.play().catch(() => undefined);
      }
    }
  }, [cameraStatus, view]);

  useEffect(() => {
    listPoseSessions()
      .then((sessions) => {
        setSessionCount(sessions.length);
        setLatestSession(sessions[0] ?? null);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (cameraStatus !== "connected" || !poseStats.startedAt) return;
    const timer = window.setInterval(() => {
      const now = currentEpochTime();
      setElapsedSeconds(
        Math.max(0, Math.floor((now - poseStats.startedAt!) / 1000)),
      );
    }, 1000);
    return () => window.clearInterval(timer);
  }, [cameraStatus, poseStats.startedAt]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 3200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    return () => {
      trackingActiveRef.current = false;
      if (poseAnimationRef.current !== null) {
        cancelAnimationFrame(poseAnimationRef.current);
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      poseLandmarkerRef.current?.close();
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelectedEvent(null);
        setBookingOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraStatus("error");
      setCameraMessage("이 브라우저에서는 카메라를 사용할 수 없어요.");
      return;
    }

    setCameraStatus("requesting");
    setCameraMessage("카메라 연결을 기다리고 있어요…");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      if (processingVideoRef.current) {
        processingVideoRef.current.srcObject = stream;
        await processingVideoRef.current.play();
      }
      setCameraStatus("connected");
      setCameraMessage("전신 스켈레톤 모델을 준비하고 있어요…");
      await startPoseTracking();
    } catch {
      setCameraStatus("error");
      setCameraMessage(
        "카메라 권한을 허용한 뒤 다시 연결해 주세요. 영상은 외부로 전송되지 않아요.",
      );
    }
  }

  function updatePoseStatus(nextStatus: PoseStatus) {
    if (poseStatusRef.current === nextStatus) return;
    poseStatusRef.current = nextStatus;
    setPoseStatus(nextStatus);
  }

  async function ensurePoseLandmarker() {
    if (poseLandmarkerRef.current) return poseLandmarkerRef.current;
    updatePoseStatus("loading");
    const { FilesetResolver, PoseLandmarker: PoseLandmarkerClass } = await import(
      "@mediapipe/tasks-vision"
    );
    const vision = await FilesetResolver.forVisionTasks("/mediapipe-wasm");
    const baseOptions = {
      modelAssetPath: "/models/pose_landmarker_lite.task",
    };

    try {
      poseLandmarkerRef.current = await PoseLandmarkerClass.createFromOptions(
        vision,
        {
          baseOptions: { ...baseOptions, delegate: "GPU" },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputSegmentationMasks: false,
        },
      );
    } catch {
      poseLandmarkerRef.current = await PoseLandmarkerClass.createFromOptions(
        vision,
        {
          baseOptions,
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputSegmentationMasks: false,
        },
      );
    }
    return poseLandmarkerRef.current;
  }

  function isFullBodyVisible(landmarks: NormalizedLandmark[]) {
    const allRequiredVisible = FULL_BODY_LANDMARKS.every((index) => {
      const point = landmarks[index];
      return (
        point &&
        point.visibility >= 0.55 &&
        point.x >= 0.015 &&
        point.x <= 0.985 &&
        point.y >= 0.015 &&
        point.y <= 0.985
      );
    });
    if (!allRequiredVisible) return false;
    const ankleY = (landmarks[27].y + landmarks[28].y) / 2;
    return ankleY - landmarks[0].y >= 0.5;
  }

  function flushPoseFrames() {
    if (poseBufferRef.current.length === 0 || !poseSessionRef.current) {
      return poseWriteQueueRef.current;
    }
    const values = new Float32Array(poseBufferRef.current);
    const detectedFrames = chunkDetectedFramesRef.current;
    const fullBodyFrames = chunkFullBodyFramesRef.current;
    const sessionId = poseSessionRef.current.id;
    poseBufferRef.current = [];
    chunkDetectedFramesRef.current = 0;
    chunkFullBodyFramesRef.current = 0;

    poseWriteQueueRef.current = poseWriteQueueRef.current.then(async () => {
      const session = poseSessionRef.current;
      if (!session || session.id !== sessionId) return;
      const updated = await appendPoseChunk(
        session,
        session.frameCount,
        values,
        detectedFrames,
        fullBodyFrames,
      );
      poseSessionRef.current = updated;
      setLatestSession(updated);
    });
    return poseWriteQueueRef.current;
  }

  function recordPoseFrame(
    timestamp: number,
    landmarks: NormalizedLandmark[] | null,
    fullBody: boolean,
  ) {
    const buffer = poseBufferRef.current;
    buffer.push(timestamp - sessionPerformanceStartRef.current);
    buffer.push(landmarks ? 1 : 0);
    buffer.push(fullBody ? 1 : 0);

    if (landmarks) {
      for (let index = 0; index < POSE_LANDMARK_COUNT; index += 1) {
        const point = landmarks[index];
        buffer.push(point.x, point.y, point.z, point.visibility);
      }
      chunkDetectedFramesRef.current += 1;
    } else {
      for (let index = 0; index < POSE_LANDMARK_COUNT * 4; index += 1) {
        buffer.push(Number.NaN);
      }
    }
    if (fullBody) chunkFullBodyFramesRef.current += 1;

    const previous = poseStatsRef.current;
    const next: PoseStats = {
      ...previous,
      frames: previous.frames + 1,
      detectedFrames: previous.detectedFrames + (landmarks ? 1 : 0),
      fullBodyFrames: previous.fullBodyFrames + (fullBody ? 1 : 0),
      storageBytes: (previous.frames + 1) * POSE_FRAME_STRIDE * 4,
    };
    poseStatsRef.current = next;
    if (timestamp - lastStatsUpdateRef.current >= 500) {
      lastStatsUpdateRef.current = timestamp;
      setPoseStats(next);
    }

    if (buffer.length / POSE_FRAME_STRIDE >= CHUNK_FRAME_COUNT) {
      void flushPoseFrames();
    }
  }

  function poseTrackingLoop() {
    if (!trackingActiveRef.current) return;
    const video = processingVideoRef.current;
    const landmarker = poseLandmarkerRef.current;
    const timestamp = currentMonotonicTime();

    if (
      video &&
      landmarker &&
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      video.currentTime !== lastVideoTimeRef.current &&
      timestamp - lastDetectionTimeRef.current >= 66
    ) {
      lastVideoTimeRef.current = video.currentTime;
      lastDetectionTimeRef.current = timestamp;
      try {
        const result = landmarker.detectForVideo(video, timestamp);
        const landmarks = result.landmarks[0] ?? null;
        lastPoseRef.current = landmarks;
        const fullBody = landmarks ? isFullBodyVisible(landmarks) : false;
        updatePoseStatus(landmarks ? (fullBody ? "full" : "partial") : "searching");

        const canvas = overlayCanvasRef.current;
        if (canvas) {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          const context = canvas.getContext("2d");
          context?.clearRect(0, 0, canvas.width, canvas.height);
          if (landmarks) drawSkeleton(canvas, landmarks, fullBody);
        }

        if (timestamp - lastSampleTimeRef.current >= 1000 / POSE_SAMPLE_RATE) {
          lastSampleTimeRef.current = timestamp;
          recordPoseFrame(timestamp, landmarks, fullBody);
        }
      } catch {
        updatePoseStatus("error");
      }
    }
    poseAnimationRef.current = requestAnimationFrame(poseTrackingLoop);
  }

  async function startPoseTracking() {
    try {
      await ensurePoseLandmarker();
      const startedAt = currentEpochTime();
      const session = await createPoseSession(
        `pose-${crypto.randomUUID()}`,
        startedAt,
      );
      poseSessionRef.current = session;
      sessionPerformanceStartRef.current = currentMonotonicTime();
      lastDetectionTimeRef.current = 0;
      lastSampleTimeRef.current = 0;
      lastVideoTimeRef.current = -1;
      lastPoseRef.current = null;
      poseBufferRef.current = [];
      chunkDetectedFramesRef.current = 0;
      chunkFullBodyFramesRef.current = 0;
      const emptyStats: PoseStats = {
        frames: 0,
        detectedFrames: 0,
        fullBodyFrames: 0,
        storageBytes: 0,
        startedAt,
      };
      poseStatsRef.current = emptyStats;
      setPoseStats(emptyStats);
      setElapsedSeconds(0);
      setSessionCount((count) => count + 1);
      updatePoseStatus("searching");
      trackingActiveRef.current = true;
      setCameraMessage("영상은 저장하지 않고 관절 좌표만 기록하고 있어요.");
      setToast("스켈레톤 좌표 상시 기록을 시작했어요");
      poseAnimationRef.current = requestAnimationFrame(poseTrackingLoop);
    } catch {
      updatePoseStatus("error");
      setCameraMessage("스켈레톤 모델을 불러오지 못했어요. 다시 연결해 주세요.");
      setToast("스켈레톤 모델을 준비하지 못했어요");
    }
  }

  async function stopPoseTracking() {
    trackingActiveRef.current = false;
    if (poseAnimationRef.current !== null) {
      cancelAnimationFrame(poseAnimationRef.current);
      poseAnimationRef.current = null;
    }
    await flushPoseFrames();
    await poseWriteQueueRef.current;
    const session = poseSessionRef.current;
    if (session) {
      const endedAt = currentEpochTime();
      const completed = await finishPoseSession(session, endedAt);
      setLatestSession(completed);
      poseSessionRef.current = null;
    }
    overlayCanvasRef.current
      ?.getContext("2d")
      ?.clearRect(
        0,
        0,
        overlayCanvasRef.current.width,
        overlayCanvasRef.current.height,
      );
    updatePoseStatus("idle");
  }

  async function stopCamera() {
    await stopPoseTracking();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    if (processingVideoRef.current) processingVideoRef.current.srcObject = null;
    setCameraStatus("idle");
    setElapsedSeconds(0);
    setCameraMessage("카메라 연결을 멈췄어요.");
    setToast("좌표 기록을 안전하게 저장하고 카메라를 종료했어요");
  }

  function markTestEvent(preset: (typeof eventPresets)[number]) {
    const session = poseSessionRef.current;
    const landmarks = lastPoseRef.current;
    if (!session || cameraStatus !== "connected") {
      setToast("먼저 카메라를 연결해 주세요");
      return;
    }
    if (!landmarks) {
      setToast("스켈레톤이 인식된 뒤 이벤트를 표시해 주세요");
      return;
    }
    const event: TimelineEvent = {
      id: `pose-event-${crypto.randomUUID()}`,
      time: currentTime(),
      title: preset.title,
      detail: `${preset.detail} · 관절 좌표 시점`,
      kind: preset.kind,
      poseSessionId: session.id,
      poseSnapshot: landmarks.flatMap((point) => [
        point.x,
        point.y,
        point.z,
        point.visibility,
      ]),
    };
    setEvents((previous) => [event, ...previous]);
    setToast("이벤트 시점을 좌표 기록에 표시했어요");
  }

  async function exportPoseData() {
    try {
      await flushPoseFrames();
      await poseWriteQueueRef.current;
      const session = poseSessionRef.current ?? latestSession;
      if (!session || session.frameCount === 0) {
        setToast("내보낼 좌표 기록이 아직 없어요");
        return;
      }
      await downloadPoseSession(session);
      setToast("학습용 좌표 데이터를 내려받았어요");
    } catch {
      setToast("좌표 데이터를 내보내지 못했어요");
    }
  }

  function startClosingCheck() {
    setClosingStatus("checking");
    setClosingStep(0);
    let step = 0;
    const timer = window.setInterval(() => {
      step += 1;
      setClosingStep(step);
      if (step >= 3) {
        window.clearInterval(timer);
        if (heaterOn) {
          setClosingStatus("attention");
        } else {
          completeClosing();
        }
      }
    }, 650);
  }

  function completeClosing() {
    setClosingStatus("done");
    setClosingStep(3);
    setEvents((previous) => {
      if (previous[0]?.title === "오늘의 마감이 완료됐어요") return previous;
      return [
        {
          id: `closing-${crypto.randomUUID()}`,
          time: currentTime(),
          title: "오늘의 마감이 완료됐어요",
          detail: "가스 · 전기 · 출입문 모두 안전",
          kind: "safety",
        },
        ...previous,
      ];
    });
    setToast("모든 항목이 안전해요. 편안히 퇴근하세요");
  }

  function turnOffHeater() {
    setHeaterOn(false);
    setClosingStatus("idle");
    setToast("온열기 전원을 차단했어요");
  }

  function saveBooking(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setEvents((previous) => [
      {
        id: `booking-${crypto.randomUUID()}`,
        time: currentTime(),
        title: "예약을 등록했어요",
        detail: `${bookingName} 고객 · ${bookingService}`,
        kind: "booking",
      },
      ...previous,
    ]);
    setSavepointOpen(false);
    setBookingOpen(false);
    setToast("하던 업무를 이어서 완료했어요");
  }

  function openTimelineEvent(event: TimelineEvent) {
    setSelectedEvent(event);
  }

  const statusText =
    cameraStatus === "connected"
      ? "카메라 연결됨"
      : cameraStatus === "requesting"
        ? "연결 중"
        : "카메라 대기";

  const poseStatusLabel =
    poseStatus === "full"
      ? "전신 인식됨"
      : poseStatus === "partial"
        ? "전신이 보이게 뒤로 이동해 주세요"
        : poseStatus === "searching"
          ? "사람을 찾고 있어요"
          : poseStatus === "loading"
            ? "스켈레톤 모델 준비 중"
            : poseStatus === "error"
              ? "스켈레톤 인식 오류"
              : "좌표 기록 대기";

  const fullBodyRatio = poseStats.detectedFrames
    ? Math.round((poseStats.fullBodyFrames / poseStats.detectedFrames) * 100)
    : 0;

  return (
    <main className="app-shell">
      <video
        ref={processingVideoRef}
        className="processing-video"
        muted
        playsInline
        aria-hidden="true"
      />
      <aside className="sidebar">
        <div className="brand" aria-label="메모리 가드">
          <span className="brand-mark" aria-hidden="true">
            M
          </span>
          <span>
            <strong>메모리 가드</strong>
            <small>Memory Guard</small>
          </span>
        </div>

        <nav className="main-nav" aria-label="주요 메뉴">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={view === item.id ? "active" : ""}
              onClick={() => setView(item.id)}
              type="button"
              aria-current={view === item.id ? "page" : undefined}
            >
              <span aria-hidden="true">{item.icon}</span>
              {item.label}
              {item.id === "closing" && heaterOn && (
                <i aria-label="확인할 항목 1개">1</i>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-care">
          <span className="care-sprout" aria-hidden="true">
            ♡
          </span>
          <p>사장님의 하루를</p>
          <strong>조용히 지켜드릴게요.</strong>
        </div>
        <div className="profile">
          <span className="avatar">김</span>
          <span>
            <strong>김메모리 사장님</strong>
            <small>오늘도 좋은 하루예요</small>
          </span>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <span className="eyebrow">8월 11일 화요일</span>
            <h1>
              {view === "today" && "안녕하세요, 사장님"}
              {view === "timeline" && "오늘의 메모리 타임라인"}
              {view === "closing" && "스마트 마감"}
              {view === "care" && "나의 케어 기록"}
            </h1>
          </div>
          <div className={`camera-pill ${cameraStatus}`}>
            <span aria-hidden="true" />
            {statusText}
          </div>
        </header>

        {view === "today" && (
          <div className="today-view">
            <section className="safety-banner">
              <div className="safety-check" aria-hidden="true">
                ✓
              </div>
              <div>
                <strong>매장은 대체로 안전해요</strong>
                <p>온열기 한 가지만 확인하면 마음 놓고 마감할 수 있어요.</p>
              </div>
              <button type="button" onClick={() => setView("closing")}>
                확인하기 <span aria-hidden="true">›</span>
              </button>
            </section>

            <div className="dashboard-grid">
              <section className="panel camera-panel">
                <div className="panel-heading">
                  <div>
                    <span className="section-kicker">기억 복원 카메라</span>
                    <h2>지금 매장 모습</h2>
                  </div>
                  {cameraStatus === "connected" && (
                    <div className="camera-badges">
                      <span className="coordinate-badge">
                        <i aria-hidden="true" /> 좌표 REC
                      </span>
                      <span className="live-badge">
                        <i aria-hidden="true" /> LIVE
                      </span>
                    </div>
                  )}
                </div>

                <div className={`camera-frame ${cameraStatus}`}>
                  <video ref={videoRef} muted playsInline aria-label="실시간 카메라 영상" />
                  <canvas
                    ref={overlayCanvasRef}
                    className="pose-overlay"
                    aria-label="실시간 전신 스켈레톤"
                  />
                  {cameraStatus !== "connected" && (
                    <div className="camera-empty">
                      <span className="camera-symbol" aria-hidden="true">
                        ●
                      </span>
                      <strong>
                        {cameraStatus === "requesting"
                          ? "카메라를 연결하고 있어요"
                          : "노트북 카메라를 연결해 주세요"}
                      </strong>
                      <p>{cameraMessage}</p>
                      <button
                        className="primary-button"
                        type="button"
                        onClick={startCamera}
                        disabled={cameraStatus === "requesting"}
                      >
                        {cameraStatus === "requesting" ? "연결 중…" : "카메라 연결"}
                      </button>
                    </div>
                  )}
                  {cameraStatus === "connected" && (
                    <div className="camera-caption">
                      <span className={`pose-state ${poseStatus}`}>
                        <i aria-hidden="true" /> {poseStatusLabel}
                      </span>
                      <button type="button" onClick={() => void stopCamera()}>
                        카메라 끄기
                      </button>
                    </div>
                  )}
                </div>

                <div className="coordinate-recorder">
                  <div className="recorder-heading">
                    <div>
                      <strong>스켈레톤 좌표 상시 기록</strong>
                      <p>{cameraMessage}</p>
                    </div>
                    <span className={cameraStatus === "connected" ? "recording" : ""}>
                      {cameraStatus === "connected" ? formatDuration(elapsedSeconds) : "대기"}
                    </span>
                  </div>
                  <div className="coordinate-stats">
                    <span><small>관절점</small><strong>33개</strong></span>
                    <span><small>기록 속도</small><strong>{POSE_SAMPLE_RATE} FPS</strong></span>
                    <span><small>누적 프레임</small><strong>{poseStats.frames.toLocaleString()}</strong></span>
                    <span><small>전신 인식률</small><strong>{fullBodyRatio}%</strong></span>
                    <span><small>예상 용량</small><strong>{formatBytes(poseStats.storageBytes)}</strong></span>
                  </div>
                  <div className="coordinate-actions">
                    <span>영상·음성 없이 좌표만 이 브라우저에 저장 · 세션 {sessionCount}개</span>
                    <button
                      type="button"
                      onClick={() => void exportPoseData()}
                      disabled={!latestSession && poseStats.frames === 0}
                    >
                      학습용 JSON 내려받기
                    </button>
                  </div>
                </div>

                <div className="camera-test">
                  <div>
                    <strong>타임라인 이벤트 표시</strong>
                    <p>버튼을 누른 시점의 스켈레톤 좌표를 타임라인에 표시해요.</p>
                  </div>
                  <div className="test-buttons">
                    {eventPresets.map((preset) => (
                      <button
                        type="button"
                        key={preset.button}
                        disabled={
                          cameraStatus !== "connected" ||
                          (poseStatus !== "partial" && poseStatus !== "full")
                        }
                        onClick={() => markTestEvent(preset)}
                      >
                        {preset.button}
                      </button>
                    ))}
                  </div>
                </div>
              </section>

              <section className="panel timeline-panel">
                <div className="panel-heading">
                  <div>
                    <span className="section-kicker">메모리 타임라인</span>
                    <h2>오늘 무슨 일이 있었나요?</h2>
                  </div>
                  <button
                    className="text-button"
                    type="button"
                    onClick={() => setView("timeline")}
                  >
                    모두 보기
                  </button>
                </div>
                <TimelineList
                  events={todayEvents}
                  onSelect={openTimelineEvent}
                  compact
                />
              </section>
            </div>

            {savepointOpen && (
              <section className="savepoint-card">
                <div className="savepoint-icon" aria-hidden="true">
                  ↗
                </div>
                <div>
                  <span className="section-kicker">업무 세이브포인트</span>
                  <h2>아까 하던 예약 입력이 남아 있어요</h2>
                  <p>김하나 고객 · 커트 · 예약 시간 확인 단계</p>
                </div>
                <button type="button" onClick={() => setBookingOpen(true)}>
                  이어서 하기 <span aria-hidden="true">›</span>
                </button>
              </section>
            )}
          </div>
        )}

        {view === "timeline" && (
          <div className="subpage timeline-page">
            <div className="subpage-intro">
              <div>
                <span className="section-kicker">하루의 기억을 한눈에</span>
                <p>중요한 순간만 모았어요. 좌표 표시가 있는 항목은 당시 스켈레톤을 확인할 수 있어요.</p>
              </div>
              <span className="count-chip">오늘 {events.length}개 기록</span>
            </div>
            <section className="panel full-timeline-panel">
              <TimelineList events={events} onSelect={openTimelineEvent} />
            </section>
          </div>
        )}

        {view === "closing" && (
          <div className="subpage closing-page">
            <section className={`closing-hero ${closingStatus}`}>
              <div className="closing-hero-icon" aria-hidden="true">
                {closingStatus === "done" ? "✓" : closingStatus === "attention" ? "!" : "⌁"}
              </div>
              <div>
                <span className="section-kicker">퇴근 전 자동 점검</span>
                <h2>
                  {closingStatus === "done"
                    ? "오늘의 마감이 끝났어요"
                    : closingStatus === "checking"
                      ? "매장을 하나씩 확인하고 있어요"
                      : closingStatus === "attention"
                        ? "온열기 전원을 확인해 주세요"
                        : "마지막으로 안전을 확인할까요?"}
                </h2>
                <p>
                  {closingStatus === "done"
                    ? "가스, 전기, 출입문 모두 안전해요. 편안히 퇴근하세요."
                    : "연결된 센서가 가스, 전기, 출입문 상태를 확인해 드려요."}
                </p>
              </div>
            </section>

            <section className="panel closing-card">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">연결된 안전 기기</span>
                  <h2>마감 확인 항목</h2>
                </div>
                <span className="device-count">3개 연결됨</span>
              </div>

              <div className="device-list">
                <div className={`device-row ${closingStep >= 1 || closingStatus === "idle" ? "checked" : ""}`}>
                  <span className="device-icon gas" aria-hidden="true">G</span>
                  <span className="device-copy">
                    <strong>가스 밸브</strong>
                    <small>주방 가스 차단기</small>
                  </span>
                  <span className="device-status safe"><i /> 닫혀 있어요</span>
                </div>
                <div className={`device-row ${!heaterOn ? "checked" : "warning"}`}>
                  <span className="device-icon power" aria-hidden="true">P</span>
                  <span className="device-copy">
                    <strong>온열기 전원</strong>
                    <small>카운터 스마트 플러그</small>
                  </span>
                  {heaterOn ? (
                    <button className="device-action" type="button" onClick={turnOffHeater}>
                      전원 끄기
                    </button>
                  ) : (
                    <span className="device-status safe"><i /> 꺼져 있어요</span>
                  )}
                </div>
                <div className={`device-row ${closingStep >= 3 || closingStatus === "idle" ? "checked" : ""}`}>
                  <span className="device-icon lock" aria-hidden="true">D</span>
                  <span className="device-copy">
                    <strong>출입문</strong>
                    <small>정문 도어락 센서</small>
                  </span>
                  <span className="device-status safe"><i /> 잠겨 있어요</span>
                </div>
              </div>

              <button
                className="closing-button"
                type="button"
                onClick={startClosingCheck}
                disabled={closingStatus === "checking" || closingStatus === "done"}
              >
                {closingStatus === "checking"
                  ? `안전 확인 중 ${closingStep}/3`
                  : closingStatus === "done"
                    ? "마감 완료"
                    : "퇴근 전 자동 점검"}
              </button>
            </section>
          </div>
        )}

        {view === "care" && (
          <div className="subpage care-page">
            <section className="care-summary">
              <span className="care-summary-mark" aria-hidden="true">♡</span>
              <div>
                <span className="section-kicker">최근 2주 케어 리포트</span>
                <h2>대체로 평소와 비슷한 흐름이에요</h2>
                <p>가스·전기 차단 알림이 3회 있었어요. 이번 주말은 푹 쉬면서 컨디션을 관리해 보세요.</p>
              </div>
            </section>

            <div className="metric-grid">
              <article className="metric-card">
                <span>안전 알림</span>
                <strong>3<small>회</small></strong>
                <p><i className="up">↑ 1회</i> 지난 2주와 비교</p>
              </article>
              <article className="metric-card">
                <span>마감 반복 확인</span>
                <strong>2<small>회</small></strong>
                <p><i>— 같음</i> 지난 2주와 비교</p>
              </article>
              <article className="metric-card">
                <span>업무 누락</span>
                <strong>1<small>건</small></strong>
                <p><i className="down">↓ 1건</i> 지난 2주와 비교</p>
              </article>
              <article className="metric-card">
                <span>업무 흐름</span>
                <strong className="word-value">평소와 비슷</strong>
                <p><i className="steady">● 안정</i> 일상 업무 소요 시간</p>
              </article>
            </div>

            <section className="panel care-note">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">따뜻한 한마디</span>
                  <h2>이번 주의 케어 메모</h2>
                </div>
                <span>8월 11일</span>
              </div>
              <p>바쁜 날에는 마감 확인이 조금 늘었지만, 업무 흐름은 평소와 비슷했어요. 충분히 쉬는 것만으로도 다음 주가 한결 가벼워질 거예요.</p>
            </section>
          </div>
        )}
      </section>

      <nav className="mobile-nav" aria-label="모바일 주요 메뉴">
        {navItems.map((item) => (
          <button
            key={item.id}
            className={view === item.id ? "active" : ""}
            onClick={() => setView(item.id)}
            type="button"
          >
            <span aria-hidden="true">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      {selectedEvent && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setSelectedEvent(null)}>
          <section
            className="video-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="video-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <button className="modal-close" type="button" onClick={() => setSelectedEvent(null)} aria-label="닫기">×</button>
            <div className="modal-heading">
              <span className={`event-type ${selectedEvent.kind}`}>{kindLabel[selectedEvent.kind]}</span>
              <span>{selectedEvent.time}</span>
              <h2 id="video-title">{selectedEvent.title}</h2>
              <p>{selectedEvent.detail}</p>
            </div>
            <div className="playback-frame">
              {selectedEvent.poseSnapshot ? (
                <div className="pose-snapshot">
                  <PoseSnapshot points={selectedEvent.poseSnapshot} />
                  <span>영상이 아닌 33개 관절점 좌표로 복원한 장면이에요.</span>
                </div>
              ) : (
                <div className="no-clip">
                  <span className="camera-symbol" aria-hidden="true">●</span>
                  <strong>이 기록은 예시 타임라인이에요</strong>
                  <p>카메라를 연결하고 이벤트 시점을 표시하면 이곳에서 당시 스켈레톤을 확인할 수 있어요.</p>
                </div>
              )}
            </div>
            <button className="modal-confirm" type="button" onClick={() => setSelectedEvent(null)}>확인했어요</button>
          </section>
        </div>
      )}

      {bookingOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setBookingOpen(false)}>
          <section
            className="booking-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="booking-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <button className="modal-close" type="button" onClick={() => setBookingOpen(false)} aria-label="닫기">×</button>
            <span className="section-kicker">아까 하던 업무</span>
            <h2 id="booking-title">예약 입력을 마무리할까요?</h2>
            <p>작성했던 내용은 그대로 임시 저장되어 있어요.</p>
            <form onSubmit={saveBooking}>
              <label>
                고객 이름
                <input value={bookingName} onChange={(event) => setBookingName(event.target.value)} required />
              </label>
              <label>
                서비스
                <select value={bookingService} onChange={(event) => setBookingService(event.target.value)}>
                  <option>커트</option>
                  <option>염색</option>
                  <option>펌</option>
                </select>
              </label>
              <label>
                예약 시간
                <input value="8월 14일 오후 3:00" readOnly />
              </label>
              <button type="submit">예약 입력 완료</button>
            </form>
          </section>
        </div>
      )}

      {toast && (
        <div className="toast" role="status">
          <span aria-hidden="true">✓</span> {toast}
        </div>
      )}
    </main>
  );
}
