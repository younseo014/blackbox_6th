"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type {
  HandLandmarker,
  NormalizedLandmark,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import {
  BODY_LANDMARK_COUNT,
  HAND_LANDMARK_COUNT,
  MOTION_FRAME_STRIDE,
  MOTION_SAMPLE_RATE,
  appendMotionChunk,
  createMotionSession,
  downloadMotionSession,
  finishMotionSession,
  listMotionSessions,
  type MotionSessionRecord,
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
  motionSnapshot?: MotionSnapshot;
};

type PoseStats = {
  frames: number;
  detectedFrames: number;
  fullBodyFrames: number;
  handDetectedFrames: number;
  storageBytes: number;
  startedAt: number | null;
};

type DemoDay = {
  day: string;
  date: string;
  safetyAlerts: number;
  doubleChecks: number;
  unfinishedTasks: number;
  microDelay: number;
  note: string;
  examples: string[];
};

const demoWeek: DemoDay[] = [
  {
    day: "월",
    date: "8/4",
    safetyAlerts: 0,
    doubleChecks: 0,
    unfinishedTasks: 0,
    microDelay: 8,
    note: "평소와 비슷한 월요일 오픈·마감 흐름",
    examples: ["08:41 매장 오픈", "19:07 스마트 마감 1회 완료"],
  },
  {
    day: "화",
    date: "8/5",
    safetyAlerts: 0,
    doubleChecks: 1,
    unfinishedTasks: 0,
    microDelay: 9,
    note: "마감 뒤 출입문 상태를 한 번 더 확인",
    examples: ["19:12 출입문 잠김 확인", "19:14 출입문 상태 재확인"],
  },
  {
    day: "수",
    date: "8/6",
    safetyAlerts: 1,
    doubleChecks: 1,
    unfinishedTasks: 1,
    microDelay: 12,
    note: "온열기 차단 알림과 결제 입력 중단이 함께 발생",
    examples: ["14:38 결제 입력이 6분간 중단", "19:21 온열기 차단 권고"],
  },
  {
    day: "목",
    date: "8/7",
    safetyAlerts: 0,
    doubleChecks: 1,
    unfinishedTasks: 0,
    microDelay: 13,
    note: "점심 혼잡 시간대의 음료 세팅 소요 시간이 늘어남",
    examples: ["12:16 음료 세팅 4분 10초", "19:09 가스 밸브 재확인"],
  },
  {
    day: "금",
    date: "8/8",
    safetyAlerts: 1,
    doubleChecks: 2,
    unfinishedTasks: 1,
    microDelay: 17,
    note: "마감 반복 확인과 미완료 주문이 평소보다 늘어남",
    examples: ["15:02 주문 입력 후 9분 지연", "19:28 스마트 플러그 차단 권고"],
  },
  {
    day: "토",
    date: "8/9",
    safetyAlerts: 1,
    doubleChecks: 2,
    unfinishedTasks: 1,
    microDelay: 19,
    note: "바쁜 날에 여러 지표가 함께 높아진 날",
    examples: ["11:47 카드 결제 재입력", "20:11 마감 항목 2회 재확인"],
  },
  {
    day: "일",
    date: "8/10",
    safetyAlerts: 0,
    doubleChecks: 1,
    unfinishedTasks: 0,
    microDelay: 14,
    note: "휴식 후 일부 회복됐지만 마감 확인은 이어짐",
    examples: ["10:26 오픈 준비 정상", "18:53 출입문 상태 재확인"],
  },
];

type HeadDirection = {
  yaw: number;
  pitch: number;
  roll: number;
  centerX: number;
  centerY: number;
};

type HandState = {
  left: NormalizedLandmark[] | null;
  right: NormalizedLandmark[] | null;
  leftScore: number;
  rightScore: number;
};

type MotionSnapshot = {
  body: number[];
  leftHand: number[] | null;
  rightHand: number[] | null;
  head: HeadDirection;
};

const BODY_CONNECTIONS: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19],
  [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20],
  [16, 22], [18, 20], [11, 23], [12, 24], [23, 24], [23, 25],
  [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31],
  [30, 32], [27, 31], [28, 32],
];

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const FULL_BODY_LANDMARKS = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28];
const CHUNK_FRAME_COUNT = MOTION_SAMPLE_RATE * 30;

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

function drawMotionSkeleton(
  canvas: HTMLCanvasElement,
  body: NormalizedLandmark[] | number[],
  leftHand: NormalizedLandmark[] | number[] | null,
  rightHand: NormalizedLandmark[] | number[] | null,
  head: HeadDirection | null,
  fullBody: boolean,
) {
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, canvas.width, canvas.height);

  const bodyPoint = (index: number) => {
    if (typeof body[0] === "number") {
      const flat = body as number[];
      const storedIndex = index - 11;
      return {
        x: flat[storedIndex * 4],
        y: flat[storedIndex * 4 + 1],
        z: flat[storedIndex * 4 + 2],
        visibility: flat[storedIndex * 4 + 3],
      };
    }
    return (body as NormalizedLandmark[])[index];
  };

  context.lineWidth = Math.max(3, canvas.width / 180);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = fullBody ? "#6ff0b7" : "#ffc36e";
  context.shadowColor = "rgba(8, 38, 32, 0.75)";
  context.shadowBlur = 8;

  for (const [from, to] of BODY_CONNECTIONS) {
    const start = bodyPoint(from);
    const end = bodyPoint(to);
    if (!start || !end || start.visibility < 0.35 || end.visibility < 0.35) continue;
    context.beginPath();
    context.moveTo(start.x * canvas.width, start.y * canvas.height);
    context.lineTo(end.x * canvas.width, end.y * canvas.height);
    context.stroke();
  }

  context.shadowBlur = 5;
  for (let index = 11; index < 33; index += 1) {
    const landmark = bodyPoint(index);
    if (!landmark || landmark.visibility < 0.35) continue;
    context.beginPath();
    context.arc(
      landmark.x * canvas.width,
      landmark.y * canvas.height,
      Math.max(3, canvas.width / 150),
      0,
      Math.PI * 2,
    );
    context.fillStyle = fullBody ? "#b9ffdc" : "#ffe0aa";
    context.fill();
  }

  if (head) {
    const leftShoulder = bodyPoint(11);
    const rightShoulder = bodyPoint(12);
    const shoulderWidth = leftShoulder && rightShoulder
      ? Math.abs(leftShoulder.x - rightShoulder.x) * canvas.width
      : canvas.width * 0.12;
    const radius = Math.max(14, shoulderWidth * 0.28);
    const centerX = head.centerX * canvas.width;
    const centerY = head.centerY * canvas.height;
    context.shadowBlur = 8;
    context.strokeStyle = "#ffffff";
    context.lineWidth = Math.max(3, canvas.width / 190);
    context.beginPath();
    context.arc(centerX, centerY, radius, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.moveTo(centerX, centerY);
    context.lineTo(
      centerX + head.yaw * radius * 1.25,
      centerY + head.pitch * radius * 1.25,
    );
    context.stroke();
  }

  const drawHand = (
    hand: NormalizedLandmark[] | number[] | null,
    color: string,
  ) => {
    if (!hand) return;
    const handPoint = (index: number) => {
      if (typeof hand[0] === "number") {
        const flat = hand as number[];
        return {
          x: flat[index * 3],
          y: flat[index * 3 + 1],
          z: flat[index * 3 + 2],
        };
      }
      return (hand as NormalizedLandmark[])[index];
    };
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = Math.max(2, canvas.width / 260);
    context.shadowBlur = 6;
    for (const [from, to] of HAND_CONNECTIONS) {
      const start = handPoint(from);
      const end = handPoint(to);
      context.beginPath();
      context.moveTo(start.x * canvas.width, start.y * canvas.height);
      context.lineTo(end.x * canvas.width, end.y * canvas.height);
      context.stroke();
    }
    for (let index = 0; index < HAND_LANDMARK_COUNT; index += 1) {
      const landmark = handPoint(index);
      context.beginPath();
      context.arc(
        landmark.x * canvas.width,
        landmark.y * canvas.height,
        Math.max(2, canvas.width / 230),
        0,
        Math.PI * 2,
      );
      context.fill();
    }
  };

  drawHand(leftHand, "#74d9ff");
  drawHand(rightHand, "#d9a0ff");
}

function PoseSnapshot({ snapshot }: { snapshot: MotionSnapshot }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    drawMotionSkeleton(
      canvasRef.current,
      snapshot.body,
      snapshot.leftHand,
      snapshot.rightHand,
      snapshot.head,
      true,
    );
  }, [snapshot]);

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
    handDetectedFrames: 0,
    storageBytes: 0,
    startedAt: null,
  });
  const [detectedHands, setDetectedHands] = useState(0);
  const [headDirectionLabel, setHeadDirectionLabel] = useState("대기");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [sessionCount, setSessionCount] = useState(0);
  const [latestSession, setLatestSession] = useState<MotionSessionRecord | null>(null);
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
  const [demoMode, setDemoMode] = useState(true);
  const [selectedDemoDay, setSelectedDemoDay] = useState(6);

  const videoRef = useRef<HTMLVideoElement>(null);
  const processingVideoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const poseAnimationRef = useRef<number | null>(null);
  const trackingActiveRef = useRef(false);
  const lastDetectionTimeRef = useRef(0);
  const lastHandDetectionTimeRef = useRef(0);
  const lastSampleTimeRef = useRef(0);
  const lastPoseRef = useRef<NormalizedLandmark[] | null>(null);
  const lastHandsRef = useRef<HandState>({
    left: null,
    right: null,
    leftScore: 0,
    rightScore: 0,
  });
  const lastHeadDirectionRef = useRef<HeadDirection | null>(null);
  const poseBufferRef = useRef<number[]>([]);
  const chunkDetectedFramesRef = useRef(0);
  const chunkFullBodyFramesRef = useRef(0);
  const chunkHandFramesRef = useRef(0);
  const poseSessionRef = useRef<MotionSessionRecord | null>(null);
  const poseWriteQueueRef = useRef<Promise<void>>(Promise.resolve());
  const sessionPerformanceStartRef = useRef(0);
  const lastStatsUpdateRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const poseStatusRef = useRef<PoseStatus>("idle");
  const poseStatsRef = useRef<PoseStats>({
    frames: 0,
    detectedFrames: 0,
    fullBodyFrames: 0,
    handDetectedFrames: 0,
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
    listMotionSessions()
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
      handLandmarkerRef.current?.close();
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
      setCameraMessage("몸·머리 방향·손가락 추적 모델을 준비하고 있어요…");
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

  async function ensureMotionLandmarkers() {
    if (poseLandmarkerRef.current && handLandmarkerRef.current) return;
    updatePoseStatus("loading");
    const {
      FilesetResolver,
      HandLandmarker: HandLandmarkerClass,
      PoseLandmarker: PoseLandmarkerClass,
    } = await import("@mediapipe/tasks-vision");
    const vision = await FilesetResolver.forVisionTasks("/mediapipe-wasm");
    const poseBaseOptions = {
      modelAssetPath: "/models/pose_landmarker_lite.task",
    };
    const handBaseOptions = {
      modelAssetPath: "/models/hand_landmarker.task",
    };

    if (!poseLandmarkerRef.current) {
      try {
        poseLandmarkerRef.current = await PoseLandmarkerClass.createFromOptions(
          vision,
          {
            baseOptions: { ...poseBaseOptions, delegate: "GPU" },
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
            baseOptions: poseBaseOptions,
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputSegmentationMasks: false,
          },
        );
      }
    }

    if (!handLandmarkerRef.current) {
      try {
        handLandmarkerRef.current = await HandLandmarkerClass.createFromOptions(
          vision,
          {
            baseOptions: { ...handBaseOptions, delegate: "GPU" },
            runningMode: "VIDEO",
            numHands: 2,
            minHandDetectionConfidence: 0.45,
            minHandPresenceConfidence: 0.45,
            minTrackingConfidence: 0.45,
          },
        );
      } catch {
        handLandmarkerRef.current = await HandLandmarkerClass.createFromOptions(
          vision,
          {
            baseOptions: handBaseOptions,
            runningMode: "VIDEO",
            numHands: 2,
            minHandDetectionConfidence: 0.45,
            minHandPresenceConfidence: 0.45,
            minTrackingConfidence: 0.45,
          },
        );
      }
    }
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
    const headY = (landmarks[7].y + landmarks[8].y) / 2;
    return ankleY - headY >= 0.5;
  }

  function getHeadDirection(
    landmarks: NormalizedLandmark[],
  ): HeadDirection | null {
    const nose = landmarks[0];
    const leftEar = landmarks[7];
    const rightEar = landmarks[8];
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    if (
      !nose ||
      !leftEar ||
      !rightEar ||
      leftEar.visibility < 0.35 ||
      rightEar.visibility < 0.35
    ) {
      return null;
    }
    const centerX = (leftEar.x + rightEar.x) / 2;
    const centerY = (leftEar.y + rightEar.y) / 2;
    const shoulderWidth = Math.max(
      0.08,
      Math.abs(leftShoulder.x - rightShoulder.x),
    );
    const clamp = (value: number) => Math.max(-1, Math.min(1, value));
    return {
      yaw: clamp((nose.x - centerX) / (shoulderWidth * 0.32)),
      pitch: clamp((nose.y - centerY) / (shoulderWidth * 0.32)),
      roll: Math.atan2(
        rightEar.y - leftEar.y,
        rightEar.x - leftEar.x,
      ),
      centerX,
      centerY,
    };
  }

  function describeHeadDirection(head: HeadDirection | null) {
    if (!head) return "머리 방향 미확인";
    if (head.pitch < -0.32) return "위를 보는 중";
    if (head.pitch > 0.32) return "아래를 보는 중";
    if (head.yaw < -0.3) return "왼쪽을 보는 중";
    if (head.yaw > 0.3) return "오른쪽을 보는 중";
    return "정면을 보는 중";
  }

  function flushPoseFrames() {
    if (poseBufferRef.current.length === 0 || !poseSessionRef.current) {
      return poseWriteQueueRef.current;
    }
    const values = new Float32Array(poseBufferRef.current);
    const detectedFrames = chunkDetectedFramesRef.current;
    const fullBodyFrames = chunkFullBodyFramesRef.current;
    const handDetectedFrames = chunkHandFramesRef.current;
    const sessionId = poseSessionRef.current.id;
    poseBufferRef.current = [];
    chunkDetectedFramesRef.current = 0;
    chunkFullBodyFramesRef.current = 0;
    chunkHandFramesRef.current = 0;

    poseWriteQueueRef.current = poseWriteQueueRef.current.then(async () => {
      const session = poseSessionRef.current;
      if (!session || session.id !== sessionId) return;
      const updated = await appendMotionChunk(
        session,
        session.frameCount,
        values,
        detectedFrames,
        fullBodyFrames,
        handDetectedFrames,
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
    head: HeadDirection | null,
    hands: HandState,
  ) {
    const buffer = poseBufferRef.current;
    buffer.push(timestamp - sessionPerformanceStartRef.current);
    buffer.push(landmarks ? 1 : 0);
    buffer.push(fullBody ? 1 : 0);
    buffer.push(head?.yaw ?? Number.NaN);
    buffer.push(head?.pitch ?? Number.NaN);
    buffer.push(head?.roll ?? Number.NaN);
    buffer.push(hands.left ? 1 : 0);
    buffer.push(hands.right ? 1 : 0);
    buffer.push(hands.leftScore);
    buffer.push(hands.rightScore);

    if (landmarks) {
      for (let index = 11; index < 33; index += 1) {
        const point = landmarks[index];
        buffer.push(point.x, point.y, point.z, point.visibility);
      }
      chunkDetectedFramesRef.current += 1;
    } else {
      for (let index = 0; index < BODY_LANDMARK_COUNT * 4; index += 1) {
        buffer.push(Number.NaN);
      }
    }

    const pushHand = (hand: NormalizedLandmark[] | null) => {
      if (hand) {
        for (let index = 0; index < HAND_LANDMARK_COUNT; index += 1) {
          const point = hand[index];
          buffer.push(point.x, point.y, point.z);
        }
      } else {
        for (let index = 0; index < HAND_LANDMARK_COUNT * 3; index += 1) {
          buffer.push(Number.NaN);
        }
      }
    };
    pushHand(hands.left);
    pushHand(hands.right);

    if (fullBody) chunkFullBodyFramesRef.current += 1;
    if (hands.left || hands.right) chunkHandFramesRef.current += 1;

    const previous = poseStatsRef.current;
    const next: PoseStats = {
      ...previous,
      frames: previous.frames + 1,
      detectedFrames: previous.detectedFrames + (landmarks ? 1 : 0),
      fullBodyFrames: previous.fullBodyFrames + (fullBody ? 1 : 0),
      handDetectedFrames:
        previous.handDetectedFrames + (hands.left || hands.right ? 1 : 0),
      storageBytes: (previous.frames + 1) * MOTION_FRAME_STRIDE * 4,
    };
    poseStatsRef.current = next;
    if (timestamp - lastStatsUpdateRef.current >= 500) {
      lastStatsUpdateRef.current = timestamp;
      setPoseStats(next);
    }

    if (buffer.length / MOTION_FRAME_STRIDE >= CHUNK_FRAME_COUNT) {
      void flushPoseFrames();
    }
  }

  function poseTrackingLoop() {
    if (!trackingActiveRef.current) return;
    const video = processingVideoRef.current;
    const landmarker = poseLandmarkerRef.current;
    const handLandmarker = handLandmarkerRef.current;
    const timestamp = currentMonotonicTime();

    if (
      video &&
      landmarker &&
      handLandmarker &&
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
        const head = landmarks ? getHeadDirection(landmarks) : null;
        lastHeadDirectionRef.current = head;
        const nextHeadLabel = describeHeadDirection(head);
        setHeadDirectionLabel((current) =>
          current === nextHeadLabel ? current : nextHeadLabel,
        );

        if (timestamp - lastHandDetectionTimeRef.current >= 100) {
          lastHandDetectionTimeRef.current = timestamp;
          const handResult = handLandmarker.detectForVideo(video, timestamp);
          const hands: HandState = {
            left: null,
            right: null,
            leftScore: 0,
            rightScore: 0,
          };
          handResult.landmarks.forEach((hand, index) => {
            const category = handResult.handedness[index]?.[0];
            if (category?.categoryName === "Left") {
              hands.left = hand;
              hands.leftScore = category.score;
            } else if (category?.categoryName === "Right") {
              hands.right = hand;
              hands.rightScore = category.score;
            }
          });
          lastHandsRef.current = hands;
          const handCount = Number(Boolean(hands.left)) + Number(Boolean(hands.right));
          setDetectedHands((current) => (current === handCount ? current : handCount));
        }
        updatePoseStatus(landmarks ? (fullBody ? "full" : "partial") : "searching");

        const canvas = overlayCanvasRef.current;
        if (canvas) {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          const context = canvas.getContext("2d");
          context?.clearRect(0, 0, canvas.width, canvas.height);
          if (landmarks) {
            drawMotionSkeleton(
              canvas,
              landmarks,
              lastHandsRef.current.left,
              lastHandsRef.current.right,
              head,
              fullBody,
            );
          }
        }

        if (timestamp - lastSampleTimeRef.current >= 1000 / MOTION_SAMPLE_RATE) {
          lastSampleTimeRef.current = timestamp;
          recordPoseFrame(
            timestamp,
            landmarks,
            fullBody,
            head,
            lastHandsRef.current,
          );
        }
      } catch {
        updatePoseStatus("error");
      }
    }
    poseAnimationRef.current = requestAnimationFrame(poseTrackingLoop);
  }

  async function startPoseTracking() {
    try {
      await ensureMotionLandmarkers();
      const startedAt = currentEpochTime();
      const session = await createMotionSession(
        `motion-${crypto.randomUUID()}`,
        startedAt,
      );
      poseSessionRef.current = session;
      sessionPerformanceStartRef.current = currentMonotonicTime();
      lastDetectionTimeRef.current = 0;
      lastHandDetectionTimeRef.current = 0;
      lastSampleTimeRef.current = 0;
      lastVideoTimeRef.current = -1;
      lastPoseRef.current = null;
      lastHandsRef.current = {
        left: null,
        right: null,
        leftScore: 0,
        rightScore: 0,
      };
      lastHeadDirectionRef.current = null;
      poseBufferRef.current = [];
      chunkDetectedFramesRef.current = 0;
      chunkFullBodyFramesRef.current = 0;
      chunkHandFramesRef.current = 0;
      const emptyStats: PoseStats = {
        frames: 0,
        detectedFrames: 0,
        fullBodyFrames: 0,
        handDetectedFrames: 0,
        storageBytes: 0,
        startedAt,
      };
      poseStatsRef.current = emptyStats;
      setPoseStats(emptyStats);
      setElapsedSeconds(0);
      setDetectedHands(0);
      setHeadDirectionLabel("머리 방향 미확인");
      setSessionCount((count) => count + 1);
      updatePoseStatus("searching");
      trackingActiveRef.current = true;
      setCameraMessage("얼굴은 제외하고 몸·머리 방향·손가락 좌표를 기록하고 있어요.");
      setToast("몸과 손가락 좌표 상시 기록을 시작했어요");
      poseAnimationRef.current = requestAnimationFrame(poseTrackingLoop);
    } catch {
      updatePoseStatus("error");
      setCameraMessage("몸·손 추적 모델을 불러오지 못했어요. 다시 연결해 주세요.");
      setToast("동작 추적 모델을 준비하지 못했어요");
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
      const completed = await finishMotionSession(session, endedAt);
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
    const head = lastHeadDirectionRef.current;
    const hands = lastHandsRef.current;
    if (!session || cameraStatus !== "connected") {
      setToast("먼저 카메라를 연결해 주세요");
      return;
    }
    if (!landmarks || !head) {
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
      motionSnapshot: {
        body: landmarks.slice(11).flatMap((point) => [
          point.x,
          point.y,
          point.z,
          point.visibility,
        ]),
        leftHand: hands.left
          ? hands.left.flatMap((point) => [point.x, point.y, point.z])
          : null,
        rightHand: hands.right
          ? hands.right.flatMap((point) => [point.x, point.y, point.z])
          : null,
        head,
      },
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
      await downloadMotionSession(session);
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
            ? "동작 추적 모델 준비 중"
            : poseStatus === "error"
              ? "스켈레톤 인식 오류"
              : "좌표 기록 대기";

  const fullBodyRatio = poseStats.detectedFrames
    ? Math.round((poseStats.fullBodyFrames / poseStats.detectedFrames) * 100)
    : 0;
  const demoDay = demoWeek[selectedDemoDay];

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
                    <div className="tracking-readout">
                      <span>머리 · {headDirectionLabel}</span>
                      <span className={detectedHands > 0 ? "hands-found" : ""}>
                        손 · {detectedHands}/2 인식
                      </span>
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
                      <strong>몸·머리 방향·손가락 좌표 상시 기록</strong>
                      <p>{cameraMessage}</p>
                    </div>
                    <span className={cameraStatus === "connected" ? "recording" : ""}>
                      {cameraStatus === "connected" ? formatDuration(elapsedSeconds) : "대기"}
                    </span>
                  </div>
                  <div className="coordinate-stats">
                    <span><small>몸 관절</small><strong>22개</strong></span>
                    <span><small>손 관절</small><strong>최대 42개</strong></span>
                    <span><small>기록 속도</small><strong>{MOTION_SAMPLE_RATE} FPS</strong></span>
                    <span><small>누적 프레임</small><strong>{poseStats.frames.toLocaleString()}</strong></span>
                    <span><small>전신 인식률</small><strong>{fullBodyRatio}%</strong></span>
                    <span><small>예상 용량</small><strong>{formatBytes(poseStats.storageBytes)}</strong></span>
                  </div>
                  <div className="coordinate-actions">
                    <span>얼굴 특징·영상·음성 없이 동작 좌표만 이 브라우저에 저장 · 세션 {sessionCount}개</span>
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
            <section className="demo-switcher">
              <div>
                <span className="section-kicker">연구용 시뮬레이션</span>
                <h2>가상 페르소나의 일주일 관찰 결과</h2>
                <p>54세 여성 · 개인 카페 사장 · 실제 사용자 데이터가 아닌 예시입니다.</p>
              </div>
              <button
                type="button"
                className={demoMode ? "active" : ""}
                onClick={() => setDemoMode((enabled) => !enabled)}
                aria-pressed={demoMode}
              >
                <span aria-hidden="true" /> {demoMode ? "시뮬레이션 보는 중" : "시뮬레이션 보기"}
              </button>
            </section>

            {demoMode ? (
              <>
                <section className="persona-card">
                  <div className="persona-avatar" aria-hidden="true">카</div>
                  <div>
                    <span className="section-kicker">페르소나: 이수진 사장님</span>
                    <h2>일주일의 흐름에서 함께 나타난 변화</h2>
                    <p>평소에는 혼자 카페를 안정적으로 운영합니다. 이번 주 후반에는 깜빡함, 마감 반복 확인, 업무 중단, 단일 업무 지연이 함께 늘어나는 패턴을 가정했습니다.</p>
                  </div>
                  <span className="simulation-chip">가상 데이터</span>
                </section>

                <section className="panel week-observation">
                  <div className="panel-heading">
                    <div>
                      <span className="section-kicker">7일 행동 흐름</span>
                      <h2>하루를 선택해 상세 기록 보기</h2>
                    </div>
                    <span className="week-range">8월 4일–10일</span>
                  </div>
                  <div className="week-days" role="tablist" aria-label="가상 관찰 날짜">
                    {demoWeek.map((day, index) => {
                      const intensity = Math.max(
                        day.safetyAlerts * 2 + day.doubleChecks + day.unfinishedTasks + Math.round(day.microDelay / 6) - 1,
                        0,
                      );
                      return (
                        <button
                          key={day.day}
                          className={selectedDemoDay === index ? "selected" : ""}
                          type="button"
                          role="tab"
                          aria-selected={selectedDemoDay === index}
                          onClick={() => setSelectedDemoDay(index)}
                        >
                          <span>{day.day}</span>
                          <small>{day.date}</small>
                          <i className={`signal-${Math.min(intensity, 5)}`} aria-label={`관찰 신호 ${intensity}단계`} />
                        </button>
                      );
                    })}
                  </div>
                  <div className="day-detail" role="tabpanel">
                    <div className="day-detail-head">
                      <span>{demoDay.day}요일 · {demoDay.date}</span>
                      <strong>{demoDay.note}</strong>
                    </div>
                    <div className="day-signal-grid">
                      <article><span>안전 알림</span><strong>{demoDay.safetyAlerts}<small>회</small></strong></article>
                      <article><span>마감 반복 확인</span><strong>{demoDay.doubleChecks}<small>회</small></strong></article>
                      <article><span>업무 미완료</span><strong>{demoDay.unfinishedTasks}<small>건</small></strong></article>
                      <article><span>미세 지연</span><strong>{demoDay.microDelay}<small>%</small></strong></article>
                    </div>
                    <ul className="day-examples">
                      {demoDay.examples.map((example) => <li key={example}>{example}</li>)}
                    </ul>
                  </div>
                </section>

                <section className="insight-grid">
                  <article className="insight-card observe">
                    <span className="insight-icon" aria-hidden="true">↗</span>
                    <div>
                      <span className="section-kicker">1. 평소 대비 변화</span>
                      <h3>후반 4일에 신호가 겹쳐요</h3>
                      <p>수–일에는 안전 알림 3회, 반복 확인 7회, 미완료 업무 3건이 함께 나타납니다. 한 가지 실수보다 여러 일상 지표가 같은 기간에 변하는지를 봅니다.</p>
                    </div>
                  </article>
                  <article className="insight-card timeline-insight">
                    <span className="insight-icon" aria-hidden="true">⌁</span>
                    <div>
                      <span className="section-kicker">2. 시간축 연결</span>
                      <h3>바쁜 시간대의 흐름을 확인해요</h3>
                      <p>금·토 오후에는 주문 입력 중단과 음료 세팅 지연이 가까운 시간대에 기록됩니다. 타임라인은 “언제 일이 끊겼는지”를 되짚게 합니다.</p>
                    </div>
                  </article>
                  <article className="insight-card care-insight">
                    <span className="insight-icon" aria-hidden="true">♡</span>
                    <div>
                      <span className="section-kicker">3. 케어로 연결</span>
                      <h3>진단 대신 휴식과 점검을 권해요</h3>
                      <p>이 예시만으로 건강 상태를 판단하지 않습니다. 다만 변화가 이어질 때는 휴식, 점검 루틴, 필요 시 전문 상담을 조심스럽게 권할 수 있습니다.</p>
                    </div>
                  </article>
                </section>

                <section className="simulation-note">
                  <strong>이 데모가 보여주는 범위</strong>
                  <p>이수진 사장님의 데이터는 설명을 위한 가상 시나리오입니다. 메모리 가드는 질환을 진단하거나 단정하지 않으며, 실제 서비스에서는 장기간의 개인 기준선과 안전·업무 변화 패턴을 함께 살펴 케어 대화를 돕는 용도로 사용합니다.</p>
                </section>
              </>
            ) : (
              <>
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
              </>
            )}
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
              {selectedEvent.motionSnapshot ? (
                <div className="pose-snapshot">
                  <PoseSnapshot snapshot={selectedEvent.motionSnapshot} />
                  <span>몸 22개·손 최대 42개 관절과 머리 방향으로 복원한 장면이에요.</span>
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
