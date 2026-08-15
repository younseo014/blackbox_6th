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
  deleteAllMotionSessions,
  downloadMotionSession,
  extractHandPointTrajectory,
  finishMotionSession,
  getSessionFrames,
  listMotionSessions,
  type MotionSessionRecord,
} from "./pose-store";
import {
  describeHeadDirection,
  getHeadDirection,
  isFullBodyVisible,
  computeHandMotionVariability,
  computeMovementSmoothness,
  type HeadDirection,
} from "./motion-analysis";
import { drawMotionSkeleton, type MotionSnapshot } from "./skeleton-draw";
import { SessionReplayPanel } from "./session-replay";
import { generateEventMotion, DEMO_MOTION_LABELS } from "./demo-motion";
import { detectMotionEvents, motionSamplesFromRawFrames } from "./motion-detection";
import {
  computeBaseline,
  detectChangeSignal,
  summarizeLog,
  type DailyLog,
} from "./care-metrics";
import {
  deleteAllCareLogs,
  estimateStorageUsage,
  getConsent,
  listRecentLogs,
  recordDoubleCheck,
  recordMicroDelay,
  recordSafetyAlert,
  recordTaskCompleted,
  recordTaskStarted,
  setBusyLevel as persistBusyLevel,
  setConsent,
  todayDateKey,
  type ConsentState,
} from "./metrics-store";
import {
  DEMO_PERSONAS,
  explainDemoEvent,
  signalLevelLabel,
  type DemoDay,
  type DemoEvent,
  type DemoPersona,
} from "./demo-personas";
import {
  OCCUPATION_TEMPLATES,
  getOccupationTemplate,
  phaseForHour,
  type OccupationId,
  type WorkPhase,
} from "./occupation-templates";
import {
  DEFAULT_PROFILE,
  buildBaseline,
  createObservationEpisode,
  extractObservationFeatures,
  type BaselineSnapshot,
  type ObservationEpisode,
  type ObservationMode,
  type ObservationProfile,
} from "./observation-engine";
import {
  deleteAllObservationData,
  getObservationProfile,
  listObservationEpisodes,
  saveObservationEpisode,
  saveObservationEpisodes,
  saveObservationProfile,
} from "./observation-store";
import {
  createSyntheticTrainingDataset,
  getSyntheticTrainingClips,
  type SyntheticTrainingClip,
} from "./synthetic-training";

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

type HandState = {
  left: NormalizedLandmark[] | null;
  right: NormalizedLandmark[] | null;
  leftScore: number;
  rightScore: number;
};

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

// Kept as a top-level helper (rather than calling Date.now() inline inside
// the component) so the React Compiler's purity check doesn't flag it -
// same reasoning as currentEpochTime() above.
function nowMs() {
  return Date.now();
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

function formatSessionTime(epochMs: number) {
  return new Intl.DateTimeFormat("ko-KR", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(epochMs));
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
  const [recentSessions, setRecentSessions] = useState<MotionSessionRecord[]>([]);
  const [replaySessionId, setReplaySessionId] = useState<string | null>(null);
  const [demoReplay, setDemoReplay] = useState<{
    key: string;
    label: string;
    frames: number[][];
    detectionExplanation?: DemoDetectionExplanation;
  } | null>(null);
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
  const [selectedPersonaIndex, setSelectedPersonaIndex] = useState(0);
  const [selectedDemoDay, setSelectedDemoDay] = useState(6);
  const [observationProfile, setObservationProfile] = useState<ObservationProfile>(DEFAULT_PROFILE);
  const [observationEpisodes, setObservationEpisodes] = useState<ObservationEpisode[]>([]);
  const [observationBaseline, setObservationBaseline] = useState<BaselineSnapshot>(() => buildBaseline([], 1));
  const [zoneSetupOpen, setZoneSetupOpen] = useState(false);
  const [selectedZoneId, setSelectedZoneId] = useState<string>("DRINK_PREP");
  const [syntheticLibraryOpen, setSyntheticLibraryOpen] = useState(false);
  const [syntheticLibraryPhase, setSyntheticLibraryPhase] = useState<WorkPhase>("business");
  const [applyingSyntheticBaseline, setApplyingSyntheticBaseline] = useState(false);

  // --- Consent, real observation metrics, and data controls ---
  // Lazy initializer instead of an effect: getConsent() is SSR-safe (it
  // checks `typeof window` itself and falls back to the default), so there's
  // no need to synchronize it via setState in an effect after mount.
  const [consent, setConsentState] = useState<ConsentState>(() => getConsent());
  const [showConsentModal, setShowConsentModal] = useState(false);
  const [careLogs, setCareLogs] = useState<DailyLog[]>([]);
  const [todayBusyLevel, setTodayBusyLevel] = useState<"quiet" | "normal" | "busy">(
    "normal",
  );
  const [myDataOpen, setMyDataOpen] = useState(false);
  const [storageUsage, setStorageUsage] = useState<{
    usageBytes: number;
    quotaBytes: number;
  } | null>(null);
  const [motionSignal, setMotionSignal] = useState<{
    variability: number | null;
    smoothness: number | null;
  } | null>(null);
  const pendingCameraStartRef = useRef(false);
  const closingDoneTodayRef = useRef(false);
  const bookingShownAtRef = useRef<number | null>(null);
  const lastTestEventAtRef = useRef<number | null>(null);
  const savepointStartRecordedRef = useRef(false);

  async function refreshCareData() {
    try {
      const logs = await listRecentLogs(28);
      setCareLogs(logs);
      const today = logs.find((log) => log.date === todayDateKey());
      if (today) setTodayBusyLevel(today.busyLevel);
    } catch {
      // local-only storage; ignore transient read errors
    }
  }

  async function refreshObservationData(profile = observationProfile) {
    try {
      const episodes = await listObservationEpisodes(500);
      setObservationEpisodes(episodes);
      const eligible = episodes.filter(
        (episode) =>
          episode.occupation === profile.occupation &&
          episode.baselineVersion === profile.baselineVersion,
      );
      setObservationBaseline(buildBaseline(eligible, profile.baselineVersion));
    } catch {
      // Device-local observation data; keep the current UI state on transient errors.
    }
  }

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
        setRecentSessions(sessions.slice(0, 5));
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    getObservationProfile()
      .then(async (profile) => {
        const saved = await saveObservationProfile(profile);
        setObservationProfile(saved);
        const template = getOccupationTemplate(saved.occupation);
        setSelectedZoneId(template.zones[0]?.id ?? "");
        const episodes = await listObservationEpisodes(500);
        setObservationEpisodes(episodes);
        setObservationBaseline(buildBaseline(
          episodes.filter(
            (episode) =>
              episode.occupation === saved.occupation &&
              episode.baselineVersion === saved.baselineVersion,
          ),
          saved.baselineVersion,
        ));
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    listRecentLogs(28)
      .then((logs) => {
        setCareLogs(logs);
        const today = logs.find((log) => log.date === todayDateKey());
        if (today) setTodayBusyLevel(today.busyLevel);
      })
      .catch(() => undefined);
    estimateStorageUsage().then((usage) => {
      setStorageUsage(usage);
      if (usage && usage.quotaBytes > 0 && usage.usageBytes / usage.quotaBytes > 0.9) {
        setToast(
          "브라우저 저장 공간이 거의 찼어요. 내 데이터 관리에서 오래된 기록을 정리해 주세요.",
        );
      }
    });
  }, []);

  useEffect(() => {
    if (
      consent.observationConsent &&
      savepointOpen &&
      !savepointStartRecordedRef.current
    ) {
      savepointStartRecordedRef.current = true;
      bookingShownAtRef.current = nowMs();
      void recordTaskStarted().then(refreshCareData);
    }
  }, [consent.observationConsent, savepointOpen]);

  useEffect(() => {
    if (!consent.observationConsent || !latestSession || latestSession.frameCount < 20) {
      return;
    }
    let cancelled = false;
    getSessionFrames(latestSession.id)
      .then((frames) => {
        if (cancelled) return;
        const rightHandTrajectory = extractHandPointTrajectory(
          frames,
          "right",
          8, // index fingertip
        );
        setMotionSignal({
          variability: computeHandMotionVariability(rightHandTrajectory),
          smoothness: computeMovementSmoothness(rightHandTrajectory),
        });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [consent.observationConsent, latestSession]);

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

  function requestCameraStart() {
    if (!consent.decided) {
      pendingCameraStartRef.current = true;
      setShowConsentModal(true);
      return;
    }
    void startCamera();
  }

  function handleConsentDecision(observationConsent: boolean) {
    const next = setConsent(observationConsent);
    setConsentState(next);
    setShowConsentModal(false);
    if (observationConsent && pendingCameraStartRef.current) {
      void startCamera();
    }
    pendingCameraStartRef.current = false;
  }

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

  // Analyzes a just-finished session's real coordinates with the same
  // motion detector used for the demo persona replay (app/motion-detection.ts)
  // and records ANY detected events into today's log - ALONGSIDE the
  // existing button/timer signals (recordDoubleCheck/recordSafetyAlert
  // elsewhere in this file), not instead of them. Best-effort: a failed
  // analysis shouldn't block ending the camera session.
  async function recordMotionDetections(sessionId: string) {
    try {
      const rawFrames = await getSessionFrames(sessionId);
      const samples = motionSamplesFromRawFrames(rawFrames);
      const detections = detectMotionEvents(samples);
      for (const detection of detections) {
        if (detection.type === "safety_alert") {
          await recordSafetyAlert();
        } else if (detection.type === "double_check") {
          await recordDoubleCheck();
        } else {
          await recordMicroDelay((detection.endMs - detection.startMs) / 1000);
        }
      }
    } catch {
      // local-only analysis; ignore transient read errors
    }
  }

  async function recordObservationSession(sessionId: string, recordedAt: number) {
    try {
      const rawFrames = await getSessionFrames(sessionId);
      const features = extractObservationFeatures(rawFrames, observationProfile.zoneGrid);
      const taskOverride = observationProfile.activeTestTaskId
        ? getOccupationTemplate(observationProfile.occupation).tasks.find(
            (task) => task.id === observationProfile.activeTestTaskId,
          )
        : undefined;
      const episode = createObservationEpisode({
        sessionId,
        recordedAt,
        profile: observationProfile,
        phase: taskOverride?.phase ?? phaseForHour(new Date(recordedAt).getHours()),
        features,
        baseline: observationBaseline,
        taskOverride,
      });
      await saveObservationEpisode(episode);
      await refreshObservationData(observationProfile);
      setToast(
        episode.disposition === "quarantined"
          ? "평소 흐름으로 확정하기 어려운 동작은 학습에서 잠시 보류했어요"
          : observationProfile.mode === "learning"
            ? `${episode.taskLabel} 패턴을 학습 기록에 추가했어요`
            : taskOverride
              ? `${episode.taskLabel} 성능 테스트를 개인 기준과 비교했어요`
              : `${episode.taskLabel} 동작을 개인 기준과 비교했어요`,
      );
    } catch {
      // Coordinate recording remains available even if contextual analysis fails.
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
      setRecentSessions((previous) => [completed, ...previous.filter((s) => s.id !== completed.id)].slice(0, 5));
      poseSessionRef.current = null;
      if (consent.observationConsent && completed.frameCount >= 20) {
        void recordMotionDetections(completed.id).then(refreshCareData);
        void recordObservationSession(completed.id, endedAt);
      }
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

    // Gap between consecutive logged moments is used as a rough proxy for
    // the "미세 지연" (micro-delay) observation metric from the PRD. This is
    // a coarse stand-in, not a precise task-timer.
    if (consent.observationConsent) {
      const now = nowMs();
      if (lastTestEventAtRef.current !== null) {
        void recordMicroDelay((now - lastTestEventAtRef.current) / 1000).then(
          refreshCareData,
        );
      }
      lastTestEventAtRef.current = now;
    }
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
    if (closingDoneTodayRef.current && consent.observationConsent) {
      // Re-running the closing check after it was already marked done today
      // is exactly the "마감 반복 확인 (Double Check)" pattern from the PRD.
      void recordDoubleCheck().then(refreshCareData);
    }
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
          if (consent.observationConsent) {
            void recordSafetyAlert().then(refreshCareData);
          }
        } else {
          completeClosing();
        }
      }
    }, 650);
  }

  function completeClosing() {
    setClosingStatus("done");
    setClosingStep(3);
    closingDoneTodayRef.current = true;
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

  function openDemoEventReplay(
    persona: DemoPersona,
    day: DemoDay,
    dayIndex: number,
    exampleIndex: number,
    example: DemoEvent,
  ) {
    // Deterministic per (persona, day, event) - the same event always opens
    // the same clip, but distinct events don't all play back identically.
    const seed = dayIndex * 4 + exampleIndex;
    const frames = generateEventMotion(example.motionType, seed);
    setDemoReplay({
      key: `${persona.id}-${dayIndex}-${exampleIndex}`,
      label: `${persona.name} · ${example.label} · ${DEMO_MOTION_LABELS[example.motionType]}`,
      frames,
      detectionExplanation: explainDemoEvent(persona, dayIndex, day, example, frames),
    });
  }

  function changeBusyLevel(level: "quiet" | "normal" | "busy") {
    setTodayBusyLevel(level);
    if (consent.observationConsent) {
      void persistBusyLevel(level).then(refreshCareData);
    }
  }

  async function changeOccupation(occupation: OccupationId) {
    const template = getOccupationTemplate(occupation);
    const next = await saveObservationProfile({
      ...observationProfile,
      occupation,
      mode: "learning",
      learningStartedAt: nowMs(),
      baselineVersion: observationProfile.baselineVersion + 1,
      baselineSource: "real",
      syntheticDatasetId: null,
      activeTestTaskId: null,
      zoneGrid: Array(9).fill(null),
    });
    setObservationProfile(next);
    setSelectedZoneId(template.zones[0]?.id ?? "");
    await refreshObservationData(next);
    setToast(`${template.label} 기본 업무 맥락으로 새 학습을 시작했어요`);
  }

  async function changeObservationMode(mode: ObservationMode) {
    if (mode === observationProfile.mode) return;
    const next = await saveObservationProfile({
      ...observationProfile,
      mode,
      learningStartedAt:
        mode === "learning" ? nowMs() : observationProfile.learningStartedAt,
      baselineVersion:
        mode === "learning"
          ? observationProfile.baselineVersion + 1
          : observationProfile.baselineVersion,
      baselineSource: mode === "learning" ? "real" : observationProfile.baselineSource,
      syntheticDatasetId: mode === "learning" ? null : observationProfile.syntheticDatasetId,
      activeTestTaskId: mode === "learning" ? null : observationProfile.activeTestTaskId,
    });
    setObservationProfile(next);
    await refreshObservationData(next);
    setToast(
      mode === "learning"
        ? "기존 기준선은 보관하고 새로운 평소 흐름을 학습해요"
        : observationBaseline.confidence < 70
          ? "기록이 아직 적어 임시 기준으로 분석을 시작해요"
          : "학습한 개인 업무 패턴을 기준으로 분석을 시작해요",
    );
  }

  async function assignZoneCell(index: number) {
    const zoneGrid = [...observationProfile.zoneGrid];
    zoneGrid[index] = zoneGrid[index] === selectedZoneId ? null : selectedZoneId;
    const next = await saveObservationProfile({ ...observationProfile, zoneGrid });
    setObservationProfile(next);
  }

  async function applySyntheticTrainingBaseline() {
    if (applyingSyntheticBaseline) return;
    setApplyingSyntheticBaseline(true);
    try {
      const baselineVersion = observationProfile.baselineVersion + 1;
      const dataset = createSyntheticTrainingDataset(
        observationProfile.occupation,
        baselineVersion,
        nowMs(),
      );
      await saveObservationEpisodes(dataset.episodes);
      const next = await saveObservationProfile({
        ...observationProfile,
        mode: "analysis",
        learningStartedAt: dataset.generatedAt - 13 * 24 * 60 * 60 * 1000,
        baselineVersion,
        baselineSource: "synthetic",
        syntheticDatasetId: dataset.id,
        activeTestTaskId: null,
        zoneGrid: dataset.zoneGrid,
      });
      setObservationProfile(next);
      await refreshObservationData(next);
      setToast(`${getOccupationTemplate(next.occupation).label} 2주 가상 기준선이 100% 적용됐어요`);
    } finally {
      setApplyingSyntheticBaseline(false);
    }
  }

  function openSyntheticClip(clip: SyntheticTrainingClip) {
    setDemoReplay({
      key: clip.id,
      label: `${occupationTemplate.label} · ${clip.taskLabel} · 정상 학습 예시`,
      frames: clip.frames,
    });
  }

  async function selectPerformanceTestTask(clip: SyntheticTrainingClip) {
    const next = await saveObservationProfile({
      ...observationProfile,
      mode: "analysis",
      activeTestTaskId: clip.taskType,
    });
    setObservationProfile(next);
    setSyntheticLibraryOpen(false);
    setToast(`${clip.taskLabel} 테스트 준비 완료 · 카메라 앞에서 동작 후 기록을 종료해 주세요`);
  }

  async function clearPerformanceTestTask() {
    const next = await saveObservationProfile({
      ...observationProfile,
      activeTestTaskId: null,
    });
    setObservationProfile(next);
    setToast("지정 동작 테스트를 종료했어요");
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
    if (consent.observationConsent) {
      void recordTaskCompleted().then(refreshCareData);
      if (bookingShownAtRef.current !== null) {
        void recordMicroDelay((nowMs() - bookingShownAtRef.current) / 1000);
      }
    }
  }

  function dismissSavepoint() {
    setSavepointOpen(false);
    setToast("나중에 다시 확인할 수 있어요. 세이브포인트는 그대로 남아있어요.");
    // Left uncompleted on purpose: this is what "업무 누락율 (Drop Rate)"
    // is meant to observe - a started task that never got finished.
  }

  function openTimelineEvent(event: TimelineEvent) {
    setSelectedEvent(event);
  }

  async function deleteAllMyData() {
    // Stop any in-flight recording first and let its queued writes settle.
    // Deleting while a session is still actively being written would let
    // the next scheduled flush silently recreate a "1 session" row right
    // after the wipe, which is confusing and defeats the point of "삭제".
    if (cameraStatus === "connected") {
      await stopCamera();
    }
    await Promise.all([
      deleteAllMotionSessions(),
      deleteAllCareLogs(),
      deleteAllObservationData(),
    ]);
    setSessionCount(0);
    setLatestSession(null);
    setRecentSessions([]);
    setReplaySessionId(null);
    setMotionSignal(null);
    setCareLogs([]);
    const resetProfile = {
      ...DEFAULT_PROFILE,
      learningStartedAt: nowMs(),
      updatedAt: nowMs(),
    };
    setObservationProfile(resetProfile);
    setObservationEpisodes([]);
    setObservationBaseline(buildBaseline([], resetProfile.baselineVersion));
    setPoseStats({
      frames: 0,
      detectedFrames: 0,
      fullBodyFrames: 0,
      handDetectedFrames: 0,
      storageBytes: 0,
      startedAt: null,
    });
    const usage = await estimateStorageUsage();
    setStorageUsage(usage);
    setToast("저장된 동작 좌표와 케어 기록을 모두 삭제했어요");
  }

  function withdrawObservationConsent() {
    const next = setConsent(false);
    setConsentState(next);
    setToast("관찰 참여를 철회했어요. 매장 안전 기능은 계속 사용할 수 있어요.");
  }

  async function shareCareSummary() {
    const recent = careLogs.slice(0, 7);
    const safetyAlerts = recent.reduce((sum, log) => sum + log.safetyAlerts, 0);
    const doubleChecks = recent.reduce((sum, log) => sum + log.doubleChecks, 0);
    const summaryText =
      `메모리 가드 케어 요약 (최근 ${recent.length || 0}일)\n` +
      `안전 알림 ${safetyAlerts}회 · 마감 반복 확인 ${doubleChecks}회\n` +
      `이 요약은 진단이 아니며 참고용 케어 정보입니다.`;

    try {
      if (navigator.share) {
        await navigator.share({ title: "메모리 가드 케어 요약", text: summaryText });
        setToast("공유 창을 열었어요");
        return;
      }
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(summaryText);
        setToast("케어 요약을 클립보드에 복사했어요");
        return;
      }
      setToast("이 브라우저에서는 공유를 지원하지 않아요");
    } catch {
      setToast("공유를 취소했어요");
    }
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
  const activePersona = DEMO_PERSONAS[selectedPersonaIndex] ?? DEMO_PERSONAS[0];
  const demoDay = activePersona.week[selectedDemoDay] ?? activePersona.week[0];

  const recentCareLogs = careLogs.slice(0, 7);
  const careBaseline = computeBaseline(careLogs);
  const changeSignal = detectChangeSignal(recentCareLogs, careBaseline);
  const todaySummary = careLogs[0] ? summarizeLog(careLogs[0]) : null;
  const totalSafetyAlerts = recentCareLogs.reduce((sum, log) => sum + log.safetyAlerts, 0);
  const totalDoubleChecks = recentCareLogs.reduce((sum, log) => sum + log.doubleChecks, 0);
  const totalDroppedTasks = recentCareLogs.reduce(
    (sum, log) => sum + Math.max(0, log.tasksStarted - log.tasksCompleted),
    0,
  );
  const occupationTemplate = getOccupationTemplate(observationProfile.occupation);
  const syntheticTrainingClips = getSyntheticTrainingClips(observationProfile.occupation);
  const visibleSyntheticClips = syntheticTrainingClips.filter(
    (clip) => clip.phase === syntheticLibraryPhase,
  );
  const activeTestClip = observationProfile.activeTestTaskId
    ? syntheticTrainingClips.find((clip) => clip.taskType === observationProfile.activeTestTaskId) ?? null
    : null;
  const currentObservationEpisodes = observationEpisodes.filter(
    (episode) =>
      episode.occupation === observationProfile.occupation &&
      episode.baselineVersion === observationProfile.baselineVersion,
  );
  const acceptedObservationCount = currentObservationEpisodes.filter(
    (episode) => episode.disposition === "accepted",
  ).length;
  const quarantinedObservationCount = currentObservationEpisodes.filter(
    (episode) => episode.disposition === "quarantined",
  ).length;
  const mappedZoneCount = new Set(observationProfile.zoneGrid.filter(Boolean)).size;
  const latestObservationEpisode = currentObservationEpisodes[0] ?? null;
  const latestPerformanceTest = currentObservationEpisodes.find(
    (episode) => episode.source === "performance_test",
  ) ?? null;
  const latestPerformanceBaseline = latestPerformanceTest
    ? observationBaseline.tasks.find((task) => task.taskType === latestPerformanceTest.taskType) ?? null
    : null;
  const analysisObservationSignals = currentObservationEpisodes.filter(
    (episode) =>
      episode.mode === "analysis" &&
      episode.source !== "performance_test" &&
      episode.disposition !== "excluded" &&
      ((episode.durationZScore ?? 0) >= 1.5 || (episode.pauseZScore ?? 0) >= 1.5),
  );

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
        <button
          type="button"
          className="my-data-entry"
          onClick={() => setMyDataOpen(true)}
        >
          <span aria-hidden="true">⚙</span> 내 데이터 관리
        </button>

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

            <section className={`observation-mode-card mode-${observationProfile.mode}`}>
              <div className="mode-card-heading">
                <div className="occupation-select-wrap">
                  <span className="occupation-icon" aria-hidden="true">{occupationTemplate.icon}</span>
                  <label>
                    <span>나의 업무 환경</span>
                    <select
                      value={observationProfile.occupation}
                      onChange={(event) => void changeOccupation(event.target.value as OccupationId)}
                      aria-label="직업 선택"
                    >
                      {OCCUPATION_TEMPLATES.map((template) => (
                        <option key={template.id} value={template.id}>{template.label}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="mode-toggle" role="group" aria-label="관찰 모드">
                  <button
                    type="button"
                    className={observationProfile.mode === "learning" ? "active" : ""}
                    onClick={() => void changeObservationMode("learning")}
                  >
                    학습 모드
                  </button>
                  <button
                    type="button"
                    className={observationProfile.mode === "analysis" ? "active" : ""}
                    onClick={() => void changeObservationMode("analysis")}
                  >
                    분석 모드
                  </button>
                </div>
              </div>
              <div className="mode-card-body">
                <div className="mode-copy">
                  <span className="mode-status-dot" aria-hidden="true" />
                  <div>
                    <strong>
                      {observationProfile.mode === "learning"
                        ? "평소 업무 흐름을 알아가고 있어요"
                        : "개인 업무 패턴과 비교하고 있어요"}
                    </strong>
                    <p>
                      {observationProfile.mode === "learning"
                        ? "확정하기 어려운 행동은 기준선에 넣지 않고 잠시 보류해요."
                        : `기준선 v${observationProfile.baselineVersion}을 고정해 최근 동작의 변화를 살펴봐요.`}
                    </p>
                  </div>
                  {observationProfile.baselineSource === "synthetic" && (
                    <span className="synthetic-baseline-chip">가상 기준선 적용 중</span>
                  )}
                </div>
                <div className="learning-progress" aria-label={`기준선 완성도 ${observationBaseline.confidence}%`}>
                  <div><span>개인 기준선 완성도</span><strong>{observationBaseline.confidence}%</strong></div>
                  <i><span style={{ width: `${observationBaseline.confidence}%` }} /></i>
                </div>
                <div className="mode-stats">
                  <span><small>학습 포함</small><strong>{acceptedObservationCount}건</strong></span>
                  <span><small>학습 보류</small><strong>{quarantinedObservationCount}건</strong></span>
                  <span><small>익힌 업무</small><strong>{observationBaseline.tasks.length}개</strong></span>
                  <span><small>설정한 구역</small><strong>{mappedZoneCount}개</strong></span>
                </div>
                <div className="mode-card-actions">
                  <p>{occupationTemplate.description}</p>
                  <div className="mode-action-buttons">
                    <button
                      type="button"
                      className="synthetic-apply-button"
                      onClick={() => void applySyntheticTrainingBaseline()}
                      disabled={applyingSyntheticBaseline}
                    >
                      {applyingSyntheticBaseline ? "2주 데이터 적용 중…" : "가상 학습 데이터 적용"}
                    </button>
                    <button type="button" onClick={() => setSyntheticLibraryOpen(true)}>가상 학습 데이터 보기</button>
                    <button type="button" onClick={() => setZoneSetupOpen(true)}>매장 구역 설정</button>
                  </div>
                </div>
                {activeTestClip && (
                  <div className="active-performance-test" role="status">
                    <div>
                      <span>성능 테스트 동작</span>
                      <strong>{activeTestClip.taskLabel}</strong>
                      <small>{activeTestClip.instruction}</small>
                    </div>
                    <div>
                      <button type="button" onClick={() => openSyntheticClip(activeTestClip)}>동작 다시 보기</button>
                      <button type="button" onClick={() => void clearPerformanceTestTask()}>테스트 해제</button>
                    </div>
                  </div>
                )}
                {latestObservationEpisode && (
                  <div className={`latest-learning-result disposition-${latestObservationEpisode.disposition}`}>
                    <span>최근 관찰 · {latestObservationEpisode.taskLabel}</span>
                    <strong>{latestObservationEpisode.dispositionReason}</strong>
                  </div>
                )}
                {latestPerformanceTest && latestPerformanceBaseline && (
                  <div className={`performance-test-result ${(latestPerformanceTest.durationZScore ?? 0) >= 1.5 ? "changed" : "within"}`}>
                    <div className="performance-result-heading">
                      <span>최근 성능 테스트 · {latestPerformanceTest.taskLabel}</span>
                      <strong>{(latestPerformanceTest.durationZScore ?? 0) >= 1.5 ? "평소 범위를 벗어난 변화 후보" : "가상 기준 범위 안"}</strong>
                    </div>
                    <div className="performance-result-metrics">
                      <span><small>가상 기준 평균</small><strong>{latestPerformanceBaseline.meanDuration.toFixed(1)}초</strong></span>
                      <span><small>이번 수행</small><strong>{latestPerformanceTest.features.durationSeconds.toFixed(1)}초</strong></span>
                      <span><small>가장 긴 멈춤</small><strong>{latestPerformanceTest.features.longestPauseSeconds.toFixed(1)}초</strong></span>
                    </div>
                    <p>{(latestPerformanceTest.durationZScore ?? 0) >= 1.5 ? "한 번의 결과로 판단하지 않고 같은 동작을 반복했을 때도 이어지는지 확인해 주세요." : "동작 속도와 흐름이 2주 가상 기준선의 일반적인 범위와 비슷해요."}</p>
                  </div>
                )}
              </div>
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
                        onClick={requestCameraStart}
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
                <div className="savepoint-actions">
                  <button
                    className="text-button"
                    type="button"
                    onClick={dismissSavepoint}
                  >
                    나중에 할게요
                  </button>
                  <button type="button" onClick={() => setBookingOpen(true)}>
                    이어서 하기 <span aria-hidden="true">›</span>
                  </button>
                </div>
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

              <div className="busy-level-picker">
                <span>오늘 매장 분위기</span>
                <div role="group" aria-label="오늘 매장 분위기">
                  {(
                    [
                      { key: "quiet", label: "한산" },
                      { key: "normal", label: "보통" },
                      { key: "busy", label: "바쁨" },
                    ] as const
                  ).map((option) => (
                    <button
                      key={option.key}
                      type="button"
                      className={todayBusyLevel === option.key ? "active" : ""}
                      onClick={() => changeBusyLevel(option.key)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
                <small>
                  바쁜 날은 케어 리포트에서 따로 표시해, 손님이 많아 생긴
                  변화와 패턴 변화를 구분하는 데 참고해요.
                </small>
              </div>

              <button
                className="closing-button"
                type="button"
                onClick={startClosingCheck}
                disabled={closingStatus === "checking"}
              >
                {closingStatus === "checking"
                  ? `안전 확인 중 ${closingStep}/3`
                  : closingStatus === "done"
                    ? "마감 완료 · 다시 확인하기"
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
                <p>실제 사용자 데이터가 아닌 예시입니다. 서로 다른 4가지 결과를 보여주는 페르소나 중 하나를 골라보세요.</p>
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
                <div className="persona-switcher" role="tablist" aria-label="가상 페르소나 선택">
                  {DEMO_PERSONAS.map((persona, index) => (
                    <button
                      key={persona.id}
                      type="button"
                      role="tab"
                      aria-selected={selectedPersonaIndex === index}
                      className={selectedPersonaIndex === index ? "selected" : ""}
                      onClick={() => {
                        setSelectedPersonaIndex(index);
                        setSelectedDemoDay(6);
                      }}
                    >
                      <span className={`persona-tab-avatar level-${persona.signal.level}`} aria-hidden="true">
                        {persona.avatarLabel}
                      </span>
                      <span className="persona-tab-copy">
                        <strong>{persona.name}</strong>
                        <small>{signalLevelLabel(persona.signal.level)}</small>
                      </span>
                    </button>
                  ))}
                </div>

                <section className="persona-card">
                  <div className="persona-avatar" aria-hidden="true">{activePersona.avatarLabel}</div>
                  <div>
                    <span className="section-kicker">페르소나: {activePersona.name}</span>
                    <h2>{activePersona.tagline}</h2>
                    <p>{activePersona.summary}</p>
                  </div>
                  <span className="simulation-chip">가상 데이터</span>
                </section>

                <section className={`signal-result level-${activePersona.signal.level}`}>
                  <div className="signal-result-head">
                    <span className={`signal-pill level-${activePersona.signal.level}`}>
                      {signalLevelLabel(activePersona.signal.level)}
                    </span>
                    <small>실제 케어 리포트와 같은 계산 로직(app/care-metrics.ts)으로 이 가상 데이터를 분석한 결과예요.</small>
                  </div>
                  <ul>
                    {activePersona.signal.reasons.map((reason) => <li key={reason}>{reason}</li>)}
                  </ul>
                  {activePersona.signal.confoundNote && (
                    <p className="confound-note">{activePersona.signal.confoundNote}</p>
                  )}
                </section>

                <section className="panel week-observation">
                  <div className="panel-heading">
                    <div>
                      <span className="section-kicker">7일 행동 흐름</span>
                      <h2>하루를 선택해 상세 기록 보기</h2>
                    </div>
                    <span className="week-range">{activePersona.weekRange}</span>
                  </div>
                  <div className="week-days" role="tablist" aria-label="가상 관찰 날짜">
                    {activePersona.week.map((day, index) => {
                      const intensity = Math.max(
                        day.safetyAlerts * 2 + day.doubleChecks + day.unfinishedTasks + Math.round(day.microDelayRate / 6) - 1,
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
                      <span>{demoDay.day}요일 · {demoDay.date}{demoDay.busy ? " · 바쁨으로 표시됨" : ""}</span>
                      <strong>{demoDay.note}</strong>
                    </div>
                    <div className="day-signal-grid">
                      <article><span>안전 알림</span><strong>{demoDay.safetyAlerts}<small>회</small></strong></article>
                      <article><span>마감 반복 확인</span><strong>{demoDay.doubleChecks}<small>회</small></strong></article>
                      <article><span>업무 미완료</span><strong>{demoDay.unfinishedTasks}<small>건</small></strong></article>
                      <article><span>미세 지연</span><strong>{demoDay.microDelayRate}<small>%</small></strong></article>
                    </div>
                    <ul className="day-examples">
                      {demoDay.examples.map((example, exampleIndex) => (
                        <li key={example.label}>
                          <span>{example.label}</span>
                          <button
                            type="button"
                            className="demo-event-replay-button"
                            onClick={() =>
                              openDemoEventReplay(
                                activePersona,
                                demoDay,
                                selectedDemoDay,
                                exampleIndex,
                                example,
                              )
                            }
                          >
                            동작 보기
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                </section>

                <section className="insight-grid">
                  {activePersona.insights.map((insight, index) => {
                    const toneClass = ["observe", "timeline-insight", "care-insight"][index] ?? "observe";
                    return (
                      <article className={`insight-card ${toneClass}`} key={insight.kicker}>
                        <span className="insight-icon" aria-hidden="true">{insight.icon}</span>
                        <div>
                          <span className="section-kicker">{insight.kicker}</span>
                          <h3>{insight.title}</h3>
                          <p>{insight.body}</p>
                        </div>
                      </article>
                    );
                  })}
                </section>

                <section className="simulation-note">
                  <strong>이 데모가 보여주는 범위</strong>
                  <p>{activePersona.note}</p>
                </section>
              </>
            ) : !consent.observationConsent ? (
              <section className="care-empty-state">
                <span className="care-summary-mark" aria-hidden="true">♡</span>
                <h2>아직 장기 관찰에 참여하고 있지 않아요</h2>
                <p>
                  카메라를 연결할 때 &ldquo;동의하고 카메라 켜기&rdquo;를 선택하면, 실제
                  사용 기록을 바탕으로 한 케어 리포트가 이곳에 쌓이기
                  시작해요. 지금은 시뮬레이션 예시만 보실 수 있어요.
                </p>
                <button type="button" onClick={() => handleConsentDecision(true)}>
                  장기 관찰에 참여하기
                </button>
              </section>
            ) : recentCareLogs.length === 0 ? (
              <section className="care-empty-state">
                <span className="care-summary-mark" aria-hidden="true">♡</span>
                <h2>아직 쌓인 기록이 없어요</h2>
                <p>
                  스마트 마감을 확인하거나 세이브포인트를 사용하시면, 그
                  기록을 바탕으로 케어 리포트가 만들어져요. 며칠 사용하시면
                  평소 흐름과 비교한 리포트를 볼 수 있어요.
                </p>
              </section>
            ) : (
              <>
                <section className="care-summary">
                  <span className="care-summary-mark" aria-hidden="true">♡</span>
                  <div>
                    <span className="section-kicker">
                      최근 {recentCareLogs.length}일 케어 리포트 · 실제 기록
                    </span>
                    <h2>
                      {changeSignal.level === "notable"
                        ? "요즘 몇 가지 흐름이 평소보다 늘었어요"
                        : changeSignal.level === "watch"
                          ? "한두 가지 변화가 눈에 띄어요"
                          : "대체로 평소와 비슷한 흐름이에요"}
                    </h2>
                    {changeSignal.reasons.map((reason) => (
                      <p key={reason}>{reason}</p>
                    ))}
                    {changeSignal.confoundNote && (
                      <p className="confound-note">{changeSignal.confoundNote}</p>
                    )}
                    {!careBaseline && (
                      <p className="confound-note">
                        아직 평소 기준을 만들 만큼(최소 3일) 기록이 쌓이지
                        않아, 변화 비교 없이 최근 기록만 보여드려요.
                      </p>
                    )}
                  </div>
                </section>

                <div className="metric-grid">
                  <article className="metric-card">
                    <span>안전 알림</span>
                    <strong>
                      {totalSafetyAlerts}
                      <small>회</small>
                    </strong>
                    <p>최근 {recentCareLogs.length}일 합계</p>
                  </article>
                  <article className="metric-card">
                    <span>마감 반복 확인</span>
                    <strong>
                      {totalDoubleChecks}
                      <small>회</small>
                    </strong>
                    <p>최근 {recentCareLogs.length}일 합계</p>
                  </article>
                  <article className="metric-card">
                    <span>업무 누락</span>
                    <strong>
                      {totalDroppedTasks}
                      <small>건</small>
                    </strong>
                    <p>최근 {recentCareLogs.length}일 합계</p>
                  </article>
                  <article className="metric-card">
                    <span>오늘 업무 흐름</span>
                    <strong className="word-value">
                      {todaySummary && todaySummary.microDelayRate > 0.3
                        ? "지연 있음"
                        : "평소와 비슷"}
                    </strong>
                    <p>
                      <i className="steady">● 참고용</i> 반복 업무 처리 시간
                      기준
                    </p>
                  </article>
                </div>

                <section className={`panel personal-pattern-panel mode-${observationProfile.mode}`}>
                  <div className="panel-heading">
                    <div>
                      <span className="section-kicker">{occupationTemplate.icon} 개인 업무 패턴 · 기준선 v{observationProfile.baselineVersion}</span>
                      <h2>
                        {observationProfile.mode === "learning"
                          ? "평소 업무 흐름을 학습하고 있어요"
                          : analysisObservationSignals.length > 0
                            ? "평소 범위를 벗어난 동작을 조금 더 살펴봐요"
                            : "최근 동작은 학습한 범위와 비슷해요"}
                      </h2>
                    </div>
                    <span className="pattern-mode-chip">{observationProfile.mode === "learning" ? "학습 모드" : "분석 모드"}</span>
                  </div>
                  <p>
                    {observationProfile.mode === "learning"
                      ? `정상 후보 ${acceptedObservationCount}건을 기준선에 포함했고, 확정하기 어려운 ${quarantinedObservationCount}건은 학습에서 보류했어요.`
                      : `업무별 평균과 표준편차를 따로 비교했습니다. 최근 변화 후보 ${analysisObservationSignals.length}건을 기록했어요.`}
                  </p>
                  <div className="pattern-summary-grid">
                    <span><small>기준선 완성도</small><strong>{observationBaseline.confidence}%</strong></span>
                    <span><small>학습된 업무</small><strong>{observationBaseline.tasks.length}개</strong></span>
                    <span><small>유효 학습일</small><strong>{observationBaseline.eligibleDays}일</strong></span>
                    <span><small>변화 후보</small><strong>{analysisObservationSignals.length}건</strong></span>
                  </div>
                  {analysisObservationSignals[0] && (
                    <p className="pattern-latest-signal">
                      최근 {analysisObservationSignals[0].taskLabel}에서 개인 기준보다 긴 지연이 관찰됐어요. 한 번의 장면으로 판단하지 않고 같은 흐름이 반복되는지 살펴봅니다.
                    </p>
                  )}
                </section>

                {motionSignal && (
                  <section className="panel motion-signal-note">
                    <div className="panel-heading">
                      <div>
                        <span className="section-kicker">참고용 · 검증되지 않음</span>
                        <h2>손 동작 신호 (실험적)</h2>
                      </div>
                    </div>
                    <p>
                      최근 카메라 세션의 손 움직임에서 계산한 참고 지표예요.
                      5FPS로 기록되기 때문에 실제 손 떨림(4~12Hz)을 정밀하게
                      측정할 수 없고, 임상적으로 검증되지 않았어요. 추세를
                      가볍게 참고하는 용도로만 사용해 주세요.
                    </p>
                    <div className="motion-signal-grid">
                      <span>
                        동작 변동성{" "}
                        <strong>
                          {motionSignal.variability !== null
                            ? motionSignal.variability.toFixed(2)
                            : "측정 중"}
                        </strong>
                      </span>
                      <span>
                        동작 매끄러움(값이 작을수록 부드러움){" "}
                        <strong>
                          {motionSignal.smoothness !== null
                            ? motionSignal.smoothness.toFixed(4)
                            : "측정 중"}
                        </strong>
                      </span>
                    </div>
                  </section>
                )}
              </>
            )}

            <section className="panel care-connect-card">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">케어로 연결</span>
                  <h2>도움이 필요할 때 연결할 수 있어요</h2>
                </div>
              </div>
              <p>
                이 리포트만으로 건강 상태를 판단하지 않아요. 다만 변화가
                계속 이어져 걱정되신다면, 아래 상담 채널을 편하게
                이용해보세요.
              </p>
              <div className="care-connect-actions">
                <a className="care-connect-link" href="tel:1899-9988">
                  치매상담콜센터(보건복지부) 1899-9988
                </a>
                <button type="button" onClick={() => void shareCareSummary()}>
                  가족과 리포트 공유
                </button>
              </div>
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
            {selectedEvent.poseSessionId && (
              <button
                className="secondary-button replay-open-button"
                type="button"
                onClick={() => setReplaySessionId(selectedEvent.poseSessionId ?? null)}
              >
                이 순간이 기록된 세션 전체 리플레이 보기
              </button>
            )}
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

      {showConsentModal && (
        <div className="modal-backdrop" role="presentation">
          <section
            className="consent-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="consent-title"
          >
            <span className="section-kicker">카메라를 켜기 전에 알려드려요</span>
            <h2 id="consent-title">이 카메라는 두 가지 목적으로 쓰일 수 있어요</h2>
            <ul className="consent-list">
              <li>
                <strong>① 기억 복원</strong> — 결제·출입 같은 순간의 몸·손
                좌표를 짧게 저장해, 나중에 &ldquo;그때 무슨 일이 있었는지&rdquo;를
                스켈레톤으로 다시 확인할 수 있게 해요.
              </li>
              <li>
                <strong>② 장기 인지 건강 관찰(선택)</strong> — 동의하시면,
                안전 알림 빈도·마감 반복 확인·업무 지연 같은 행동 패턴을
                오랜 기간 관찰해 케어 리포트를 만드는 데도 사용해요. 이
                데이터는 진단이 아니라 변화를 알아차리는 참고용이며, 언제든
                내 데이터 관리에서 철회하고 전부 삭제할 수 있어요.
              </li>
            </ul>
            <p className="consent-note">
              얼굴·영상·음성은 저장하지 않고, 좌표 데이터는 이 브라우저에만
              남아요. 동의하지 않아도 타임라인·스마트 마감 같은 매장 안전
              기능은 카메라 없이 그대로 사용할 수 있어요.
            </p>
            <div className="consent-actions">
              <button
                type="button"
                className="secondary-button"
                onClick={() => handleConsentDecision(false)}
              >
                매장 안전 기능만 사용할게요
              </button>
              <button
                type="button"
                className="primary-button"
                onClick={() => handleConsentDecision(true)}
              >
                동의하고 카메라 켜기
              </button>
            </div>
          </section>
        </div>
      )}

      {myDataOpen && (
        <div
          className="modal-backdrop"
          role="presentation"
          onMouseDown={() => setMyDataOpen(false)}
        >
          <section
            className="my-data-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="my-data-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <button
              className="modal-close"
              type="button"
              onClick={() => setMyDataOpen(false)}
              aria-label="닫기"
            >
              ×
            </button>
            <span className="section-kicker">내 데이터 관리</span>
            <h2 id="my-data-title">저장된 데이터를 확인하고 관리하세요</h2>

            <div className="my-data-grid">
              <article>
                <span>저장된 동작 좌표 세션</span>
                <strong>{sessionCount}개</strong>
              </article>
              <article>
                <span>케어 관찰 기록</span>
                <strong>{careLogs.length}일치</strong>
              </article>
              <article>
                <span>브라우저 저장 용량</span>
                <strong>
                  {storageUsage
                    ? `${formatBytes(storageUsage.usageBytes)} 사용 중`
                    : "확인 불가"}
                </strong>
              </article>
              <article>
                <span>장기 관찰 동의 상태</span>
                <strong>
                  {consent.decided
                    ? consent.observationConsent
                      ? "동의함"
                      : "동의 안 함"
                    : "아직 결정 안 함"}
                </strong>
              </article>
            </div>

            <p className="my-data-note">
              얼굴·영상·음성은 저장되지 않으며, 모든 데이터는 이 브라우저
              안에만 있어요. 다른 기기에서는 보이지 않고, 브라우저 데이터를
              지우면 함께 사라져요.
            </p>

            {recentSessions.length > 0 && (
              <div className="my-data-sessions">
                <span className="section-kicker">개발용 · 최근 좌표 세션 리플레이</span>
                <ul>
                  {recentSessions.map((session) => (
                    <li key={session.id}>
                      <span>
                        {formatSessionTime(session.startedAt)} · {session.frameCount}프레임
                      </span>
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => setReplaySessionId(session.id)}
                        disabled={session.frameCount === 0}
                      >
                        리플레이
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="my-data-actions">
              {consent.observationConsent ? (
                <button type="button" onClick={withdrawObservationConsent}>
                  장기 관찰 참여 철회하기
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => handleConsentDecision(true)}
                >
                  장기 관찰에 참여하기
                </button>
              )}
              <button
                type="button"
                className="danger-button"
                onClick={() => void deleteAllMyData()}
              >
                내 데이터 전체 삭제
              </button>
            </div>
          </section>
        </div>
      )}

      {syntheticLibraryOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setSyntheticLibraryOpen(false)}>
          <section className="synthetic-library-modal" role="dialog" aria-modal="true" aria-labelledby="synthetic-library-title" onMouseDown={(event) => event.stopPropagation()}>
            <button className="modal-close" type="button" onClick={() => setSyntheticLibraryOpen(false)} aria-label="닫기">×</button>
            <header className="synthetic-library-header">
              <div>
                <span className="section-kicker">{occupationTemplate.icon} {occupationTemplate.label} · 14일 정상 업무 표본</span>
                <h2 id="synthetic-library-title">가상 학습 행동을 보고 직접 따라 해보세요</h2>
                <p>실제 사용자의 영상이 아닌 좌표로 만든 예시입니다. 행동을 고르면 이번 웹캠 기록을 해당 업무 기준과 직접 비교해요.</p>
              </div>
              <span className="synthetic-library-count">{syntheticTrainingClips.length}개 행동</span>
            </header>
            <div className="synthetic-phase-tabs" role="tablist" aria-label="업무 시간대">
              {(["open", "business", "close"] as WorkPhase[]).map((phase) => (
                <button
                  key={phase}
                  type="button"
                  role="tab"
                  aria-selected={syntheticLibraryPhase === phase}
                  className={syntheticLibraryPhase === phase ? "active" : ""}
                  onClick={() => setSyntheticLibraryPhase(phase)}
                >
                  {phase === "open" ? "오픈" : phase === "business" ? "영업 중" : "마감"}
                </button>
              ))}
            </div>
            {visibleSyntheticClips.length > 0 ? (
              <div className="synthetic-clip-grid">
                {visibleSyntheticClips.map((clip) => (
                  <article className="synthetic-clip-card" key={clip.id}>
                    <div className="synthetic-clip-heading">
                      <span>{clip.phaseLabel}</span>
                      <small>{clip.zoneLabel}</small>
                    </div>
                    <h3>{clip.taskLabel}</h3>
                    <p>{clip.instruction}</p>
                    <div className="synthetic-motion-tags">
                      {clip.primitiveLabels.slice(0, 3).map((label) => <span key={label}>{label}</span>)}
                    </div>
                    <dl>
                      <div><dt>정상 소요 범위</dt><dd>{clip.expectedMinSeconds}–{clip.expectedMaxSeconds}초</dd></div>
                      <div><dt>대표 구역</dt><dd>{clip.zoneLabel}</dd></div>
                    </dl>
                    <div className="synthetic-clip-actions">
                      <button type="button" onClick={() => openSyntheticClip(clip)}>스켈레톤 보기</button>
                      <button type="button" className="primary" onClick={() => void selectPerformanceTestTask(clip)}>이 동작 테스트하기</button>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="synthetic-library-empty">
                <strong>이 시간대에 등록된 대표 행동이 없어요</strong>
                <p>다른 시간대를 선택해 주세요.</p>
              </div>
            )}
            <footer className="synthetic-library-footer">
              <span>가상 데이터는 실제 케어 리포트와 분리되며 이 기기에만 저장됩니다.</span>
              <button type="button" onClick={() => setSyntheticLibraryOpen(false)}>닫기</button>
            </footer>
          </section>
        </div>
      )}

      {zoneSetupOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setZoneSetupOpen(false)}>
          <section className="zone-setup-modal" role="dialog" aria-modal="true" aria-labelledby="zone-setup-title" onMouseDown={(event) => event.stopPropagation()}>
            <button className="modal-close" type="button" onClick={() => setZoneSetupOpen(false)} aria-label="닫기">×</button>
            <span className="section-kicker">{occupationTemplate.icon} {occupationTemplate.label} 관찰 맥락</span>
            <h2 id="zone-setup-title">카메라 화면에 매장 구역을 표시해 주세요</h2>
            <p>구역을 고른 뒤 실제 카메라 화면에서 해당 위치에 가까운 칸을 눌러주세요. 카메라 위치가 바뀌면 다시 설정해야 해요.</p>
            <label className="zone-picker">
              <span>지정할 구역</span>
              <select value={selectedZoneId} onChange={(event) => setSelectedZoneId(event.target.value)}>
                {occupationTemplate.zones.map((zone) => <option key={zone.id} value={zone.id}>{zone.label}</option>)}
              </select>
            </label>
            <div className="zone-camera-grid" aria-label="카메라 화면 3×3 구역 설정">
              {observationProfile.zoneGrid.map((zoneId, index) => {
                const zone = occupationTemplate.zones.find((item) => item.id === zoneId);
                return (
                  <button key={index} type="button" className={zoneId ? "mapped" : ""} onClick={() => void assignZoneCell(index)}>
                    <small>{index + 1}</small>
                    <strong>{zone?.label ?? "구역 지정"}</strong>
                  </button>
                );
              })}
            </div>
            <div className="zone-setup-footer">
              <span>{mappedZoneCount}개 구역 설정됨 · 정밀 거리 대신 화면상 위치를 사용해요.</span>
              <button type="button" onClick={() => setZoneSetupOpen(false)}>설정 완료</button>
            </div>
          </section>
        </div>
      )}

      {replaySessionId && (
        <SessionReplayPanel
          key={replaySessionId}
          source={{ kind: "recorded", sessionId: replaySessionId }}
          sessionLabel={
            recentSessions.find((session) => session.id === replaySessionId)
              ? `${formatSessionTime(
                  recentSessions.find((session) => session.id === replaySessionId)!.startedAt,
                )} 세션`
              : undefined
          }
          feedbackEventId={
            observationEpisodes.find((episode) => episode.sessionId === replaySessionId)?.id
          }
          observationMode={observationProfile.mode}
          baselineVersion={observationProfile.baselineVersion}
          onClose={() => setReplaySessionId(null)}
        />
      )}

      {demoReplay && (
        <SessionReplayPanel
          key={demoReplay.key}
          source={{ kind: "synthetic", frames: demoReplay.frames }}
          sessionLabel={demoReplay.label}
          detectionExplanation={demoReplay.detectionExplanation}
          feedbackEventId={demoReplay.key}
          observationMode={observationProfile.mode}
          baselineVersion={observationProfile.baselineVersion}
          onClose={() => setDemoReplay(null)}
        />
      )}

      {toast && (
        <div className="toast" role="status">
          <span aria-hidden="true">✓</span> {toast}
        </div>
      )}
    </main>
  );
}
