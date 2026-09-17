"use client";

import { FormEvent, ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Toaster, toast } from "sonner";
import NumberFlow from "@number-flow/react";
import { Dialog } from "@base-ui-components/react/dialog";
import type {
  HandLandmarker,
  NormalizedLandmark,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import {
  HAND_LANDMARK_COUNT,
  MOTION_FRAME_STRIDE,
  MOTION_SAMPLE_RATE,
  appendCameraHealthChunk,
  appendMotionChunk,
  createMotionSession,
  deleteAllMotionSessions,
  downloadMotionDataFolder,
  extractHandPointTrajectory,
  finishCaptureManifest,
  finishMotionSession,
  getGlobalSessionCameraFrames,
  getGlobalSessionFrames,
  getSessionFrames,
  getLatestBodyProportionProfile,
  listMotionSessions,
  saveCaptureManifest,
  encodeCameraHealth,
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
  shouldShowCognitiveSupport,
  summarizeLog,
  type DailyLog,
} from "./care-metrics";
import {
  deleteAllCareLogs,
  clearConsent,
  getConsent,
  listRecentLogs,
  recordDoubleCheck,
  recordMicroDelay,
  recordTaskCompleted,
  recordTaskStarted,
  setConsent,
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
  getOccupationTemplate,
  inferOccupationContext,
  inferZoneContext,
  phaseForHour,
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
  saveObservationProfile,
} from "./observation-store";
import {
  classifyLearnedMotion,
  clearLearnedMotionActions,
  loadLearnedMotionActions,
  saveLearnedMotionActions,
  type LearnedMotionAction,
  type LearnedMotionResult,
  type LearnedMotionSample,
} from "./custom-motion-training";
import { sliceTargetMotion } from "./motion-segmentation";
import {
  createGlobalCaptureSession,
  isExternalCamera,
  selectExternalCameraSlots,
  type CameraDeviceIdentity,
  type CameraSlot,
  type GlobalCaptureSession,
} from "./multi-camera";
import {
  ACTION_AMBIGUITY_REASON_LABELS,
  applyManualActionLabel,
  createActionReview,
  listPendingActionReviews,
  markActionReviewUnresolved,
  normalizeActionCandidates,
  type ReviewableObservationEpisode,
} from "./action-review";
import {
  buildWorkContextCandidates,
  listWorkContextLabels,
} from "./work-context-inference";
import {
  matchHandToPose,
  selectLockedPose,
  type PoseTargetLock,
} from "./pose-target-lock";

type View = "home" | "settings" | "today" | "timeline" | "care" | "onboarding";
type OnboardingStep = 1 | 2 | 3 | 4;
type SettingsSection = "occupation" | "observation" | "context" | "checklist" | null;
type InterfaceMode = "user" | "developer";
type CameraStatus = "idle" | "requesting" | "connected" | "error";
type PoseStatus = "idle" | "loading" | "searching" | "holding" | "partial" | "full" | "error";
type EventKind = "payment" | "booking";
type QuickMotionMode = "idle" | "training" | "labeling" | "testing";

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

function ShimmerText({ children }: { children: string }) {
  return (
    <span className="t-shimmer" data-text={children}>
      {children}
    </span>
  );
}

function cameraDisplayName(camera: MediaDeviceInfo, index: number) {
  return camera.label ? `${camera.label} · ${camera.deviceId.slice(0, 4)}` : `카메라 ${index + 1}`;
}

async function receivesVideoFrames(stream: MediaStream, timeoutMs = 4_000) {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;

  try {
    return await new Promise<boolean>((resolve) => {
      let settled = false;
      const finish = (received: boolean) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeout);
        resolve(received);
      };
      const timeout = window.setTimeout(() => finish(false), timeoutMs);

      if ("requestVideoFrameCallback" in video) {
        video.requestVideoFrameCallback(() => finish(true));
      } else {
        const onFrame = () => finish(video.videoWidth > 0 && video.videoHeight > 0);
        video.addEventListener("loadeddata", onFrame, { once: true });
        video.addEventListener("timeupdate", onFrame, { once: true });
      }
      void video.play().catch(() => finish(false));
    });
  } finally {
    video.pause();
    video.srcObject = null;
  }
}

type ClosingChecklistItem = {
  id: string;
  label: string;
  done: boolean;
};

type WeekdayIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6;

type WorkRoutineScheduleItem = {
  id: string;
  time: string;
  label: string;
  repeats: boolean;
};

type WorkRoutine = {
  id: string;
  name: string;
  days: WeekdayIndex[];
  openTime: string;
  closeTime: string;
  schedule: WorkRoutineScheduleItem[];
};

type WorkContextConfig = {
  routines: WorkRoutine[];
  closedDays: WeekdayIndex[];
};

type UserInstallState = {
  startedAt: number;
  startDate: string;
  careOffset: DailyLog | null;
};

const DEFAULT_CLOSING_CHECKLIST: ClosingChecklistItem[] = [
  { id: "pos", label: "포스기 닫기", done: false },
  { id: "revenue", label: "오늘 매출 정산하기", done: false },
  { id: "door", label: "출입문 잠금 확인하기", done: false },
];

const CHECKLIST_STORAGE_KEY = "memory-guard-closing-checklist-v1";
const WORK_CONTEXT_CONFIG_STORAGE_KEY = "memory-guard-work-context-config-v2";
const WEEKDAY_LABELS: Record<WeekdayIndex, string> = { 0: "일", 1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토" };
const WEEKDAY_DISPLAY_ORDER: WeekdayIndex[] = [1, 2, 3, 4, 5, 6, 0];
const DEFAULT_WORK_ROUTINE: WorkRoutine = {
  id: "default",
  name: "기본 루틴",
  days: [0, 1, 2, 3, 4, 5, 6],
  openTime: "08:00",
  closeTime: "21:00",
  schedule: [],
};
const DEFAULT_WORK_CONTEXT_CONFIG: WorkContextConfig = {
  routines: [DEFAULT_WORK_ROUTINE],
  closedDays: [],
};
const INTERFACE_MODE_STORAGE_KEY = "memory-guard-interface-mode-v1";
const USER_INSTALL_STORAGE_KEY = "memory-guard-user-install-v1";
const CAMERA_SLOT_STORAGE_KEY = "memory-guard-external-camera-slots-v1";
const POSE_DISPLAY_HOLD_MS = 450;
const CAMERA_HEALTH_CHUNK_SECONDS = 300;

function loadInterfaceMode(): InterfaceMode {
  if (typeof window === "undefined") return "user";
  return window.localStorage.getItem(INTERFACE_MODE_STORAGE_KEY) === "developer"
    ? "developer"
    : "user";
}

function loadUserInstallState(): UserInstallState | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = window.localStorage.getItem(USER_INSTALL_STORAGE_KEY);
    return stored ? JSON.parse(stored) as UserInstallState : null;
  } catch {
    return null;
  }
}

function loadCameraAssignments(): CameraDeviceIdentity[] {
  try {
    const stored = window.localStorage.getItem(CAMERA_SLOT_STORAGE_KEY);
    return stored ? JSON.parse(stored) as CameraDeviceIdentity[] : [];
  } catch {
    return [];
  }
}

function scopeCareLogsToUserInstall(
  logs: DailyLog[],
  install: UserInstallState | null,
): DailyLog[] {
  if (!install) return [];
  return logs
    .filter((log) => log.date >= install.startDate)
    .map((log) => {
      if (!install.careOffset || log.date !== install.startDate) return log;
      return {
        ...log,
        safetyAlerts: Math.max(0, log.safetyAlerts - install.careOffset.safetyAlerts),
        doubleChecks: Math.max(0, log.doubleChecks - install.careOffset.doubleChecks),
        tasksStarted: Math.max(0, log.tasksStarted - install.careOffset.tasksStarted),
        tasksCompleted: Math.max(0, log.tasksCompleted - install.careOffset.tasksCompleted),
        microDelaySeconds: log.microDelaySeconds.slice(install.careOffset.microDelaySeconds.length),
      };
    });
}

function loadChecklistSettings() {
  const fallback = DEFAULT_CLOSING_CHECKLIST;
  if (typeof window === "undefined") return fallback;
  try {
    const stored = window.localStorage.getItem(CHECKLIST_STORAGE_KEY);
    if (!stored) return fallback;
    const parsed = JSON.parse(stored) as {
      date?: string;
      time?: string;
      items?: ClosingChecklistItem[];
    };
    const items = Array.isArray(parsed.items) ? parsed.items : DEFAULT_CLOSING_CHECKLIST;
    return parsed.date === currentDateKey()
      ? items
      : items.map((item) => ({ ...item, done: false }));
  } catch {
    return fallback;
  }
}

function isWeekdayIndex(value: unknown): value is WeekdayIndex {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 6;
}

function loadWorkContextConfig(): WorkContextConfig {
  if (typeof window === "undefined") return DEFAULT_WORK_CONTEXT_CONFIG;
  try {
    const stored = window.localStorage.getItem(WORK_CONTEXT_CONFIG_STORAGE_KEY);
    if (!stored) return DEFAULT_WORK_CONTEXT_CONFIG;
    const parsed = JSON.parse(stored) as Partial<WorkContextConfig>;
    const routines = Array.isArray(parsed.routines) && parsed.routines.length > 0
      ? parsed.routines.map((routine) => ({
        id: routine.id ?? crypto.randomUUID(),
        name: routine.name ?? "루틴",
        days: Array.isArray(routine.days) ? routine.days.filter(isWeekdayIndex) : [],
        openTime: routine.openTime ?? DEFAULT_WORK_ROUTINE.openTime,
        closeTime: routine.closeTime ?? DEFAULT_WORK_ROUTINE.closeTime,
        schedule: Array.isArray(routine.schedule)
          ? routine.schedule.map((item) => ({ ...item, repeats: item.repeats ?? false }))
          : [],
      }))
      : [DEFAULT_WORK_ROUTINE];
    const closedDays = Array.isArray(parsed.closedDays) ? parsed.closedDays.filter(isWeekdayIndex) : [];
    return { routines, closedDays };
  } catch {
    return DEFAULT_WORK_CONTEXT_CONFIG;
  }
}

function currentDateKey() {
  return new Intl.DateTimeFormat("en-CA", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}

function formatClockLabel(value: string) {
  const [hours, minutes] = value.split(":").map(Number);
  const period = hours < 12 ? "오전" : "오후";
  const displayHour = hours % 12 === 0 ? 12 : hours % 12;
  return minutes === 0 ? `${period} ${displayHour}시` : `${period} ${displayHour}시 ${minutes}분`;
}

function routineNameForDay(config: WorkContextConfig, day: WeekdayIndex): string {
  if (config.closedDays.includes(day)) return "휴무";
  const owner = config.routines.find((routine) => routine.days.includes(day));
  return owner ? owner.name : "미지정";
}

const CHUNK_FRAME_COUNT = MOTION_SAMPLE_RATE * 30;

const initialEvents: TimelineEvent[] = [
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
];

const userNavItems: Array<{ id: View; label: string }> = [
  { id: "home", label: "홈" },
  { id: "timeline", label: "기록" },
  { id: "care", label: "변화" },
  { id: "settings", label: "설정" },
];

function NavIcon({ view }: { view: View }) {
  const paths: Partial<Record<View, ReactNode>> = {
    home: <><path d="M3.5 10.8 12 3.6l8.5 7.2" /><path d="M5.8 9.3v10.2h12.4V9.3M9.4 19.5v-6h5.2v6" /></>,
    today: <><rect x="3" y="6" width="18" height="13" rx="3" /><path d="m8 6 1.2-2h5.6L16 6" /><circle cx="12" cy="12.5" r="3.2" /></>,
    timeline: <><path d="M6 3.8h12a2 2 0 0 1 2 2v14.4H6a2 2 0 0 1-2-2V5.8a2 2 0 0 1 2-2Z" /><path d="M8 8h8M8 12h8M8 16h5" /></>,
    care: <><path d="M4 12h3l2-5 4 10 2-5h5" /><path d="M12 21C6.8 18.2 3 14.8 3 10.2 3 6.8 5.4 4.5 8.4 4.5c1.7 0 2.9.8 3.6 2 .7-1.2 1.9-2 3.6-2 3 0 5.4 2.3 5.4 5.7" /></>,
    settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.6v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z" /></>,
  };

  return (
    <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {paths[view] ?? <circle cx="12" cy="12" r="8" />}
    </svg>
  );
}

function CareMark() {
  return (
    <span className="care-summary-mark" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20.8 9.4c0 5.2-4.3 8.3-8.8 11-4.5-2.7-8.8-5.8-8.8-11A4.9 4.9 0 0 1 8 4.5c1.8 0 3.2.9 4 2.2.8-1.3 2.2-2.2 4-2.2a4.9 4.9 0 0 1 4.8 4.9Z" />
      </svg>
    </span>
  );
}

function StoreIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 10v10h16V10M3 10l2-6h14l2 6" /><path d="M8 20v-5h4v5M3 10c0 1.5 1 2.5 2.5 2.5S8 11.5 8 10c0 1.5 1 2.5 2.5 2.5S13 11.5 13 10c0 1.5 1 2.5 2.5 2.5S18 11.5 18 10c0 1.5 1 2.5 2.5 2.5" />
    </svg>
  );
}

const developerNavItems: Array<{ id: View; label: string }> = [
  { id: "today", label: "카메라 테스트" },
  { id: "care", label: "가상 리포트" },
  { id: "settings", label: "테스트 설정" },
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
  ];

const kindLabel: Record<EventKind, string> = {
  payment: "결제",
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

function Modal({
  open,
  onClose,
  labelledBy,
  className,
  disableDismiss = false,
  children,
}: {
  open: boolean;
  onClose: () => void;
  labelledBy: string;
  className: string;
  disableDismiss?: boolean;
  children: ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={(next) => !next && !disableDismiss && onClose()}>
      <Dialog.Portal>
        <Dialog.Backdrop className="modal-backdrop" />
        <Dialog.Viewport className="modal-viewport">
          <Dialog.Popup className={className} aria-labelledby={labelledBy}>
            {children}
          </Dialog.Popup>
        </Dialog.Viewport>
      </Dialog.Portal>
    </Dialog.Root>
  );
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
          <span className={`event-node ${event.kind}`} aria-hidden="true" />
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
  const [view, setView] = useState<View>("home");
  const [interfaceMode, setInterfaceMode] = useState<InterfaceMode>("user");
  const [userInstall, setUserInstall] = useState<UserInstallState | null>(null);
  const [clientPreferencesLoaded, setClientPreferencesLoaded] = useState(false);
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("idle");
  const [cameraMessage, setCameraMessage] = useState(
    "카메라를 연결하면 오늘의 장면을 확인할 수 있어요.",
  );
  const [poseStatus, setPoseStatus] = useState<PoseStatus>("idle");
  const [secondaryPoseStatus, setSecondaryPoseStatus] = useState<PoseStatus>("idle");
  const [tertiaryPoseStatus, setTertiaryPoseStatus] = useState<PoseStatus>("idle");
  const [poseStats, setPoseStats] = useState<PoseStats>({
    frames: 0,
    detectedFrames: 0,
    fullBodyFrames: 0,
    handDetectedFrames: 0,
    storageBytes: 0,
    startedAt: null,
  });
  const [detectedHands, setDetectedHands] = useState(0);
  const [detectedHandSides, setDetectedHandSides] = useState({
    left: false,
    right: false,
  });
  const [headDirectionLabel, setHeadDirectionLabel] = useState("대기");
  const [targetLocked, setTargetLocked] = useState(false);
  const [cameraFullscreen, setCameraFullscreen] = useState(false);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [primaryCameraId, setPrimaryCameraId] = useState("");
  const [secondaryCameraId, setSecondaryCameraId] = useState("");
  const [tertiaryCameraId, setTertiaryCameraId] = useState("");
  const [secondaryCameraConnected, setSecondaryCameraConnected] = useState(false);
  const [tertiaryCameraConnected, setTertiaryCameraConnected] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [sessionCount, setSessionCount] = useState(0);
  const [latestSession, setLatestSession] = useState<MotionSessionRecord | null>(null);
  const [recentSessions, setRecentSessions] = useState<MotionSessionRecord[]>([]);
  const [replaySessionId, setReplaySessionId] = useState<string | null>(null);
  const [selectedReviewEpisodeId, setSelectedReviewEpisodeId] = useState<string | null>(null);
  const [selectedReviewTaskType, setSelectedReviewTaskType] = useState("");
  const [demoReplay, setDemoReplay] = useState<{
    key: string;
    label: string;
    frames: number[][];
    detectionExplanation?: ReturnType<typeof explainDemoEvent>;
  } | null>(null);
  const [events, setEvents] = useState<TimelineEvent[]>(initialEvents);
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const [savepointOpen, setSavepointOpen] = useState(false);
  const [bookingOpen, setBookingOpen] = useState(false);
  const [bookingName, setBookingName] = useState("김하나");
  const [bookingService, setBookingService] = useState("커트");
  const [demoMode, setDemoMode] = useState(true);
  const [selectedPersonaIndex, setSelectedPersonaIndex] = useState(0);
  const [selectedDemoDay, setSelectedDemoDay] = useState(6);
  const [observationProfile, setObservationProfile] = useState<ObservationProfile>(DEFAULT_PROFILE);
  const [occupationInput, setOccupationInput] = useState("");
  const [onboardingStep, setOnboardingStep] = useState<OnboardingStep>(1);
  const [settingsSection, setSettingsSection] = useState<SettingsSection>(null);
  const [observationEpisodes, setObservationEpisodes] = useState<ObservationEpisode[]>([]);
  const [observationBaseline, setObservationBaseline] = useState<BaselineSnapshot>(() => buildBaseline([], 1));
  const [zoneSetupOpen, setZoneSetupOpen] = useState(false);
  const [selectedZoneId, setSelectedZoneId] = useState<string>("DRINK_PREP");
  const [zoneCameraSlot, setZoneCameraSlot] = useState<CameraSlot>(1);
  const [customZoneName, setCustomZoneName] = useState("");
  const [closingChecklist, setClosingChecklist] = useState<ClosingChecklistItem[]>(
    loadChecklistSettings,
  );
  const [newChecklistItem, setNewChecklistItem] = useState("");
  const [workContextConfig, setWorkContextConfig] = useState<WorkContextConfig>(loadWorkContextConfig);
  const [brainHealthOpen, setBrainHealthOpen] = useState(false);
  const [learnedMotions, setLearnedMotions] = useState<LearnedMotionAction[]>([]);
  const [quickMotionMode, setQuickMotionMode] = useState<QuickMotionMode>("idle");
  const [trainingTargetCount, setTrainingTargetCount] = useState<5 | 10>(5);
  const [draftMotionSamples, setDraftMotionSamples] = useState<LearnedMotionSample[]>([]);
  const [customMotionLabel, setCustomMotionLabel] = useState("");
  const [motionCapture, setMotionCapture] = useState<Omit<LearnedMotionSample, "endMs"> | null>(null);
  const [learnedMotionResult, setLearnedMotionResult] = useState<LearnedMotionResult | null>(null);
  const [quickMotionBusy, setQuickMotionBusy] = useState(false);
  const [quickMotionCameraOpen, setQuickMotionCameraOpen] = useState(false);
  const [multiCameraModalOpen, setMultiCameraModalOpen] = useState(false);
  const mainNavRef = useRef<HTMLElement>(null);
  const mainNavPillRef = useRef<HTMLSpanElement>(null);
  const navPillReadyRef = useRef(false);
  const previousNavModeRef = useRef<InterfaceMode>(interfaceMode);
  const pageMotionReadyRef = useRef(false);
  const [animatedPageKey, setAnimatedPageKey] = useState<string | null>(null);

  // --- Consent, real observation metrics, and data controls ---
  // Lazy initializer instead of an effect: getConsent() is SSR-safe (it
  // checks `typeof window` itself and falls back to the default), so there's
  // no need to synchronize it via setState in an effect after mount.
  const [consent, setConsentState] = useState<ConsentState>(() => getConsent());
  const [showConsentModal, setShowConsentModal] = useState(false);
  const [careLogs, setCareLogs] = useState<DailyLog[]>([]);
  const [myDataOpen, setMyDataOpen] = useState(false);
  const [storedMotionBytes, setStoredMotionBytes] = useState(0);
  const [motionSignal, setMotionSignal] = useState<{
    variability: number | null;
    smoothness: number | null;
  } | null>(null);
  const pendingCameraStartRef = useRef(false);
  const quickMotionCameraRequestedRef = useRef(false);
  const checklistTaskStartedRef = useRef(false);
  const bookingShownAtRef = useRef<number | null>(null);
  const lastTestEventAtRef = useRef<number | null>(null);
  const savepointStartRecordedRef = useRef(false);

  async function refreshCareData() {
    try {
      const logs = await listRecentLogs(28);
      setCareLogs(logs);
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
  const quickMotionVideoRef = useRef<HTMLVideoElement>(null);
  const secondaryVideoRef = useRef<HTMLVideoElement>(null);
  const secondaryOverlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const tertiaryVideoRef = useRef<HTMLVideoElement>(null);
  const tertiaryOverlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const processingVideoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const cameraFrameRef = useRef<HTMLDivElement>(null);
  const cameraFeedLayoutRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const secondaryStreamRef = useRef<MediaStream | null>(null);
  const tertiaryStreamRef = useRef<MediaStream | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const secondaryPoseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const tertiaryPoseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const poseAnimationRef = useRef<number | null>(null);
  const trackingActiveRef = useRef(false);
  const lastDetectionTimeRef = useRef(0);
  const lastHandDetectionTimeRef = useRef(0);
  const lastSampleTimeRef = useRef(0);
  const lastPoseRef = useRef<NormalizedLandmark[] | null>(null);
  const lastPoseSeenAtRef = useRef(0);
  const poseTargetLockRef = useRef<PoseTargetLock | null>(null);
  const secondaryPoseTargetLockRef = useRef<PoseTargetLock | null>(null);
  const tertiaryPoseTargetLockRef = useRef<PoseTargetLock | null>(null);
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
  const secondaryPoseSessionRef = useRef<MotionSessionRecord | null>(null);
  const tertiaryPoseSessionRef = useRef<MotionSessionRecord | null>(null);
  const globalCaptureSessionRef = useRef<GlobalCaptureSession | null>(null);
  const cameraHealthBuffersRef = useRef<Record<CameraSlot, { startSecond: number; samples: number[] }>>({
    1: { startSecond: -1, samples: [] },
    2: { startSecond: -1, samples: [] },
    3: { startSecond: -1, samples: [] },
  });
  const lastCameraHealthSecondRef = useRef(-1);
  const lastPersonSeenByCameraRef = useRef<Record<CameraSlot, number>>({ 1: 0, 2: 0, 3: 0 });
  const cameraTrackingErrorRef = useRef<Record<CameraSlot, boolean>>({ 1: false, 2: false, 3: false });
  const cameraHealthWriteQueueRef = useRef<Promise<void>>(Promise.resolve());
  const poseWriteQueueRef = useRef<Promise<void>>(Promise.resolve());
  const secondaryPoseWriteQueueRef = useRef<Promise<void>>(Promise.resolve());
  const tertiaryPoseWriteQueueRef = useRef<Promise<void>>(Promise.resolve());
  const sessionPerformanceStartRef = useRef(0);
  const lastStatsUpdateRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const lastSecondaryVideoTimeRef = useRef(-1);
  const lastSecondaryDetectionTimeRef = useRef(0);
  const lastSecondaryHandDetectionTimeRef = useRef(0);
  const lastSecondarySampleTimeRef = useRef(0);
  const lastTertiaryVideoTimeRef = useRef(-1);
  const lastTertiaryDetectionTimeRef = useRef(0);
  const lastTertiaryHandDetectionTimeRef = useRef(0);
  const lastTertiarySampleTimeRef = useRef(0);
  const secondaryHandsRef = useRef<HandState>({ left: null, right: null, leftScore: 0, rightScore: 0 });
  const tertiaryHandsRef = useRef<HandState>({ left: null, right: null, leftScore: 0, rightScore: 0 });
  const secondaryPoseBufferRef = useRef<number[]>([]);
  const secondaryChunkDetectedFramesRef = useRef(0);
  const secondaryChunkFullBodyFramesRef = useRef(0);
  const secondaryChunkHandFramesRef = useRef(0);
  const tertiaryPoseBufferRef = useRef<number[]>([]);
  const tertiaryChunkDetectedFramesRef = useRef(0);
  const tertiaryChunkFullBodyFramesRef = useRef(0);
  const tertiaryChunkHandFramesRef = useRef(0);
  const poseStatusRef = useRef<PoseStatus>("idle");
  const poseStatsRef = useRef<PoseStats>({
    frames: 0,
    detectedFrames: 0,
    fullBodyFrames: 0,
    handDetectedFrames: 0,
    storageBytes: 0,
    startedAt: null,
  });

  useEffect(() => {
    const storedMode = loadInterfaceMode();
    const storedInstall = loadUserInstallState();
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      setInterfaceMode(storedMode);
      setUserInstall(storedInstall);
      setView(storedMode === "developer" ? "today" : "home");
      setClientPreferencesLoaded(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!clientPreferencesLoaded) return;
    window.localStorage.setItem(INTERFACE_MODE_STORAGE_KEY, interfaceMode);
  }, [clientPreferencesLoaded, interfaceMode]);

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
      if (quickMotionVideoRef.current) {
        quickMotionVideoRef.current.srcObject = streamRef.current;
        quickMotionVideoRef.current.play().catch(() => undefined);
      }
      if (secondaryVideoRef.current) {
        secondaryVideoRef.current.srcObject = secondaryStreamRef.current;
        secondaryVideoRef.current.play().catch(() => undefined);
      }
      if (tertiaryVideoRef.current) {
        tertiaryVideoRef.current.srcObject = tertiaryStreamRef.current;
        tertiaryVideoRef.current.play().catch(() => undefined);
      }
    }
  }, [cameraStatus, view, quickMotionCameraOpen, multiCameraModalOpen]);

  useEffect(() => {
    queueMicrotask(() => setLearnedMotions(loadLearnedMotionActions()));
  }, []);

  useEffect(() => {
    listMotionSessions()
      .then((sessions) => {
        const captureSessions = sessions.filter((session) => !session.cameraSlot || session.cameraSlot === 1);
        setSessionCount(captureSessions.length);
        setLatestSession(captureSessions[0] ?? null);
        setRecentSessions(captureSessions.slice(0, 5));
        setStoredMotionBytes(sessions.reduce((sum, session) => sum + session.storageBytes, 0));
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    getObservationProfile()
      .then(async (profile) => {
        const saved = await saveObservationProfile(profile);
        setObservationProfile(saved);
        setOccupationInput(saved.occupationName || getOccupationTemplate(saved.occupation).label);
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
        const latestBodyProfile = await getLatestBodyProportionProfile().catch(() => null);
        if (latestBodyProfile && latestBodyProfile.sourceSessionId !== saved.bodyProportionProfile?.sourceSessionId) {
          const profiled = await saveObservationProfile({
            ...saved,
            bodyProportionProfile: latestBodyProfile,
          });
          setObservationProfile(profiled);
        }
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    listRecentLogs(28)
      .then((logs) => {
        setCareLogs(logs);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(
      CHECKLIST_STORAGE_KEY,
      JSON.stringify({
        date: currentDateKey(),
        items: closingChecklist,
      }),
    );
  }, [closingChecklist]);

  useEffect(() => {
    window.localStorage.setItem(WORK_CONTEXT_CONFIG_STORAGE_KEY, JSON.stringify(workContextConfig));
  }, [workContextConfig]);

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
    (latestSession.globalSessionId
      ? getGlobalSessionFrames(latestSession.globalSessionId)
      : getSessionFrames(latestSession.id))
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
    const syncFullscreenState = () => {
      setCameraFullscreen(document.fullscreenElement === cameraFeedLayoutRef.current);
    };
    document.addEventListener("fullscreenchange", syncFullscreenState);
    return () => document.removeEventListener("fullscreenchange", syncFullscreenState);
  }, []);

  useEffect(() => {
    return () => {
      trackingActiveRef.current = false;
      if (poseAnimationRef.current !== null) {
        cancelAnimationFrame(poseAnimationRef.current);
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      secondaryStreamRef.current?.getTracks().forEach((track) => track.stop());
      tertiaryStreamRef.current?.getTracks().forEach((track) => track.stop());
      poseLandmarkerRef.current?.close();
      secondaryPoseLandmarkerRef.current?.close();
      tertiaryPoseLandmarkerRef.current?.close();
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
    quickMotionCameraRequestedRef.current = false;
    if (!consent.decided) {
      pendingCameraStartRef.current = true;
      setShowConsentModal(true);
      return;
    }
    setMultiCameraModalOpen(true);
    void startCamera();
  }

  async function toggleCameraFullscreen() {
    const cameraFeedLayout = cameraFeedLayoutRef.current;
    if (!cameraFeedLayout) return;

    try {
      if (document.fullscreenElement === cameraFeedLayout) {
        await document.exitFullscreen();
      } else {
        await cameraFeedLayout.requestFullscreen();
      }
    } catch {
      toast.success("이 브라우저에서는 카메라 전체화면을 열 수 없어요");
    }
  }

  function handleConsentDecision(observationConsent: boolean) {
    const next = setConsent(observationConsent);
    setConsentState(next);
    setShowConsentModal(false);
    if (observationConsent && pendingCameraStartRef.current) {
      if (!quickMotionCameraRequestedRef.current) setMultiCameraModalOpen(true);
      void startCamera(undefined, undefined, undefined, quickMotionCameraRequestedRef.current);
    } else if (pendingCameraStartRef.current && !quickMotionCameraRequestedRef.current) {
      setMultiCameraModalOpen(false);
    }
    pendingCameraStartRef.current = false;
  }

  async function startCamera(
    requestedPrimaryCameraId?: string,
    requestedSecondaryCameraId?: string,
    requestedTertiaryCameraId?: string,
    primaryOnly = false,
  ) {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraStatus("error");
      setCameraMessage("이 브라우저에서는 카메라를 사용할 수 없어요.");
      return;
    }

    setCameraStatus("requesting");
    setCameraMessage("카메라 연결을 기다리고 있어요…");

    try {
      const savedAssignments = loadCameraAssignments();
      let cameras = (await navigator.mediaDevices.enumerateDevices())
        .filter((device) => device.kind === "videoinput");
      if (cameras.some((camera) => !camera.label)) {
        const savedDevice = cameras.find((camera) =>
          savedAssignments.some((saved) => saved.deviceId === camera.deviceId),
        );
        // Reuse a known external device for permission discovery. Opening
        // plain `video: true` here would make the browser select the laptop camera.
        const permissionStream = await navigator.mediaDevices.getUserMedia({
          video: savedDevice ? { deviceId: { exact: savedDevice.deviceId } } : true,
          audio: false,
        });
        permissionStream.getTracks().forEach((track) => track.stop());
        cameras = (await navigator.mediaDevices.enumerateDevices())
          .filter((device) => device.kind === "videoinput");
      }
      if (cameras.length === 0) throw new Error("No video input devices found");

      const externalCameras = cameras.filter(isExternalCamera);
      const selected = selectExternalCameraSlots(
        externalCameras,
        savedAssignments,
        [
          requestedPrimaryCameraId ?? primaryCameraId,
          requestedSecondaryCameraId ?? secondaryCameraId,
          requestedTertiaryCameraId ?? tertiaryCameraId,
        ],
      );
      const [preferredPrimary, preferredSecondary, preferredTertiary] = selected;
      if (!preferredPrimary) throw new Error("No external video input devices found");

      setAvailableCameras(externalCameras);
      setPrimaryCameraId(preferredPrimary.deviceId);
      setSecondaryCameraId(preferredSecondary?.deviceId ?? "");
      setTertiaryCameraId(preferredTertiary?.deviceId ?? "");
      window.localStorage.setItem(CAMERA_SLOT_STORAGE_KEY, JSON.stringify(selected));

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: preferredPrimary.deviceId },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      let secondaryStream: MediaStream | null = null;
      let tertiaryStream: MediaStream | null = null;
      if (!primaryOnly && preferredSecondary) {
        try {
          secondaryStream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: preferredSecondary.deviceId },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: false,
          });
        } catch {
          // Keep the primary camera usable if a second camera is busy or unavailable.
          secondaryStream = null;
        }
      }
      if (!primaryOnly && preferredTertiary) {
        try {
          tertiaryStream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: preferredTertiary.deviceId },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: false,
          });
        } catch {
          tertiaryStream = null;
        }
      }

      // The first two streams are the proven working path. Safari can starve
      // extra detached video elements, so only probe the newly added third stream.
      const tertiaryHasFrames = tertiaryStream
        ? await receivesVideoFrames(tertiaryStream)
        : false;
      const camerasWithoutFrames: string[] = [];
      if (tertiaryStream && !tertiaryHasFrames) {
        camerasWithoutFrames.push(cameraDisplayName(preferredTertiary!, 2));
        tertiaryStream.getTracks().forEach((track) => track.stop());
        tertiaryStream = null;
      }

      if (primaryOnly && !quickMotionCameraRequestedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        secondaryStream?.getTracks().forEach((track) => track.stop());
        tertiaryStream?.getTracks().forEach((track) => track.stop());
        setCameraStatus("idle");
        return;
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());
      secondaryStreamRef.current?.getTracks().forEach((track) => track.stop());
      tertiaryStreamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = stream;
      secondaryStreamRef.current = secondaryStream;
      tertiaryStreamRef.current = tertiaryStream;
      setSecondaryCameraConnected(Boolean(secondaryStream));
      setTertiaryCameraConnected(Boolean(tertiaryStream));
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      if (processingVideoRef.current) {
        processingVideoRef.current.srcObject = stream;
        await processingVideoRef.current.play();
      }
      if (quickMotionVideoRef.current) {
        quickMotionVideoRef.current.srcObject = stream;
        await quickMotionVideoRef.current.play();
      }
      if (secondaryVideoRef.current && secondaryStream) {
        secondaryVideoRef.current.srcObject = secondaryStream;
        await secondaryVideoRef.current.play();
      }
      if (tertiaryVideoRef.current && tertiaryStream) {
        tertiaryVideoRef.current.srcObject = tertiaryStream;
        await tertiaryVideoRef.current.play();
      }
      setCameraStatus("connected");
      const connectedCount = 1 + Number(Boolean(secondaryStream)) + Number(Boolean(tertiaryStream));
      const connectionMessage = camerasWithoutFrames.length > 0
        ? `${camerasWithoutFrames.join(", ")}이(가) 연결됐지만 영상 프레임이 오지 않아 제외했어요. 여러 카메라가 같은 USB 허브를 공유하면 대역폭이 부족할 수 있어요. 해당 카메라를 노트북의 다른 USB 포트에 직접 연결한 뒤 다시 연결해 주세요.`
        : connectedCount < 3
        ? `외장 웹캠 ${connectedCount}대만 연결됐어요. 나머지 웹캠이 다른 앱에서 사용 중인지 확인해 주세요.`
        : savedAssignments.length < 3
          ? "새 노트북의 첫 연결이에요. 세 화면의 순서를 확인해 주세요. 이후부터 같은 순서로 연결해요."
          : "";
      await startPoseTracking(
        preferredPrimary.deviceId,
        preferredSecondary?.deviceId,
        preferredTertiary?.deviceId,
        connectionMessage,
        {
          1: preferredPrimary.label || "카메라 1",
          2: preferredSecondary?.label || "카메라 2",
          3: preferredTertiary?.label || "카메라 3",
        },
      );
    } catch (error) {
      console.error("Camera connection failed", error);
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

  function resetMotionLandmarkers() {
    poseLandmarkerRef.current?.close();
    secondaryPoseLandmarkerRef.current?.close();
    tertiaryPoseLandmarkerRef.current?.close();
    handLandmarkerRef.current?.close();
    poseLandmarkerRef.current = null;
    secondaryPoseLandmarkerRef.current = null;
    tertiaryPoseLandmarkerRef.current = null;
    handLandmarkerRef.current = null;
  }

  async function ensureMotionLandmarkers() {
    if (
      poseLandmarkerRef.current &&
      handLandmarkerRef.current &&
      (!secondaryStreamRef.current || secondaryPoseLandmarkerRef.current) &&
      (!tertiaryStreamRef.current || tertiaryPoseLandmarkerRef.current)
    ) return;
    updatePoseStatus("loading");
    const {
      FilesetResolver,
      HandLandmarker: HandLandmarkerClass,
      PoseLandmarker: PoseLandmarkerClass,
    } = await import("@mediapipe/tasks-vision");
    const vision = await FilesetResolver.forVisionTasks("/mediapipe-wasm");
    const poseBaseOptions = {
      modelAssetPath: "/models/pose_landmarker_heavy.task",
    };
    const handBaseOptions = {
      modelAssetPath: "/models/hand_landmarker.task",
    };
    const poseTrackingOptions = {
      runningMode: "VIDEO" as const,
      numPoses: 3,
      minPoseDetectionConfidence: 0.25,
      minPosePresenceConfidence: 0.25,
      minTrackingConfidence: 0.25,
      outputSegmentationMasks: false,
    };

    if (!poseLandmarkerRef.current) {
      try {
        poseLandmarkerRef.current = await PoseLandmarkerClass.createFromOptions(
          vision,
          { baseOptions: { ...poseBaseOptions, delegate: "GPU" }, ...poseTrackingOptions },
        );
      } catch {
        poseLandmarkerRef.current = await PoseLandmarkerClass.createFromOptions(
          vision,
          { baseOptions: poseBaseOptions, ...poseTrackingOptions },
        );
      }
    }

    const createPoseLandmarker = async () => {
      try {
        return await PoseLandmarkerClass.createFromOptions(
          vision,
          { baseOptions: { ...poseBaseOptions, delegate: "GPU" }, ...poseTrackingOptions },
        );
      } catch {
        return PoseLandmarkerClass.createFromOptions(
          vision,
          { baseOptions: poseBaseOptions, ...poseTrackingOptions },
        );
      }
    };
    if (secondaryStreamRef.current && !secondaryPoseLandmarkerRef.current) {
      secondaryPoseLandmarkerRef.current = await createPoseLandmarker();
    }
    if (tertiaryStreamRef.current && !tertiaryPoseLandmarkerRef.current) {
      tertiaryPoseLandmarkerRef.current = await createPoseLandmarker();
    }

    const createHandLandmarker = async () => {
      const options = {
        runningMode: "VIDEO" as const,
        numHands: 2,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
      };
      try {
        return await HandLandmarkerClass.createFromOptions(
          vision,
          { baseOptions: { ...handBaseOptions, delegate: "GPU" }, ...options },
        );
      } catch {
        try {
          return await HandLandmarkerClass.createFromOptions(
            vision,
            { baseOptions: handBaseOptions, ...options },
          );
        } catch {
          return null;
        }
      }
    };
    if (!handLandmarkerRef.current) handLandmarkerRef.current = await createHandLandmarker();
  }

  function flushPoseFrames(cameraSlot: CameraSlot = 1) {
    const bufferRef = cameraSlot === 1 ? poseBufferRef : cameraSlot === 2 ? secondaryPoseBufferRef : tertiaryPoseBufferRef;
    const sessionRef = cameraSlot === 1 ? poseSessionRef : cameraSlot === 2 ? secondaryPoseSessionRef : tertiaryPoseSessionRef;
    const writeQueueRef = cameraSlot === 1 ? poseWriteQueueRef : cameraSlot === 2 ? secondaryPoseWriteQueueRef : tertiaryPoseWriteQueueRef;
    const detectedRef = cameraSlot === 1 ? chunkDetectedFramesRef : cameraSlot === 2 ? secondaryChunkDetectedFramesRef : tertiaryChunkDetectedFramesRef;
    const fullBodyRef = cameraSlot === 1 ? chunkFullBodyFramesRef : cameraSlot === 2 ? secondaryChunkFullBodyFramesRef : tertiaryChunkFullBodyFramesRef;
    const handRef = cameraSlot === 1 ? chunkHandFramesRef : cameraSlot === 2 ? secondaryChunkHandFramesRef : tertiaryChunkHandFramesRef;
    if (bufferRef.current.length === 0 || !sessionRef.current) return writeQueueRef.current;
    const values = new Float32Array(bufferRef.current);
    const detectedFrames = detectedRef.current;
    const fullBodyFrames = fullBodyRef.current;
    const handDetectedFrames = handRef.current;
    const sessionId = sessionRef.current.id;
    bufferRef.current = [];
    detectedRef.current = 0;
    fullBodyRef.current = 0;
    handRef.current = 0;

    writeQueueRef.current = writeQueueRef.current.then(async () => {
      const session = sessionRef.current;
      if (!session || session.id !== sessionId) return;
      const updated = await appendMotionChunk(
        session,
        session.frameCount,
        values,
        detectedFrames,
        fullBodyFrames,
        handDetectedFrames,
      );
      setStoredMotionBytes((current) => current + updated.storageBytes - session.storageBytes);
      sessionRef.current = updated;
      if (cameraSlot === 1) setLatestSession(updated);
    });
    return writeQueueRef.current;
  }

  function recordPoseFrame(
    timestamp: number,
    landmarks: NormalizedLandmark[] | null,
    fullBody: boolean,
    head: HeadDirection | null,
    hands: HandState,
    cameraSlot: CameraSlot = 1,
  ) {
    if (!landmarks) return;
    const bufferRef = cameraSlot === 1 ? poseBufferRef : cameraSlot === 2 ? secondaryPoseBufferRef : tertiaryPoseBufferRef;
    const detectedRef = cameraSlot === 1 ? chunkDetectedFramesRef : cameraSlot === 2 ? secondaryChunkDetectedFramesRef : tertiaryChunkDetectedFramesRef;
    const fullBodyRef = cameraSlot === 1 ? chunkFullBodyFramesRef : cameraSlot === 2 ? secondaryChunkFullBodyFramesRef : tertiaryChunkFullBodyFramesRef;
    const handRef = cameraSlot === 1 ? chunkHandFramesRef : cameraSlot === 2 ? secondaryChunkHandFramesRef : tertiaryChunkHandFramesRef;
    const buffer = bufferRef.current;
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

    for (let index = 11; index < 33; index += 1) {
      const point = landmarks[index];
      buffer.push(point.x, point.y, point.z, point.visibility);
    }
    detectedRef.current += 1;

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

    if (fullBody) fullBodyRef.current += 1;
    if (hands.left || hands.right) handRef.current += 1;

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
      void flushPoseFrames(cameraSlot);
    }
  }

  function flushCameraHealth(cameraSlot: CameraSlot) {
    const globalSessionId = globalCaptureSessionRef.current?.id;
    const buffer = cameraHealthBuffersRef.current[cameraSlot];
    if (!globalSessionId || buffer.startSecond < 0 || buffer.samples.length === 0) {
      return cameraHealthWriteQueueRef.current;
    }
    const startSecond = buffer.startSecond;
    const samples = Uint8Array.from(buffer.samples);
    cameraHealthBuffersRef.current[cameraSlot] = { startSecond: -1, samples: [] };
    cameraHealthWriteQueueRef.current = cameraHealthWriteQueueRef.current.then(() =>
      appendCameraHealthChunk(globalSessionId, cameraSlot, startSecond, samples),
    );
    return cameraHealthWriteQueueRef.current;
  }

  function sampleCameraHealth(timestamp: number) {
    const origin = sessionPerformanceStartRef.current;
    if (!origin) return;
    const second = Math.max(0, Math.floor((timestamp - origin) / 1000));
    if (second <= lastCameraHealthSecondRef.current) return;
    const lastDetections: Record<CameraSlot, number> = {
      1: lastDetectionTimeRef.current,
      2: lastSecondaryDetectionTimeRef.current,
      3: lastTertiaryDetectionTimeRef.current,
    };
    const streams: Record<CameraSlot, MediaStream | null> = {
      1: streamRef.current,
      2: secondaryStreamRef.current,
      3: tertiaryStreamRef.current,
    };

    for (let sampledSecond = lastCameraHealthSecondRef.current + 1; sampledSecond <= second; sampledSecond += 1) {
      ([1, 2, 3] as CameraSlot[]).forEach((cameraSlot) => {
        const stream = streams[cameraSlot];
        const connected = Boolean(stream?.getVideoTracks().some((track) => track.readyState === "live"));
        const value = encodeCameraHealth({
          connected,
          receivingFrames: connected && timestamp - lastDetections[cameraSlot] < 1_500,
          personDetected: timestamp - lastPersonSeenByCameraRef.current[cameraSlot] < 1_500,
          trackingError: cameraTrackingErrorRef.current[cameraSlot],
        });
        const buffer = cameraHealthBuffersRef.current[cameraSlot];
        if (buffer.startSecond < 0) buffer.startSecond = sampledSecond;
        buffer.samples.push(value);
        if (buffer.samples.length >= CAMERA_HEALTH_CHUNK_SECONDS) void flushCameraHealth(cameraSlot);
      });
    }
    lastCameraHealthSecondRef.current = second;
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
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      video.currentTime !== lastVideoTimeRef.current &&
      timestamp - lastDetectionTimeRef.current >= 66
    ) {
      lastVideoTimeRef.current = video.currentTime;
      lastDetectionTimeRef.current = timestamp;
      try {
        const result = landmarker.detectForVideo(video, timestamp);
        cameraTrackingErrorRef.current[1] = false;
        const targetSelection = selectLockedPose(
          result.landmarks,
          poseTargetLockRef.current,
          timestamp,
        );
        poseTargetLockRef.current = targetSelection.lock;
        const liveLandmarks = targetSelection.landmarks;
        if (liveLandmarks) {
          lastPoseRef.current = liveLandmarks;
          lastPoseSeenAtRef.current = timestamp;
          lastPersonSeenByCameraRef.current[1] = timestamp;
        }
        const holdingLastPose = Boolean(
          !liveLandmarks &&
          targetSelection.lock &&
          lastPoseRef.current &&
          timestamp - lastPoseSeenAtRef.current <= POSE_DISPLAY_HOLD_MS,
        );
        const displayLandmarks = liveLandmarks ?? (holdingLastPose ? lastPoseRef.current : null);
        const nextTargetLocked = Boolean(liveLandmarks);
        setTargetLocked((current) => current === nextTargetLocked ? current : nextTargetLocked);
        const fullBody = liveLandmarks ? isFullBodyVisible(liveLandmarks) : false;
        const displayFullBody = displayLandmarks ? isFullBodyVisible(displayLandmarks) : false;
        const head = liveLandmarks ? getHeadDirection(liveLandmarks) : null;
        if (liveLandmarks) {
          lastHeadDirectionRef.current = head;
          const nextHeadLabel = describeHeadDirection(head);
          setHeadDirectionLabel((current) =>
            current === nextHeadLabel ? current : nextHeadLabel,
          );
        }
        const displayHead = liveLandmarks
          ? head
          : holdingLastPose
            ? lastHeadDirectionRef.current
            : null;

        if (liveLandmarks && handLandmarker && timestamp - lastHandDetectionTimeRef.current >= 100) {
          lastHandDetectionTimeRef.current = timestamp;
          const handResult = handLandmarker.detectForVideo(video, timestamp);
          const hands: HandState = {
            left: null,
            right: null,
            leftScore: 0,
            rightScore: 0,
          };
          handResult.landmarks.forEach((hand, index) => {
            const matchedSide = matchHandToPose(hand, liveLandmarks);
            if (!matchedSide) return;
            const category = handResult.handedness[index]?.[0];
            if (matchedSide === "left") {
              hands.left = hand;
              hands.leftScore = category?.score ?? 0;
            } else {
              hands.right = hand;
              hands.rightScore = category?.score ?? 0;
            }
          });
          lastHandsRef.current = hands;
          const handCount = Number(Boolean(hands.left)) + Number(Boolean(hands.right));
          setDetectedHands((current) => (current === handCount ? current : handCount));
          setDetectedHandSides((current) => {
            const next = {
              left: Boolean(hands.left),
              right: Boolean(hands.right),
            };
            return current.left === next.left && current.right === next.right ? current : next;
          });
        }
        if (!liveLandmarks && !holdingLastPose) {
          lastHandsRef.current = {
            left: null,
            right: null,
            leftScore: 0,
            rightScore: 0,
          };
          setDetectedHands((current) => current === 0 ? current : 0);
          setDetectedHandSides((current) =>
            !current.left && !current.right ? current : { left: false, right: false },
          );
        }
        updatePoseStatus(
          liveLandmarks
            ? (fullBody ? "full" : "partial")
            : holdingLastPose
              ? "holding"
              : "searching",
        );

        const canvas = overlayCanvasRef.current;
        if (canvas) {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          const context = canvas.getContext("2d");
          context?.clearRect(0, 0, canvas.width, canvas.height);
          if (displayLandmarks) {
            drawMotionSkeleton(
              canvas,
              displayLandmarks,
              lastHandsRef.current.left,
              lastHandsRef.current.right,
              displayHead,
              displayFullBody,
            );
          }
        }

        if (timestamp - lastSampleTimeRef.current >= 1000 / MOTION_SAMPLE_RATE) {
          lastSampleTimeRef.current = timestamp;
          const recordingHands = displayLandmarks
            ? lastHandsRef.current
            : { left: null, right: null, leftScore: 0, rightScore: 0 };
          recordPoseFrame(
            timestamp,
            liveLandmarks,
            fullBody,
            head,
            liveLandmarks ? recordingHands : { left: null, right: null, leftScore: 0, rightScore: 0 },
          );
        }
      } catch (error) {
        console.error("Primary camera pose tracking failed", error);
        cameraTrackingErrorRef.current[1] = true;
        updatePoseStatus("error");
      }
    }
    trackAuxiliaryPose(timestamp, 2);
    trackAuxiliaryPose(timestamp, 3);
    sampleCameraHealth(timestamp);
    poseAnimationRef.current = requestAnimationFrame(poseTrackingLoop);
  }

  function trackAuxiliaryPose(timestamp: number, cameraSlot: 2 | 3) {
    const video = cameraSlot === 2 ? secondaryVideoRef.current : tertiaryVideoRef.current;
    const canvas = cameraSlot === 2 ? secondaryOverlayCanvasRef.current : tertiaryOverlayCanvasRef.current;
    const landmarker = cameraSlot === 2 ? secondaryPoseLandmarkerRef.current : tertiaryPoseLandmarkerRef.current;
    // Hand detection is stateless enough for our sampled frames; sharing one
    // model avoids loading three large WASM graphs at camera startup.
    const handLandmarker = handLandmarkerRef.current;
    const targetLockRef = cameraSlot === 2 ? secondaryPoseTargetLockRef : tertiaryPoseTargetLockRef;
    const handsRef = cameraSlot === 2 ? secondaryHandsRef : tertiaryHandsRef;
    const lastVideoTimeRef = cameraSlot === 2 ? lastSecondaryVideoTimeRef : lastTertiaryVideoTimeRef;
    const lastDetectionRef = cameraSlot === 2 ? lastSecondaryDetectionTimeRef : lastTertiaryDetectionTimeRef;
    const lastHandDetectionRef = cameraSlot === 2 ? lastSecondaryHandDetectionTimeRef : lastTertiaryHandDetectionTimeRef;
    const lastSampleRef = cameraSlot === 2 ? lastSecondarySampleTimeRef : lastTertiarySampleTimeRef;
    const setStatus = cameraSlot === 2 ? setSecondaryPoseStatus : setTertiaryPoseStatus;
    if (!video || !landmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.currentTime === lastVideoTimeRef.current || timestamp - lastDetectionRef.current < 100) return;
    lastVideoTimeRef.current = video.currentTime;
    lastDetectionRef.current = timestamp;
    try {
      const result = landmarker.detectForVideo(video, timestamp);
      cameraTrackingErrorRef.current[cameraSlot] = false;
      const targetSelection = selectLockedPose(
        result.landmarks,
        targetLockRef.current,
        timestamp,
      );
      targetLockRef.current = targetSelection.lock;
      const landmarks = targetSelection.landmarks;
      if (landmarks) lastPersonSeenByCameraRef.current[cameraSlot] = timestamp;
      if (landmarks && handLandmarker && timestamp - lastHandDetectionRef.current >= 100) {
        lastHandDetectionRef.current = timestamp;
        const hands: HandState = { left: null, right: null, leftScore: 0, rightScore: 0 };
        const handResult = handLandmarker.detectForVideo(video, timestamp + cameraSlot - 1);
        handResult.landmarks.forEach((hand, index) => {
          const side = matchHandToPose(hand, landmarks);
          if (!side) return;
          const score = handResult.handedness[index]?.[0]?.score ?? 0;
          if (side === "left") { hands.left = hand; hands.leftScore = score; }
          else { hands.right = hand; hands.rightScore = score; }
        });
        handsRef.current = hands;
      }
      if (!landmarks) handsRef.current = { left: null, right: null, leftScore: 0, rightScore: 0 };
      const fullBody = landmarks ? isFullBodyVisible(landmarks) : false;
      const head = landmarks ? getHeadDirection(landmarks) : null;
      if (canvas) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        const context = canvas.getContext("2d");
        context?.clearRect(0, 0, canvas.width, canvas.height);
        if (landmarks) drawMotionSkeleton(canvas, landmarks, handsRef.current.left, handsRef.current.right, head, fullBody);
      }
      if (timestamp - lastSampleRef.current >= 1000 / MOTION_SAMPLE_RATE) {
        lastSampleRef.current = timestamp;
        recordPoseFrame(timestamp, landmarks, fullBody, head, handsRef.current, cameraSlot);
      }
      setStatus(landmarks ? (fullBody ? "full" : "partial") : "searching");
    } catch (error) {
      console.error(`Camera ${cameraSlot} pose tracking failed`, error);
      cameraTrackingErrorRef.current[cameraSlot] = true;
      setStatus("error");
    }
  }

  async function startPoseTracking(
    primaryId = primaryCameraId,
    secondaryId = secondaryCameraId,
    tertiaryId = tertiaryCameraId,
    connectionMessage = "",
    cameraLabels: Partial<Record<CameraSlot, string>> = {},
  ) {
    try {
      await ensureMotionLandmarkers();
      const startedAt = currentEpochTime();
      const connectedCameras = [
        { cameraId: primaryId, slot: 1 as const, label: cameraLabels[1] || "카메라 1" },
        ...(secondaryStreamRef.current ? [{ cameraId: secondaryId, slot: 2 as const, label: cameraLabels[2] || "카메라 2" }] : []),
        ...(tertiaryStreamRef.current ? [{ cameraId: tertiaryId, slot: 3 as const, label: cameraLabels[3] || "카메라 3" }] : []),
      ];
      globalCaptureSessionRef.current = createGlobalCaptureSession(connectedCameras);
      const sharedSession = {
        globalSessionId: globalCaptureSessionRef.current.id,
        timelineOriginMs: globalCaptureSessionRef.current.timelineOriginMs,
      };
      const session = await createMotionSession(
        `motion-${crypto.randomUUID()}`,
        startedAt,
        { ...sharedSession, cameraId: primaryId, cameraSlot: 1, cameraLabel: cameraLabels[1], logicalCameraId: "CAMERA_1" },
      );
      poseSessionRef.current = session;
      secondaryPoseSessionRef.current = secondaryStreamRef.current
        ? await createMotionSession(
          `motion-${crypto.randomUUID()}`,
          startedAt,
          { ...sharedSession, cameraId: secondaryId, cameraSlot: 2, cameraLabel: cameraLabels[2], logicalCameraId: "CAMERA_2" },
        )
        : null;
      tertiaryPoseSessionRef.current = tertiaryStreamRef.current
        ? await createMotionSession(
          `motion-${crypto.randomUUID()}`,
          startedAt,
          { ...sharedSession, cameraId: tertiaryId, cameraSlot: 3, cameraLabel: cameraLabels[3], logicalCameraId: "CAMERA_3" },
        )
        : null;
      await saveCaptureManifest({
        id: globalCaptureSessionRef.current.id,
        startedAt,
        endedAt: null,
        timelineOriginMs: globalCaptureSessionRef.current.timelineOriginMs,
        expectedCameraSlots: [1, 2, 3],
        cameras: ([1, 2, 3] as CameraSlot[]).map((slot) => {
          const ids = [primaryId, secondaryId, tertiaryId];
          const sessions = [session, secondaryPoseSessionRef.current, tertiaryPoseSessionRef.current];
          return {
            slot,
            logicalCameraId: `CAMERA_${slot}`,
            deviceId: ids[slot - 1] || "",
            label: cameraLabels[slot] || `카메라 ${slot}`,
            sessionId: sessions[slot - 1]?.id ?? null,
            connectedAtStart: Boolean(sessions[slot - 1]),
          };
        }),
        zoneSnapshot: {
          primary: [...observationProfile.zoneGrid],
          secondary: [...observationProfile.secondaryZoneGrid],
          tertiary: [...observationProfile.tertiaryZoneGrid],
          customZones: observationProfile.customZones.map((zone) => ({ ...zone })),
          profileUpdatedAt: observationProfile.updatedAt,
        },
        healthSampleRateHz: 1,
      });
      sessionPerformanceStartRef.current = globalCaptureSessionRef.current.timelineOriginMs;
      lastDetectionTimeRef.current = 0;
      lastHandDetectionTimeRef.current = 0;
      lastSampleTimeRef.current = 0;
      lastVideoTimeRef.current = -1;
      lastSecondaryVideoTimeRef.current = -1;
      lastSecondaryDetectionTimeRef.current = 0;
      lastSecondaryHandDetectionTimeRef.current = 0;
      lastSecondarySampleTimeRef.current = 0;
      lastTertiaryVideoTimeRef.current = -1;
      lastTertiaryDetectionTimeRef.current = 0;
      lastTertiaryHandDetectionTimeRef.current = 0;
      lastTertiarySampleTimeRef.current = 0;
      lastCameraHealthSecondRef.current = -1;
      cameraHealthBuffersRef.current = {
        1: { startSecond: -1, samples: [] },
        2: { startSecond: -1, samples: [] },
        3: { startSecond: -1, samples: [] },
      };
      lastPersonSeenByCameraRef.current = { 1: 0, 2: 0, 3: 0 };
      cameraTrackingErrorRef.current = { 1: false, 2: false, 3: false };
      secondaryHandsRef.current = { left: null, right: null, leftScore: 0, rightScore: 0 };
      tertiaryHandsRef.current = { left: null, right: null, leftScore: 0, rightScore: 0 };
      lastPoseRef.current = null;
      lastPoseSeenAtRef.current = 0;
      poseTargetLockRef.current = null;
      secondaryPoseTargetLockRef.current = null;
      tertiaryPoseTargetLockRef.current = null;
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
      secondaryPoseBufferRef.current = [];
      secondaryChunkDetectedFramesRef.current = 0;
      secondaryChunkFullBodyFramesRef.current = 0;
      secondaryChunkHandFramesRef.current = 0;
      tertiaryPoseBufferRef.current = [];
      tertiaryChunkDetectedFramesRef.current = 0;
      tertiaryChunkFullBodyFramesRef.current = 0;
      tertiaryChunkHandFramesRef.current = 0;
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
      setDetectedHandSides({ left: false, right: false });
      setTargetLocked(false);
      setHeadDirectionLabel("머리 방향 미확인");
      setSessionCount((count) => count + 1);
      updatePoseStatus("searching");
      setSecondaryPoseStatus(secondaryStreamRef.current ? "searching" : "idle");
      setTertiaryPoseStatus(tertiaryStreamRef.current ? "searching" : "idle");
      trackingActiveRef.current = true;
      if (connectionMessage) {
        setCameraMessage(connectionMessage);
      } else if (handLandmarkerRef.current) {
        setCameraMessage(secondaryStreamRef.current || tertiaryStreamRef.current
          ? `웹캠 ${connectedCameras.length}대의 몸·손 좌표를 한 명의 공통 시간축에 기록하고 있어요.`
          : "얼굴은 제외하고 몸·머리 방향·손가락 좌표를 기록하고 있어요.");
      } else {
        setCameraMessage("몸·머리 방향 좌표를 기록 중이에요. 손가락 추적은 이 기기에서 준비하지 못했어요.");
      }
      poseAnimationRef.current = requestAnimationFrame(poseTrackingLoop);
    } catch (error) {
      console.error("Motion tracking initialization failed", error);
      resetMotionLandmarkers();
      updatePoseStatus("error");
      setCameraMessage("몸·손 추적 모델을 불러오지 못했어요. 다시 연결해 주세요.");
      toast.success("동작 추적 모델을 준비하지 못했어요");
    }
  }

  // Analyzes a just-finished session's real coordinates with the same
  // motion detector used for the demo persona replay (app/motion-detection.ts)
  // and records behavior-pattern events into today's log. Safety-style
  // detections are deliberately ignored: camera movement cannot prove that
  // a device was left on or that a guest entered or exited.
  async function recordMotionDetections(sessionId: string, globalSessionId?: string) {
    try {
      const rawFrames = globalSessionId
        ? await getGlobalSessionFrames(globalSessionId)
        : await getSessionFrames(sessionId);
      const samples = motionSamplesFromRawFrames(rawFrames);
      const detections = detectMotionEvents(samples);
      for (const detection of detections) {
        if (detection.type === "double_check") {
          await recordDoubleCheck();
        } else if (detection.type === "micro_delay") {
          await recordMicroDelay((detection.endMs - detection.startMs) / 1000);
        }
      }
    } catch {
      // local-only analysis; ignore transient read errors
    }
  }

  async function recordObservationSession(
    sessionId: string,
    recordedAt: number,
    globalSessionId?: string,
  ) {
    try {
      const cameraStreams = globalSessionId
        ? await getGlobalSessionCameraFrames(globalSessionId)
        : [{ cameraSlot: 1 as const, frames: await getSessionFrames(sessionId) }];
      const rawFrames = globalSessionId
        ? await getGlobalSessionFrames(globalSessionId)
        : cameraStreams[0].frames;
      const motionSlice = sliceTargetMotion(rawFrames);
      const zoneAliases = Object.fromEntries(
        observationProfile.customZones
          .filter((zone) => zone.contextZoneId)
          .map((zone) => [zone.id, zone.contextZoneId!]),
      );
      const bestCamera = cameraStreams
        .map((stream) => ({
          ...stream,
          frames: stream.frames.filter((frame) => frame[0] >= motionSlice.startMs && frame[0] <= motionSlice.endMs),
        }))
        .sort((a, b) => b.frames.filter((frame) => frame[1] === 1).length - a.frames.filter((frame) => frame[1] === 1).length)[0];
      const features = extractObservationFeatures(
        bestCamera?.frames ?? motionSlice.frames,
        bestCamera?.cameraSlot === 3
          ? observationProfile.tertiaryZoneGrid
          : bestCamera?.cameraSlot === 2
            ? observationProfile.secondaryZoneGrid
            : observationProfile.zoneGrid,
        zoneAliases,
      );
      let episode: ReviewableObservationEpisode = createObservationEpisode({
        sessionId,
        recordedAt,
        profile: observationProfile,
        phase: phaseForHour(new Date(recordedAt).getHours()),
        features,
        baseline: observationBaseline,
        motionSlice: {
          startMs: motionSlice.startMs,
          endMs: motionSlice.endMs,
          durationSeconds: motionSlice.durationSeconds,
          originalDurationSeconds: motionSlice.originalDurationSeconds,
          excludedFrameCount: motionSlice.excludedFrameCount,
          reason: motionSlice.reason,
        },
      });
      const labelOptions = listWorkContextLabels(workContextConfig);
      let motionCandidates: Array<{ label: string; confidence: number }> = [];
      if (learnedMotions.length > 0) {
        const loadedActions = await Promise.all(learnedMotions.map(async (action) => ({
          id: action.id,
          label: action.label,
          samples: (await Promise.all(action.samples.map(framesForMotionSample))).filter(
            (frames) => frames.length >= 6,
          ),
        })));
        const learnedResult = classifyLearnedMotion(motionSlice.frames, loadedActions);
        motionCandidates = learnedResult.candidates.map((candidate) => ({
          label: candidate.label,
          confidence: Math.max(0, Math.min(1, Math.exp(-candidate.distance * 4.2))),
        }));
      }
      const candidates = normalizeActionCandidates(buildWorkContextCandidates({
        config: workContextConfig,
        recordedAt,
        motionCandidates,
      }));
      const bestCandidate = candidates[0];
      const hasGroundedMotionLabel = motionCandidates.some((motionCandidate) =>
        labelOptions.some((option) =>
          option.taskLabel.trim().toLocaleLowerCase() === motionCandidate.label.trim().toLocaleLowerCase(),
        ),
      );
      episode = {
        ...episode,
        taskType: bestCandidate?.taskType ?? "UNCLASSIFIED",
        taskLabel: bestCandidate?.taskLabel ?? "미분류",
        taskConfidence: bestCandidate?.confidence ?? 0,
      };
      const qualityReasons = [] as Array<"insufficient_frames" | "missing_zone" | "missing_time_context" | "unknown_motion">;
      if (motionSlice.frames.length < 6) qualityReasons.push("insufficient_frames");
      if (!features.dominantZone) qualityReasons.push("missing_zone");
      if (labelOptions.length === 0) qualityReasons.push("missing_time_context");
      if (!hasGroundedMotionLabel) qualityReasons.push("unknown_motion");
      const actionReview = createActionReview(episode, { candidates, qualityReasons });
      episode = {
        ...episode,
        globalSessionId,
        ...(actionReview ? {
          actionReview,
          disposition: "quarantined" as const,
          dispositionReason: "업무 라벨이 애매해 개발자 검토 전까지 기준선 학습에서 보류했어요.",
        } : {}),
      };
      await saveObservationEpisode(episode);
      await refreshObservationData(observationProfile);
      toast.success(
        episode.actionReview?.status === "pending"
          ? "업무 라벨이 애매한 동작을 개발자 검토함에 보관했어요"
          : episode.disposition === "quarantined"
            ? "평소 흐름으로 확정하기 어려운 동작은 학습에서 잠시 보류했어요"
            : observationProfile.mode === "learning"
              ? `${episode.taskLabel} 패턴을 학습 기록에 추가했어요`
              : `${episode.taskLabel} 동작을 개인 기준과 비교했어요`,
      );
    } catch {
      // Coordinate recording remains available even if contextual analysis fails.
    }
  }

  async function stopPoseTracking(analyzeSession = true) {
    sampleCameraHealth(currentMonotonicTime());
    trackingActiveRef.current = false;
    if (poseAnimationRef.current !== null) {
      cancelAnimationFrame(poseAnimationRef.current);
      poseAnimationRef.current = null;
    }
    await Promise.all([flushPoseFrames(1), flushPoseFrames(2), flushPoseFrames(3)]);
    await Promise.all([poseWriteQueueRef.current, secondaryPoseWriteQueueRef.current, tertiaryPoseWriteQueueRef.current]);
    await Promise.all([flushCameraHealth(1), flushCameraHealth(2), flushCameraHealth(3)]);
    await cameraHealthWriteQueueRef.current;
    const session = poseSessionRef.current;
    const secondarySession = secondaryPoseSessionRef.current;
    const tertiarySession = tertiaryPoseSessionRef.current;
    const endedAt = currentEpochTime();
    if (globalCaptureSessionRef.current) {
      await finishCaptureManifest(globalCaptureSessionRef.current.id, endedAt);
    }
    let secondaryFrameCount = 0;
    if (secondarySession) {
      const completedSecondary = await finishMotionSession(secondarySession, endedAt);
      secondaryFrameCount = completedSecondary.frameCount;
      secondaryPoseSessionRef.current = null;
    }
    if (tertiarySession) {
      const completedTertiary = await finishMotionSession(tertiarySession, endedAt);
      secondaryFrameCount += completedTertiary.frameCount;
      tertiaryPoseSessionRef.current = null;
    }
    if (session) {
      const completed = await finishMotionSession(session, endedAt);
      setLatestSession(completed);
      setRecentSessions((previous) => [completed, ...previous.filter((s) => s.id !== completed.id)].slice(0, 5));
      poseSessionRef.current = null;
      if (analyzeSession && consent.observationConsent && completed.frameCount + secondaryFrameCount >= 20) {
        void recordMotionDetections(completed.id, completed.globalSessionId).then(refreshCareData);
        void recordObservationSession(completed.id, endedAt, completed.globalSessionId);
      }
      void getLatestBodyProportionProfile().then(async (bodyProportionProfile) => {
        if (!bodyProportionProfile) return;
        const currentProfile = await getObservationProfile();
        const next = await saveObservationProfile({
          ...currentProfile,
          bodyProportionProfile,
        });
        setObservationProfile(next);
      }).catch(() => undefined);
    }
    overlayCanvasRef.current
      ?.getContext("2d")
      ?.clearRect(
        0,
        0,
        overlayCanvasRef.current.width,
        overlayCanvasRef.current.height,
      );
    secondaryOverlayCanvasRef.current?.getContext("2d")?.clearRect(0, 0, secondaryOverlayCanvasRef.current.width, secondaryOverlayCanvasRef.current.height);
    tertiaryOverlayCanvasRef.current?.getContext("2d")?.clearRect(0, 0, tertiaryOverlayCanvasRef.current.width, tertiaryOverlayCanvasRef.current.height);
    updatePoseStatus("idle");
    setSecondaryPoseStatus("idle");
    setTertiaryPoseStatus("idle");
  }

  async function stopCamera({ analyzeSession = true, notify = true } = {}) {
    if (document.fullscreenElement === cameraFeedLayoutRef.current) {
      await document.exitFullscreen().catch(() => undefined);
    }
    await stopPoseTracking(analyzeSession);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    secondaryStreamRef.current?.getTracks().forEach((track) => track.stop());
    tertiaryStreamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    secondaryStreamRef.current = null;
    tertiaryStreamRef.current = null;
    setSecondaryCameraConnected(false);
    setTertiaryCameraConnected(false);
    if (videoRef.current) videoRef.current.srcObject = null;
    if (secondaryVideoRef.current) secondaryVideoRef.current.srcObject = null;
    if (tertiaryVideoRef.current) tertiaryVideoRef.current.srcObject = null;
    if (processingVideoRef.current) processingVideoRef.current.srcObject = null;
    setCameraStatus("idle");
    poseTargetLockRef.current = null;
    setTargetLocked(false);
    setElapsedSeconds(0);
    setCameraMessage("카메라 연결을 멈췄어요.");
    if (notify) toast.success("좌표 기록을 안전하게 저장하고 카메라를 종료했어요");
  }

  async function changeCameraSource(slot: CameraSlot, deviceId: string) {
    const next = [primaryCameraId, secondaryCameraId, tertiaryCameraId];
    const previousIndex = next.indexOf(deviceId);
    if (previousIndex >= 0) next[previousIndex] = next[slot - 1];
    next[slot - 1] = deviceId;
    const [nextPrimary, nextSecondary, nextTertiary] = next;
    if (!nextPrimary || new Set(next.filter(Boolean)).size !== next.filter(Boolean).length) return;
    setPrimaryCameraId(nextPrimary);
    setSecondaryCameraId(nextSecondary);
    setTertiaryCameraId(nextTertiary);
    if (cameraStatus === "connected") {
      await stopCamera({ analyzeSession: false, notify: false });
      await startCamera(nextPrimary, nextSecondary, nextTertiary);
    }
  }

  function markTestEvent(preset: (typeof eventPresets)[number]) {
    const session = poseSessionRef.current;
    const landmarks = lastPoseRef.current;
    const head = lastHeadDirectionRef.current;
    const hands = lastHandsRef.current;
    if (!session || cameraStatus !== "connected") {
      toast.success("먼저 카메라를 연결해 주세요");
      return;
    }
    if (!landmarks || !head) {
      toast.success("스켈레톤이 인식된 뒤 이벤트를 표시해 주세요");
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
    toast.success("이벤트 시점을 좌표 기록에 표시했어요");

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
      const exported = await downloadMotionDataFolder();
      if (!exported) {
        toast.success("내보낼 좌표 기록이 아직 없어요");
        return;
      }
      toast.success("전체 기록을 하나의 데이터 폴더로 내려받았어요");
    } catch {
      toast.success("좌표 데이터를 내보내지 못했어요");
    }
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
      detectionExplanation: explainDemoEvent(persona, dayIndex, day, frames),
    });
  }

  async function changeOccupation(occupationName: string) {
    const inference = inferOccupationContext(occupationName);
    const occupation = inference.template.id;
    const template = getOccupationTemplate(occupation);
    const next = await saveObservationProfile({
      ...observationProfile,
      occupation,
      occupationName,
      mode: "learning",
      learningStartedAt: nowMs(),
      baselineVersion: observationProfile.baselineVersion + 1,
      zoneGrid: Array(9).fill(null),
      secondaryZoneGrid: Array(9).fill(null),
      tertiaryZoneGrid: Array(9).fill(null),
      customZones: observationProfile.customZones.map((zone) => ({
        ...zone,
        contextZoneId: inferZoneContext(template, zone.label)?.id ?? null,
      })),
    });
    setObservationProfile(next);
    setOccupationInput(occupationName);
    setSelectedZoneId(template.zones[0]?.id ?? "");
    await refreshObservationData(next);
    toast.success(`‘${occupationName}’ 업종을 ${template.label} 업무 맥락으로 이해했어요`);
  }

  function saveOccupation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const name = occupationInput.trim();
    if (!name) {
      toast.success("업종 이름을 입력해 주세요");
      return;
    }
    void changeOccupation(name);
  }

  function beginOnboarding(step: OnboardingStep = 1) {
    setOnboardingStep(step);
    setView("onboarding");
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  function openSettings(section: SettingsSection = null) {
    setSettingsSection(section);
    setView("settings");
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  async function continueOnboardingFromOccupation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const name = occupationInput.trim();
    if (!name) {
      toast.error("업종 이름을 입력해 주세요");
      return;
    }
    if (name !== observationProfile.occupationName.trim()) {
      await changeOccupation(name);
    }
    setOnboardingStep(2);
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  function continueOnboardingFromZones() {
    const configuredCells = [
      ...observationProfile.zoneGrid,
      ...observationProfile.secondaryZoneGrid,
      ...observationProfile.tertiaryZoneGrid,
    ].filter(Boolean).length;
    if (configuredCells === 0) {
      toast.error("매장 구역을 하나 이상 지정해 주세요");
      return;
    }
    setOnboardingStep(3);
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  function continueOnboardingFromRoutine() {
    const routine = workContextConfig.routines[0];
    if (!routine?.name.trim()) {
      toast.error("업무 루틴 이름을 입력해 주세요");
      return;
    }
    if (routine.days.length === 0) {
      toast.error("업무 루틴을 적용할 요일을 하나 이상 선택해 주세요");
      return;
    }
    setOnboardingStep(4);
    window.scrollTo({ top: 0, behavior: "auto" });
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
    });
    setObservationProfile(next);
    await refreshObservationData(next);
    toast.success(
      mode === "learning"
        ? "기존 기준선은 보관하고 새로운 평소 흐름을 학습해요"
        : observationBaseline.confidence < 70
          ? "기록이 아직 적어 임시 기준으로 분석을 시작해요"
          : "학습한 개인 업무 패턴을 기준으로 분석을 시작해요",
    );
  }

  async function assignZoneCell(index: number) {
    const key = zoneCameraSlot === 1 ? "zoneGrid" : zoneCameraSlot === 2 ? "secondaryZoneGrid" : "tertiaryZoneGrid";
    const zoneGrid = [...observationProfile[key]];
    zoneGrid[index] = zoneGrid[index] === selectedZoneId ? null : selectedZoneId;
    const next = await saveObservationProfile({ ...observationProfile, [key]: zoneGrid });
    setObservationProfile(next);
  }

  async function addCustomZone() {
    const label = customZoneName.trim();
    if (!label) return;
    const existing = observationProfile.customZones.find((zone) => zone.label === label);
    if (existing) {
      setSelectedZoneId(existing.id);
      setCustomZoneName("");
      return;
    }
    const context = inferZoneContext(getOccupationTemplate(observationProfile.occupation), label);
    const zone = { id: `CUSTOM:${crypto.randomUUID()}`, label, contextZoneId: context?.id ?? null };
    const next = await saveObservationProfile({
      ...observationProfile,
      customZones: [...observationProfile.customZones, zone],
    });
    setObservationProfile(next);
    setSelectedZoneId(zone.id);
    setCustomZoneName("");
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
    toast.success("하던 업무를 이어서 완료했어요");
    if (consent.observationConsent) {
      void recordTaskCompleted().then(refreshCareData);
      if (bookingShownAtRef.current !== null) {
        void recordMicroDelay((nowMs() - bookingShownAtRef.current) / 1000);
      }
    }
  }

  function dismissSavepoint() {
    setSavepointOpen(false);
    toast.success("나중에 다시 확인할 수 있어요. 세이브포인트는 그대로 남아있어요.");
    // Left uncompleted on purpose: this is what "업무 누락율 (Drop Rate)"
    // is meant to observe - a started task that never got finished.
  }

  function openTimelineEvent(event: TimelineEvent) {
    setSelectedEvent(event);
  }

  function openQuickMotionMode(mode: "training" | "testing") {
    setQuickMotionMode(mode);
    setDraftMotionSamples([]);
    setCustomMotionLabel("");
    setMotionCapture(null);
    setLearnedMotionResult(null);
  }

  async function connectQuickMotionCamera() {
    setQuickMotionCameraOpen(true);
    quickMotionCameraRequestedRef.current = true;
    if (!consent.decided) {
      pendingCameraStartRef.current = true;
      setShowConsentModal(true);
      return;
    }
    if (cameraStatus === "connected") await stopCamera();
    await startCamera(undefined, undefined, undefined, true);
  }

  async function closeQuickMotionCamera() {
    setQuickMotionCameraOpen(false);
    setMotionCapture(null);
    quickMotionCameraRequestedRef.current = false;
    if (cameraStatus === "connected") await stopCamera();
  }

  function startQuickMotionCapture() {
    const session = poseSessionRef.current;
    if (cameraStatus !== "connected" || !session) {
      toast.success("카메라 연결이 끝난 뒤 다시 눌러 주세요");
      return;
    }
    setLearnedMotionResult(null);
    setMotionCapture({
      sessionId: session.id,
      globalSessionId: session.globalSessionId,
      startMs: currentMonotonicTime() - sessionPerformanceStartRef.current,
    });
  }

  async function framesForMotionSample(sample: LearnedMotionSample) {
    const frames = sample.globalSessionId
      ? await getGlobalSessionFrames(sample.globalSessionId)
      : await getSessionFrames(sample.sessionId);
    return frames.filter((frame) => frame[0] >= sample.startMs && frame[0] <= sample.endMs);
  }

  async function finishQuickMotionCapture() {
    if (!motionCapture || quickMotionBusy) return;
    const endMs = currentMonotonicTime() - sessionPerformanceStartRef.current;
    if (endMs - motionCapture.startMs < 700) {
      toast.success("동작을 1초 정도 보여준 뒤 완료해 주세요");
      return;
    }

    const sample: LearnedMotionSample = { ...motionCapture, endMs };
    setQuickMotionBusy(true);
    try {
      await Promise.all([flushPoseFrames(1), flushPoseFrames(2)]);
      await Promise.all([poseWriteQueueRef.current, secondaryPoseWriteQueueRef.current]);
      setMotionCapture(null);

      if (quickMotionMode === "training") {
        const frames = await framesForMotionSample(sample);
        if (frames.length < 6) {
          toast.success("상반신 좌표가 부족해요. 얼굴·어깨·양팔이 보이도록 다시 해주세요");
          return;
        }
        const nextSamples = [...draftMotionSamples, sample];
        setDraftMotionSamples(nextSamples);
        if (nextSamples.length >= trainingTargetCount) {
          setQuickMotionMode("labeling");
          await closeQuickMotionCamera();
        }
        return;
      }

      const observedFrames = await framesForMotionSample(sample);
      const loadedActions = await Promise.all(learnedMotions.map(async (action) => ({
        id: action.id,
        label: action.label,
        samples: (await Promise.all(action.samples.map(framesForMotionSample))).filter(
          (frames) => frames.length >= 6,
        ),
      })));
      setLearnedMotionResult(classifyLearnedMotion(observedFrames, loadedActions));
    } catch {
      toast.success("좌표를 불러오지 못했어요. 다시 시도해 주세요");
    } finally {
      setQuickMotionBusy(false);
    }
  }

  function saveQuickMotion() {
    const label = customMotionLabel.trim();
    if (!label) {
      toast.success("동작 이름을 입력해 주세요");
      return;
    }
    if (learnedMotions.some((motion) => motion.label.toLocaleLowerCase() === label.toLocaleLowerCase())) {
      toast.success("이미 같은 이름의 동작이 있어요");
      return;
    }
    const next = [...learnedMotions, {
      id: `learned-${crypto.randomUUID()}`,
      label,
      samples: draftMotionSamples,
      createdAt: nowMs(),
    }];
    saveLearnedMotionActions(next);
    setLearnedMotions(next);
    setQuickMotionMode("idle");
    setDraftMotionSamples([]);
    setCustomMotionLabel("");
    setQuickMotionCameraOpen(false);
    toast.success(`${label} 동작을 ${trainingTargetCount}회 표본으로 저장했어요`);
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
    clearConsent();
    window.localStorage.removeItem(CHECKLIST_STORAGE_KEY);
    window.localStorage.removeItem(WORK_CONTEXT_CONFIG_STORAGE_KEY);
    window.localStorage.removeItem(USER_INSTALL_STORAGE_KEY);
    clearLearnedMotionActions();
    setConsentState(getConsent());
    setUserInstall(null);
    setSessionCount(0);
    setLatestSession(null);
    setRecentSessions([]);
    setReplaySessionId(null);
    setMotionSignal(null);
    setCareLogs([]);
    setLearnedMotions([]);
    setQuickMotionMode("idle");
    setClosingChecklist(DEFAULT_CLOSING_CHECKLIST);
    setWorkContextConfig(DEFAULT_WORK_CONTEXT_CONFIG);
    const resetProfile = {
      ...DEFAULT_PROFILE,
      learningStartedAt: nowMs(),
      updatedAt: nowMs(),
    };
    setObservationProfile(resetProfile);
    setOccupationInput("");
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
    setStoredMotionBytes(0);
    toast.success("저장된 동작 좌표와 케어 기록을 모두 삭제했어요");
  }

  async function resetToFirstScreen() {
    const confirmed = window.confirm(
      "모든 사용자 기록, 학습 기준선, 동작 좌표와 설정을 지우고 처음 화면으로 돌아갈까요? 이 작업은 되돌릴 수 없어요.",
    );
    if (!confirmed) return;
    await deleteAllMyData();
    window.localStorage.removeItem(INTERFACE_MODE_STORAGE_KEY);
    setInterfaceMode("user");
    setDemoMode(true);
    setMyDataOpen(false);
    setView("home");
    toast.success("모든 내용을 초기화하고 처음 시작 화면으로 돌아왔어요.");
  }

  function withdrawObservationConsent() {
    const next = setConsent(false);
    setConsentState(next);
    toast.success("관찰 참여를 철회했어요. 매장 안전 기능은 계속 사용할 수 있어요.");
  }

  function toggleChecklistItem(itemId: string) {
    const nextItems = closingChecklist.map((item) =>
      item.id === itemId ? { ...item, done: !item.done } : item,
    );
    setClosingChecklist(nextItems);
    if (consent.observationConsent && !checklistTaskStartedRef.current) {
      checklistTaskStartedRef.current = true;
      void recordTaskStarted().then(refreshCareData);
    }
    if (
      consent.observationConsent &&
      nextItems.length > 0 &&
      nextItems.every((item) => item.done)
    ) {
      void recordTaskCompleted().then(refreshCareData);
    }
  }

  function addChecklistItem(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const label = newChecklistItem.trim();
    if (!label) return;
    setClosingChecklist((items) => [
      ...items,
      { id: crypto.randomUUID(), label, done: false },
    ]);
    setNewChecklistItem("");
  }

  function removeChecklistItem(itemId: string) {
    setClosingChecklist((items) => items.filter((item) => item.id !== itemId));
  }

  function addRoutine() {
    setWorkContextConfig((config) => ({
      ...config,
      routines: [
        ...config.routines,
        {
          id: crypto.randomUUID(),
          name: `루틴 ${config.routines.length + 1}`,
          days: [],
          openTime: DEFAULT_WORK_ROUTINE.openTime,
          closeTime: DEFAULT_WORK_ROUTINE.closeTime,
          schedule: [],
        },
      ],
    }));
  }

  function removeRoutine(routineId: string) {
    setWorkContextConfig((config) => {
      if (config.routines.length <= 1) return config;
      const removed = config.routines.find((routine) => routine.id === routineId);
      const remaining = config.routines.filter((routine) => routine.id !== routineId);
      if (!removed || removed.days.length === 0) return { ...config, routines: remaining };
      const [first, ...rest] = remaining;
      const merged = { ...first, days: [...new Set([...first.days, ...removed.days])].sort() as WeekdayIndex[] };
      return { ...config, routines: [merged, ...rest] };
    });
  }

  function renameRoutine(routineId: string, name: string) {
    setWorkContextConfig((config) => ({
      ...config,
      routines: config.routines.map((routine) => (routine.id === routineId ? { ...routine, name } : routine)),
    }));
  }

  function setRoutineTime(routineId: string, field: "openTime" | "closeTime", value: string) {
    setWorkContextConfig((config) => ({
      ...config,
      routines: config.routines.map((routine) => (routine.id === routineId ? { ...routine, [field]: value } : routine)),
    }));
  }

  function assignDayToRoutine(routineId: string, day: WeekdayIndex) {
    setWorkContextConfig((config) => ({
      closedDays: config.closedDays.filter((closedDay) => closedDay !== day),
      routines: config.routines.map((routine) => ({
        ...routine,
        days: routine.id === routineId
          ? (routine.days.includes(day) ? routine.days : [...routine.days, day].sort())
          : routine.days.filter((existing) => existing !== day),
      })),
    }));
  }

  function toggleClosedDay(day: WeekdayIndex) {
    setWorkContextConfig((config) => {
      const isClosed = config.closedDays.includes(day);
      if (isClosed) {
        const [first, ...rest] = config.routines;
        return {
          closedDays: config.closedDays.filter((closedDay) => closedDay !== day),
          routines: first ? [{ ...first, days: [...first.days, day].sort() }, ...rest] : config.routines,
        };
      }
      return {
        closedDays: [...config.closedDays, day].sort(),
        routines: config.routines.map((routine) => ({ ...routine, days: routine.days.filter((existing) => existing !== day) })),
      };
    });
  }

  function removeRoutineScheduleItem(routineId: string, itemId: string) {
    setWorkContextConfig((config) => ({
      ...config,
      routines: config.routines.map((routine) =>
        routine.id === routineId
          ? { ...routine, schedule: routine.schedule.filter((item) => item.id !== itemId) }
          : routine,
      ),
    }));
  }

  function handleAddRoutineItem(routineId: string) {
    return (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const form = event.currentTarget;
      const data = new FormData(form);
      const time = String(data.get("time") ?? "");
      const label = String(data.get("label") ?? "").trim();
      const repeats = data.get("repeats") === "on";
      if (!label || !time) return;
      setWorkContextConfig((config) => ({
        ...config,
        routines: config.routines.map((routine) =>
          routine.id === routineId
            ? {
              ...routine,
              schedule: [...routine.schedule, { id: crypto.randomUUID(), time, label, repeats }].sort((a, b) =>
                a.time.localeCompare(b.time),
              ),
            }
            : routine,
        ),
      }));
      form.reset();
    };
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
        : poseStatus === "holding"
          ? "대상 잠금 유지 중"
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

  const userCareLogs = scopeCareLogsToUserInstall(careLogs, userInstall);
  const recentCareLogs = userCareLogs.slice(0, 7);
  const careBaseline = computeBaseline(userCareLogs);
  const changeSignal = detectChangeSignal(recentCareLogs, careBaseline);
  const cognitiveConcernDetected = shouldShowCognitiveSupport(
    changeSignal,
    recentCareLogs,
    careBaseline,
  );
  const todaySummary = userCareLogs[0] ? summarizeLog(userCareLogs[0]) : null;
  const activeNavItems = interfaceMode === "developer" ? developerNavItems : userNavItems;
  const pageKey = `${interfaceMode}-${view}-${onboardingStep}-${settingsSection ?? "root"}`;
  const pageMotionClass = animatedPageKey === pageKey ? " page-transition" : "";

  useEffect(() => {
    if (!pageMotionReadyRef.current) {
      pageMotionReadyRef.current = true;
      return;
    }
    setAnimatedPageKey(pageKey);
    const timeout = window.setTimeout(() => setAnimatedPageKey(null), 260);
    return () => window.clearTimeout(timeout);
  }, [pageKey]);

  useLayoutEffect(() => {
    const nav = mainNavRef.current;
    const pill = mainNavPillRef.current;
    if (!nav || !pill || view === "onboarding") return;

    const placePill = (animate: boolean) => {
      const activeTab = nav.querySelector<HTMLButtonElement>('.t-tab[aria-current="page"]');
      if (!activeTab) return;
      if (!animate) pill.style.transition = "none";
      pill.style.width = `${activeTab.offsetWidth}px`;
      pill.style.transform = `translateX(${activeTab.offsetLeft}px)`;
      if (!animate) {
        void pill.offsetWidth;
        pill.style.transition = "";
      }
    };

    const modeChanged = previousNavModeRef.current !== interfaceMode;
    placePill(navPillReadyRef.current && !modeChanged);
    navPillReadyRef.current = true;
    previousNavModeRef.current = interfaceMode;

    const handleResize = () => placePill(false);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [interfaceMode, view]);
  const realHasNotice = changeSignal.level !== "none";
  const realFlowMessage = !careBaseline
    ? "평소 흐름을 알아가는 중이에요."
    : changeSignal.level === "none"
      ? "최근 일주일은 대체로 평소와 비슷했어요."
      : changeSignal.reasons[0] ?? "평소와 다른 흐름이 조금 관찰됐어요.";
  const totalDoubleChecks = recentCareLogs.reduce((sum, log) => sum + log.doubleChecks, 0);
  const totalDroppedTasks = recentCareLogs.reduce(
    (sum, log) => sum + Math.max(0, log.tasksStarted - log.tasksCompleted),
    0,
  );
  const occupationTemplate = getOccupationTemplate(observationProfile.occupation);
  const occupationDisplayName = observationProfile.occupationName.trim() || occupationTemplate.label;
  const occupationDraftInference = inferOccupationContext(occupationInput);
  const onboardingRoutine = workContextConfig.routines[0] ?? DEFAULT_WORK_ROUTINE;
  const settingsSectionTitle = settingsSection === "occupation"
    ? "업종 변경"
    : settingsSection === "observation"
      ? "관찰 방식"
      : settingsSection === "context"
        ? "업무 맥락 등록"
        : settingsSection === "checklist"
          ? "마감 체크리스트 수정"
          : "설정";
  const zoneOptions = [...occupationTemplate.zones, ...observationProfile.customZones];
  const activeZoneGrid = zoneCameraSlot === 1
    ? observationProfile.zoneGrid
    : zoneCameraSlot === 2
      ? observationProfile.secondaryZoneGrid
      : observationProfile.tertiaryZoneGrid;
  const selectedCustomZone = observationProfile.customZones.find((zone) => zone.id === selectedZoneId);
  const selectedZoneContext = selectedCustomZone?.contextZoneId
    ? occupationTemplate.zones.find((zone) => zone.id === selectedCustomZone.contextZoneId)
    : null;
  const draftZoneContext = customZoneName.trim()
    ? inferZoneContext(occupationTemplate, customZoneName)
    : null;
  function routineSummary(routine: WorkRoutine) {
    return [
      `${formatClockLabel(routine.openTime)} 오픈 준비`,
      ...routine.schedule.map((item) => `${formatClockLabel(item.time)} ${item.label}${item.repeats ? " (반복)" : ""}`),
      `${formatClockLabel(routine.closeTime)} 마감`,
    ].join(" → ");
  }
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
  const mappedZoneCount = new Set(activeZoneGrid.filter(Boolean)).size;
  const latestObservationEpisode = currentObservationEpisodes[0] ?? null;
  const analysisObservationSignals = currentObservationEpisodes.filter(
    (episode) =>
      episode.mode === "analysis" &&
      episode.disposition !== "excluded" &&
      ((episode.durationZScore ?? 0) >= 1.5 || (episode.pauseZScore ?? 0) >= 1.5),
  );
  const pendingActionReviews = listPendingActionReviews(
    currentObservationEpisodes as ReviewableObservationEpisode[],
  );
  const selectedReviewEpisode = selectedReviewEpisodeId
    ? (currentObservationEpisodes.find((episode) => episode.id === selectedReviewEpisodeId) as ReviewableObservationEpisode | undefined)
    : undefined;
  const reviewLabelOptions = listWorkContextLabels(workContextConfig);
  const registeredReviewTaskTypes = new Set(reviewLabelOptions.map((option) => option.taskType));

  function actionReviewDisplayLabel(episode: ReviewableObservationEpisode) {
    return episode.actionReview?.candidates.find((candidate) => registeredReviewTaskTypes.has(candidate.taskType))?.taskLabel
      ?? (registeredReviewTaskTypes.has(episode.taskType) ? episode.taskLabel : "미분류");
  }

  function openActionReview(episode: ReviewableObservationEpisode) {
    const firstOption = reviewLabelOptions.find(
      (option) => option.taskType === episode.actionReview?.candidates[0]?.taskType,
    ) ?? reviewLabelOptions[0];
    setSelectedReviewTaskType(firstOption?.taskType ?? "");
    setSelectedReviewEpisodeId(episode.id);
  }

  async function confirmActionReview() {
    if (!selectedReviewEpisode || !selectedReviewTaskType) return;
    const selectedLabel = reviewLabelOptions.find((option) => option.taskType === selectedReviewTaskType);
    if (!selectedLabel) return;
    const updated = applyManualActionLabel(selectedReviewEpisode, {
      ...selectedLabel,
      reviewer: "developer",
    });
    await saveObservationEpisode(updated);
    if (updated.motionSlice) {
      const sample: LearnedMotionSample = {
        sessionId: updated.sessionId,
        globalSessionId: updated.globalSessionId,
        startMs: updated.motionSlice.startMs,
        endMs: updated.motionSlice.endMs,
      };
      const matching = learnedMotions.find(
        (motion) => motion.label.toLocaleLowerCase() === selectedLabel.taskLabel.toLocaleLowerCase(),
      );
      const next = matching
        ? learnedMotions.map((motion) => motion.id === matching.id
          ? { ...motion, samples: [...motion.samples.filter((item) => item.sessionId !== sample.sessionId), sample] }
          : motion)
        : [...learnedMotions, {
          id: `learned-${crypto.randomUUID()}`,
          label: selectedLabel.taskLabel,
          samples: [sample],
          createdAt: nowMs(),
        }];
      saveLearnedMotionActions(next);
      setLearnedMotions(next);
    }
    await refreshObservationData(observationProfile);
    setSelectedReviewEpisodeId(null);
    toast.success(`${selectedLabel.taskLabel} 업무로 확정하고 학습 표본에 반영했어요`);
  }

  async function leaveActionReviewUnresolved() {
    if (!selectedReviewEpisode) return;
    await saveObservationEpisode(markActionReviewUnresolved(selectedReviewEpisode));
    await refreshObservationData(observationProfile);
    setSelectedReviewEpisodeId(null);
    toast.success("판별 불가로 표시하고 기준선 학습에서 제외했어요");
  }

  function switchInterfaceMode() {
    window.scrollTo({ top: 0, behavior: "auto" });
    if (interfaceMode === "user") {
      setInterfaceMode("developer");
      setDemoMode(true);
      setView("today");
      toast.success("개발자 모드로 전환했어요. 테스트 도구만 보여드릴게요.");
      return;
    }
    setInterfaceMode("user");
    setView("home");
    toast.success("사용자 모드로 전환했어요. 실제 사용 화면만 보여드릴게요.");
  }

  async function completeFirstUserSetup() {
    const startedAt = nowMs();
    const startDate = currentDateKey();
    const install: UserInstallState = {
      startedAt,
      startDate,
      careOffset: careLogs.find((log) => log.date === startDate) ?? null,
    };
    const nextProfile = await saveObservationProfile({
      ...observationProfile,
      mode: "learning",
      learningStartedAt: startedAt,
      baselineVersion: observationProfile.baselineVersion + 1,
    });
    window.localStorage.setItem(USER_INSTALL_STORAGE_KEY, JSON.stringify(install));
    clearConsent();
    setConsentState(getConsent());
    setUserInstall(install);
    setObservationProfile(nextProfile);
    await refreshObservationData(nextProfile);
    setOnboardingStep(1);
    setView("home");
    toast.success("초기 설정을 마쳤어요. 새로운 사용자 기준으로 학습을 시작합니다.");
  }

  return (
    <main className={`app-shell interface-${interfaceMode}${view === "onboarding" ? " onboarding-active" : ""}`}>
      <video
        ref={processingVideoRef}
        className="processing-video"
        hidden
        muted
        playsInline
        aria-hidden="true"
      />
      <a className="skip-link" href="#main-content">본문으로 건너뛰기</a>
      <aside className="sidebar">
        <div className="brand" aria-label="메모리 가드">
          <span>
            <strong>Memory Guard</strong>
            <small>메모리 가드</small>
          </span>
        </div>

        <nav ref={mainNavRef} className="main-nav t-tabs" aria-label="주요 메뉴">
          <span ref={mainNavPillRef} className="t-tabs-pill" aria-hidden="true" />
          {activeNavItems.map((item) => (
            <button
              key={item.id}
              className={`t-tab ${view === item.id ? "active" : ""}`}
              onClick={() => {
                if (interfaceMode === "user" && !userInstall && item.id === "settings") {
                  beginOnboarding();
                  return;
                }
                if (item.id === "settings") setSettingsSection(null);
                setView(item.id);
                window.scrollTo({ top: 0, behavior: "auto" });
              }}
              type="button"
              aria-current={view === item.id ? "page" : undefined}
            >
              <NavIcon view={item.id} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-care">
          <p>{interfaceMode === "user" ? "사장님의 하루를" : "개발자 전용 공간"}</p>
          <strong>{interfaceMode === "user" ? "조용히 지켜드릴게요." : "실험 기능을 확인해요."}</strong>
        </div>
        <button
          type="button"
          className="my-data-entry"
          onClick={() => setMyDataOpen(true)}
        >
          내 데이터 관리
        </button>

        <div className="profile">
          <span>
            <strong>김메모리</strong>
            <small>오늘도 좋은 하루입니다.</small>
          </span>
        </div>
      </aside>

      <section id="main-content" className="workspace" tabIndex={-1}>
        <header className="mobile-brand-header">
          <button
            className="mobile-brand"
            type="button"
            onClick={() => {
              setView(interfaceMode === "user" ? "home" : "today");
              window.scrollTo({ top: 0, behavior: "auto" });
            }}
            aria-label={interfaceMode === "user" ? "메모리 가드 홈으로 이동" : "개발자 도구 홈으로 이동"}
          >
            Memory Guard
          </button>
          <div className="mobile-brand-actions">
            {interfaceMode === "user" ? (
              <>
              <button
                type="button"
                className={`camera-pill ${cameraStatus}`}
                onClick={() => cameraStatus === "connected" ? void stopCamera() : requestCameraStart()}
                disabled={cameraStatus === "requesting"}
                aria-label={cameraStatus === "connected" ? "카메라 연결 끄기" : "카메라 연결하기"}
              >
                <span aria-hidden="true" />
                {cameraStatus === "connected" ? "연결됨" : cameraStatus === "requesting" ? <ShimmerText>연결 중</ShimmerText> : "카메라 연결"}
              </button>
              <button className="notification-button" type="button" onClick={() => setView("care")} aria-label="변화 알림 보기">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" /><path d="M10 21h4" /></svg>
                {realHasNotice && <span aria-hidden="true" />}
              </button>
              </>
            ) : (
              <>
                <span className="developer-mode-pill"><i aria-hidden="true" />개발자 도구</span>
                <button className="return-user-button" type="button" onClick={switchInterfaceMode} aria-label="사용자 모드로 돌아가기">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m9 18-6-6 6-6" /><path d="M3 12h18" /></svg>
                  <span>사용자 모드</span>
                </button>
              </>
            )}
          </div>
        </header>
        <header className="topbar">
          <div>
            <span className={`interface-mode-badge ${interfaceMode}`}>
              <span className="interface-mode-badge-dot" aria-hidden="true" />
              {interfaceMode === "user" ? "사용자 모드" : "개발자 모드"}
            </span>
            <h1>
              {view === "home" && (interfaceMode === "user" ? "안녕하세요, 김메모리 사장님" : "오늘의 케어")}
              {view === "today" && "카메라 기능 테스트"}
              {view === "timeline" && "오늘의 기록"}
              {view === "care" && (interfaceMode === "user" ? "이번 주 변화" : "가상 케어 리포트")}
              {view === "settings" && (interfaceMode === "user" ? settingsSectionTitle : "테스트 설정")}
              {view === "onboarding" && `초기 설정 ${onboardingStep}/4`}
            </h1>
          </div>
          <div className="topbar-actions">
            <div className={`camera-pill ${cameraStatus}`}>
              <span aria-hidden="true" />
              {interfaceMode === "user"
                ? cameraStatus === "connected"
                  ? tertiaryCameraConnected ? "웹캠 3대 연결됨" : secondaryCameraConnected ? "웹캠 2대 연결됨" : "웹캠 연결됨"
                  : "카메라 연결 대기"
                : cameraStatus === "requesting"
                  ? <ShimmerText>{statusText}</ShimmerText>
                  : statusText}
            </div>
            <button
              type="button"
              className={`interface-mode-switch ${interfaceMode}`}
              onClick={switchInterfaceMode}
              aria-label={`${interfaceMode === "user" ? "개발자" : "사용자"} 모드로 전환`}
            >
              <span>
                <small>{interfaceMode === "user" ? "테스트 도구가 필요하신가요?" : "실제 화면으로 돌아가기"}</small>
                <strong>{interfaceMode === "user" ? "개발자 모드" : "사용자 모드"}로 전환</strong>
              </span>
            </button>
          </div>
        </header>

        {view === "home" && (
          <div className={`mobile-home-view${pageMotionClass}`}>
            {!userInstall ? (
              <section className="first-user-welcome">
                <span className="welcome-mark" aria-hidden="true">M</span>
                <span className="section-kicker">처음 시작하기</span>
                <h2>사장님의 평소 업무 흐름부터 알아갈게요</h2>
                <p>아직 연결된 기록이 없어요. 업종, 매장 구역과 업무 루틴을 확인한 뒤, 1~2주 동안 학습 모드로 평소 흐름을 익힙니다.</p>
                <div className="first-user-steps" aria-label="초기 설정 순서">
                  <span><i>1</i><strong>업종 입력</strong></span>
                  <span><i>2</i><strong>매장 구역 설정</strong></span>
                  <span><i>3</i><strong>업무 루틴 입력</strong></span>
                  <span><i>4</i><strong>학습 시작</strong></span>
                </div>
                <button type="button" onClick={() => beginOnboarding()}>초기 설정 시작하기</button>
                <small>개발자 모드에서 만든 가상 데이터와 테스트 기록은 사용자 화면에 나타나지 않아요.</small>
              </section>
            ) : (
              <>
                <section className={`home-status-card ${realHasNotice ? "has-notice" : ""}`}>
                  <span className="home-status-icon" aria-hidden="true">{realHasNotice ? "!" : "✓"}</span>
                  <div>
                    <span className="section-kicker">오늘의 상태</span>
                    <h2>
                      {userCareLogs.length === 0
                        ? "첫 기록을 기다리고 있어요."
                        : realHasNotice
                          ? "평소와 다른 흐름이 조금 관찰됐어요."
                          : "오늘은 평소와 비슷한 흐름이에요."}
                    </h2>
                    <p>
                      {realHasNotice
                        ? "한 장면만으로 판단하지 않고, 같은 변화가 반복되는지 차분히 살펴볼게요."
                        : userCareLogs.length === 0
                          ? "마감 체크와 동작 분석 기록이 쌓이면 개인의 평소 흐름과 비교해 드려요."
                          : "필요한 변화가 생기면 이유와 함께 알려드릴게요."}
                    </p>
                  </div>
                </section>

                <button className="home-observation-mode" type="button" onClick={() => openSettings("observation")}>
                  <span className={`home-mode-icon mode-${observationProfile.mode}`} aria-hidden="true">
                    {observationProfile.mode === "learning" ? "↻" : "⌁"}
                  </span>
                  <span>
                    <small>현재 관찰 방식</small>
                    <strong>{observationProfile.mode === "learning" ? "학습 모드" : "분석 모드"}</strong>
                    <em>
                      {observationProfile.mode === "learning"
                        ? "나의 평소 업무 흐름을 익히고 있어요."
                        : "학습한 평소 흐름과 오늘의 동작을 비교해요."}
                    </em>
                  </span>
                  <i>변경하기 ›</i>
                </button>

                {realHasNotice && (
                  <button className="home-notice-card" type="button" onClick={() => setView("care")}>
                    <span className="notice-dot" aria-hidden="true" />
                    <span>
                      <small>확인할 기록</small>
                      <strong>{changeSignal.reasons[0] ?? "평소와 다른 동작 흐름"}</strong>
                      <em>왜 기록됐는지 보기 ›</em>
                    </span>
                  </button>
                )}

                <section className="home-checklist-card">
                  <div className="home-checklist-heading">
                    <div>
                      <span className="section-kicker">오늘의 마감 루틴</span>
                      <h2>필요할 때 직접 확인해 주세요</h2>
                    </div>
                    <span>{closingChecklist.filter((item) => item.done).length}/{closingChecklist.length}</span>
                  </div>
                  <div className="home-checklist-items">
                    {closingChecklist.map((item) => (
                      <label key={item.id} className={item.done ? "done" : ""}>
                        <input
                          type="checkbox"
                          checked={item.done}
                          onChange={() => toggleChecklistItem(item.id)}
                        />
                        <span aria-hidden="true">{item.done ? "✓" : ""}</span>
                        <strong>{item.label}</strong>
                      </label>
                    ))}
                  </div>
                  <button className="checklist-settings-link" type="button" onClick={() => openSettings("checklist")}>항목 수정하기</button>
                </section>

                <section className="home-flow-card">
                  <div className="home-flow-heading">
                    <div>
                      <span className="section-kicker">최근 흐름</span>
                      <h2>지난 일주일을 살펴봤어요</h2>
                    </div>
                    <button type="button" className="text-button" onClick={() => setView("care")}>기록 보기</button>
                  </div>
                  <p className="home-flow-narrative">{realFlowMessage}</p>
                  <div className="mini-flow-chart" aria-label="최근 7일 평소 흐름 일치도">
                    {[...recentCareLogs].reverse().map((day, index) => {
                      const summary = summarizeLog(day);
                      const changeCount = careBaseline
                        ? [
                          summary.doubleChecks >= careBaseline.doubleChecks + 1,
                          summary.dropRate >= careBaseline.dropRate + 0.15,
                          summary.microDelayRate >= careBaseline.microDelayRate + 0.15,
                        ].filter(Boolean).length
                        : 0;
                      const score = careBaseline ? Math.max(52, 94 - changeCount * 12) : 52;
                      const date = new Date(`${day.date}T00:00:00`);
                      const dayLabel = new Intl.DateTimeFormat("ko-KR", { weekday: "short" }).format(date);
                      return (
                        <button
                          type="button"
                          key={day.date}
                          className={index === recentCareLogs.length - 1 ? "selected" : ""}
                          onClick={() => setView("care")}
                          aria-label={`${dayLabel}, ${careBaseline ? `평소 흐름 일치도 ${score}점` : "기준선 학습 중"}`}
                        >
                          <i />
                          <span>{dayLabel.replace("요일", "")}</span>
                        </button>
                      );
                    })}
                    {recentCareLogs.length === 0 && <p className="mini-flow-empty">기록이 쌓이면 여기에 흐름이 표시돼요.</p>}
                  </div>
                </section>

                <section className="home-report-card">
                  <div>
                    <span className="section-kicker">최근 케어 기록</span>
                    <h2>김메모리 사장님의 최근 요약</h2>
                    <p>{changeSignal.reasons[0] ?? "아직 비교할 만큼의 기록이 쌓이지 않았어요."}</p>
                  </div>
                  <button type="button" onClick={() => setView("care")}>자세히 보기 <span aria-hidden="true">›</span></button>
                </section>

                <p className="home-disclaimer">이 결과는 진단이 아닌, 평소 업무 흐름의 변화를 알아차리기 위한 참고 정보예요.</p>
              </>
            )}
          </div>
        )}

        {view === "onboarding" && interfaceMode === "user" && !userInstall && (
          <div className="onboarding-page">
            <div className="onboarding-progress" aria-label={`초기 설정 4단계 중 ${onboardingStep}단계`}>
              <div>
                {[1, 2, 3, 4].map((step) => (
                  <span key={step} className={onboardingStep >= step ? "active" : ""} aria-hidden="true" />
                ))}
              </div>
              <small>{onboardingStep === 1 ? "업종 입력" : onboardingStep === 2 ? "매장 구역 설정" : onboardingStep === 3 ? "업무 루틴 입력" : "학습 시작"}</small>
            </div>

            {onboardingStep === 1 && (
              <section className={`onboarding-step${pageMotionClass}`} aria-labelledby="onboarding-occupation-title">
                <span className="onboarding-step-number">1</span>
                <span className="section-kicker">첫 번째 단계</span>
                <h2 id="onboarding-occupation-title">어떤 매장을 운영하고 계신가요?</h2>
                <p>업종에 맞는 기본 업무 흐름과 추천 구역을 준비할게요.</p>
                <form className="onboarding-form" onSubmit={(event) => void continueOnboardingFromOccupation(event)}>
                  <label htmlFor="onboarding-occupation">
                    <span>업종 이름</span>
                    <input
                      id="onboarding-occupation"
                      value={occupationInput}
                      onChange={(event) => setOccupationInput(event.target.value)}
                      placeholder="예: 디저트 카페, 네일숍, 꽃 공방"
                      autoComplete="organization"
                    />
                  </label>
                  {occupationInput.trim() && (
                    <small className="onboarding-context-hint">
                      {occupationDraftInference.matched
                        ? `${occupationDraftInference.template.label} 업무 맥락으로 시작해요.`
                        : "기본 매장 맥락으로 시작하고, 다음 단계에서 실제 구역에 맞게 보완해요."}
                    </small>
                  )}
                  <button type="submit">다음: 매장 구역 설정</button>
                </form>
              </section>
            )}

            {onboardingStep === 2 && (
              <section className={`onboarding-step${pageMotionClass}`} aria-labelledby="onboarding-zone-title">
                <span className="onboarding-step-number">2</span>
                <span className="section-kicker">두 번째 단계</span>
                <h2 id="onboarding-zone-title">카메라 화면에 매장 구역을 표시해 주세요</h2>
                <p>계산대, 작업대처럼 자주 일하는 위치를 하나 이상 지정하면 동작의 맥락을 더 정확히 이해할 수 있어요.</p>
                <div className="onboarding-zone-summary">
                  <span>현재 업종</span>
                  <strong>{occupationDisplayName}</strong>
                  <small>설정 창에서 카메라별 3×3 화면을 눌러 구역을 지정합니다.</small>
                </div>
                <div className="onboarding-actions">
                  <button className="onboarding-secondary" type="button" onClick={() => setOnboardingStep(1)}>이전</button>
                  <button type="button" onClick={() => setZoneSetupOpen(true)}>매장 구역 설정하기</button>
                  <button className="onboarding-next" type="button" onClick={continueOnboardingFromZones}>다음: 업무 루틴 입력</button>
                </div>
              </section>
            )}

            {onboardingStep === 3 && (
              <section className={`onboarding-step${pageMotionClass}`} aria-labelledby="onboarding-routine-title">
                <span className="onboarding-step-number">3</span>
                <span className="section-kicker">세 번째 단계</span>
                <h2 id="onboarding-routine-title">평소 업무 루틴을 알려주세요</h2>
                <p>주로 일하는 요일과 오픈·마감 시각을 기준으로, 같은 시간대의 업무 흐름을 비교해요.</p>
                <div className="onboarding-routine-card">
                  <label className="onboarding-routine-name" htmlFor="onboarding-routine-name">
                    <span>루틴 이름</span>
                    <input
                      id="onboarding-routine-name"
                      value={onboardingRoutine.name}
                      onChange={(event) => renameRoutine(onboardingRoutine.id, event.target.value)}
                      placeholder="예: 평일 루틴"
                    />
                  </label>
                  <fieldset className="onboarding-routine-days">
                    <legend>이 루틴을 적용할 요일</legend>
                    <div className="weekday-chip-row">
                      {WEEKDAY_DISPLAY_ORDER.map((day) => (
                        <button
                          key={day}
                          type="button"
                          className={`weekday-toggle-chip ${onboardingRoutine.days.includes(day) ? "active" : ""}`}
                          onClick={() => assignDayToRoutine(onboardingRoutine.id, day)}
                          aria-pressed={onboardingRoutine.days.includes(day)}
                        >
                          {WEEKDAY_LABELS[day]}
                        </button>
                      ))}
                    </div>
                  </fieldset>
                  <div className="onboarding-routine-times">
                    <label>
                      <span>출근·오픈 시각</span>
                      <input type="time" value={onboardingRoutine.openTime} onChange={(event) => setRoutineTime(onboardingRoutine.id, "openTime", event.target.value)} />
                    </label>
                    <label>
                      <span>마감 시각</span>
                      <input type="time" value={onboardingRoutine.closeTime} onChange={(event) => setRoutineTime(onboardingRoutine.id, "closeTime", event.target.value)} />
                    </label>
                  </div>
                  <small>세부 반복 업무와 추가 루틴은 설정 탭에서 언제든 수정할 수 있어요.</small>
                </div>
                <div className="onboarding-actions">
                  <button className="onboarding-secondary" type="button" onClick={() => setOnboardingStep(2)}>이전</button>
                  <button className="onboarding-next" type="button" onClick={continueOnboardingFromRoutine}>다음: 학습 시작 확인</button>
                </div>
              </section>
            )}

            {onboardingStep === 4 && (
              <section className={`onboarding-step${pageMotionClass}`} aria-labelledby="onboarding-learning-title">
                <span className="onboarding-step-number">4</span>
                <span className="section-kicker">마지막 단계</span>
                <h2 id="onboarding-learning-title">이제 평소 업무 흐름을 학습할게요</h2>
                <p>처음 1~2주 동안은 특이하거나 확실하지 않은 동작을 제외하고, 사장님의 평소 루틴만 차분히 익힙니다.</p>
                <dl className="onboarding-review">
                  <div><dt>업종</dt><dd>{occupationDisplayName}</dd></div>
                  <div><dt>업무 루틴</dt><dd>{onboardingRoutine.name}</dd></div>
                  <div><dt>운영 시각</dt><dd>{formatClockLabel(onboardingRoutine.openTime)}–{formatClockLabel(onboardingRoutine.closeTime)}</dd></div>
                  <div><dt>시작 모드</dt><dd>학습 모드</dd></div>
                  <div><dt>데이터 기준</dt><dd>새 사용자 기록만 사용</dd></div>
                </dl>
                <div className="onboarding-actions final">
                  <button className="onboarding-secondary" type="button" onClick={() => setOnboardingStep(3)}>이전</button>
                  <button type="button" onClick={() => void completeFirstUserSetup()}>설정 완료하고 학습 시작하기</button>
                </div>
                <small className="onboarding-privacy-note">개발자 모드의 가상 데이터와 테스트 기록은 사용자 기준선에 포함되지 않아요.</small>
              </section>
            )}
          </div>
        )}

        {view === "today" && (
          <div className={`today-view${pageMotionClass}`}>
            <section className="safety-banner">
              <div className="safety-check" aria-hidden="true">
                ✓
              </div>
              <div>
                <strong>테스트 중인 기능이에요</strong>
                <p>웹캠 3대를 고정한 뒤, 세 화면 사이로 이동하며 예시 동작을 따라 해보세요.</p>
              </div>
              <button type="button" onClick={() => setView("settings")}>
                설정으로 <span aria-hidden="true">›</span>
              </button>
            </section>

            <section className="manual-review-queue" aria-labelledby="manual-review-title">
              <div className="manual-review-heading">
                <div>
                  <span>수동 라벨 검토</span>
                  <h2 id="manual-review-title">판별이 애매한 행동만 모았어요</h2>
                  <p>영상·음성 없이 스켈레톤 구간과 분류 근거를 확인한 뒤 업무 이름을 확정합니다.</p>
                </div>
                <strong aria-label={`검토 대기 ${pendingActionReviews.length}건`}>
                  {pendingActionReviews.length}<small>건 대기</small>
                </strong>
              </div>

              {pendingActionReviews.length > 0 ? (
                <div className="manual-review-list">
                  {pendingActionReviews.slice(0, 6).map((episode) => (
                    <article key={episode.id} aria-label={`검토 대기 행동 ${formatSessionTime(episode.recordedAt)}`}>
                      <div className="manual-review-time">
                        <span>{formatSessionTime(episode.recordedAt)}</span>
                        <small>{episode.features.dominantZone
                          ? occupationTemplate.zones.find((zone) => zone.id === episode.features.dominantZone)?.label ?? episode.features.dominantZone
                          : "구역 미확인"}</small>
                      </div>
                      <div className="manual-review-summary">
                        <strong>{actionReviewDisplayLabel(episode)}</strong>
                        <span>{episode.actionReview?.reasons.map((reason) => ACTION_AMBIGUITY_REASON_LABELS[reason]).join(" · ")}</span>
                      </div>
                      <button type="button" onClick={() => openActionReview(episode)}>스켈레톤 확인</button>
                    </article>
                  ))}
                </div>
              ) : (
                <div className="manual-review-empty">
                  <span aria-hidden="true">✓</span>
                  <div><strong>지금은 확인할 행동이 없어요</strong><small>후보가 비슷하거나 좌표가 부족한 행동만 이곳에 나타납니다.</small></div>
                </div>
              )}
            </section>

            <section className="quick-motion-lab" aria-labelledby="quick-motion-title">
              <div className="quick-motion-heading">
                <div>
                  <span>직접 학습 판별</span>
                  <h2 id="quick-motion-title">짧게 가르치고 바로 맞혀보기</h2>
                  <p>같은 동작을 여러 번 저장한 뒤, 새 촬영을 어떤 동작으로 판단하는지 확인합니다.</p>
                </div>
                <div className="quick-motion-actions">
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => openQuickMotionMode("training")}
                  >
                    새 동작 추가
                  </button>
                  <button
                    type="button"
                    onClick={() => openQuickMotionMode("testing")}
                    disabled={learnedMotions.length === 0}
                  >
                    판별 테스트
                  </button>
                </div>
              </div>

              <div className="learned-motion-list" aria-label="학습된 동작">
                {learnedMotions.length > 0 ? learnedMotions.map((motion) => (
                  <span key={motion.id}><strong>{motion.label}</strong>{motion.samples.length}회</span>
                )) : <p>아직 직접 학습한 동작이 없습니다.</p>}
              </div>

              {quickMotionMode !== "idle" && (
                <div className="quick-motion-workspace">
                  {quickMotionMode === "training" && (
                    <>
                      <div className="quick-motion-step">
                        <div>
                          <span>학습 촬영</span>
                          <strong>{draftMotionSamples.length + 1}번째 동작을 보여주세요</strong>
                          <small>매번 같은 동작을 처음부터 끝까지 한 번씩 수행합니다.</small>
                        </div>
                        <div className="sample-count-toggle" role="group" aria-label="반복 횟수">
                          {[5, 10].map((count) => (
                            <button
                              key={count}
                              type="button"
                              className={trainingTargetCount === count ? "active" : ""}
                              onClick={() => setTrainingTargetCount(count as 5 | 10)}
                              disabled={draftMotionSamples.length > 0}
                            >
                              {count}회
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className="quick-motion-progress" aria-label={`${trainingTargetCount}회 중 ${draftMotionSamples.length}회 완료`}>
                        {Array.from({ length: trainingTargetCount }, (_, index) => (
                          <i key={index} className={index < draftMotionSamples.length ? "done" : index === draftMotionSamples.length ? "current" : ""} />
                        ))}
                      </div>
                    </>
                  )}

                  {quickMotionMode === "labeling" && (
                    <form className="quick-motion-label-form" onSubmit={(event) => { event.preventDefault(); saveQuickMotion(); }}>
                      <label htmlFor="custom-motion-label">이 동작의 이름</label>
                      <div>
                        <input
                          id="custom-motion-label"
                          value={customMotionLabel}
                          onChange={(event) => setCustomMotionLabel(event.target.value)}
                          placeholder="예: 서빙, 설거지"
                          autoFocus
                        />
                        <button className="primary-button" type="submit">학습 동작 저장</button>
                      </div>
                      <small>{draftMotionSamples.length}회 촬영한 스켈레톤 구간을 이 이름으로 묶습니다.</small>
                    </form>
                  )}

                  {quickMotionMode === "testing" && (
                    <div className="quick-motion-test-copy">
                      <span>임시 분석 모드</span>
                      <strong>{motionCapture ? "판별할 동작을 수행하고 있어요" : "학습시킨 동작 중 하나를 보여주세요"}</strong>
                      <small>정면이나 후면 어느 쪽에서 수행해도 좌우 반전 좌표를 함께 비교합니다.</small>
                    </div>
                  )}

                  {(quickMotionMode === "training" || quickMotionMode === "testing") && (
                    <div className="quick-motion-controls">
                      <button
                        className="primary-button"
                        type="button"
                        onClick={() => void connectQuickMotionCamera()}
                        disabled={quickMotionBusy}
                      >
                        동작 촬영 시작
                      </button>
                      <span>저장된 1번 웹캠 촬영 창이 별도로 열립니다.</span>
                    </div>
                  )}

                  {learnedMotionResult && (
                    <div className={`learned-motion-result status-${learnedMotionResult.status}`} aria-live="polite">
                      <span>{learnedMotionResult.status === "matched" ? "판별 결과" : learnedMotionResult.status === "uncertain" ? "가장 가까운 후보" : "판별 보류"}</span>
                      <strong>{learnedMotionResult.label ?? "상반신 좌표가 부족합니다"}</strong>
                      <b>신뢰도 <NumberFlow value={Math.round(learnedMotionResult.confidence * 100)} suffix="%" /></b>
                      {learnedMotionResult.candidates.length > 1 && (
                        <small>다음 후보 · {learnedMotionResult.candidates[1].label}</small>
                      )}
                    </div>
                  )}

                  <button
                    className="quick-motion-cancel"
                    type="button"
                    onClick={() => { setQuickMotionMode("idle"); setMotionCapture(null); }}
                  >
                    닫기
                  </button>
                </div>
              )}
            </section>

            <section className={`observation-mode-card mode-${observationProfile.mode}`}>
              <div className="mode-card-heading">
                <div className="occupation-select-wrap">
                  <span className="occupation-icon"><StoreIcon /></span>
                  <form onSubmit={saveOccupation}>
                    <label>
                      <span>업종 입력</span>
                      <input
                        value={occupationInput}
                        onChange={(event) => setOccupationInput(event.target.value)}
                        placeholder="예: 디저트 카페, 네일숍, 꽃 공방"
                        aria-label="업종 입력"
                      />
                    </label>
                    <button type="submit">저장</button>
                    {occupationInput.trim() && (
                      <small>
                        {occupationDraftInference.matched
                          ? `${occupationDraftInference.template.label} 업무 맥락으로 이해해요.`
                          : "가장 가까운 기본 매장 업무 맥락으로 시작하고, 등록한 구역과 루틴으로 보완해요."}
                      </small>
                    )}
                  </form>
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
                  {observationProfile.bodyProportionProfile && (
                    <span className="body-profile-chip">
                      최근 촬영 비율 · {observationProfile.bodyProportionProfile.usableFrames}프레임
                    </span>
                  )}
                </div>
                <div className="learning-progress" aria-label={`기준선 완성도 ${observationBaseline.confidence}%`}>
                  <div><span>개인 기준선 완성도</span><strong><NumberFlow value={observationBaseline.confidence} suffix="%" /></strong></div>
                  <i><span /></i>
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
                    <button type="button" onClick={() => setZoneSetupOpen(true)}>매장 구역 설정</button>
                  </div>
                </div>
                {latestObservationEpisode && (
                  <div className={`latest-learning-result disposition-${latestObservationEpisode.disposition}`}>
                    <span>최근 관찰 · {latestObservationEpisode.taskLabel}</span>
                    <strong>{latestObservationEpisode.dispositionReason}</strong>
                  </div>
                )}
              </div>
            </section>

            <div className="dashboard-grid">
              <section className={`panel camera-panel ${quickMotionMode !== "idle" ? "quick-motion-hidden" : ""}`}>
                <div className="panel-heading">
                  <div>
                    <span className="section-kicker">외장 웹캠 3대 테스트</span>
                    <h2>웹캠 3대 연결 확인</h2>
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

                {availableCameras.length > 0 && (
                  <div className="camera-source-controls" aria-label="카메라 선택">
                    <label>
                      <span>카메라 1</span>
                      <select
                        value={primaryCameraId}
                        onChange={(event) => void changeCameraSource(1, event.target.value)}
                        disabled={cameraStatus === "requesting"}
                      >
                        {availableCameras.map((camera, index) => (
                          <option key={camera.deviceId} value={camera.deviceId}>
                            {cameraDisplayName(camera, index)}
                          </option>
                        ))}
                      </select>
                    </label>
                    {availableCameras.length > 1 && (
                      <label>
                        <span>카메라 2</span>
                        <select
                          value={secondaryCameraId}
                          onChange={(event) => void changeCameraSource(2, event.target.value)}
                          disabled={cameraStatus === "requesting"}
                        >
                          {availableCameras
                            .filter((camera) => camera.deviceId !== primaryCameraId)
                            .map((camera, index) => (
                              <option key={camera.deviceId} value={camera.deviceId}>
                                {cameraDisplayName(camera, index)}
                              </option>
                            ))}
                        </select>
                      </label>
                    )}
                    {availableCameras.length > 2 && (
                      <label>
                        <span>카메라 3</span>
                        <select
                          value={tertiaryCameraId}
                          onChange={(event) => void changeCameraSource(3, event.target.value)}
                          disabled={cameraStatus === "requesting"}
                        >
                          {availableCameras
                            .filter((camera) => camera.deviceId !== primaryCameraId && camera.deviceId !== secondaryCameraId)
                            .map((camera, index) => (
                              <option key={camera.deviceId} value={camera.deviceId}>{cameraDisplayName(camera, index)}</option>
                            ))}
                        </select>
                      </label>
                    )}
                  </div>
                )}

                {cameraStatus !== "connected" && (
                  <div className="camera-connect-action" aria-live="polite">
                    <p>{cameraMessage}</p>
                    <button className="primary-button" type="button" onClick={requestCameraStart} disabled={cameraStatus === "requesting"}>
                      {cameraStatus === "requesting" ? <ShimmerText>연결 중…</ShimmerText> : "카메라 연결"}
                    </button>
                  </div>
                )}

                {cameraStatus === "connected" && (
                  <div className="camera-connect-action camera-connected-action" aria-live="polite">
                    <p>카메라 화면은 팝업에서 크게 확인할 수 있어요.</p>
                    <button className="primary-button" type="button" onClick={() => setMultiCameraModalOpen(true)}>
                      카메라 화면 열기
                    </button>
                  </div>
                )}

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
                  <details className="recorder-details">
                    <summary>
                      <span>기록 정보 자세히 보기</span>
                      <small>{poseStats.frames.toLocaleString()}프레임 · 세션 {sessionCount}개</small>
                    </summary>
                    <div className="coordinate-stats">
                      <span><small>몸 관절</small><strong>22개</strong></span>
                      <span><small>손 관절</small><strong>최대 42개</strong></span>
                      <span><small>기록 속도</small><strong>카메라당 {MOTION_SAMPLE_RATE} FPS</strong></span>
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
                        전체 데이터 폴더 내려받기
                      </button>
                    </div>
                  </details>
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
                  events={events}
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
          <div className={`subpage timeline-page${pageMotionClass}`}>
            <div className="subpage-intro">
              <div>
                <span className="section-kicker">하루의 기억을 한눈에</span>
                <p>좌표 표시가 있는 항목은 당시 스켈레톤을 확인할 수 있습니다.</p>
              </div>
              <span className="count-chip">오늘 {events.length}개 기록</span>
            </div>
            <section className="panel full-timeline-panel">
              <TimelineList events={events} onSelect={openTimelineEvent} />
            </section>
          </div>
        )}

        {view === "settings" && (
          <div className={`settings-page${pageMotionClass}`}>
            {interfaceMode === "user" ? (
              <>
                {settingsSection !== null && (
                  <div className="settings-subpage-head">
                    <button type="button" onClick={() => { setSettingsSection(null); window.scrollTo({ top: 0, behavior: "auto" }); }} aria-label="설정 목록으로 돌아가기">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m15 18-6-6 6-6" /></svg>
                    </button>
                    <div><span className="section-kicker">설정</span><strong>{settingsSectionTitle}</strong></div>
                  </div>
                )}

                {settingsSection === null && !userInstall && (
                  <section className="first-user-settings-head">
                    <span>초기 설정</span>
                    <strong>아래 내용을 확인하면 새로운 학습을 시작할 수 있어요.</strong>
                  </section>
                )}

                {settingsSection === null && (
                  <section className="settings-section-menu" aria-label="사용자 설정 메뉴">
                    <button type="button" onClick={() => setSettingsSection("occupation")}>
                      <span className="settings-menu-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M4 10h16v10H4zM3 10l2-6h14l2 6M8 20v-6h4v6" /></svg></span>
                      <span><strong>업종 변경</strong><small>{occupationDisplayName} · 매장 구역도 함께 관리해요</small></span><i aria-hidden="true">›</i>
                    </button>
                    <button type="button" onClick={() => setSettingsSection("observation")}>
                      <span className="settings-menu-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" /><circle cx="12" cy="12" r="2.5" /></svg></span>
                      <span><strong>관찰 방식</strong><small>{observationProfile.mode === "learning" ? "학습 모드" : "분석 모드"}</small></span><i aria-hidden="true">›</i>
                    </button>
                    <button type="button" onClick={() => setSettingsSection("context")}>
                      <span className="settings-menu-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M6 3v3M18 3v3M4 8h16M5 5h14a1 1 0 0 1 1 1v14H4V6a1 1 0 0 1 1-1Z" /></svg></span>
                      <span><strong>업무 맥락 등록</strong><small>{workContextConfig.routines.length}개 루틴 · 정기 휴일 {workContextConfig.closedDays.length}일</small></span><i aria-hidden="true">›</i>
                    </button>
                    <button type="button" onClick={() => setSettingsSection("checklist")}>
                      <span className="settings-menu-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="m5 7 2 2 4-4M5 13l2 2 4-4M13 8h6M13 14h6M5 20h14" /></svg></span>
                      <span><strong>마감 체크리스트 수정</strong><small>{closingChecklist.length}개 항목</small></span><i aria-hidden="true">›</i>
                    </button>
                  </section>
                )}

                {settingsSection === "occupation" && <section className="settings-intro-card">
                  <span className="section-kicker">나의 업무 환경</span>
                  <h2>{occupationDisplayName}</h2>
                  <p>입력한 업종 이름을 로컬에서 가장 가까운 업무 맥락으로 해석해요.</p>
                  <div className="settings-inline-actions">
                    <button type="button" onClick={() => setZoneSetupOpen(true)}>매장 구역 설정</button>
                    <form onSubmit={saveOccupation}>
                      <label>
                        <span>업종 입력</span>
                        <input
                          value={occupationInput}
                          onChange={(event) => setOccupationInput(event.target.value)}
                          placeholder="예: 디저트 카페, 네일숍, 꽃 공방"
                          aria-label="업종 입력"
                        />
                      </label>
                      <button type="submit">저장</button>
                    </form>
                  </div>
                  {occupationInput.trim() && (
                    <small>
                      {occupationDraftInference.matched
                        ? `${occupationDraftInference.template.label} 업무 맥락으로 연결됩니다.`
                        : "정확히 일치하는 업종이 없어 기본 매장 맥락으로 시작합니다. 매장 구역과 업무 루틴을 입력하면 실제 환경에 맞게 보완됩니다."}
                    </small>
                  )}
                </section>}

                {settingsSection === "observation" && <section className={`settings-observation-mode mode-${userInstall ? observationProfile.mode : "learning"}`}>
                  <div>
                    <span className="section-kicker">관찰 방식</span>
                    <h2>{!userInstall || observationProfile.mode === "learning" ? "평소 업무 흐름부터 학습해요" : "평소 흐름과 오늘의 동작을 비교하고 있어요"}</h2>
                    <p>
                      {!userInstall || observationProfile.mode === "learning"
                        ? "매장 설치 후 1~2주 동안 켜두는 방식이에요. 특이하거나 확정하기 어려운 동작은 평소 기준에 넣지 않아요."
                        : "평소 업무 패턴 학습이 끝난 뒤 사용하는 방식이에요. 달라진 흐름이 반복될 때 케어 기록으로 알려드려요."}
                    </p>
                  </div>
                  <div className="user-mode-toggle" role="group" aria-label="학습 모드 또는 분석 모드 선택">
                    <button
                      type="button"
                      className={!userInstall || observationProfile.mode === "learning" ? "active" : ""}
                      onClick={() => void changeObservationMode("learning")}
                      aria-pressed={!userInstall || observationProfile.mode === "learning"}
                      disabled={!userInstall}
                    >
                      <span aria-hidden="true">↻</span>
                      <strong>학습 모드</strong>
                      <small>평소 루틴 익히기</small>
                    </button>
                    <button
                      type="button"
                      className={userInstall && observationProfile.mode === "analysis" ? "active" : ""}
                      onClick={() => void changeObservationMode("analysis")}
                      aria-pressed={Boolean(userInstall && observationProfile.mode === "analysis")}
                      disabled={!userInstall}
                    >
                      <span aria-hidden="true">⌁</span>
                      <strong>분석 모드</strong>
                      <small>평소와 비교하기</small>
                    </button>
                  </div>
                  <p className="mode-change-note">
                    {userInstall
                      ? "모드를 바꿔도 기존에 학습한 기준선과 케어 기록은 삭제되지 않아요."
                      : "처음 설정을 마치면 개발 테스트 기록과 분리된 새 학습 기준선이 만들어져요."}
                  </p>
                </section>}

                {settingsSection === null && !userInstall && (
                  <button className="complete-user-setup" type="button" onClick={() => void completeFirstUserSetup()}>
                    설정 완료하고 학습 시작하기
                    <span>이전 개발 테스트 기록과 분리된 새 사용자 기준선이 만들어져요.</span>
                  </button>
                )}

                {settingsSection === "context" && <section className="settings-checklist-card work-context-card">
                  <div>
                    <span className="section-kicker">업무 맥락 등록</span>
                    <h2>사장님의 실제 업무 루틴을 알려주세요</h2>
                    <p>업종의 기본 맥락에 더해, 매장에서 실제로 반복되는 요일별 루틴과 정기 휴일을 등록하면 더 정확하게 비교할 수 있어요. 평일처럼 같은 루틴을 쓰는 요일은 하나의 루틴에 묶어서 지정할 수 있어요.</p>
                  </div>

                  <div className="work-context-week-overview">
                    <span className="section-kicker">요일별 배정 현황</span>
                    <div className="weekday-chip-row">
                      {WEEKDAY_DISPLAY_ORDER.map((day) => (
                        <span key={day} className={`weekday-overview-chip ${workContextConfig.closedDays.includes(day) ? "closed" : ""}`}>
                          <strong>{WEEKDAY_LABELS[day]}</strong>
                          <small>{routineNameForDay(workContextConfig, day)}</small>
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="work-context-closed-days">
                    <span className="section-kicker">정기 휴일</span>
                    <div className="weekday-chip-row">
                      {WEEKDAY_DISPLAY_ORDER.map((day) => (
                        <button
                          key={day}
                          type="button"
                          className={`weekday-toggle-chip ${workContextConfig.closedDays.includes(day) ? "active" : ""}`}
                          onClick={() => toggleClosedDay(day)}
                          aria-pressed={workContextConfig.closedDays.includes(day)}
                        >
                          {WEEKDAY_LABELS[day]}
                        </button>
                      ))}
                    </div>
                    <p className="work-context-hint">정기 휴일로 지정하면 그 요일은 어떤 루틴에도 속하지 않아요. 다시 누르면 첫 번째 루틴으로 돌아갑니다.</p>
                  </div>

                  {workContextConfig.routines.map((routine) => (
                    <div className="work-routine-card" key={routine.id}>
                      <div className="work-routine-head">
                        <label className="work-routine-name-field">
                          <span>루틴 이름</span>
                          <input
                            className="work-routine-name"
                            value={routine.name}
                            onChange={(event) => renameRoutine(routine.id, event.target.value)}
                          />
                        </label>
                        {workContextConfig.routines.length > 1 && (
                          <button type="button" className="work-routine-remove" onClick={() => removeRoutine(routine.id)}>
                            루틴 삭제
                          </button>
                        )}
                      </div>

                      <fieldset className="work-routine-weekdays">
                        <legend>이 루틴을 적용할 요일</legend>
                        <div className="weekday-chip-row">
                          {WEEKDAY_DISPLAY_ORDER.map((day) => (
                            <button
                              key={day}
                              type="button"
                              className={`weekday-toggle-chip ${routine.days.includes(day) ? "active" : ""}`}
                              onClick={() => assignDayToRoutine(routine.id, day)}
                              aria-pressed={routine.days.includes(day)}
                            >
                              {WEEKDAY_LABELS[day]}
                            </button>
                          ))}
                        </div>
                      </fieldset>

                      <div className="work-context-times">
                        <label>
                          <span>출근·오픈 시각</span>
                          <input
                            type="time"
                            value={routine.openTime}
                            onChange={(event) => setRoutineTime(routine.id, "openTime", event.target.value)}
                          />
                        </label>
                        <label>
                          <span>마감 시각</span>
                          <input
                            type="time"
                            value={routine.closeTime}
                            onChange={(event) => setRoutineTime(routine.id, "closeTime", event.target.value)}
                          />
                        </label>
                      </div>

                      <ul className="settings-checklist-items">
                        {routine.schedule.map((item) => (
                          <li key={item.id}>
                            <span>{item.time} · {item.label}{item.repeats ? " · 반복" : ""}</span>
                            <button type="button" onClick={() => removeRoutineScheduleItem(routine.id, item.id)} aria-label={`${item.label} 삭제`}>삭제</button>
                          </li>
                        ))}
                      </ul>
                      <form className="checklist-add-form" onSubmit={handleAddRoutineItem(routine.id)}>
                        <label className="schedule-time-field">
                          <span>업무 시간</span>
                          <input type="time" name="time" defaultValue="08:30" />
                        </label>
                        <label className="schedule-label-field">
                          <span>업무 이름</span>
                          <input name="label" placeholder="예: 홀 청소" />
                        </label>
                        <label className="schedule-repeat-toggle">
                          <input type="checkbox" name="repeats" />
                          <span>시간대별 반복 업무</span>
                        </label>
                        <button type="submit">업무 추가</button>
                      </form>

                      <div className="work-context-summary">
                        <span className="section-kicker">{routine.name} 요약</span>
                        <p>{routineSummary(routine)}</p>
                      </div>
                    </div>
                  ))}

                  <button type="button" className="work-routine-add" onClick={addRoutine}>+ 새 루틴 추가</button>
                </section>}

                {settingsSection === "checklist" && <section className="settings-checklist-card">
                  <div>
                    <span className="section-kicker">마감 체크리스트</span>
                    <h2>필요한 항목을 직접 확인해요</h2>
                    <p>자동 감지나 알림 없이, 사장님이 필요할 때 열어보는 체크리스트예요.</p>
                  </div>
                  <ul className="settings-checklist-items">
                    {closingChecklist.map((item) => (
                      <li key={item.id}>
                        <span>{item.label}</span>
                        <button type="button" onClick={() => removeChecklistItem(item.id)} aria-label={`${item.label} 삭제`}>삭제</button>
                      </li>
                    ))}
                  </ul>
                  <form className="checklist-add-form" onSubmit={addChecklistItem}>
                    <input
                      value={newChecklistItem}
                      onChange={(event) => setNewChecklistItem(event.target.value)}
                      placeholder="예: 냉장고 문 확인하기"
                      aria-label="새 체크리스트 항목"
                    />
                    <button type="submit">항목 추가</button>
                  </form>
                </section>}

              </>
            ) : (
              <section className="settings-intro-card developer-settings-intro">
                <span className="section-kicker">개발자 전용 설정</span>
                <h2>카메라 인식을 검증해요</h2>
                <p>이 화면의 기능과 데이터는 실제 사용자가 보는 사용자 모드에는 표시되지 않아요.</p>
                <div className="developer-setting-summary">
                  <span><small>입력 장치</small><strong>외장 웹캠 3대</strong></span>
                </div>
              </section>
            )}

            {(interfaceMode === "developer" || settingsSection === null) && <section className="settings-list" aria-label="설정 목록">
              <button type="button" onClick={() => setMyDataOpen(true)}>
                <span aria-hidden="true">◌</span>
                <span><strong>내 데이터 관리</strong><small>저장된 테스트 기록을 확인하거나 삭제해요</small></span>
                <i aria-hidden="true">›</i>
              </button>
              <button type="button" onClick={switchInterfaceMode}>
                <span className="settings-row-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M8 9 4 12l4 3M16 9l4 3-4 3M14 5l-4 14" />
                  </svg>
                </span>
                <span>
                  <strong>{interfaceMode === "user" ? "개발자 모드" : "사용자 모드로 돌아가기"}</strong>
                  <small>{interfaceMode === "user" ? "카메라·동작 분석 테스트 도구를 열어요" : "실제 사용자 화면으로 전환해요"}</small>
                </span>
                <i aria-hidden="true">›</i>
              </button>
              {interfaceMode === "developer" && (
                <button type="button" onClick={() => setView("today")}>
                  <span aria-hidden="true">⌁</span>
                  <span><strong>웹캠 3대 기능 테스트</strong><small>카메라별 재생과 시공간 인계 분석을 확인해요</small></span>
                  <i aria-hidden="true">›</i>
                </button>
              )}
            </section>}

            {interfaceMode === "developer" && (
              <section className="settings-note">
                <strong>개발자 모드 안내</strong>
                <p>현재는 외장 웹캠 3대를 이용한 동작 테스트와 카메라 간 인계 검증을 지원해요. IoT·POS 연동은 이후 단계에서 추가합니다.</p>
              </section>
            )}

            {(interfaceMode === "developer" || settingsSection === null) && <section className="factory-reset-card">
              <div>
                <strong>처음 화면으로 돌아가기</strong>
                <p>사용자 기록, 학습 기준선, 동작 좌표와 설정을 모두 지우고 처음 설치한 상태로 돌아가요.</p>
              </div>
              <button type="button" onClick={() => void resetToFirstScreen()}>전체 초기화</button>
            </section>}
          </div>
        )}

        {view === "care" && (
          <div className={`subpage care-page${pageMotionClass}`}>
            {interfaceMode === "developer" && <section className="demo-switcher">
              <div>
                <span className="section-kicker">테스트용 케어 기록</span>
                <h2>가상 기준선으로 만든 이번 주 기록</h2>
                <p>실제 사용 기록이 쌓이기 전, 화면과 알림 흐름을 확인하기 위한 예시예요.</p>
              </div>
              <button
                type="button"
                className={demoMode ? "active" : ""}
                onClick={() => setDemoMode((enabled) => !enabled)}
                aria-pressed={demoMode}
              >
                <span aria-hidden="true" /> {demoMode ? "예시 기록 보는 중" : "예시 기록 보기"}
              </button>
            </section>}

            {interfaceMode === "user" && !userInstall ? (
              <section className="care-empty-state first-install-care">
                <CareMark />
                <h2>아직 사용자 기록이 없어요</h2>
                <p>초기 설정을 마치고 학습을 시작하면, 새로 쌓이는 기록만 이곳에 표시됩니다. 개발자 모드의 가상·테스트 기록은 포함되지 않아요.</p>
                <button type="button" onClick={() => beginOnboarding()}>초기 설정 시작하기</button>
              </section>
            ) : interfaceMode === "developer" && demoMode ? (
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
                    <small>가상 기준선과 비교해 평소와 달랐던 흐름만 정리한 결과예요.</small>
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
                      <span className="section-kicker">이번 주 흐름</span>
                      <h2>날짜를 눌러 확인할 기록 보기</h2>
                    </div>
                    <span className="week-range">{activePersona.weekRange}</span>
                  </div>
                  <div className="week-days" role="tablist" aria-label="가상 관찰 날짜">
                    {activePersona.week.map((day, index) => {
                      const intensity = Math.max(
                        day.doubleChecks + day.unfinishedTasks + Math.round(day.microDelayRate / 6) - 1,
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
                      <article><span>반복 확인</span><strong>{demoDay.doubleChecks}<small>회</small></strong></article>
                      <article><span>마무리 전 이탈</span><strong>{demoDay.unfinishedTasks}<small>건</small></strong></article>
                      <article><span>업무 흐름</span><strong className="word-value">{demoDay.microDelayRate >= 12 ? "변화 있음" : "평소와 비슷"}</strong></article>
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
                        <div className="insight-card-heading">
                          <span className="insight-icon" aria-hidden="true">{insight.icon}</span>
                          <div>
                            <span className="section-kicker">{insight.kicker}</span>
                            <h3>{insight.title}</h3>
                          </div>
                        </div>
                        <p>{insight.body}</p>
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
                <CareMark />
                <h2>아직 장기 관찰에 참여하고 있지 않아요</h2>
                <p>
                  카메라를 연결할 때 &ldquo;동의하고 카메라 켜기&rdquo;를 선택하면, 실제
                  사용 기록을 바탕으로 한 케어 리포트가 이곳에 쌓이기
                  시작해요. 개발자용 예시 기록은 개발자 모드에서만 확인할 수 있어요.
                </p>
                <button type="button" onClick={() => handleConsentDecision(true)}>
                  장기 관찰에 참여하기
                </button>
              </section>
            ) : recentCareLogs.length === 0 ? (
              <section className="care-empty-state">
                <CareMark />
                <h2>아직 쌓인 기록이 없어요</h2>
                <p>
                  마감 체크리스트를 사용하고 카메라 동작 테스트를 진행하면,
                  그 기록을 바탕으로 케어 리포트가 만들어져요. 며칠 사용하시면
                  본인의 평소 흐름과 비교한 내용을 볼 수 있어요.
                </p>
              </section>
            ) : (
              <>
                <section className="care-summary">
                  <CareMark />
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
                      <span className="section-kicker">개인 업무 패턴 · 기준선 v{observationProfile.baselineVersion}</span>
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
                    <span><small>기준선 완성도</small><strong><NumberFlow value={observationBaseline.confidence} suffix="%" /></strong></span>
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
                      {MOTION_SAMPLE_RATE}FPS로 기록되기 때문에 실제 손 떨림(4~12Hz)을
                      정밀하게 측정할 수 없고, 임상적으로 검증되지 않았어요. 추세를
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

            {interfaceMode === "user" && cognitiveConcernDetected && (
            <section className="panel care-connect-card">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">케어로 연결</span>
                  <h2>도움이 필요할 때 연결할 수 있어요</h2>
                </div>
              </div>
              <p>
                이 기록만으로 건강 상태를 판단하지 않아요. 피로·수면·스트레스처럼
                다른 원인도 함께 살펴보고, 변화가 반복되어 걱정된다면 다음 단계 중
                편한 것부터 선택해 보세요.
              </p>
              <div className="care-next-steps">
                <span><i>1</i><strong>정보 확인</strong><small>변화를 이해해요</small></span>
                <span><i>2</i><strong>기관 찾기</strong><small>가까운 곳을 찾아요</small></span>
                <span><i>3</i><strong>상담 준비</strong><small>기록을 정리해요</small></span>
              </div>
              <div className="care-connect-actions">
                <button type="button" onClick={() => setBrainHealthOpen(true)}>뇌 건강 정보 보기</button>
                <a className="care-connect-link" href="https://www.nid.or.kr" target="_blank" rel="noreferrer">가까운 치매안심센터 찾기</a>
                <a className="care-connect-link" href="tel:1899-9988">상담전화 1899-9988</a>
              </div>
            </section>
            )}
          </div>
        )}
      </section>

      <Modal
        open={multiCameraModalOpen}
        onClose={() => setMultiCameraModalOpen(false)}
        labelledBy="multi-camera-modal-title"
        className={`multi-camera-modal ${tertiaryCameraConnected ? "has-tertiary-camera" : secondaryCameraConnected ? "has-secondary-camera" : "single-camera"}`}
      >
        <button className="modal-close" type="button" onClick={() => setMultiCameraModalOpen(false)} aria-label="카메라 팝업 닫기">×</button>
        <header className="multi-camera-modal-header">
          <div className="multi-camera-modal-heading">
            <span className="section-kicker">외장 웹캠 3대 · 영상 저장 안 함</span>
            <h2 id="multi-camera-modal-title">카메라 화면 확인</h2>
            <p>{cameraMessage}</p>
          </div>

          {availableCameras.length > 0 && (
            <div className="camera-source-controls multi-camera-source-controls" aria-label="팝업 카메라 선택">
              <label>
                <span>카메라 1</span>
                <select
                  value={primaryCameraId}
                  onChange={(event) => void changeCameraSource(1, event.target.value)}
                  disabled={cameraStatus === "requesting"}
                >
                  {availableCameras.map((camera, index) => (
                    <option key={camera.deviceId} value={camera.deviceId}>{cameraDisplayName(camera, index)}</option>
                  ))}
                </select>
              </label>
              {availableCameras.length > 1 && (
                <label>
                  <span>카메라 2</span>
                  <select
                    value={secondaryCameraId}
                    onChange={(event) => void changeCameraSource(2, event.target.value)}
                    disabled={cameraStatus === "requesting"}
                  >
                    {availableCameras
                      .filter((camera) => camera.deviceId !== primaryCameraId)
                      .map((camera, index) => (
                        <option key={camera.deviceId} value={camera.deviceId}>{cameraDisplayName(camera, index)}</option>
                      ))}
                  </select>
                </label>
              )}
              {availableCameras.length > 2 && (
                <label>
                  <span>카메라 3</span>
                  <select
                    value={tertiaryCameraId}
                    onChange={(event) => void changeCameraSource(3, event.target.value)}
                    disabled={cameraStatus === "requesting"}
                  >
                    {availableCameras
                      .filter((camera) => camera.deviceId !== primaryCameraId && camera.deviceId !== secondaryCameraId)
                      .map((camera, index) => (
                        <option key={camera.deviceId} value={camera.deviceId}>{cameraDisplayName(camera, index)}</option>
                      ))}
                  </select>
                </label>
              )}
            </div>
          )}
        </header>

        {cameraStatus === "connected" ? (
          <div ref={cameraFeedLayoutRef} className="camera-feed-layout multi-camera-feed-layout">
            <div ref={cameraFrameRef} className={`camera-frame ${cameraStatus}`}>
              <div className="camera-main-view">
                <video ref={videoRef} muted playsInline aria-label="카메라 1 영상" />
                <canvas ref={overlayCanvasRef} className="pose-overlay" aria-label="실시간 전신 스켈레톤" />
              </div>
              <div className="tracking-readout">
                <span className={targetLocked ? "target-locked" : ""}>대상 · {targetLocked ? "고정됨" : "찾는 중"}</span>
                <span>머리 · {headDirectionLabel}</span>
                <span
                  className={detectedHands > 0 ? "hands-found hand-status" : "hand-status"}
                  aria-label={`손가락 관절 인식: 왼손 ${detectedHandSides.left ? "인식됨" : "미인식"}, 오른손 ${detectedHandSides.right ? "인식됨" : "미인식"}`}
                >
                  손 · <b className={detectedHandSides.left ? "detected" : "missing"}>왼손 {detectedHandSides.left ? "✓" : "—"}</b>
                  <b className={detectedHandSides.right ? "detected" : "missing"}>오른손 {detectedHandSides.right ? "✓" : "—"}</b>
                </span>
              </div>
              <button
                className="camera-fullscreen-button"
                type="button"
                onClick={() => void toggleCameraFullscreen()}
                aria-label={cameraFullscreen ? "카메라 전체화면 닫기" : "카메라 전체화면으로 보기"}
              >
                <span aria-hidden="true">{cameraFullscreen ? "↙" : "↗"}</span>
                {cameraFullscreen ? "전체화면 닫기" : "전체화면"}
              </button>
              <div className="camera-caption">
                <span className={`pose-state ${poseStatus}`}>
                  <i aria-hidden="true" /> {poseStatus === "loading" ? <ShimmerText>{poseStatusLabel}</ShimmerText> : poseStatusLabel}
                </span>
              </div>
            </div>

            {secondaryCameraConnected && (
              <section className="secondary-camera-frame" aria-label="카메라 2 화면">
                <div className="secondary-camera-heading">
                  <span>카메라 2</span>
                  <small>공통 대상 · 좌표 기록</small>
                </div>
                <div className="camera-secondary-view">
                  <video ref={secondaryVideoRef} muted playsInline aria-label="카메라 2 영상" />
                  <canvas ref={secondaryOverlayCanvasRef} className="pose-overlay" aria-label="카메라 2 전신 스켈레톤" />
                  <span className={`secondary-pose-state ${secondaryPoseStatus}`}>좌표 · {secondaryPoseStatus === "full" ? "전신 인식" : secondaryPoseStatus === "partial" ? "일부 인식" : secondaryPoseStatus === "error" ? "추적 오류" : "찾는 중"}</span>
                </div>
              </section>
            )}
            {tertiaryCameraConnected && (
              <section className="secondary-camera-frame" aria-label="카메라 3 화면">
                <div className="secondary-camera-heading">
                  <span>카메라 3</span>
                  <small>공통 대상 · 좌표 기록</small>
                </div>
                <div className="camera-secondary-view">
                  <video ref={tertiaryVideoRef} muted playsInline aria-label="카메라 3 영상" />
                  <canvas ref={tertiaryOverlayCanvasRef} className="pose-overlay" aria-label="카메라 3 전신 스켈레톤" />
                  <span className={`secondary-pose-state ${tertiaryPoseStatus}`}>좌표 · {tertiaryPoseStatus === "full" ? "전신 인식" : tertiaryPoseStatus === "partial" ? "일부 인식" : tertiaryPoseStatus === "error" ? "추적 오류" : "찾는 중"}</span>
                </div>
              </section>
            )}
          </div>
        ) : (
          <div className={`multi-camera-modal-waiting status-${cameraStatus}`} aria-live="polite">
            <span aria-hidden="true" />
            <strong>
              {cameraStatus === "requesting"
                ? <ShimmerText>카메라를 연결하고 있어요</ShimmerText>
                : cameraStatus === "error"
                  ? "카메라를 연결하지 못했어요"
                  : "카메라 연결을 기다리고 있어요"}
            </strong>
            <p>{cameraMessage}</p>
            {cameraStatus !== "requesting" && (
              <button className="primary-button" type="button" onClick={requestCameraStart}>다시 연결</button>
            )}
          </div>
        )}

        <div className="multi-camera-modal-actions">
          {cameraStatus === "connected" && (
            <button className="camera-stop-button" type="button" onClick={() => void stopCamera().then(() => setMultiCameraModalOpen(false))}>카메라 끄기</button>
          )}
          <button className="modal-confirm-button" type="button" onClick={() => setMultiCameraModalOpen(false)}>확인 완료</button>
        </div>
      </Modal>

      <Modal
        open={quickMotionCameraOpen}
        onClose={() => void closeQuickMotionCamera()}
        labelledBy="quick-motion-camera-title"
        className="quick-motion-camera-modal"
      >
        <button className="modal-close" type="button" onClick={() => void closeQuickMotionCamera()} aria-label="촬영 창 닫기">×</button>
        <div className="quick-motion-camera-heading">
          <span className="section-kicker">1번 웹캠 · 영상 저장 안 함</span>
          <h2 id="quick-motion-camera-title">
            {quickMotionMode === "training"
              ? `${draftMotionSamples.length + 1}번째 동작을 보여주세요`
              : "판별할 동작을 보여주세요"}
          </h2>
          <p>얼굴·어깨·양팔이 화면에 들어오도록 선 뒤, 시작 버튼을 누르세요.</p>
        </div>

        <div className={`quick-motion-camera-preview status-${cameraStatus}`}>
          <video ref={quickMotionVideoRef} muted playsInline aria-label="1번 웹캠 동작 촬영 영상" />
          {cameraStatus !== "connected" && (
            <div className="quick-motion-camera-waiting">
              <strong>
                {cameraStatus === "error"
                  ? "1번 웹캠을 연결하지 못했어요"
                  : cameraStatus === "requesting"
                    ? <ShimmerText>1번 웹캠을 준비하고 있어요</ShimmerText>
                    : "1번 웹캠을 준비하고 있어요"}
              </strong>
              <span>{cameraMessage}</span>
            </div>
          )}
          {cameraStatus === "connected" && (
            <div className="quick-motion-camera-status">
              <span className={motionCapture ? "recording" : ""}>{motionCapture ? "● 좌표 기록 중" : "촬영 준비됨"}</span>
              <span>{targetLocked ? "상반신 인식됨" : "분석 대상 찾는 중"}</span>
            </div>
          )}
        </div>

        {quickMotionMode === "training" && (
          <div className="quick-motion-modal-progress">
            <span>학습 진행</span>
            <strong>{draftMotionSamples.length} / {trainingTargetCount}</strong>
          </div>
        )}

        {learnedMotionResult && quickMotionMode === "testing" && (
          <div className={`learned-motion-result status-${learnedMotionResult.status}`} aria-live="polite">
            <span>{learnedMotionResult.status === "matched" ? "판별 결과" : learnedMotionResult.status === "uncertain" ? "가장 가까운 후보" : "판별 보류"}</span>
            <strong>{learnedMotionResult.label ?? "상반신 좌표가 부족합니다"}</strong>
            <b>신뢰도 <NumberFlow value={Math.round(learnedMotionResult.confidence * 100)} suffix="%" /></b>
            {learnedMotionResult.candidates.length > 1 && <small>다음 후보 · {learnedMotionResult.candidates[1].label}</small>}
          </div>
        )}

        <div className="quick-motion-camera-actions">
          <button
            className={motionCapture ? "capture-stop-button" : "primary-button"}
            type="button"
            onClick={() => motionCapture ? void finishQuickMotionCapture() : startQuickMotionCapture()}
            disabled={cameraStatus !== "connected" || quickMotionBusy}
          >
            {quickMotionBusy ? "좌표 확인 중…" : motionCapture ? "이번 동작 완료" : "녹화 시작"}
          </button>
          <button type="button" onClick={() => void closeQuickMotionCamera()}>촬영 창 닫기</button>
        </div>
      </Modal>

      <Modal open={selectedEvent !== null} onClose={() => setSelectedEvent(null)} labelledBy="video-title" className="video-modal">
        {selectedEvent && (
          <>
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
          </>
        )}
      </Modal>

      <Modal open={brainHealthOpen && cognitiveConcernDetected} onClose={() => setBrainHealthOpen(false)} labelledBy="brain-health-title" className="brain-health-modal">
        <button className="modal-close" type="button" onClick={() => setBrainHealthOpen(false)} aria-label="닫기">×</button>
        <span className="section-kicker">뇌 건강 정보</span>
        <h2 id="brain-health-title">한 번의 실수보다 반복되는 변화가 중요해요</h2>
        <p>익숙한 업무가 평소보다 오래 걸리거나 확인 행동이 반복되는 데에는 피로, 수면 부족, 스트레스, 신체 컨디션 등 여러 이유가 있을 수 있어요. 메모리 가드는 원인을 진단하지 않고, 본인의 평소 흐름과 달라진 장면을 정리해 드립니다.</p>
        <div className="brain-health-guide">
          <article><strong>먼저 돌아보기</strong><span>최근 수면, 피로, 매장 혼잡도와 함께 확인해 보세요.</span></article>
          <article><strong>며칠 더 살펴보기</strong><span>같은 변화가 여러 날 이어지는지 기록을 확인해 보세요.</span></article>
          <article><strong>걱정되면 상담하기</strong><span>상담할 때 언제부터 어떤 업무가 달라졌는지 이 기록을 보여주세요.</span></article>
        </div>
        <div className="brain-health-actions">
          <a href="https://www.nid.or.kr" target="_blank" rel="noreferrer">공식 정보와 센터 찾기</a>
          <a href="tel:1899-9988">치매상담콜센터 연결</a>
        </div>
        <small>이 정보와 앱의 기록은 의료 진단을 대신하지 않습니다.</small>
      </Modal>

      <Modal open={bookingOpen} onClose={() => setBookingOpen(false)} labelledBy="booking-title" className="booking-modal">
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
      </Modal>

      <Modal open={showConsentModal} onClose={() => { }} disableDismiss labelledBy="consent-title" className="consent-modal">
        <span className="section-kicker">카메라를 켜기 전에 알려드려요</span>
        <h2 id="consent-title">이 카메라는 두 가지 목적으로 쓰일 수 있어요</h2>
        <ul className="consent-list">
          <li>
            <strong>① 기억 복원</strong> — 업무 중 특정 순간의 몸·손
            좌표를 짧게 저장해, 나중에 &ldquo;그때 무슨 일이 있었는지&rdquo;를
            스켈레톤으로 다시 확인할 수 있게 해요.
          </li>
          <li>
            <strong>② 장기 업무 패턴 관찰(선택)</strong> — 동의하시면,
            반복 확인·업무 지연 같은 행동 패턴을
            오랜 기간 관찰해 케어 리포트를 만드는 데도 사용해요. 이
            데이터는 진단이 아니라 변화를 알아차리는 참고용이며, 언제든
            내 데이터 관리에서 철회하고 전부 삭제할 수 있어요.
          </li>
        </ul>
        <p className="consent-note">
          얼굴·영상·음성은 저장하지 않고, 좌표 데이터는 이 브라우저에만
          남아요. 동의하지 않아도 기록과 수동 체크리스트는 그대로 사용할 수 있어요.
        </p>
        <div className="consent-actions">
          <button
            type="button"
            className="secondary-button"
            onClick={() => handleConsentDecision(false)}
          >
            장기 관찰 없이 사용할게요
          </button>
          <button
            type="button"
            className="primary-button"
            onClick={() => handleConsentDecision(true)}
          >
            동의하고 카메라 켜기
          </button>
        </div>
      </Modal>

      <Modal open={myDataOpen} onClose={() => setMyDataOpen(false)} labelledBy="my-data-title" className="my-data-modal">
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
            <span>저장된 동작 좌표</span>
            <strong>{formatBytes(storedMotionBytes)} 사용 중</strong>
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
          <button
            type="button"
            className="secondary-button"
            onClick={() => void exportPoseData()}
            disabled={sessionCount === 0 && poseStats.frames === 0}
          >
            전체 데이터 폴더 내려받기
          </button>
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
      </Modal>

      <Modal open={zoneSetupOpen} onClose={() => setZoneSetupOpen(false)} labelledBy="zone-setup-title" className="zone-setup-modal">
        <button className="modal-close" type="button" onClick={() => setZoneSetupOpen(false)} aria-label="닫기">×</button>
        <span className="section-kicker">{occupationDisplayName} 관찰 맥락</span>
        <h2 id="zone-setup-title">카메라 화면에 매장 구역을 표시해 주세요</h2>
        <p>카메라를 고르고, 실제 화면에서 구역에 가까운 칸을 눌러주세요. 카메라 위치가 바뀌면 다시 설정해야 해요.</p>
        <div className="zone-camera-tabs" role="tablist" aria-label="편집할 카메라">
          {([1, 2, 3] as CameraSlot[]).map((slot) => (
            <button
              key={slot}
              type="button"
              role="tab"
              aria-selected={zoneCameraSlot === slot}
              className={zoneCameraSlot === slot ? "active" : ""}
              onClick={() => setZoneCameraSlot(slot)}
            >
              <strong>카메라 {slot}</strong>
              <span>{availableCameras.find((camera) => camera.deviceId === [primaryCameraId, secondaryCameraId, tertiaryCameraId][slot - 1])?.label || `카메라 ${slot}`}</span>
            </button>
          ))}
        </div>
        <label className="zone-picker">
          <span>지정할 구역</span>
          <select value={selectedZoneId} onChange={(event) => setSelectedZoneId(event.target.value)}>
            <optgroup label="추천 구역">
              {occupationTemplate.zones.map((zone) => <option key={zone.id} value={zone.id}>{zone.label}</option>)}
            </optgroup>
            {observationProfile.customZones.length > 0 && (
              <optgroup label="직접 추가한 구역">
                {observationProfile.customZones.map((zone) => <option key={zone.id} value={zone.id}>{zone.label}</option>)}
              </optgroup>
            )}
          </select>
        </label>
        <div className="custom-zone-box">
          <label htmlFor="custom-zone-name">목록에 없는 구역인가요?</label>
          <div className="custom-zone-entry">
            <input
              id="custom-zone-name"
              value={customZoneName}
              maxLength={24}
              placeholder="예: 테라스, 포장대, 직원 휴게실"
              onChange={(event) => setCustomZoneName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  void addCustomZone();
                }
              }}
            />
            <button type="button" disabled={!customZoneName.trim()} onClick={() => void addCustomZone()}>구역 추가</button>
          </div>
        </div>
        {(draftZoneContext || selectedCustomZone) && (
          <p className="zone-context-hint">
            {draftZoneContext || selectedZoneContext
              ? `‘${(draftZoneContext ?? selectedZoneContext)!.label}’ 맥락으로 이해해요.`
              : "새 구역으로 기록하고 행동·시간대와 함께 학습해요."}
          </p>
        )}
        <div key={zoneCameraSlot} className="zone-camera-grid" aria-label={`카메라 ${zoneCameraSlot} 화면 3×3 구역 설정`}>
          {activeZoneGrid.map((zoneId, index) => {
            const zone = zoneOptions.find((item) => item.id === zoneId);
            return (
              <button key={index} type="button" className={zoneId ? "mapped" : ""} onClick={() => void assignZoneCell(index)}>
                <small>{index + 1}</small>
                <strong>{zone?.label ?? "구역 지정"}</strong>
              </button>
            );
          })}
        </div>
        <div className="zone-setup-footer">
          <span>카메라 {zoneCameraSlot}에 {mappedZoneCount}개 구역 설정됨 · 화면상 위치를 사용해요.</span>
          <button type="button" onClick={() => setZoneSetupOpen(false)}>설정 완료</button>
        </div>
      </Modal>

      {replaySessionId && (
        <SessionReplayPanel
          key={replaySessionId}
          source={{
            kind: "recorded",
            sessionId: replaySessionId,
            globalSessionId: recentSessions.find((session) => session.id === replaySessionId)?.globalSessionId,
          }}
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
          showMultiCameraDiagnostics={interfaceMode === "developer"}
          onClose={() => setReplaySessionId(null)}
        />
      )}

      {selectedReviewEpisode && (
        <SessionReplayPanel
          key={selectedReviewEpisode.id}
          source={{
            kind: "recorded",
            sessionId: selectedReviewEpisode.sessionId,
            globalSessionId: selectedReviewEpisode.globalSessionId,
            window: selectedReviewEpisode.motionSlice
              ? {
                startMs: selectedReviewEpisode.motionSlice.startMs,
                endMs: selectedReviewEpisode.motionSlice.endMs,
              }
              : undefined,
          }}
          sessionLabel={`${formatSessionTime(selectedReviewEpisode.recordedAt)} · 업무 라벨 검토`}
          observationMode={selectedReviewEpisode.mode}
          baselineVersion={selectedReviewEpisode.baselineVersion}
          hideAnalysisFeedback
          reviewPanel={(
            <section className="manual-review-panel" aria-labelledby="manual-review-panel-title">
              <div className="manual-review-evidence">
                <span>자동 판별 근거</span>
                <strong id="manual-review-panel-title">
                  {actionReviewDisplayLabel(selectedReviewEpisode)}
                </strong>
                <p>
                  {selectedReviewEpisode.actionReview?.reasons
                    .map((reason) => ACTION_AMBIGUITY_REASON_LABELS[reason])
                    .join(" · ")}
                </p>
                <small>
                  {selectedReviewEpisode.features.dominantZone
                    ? `감지 구역 · ${occupationTemplate.zones.find((zone) => zone.id === selectedReviewEpisode.features.dominantZone)?.label ?? selectedReviewEpisode.features.dominantZone}`
                    : "감지 구역을 확인하지 못했어요"}
                </small>
              </div>

              {reviewLabelOptions.length > 0 ? (
                <div className="manual-review-choice" role="radiogroup" aria-label="업무 라벨 후보">
                  <span>스켈레톤을 보고 실제 업무를 선택하세요</span>
                  <div>
                    {reviewLabelOptions.map((option) => (
                      <label key={option.taskType} className={selectedReviewTaskType === option.taskType ? "selected" : ""}>
                        <input
                          type="radio"
                          name="manual-task-label"
                          value={option.taskType}
                          checked={selectedReviewTaskType === option.taskType}
                          onChange={() => setSelectedReviewTaskType(option.taskType)}
                        />
                        <span>{option.taskLabel}</span>
                      </label>
                    ))}
                  </div>
                  <div className="manual-review-actions">
                    <button className="primary-button" type="button" onClick={() => void confirmActionReview()} disabled={!selectedReviewTaskType}>라벨 확정</button>
                    <button type="button" onClick={() => void leaveActionReviewUnresolved()}>판별 불가</button>
                  </div>
                </div>
              ) : (
                <div className="manual-review-no-labels">
                  <strong>먼저 업무 맥락에 업무 이름을 등록해 주세요</strong>
                  <button type="button" onClick={() => { setSelectedReviewEpisodeId(null); openSettings("context"); }}>업무 맥락 등록으로 이동</button>
                </div>
              )}
            </section>
          )}
          onClose={() => setSelectedReviewEpisodeId(null)}
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

      <Toaster
        position="bottom-center"
        icons={{ success: <span aria-hidden="true">✓</span> }}
        toastOptions={{
          unstyled: true,
          classNames: { toast: "toast", icon: "toast-icon" },
        }}
      />
    </main>
  );
}
