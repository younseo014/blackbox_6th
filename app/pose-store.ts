import { mergeCameraFrameStreams, type CameraSlot } from "./multi-camera";
import {
  createDataFolderZip,
  safeDataPathSegment,
  type DataFolderEntry,
} from "./data-folder-export";

export const BODY_LANDMARK_COUNT = 22;
export const HAND_LANDMARK_COUNT = 21;
export const MOTION_SAMPLE_RATE = 10;

// time, body/full-body flags, head yaw/pitch/roll,
// left/right hand flags and confidence, 22 body points × 4, 42 hand points × 3.
export const MOTION_FRAME_STRIDE =
  10 + BODY_LANDMARK_COUNT * 4 + HAND_LANDMARK_COUNT * 2 * 3;

export type TimeZoneMetadata = {
  timeZone: string;
  utcOffsetMinutes: number;
};

export function getTimeZoneMetadata(at: number): TimeZoneMetadata {
  let timeZone = "UTC";
  try {
    timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    // UTC is a safe fallback on runtimes without full Intl time-zone data.
  }
  return {
    timeZone,
    utcOffsetMinutes: -new Date(at).getTimezoneOffset(),
  };
}

export const BODY_LANDMARK_NAMES = [
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_pinky_anchor",
  "right_pinky_anchor",
  "left_index_anchor",
  "right_index_anchor",
  "left_thumb_anchor",
  "right_thumb_anchor",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
  "left_heel",
  "right_heel",
  "left_foot_index",
  "right_foot_index",
] as const;

export const HAND_LANDMARK_NAMES = [
  "wrist",
  "thumb_cmc",
  "thumb_mcp",
  "thumb_ip",
  "thumb_tip",
  "index_mcp",
  "index_pip",
  "index_dip",
  "index_tip",
  "middle_mcp",
  "middle_pip",
  "middle_dip",
  "middle_tip",
  "ring_mcp",
  "ring_pip",
  "ring_dip",
  "ring_tip",
  "pinky_mcp",
  "pinky_pip",
  "pinky_dip",
  "pinky_tip",
] as const;

export type MotionSessionRecord = {
  id: string;
  startedAt: number;
  endedAt: number | null;
  frameCount: number;
  detectedFrameCount: number;
  fullBodyFrameCount: number;
  handDetectedFrameCount: number;
  storageBytes: number;
  sampleRate: number;
  bodyLandmarkCount: number;
  handLandmarkCount: number;
  coordinateSpace: "normalized_image";
  mirroredPreview: boolean;
  source: "local_camera";
  faceLandmarksStored: false;
  globalSessionId?: string;
  cameraId?: string;
  cameraSlot?: CameraSlot;
  cameraLabel?: string;
  logicalCameraId?: string;
  timelineOriginMs?: number;
  /** IANA zone and local UTC offset captured with this session. Optional for legacy records. */
  timeZone?: string;
  utcOffsetMinutes?: number;
};

export type CaptureManifestRecord = {
  id: string;
  startedAt: number;
  endedAt: number | null;
  timelineOriginMs: number;
  timeZone?: string;
  utcOffsetMinutes?: number;
  expectedCameraSlots: CameraSlot[];
  cameras: Array<{
    slot: CameraSlot;
    logicalCameraId: string;
    deviceId: string;
    label: string;
    sessionId: string | null;
    connectedAtStart: boolean;
  }>;
  zoneSnapshot: {
    primary: Array<string | null>;
    secondary: Array<string | null>;
    tertiary: Array<string | null>;
    customZones: Array<{ id: string; label: string; contextZoneId: string | null }>;
    profileUpdatedAt: number;
  };
  healthSampleRateHz: 1;
};

export type CameraHealthState = {
  connected: boolean;
  receivingFrames: boolean;
  personDetected: boolean;
  trackingError: boolean;
};

export type CameraHealthChunkRecord = {
  id: string;
  globalSessionId: string;
  cameraSlot: CameraSlot;
  startSecond: number;
  sampleCount: number;
  data: ArrayBuffer;
};

export function encodeCameraHealth(state: CameraHealthState) {
  return Number(state.connected) |
    (Number(state.receivingFrames) << 1) |
    (Number(state.personDetected) << 2) |
    (Number(state.trackingError) << 3);
}

export function decodeCameraHealth(value: number): CameraHealthState {
  return {
    connected: Boolean(value & 1),
    receivingFrames: Boolean(value & 2),
    personDetected: Boolean(value & 4),
    trackingError: Boolean(value & 8),
  };
}

export type MotionChunkRecord = {
  id: string;
  sessionId: string;
  startFrame: number;
  frameCount: number;
  createdAt: number;
  data: ArrayBuffer;
};

const DB_NAME = "memory-guard-motion-v2";
const DB_VERSION = 2;
const SESSION_STORE = "motion_sessions";
const CHUNK_STORE = "motion_chunks";
const CAPTURE_MANIFEST_STORE = "capture_manifests";
const CAMERA_HEALTH_STORE = "camera_health_chunks";

function requestResult<T>(request: IDBRequest<T>) {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function transactionDone(transaction: IDBTransaction) {
  return new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });
}

export function openMotionDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(SESSION_STORE)) {
        database.createObjectStore(SESSION_STORE, { keyPath: "id" });
      }
      if (!database.objectStoreNames.contains(CHUNK_STORE)) {
        const chunkStore = database.createObjectStore(CHUNK_STORE, {
          keyPath: "id",
        });
        chunkStore.createIndex("sessionId", "sessionId", { unique: false });
      }
      if (!database.objectStoreNames.contains(CAPTURE_MANIFEST_STORE)) {
        database.createObjectStore(CAPTURE_MANIFEST_STORE, { keyPath: "id" });
      }
      if (!database.objectStoreNames.contains(CAMERA_HEALTH_STORE)) {
        const healthStore = database.createObjectStore(CAMERA_HEALTH_STORE, { keyPath: "id" });
        healthStore.createIndex("globalSessionId", "globalSessionId", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function createMotionSession(
  id: string,
  startedAt: number,
  multiCamera?: Pick<MotionSessionRecord, "globalSessionId" | "cameraId" | "cameraSlot" | "cameraLabel" | "logicalCameraId" | "timelineOriginMs">,
) {
  const timeZone = getTimeZoneMetadata(startedAt);
  const database = await openMotionDatabase();
  const transaction = database.transaction(SESSION_STORE, "readwrite");
  const record: MotionSessionRecord = {
    id,
    startedAt,
    endedAt: null,
    frameCount: 0,
    detectedFrameCount: 0,
    fullBodyFrameCount: 0,
    handDetectedFrameCount: 0,
    storageBytes: 0,
    sampleRate: MOTION_SAMPLE_RATE,
    bodyLandmarkCount: BODY_LANDMARK_COUNT,
    handLandmarkCount: HAND_LANDMARK_COUNT,
    coordinateSpace: "normalized_image",
    mirroredPreview: true,
    source: "local_camera",
    faceLandmarksStored: false,
    ...timeZone,
    ...multiCamera,
  };
  transaction.objectStore(SESSION_STORE).put(record);
  await transactionDone(transaction);
  database.close();
  return record;
}

export async function saveCaptureManifest(manifest: CaptureManifestRecord) {
  const fallbackTimeZone = getTimeZoneMetadata(manifest.startedAt);
  const record: CaptureManifestRecord = {
    ...manifest,
    timeZone: manifest.timeZone ?? fallbackTimeZone.timeZone,
    utcOffsetMinutes: manifest.utcOffsetMinutes ?? fallbackTimeZone.utcOffsetMinutes,
  };
  const database = await openMotionDatabase();
  const transaction = database.transaction(CAPTURE_MANIFEST_STORE, "readwrite");
  transaction.objectStore(CAPTURE_MANIFEST_STORE).put(record);
  await transactionDone(transaction);
  database.close();
  return record;
}

export async function finishCaptureManifest(id: string, endedAt: number) {
  const database = await openMotionDatabase();
  const transaction = database.transaction(CAPTURE_MANIFEST_STORE, "readwrite");
  const store = transaction.objectStore(CAPTURE_MANIFEST_STORE);
  const manifest = await requestResult<CaptureManifestRecord | undefined>(store.get(id));
  if (manifest) store.put({ ...manifest, endedAt });
  await transactionDone(transaction);
  database.close();
}

export async function getCaptureManifest(id: string) {
  const database = await openMotionDatabase();
  const transaction = database.transaction(CAPTURE_MANIFEST_STORE, "readonly");
  const manifest = await requestResult<CaptureManifestRecord | undefined>(
    transaction.objectStore(CAPTURE_MANIFEST_STORE).get(id),
  );
  await transactionDone(transaction);
  database.close();
  return manifest ?? null;
}

export async function appendCameraHealthChunk(
  globalSessionId: string,
  cameraSlot: CameraSlot,
  startSecond: number,
  samples: Uint8Array,
) {
  if (samples.length === 0) return;
  const database = await openMotionDatabase();
  const transaction = database.transaction(CAMERA_HEALTH_STORE, "readwrite");
  const data = samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength);
  const record: CameraHealthChunkRecord = {
    id: `${globalSessionId}:${cameraSlot}:${startSecond}`,
    globalSessionId,
    cameraSlot,
    startSecond,
    sampleCount: samples.length,
    data,
  };
  transaction.objectStore(CAMERA_HEALTH_STORE).put(record);
  await transactionDone(transaction);
  database.close();
}

async function getCameraHealthChunks(globalSessionId: string) {
  const database = await openMotionDatabase();
  const transaction = database.transaction(CAMERA_HEALTH_STORE, "readonly");
  const chunks = await requestResult<CameraHealthChunkRecord[]>(
    transaction.objectStore(CAMERA_HEALTH_STORE).index("globalSessionId").getAll(globalSessionId),
  );
  await transactionDone(transaction);
  database.close();
  return chunks.sort((a, b) => a.cameraSlot - b.cameraSlot || a.startSecond - b.startSecond);
}

export async function getCameraHealthTimeline(globalSessionId: string) {
  return (await getCameraHealthChunks(globalSessionId))
    .flatMap((chunk) => Array.from(new Uint8Array(chunk.data), (value, offset) => ({
      cameraSlot: chunk.cameraSlot,
      second: chunk.startSecond + offset,
      ...decodeCameraHealth(value),
    })));
}

export async function getCameraHealthIntervals(globalSessionId: string) {
  const intervals: Array<CameraHealthState & {
    cameraSlot: CameraSlot;
    startSecond: number;
    endSecond: number;
  }> = [];
  for (const chunk of await getCameraHealthChunks(globalSessionId)) {
    Array.from(new Uint8Array(chunk.data)).forEach((value, offset) => {
      const second = chunk.startSecond + offset;
      const state = decodeCameraHealth(value);
      const previous = intervals[intervals.length - 1];
      const sameState = previous &&
        previous.cameraSlot === chunk.cameraSlot &&
        previous.endSecond + 1 === second &&
        previous.connected === state.connected &&
        previous.receivingFrames === state.receivingFrames &&
        previous.personDetected === state.personDetected &&
        previous.trackingError === state.trackingError;
      if (sameState) previous.endSecond = second;
      else intervals.push({ cameraSlot: chunk.cameraSlot, startSecond: second, endSecond: second, ...state });
    });
  }
  return intervals;
}

export async function appendMotionChunk(
  session: MotionSessionRecord,
  startFrame: number,
  data: Float32Array,
  detectedFrames: number,
  fullBodyFrames: number,
  handDetectedFrames: number,
) {
  const database = await openMotionDatabase();
  const transaction = database.transaction(
    [SESSION_STORE, CHUNK_STORE],
    "readwrite",
  );
  const frameCount = data.length / MOTION_FRAME_STRIDE;
  const storedBuffer = data.buffer.slice(0);
  const chunk: MotionChunkRecord = {
    id: `${session.id}:${startFrame}`,
    sessionId: session.id,
    startFrame,
    frameCount,
    createdAt: Date.now(),
    data: storedBuffer,
  };
  const updatedSession: MotionSessionRecord = {
    ...session,
    frameCount: session.frameCount + frameCount,
    detectedFrameCount: session.detectedFrameCount + detectedFrames,
    fullBodyFrameCount: session.fullBodyFrameCount + fullBodyFrames,
    handDetectedFrameCount:
      session.handDetectedFrameCount + handDetectedFrames,
    storageBytes: session.storageBytes + storedBuffer.byteLength,
  };
  transaction.objectStore(CHUNK_STORE).put(chunk);
  transaction.objectStore(SESSION_STORE).put(updatedSession);
  await transactionDone(transaction);
  database.close();
  return updatedSession;
}

export async function finishMotionSession(
  session: MotionSessionRecord,
  endedAt: number,
) {
  const database = await openMotionDatabase();
  const transaction = database.transaction(SESSION_STORE, "readwrite");
  const completed = { ...session, endedAt };
  transaction.objectStore(SESSION_STORE).put(completed);
  await transactionDone(transaction);
  database.close();
  return completed;
}

export async function listMotionSessions() {
  const database = await openMotionDatabase();
  const transaction = database.transaction(SESSION_STORE, "readonly");
  const sessions = await requestResult<MotionSessionRecord[]>(
    transaction.objectStore(SESSION_STORE).getAll(),
  );
  await transactionDone(transaction);
  database.close();
  return sessions.sort((a, b) => b.startedAt - a.startedAt);
}

async function getMotionChunks(sessionId: string) {
  const database = await openMotionDatabase();
  const transaction = database.transaction(CHUNK_STORE, "readonly");
  const chunks = await requestResult<MotionChunkRecord[]>(
    transaction.objectStore(CHUNK_STORE).index("sessionId").getAll(sessionId),
  );
  await transactionDone(transaction);
  database.close();
  return chunks.sort((a, b) => a.startFrame - b.startFrame);
}

/** Parses a session's stored chunks back into per-frame numeric arrays. */
export async function getSessionFrames(sessionId: string): Promise<number[][]> {
  const chunks = await getMotionChunks(sessionId);
  const frames: number[][] = [];
  for (const chunk of chunks) {
    const values = new Float32Array(chunk.data);
    for (let offset = 0; offset < values.length; offset += MOTION_FRAME_STRIDE) {
      frames.push(Array.from(values.subarray(offset, offset + MOTION_FRAME_STRIDE)));
    }
  }
  return frames;
}

export async function getGlobalSessionCameraFrames(globalSessionId: string) {
  const sessions = (await listMotionSessions()).filter(
    (session) => session.globalSessionId === globalSessionId,
  );
  return Promise.all(sessions.map(async (session) => ({
    sessionId: session.id,
    cameraSlot: session.cameraSlot ?? 1,
    cameraId: session.cameraId ?? "",
    cameraLabel: session.cameraLabel ?? `카메라 ${session.cameraSlot ?? 1}`,
    logicalCameraId: session.logicalCameraId ?? `CAMERA_${session.cameraSlot ?? 1}`,
    frames: await getSessionFrames(session.id),
  })));
}

/** Reads every camera in one capture session as a single-owner timeline. */
export async function getGlobalSessionFrames(globalSessionId: string): Promise<number[][]> {
  return mergeCameraFrameStreams(await getGlobalSessionCameraFrames(globalSessionId));
}

export type ParsedMotionFrame = {
  relativeTimeMs: number;
  bodyDetected: boolean;
  fullBodyVisible: boolean;
  /** Raw yaw/pitch/roll only - the ear midpoint used to *position* the head
   * on screen isn't part of the compact stored frame (see BODY_LANDMARK_NAMES:
   * storage starts at the shoulders, not the ears), so callers that need to
   * draw a head marker must approximate its center from the shoulders. */
  head: { yaw: number; pitch: number; roll: number } | null;
  leftHandDetected: boolean;
  rightHandDetected: boolean;
  leftHandConfidence: number;
  rightHandConfidence: number;
  /** Flat [x,y,z,visibility] * BODY_LANDMARK_COUNT, same layout drawMotionSkeleton expects. */
  body: number[];
  /** Flat [x,y,z] * HAND_LANDMARK_COUNT, or null when that hand wasn't detected this frame. */
  leftHand: number[] | null;
  rightHand: number[] | null;
};

export type BodyProportionProfile = {
  sourceSessionId: string;
  sampledAt: number;
  usableFrames: number;
  shoulderToTorso: number;
  hipToTorso: number;
  upperArmToTorso: number;
  forearmToTorso: number;
  thighToTorso: number;
  shinToTorso: number;
  handToForearm: number;
};

/**
 * Parses one raw stored frame (see MOTION_FRAME_STRIDE) back into a
 * structured shape usable for rendering (session-replay.tsx) or analysis.
 * Pure and DOM-free so it's directly unit-testable.
 */
export function parseSessionFrame(frame: number[]): ParsedMotionFrame {
  const bodyDetected = frame[1] === 1;
  const fullBodyVisible = frame[2] === 1;
  const leftHandDetected = frame[6] === 1;
  const rightHandDetected = frame[7] === 1;

  const headerLength = 10;
  const bodyLength = BODY_LANDMARK_COUNT * 4;
  const handLength = HAND_LANDMARK_COUNT * 3;
  const leftHandOffset = headerLength + bodyLength;
  const rightHandOffset = leftHandOffset + handLength;

  return {
    relativeTimeMs: frame[0],
    bodyDetected,
    fullBodyVisible,
    head: bodyDetected ? { yaw: frame[3], pitch: frame[4], roll: frame[5] } : null,
    leftHandDetected,
    rightHandDetected,
    leftHandConfidence: frame[8],
    rightHandConfidence: frame[9],
    body: frame.slice(headerLength, headerLength + bodyLength),
    leftHand: leftHandDetected
      ? frame.slice(leftHandOffset, leftHandOffset + handLength)
      : null,
    rightHand: rightHandDetected
      ? frame.slice(rightHandOffset, rightHandOffset + handLength)
      : null,
  };
}

const DISPLAY_ASPECT_RATIO = 16 / 9;

function median(values: number[]) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function clampRatio(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

/**
 * Builds an anonymous body-proportion profile from coordinate frames only.
 * Absolute position, video pixels, face geometry and identity are not retained.
 */
export function deriveBodyProportionProfile(
  frames: number[][],
  sourceSessionId = "local-session",
  sampledAt = Date.now(),
): BodyProportionProfile | null {
  const samples: Array<Omit<BodyProportionProfile, "sourceSessionId" | "sampledAt" | "usableFrames">> = [];
  const metricDistance = (body: number[], from: number, to: number) => {
    const dx = (body[from * 4] - body[to * 4]) * DISPLAY_ASPECT_RATIO;
    const dy = body[from * 4 + 1] - body[to * 4 + 1];
    return Math.hypot(dx, dy);
  };
  const midpoint = (body: number[], left: number, right: number) => ({
    x: (body[left * 4] + body[right * 4]) / 2,
    y: (body[left * 4 + 1] + body[right * 4 + 1]) / 2,
  });
  const midpointDistance = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.hypot((a.x - b.x) * DISPLAY_ASPECT_RATIO, a.y - b.y);
  const handLength = (hand: number[] | null) => {
    if (!hand) return null;
    const dx = (hand[9 * 3] - hand[12 * 3]) * DISPLAY_ASPECT_RATIO;
    const dy = hand[9 * 3 + 1] - hand[12 * 3 + 1];
    return Math.hypot(dx, dy);
  };

  for (const rawFrame of frames) {
    const frame = parseSessionFrame(rawFrame);
    if (!frame.bodyDetected || !frame.fullBodyVisible) continue;
    const body = frame.body;
    const required = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17];
    if (required.some((index) => (body[index * 4 + 3] ?? 0) < 0.5)) continue;
    const shoulders = midpoint(body, 0, 1);
    const hips = midpoint(body, 12, 13);
    const torso = midpointDistance(shoulders, hips);
    if (torso < 0.05) continue;
    const shoulder = metricDistance(body, 0, 1);
    const hip = metricDistance(body, 12, 13);
    const upperArm = (metricDistance(body, 0, 2) + metricDistance(body, 1, 3)) / 2;
    const forearm = (metricDistance(body, 2, 4) + metricDistance(body, 3, 5)) / 2;
    const thigh = (metricDistance(body, 12, 14) + metricDistance(body, 13, 15)) / 2;
    const shin = (metricDistance(body, 14, 16) + metricDistance(body, 15, 17)) / 2;
    if (shoulder < 0.04 || forearm < 0.04 || thigh + shin < 0.12) continue;
    const measuredHands = [handLength(frame.leftHand), handLength(frame.rightHand)].filter(
      (value): value is number => value !== null && Number.isFinite(value) && value > 0.01,
    );
    const hand = measuredHands.length
      ? measuredHands.reduce((sum, value) => sum + value, 0) / measuredHands.length
      : forearm * 0.42;
    samples.push({
      shoulderToTorso: shoulder / torso,
      hipToTorso: hip / torso,
      upperArmToTorso: upperArm / torso,
      forearmToTorso: forearm / torso,
      thighToTorso: thigh / torso,
      shinToTorso: shin / torso,
      handToForearm: hand / forearm,
    });
  }
  if (samples.length < 3) return null;
  const value = (key: keyof (typeof samples)[number]) => median(samples.map((sample) => sample[key])) ?? 1;
  return {
    sourceSessionId,
    sampledAt,
    usableFrames: samples.length,
    shoulderToTorso: clampRatio(value("shoulderToTorso"), 0.45, 1.45),
    hipToTorso: clampRatio(value("hipToTorso"), 0.35, 1.2),
    upperArmToTorso: clampRatio(value("upperArmToTorso"), 0.45, 1.15),
    forearmToTorso: clampRatio(value("forearmToTorso"), 0.4, 1.05),
    thighToTorso: clampRatio(value("thighToTorso"), 0.65, 1.65),
    shinToTorso: clampRatio(value("shinToTorso"), 0.6, 1.55),
    handToForearm: clampRatio(value("handToForearm"), 0.25, 0.72),
  };
}

export async function getLatestBodyProportionProfile(): Promise<BodyProportionProfile | null> {
  const sessions = await listMotionSessions();
  for (const session of sessions) {
    if (session.frameCount < 3 || session.fullBodyFrameCount < 3) continue;
    const frames = await getSessionFrames(session.id);
    const profile = deriveBodyProportionProfile(frames, session.id, session.endedAt ?? session.startedAt);
    if (profile) return profile;
  }
  return null;
}

/** Extracts a single hand landmark's (x,y,z) trajectory across frames. */
export function extractHandPointTrajectory(
  frames: number[][],
  hand: "left" | "right",
  landmarkIndex: number,
): Array<{ x: number; y: number; z: number } | null> {
  // Frame layout (see MOTION_FRAME_STRIDE): 10 header values, then
  // 22 body landmarks * 4, then 21 left-hand * 3, then 21 right-hand * 3.
  const headerLength = 10;
  const bodyLength = BODY_LANDMARK_COUNT * 4;
  const leftHandOffset = headerLength + bodyLength;
  const rightHandOffset = leftHandOffset + HAND_LANDMARK_COUNT * 3;
  const baseOffset = hand === "left" ? leftHandOffset : rightHandOffset;

  return frames.map((frame) => {
    const x = frame[baseOffset + landmarkIndex * 3];
    const y = frame[baseOffset + landmarkIndex * 3 + 1];
    const z = frame[baseOffset + landmarkIndex * 3 + 2];
    if (Number.isNaN(x) || Number.isNaN(y) || Number.isNaN(z)) return null;
    return { x, y, z };
  });
}

/** Deletes every locally stored motion session and chunk. Irreversible. */
export async function deleteAllMotionSessions(): Promise<void> {
  await requestResult(indexedDB.deleteDatabase(DB_NAME));
}

/**
 * Builds one absolute timestamp row per stored skeleton frame. The UTC offset
 * is calculated for each frame so a capture spanning a daylight-saving change
 * remains unambiguous.
 */
export function buildFrameTimeIndex(frames: number[][], sessionStartedAt: number) {
  return frames.map((frame) => {
    const epochMs = sessionStartedAt + (Number.isFinite(frame[0]) ? frame[0] : 0);
    return [epochMs, -new Date(epochMs).getTimezoneOffset()];
  });
}

async function buildMotionSessionDataset(session: MotionSessionRecord) {
  const chunks = await getMotionChunks(session.id);
  const captureManifest = session.globalSessionId
    ? await getCaptureManifest(session.globalSessionId)
    : null;
  const cameraHealth = session.globalSessionId
    ? await getCameraHealthIntervals(session.globalSessionId)
    : [];
  const frames: number[][] = [];
  for (const chunk of chunks) {
    const values = new Float32Array(chunk.data);
    for (let offset = 0; offset < values.length; offset += MOTION_FRAME_STRIDE) {
      frames.push(
        Array.from(values.subarray(offset, offset + MOTION_FRAME_STRIDE)),
      );
    }
  }
  const fallbackTimeZone = getTimeZoneMetadata(session.startedAt);
  const timeZone = session.timeZone ?? captureManifest?.timeZone ?? fallbackTimeZone.timeZone;
  const utcOffsetMinutes = session.utcOffsetMinutes
    ?? captureManifest?.utcOffsetMinutes
    ?? fallbackTimeZone.utcOffsetMinutes;

  const dataset = {
    format: "memory-guard-motion-v2",
    description:
      "Body, head-direction and detailed hand coordinates only; no facial landmarks, video or audio.",
    schema: {
      sample_rate_hz: MOTION_SAMPLE_RATE,
      coordinate_space: "normalized_image",
      mirrored_preview: true,
      time_reference: {
        session_started_at_epoch_ms: session.startedAt,
        time_zone_iana: timeZone,
        utc_offset_minutes_at_start: utcOffsetMinutes,
        frame_time_index_layout: ["captured_at_epoch_ms", "utc_offset_minutes"],
        note: "frame_time_index has one row for every skeleton row in frames.",
      },
      frame_layout: [
        "relative_time_ms",
        "body_detected_0_or_1",
        "full_body_visible_0_or_1",
        "head_yaw_normalized",
        "head_pitch_normalized",
        "head_roll_radians",
        "left_hand_detected_0_or_1",
        "right_hand_detected_0_or_1",
        "left_hand_confidence",
        "right_hand_confidence",
        "22_body_landmarks_repeated_as_x_y_z_visibility",
        "21_left_hand_landmarks_repeated_as_x_y_z",
        "21_right_hand_landmarks_repeated_as_x_y_z",
      ],
      privacy: {
        facial_landmarks_stored: false,
        head_output: "direction_only",
        video_stored: false,
        audio_stored: false,
      },
      body_landmark_names: BODY_LANDMARK_NAMES,
      hand_landmark_names: HAND_LANDMARK_NAMES,
      camera_health: {
        sample_rate_hz: 1,
        format: "run_length_intervals",
        time_fields: ["startSecond", "endSecond"],
        fields: ["connected", "receivingFrames", "personDetected", "trackingError"],
      },
    },
    session,
    capture_manifest: captureManifest,
    camera_health: cameraHealth,
    frame_time_index: buildFrameTimeIndex(frames, session.startedAt),
    frames,
  };
  return dataset;
}

export async function downloadMotionSession(session: MotionSessionRecord) {
  const dataset = await buildMotionSessionDataset(session);
  const blob = new Blob([JSON.stringify(dataset)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `memory-guard-motion-${new Date(session.startedAt).toISOString()}.json`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function captureDirectoryName(session: MotionSessionRecord) {
  const stamp = new Date(session.startedAt).toISOString().replace(/[:.]/g, "-");
  const id = safeDataPathSegment(session.globalSessionId ?? session.id, "session").slice(-18);
  return `${stamp}_${id}`;
}

/** Downloads every locally retained coordinate session in one identifiable folder archive. */
export async function downloadMotionDataFolder() {
  const sessions = await listMotionSessions();
  if (sessions.length === 0) return false;

  const groups = new Map<string, MotionSessionRecord[]>();
  for (const session of sessions) {
    const key = session.globalSessionId ?? session.id;
    groups.set(key, [...(groups.get(key) ?? []), session]);
  }

  const entries: DataFolderEntry[] = [];
  const catalog: Array<{
    captureId: string;
    startedAt: string;
    directory: string;
    cameras: Array<{ slot: number; logicalCameraId: string; label: string; file: string }>;
  }> = [];

  for (const [captureId, captureSessions] of groups) {
    const ordered = [...captureSessions].sort((a, b) => (a.cameraSlot ?? 1) - (b.cameraSlot ?? 1));
    const primary = ordered[0];
    const dateDirectory = new Date(primary.startedAt).toISOString().slice(0, 10);
    const sessionDirectory = `${dateDirectory}/${captureDirectoryName(primary)}`;
    const manifest = primary.globalSessionId ? await getCaptureManifest(primary.globalSessionId) : null;
    const health = primary.globalSessionId ? await getCameraHealthIntervals(primary.globalSessionId) : [];
    entries.push({
      path: `memory-guard-data/${sessionDirectory}/capture-manifest.json`,
      contents: JSON.stringify(manifest ?? {
        id: captureId,
        startedAt: primary.startedAt,
        endedAt: primary.endedAt,
        legacySession: true,
      }, null, 2),
    });
    entries.push({
      path: `memory-guard-data/${sessionDirectory}/camera-health.json`,
      contents: JSON.stringify(health, null, 2),
    });

    const cameras = [];
    for (const session of ordered) {
      const slot = session.cameraSlot ?? 1;
      const label = session.cameraLabel ?? `카메라 ${slot}`;
      const cameraDirectory = `camera-${slot}_${safeDataPathSegment(label, `camera-${slot}`)}`;
      const file = `${sessionDirectory}/${cameraDirectory}/motion.json`;
      const dataset = await buildMotionSessionDataset(session);
      entries.push({
        path: `memory-guard-data/${file}`,
        contents: JSON.stringify(dataset),
      });
      cameras.push({
        slot,
        logicalCameraId: session.logicalCameraId ?? `CAMERA_${slot}`,
        label,
        file,
      });
    }
    catalog.push({
      captureId,
      startedAt: new Date(primary.startedAt).toISOString(),
      directory: sessionDirectory,
      cameras,
    });
  }

  entries.unshift({
    path: "memory-guard-data/catalog.json",
    contents: JSON.stringify({
      format: "memory-guard-data-folder-v1",
      exportedAt: new Date().toISOString(),
      captureCount: catalog.length,
      cameraSessionCount: sessions.length,
      captures: catalog.sort((a, b) => a.startedAt.localeCompare(b.startedAt)),
    }, null, 2),
  });
  entries.unshift({
    path: "memory-guard-data/README.txt",
    contents: [
      "Memory Guard 좌표 데이터 폴더",
      "",
      "catalog.json: 전체 촬영 세션과 카메라 파일 색인",
      "날짜/세션/capture-manifest.json: 카메라·구역 매핑 스냅샷",
      "날짜/세션/camera-health.json: 카메라별 연결·프레임·사람 인식 상태",
      "날짜/세션/camera-N_라벨/motion.json: 해당 카메라의 스켈레톤 좌표와 프레임별 절대 시각·시간대 오프셋",
      "",
      "영상·음성·얼굴 특징점은 포함되지 않습니다.",
    ].join("\n"),
  });

  const blob = createDataFolderZip(entries);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `memory-guard-data_${new Date().toISOString().slice(0, 10)}.zip`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  return true;
}
