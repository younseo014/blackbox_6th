export const BODY_LANDMARK_COUNT = 22;
export const HAND_LANDMARK_COUNT = 21;
export const MOTION_SAMPLE_RATE = 5;

// time, body/full-body flags, head yaw/pitch/roll,
// left/right hand flags and confidence, 22 body points × 4, 42 hand points × 3.
export const MOTION_FRAME_STRIDE =
  10 + BODY_LANDMARK_COUNT * 4 + HAND_LANDMARK_COUNT * 2 * 3;

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
};

export type MotionChunkRecord = {
  id: string;
  sessionId: string;
  startFrame: number;
  frameCount: number;
  createdAt: number;
  data: ArrayBuffer;
};

const DB_NAME = "memory-guard-motion-v2";
const DB_VERSION = 1;
const SESSION_STORE = "motion_sessions";
const CHUNK_STORE = "motion_chunks";

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
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function createMotionSession(id: string, startedAt: number) {
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
  };
  transaction.objectStore(SESSION_STORE).put(record);
  await transactionDone(transaction);
  database.close();
  return record;
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
  const database = await openMotionDatabase();
  const transaction = database.transaction(
    [SESSION_STORE, CHUNK_STORE],
    "readwrite",
  );
  transaction.objectStore(SESSION_STORE).clear();
  transaction.objectStore(CHUNK_STORE).clear();
  await transactionDone(transaction);
  database.close();
}

export async function downloadMotionSession(session: MotionSessionRecord) {
  const chunks = await getMotionChunks(session.id);
  const frames: number[][] = [];
  for (const chunk of chunks) {
    const values = new Float32Array(chunk.data);
    for (let offset = 0; offset < values.length; offset += MOTION_FRAME_STRIDE) {
      frames.push(
        Array.from(values.subarray(offset, offset + MOTION_FRAME_STRIDE)),
      );
    }
  }

  const dataset = {
    format: "memory-guard-motion-v2",
    description:
      "Body, head-direction and detailed hand coordinates only; no facial landmarks, video or audio.",
    schema: {
      sample_rate_hz: MOTION_SAMPLE_RATE,
      coordinate_space: "normalized_image",
      mirrored_preview: true,
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
    },
    session,
    frames,
  };
  const blob = new Blob([JSON.stringify(dataset)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `memory-guard-motion-${new Date(session.startedAt).toISOString()}.json`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
