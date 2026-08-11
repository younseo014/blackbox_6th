export const POSE_LANDMARK_COUNT = 33;
export const POSE_SAMPLE_RATE = 5;
export const POSE_FRAME_STRIDE = 3 + POSE_LANDMARK_COUNT * 4;

export const POSE_LANDMARK_NAMES = [
  "nose",
  "left_eye_inner",
  "left_eye",
  "left_eye_outer",
  "right_eye_inner",
  "right_eye",
  "right_eye_outer",
  "left_ear",
  "right_ear",
  "mouth_left",
  "mouth_right",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_pinky",
  "right_pinky",
  "left_index",
  "right_index",
  "left_thumb",
  "right_thumb",
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

export type PoseSessionRecord = {
  id: string;
  startedAt: number;
  endedAt: number | null;
  frameCount: number;
  detectedFrameCount: number;
  fullBodyFrameCount: number;
  storageBytes: number;
  sampleRate: number;
  landmarkCount: number;
  coordinateSpace: "normalized_image";
  mirroredPreview: boolean;
  source: "local_camera";
};

export type PoseChunkRecord = {
  id: string;
  sessionId: string;
  startFrame: number;
  frameCount: number;
  createdAt: number;
  data: ArrayBuffer;
};

const DB_NAME = "memory-guard-pose-v1";
const DB_VERSION = 1;
const SESSION_STORE = "pose_sessions";
const CHUNK_STORE = "pose_chunks";

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

export function openPoseDatabase() {
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

export async function createPoseSession(id: string, startedAt: number) {
  const database = await openPoseDatabase();
  const transaction = database.transaction(SESSION_STORE, "readwrite");
  const record: PoseSessionRecord = {
    id,
    startedAt,
    endedAt: null,
    frameCount: 0,
    detectedFrameCount: 0,
    fullBodyFrameCount: 0,
    storageBytes: 0,
    sampleRate: POSE_SAMPLE_RATE,
    landmarkCount: POSE_LANDMARK_COUNT,
    coordinateSpace: "normalized_image",
    mirroredPreview: true,
    source: "local_camera",
  };
  transaction.objectStore(SESSION_STORE).put(record);
  await transactionDone(transaction);
  database.close();
  return record;
}

export async function appendPoseChunk(
  session: PoseSessionRecord,
  startFrame: number,
  data: Float32Array,
  detectedFrames: number,
  fullBodyFrames: number,
) {
  const database = await openPoseDatabase();
  const transaction = database.transaction(
    [SESSION_STORE, CHUNK_STORE],
    "readwrite",
  );
  const frameCount = data.length / POSE_FRAME_STRIDE;
  const storedBuffer = data.buffer.slice(0);
  const chunk: PoseChunkRecord = {
    id: `${session.id}:${startFrame}`,
    sessionId: session.id,
    startFrame,
    frameCount,
    createdAt: Date.now(),
    data: storedBuffer,
  };
  const updatedSession: PoseSessionRecord = {
    ...session,
    frameCount: session.frameCount + frameCount,
    detectedFrameCount: session.detectedFrameCount + detectedFrames,
    fullBodyFrameCount: session.fullBodyFrameCount + fullBodyFrames,
    storageBytes: session.storageBytes + storedBuffer.byteLength,
  };
  transaction.objectStore(CHUNK_STORE).put(chunk);
  transaction.objectStore(SESSION_STORE).put(updatedSession);
  await transactionDone(transaction);
  database.close();
  return updatedSession;
}

export async function finishPoseSession(
  session: PoseSessionRecord,
  endedAt: number,
) {
  const database = await openPoseDatabase();
  const transaction = database.transaction(SESSION_STORE, "readwrite");
  const completed = { ...session, endedAt };
  transaction.objectStore(SESSION_STORE).put(completed);
  await transactionDone(transaction);
  database.close();
  return completed;
}

export async function listPoseSessions() {
  const database = await openPoseDatabase();
  const transaction = database.transaction(SESSION_STORE, "readonly");
  const sessions = await requestResult<PoseSessionRecord[]>(
    transaction.objectStore(SESSION_STORE).getAll(),
  );
  await transactionDone(transaction);
  database.close();
  return sessions.sort((a, b) => b.startedAt - a.startedAt);
}

async function getPoseChunks(sessionId: string) {
  const database = await openPoseDatabase();
  const transaction = database.transaction(CHUNK_STORE, "readonly");
  const chunks = await requestResult<PoseChunkRecord[]>(
    transaction.objectStore(CHUNK_STORE).index("sessionId").getAll(sessionId),
  );
  await transactionDone(transaction);
  database.close();
  return chunks.sort((a, b) => a.startFrame - b.startFrame);
}

export async function downloadPoseSession(session: PoseSessionRecord) {
  const chunks = await getPoseChunks(session.id);
  const frames: number[][] = [];
  for (const chunk of chunks) {
    const values = new Float32Array(chunk.data);
    for (let offset = 0; offset < values.length; offset += POSE_FRAME_STRIDE) {
      frames.push(Array.from(values.subarray(offset, offset + POSE_FRAME_STRIDE)));
    }
  }

  const dataset = {
    format: "memory-guard-pose-v1",
    description: "Local-camera pose coordinates only; no video or audio.",
    schema: {
      sample_rate_hz: POSE_SAMPLE_RATE,
      coordinate_space: "normalized_image",
      mirrored_preview: true,
      frame_layout: [
        "relative_time_ms",
        "pose_detected_0_or_1",
        "full_body_visible_0_or_1",
        "33_landmarks_repeated_as_x_y_z_visibility",
      ],
      landmark_names: POSE_LANDMARK_NAMES,
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
  link.download = `memory-guard-pose-${new Date(session.startedAt).toISOString()}.json`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
