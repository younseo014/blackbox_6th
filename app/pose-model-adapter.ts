import type { NormalizedLandmarkLike } from "./motion-analysis";

/**
 * Swappable pose backend.
 *
 * The tracking stack above this line - target lock, presence gate, global
 * person fusion - only ever sees normalized landmarks. This adapter is the one
 * seam where the model behind them can be exchanged, so RTMW/RTMPose can be
 * A/B compared against MediaPipe Heavy on the same evaluation scenarios
 * without editing the loop in `page.tsx`.
 *
 * Landmark indices are MediaPipe Pose's 33-point layout. A backend with a
 * different skeleton (RTMW WholeBody has 133 keypoints) must map into that
 * layout in its own adapter, because the stored frame format and every
 * downstream analyser are defined against it.
 */

export type PoseBackendId = "mediapipe-heavy" | "rtmw-l-384x288" | "rtmw-l-256x192" | "rtmw-m-256x192";

export type PoseBackendCapabilities = {
  /** Where inference runs. Anything but "worker" can block the UI thread. */
  execution: "main-thread" | "worker";
  /** Compute backend actually selected at load time. */
  delegate: "gpu" | "webgpu" | "wasm" | "cpu";
  /** True when the backend also returns hand/finger keypoints. */
  wholeBody: boolean;
};

export type PoseBackendSource = {
  /** Where the weights came from. Required - an unattributed model is not usable evidence. */
  url: string;
  /** sha256 of the model file as shipped in `public/models`. */
  sha256: string;
  license: string;
};

export type PoseInferenceResult = {
  /** Candidate people, each as a 33-point MediaPipe-layout landmark list. */
  landmarks: NormalizedLandmarkLike[][];
  /** Monotonic timestamp the frame was submitted with. */
  timestamp: number;
};

export type PoseBackend = {
  id: PoseBackendId;
  label: string;
  capabilities: PoseBackendCapabilities;
  source: PoseBackendSource;
  detect(frame: CanvasImageSource, timestamp: number): Promise<PoseInferenceResult> | PoseInferenceResult;
  close(): void;
};

export type PoseBackendFactory = {
  id: PoseBackendId;
  label: string;
  create(): Promise<PoseBackend>;
};

export type BackendLoadAttempt = {
  id: PoseBackendId;
  ok: boolean;
  error?: string;
};

export type BackendLoadOutcome = {
  backend: PoseBackend | null;
  attempts: BackendLoadAttempt[];
  /** True when the requested backend failed and a later one was used instead. */
  usedFallback: boolean;
};

/**
 * Loads the first backend in `order` that actually initialises.
 *
 * Model loading is the failure most likely to happen on a machine other than
 * the developer's - no WebGPU, a 200MB download that never finishes, a driver
 * that rejects the graph. Falling back silently to MediaPipe is the correct
 * behaviour for the study; failing to *report* that it happened is not, so the
 * attempt log is part of the return value and is written into the session
 * record by the caller.
 */
export async function loadPoseBackend(
  order: PoseBackendFactory[],
): Promise<BackendLoadOutcome> {
  const attempts: BackendLoadAttempt[] = [];
  for (const factory of order) {
    try {
      const backend = await factory.create();
      attempts.push({ id: factory.id, ok: true });
      return { backend, attempts, usedFallback: attempts.length > 1 };
    } catch (error) {
      attempts.push({
        id: factory.id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
  return { backend: null, attempts, usedFallback: false };
}

/**
 * Preference order from the study plan. MediaPipe Heavy is last on purpose:
 * it is the known-good floor, so it must never be skipped past when the
 * faster or more accurate backends fail to load.
 */
export const BACKEND_PREFERENCE: PoseBackendId[] = [
  "rtmw-l-384x288",
  "rtmw-l-256x192",
  "rtmw-m-256x192",
  "mediapipe-heavy",
];

export function orderFactories(factories: PoseBackendFactory[]): PoseBackendFactory[] {
  const rank = (id: PoseBackendId) => {
    const index = BACKEND_PREFERENCE.indexOf(id);
    return index === -1 ? BACKEND_PREFERENCE.length : index;
  };
  return [...factories].sort((a, b) => rank(a.id) - rank(b.id));
}
