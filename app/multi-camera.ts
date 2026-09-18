export type CameraSlot = 1 | 2 | 3;

export type CameraDeviceIdentity = Pick<MediaDeviceInfo, "deviceId" | "groupId" | "label">;

const BUILT_IN_CAMERA = /built-in|facetime|integrated|internal|macbook|continuity|iphone|desk ?view|내장|데스크뵰|아이폰|‘[^’]+’ 카메라/i;
const CONTINUITY_CAMERA = /continuity|iphone|desk ?view|아이폰|데스크뵰|데스크뷰|‘[^’]+’ 카메라/i;

export const isExternalCamera = (camera: CameraDeviceIdentity) =>
  !BUILT_IN_CAMERA.test(camera.label) && !/데스크뷰/i.test(camera.label);

/**
 * Keeps dedicated webcams first without hiding a usable camera just because
 * its driver reports a generic "Integrated Camera" style label. Continuity
 * cameras stay last so they never displace a physically connected webcam.
 */
export function listCameraCandidates<T extends CameraDeviceIdentity>(cameras: T[]): T[] {
  const unique = cameras.filter((camera, index) =>
    camera.deviceId
      ? cameras.findIndex((candidate) => candidate.deviceId === camera.deviceId) === index
      : true,
  );
  const byStableName = (a: T, b: T) =>
    a.label.localeCompare(b.label) || a.deviceId.localeCompare(b.deviceId);
  const external = unique.filter(isExternalCamera).sort(byStableName);
  const localFallback = unique
    .filter((camera) => !isExternalCamera(camera) && !CONTINUITY_CAMERA.test(camera.label))
    .sort(byStableName);
  const continuity = unique
    .filter((camera) => !isExternalCamera(camera) && CONTINUITY_CAMERA.test(camera.label))
    .sort(byStableName);
  return [...external, ...localFallback, ...continuity];
}

/** Prefers external cameras and restores up to three stable camera slots. */
export function selectExternalCameraSlots(
  cameras: CameraDeviceIdentity[],
  saved: CameraDeviceIdentity[] = [],
  requestedIds: string[] = [],
) {
  const remaining = listCameraCandidates(cameras);
  const selected: CameraDeviceIdentity[] = [];

  for (let slot = 0; slot < 3 && remaining.length > 0; slot += 1) {
    const savedCamera = saved[slot];
    const requestedIndex = remaining.findIndex((camera) => camera.deviceId === requestedIds[slot]);
    const savedIdIndex = remaining.findIndex((camera) => camera.deviceId === savedCamera?.deviceId);
    const savedGroupIndex = remaining.findIndex((camera) =>
      Boolean(savedCamera?.groupId && camera.groupId === savedCamera.groupId && camera.label === savedCamera.label),
    );
    const savedLabelIndex = savedCamera?.label && remaining.filter((camera) => camera.label === savedCamera.label).length === 1
      ? remaining.findIndex((camera) => camera.label === savedCamera.label)
      : -1;
    const index = [requestedIndex, savedIdIndex, savedGroupIndex, savedLabelIndex].find((candidate) => candidate >= 0) ?? 0;
    selected.push(remaining.splice(index >= 0 ? index : 0, 1)[0]);
  }
  return selected;
}

export type CameraCalibration = {
  cameraId: string;
  slot: CameraSlot;
  label: string;
  // All browser streams share performance.now(), so timestamps use this
  // common monotonic timeline rather than each video's presentation time.
  timelineOriginMs: number;
};

export type GlobalCaptureSession = {
  id: string;
  startedAt: number;
  timelineOriginMs: number;
  cameras: CameraCalibration[];
};

export type CameraFrameStream = {
  cameraSlot: CameraSlot;
  frames: number[][];
};

export type CameraHandoff = {
  from: CameraSlot;
  to: CameraSlot;
  atMs: number;
  gapMs: number;
};

export type CameraContinuityReport = {
  status: "pass" | "waiting" | "warning";
  handoffs: CameraHandoff[];
  overlapSamples: number;
  longestBlindGapMs: number;
  facingCorrectedFrames: number;
};

export function createGlobalCaptureSession(cameras: Omit<CameraCalibration, "timelineOriginMs">[]): GlobalCaptureSession {
  const timelineOriginMs = performance.now();
  return {
    id: `global-${crypto.randomUUID()}`,
    startedAt: Date.now(),
    timelineOriginMs,
    cameras: cameras.map((camera) => ({ ...camera, timelineOriginMs })),
  };
}

function frameQuality(frame: number[]) {
  if (frame[1] !== 1) return 0;
  let visibility = 0;
  for (let offset = 13; offset < 98; offset += 4) visibility += frame[offset] ?? 0;
  return 100 + (frame[2] === 1 ? 40 : 0) + (frame[6] === 1 ? 5 : 0) +
    (frame[7] === 1 ? 5 : 0) + visibility / 22;
}

function bodyCenter(frame: number[]) {
  if (frame[1] !== 1) return null;
  const leftHip = 10 + 12 * 4;
  const rightHip = 10 + 13 * 4;
  const values = [frame[leftHip], frame[leftHip + 1], frame[rightHip], frame[rightHip + 1]];
  return values.every(Number.isFinite)
    ? { x: (values[0] + values[2]) / 2, y: (values[1] + values[3]) / 2 }
    : null;
}

function translateFrame(frame: number[], dx: number, dy: number) {
  const translated = [...frame];
  for (let offset = 10; offset < 10 + 22 * 4; offset += 4) {
    if (Number.isFinite(translated[offset])) translated[offset] += dx;
    if (Number.isFinite(translated[offset + 1])) translated[offset + 1] += dy;
  }
  for (let offset = 10 + 22 * 4; offset < translated.length; offset += 3) {
    if (Number.isFinite(translated[offset])) translated[offset] += dx;
    if (Number.isFinite(translated[offset + 1])) translated[offset + 1] += dy;
  }
  return translated;
}

function canonicalizeFacing(frame: number[]) {
  const leftShoulderX = frame[10];
  const rightShoulderX = frame[14];
  const center = bodyCenter(frame);
  if (!center || !Number.isFinite(leftShoulderX) || !Number.isFinite(rightShoulderX) || leftShoulderX >= rightShoulderX) {
    return [...frame];
  }
  const mirrored = [...frame];
  mirrored[3] = -mirrored[3];
  for (let offset = 10; offset < 10 + 22 * 4; offset += 4) {
    if (Number.isFinite(mirrored[offset])) mirrored[offset] = center.x * 2 - mirrored[offset];
  }
  for (let offset = 10 + 22 * 4; offset < mirrored.length; offset += 3) {
    if (Number.isFinite(mirrored[offset])) mirrored[offset] = center.x * 2 - mirrored[offset];
  }
  return mirrored;
}

/**
 * Produces one owner timeline from synchronized camera streams. Frames in
 * the same sample window are duplicate observations of the study participant,
 * so the clearest skeleton wins instead of counting the action twice.
 */
export function mergeCameraFrameStreams(
  streams: CameraFrameStream[],
  sampleWindowMs = 100,
) {
  const buckets = new Map<number, { cameraSlot: CameraSlot; frame: number[]; quality: number }>();
  for (const { cameraSlot, frames } of streams) {
    for (const frame of frames) {
      const bucket = Math.round((frame[0] ?? 0) / sampleWindowMs);
      const quality = frameQuality(frame);
      const current = buckets.get(bucket);
      if (!current || quality > current.quality) {
        buckets.set(bucket, { cameraSlot, frame: canonicalizeFacing(frame), quality });
      }
    }
  }
  let previousSlot: CameraSlot | null = null;
  let previousCenter: { x: number; y: number } | null = null;
  const offsets = new Map<CameraSlot, { x: number; y: number }>();
  return [...buckets.entries()]
    .sort(([a], [b]) => a - b)
    .map(([bucket, item]) => {
      const rawCenter = bodyCenter(item.frame);
      let offset = offsets.get(item.cameraSlot) ?? { x: 0, y: 0 };
      if (rawCenter && previousCenter && previousSlot !== item.cameraSlot) {
        // ponytail: translation-only handoff is enough for the fixed-camera setup;
        // replace with floor-point homography when measured store coordinates exist.
        offset = { x: previousCenter.x - rawCenter.x, y: previousCenter.y - rawCenter.y };
        offsets.set(item.cameraSlot, offset);
      }
      const frame = translateFrame(item.frame, offset.x, offset.y);
      frame[0] = bucket * sampleWindowMs;
      previousCenter = bodyCenter(frame) ?? previousCenter;
      previousSlot = item.cameraSlot;
      return frame;
    });
}

export function cameraPresenceSegments(frames: number[][], maxGapMs = 250) {
  const times = frames.filter((frame) => frame[1] === 1).map((frame) => frame[0]).sort((a, b) => a - b);
  if (!times.length) return [];
  const segments = [{ startMs: times[0], endMs: times[0] }];
  for (const time of times.slice(1)) {
    const current = segments[segments.length - 1];
    if (time - current.endMs > maxGapMs) segments.push({ startMs: time, endMs: time });
    else current.endMs = time;
  }
  return segments;
}

export function analyzeCameraContinuity(
  streams: CameraFrameStream[],
  sampleWindowMs = 100,
  maxHandoffGapMs = 8_000,
): CameraContinuityReport {
  const presence = new Map<number, Set<CameraSlot>>();
  let facingCorrectedFrames = 0;
  for (const stream of streams) {
    for (const frame of stream.frames) {
      if (frame[1] !== 1) continue;
      const bucket = Math.round(frame[0] / sampleWindowMs) * sampleWindowMs;
      const slots = presence.get(bucket) ?? new Set<CameraSlot>();
      slots.add(stream.cameraSlot);
      presence.set(bucket, slots);
      if (Number.isFinite(frame[10]) && Number.isFinite(frame[14]) && frame[10] < frame[14]) {
        facingCorrectedFrames += 1;
      }
    }
  }
  const times = [...presence.keys()].sort((a, b) => a - b);
  const lastSeen = new Map<CameraSlot, number>();
  const handoffs: CameraHandoff[] = [];
  let active: CameraSlot | null = null;
  let longestBlindGapMs = 0;
  times.forEach((time, index) => {
    const slots = [...presence.get(time)!].sort();
    slots.forEach((slot) => lastSeen.set(slot, time));
    if (index > 0) longestBlindGapMs = Math.max(longestBlindGapMs, time - times[index - 1] - sampleWindowMs);
    if (active === null) active = slots[0];
    else if (!slots.includes(active)) {
      const next = slots[0];
      handoffs.push({ from: active, to: next, atMs: time, gapMs: Math.max(0, time - (lastSeen.get(active) ?? time)) });
      active = next;
    }
  });
  const overlapSamples = times.filter((time) => presence.get(time)!.size > 1).length;
  const hasTwoStreams = streams.filter((stream) => stream.frames.some((frame) => frame[1] === 1)).length > 1;
  const failed = longestBlindGapMs > maxHandoffGapMs || handoffs.some((handoff) => handoff.gapMs > maxHandoffGapMs);
  return {
    status: failed ? "warning" : hasTwoStreams && handoffs.length > 0 ? "pass" : "waiting",
    handoffs,
    overlapSamples,
    longestBlindGapMs,
    facingCorrectedFrames,
  };
}
