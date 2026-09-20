export const BACKGROUND_WIDTH = 32;
export const BACKGROUND_HEIGHT = 18;

export type BackgroundGateState = {
  pixels: Float32Array | null;
  learnedFrames: number;
  foregroundRatio: number;
  lastForegroundAt: number;
};

export const EMPTY_BACKGROUND_GATE: BackgroundGateState = {
  pixels: null,
  learnedFrames: 0,
  foregroundRatio: 0,
  lastForegroundAt: 0,
};

export function grayscaleSignature(data: Uint8ClampedArray) {
  const signature = new Float32Array(data.length / 4);
  for (let source = 0, target = 0; source < data.length; source += 4, target += 1) {
    signature[target] = data[source] * 0.299 + data[source + 1] * 0.587 + data[source + 2] * 0.114;
  }
  return signature;
}

export function updateBackgroundGate(
  previous: BackgroundGateState,
  current: Float32Array,
  timestamp: number,
  mayLearnBackground: boolean,
) {
  if (!previous.pixels || previous.pixels.length !== current.length) {
    return mayLearnBackground
      ? {
          pixels: current.slice(),
          learnedFrames: 1,
          foregroundRatio: 0,
          lastForegroundAt: 0,
        }
      : { ...EMPTY_BACKGROUND_GATE };
  }

  let changed = 0;
  for (let index = 0; index < current.length; index += 1) {
    if (Math.abs(current[index] - previous.pixels[index]) >= 22) changed += 1;
  }
  const foregroundRatio = changed / current.length;
  const foregroundDetected =
    (previous.learnedFrames >= 3 && foregroundRatio >= 0.045) ||
    // If a headless person enters just after startup, one clean empty-room
    // sample plus a large scene change is safer than learning that person into
    // the background forever.
    (previous.learnedFrames >= 1 && foregroundRatio >= 0.18);
  const pixels = previous.pixels.slice();
  // Learn only genuinely quiet, person-free frames. Slow adaptation handles
  // daylight changes without absorbing a newly arrived person into the room.
  if (mayLearnBackground && foregroundRatio < 0.025) {
    for (let index = 0; index < pixels.length; index += 1) {
      pixels[index] = pixels[index] * 0.98 + current[index] * 0.02;
    }
  }
  return {
    pixels,
    learnedFrames: mayLearnBackground && foregroundRatio < 0.025
      ? previous.learnedFrames + 1
      : previous.learnedFrames,
    foregroundRatio,
    lastForegroundAt: foregroundDetected ? timestamp : previous.lastForegroundAt,
  };
}

export function personPresenceAllowsAcquisition(args: {
  timestamp: number;
  lastPersonDetectionAt: number;
  background: BackgroundGateState;
}) {
  const detectorIsRecent = args.lastPersonDetectionAt > 0 &&
    args.timestamp - args.lastPersonDetectionAt <= 1_200;
  const foregroundIsRecent = args.background.learnedFrames >= 1 &&
    args.background.lastForegroundAt > 0 &&
    args.timestamp - args.background.lastForegroundAt <= 1_200;
  return detectorIsRecent || foregroundIsRecent;
}
