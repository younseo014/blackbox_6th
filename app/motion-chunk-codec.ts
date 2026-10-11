/** Versioned storage codec. Decoded frames keep the legacy Float32 layout. */
export const MOTION_CHUNK_CODEC = "q16-gzip-v1" as const;
export type MotionChunkCodec = "q16-gzip-v1" | "q16-v1" | "f32-v1";

export const MOTION_CODEC_FRAME_VALUES = 224;
const FRAME_VALUES = MOTION_CODEC_FRAME_VALUES;
const QUANTIZED_VALUES = FRAME_VALUES - 1;
const FRAME_BYTES = 4 + QUANTIZED_VALUES * 2;
const MAX_QUANTIZED = 65_533;
const NAN_CODE = 65_534;
const ESCAPE_CODE = 65_535;

function rangeForIndex(index: number): readonly [number, number] {
  if (index === 5) return [-Math.PI, Math.PI];
  if (index === 3 || index === 4) return [-1, 1];
  if (index >= 10 && index < 98) {
    return (index - 10) % 4 === 2 ? [-2, 2] : [0, 1];
  }
  if (index >= 98) {
    return (index - 98) % 3 === 2 ? [-2, 2] : [0, 1];
  }
  return [0, 1];
}

const VALUE_RANGES = Array.from({ length: FRAME_VALUES }, (_, index) => rangeForIndex(index));

function needsEscape(value: number, index: number) {
  if (Number.isNaN(value)) return false;
  const [minimum, maximum] = VALUE_RANGES[index];
  return !Number.isFinite(value) || value < minimum || value > maximum;
}

/** Stores Float32 time exactly; all other values use 16-bit fixed point or a Float32 escape. */
export function encodeQuantizedFrames(values: Float32Array): ArrayBuffer {
  if (values.length % FRAME_VALUES !== 0) throw new Error("Invalid motion frame length.");
  const frameCount = values.length / FRAME_VALUES;
  let escapes = 0;
  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let index = 1; index < FRAME_VALUES; index += 1) {
      if (needsEscape(values[frame * FRAME_VALUES + index], index)) escapes += 1;
    }
  }
  const buffer = new ArrayBuffer(frameCount * FRAME_BYTES + escapes * 4);
  const view = new DataView(buffer);
  let offset = 0;
  for (let frame = 0; frame < frameCount; frame += 1) {
    const base = frame * FRAME_VALUES;
    view.setFloat32(offset, values[base], true);
    offset += 4;
    const escaped: number[] = [];
    for (let index = 1; index < FRAME_VALUES; index += 1) {
      const value = values[base + index];
      let code: number;
      if (Number.isNaN(value)) code = NAN_CODE;
      else if (needsEscape(value, index)) {
        code = ESCAPE_CODE;
        escaped.push(value);
      } else {
        const [minimum, maximum] = VALUE_RANGES[index];
        code = Math.round(((value - minimum) / (maximum - minimum)) * MAX_QUANTIZED);
      }
      view.setUint16(offset, code, true);
      offset += 2;
    }
    for (const value of escaped) {
      view.setFloat32(offset, value, true);
      offset += 4;
    }
  }
  return buffer;
}

export function decodeQuantizedFrames(buffer: ArrayBuffer, frameCount: number): Float32Array {
  if (buffer.byteLength < frameCount * FRAME_BYTES) throw new Error("Motion chunk is incomplete.");
  const values = new Float32Array(frameCount * FRAME_VALUES);
  const view = new DataView(buffer);
  let offset = 0;
  for (let frame = 0; frame < frameCount; frame += 1) {
    const base = frame * FRAME_VALUES;
    values[base] = view.getFloat32(offset, true);
    offset += 4;
    const escapedIndices: number[] = [];
    for (let index = 1; index < FRAME_VALUES; index += 1) {
      if (offset + 2 > buffer.byteLength) throw new Error("Motion chunk is incomplete.");
      const code = view.getUint16(offset, true);
      offset += 2;
      if (code === NAN_CODE) values[base + index] = Number.NaN;
      else if (code === ESCAPE_CODE) escapedIndices.push(index);
      else {
        const [minimum, maximum] = VALUE_RANGES[index];
        values[base + index] = minimum + (code / MAX_QUANTIZED) * (maximum - minimum);
      }
    }
    for (const index of escapedIndices) {
      if (offset + 4 > buffer.byteLength) throw new Error("Motion chunk is incomplete.");
      values[base + index] = view.getFloat32(offset, true);
      offset += 4;
    }
  }
  if (offset !== buffer.byteLength) throw new Error("Motion chunk has unexpected trailing bytes.");
  return values;
}

async function gzip(buffer: ArrayBuffer) {
  const stream = new Blob([buffer]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

async function gunzip(buffer: ArrayBuffer) {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

export async function encodeMotionChunk(values: Float32Array): Promise<{ codec: MotionChunkCodec; data: ArrayBuffer }> {
  const quantized = encodeQuantizedFrames(values);
  if (typeof CompressionStream === "undefined" || typeof DecompressionStream === "undefined") {
    console.warn("[motion-storage] gzip codec unavailable; storing 16-bit coordinates without compression");
    return { codec: "q16-v1", data: quantized };
  }
  try {
    return { codec: MOTION_CHUNK_CODEC, data: await gzip(quantized) };
  } catch (error) {
    console.warn("[motion-storage] gzip failed; storing 16-bit coordinates without compression", error);
    return { codec: "q16-v1", data: quantized };
  }
}

export async function decodeMotionChunk(
  codec: MotionChunkCodec | undefined,
  data: ArrayBuffer,
  frameCount: number,
): Promise<Float32Array> {
  if (!codec || codec === "f32-v1") {
    if (data.byteLength !== frameCount * FRAME_VALUES * 4) throw new Error("Legacy motion chunk has invalid length.");
    return new Float32Array(data);
  }
  if (codec === "q16-v1") return decodeQuantizedFrames(data, frameCount);
  if (codec === "q16-gzip-v1") {
    if (typeof DecompressionStream === "undefined") throw new Error("gzip decompression is unavailable.");
    return decodeQuantizedFrames(await gunzip(data), frameCount);
  }
  throw new Error(`Unsupported motion chunk codec: ${codec}`);
}
