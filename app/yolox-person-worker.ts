/// <reference lib="webworker" />

import * as ort from "onnxruntime-web/wasm";

const PERSON_CLASS = 0;
const canvas = new OffscreenCanvas(1, 1);
const context = canvas.getContext("2d", { willReadFrequently: true });
let session: ort.InferenceSession | null = null;
let inputWidth = 0;
let inputHeight = 0;

type Request =
  | { type: "load"; modelUrl: string; runtimeUrl: string }
  | { type: "detect"; id: number; bitmap: ImageBitmap };

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function frameTensor(bitmap: ImageBitmap) {
  if (!context) throw new Error("2D canvas is unavailable in the detector worker.");
  canvas.width = inputWidth;
  canvas.height = inputHeight;
  const scale = Math.min(inputWidth / bitmap.width, inputHeight / bitmap.height);
  const width = Math.round(bitmap.width * scale);
  const height = Math.round(bitmap.height * scale);
  context.fillStyle = "rgb(114,114,114)";
  context.fillRect(0, 0, inputWidth, inputHeight);
  context.drawImage(bitmap, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, inputWidth, inputHeight).data;
  const plane = inputWidth * inputHeight;
  const values = new Float32Array(plane * 3);
  for (let index = 0; index < plane; index += 1) {
    // Match scripts/person_detector.py: BGR, float32, no normalisation.
    values[index] = pixels[index * 4 + 2];
    values[plane + index] = pixels[index * 4 + 1];
    values[plane * 2 + index] = pixels[index * 4];
  }
  return new ort.Tensor("float32", values, [1, 3, inputHeight, inputWidth]);
}

function bestPersonScore(outputs: ort.InferenceSession.OnnxValueMapType) {
  if (!session) return 0;
  const detections = outputs[session.outputNames[0]];
  const labels = outputs[session.outputNames[1]];
  if (!detections || !labels) throw new Error("YOLOX output tensors are missing.");
  if (detections.dims.at(-1) !== 5) throw new Error("Unexpected YOLOX detection shape.");
  const boxes = detections.data;
  const classes = labels.data;
  let score = 0;
  for (let index = 0; index < classes.length; index += 1) {
    if (Number(classes[index]) === PERSON_CLASS) {
      score = Math.max(score, Number(boxes[index * 5 + 4]));
    }
  }
  return score;
}

self.onmessage = async (event: MessageEvent<Request>) => {
  const request = event.data;
  if (request.type === "load") {
    try {
      ort.env.wasm.wasmPaths = request.runtimeUrl;
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create(request.modelUrl, {
        executionProviders: ["wasm"],
      });
      const input = session.inputMetadata[0];
      if (!input?.isTensor || !Number.isInteger(input.shape[2]) || !Number.isInteger(input.shape[3])) {
        throw new Error("YOLOX input dimensions are unavailable.");
      }
      inputHeight = Number(input.shape[2]);
      inputWidth = Number(input.shape[3]);
      self.postMessage({ type: "ready" });
    } catch (error) {
      self.postMessage({ type: "load-error", error: errorMessage(error) });
    }
    return;
  }

  try {
    if (!session) throw new Error("YOLOX model has not loaded.");
    const tensor = frameTensor(request.bitmap);
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    self.postMessage({ type: "result", id: request.id, score: bestPersonScore(outputs) });
  } catch (error) {
    self.postMessage({ type: "result-error", id: request.id, error: errorMessage(error) });
  } finally {
    request.bitmap.close();
  }
};
