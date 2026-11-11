/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/wasm";
import { rankVisualContext, type VisualActionKind } from "./local-action-context";

let session: ort.InferenceSession | null = null;
let prototypes: Array<{ label: string; kind: VisualActionKind; embedding: number[] }> = [];
const canvas = new OffscreenCanvas(224, 224);
const context = canvas.getContext("2d", { willReadFrequently: true });
const mean = [0.48145466, 0.4578275, 0.40821073], std = [0.26862954, 0.26130258, 0.27577711];

self.onmessage = async (event: MessageEvent) => {
  const request = event.data;
  try {
    if (request.type === "load") {
      ort.env.wasm.wasmPaths = request.runtimeUrl;
      ort.env.wasm.numThreads = 1;
      const response = await fetch(request.prototypesUrl);
      if (!response.ok) throw new Error("Local action prototypes are missing");
      prototypes = (await response.json()).prototypes;
      session = await ort.InferenceSession.create(request.modelUrl, { executionProviders: ["wasm"] });
      self.postMessage({ type: "ready" });
      return;
    }
    if (!session || !context) throw new Error("Local action model is not ready");
    const bitmap = request.bitmap as ImageBitmap;
    try {
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      const side = Math.min(bitmap.width, bitmap.height);
      context.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, 224, 224);
      const rgba = context.getImageData(0, 0, 224, 224).data;
      const values = new Float32Array(3 * 224 * 224), plane = 224 * 224;
      for (let i = 0; i < plane; i++) for (let channel = 0; channel < 3; channel++) {
        values[channel * plane + i] = (rgba[4 * i + channel] / 255 - mean[channel]) / std[channel];
      }
      const tensor = new ort.Tensor("float32", values, [1, 3, 224, 224]);
      let outputs: ort.InferenceSession.OnnxValueMapType | undefined;
      try {
        outputs = await session.run({ pixel_values: tensor });
        const result = rankVisualContext(outputs.image_embeds.data as Float32Array, prototypes, request.taskLabels);
        self.postMessage({ type: "result", id: request.id, result });
      } finally { tensor.dispose(); if (outputs) Object.values(outputs).forEach(value => value.dispose()); }
    } finally { bitmap.close(); }
  } catch (error) {
    self.postMessage({ type: request.type === "load" ? "load-error" : "result-error", id: request.id,
      error: error instanceof Error ? error.message : String(error) });
  }
};
