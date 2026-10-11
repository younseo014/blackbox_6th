import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const modelRoot = join(root, "models", "yolox", "yolox_tiny");
const runtimeRoot = join(root, "node_modules", "onnxruntime-web", "dist");
const targetRoot = join(root, "public");

function findModel(directory) {
  if (!existsSync(directory)) return null;
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      const found = findModel(path);
      if (found) return found;
    } else if (entry.name === "end2end.onnx") return path;
  }
  return null;
}

const model = findModel(modelRoot);
if (!model) throw new Error(`Extract the YOLOX-m SDK archive into ${modelRoot} first.`);
if (!existsSync(runtimeRoot)) throw new Error("Install dependencies before preparing YOLOX.");

mkdirSync(join(targetRoot, "models"), { recursive: true });
mkdirSync(join(targetRoot, "ort"), { recursive: true });
copyFileSync(model, join(targetRoot, "models", "yolox_tiny.onnx"));
for (const filename of readdirSync(runtimeRoot)) {
  if (/^ort-wasm-simd-threaded(?:\.[^.]+)?\.(?:mjs|wasm)$/.test(filename)) {
    copyFileSync(join(runtimeRoot, filename), join(targetRoot, "ort", filename));
  }
}
console.log(`YOLOX-tiny ready (${Math.round(statSync(model).size / 1_000_000)} MB model).`);
