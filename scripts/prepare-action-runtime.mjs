import { createHash } from "node:crypto";
import { mkdir, readFile, rename, copyFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
const directory = path.join(root, "public/models/action-context");
const metadata = JSON.parse(await readFile(path.join(directory, "prototypes.json"), "utf8"));
const modelPath = path.join(directory, "vision-int8.onnx");
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const existing = await readFile(modelPath).catch(() => null);
if (!existing || digest(existing) !== metadata.sha256) {
  const response = await fetch(`https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/${metadata.revision}/onnx/vision_model_quantized.onnx`);
  if (!response.ok) throw new Error(`Model download failed: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (digest(bytes) !== metadata.sha256) throw new Error("Action model checksum mismatch");
  const { writeFile } = await import("node:fs/promises");
  await writeFile(`${modelPath}.download`, bytes);
  await rename(`${modelPath}.download`, modelPath);
}
await mkdir(path.join(root, "public/ort"), { recursive: true });
for (const file of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  await copyFile(path.join(root, "node_modules/onnxruntime-web/dist", file), path.join(root, "public/ort", file));
}
console.log(`Local action model ready: ${metadata.model}; SHA-256 verified; license ${metadata.license}`);
