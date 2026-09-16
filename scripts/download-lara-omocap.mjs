import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, open, rename, rm, stat } from "node:fs/promises";
import { spawn } from "node:child_process";
import { pipeline } from "node:stream/promises";
import path from "node:path";

const URL = "https://zenodo.org/api/records/3862782/files/OMoCap%20data.zip/content";
const TOTAL_BYTES = 5_903_274_844;
const EXPECTED_MD5 = "9055d879ec6c0e63a8e3190276f06a66";
const PART_COUNT = 16;
const CONCURRENCY = 4;
const outputDir = path.resolve("data/external/lara");
const outputPath = path.join(outputDir, "OMoCap data.zip");
const partDir = path.join(outputDir, ".parts");
const chunkSize = Math.ceil(TOTAL_BYTES / PART_COUNT);

async function sizeOf(filePath) {
  try {
    return (await stat(filePath)).size;
  } catch {
    return 0;
  }
}

function partSpec(index) {
  const start = index * chunkSize;
  const end = Math.min(TOTAL_BYTES - 1, start + chunkSize - 1);
  return {
    index,
    start,
    end,
    size: end - start + 1,
    path: path.join(partDir, `part-${String(index).padStart(2, "0")}`),
  };
}

async function runCurl(part) {
  const temporaryPath = `${part.path}.partial`;
  const tailPath = `${part.path}.tail`;
  const existingTail = await sizeOf(tailPath);
  const existingPartial = await sizeOf(temporaryPath);
  if (existingTail > 0 && existingPartial + existingTail <= part.size) {
    await pipeline(createReadStream(tailPath), createWriteStream(temporaryPath, { flags: "a" }));
    await rm(tailPath, { force: true });
    return runCurl(part);
  }
  if (existingTail > 0) await rm(tailPath, { force: true });
  const downloaded = await sizeOf(temporaryPath);
  if (downloaded > part.size) {
    await rm(`${part.path}.partial`, { force: true });
    return runCurl(part);
  }
  if (downloaded === part.size) {
    await rename(`${part.path}.partial`, part.path);
    return;
  }
  return new Promise((resolve, reject) => {
    const remaining = part.size - downloaded;
    const child = spawn("curl", [
      "-sS", "-L", "--fail", "--retry", "20", "--retry-delay", "5",
      "-r", `${part.start + downloaded}-${part.end}`,
      "-o", tailPath,
      URL,
    ], { stdio: ["ignore", "ignore", "inherit"] });
    child.on("error", reject);
    child.on("exit", async (code) => {
      if (code !== 0) {
        resolve(runCurl(part));
        return;
      }
      const tailSize = await sizeOf(tailPath);
      if (tailSize !== remaining) {
        reject(new Error(`part ${part.index} tail has ${tailSize} bytes; expected ${remaining}`));
        return;
      }
      await pipeline(createReadStream(tailPath), createWriteStream(temporaryPath, { flags: "a" }));
      await rm(tailPath, { force: true });
      const completeSize = await sizeOf(temporaryPath);
      if (completeSize !== part.size) {
        reject(new Error(`part ${part.index} has ${completeSize} bytes; expected ${part.size}`));
        return;
      }
      await rename(temporaryPath, part.path);
      resolve();
    });
  });
}

async function downloadParts(parts) {
  let cursor = 0;
  async function worker() {
    while (cursor < parts.length) {
      const part = parts[cursor];
      cursor += 1;
      if (await sizeOf(part.path) === part.size) continue;
      await rm(part.path, { force: true });
      await runCurl(part);
    }
  }
  await Promise.all(Array.from({ length: CONCURRENCY }, worker));
}

async function progress(parts) {
  const bytes = (await Promise.all(parts.flatMap((part) => [
    sizeOf(part.path),
    sizeOf(`${part.path}.partial`),
    sizeOf(`${part.path}.tail`),
  ]))).reduce((sum, value) => sum + value, 0);
  console.log(`downloaded ${(bytes / 1e9).toFixed(2)} / ${(TOTAL_BYTES / 1e9).toFixed(2)} GB (${(bytes / TOTAL_BYTES * 100).toFixed(1)}%)`);
}

async function md5(filePath) {
  const hash = createHash("md5");
  for await (const chunk of createReadStream(filePath)) hash.update(chunk);
  return hash.digest("hex");
}

async function main() {
  await mkdir(partDir, { recursive: true });
  if (await sizeOf(outputPath) === TOTAL_BYTES && await md5(outputPath) === EXPECTED_MD5) {
    console.log("LARa OMoCap archive is already complete and verified.");
    return;
  }
  await rm(outputPath, { force: true });
  const parts = Array.from({ length: PART_COUNT }, (_, index) => partSpec(index));
  const timer = setInterval(() => void progress(parts), 10_000);
  try {
    await downloadParts(parts);
  } finally {
    clearInterval(timer);
  }
  await progress(parts);
  for (const part of parts) {
    await pipeline(createReadStream(part.path), createWriteStream(outputPath, { flags: part.index === 0 ? "w" : "a" }));
  }
  const finalSize = await sizeOf(outputPath);
  const checksum = await md5(outputPath);
  if (finalSize !== TOTAL_BYTES || checksum !== EXPECTED_MD5) {
    throw new Error(`archive verification failed: size=${finalSize}, md5=${checksum}`);
  }
  for (const part of parts) await rm(part.path, { force: true });
  await rm(partDir, { recursive: true, force: true });
  const handle = await open(path.join(outputDir, "OMoCap data.md5"), "w");
  await handle.writeFile(`${EXPECTED_MD5}  OMoCap data.zip\n`);
  await handle.close();
  console.log(`verified ${outputPath} (${finalSize} bytes, md5 ${checksum})`);
}

await main();
