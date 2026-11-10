import { _electron } from "playwright";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const profile = await mkdtemp(path.join(os.tmpdir(), "memory-guard-action-smoke-"));
const app = await _electron.launch({ args: ["desktop/main.mjs", `--user-data-dir=${profile}`,
  "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
try {
  const page = await app.firstWindow();
  page.setDefaultTimeout(45_000);
  await page.getByRole("button", { name: "설정", exact: true }).click();
  await page.getByRole("button", { name: /개발자 모드 카메라·동작 분석/ }).click();
  await page.getByRole("button", { name: "자체테스트", exact: true }).click();
  await page.getByPlaceholder("예: 참가자 A").fill("desktop-model-smoke");
  const discover = page.getByRole("button", { name: "카메라 찾기", exact: true });
  if (await discover.isVisible()) await discover.click();
  await page.getByRole("button", { name: "카메라 연결", exact: true }).click();
  await page.getByRole("button", { name: "동의하고 카메라 켜기", exact: true }).click();
  await page.getByText("로컬 영상 맥락 모델 준비 완료", { exact: true }).waitFor();
  await page.waitForFunction(() => [...document.querySelectorAll("button")].some(b => b.textContent === "학습 촬영 시작" && !b.disabled));
  await page.getByRole("button", { name: "학습 촬영 시작", exact: true }).click();
  // The fake camera supplies a test pattern. This checks runtime and provenance, not cafe accuracy.
  await page.waitForTimeout(4500);
  await page.getByRole("button", { name: "촬영 종료·분석", exact: true }).evaluate(button => { button.click(); button.click(); });
  await page.getByRole("button", { name: "카메라 연결", exact: true }).waitFor();
  assert.equal(await page.getByText("좌표 기록을 안전하게 저장하고 카메라를 종료했어요", { exact: true }).count(), 1);
  const manifest = await page.evaluate(async () => {
    const db = await new Promise((resolve, reject) => {
      const request = indexedDB.open("memory-guard-motion-v2");
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    const records = await new Promise((resolve, reject) => {
      const request = db.transaction("capture_manifests").objectStore("capture_manifests").getAll();
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    db.close(); return records.at(-1);
  });
  assert.equal(new URL(page.url()).protocol, "memoryguard:");
  assert.equal(manifest.visualContext.model, "clip-vit-base-patch32-int8");
  assert.ok(manifest.visualContext.samples.length >= 1);
  assert.ok(manifest.visualContext.samples.every(s => Number.isFinite(s.similarity) && !s.bodyObserved));
  assert.deepEqual(Object.keys(manifest.visualContext).sort(), ["cameraSlot", "model", "samples"]);
  console.log(`PASS: Electron local model inference; ${manifest.visualContext.samples.length} metadata samples; no images stored`);
} finally { await app.close(); await rm(profile, { recursive: true, force: true }); }
