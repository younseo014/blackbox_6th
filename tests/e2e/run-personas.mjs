// Persona-driven browser tests for 메모리 가드.
//
// These need a real browser (camera permission flows, IndexedDB, computed
// CSS) so they're kept separate from the fast unit tests. Run with:
//   npm run test:e2e
// First run needs the Chromium browser binary once:
//   npx playwright install chromium
//
// Each `test(...)` below is named after the persona/scenario it checks, per
// the product review's test-case notes. A couple of personas (genuinely
// low-end consumer hardware, real GPU-less drivers) can't be fully
// reproduced in a headless sandbox - those are approximated, with a comment
// explaining the gap.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { chromium } from "playwright";

const PORT = Number(process.env.E2E_PORT ?? 8799);
const BASE_URL = `http://localhost:${PORT}`;
const repoRoot = new URL("../../", import.meta.url);

let serverProcess;
let browser;

async function waitForServer(url, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // server not up yet
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  throw new Error(`Server did not become ready at ${url} within ${timeoutMs}ms`);
}

before(async () => {
  serverProcess = spawn("npm", ["run", "start", "--", "--port", String(PORT)], {
    cwd: repoRoot,
    stdio: "ignore",
    env: process.env,
  });
  await waitForServer(BASE_URL);

  browser = await chromium.launch({
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH || undefined,
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
  });
});

after(async () => {
  await browser?.close();
  if (serverProcess) serverProcess.kill();
});

async function withPage(options, run) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    ...options,
  });
  const page = await context.newPage();
  // 25s used to be comfortable margin, but the app now does noticeably more
  // per-frame work while the camera/pose loop is running (occupation-aware
  // baselines, personalized skeleton proportions, etc.), and this sandbox's
  // software-rendered WebGL fallback (no real GPU) makes that contention
  // worse than it would be on real hardware. A slow click here is a real,
  // reproducible timing margin issue - not a broken feature (confirmed by
  // manually clicking with a much longer timeout, which always succeeds) -
  // so the fix is more headroom, not retrying or skipping the assertion.
  page.setDefaultTimeout(45000);
  try {
    await run(page, context);
  } finally {
    // Always release the context (and its camera stream / WASM instance)
    // so a failure in one persona test doesn't slow down or destabilize
    // the ones that run after it.
    await context.close().catch(() => undefined);
  }
}

async function acceptObservationConsent(page) {
  await page.getByRole("button", { name: "카메라 연결", exact: true }).click();
  await page.getByRole("button", { name: "동의하고 카메라 켜기", exact: true }).click();
  await page.waitForTimeout(3000);
}

// --- Persona: 카메라 권한을 거부한 사용자 -----------------------------

test("페르소나: 카메라 권한 거부 - 에러 메시지가 뜨고 다른 탭은 계속 동작한다", async () => {
  await withPage({}, async (page) => {
    // Simulate the browser permission prompt being denied, regardless of
    // the fake-media flags used for the other tests.
    await page.addInitScript(() => {
      navigator.mediaDevices.getUserMedia = () =>
        Promise.reject(new DOMException("Permission denied", "NotAllowedError"));
    });
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });

    await page.getByRole("button", { name: "카메라 연결", exact: true }).click();
    await page.getByRole("button", { name: "동의하고 카메라 켜기", exact: true }).click();
    await page.waitForTimeout(1000);

    await assert.doesNotReject(
      page
        .getByText("카메라 권한을 허용한 뒤 다시 연결해 주세요", { exact: false })
        .first()
        .waitFor({ timeout: 5000 }),
      "expected the camera-permission-denied message to be shown",
    );

    // Store-safety features must keep working with no camera at all.
    await page.getByRole("button", { name: "스마트 마감", exact: false }).first().click();
    await assert.doesNotReject(
      page.getByRole("button", { name: "퇴근 전 자동 점검", exact: true }).waitFor(),
      "스마트 마감 should stay usable without a camera",
    );

    await page.getByRole("button", { name: "케어 기록", exact: false }).first().click();
    await assert.doesNotReject(
      page.getByText("가상 페르소나의 일주일 관찰 결과").waitFor(),
      "케어 기록 (demo) should stay usable without a camera",
    );
  });
});

// --- Persona: 관찰에 동의하지 않고 매장 안전 기능만 쓰는 사용자 -------

test("페르소나: 관찰 비동의 - 카메라는 꺼진 채로 남고 케어 리포트는 참여 유도 상태를 보여준다", async () => {
  await withPage({}, async (page) => {
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });

    await page.getByRole("button", { name: "카메라 연결", exact: true }).click();
    await page.getByRole("button", { name: "매장 안전 기능만 사용할게요", exact: true }).click();
    await page.waitForTimeout(500);

    await assert.doesNotReject(
      page.getByRole("button", { name: "카메라 연결", exact: true }).waitFor(),
      "camera should remain off (still showing the connect button) after declining",
    );

    await page.getByRole("button", { name: "케어 기록", exact: false }).first().click();
    await page.locator(".demo-switcher button").click();
    await assert.doesNotReject(
      page.getByText("아직 장기 관찰에 참여하고 있지 않아요").waitFor(),
      "declining observation consent should show the opt-in empty state, not fabricated numbers",
    );
  });
});

// --- Persona: 아주 바쁜 성수기 매장, 마감 반복 확인 (혼란 변수 인지) ---

test("페르소나: 바쁜 날 첫 사용 - 기준 데이터가 없을 땐 비교 없이 안내만 하고, 안전/반복확인 지표는 실제 조작대로 기록된다", async () => {
  await withPage({}, async (page, context) => {
    await context.grantPermissions(["camera"], { origin: BASE_URL });
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
    await acceptObservationConsent(page);

    await page.getByRole("button", { name: "스마트 마감", exact: false }).first().click();
    await page.getByRole("button", { name: "바쁨", exact: true }).click();
    await page.getByRole("button", { name: "퇴근 전 자동 점검", exact: true }).click();
    await page.waitForTimeout(2200);
    const heaterOff = page.getByRole("button", { name: "전원 끄기", exact: true });
    if (await heaterOff.count()) await heaterOff.click();
    await page.getByRole("button", { name: "퇴근 전 자동 점검", exact: true }).click();
    await page.waitForTimeout(2200);
    // Re-run once more while already done today -> counts as a double-check.
    await page.getByRole("button", { name: "마감 완료 · 다시 확인하기", exact: true }).click();
    await page.waitForTimeout(2200);

    await page.getByRole("button", { name: "케어 기록", exact: false }).first().click();
    await page.locator(".demo-switcher button").click();
    await page.waitForTimeout(500);

    // On day one there isn't enough history for a fair personal baseline
    // yet, so the app should say so plainly instead of fabricating a
    // "평소보다 늘었어요" comparison against nothing.
    await assert.doesNotReject(
      page.getByText("아직 평소 기준을 만들 만큼", { exact: false }).waitFor({ timeout: 8000 }),
      "expected the 'not enough baseline yet' notice on a fresh profile",
    );

    const metricCards = page.locator(".metric-card strong");
    const safetyAlerts = Number.parseInt((await metricCards.nth(0).innerText()).trim(), 10);
    const doubleChecks = Number.parseInt((await metricCards.nth(1).innerText()).trim(), 10);
    assert.ok(safetyAlerts >= 1, `expected at least 1 recorded safety alert, got ${safetyAlerts}`);
    assert.ok(doubleChecks >= 1, `expected at least 1 recorded double-check, got ${doubleChecks}`);
  });
});

// --- Persona: 장기간 여러 날 사용하는 사용자 (세션 누적) ----------------

test("페르소나: 여러 세션 누적 - 카메라를 두 번 연결하면 세션 수가 늘어난다", async () => {
  await withPage({}, async (page, context) => {
    await context.grantPermissions(["camera"], { origin: BASE_URL });
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });

    await acceptObservationConsent(page);
    await page.getByRole("button", { name: "카메라 끄기", exact: true }).click();
    await page.waitForTimeout(500);

    await page.getByRole("button", { name: "카메라 연결", exact: true }).click();
    await page.waitForTimeout(3000);
    await page.getByRole("button", { name: "카메라 끄기", exact: true }).click();
    await page.waitForTimeout(500);

    await page.getByRole("button", { name: "내 데이터 관리", exact: false }).click();
    const sessionCountText = await page
      .locator(".my-data-grid article")
      .first()
      .locator("strong")
      .innerText();
    const sessionCount = Number.parseInt(sessionCountText, 10);
    assert.ok(sessionCount >= 2, `expected at least 2 accumulated sessions, got "${sessionCountText}"`);
  });
});

// --- Persona: 개인정보에 민감한 사용자 (데이터 삭제) --------------------

test("페르소나: 전체 삭제 - 내 데이터 전체 삭제 후 세션/케어 기록이 0으로 초기화된다", async () => {
  await withPage({}, async (page, context) => {
    await context.grantPermissions(["camera"], { origin: BASE_URL });
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
    await acceptObservationConsent(page);

    await page.getByRole("button", { name: "내 데이터 관리", exact: false }).click();
    await page.getByRole("button", { name: "내 데이터 전체 삭제", exact: true }).click();
    await page.waitForTimeout(500);

    const sessionCountText = await page
      .locator(".my-data-grid article")
      .first()
      .locator("strong")
      .innerText();
    assert.equal(sessionCountText.trim(), "0개");
  });
});

// --- Persona: 디지털 기기에 익숙하지 않은 58세 사용자 (접근성) ----------

test("페르소나: 저시력/고령 사용자 - 주요 제목과 버튼 글자 크기가 충분히 크다", async () => {
  await withPage({}, async (page) => {
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });

    const headingSize = await page
      .locator("h1")
      .first()
      .evaluate((el) => Number.parseFloat(getComputedStyle(el).fontSize));
    const navLabelSize = await page
      .locator(".main-nav button")
      .first()
      .evaluate((el) => Number.parseFloat(getComputedStyle(el).fontSize));

    assert.ok(headingSize >= 20, `main heading font-size too small for older users: ${headingSize}px`);
    assert.ok(navLabelSize >= 13, `nav label font-size too small for older users: ${navLabelSize}px`);
  });
});

// --- 시뮬레이션 페르소나 전환 - 4개 페르소나가 서로 다른 신호를 보여준다 ---

test("시뮬레이션: 페르소나를 전환하면 신호 배지와 요일 데이터가 함께 바뀐다", async () => {
  await withPage({}, async (page) => {
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
    await page.getByRole("button", { name: "케어 기록", exact: false }).first().click();
    await page.waitForTimeout(300);

    const personaTabs = page.locator(".persona-switcher button");
    await assert.doesNotReject(
      personaTabs.first().waitFor(),
      "expected the persona switcher to render at least one persona tab",
    );
    const personaCount = await personaTabs.count();
    assert.ok(personaCount >= 4, `expected at least 4 demo personas, found ${personaCount}`);

    const seenLevels = new Set();
    for (let i = 0; i < personaCount; i += 1) {
      await personaTabs.nth(i).click();
      await page.waitForTimeout(150);
      const pillClass = await page.locator(".signal-result .signal-pill").getAttribute("class");
      const level = pillClass?.match(/level-(\w+)/)?.[1];
      assert.ok(level, `expected a level-* class on the signal pill for persona index ${i}`);
      seenLevels.add(level);

      // The signal reasons list should never use a diagnostic label.
      const reasonsText = await page.locator(".signal-result ul").innerText();
      assert.doesNotMatch(reasonsText, /치매|진단|질환/);
    }

    assert.ok(
      seenLevels.size >= 2,
      `expected the 4 demo personas to cover more than one outcome level, saw: ${[...seenLevels].join(", ")}`,
    );
  });
});

// --- Persona: GPU 가속을 지원하지 않는 저사양 기기 (근사치) -------------
// A full low-end-hardware reproduction isn't possible in this sandbox; this
// approximates it by disabling GPU acceleration in the browser and checking
// the app's existing GPU->CPU delegate fallback (see page.tsx
// ensureMotionLandmarkers) doesn't crash the page.

// --- 개발용 세션 리플레이 - 기록된 스켈레톤 좌표를 재생해서 확인한다 ------

test("개발용 세션 리플레이: 기록된 세션을 재생하면 스켈레톤과 재생 컨트롤이 나타난다", async () => {
  await withPage({}, async (page, context) => {
    await context.grantPermissions(["camera"], { origin: BASE_URL });
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
    await acceptObservationConsent(page);
    // Let a few seconds of pose frames accumulate before disconnecting -
    // stopping the camera flushes any buffered frames into a chunk, so the
    // session becomes replayable right after.
    await page.waitForTimeout(3000);
    await page.getByRole("button", { name: "카메라 끄기", exact: true }).click();
    await page.waitForTimeout(1000);

    await page.getByRole("button", { name: "내 데이터 관리", exact: false }).click();
    const replayButtons = page.locator(".my-data-sessions li button:not([disabled])");
    await assert.doesNotReject(
      replayButtons.first().waitFor({ timeout: 8000 }),
      "expected at least one replayable recorded session to be listed",
    );
    await replayButtons.first().click();

    await assert.doesNotReject(
      page.locator(".replay-modal").waitFor({ timeout: 8000 }),
      "expected the replay modal to open",
    );
    await assert.doesNotReject(
      page.locator(".replay-canvas-wrap canvas").waitFor(),
      "expected the replay skeleton canvas to render",
    );
    await assert.doesNotReject(
      page.locator(".replay-scrubber").waitFor(),
      "expected a scrubber control for stepping through frames",
    );

    // The frame-by-frame stats (including the frame counter) live inside a
    // collapsed "분석 상세 정보 보기" <details> panel in the redesigned replay
    // UI - it must be expanded before its text is actually visible/readable.
    await page.locator(".analysis-details summary").click();

    // Scrub to a specific frame and confirm the frame counter follows it -
    // this is the core "let me see exactly what motion happened here" flow.
    const scrubber = page.locator(".replay-scrubber");
    const maxFrame = await scrubber.getAttribute("max");
    if (maxFrame && Number(maxFrame) > 0) {
      await scrubber.fill(maxFrame);
      await page.waitForTimeout(200);
      const frameCounterText = await page.locator(".replay-stats article").nth(2).locator("strong").innerText();
      assert.match(frameCounterText.trim(), /^\d+ \/ \d+$/);
    }

    await page.locator(".replay-modal .modal-close").click();
    await assert.doesNotReject(
      page.locator(".replay-modal").waitFor({ state: "detached", timeout: 5000 }),
      "expected the replay modal to close",
    );
  });
});

// --- 개발용 합성 시나리오 리플레이 - 가상 페르소나 이벤트의 예시 동작을 확인한다 ---

test("개발용 합성 시나리오 리플레이: 페르소나의 '동작 보기'를 누르면 합성 스켈레톤과 실제 기록이 아니라는 안내가 뜬다", async () => {
  await withPage({}, async (page) => {
    await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
    await page.getByRole("button", { name: "케어 기록", exact: false }).first().click();
    await page.waitForTimeout(300);

    // The default persona/day already has at least one example event, so no
    // extra tab navigation is needed before this button is present.
    const eventReplayButtons = page.locator(".demo-event-replay-button");
    await assert.doesNotReject(
      eventReplayButtons.first().waitFor({ timeout: 8000 }),
      "expected at least one demo event with a '동작 보기' replay button",
    );
    await eventReplayButtons.first().click();

    await assert.doesNotReject(
      page.locator(".replay-modal").waitFor({ timeout: 8000 }),
      "expected the synthetic replay modal to open",
    );
    await assert.doesNotReject(
      page.locator(".replay-canvas-wrap canvas").waitFor(),
      "expected the replay skeleton canvas to render",
    );

    // This is the critical distinction the user asked for: a synthetic demo
    // clip must never be mistaken for a real recorded session.
    const chipText = await page.locator(".replay-source-chip.synthetic").innerText();
    assert.match(chipText, /가상 시나리오/);

    // The default persona/day's first example is a routine ("normal_task")
    // event, so the detection-basis rule (rendered inside the collapsed
    // "분석 상세 정보 보기" panel) should honestly say it isn't a counted
    // signal rather than inventing a fake reason.
    await page.locator(".analysis-details summary").click();
    const routineRuleText = await page.locator(".technical-evidence").last().innerText();
    assert.match(routineRuleText, /집계되지 않는 일반 업무 장면이에요/);

    await assert.doesNotReject(
      page.locator(".replay-scrubber").waitFor(),
      "expected a scrubber control for stepping through the synthetic clip",
    );

    await page.locator(".replay-modal .modal-close").click();
    await assert.doesNotReject(
      page.locator(".replay-modal").waitFor({ state: "detached", timeout: 5000 }),
      "expected the replay modal to close",
    );

    // Now check a COUNTED event (마감 반복 확인) shows the real rule plus a
    // baseline comparison, not just "not counted".
    const personaTabs = page.locator(".persona-switcher button");
    await personaTabs.nth(1).click(); // "decline" persona - has counted events
    await page.waitForTimeout(150);
    const weekDayButtons = page.locator(".week-days button");
    await weekDayButtons.nth(1).click(); // 화 - has a double_check example
    await page.waitForTimeout(150);

    const doubleCheckButton = page
      .locator(".day-examples li")
      .filter({ hasText: "출입문 상태 재확인" })
      .locator(".demo-event-replay-button");
    await doubleCheckButton.click();

    await assert.doesNotReject(
      page.locator(".replay-modal").waitFor({ timeout: 8000 }),
      "expected the replay modal to open for a counted event",
    );
    // The baseline comparison line renders directly in the analysis panel
    // (not behind the collapsed details), so it's checked first.
    const metricText = await page.locator(".analysis-message small").innerText();
    assert.match(metricText, /평소 하루 평균/);
    await page.locator(".analysis-details summary").click();
    const countedRuleText = await page.locator(".technical-evidence").last().innerText();
    assert.match(countedRuleText, /마감 반복 확인/);

    await page.locator(".replay-modal .modal-close").click();
    await assert.doesNotReject(
      page.locator(".replay-modal").waitFor({ state: "detached", timeout: 5000 }),
      "expected the replay modal to close",
    );
  });
});

test("페르소나: GPU 미가속 환경 - GPU 비활성화 상태에서도 카메라 연결이 에러 없이 진행된다", async () => {
  const gpuDisabledBrowser = await chromium.launch({
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH || undefined,
    args: [
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
      "--disable-gpu",
      "--disable-software-rasterizer",
    ],
  });
  try {
    const context = await gpuDisabledBrowser.newContext({ viewport: { width: 1440, height: 900 } });
    try {
      await context.grantPermissions(["camera"], { origin: BASE_URL });
      const page = await context.newPage();
      // 25s used to be comfortable margin, but the app now does noticeably more
  // per-frame work while the camera/pose loop is running (occupation-aware
  // baselines, personalized skeleton proportions, etc.), and this sandbox's
  // software-rendered WebGL fallback (no real GPU) makes that contention
  // worse than it would be on real hardware. A slow click here is a real,
  // reproducible timing margin issue - not a broken feature (confirmed by
  // manually clicking with a much longer timeout, which always succeeds) -
  // so the fix is more headroom, not retrying or skipping the assertion.
  page.setDefaultTimeout(45000);
      const pageErrors = [];
      page.on("pageerror", (err) => pageErrors.push(err.message));

      await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
      await acceptObservationConsent(page);
      await page.waitForTimeout(5000);

      assert.deepEqual(pageErrors, [], `expected no uncaught page errors, got: ${pageErrors.join(", ")}`);
    } finally {
      await context.close().catch(() => undefined);
    }
  } finally {
    await gpuDisabledBrowser.close();
  }
});
