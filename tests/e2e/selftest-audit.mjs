import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const port = Number(process.env.E2E_PORT ?? 8803);
const origin = `http://localhost:${port}`;
let server, browser;
const moduleUrl = (file) => `/@fs${path.join(root, 'app', file)}`;
before(async () => {
  server = spawn('npx', ['vite', '--config', 'vite.desktop.config.ts', '--port', String(port)], { cwd: root, stdio: 'ignore' });
  for (let attempt = 0; attempt < 100; attempt++) {
    if (await fetch(origin).then(r => r.ok).catch(() => false)) break;
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH || undefined,
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
});
after(async () => { await browser?.close(); server?.kill(); });

async function openTest() {
  const context = await browser.newContext({ reducedMotion: "reduce" });
  await context.grantPermissions(['camera'], { origin });
  const page = await context.newPage();
  page.on('pageerror', error => console.error('Page error:', error.message));
  page.on('console', message => { if (message.type() === 'error') console.error('Browser error:', message.text()); });
  page.on('requestfailed', request => console.error('Request failed:', request.url(), request.failure()?.errorText));
  page.setDefaultTimeout(20000);
  await page.goto(origin);
  await page.getByRole('button', { name: '설정', exact: true }).click();
  await page.getByRole('button', { name: /개발자 모드 카메라·동작 분석/ }).click();
  await page.getByRole('button', { name: '자체테스트', exact: true }).click();
  return { page, context };
}

test('the bundled action model runs in a local worker and returns finite context scores', async () => {
  const { page, context } = await openTest();
  try {
    const origins = new Set();
    context.on('request', request => { if (request.url().startsWith('http')) origins.add(new URL(request.url()).origin); });
    const result = await page.evaluate(async url => {
      const { LocalActionModel } = await import(url);
      const started = performance.now();
      const model = await LocalActionModel.create();
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = 224;
      canvas.getContext('2d').fillRect(0, 0, 224, 224);
      const video = document.createElement('video'); video.muted = true;
      const stream = canvas.captureStream(10); video.srcObject = stream; await video.play();
      try {
        const inferenceStart = performance.now();
        const result = await model.classify(video);
        return { ...result, inferenceMs: performance.now() - inferenceStart, totalMs: performance.now() - started };
      } finally { model.close(); stream.getTracks().forEach(t => t.stop()); video.srcObject = null; }
    }, moduleUrl('local-action-model.ts'));
    assert.ok(Number.isFinite(result.similarity) && Number.isFinite(result.margin));
    assert.ok(['task', 'phone', 'interaction', 'other'].includes(result.kind));
    assert.ok(result.label.length > 0);
    assert.deepEqual([...origins], [origin]);
    console.log('Local model smoke timing:', JSON.stringify({ inferenceMs: result.inferenceMs, totalMs: result.totalMs }));
  } finally { await context.close(); }
});

test('self-test setup fits a narrow mobile screen without horizontal scrolling', async () => {
  const { page, context } = await openTest();
  try {
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    await page.getByRole('heading', { name: '업무 동작 자체테스트', exact: true }).waitFor();
    await page.screenshot({ path: '/private/tmp/mg-audit-selftest-mobile.png', fullPage: true });
  } finally { await context.close(); }
});

test('legacy chunks survive the ordered-read migration and window reads skip unrelated compressed chunks', async () => {
  const { page, context } = await openTest();
  try {
    const result = await page.evaluate(async (url) => {
      await new Promise((resolve, reject) => {
        const request = indexedDB.deleteDatabase('memory-guard-motion-v2');
        request.onsuccess = resolve; request.onerror = () => reject(request.error);
      });
      const oldDb = await new Promise((resolve, reject) => {
        const request = indexedDB.open('memory-guard-motion-v2', 2);
        request.onupgradeneeded = () => {
          request.result.createObjectStore('motion_sessions', { keyPath: 'id' });
          const store = request.result.createObjectStore('motion_chunks', { keyPath: 'id' });
          store.createIndex('sessionId', 'sessionId');
        };
        request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
      });
      const tx = oldDb.transaction('motion_chunks', 'readwrite');
      for (const start of [20, 0, 10]) {
        const data = new Float32Array(224 * 10);
        for (let i = 0; i < 10; i++) data[i * 224] = (start + i) * 100;
        tx.objectStore('motion_chunks').put({ id: `legacy:${start}`, sessionId: 'legacy', startFrame: start,
          frameCount: 10, createdAt: 1, data: data.buffer });
      }
      await new Promise((resolve, reject) => { tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); });
      oldDb.close();
      const pose = await import(url);
      const legacyTimes = (await pose.getSessionFrames('legacy')).map(f => f[0]);
      let session = await pose.createMotionSession('windowed', Date.now());
      for (const start of [0, 10, 20]) {
        const data = new Float32Array(224 * 10);
        for (let i = 0; i < 10; i++) data[i * 224] = (start + i) * 100;
        session = await pose.appendMotionChunk(session, start, data, 0, 0, 0);
      }
      // A corrupt chunk outside the requested interval must not be decoded.
      const db = await pose.openMotionDatabase();
      const broken = db.transaction('motion_chunks', 'readwrite');
      broken.objectStore('motion_chunks').put({ id: 'windowed:0', sessionId: 'windowed', startFrame: 0,
        frameCount: 10, startMs: 0, endMs: 900, codec: 'q16-gzip-v1', data: new ArrayBuffer(0) });
      await new Promise((resolve, reject) => { broken.oncomplete = resolve; broken.onerror = () => reject(broken.error); });
      db.close();
      const windowTimes = (await pose.getSessionFrames('windowed', { startMs: 1200, endMs: 1700 })).map(f => f[0]);
      let corruptionReported = false;
      try { await pose.getSessionFrames('windowed'); } catch { corruptionReported = true; }
      return { legacyTimes, windowTimes, corruptionReported };
    }, moduleUrl('pose-store.ts'));
    assert.deepEqual(result.legacyTimes, Array.from({ length: 30 }, (_, i) => i * 100));
    assert.deepEqual(result.windowTimes, [1200, 1300, 1400, 1500, 1600, 1700]);
    assert.equal(result.corruptionReported, true);
  } finally { await context.close(); }
});

test('fresh install can reach self-test; preview saves no pose frames; capture records its own one-camera context', async () => {
  const { page, context } = await openTest();
  try {
    await page.getByPlaceholder('예: 참가자 A').fill('preview-test');
    const discover = page.getByRole('button', { name: '카메라 찾기', exact: true });
    if (await discover.isVisible()) await discover.click();
    await page.getByRole('button', { name: '카메라 연결', exact: true }).click();
    await page.getByRole('button', { name: '동의하고 카메라 켜기', exact: true }).click();
    await page.getByRole('button', { name: '학습 촬영 시작', exact: true }).waitFor({ timeout: 45000 });
    await page.waitForFunction(() => [...document.querySelectorAll('button')].some(b => b.textContent === '학습 촬영 시작' && !b.disabled), { timeout: 45000 });
    const sessions = await page.evaluate(async (url) => (await import(url)).listMotionSessions(), moduleUrl('pose-store.ts'));
    assert.equal(sessions.length, 1);
    assert.equal(sessions[0].frameCount, 0);
    assert.equal(await page.getByRole('combobox').count(), 1);
    assert.equal(await page.getByRole('button', {name:'구역 5',exact:true}).count(),0);
    await page.getByRole('heading',{name:'음료 제조대 · 고정 시험 조건',exact:true}).waitFor();
    await page.getByRole('button', { name: '학습 촬영 시작', exact: true }).click();
    await page.getByRole('button', { name: '촬영 종료·분석', exact: true }).waitFor();
    const manifest = await page.evaluate(async ({url,id}) => (await import(url)).getCaptureManifest(id), {url:moduleUrl('pose-store.ts'),id:sessions[0].globalSessionId});
    assert.deepEqual(manifest.expectedCameraSlots, [1]);
    assert.equal(manifest.purpose, 'self_test');
    assert.equal(manifest.multiViewCapture, undefined);
    assert.deepEqual(manifest.zoneSnapshot.primary, Array(9).fill('DRINK_PREP'));
    assert.equal(manifest.selfTest.protocol,'workbench-v2');
    assert.equal(manifest.selfTest.fixedContext,'DRINK_PREP');
    assert.equal(manifest.selfTest.participant, 'preview-test');
    assert.ok(await page.getByRole('combobox').last().isDisabled());
    await page.getByRole('button', { name: '촬영 종료·분석', exact: true }).click();
    await page.getByRole('button', { name: '카메라 연결', exact: true }).waitFor();
  } catch (error) { console.error(error); await page.screenshot({path: `/private/tmp/mg-audit-failed-${port}.png`, fullPage: true}); throw error; } finally { await context.close(); }
});

test('all 601 retained results are exported rather than only the latest 500', async () => {
  const { page, context } = await openTest();
  try {
    await page.evaluate(async ({engineUrl,storeUrl}) => {
      const engine=await import(engineUrl), store=await import(storeUrl);
      for(let i=0;i<601;i++) {
        const episode=engine.createObservationEpisode({sessionId:`archive-${i}`,recordedAt:Date.now()+i,
          profile:engine.DEFAULT_PROFILE,phase:'business',features:engine.extractObservationFeatures([],Array(9).fill(null)),baseline:engine.buildBaseline([])});
        await store.saveObservationEpisode({...episode,source:'self_test_training',selfTestParticipant:'archive-test'});
      }
      if((await store.listObservationEpisodes()).length!==601)throw new Error('truncated episode query');
    }, {engineUrl:moduleUrl('observation-engine.ts'),storeUrl:moduleUrl('observation-store.ts')});
    await page.reload();
    await page.getByRole('button', {name:'자체테스트',exact:true}).click();
    await page.getByPlaceholder('예: 참가자 A').fill('archive-test');
    await page.getByText('archive-test · 학습 결과 601건',{exact:true}).waitFor();
    const downloaded=page.waitForEvent('download');
    await page.getByRole('button',{name:'전체 결과 CSV 내보내기',exact:true}).click();
    const download=await downloaded; const stream=await download.createReadStream();let csv='';
    for await(const chunk of stream)csv+=chunk.toString('utf8');
    assert.equal(csv.trim().split('\r\n').length,602);
    assert.ok(csv.includes('archive-0"'));
    assert.ok(csv.includes('archive-600"'));
  } catch (error) { console.error(error); await page.screenshot({path: `/private/tmp/mg-audit-failed-${port}.png`, fullPage: true}); throw error; } finally { await context.close(); }
});

test('a confirmed label and interval can be corrected, then withdrawn without leaving stale training samples', async () => {
  const { page, context } = await openTest();
  try {
    await page.evaluate(async (urls) => {
      const pose=await import(urls.pose), demo=await import(urls.demo), engine=await import(urls.engine), store=await import(urls.store),review=await import(urls.review);
      const frames=demo.generateEventMotion('normal_task',1);
      let session=await pose.createMotionSession('review-session',Date.now());
      session=await pose.appendMotionChunk(session,0,Float32Array.from(frames.flat()),frames.length,frames.length,frames.length);
      await pose.finishMotionSession(session,Date.now());
      const endMs=frames.at(-1)[0];
      const episode={...engine.createObservationEpisode({sessionId:session.id,recordedAt:session.startedAt,profile:engine.DEFAULT_PROFILE,phase:'business',features:engine.extractObservationFeatures(frames,Array(9).fill('DRINK_PREP')),baseline:engine.buildBaseline([])}),id:'review-episode',source:'self_test_training',selfTestParticipant:'review-test',motionSlice:{startMs:0,endMs,durationSeconds:endMs/1000,originalDurationSeconds:endMs/1000,excludedFrameCount:0,reason:'test'}};
      await store.saveObservationEpisode(review.applyManualActionLabel(episode,{taskType:'WORK_CONTEXT:테이블 닦기',taskLabel:'테이블 닦기'}));
      localStorage.setItem('memory-guard-self-test-learned-v1',JSON.stringify({'review-test':[{id:'old',label:'테이블 닦기',createdAt:1,samples:[{sessionId:session.id,startMs:0,endMs}]}]}));
    },{pose:moduleUrl('pose-store.ts'),demo:moduleUrl('demo-motion.ts'),engine:moduleUrl('observation-engine.ts'),store:moduleUrl('observation-store.ts'),review:moduleUrl('action-review.ts')});
    await page.reload();await page.getByRole('button',{name:'자체테스트',exact:true}).click();
    await page.getByPlaceholder('예: 참가자 A').fill('review-test');
    await page.getByRole('button',{name:'스켈레톤 검토',exact:true}).click();
    await page.getByRole('spinbutton',{name:'시작',exact:true}).fill('0.1');
    await page.getByRole('spinbutton',{name:'종료',exact:true}).fill('1');
    await page.getByRole('radio',{name:'음료 재료 혼합',exact:true}).check();
    await page.getByRole('button',{name:'라벨 확정',exact:true}).click();
    await page.getByRole('button',{name:'스켈레톤 검토',exact:true}).waitFor();
    await page.waitForFunction(() => JSON.parse(localStorage.getItem('memory-guard-self-test-learned-v1'))?.['review-test']?.[0]?.label === '음료 재료 혼합');
    const learned=await page.evaluate(()=>JSON.parse(localStorage.getItem('memory-guard-self-test-learned-v1'))['review-test']);
    assert.equal(learned.length,1);assert.equal(learned[0].label,'음료 재료 혼합');
    assert.equal(learned[0].samples[0].startMs,100);assert.equal(learned[0].samples[0].endMs,1000);
    await page.getByRole('button',{name:'스켈레톤 검토',exact:true}).click();
    await page.getByRole('button',{name:'판별 불가',exact:true}).click();
    await page.getByRole('button',{name:'스켈레톤 검토',exact:true}).waitFor();
    await page.waitForFunction(() => JSON.parse(localStorage.getItem('memory-guard-self-test-learned-v1'))?.['review-test']?.length === 0);
    const remaining=await page.evaluate(()=>JSON.parse(localStorage.getItem('memory-guard-self-test-learned-v1'))['review-test']);
    assert.deepEqual(remaining,[]);
  } catch (error) { console.error(error); await page.screenshot({path: `/private/tmp/mg-audit-failed-${port}.png`, fullPage: true}); throw error; } finally { await context.close(); }
});
