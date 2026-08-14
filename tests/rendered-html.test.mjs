import assert from "node:assert/strict";
import test from "node:test";

// This used to check the vinext starter template's placeholder loading
// skeleton. That skeleton was replaced by the real "메모리 가드" app in
// app/page.tsx, so these checks were rewritten to assert against the real
// server-rendered output instead (they were previously failing outright -
// see the product review notes for context).

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Memory Guard app shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>메모리 가드 \| 매장 안전 스마트 케어<\/title>/);
  assert.match(html, /class="app-shell"/);
});

test("renders all four primary nav tabs", async () => {
  const response = await render();
  const html = await response.text();

  for (const label of ["오늘", "타임라인", "스마트 마감", "케어 기록"]) {
    assert.ok(
      html.includes(label),
      `expected nav label "${label}" to be present in server-rendered HTML`,
    );
  }
});

test("renders the seeded timeline events on first load", async () => {
  const response = await render();
  const html = await response.text();

  for (const title of [
    "손님이 나가셨어요",
    "결제가 완료됐어요",
    "예약을 등록했어요",
    "매장 문을 열었어요",
  ]) {
    assert.ok(
      html.includes(title),
      `expected seeded timeline event "${title}" to be present`,
    );
  }
});

test("does not disclose camera video/audio off-device (privacy copy present)", async () => {
  const response = await render();
  const html = await response.text();

  assert.ok(
    html.includes("얼굴 특징") && html.includes("이 브라우저에"),
    "expected the local-only storage disclosure copy to render on first load",
  );
});
