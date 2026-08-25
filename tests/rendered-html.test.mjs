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
  assert.match(html, /class="app-shell interface-user"/);
});

test("renders only the three user-mode navigation tabs", async () => {
  const response = await render();
  const html = await response.text();

  for (const label of ["홈", "기록", "설정"]) {
    assert.ok(
      html.includes(label),
      `expected nav label "${label}" to be present in server-rendered HTML`,
    );
  }
  assert.ok(!html.includes("카메라 기능 테스트"));
  assert.ok(!html.includes("가상 학습 데이터 보기"));
});

test("renders the first-install user home without seeded persona events", async () => {
  const response = await render();
  const html = await response.text();

  assert.ok(html.includes("처음 시작하기"));
  assert.ok(html.includes("초기 설정 시작하기"));
  assert.ok(html.includes("아직 연결된 기록이 없어요"));
  assert.ok(!html.includes("결제가 완료됐어요"));
  assert.ok(!html.includes("가상 학습 데이터 보기"));
});

test("renders a dedicated developer-mode switch", async () => {
  const response = await render();
  const html = await response.text();

  assert.ok(html.includes("사용자 모드"));
  assert.ok(html.includes("개발자 모드<!-- -->로 전환"));
  assert.ok(html.includes('aria-label="개발자 모드로 전환"'));
});
