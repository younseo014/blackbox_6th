// Optional, off-by-default sync endpoint for daily care-metric summaries.
//
// Not called anywhere in the current UI. It exists as ready-to-enable
// scaffolding: once D1 hosting is turned on (see README.md, "선택적 서버
// 동기화") a future cross-device sync feature can POST/GET through this
// route instead of leaving everything in browser-local IndexedDB. Until
// then `getDb()` throws and this route responds with a clear "not
// configured" error rather than a stack trace.
// NOTE: db/index.ts statically imports "cloudflare:workers", which only
// resolves inside the real Workers/Miniflare runtime. This route is loaded
// as part of the single worker entry bundle, so importing it at module scope
// here would make "cloudflare:workers" unresolvable for anything that loads
// this bundle outside that runtime (e.g. tests/rendered-html.test.mjs, which
// imports dist/server/index.js directly under plain Node). Importing lazily
// inside each handler keeps that import out of the always-loaded module
// graph and only pulls it in when a request actually hits this route.
async function loadDb() {
  const [{ getDb }, { dailyCareMetrics }] = await Promise.all([
    import("../../../db"),
    import("../../../db/schema"),
  ]);
  return { getDb, dailyCareMetrics };
}

function toRouteErrorMessage(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected error";
  if (message.includes("D1 binding") || message.includes("no such table")) {
    return "서버 동기화가 아직 활성화되지 않았어요. .openai/hosting.json의 d1 설정을 켜고 `npm run db:generate` 후 배포해야 이 기능이 동작해요. 그 전까지는 브라우저 로컬 저장만 사용돼요.";
  }
  return message;
}

type SyncPayload = {
  deviceId?: string;
  logs?: Array<{
    date: string;
    safetyAlerts: number;
    doubleChecks: number;
    tasksStarted: number;
    tasksCompleted: number;
    microDelaySeconds: number[];
    busyLevel: string;
  }>;
};

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const deviceId = url.searchParams.get("deviceId");
    if (!deviceId) {
      return Response.json({ error: "deviceId is required" }, { status: 400 });
    }
    const { getDb, dailyCareMetrics } = await loadDb();
    const db = getDb();
    const rows = await db
      .select()
      .from(dailyCareMetrics)
      .where(eq(dailyCareMetrics.deviceId, deviceId));
    return Response.json({ logs: rows });
  } catch (error) {
    return Response.json({ error: toRouteErrorMessage(error) }, { status: 503 });
  }
}

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as SyncPayload;
    const deviceId = payload.deviceId?.trim();
    const logs = payload.logs ?? [];
    if (!deviceId) {
      return Response.json({ error: "deviceId is required" }, { status: 400 });
    }

    const { getDb, dailyCareMetrics } = await loadDb();
    const db = getDb();
    for (const log of logs) {
      const slowSamples = log.microDelaySeconds.filter((s) => s >= 120).length;
      const existing = await db
        .select({ id: dailyCareMetrics.id })
        .from(dailyCareMetrics)
        .where(
          and(
            eq(dailyCareMetrics.deviceId, deviceId),
            eq(dailyCareMetrics.date, log.date),
          ),
        )
        .limit(1);

      const values = {
        deviceId,
        date: log.date,
        safetyAlerts: log.safetyAlerts,
        doubleChecks: log.doubleChecks,
        tasksStarted: log.tasksStarted,
        tasksCompleted: log.tasksCompleted,
        microDelaySamples: log.microDelaySeconds.length,
        microDelaySlowSamples: slowSamples,
        busyLevel: log.busyLevel,
      };

      if (existing[0]) {
        await db
          .update(dailyCareMetrics)
          .set(values)
          .where(eq(dailyCareMetrics.id, existing[0].id));
      } else {
        await db.insert(dailyCareMetrics).values(values);
      }
    }

    return Response.json({ synced: logs.length }, { status: 200 });
  } catch (error) {
    return Response.json({ error: toRouteErrorMessage(error) }, { status: 503 });
  }
}
