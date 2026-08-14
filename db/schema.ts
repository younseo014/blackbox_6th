// Optional server-side sync target for daily care-metric summaries.
//
// This table is NOT active by default: `.openai/hosting.json` currently has
// `"d1": null`, so `db/index.ts`'s `getDb()` throws until D1 hosting is
// enabled and `npm run db:generate` + a deploy apply this schema. See
// README.md ("선택적 서버 동기화") before turning this on.
//
// Deliberately narrow: only day-level aggregate counts sync here, never raw
// pose/hand landmark frames (those stay local-only in IndexedDB, per the
// product review's data-minimization recommendation).
import { sql } from "drizzle-orm";
import { integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const dailyCareMetrics = sqliteTable("daily_care_metrics", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  // Per-device pseudonymous id, not a real account/user identity.
  deviceId: text("device_id").notNull(),
  date: text("date").notNull(),
  safetyAlerts: integer("safety_alerts").notNull().default(0),
  doubleChecks: integer("double_checks").notNull().default(0),
  tasksStarted: integer("tasks_started").notNull().default(0),
  tasksCompleted: integer("tasks_completed").notNull().default(0),
  microDelaySamples: integer("micro_delay_samples").notNull().default(0),
  microDelaySlowSamples: integer("micro_delay_slow_samples").notNull().default(0),
  busyLevel: text("busy_level").notNull().default("normal"),
  syncedAt: text("synced_at").notNull().default(sql`CURRENT_TIMESTAMP`),
});
