// Browser-local persistence for the care/observation metrics and for the
// user's consent choice. Mirrors the pattern already used in pose-store.ts
// (IndexedDB, local-only, nothing sent off the device unless the optional
// server sync in app/api/metrics/route.ts is explicitly enabled and used).
import {
  type BusyLevel,
  type DailyLog,
  emptyDailyLog,
} from "./care-metrics";

const DB_NAME = "memory-guard-care-v1";
const DB_VERSION = 1;
const LOG_STORE = "daily_logs";

function requestResult<T>(request: IDBRequest<T>) {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function transactionDone(transaction: IDBTransaction) {
  return new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });
}

function openCareDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(LOG_STORE)) {
        database.createObjectStore(LOG_STORE, { keyPath: "date" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export function todayDateKey(reference = new Date()): string {
  const year = reference.getFullYear();
  const month = `${reference.getMonth() + 1}`.padStart(2, "0");
  const day = `${reference.getDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export async function getLog(date: string): Promise<DailyLog | null> {
  const database = await openCareDatabase();
  const transaction = database.transaction(LOG_STORE, "readonly");
  const record = await requestResult<DailyLog | undefined>(
    transaction.objectStore(LOG_STORE).get(date),
  );
  await transactionDone(transaction);
  database.close();
  return record ?? null;
}

async function saveLog(log: DailyLog): Promise<void> {
  const database = await openCareDatabase();
  const transaction = database.transaction(LOG_STORE, "readwrite");
  transaction.objectStore(LOG_STORE).put(log);
  await transactionDone(transaction);
  database.close();
}

async function getOrCreateTodayLog(): Promise<DailyLog> {
  const date = todayDateKey();
  const existing = await getLog(date);
  return existing ?? emptyDailyLog(date);
}

async function updateToday(mutate: (log: DailyLog) => DailyLog): Promise<DailyLog> {
  const current = await getOrCreateTodayLog();
  const next = mutate(current);
  await saveLog(next);
  return next;
}

export async function recordSafetyAlert(): Promise<DailyLog> {
  return updateToday((log) => ({ ...log, safetyAlerts: log.safetyAlerts + 1 }));
}

export async function recordDoubleCheck(): Promise<DailyLog> {
  return updateToday((log) => ({ ...log, doubleChecks: log.doubleChecks + 1 }));
}

export async function recordTaskStarted(): Promise<DailyLog> {
  return updateToday((log) => ({ ...log, tasksStarted: log.tasksStarted + 1 }));
}

export async function recordTaskCompleted(): Promise<DailyLog> {
  return updateToday((log) => ({
    ...log,
    tasksCompleted: log.tasksCompleted + 1,
  }));
}

export async function recordMicroDelay(seconds: number): Promise<DailyLog> {
  return updateToday((log) => ({
    ...log,
    microDelaySeconds: [...log.microDelaySeconds, Math.max(0, Math.round(seconds))],
  }));
}

export async function setBusyLevel(level: BusyLevel): Promise<DailyLog> {
  return updateToday((log) => ({ ...log, busyLevel: level }));
}

export async function listRecentLogs(days = 28): Promise<DailyLog[]> {
  const database = await openCareDatabase();
  const transaction = database.transaction(LOG_STORE, "readonly");
  const all = await requestResult<DailyLog[]>(
    transaction.objectStore(LOG_STORE).getAll(),
  );
  await transactionDone(transaction);
  database.close();
  return all.sort((a, b) => (a.date < b.date ? 1 : -1)).slice(0, days);
}

export async function deleteAllCareLogs(): Promise<void> {
  const database = await openCareDatabase();
  const transaction = database.transaction(LOG_STORE, "readwrite");
  transaction.objectStore(LOG_STORE).clear();
  await transactionDone(transaction);
  database.close();
}

// --- Consent -----------------------------------------------------------
// Two things are always true regardless of consent:
//  1. Store-safety features (타임라인, 스마트 마감) work with no consent at all.
//  2. Camera-based long-term observation (오늘 탭의 좌표 기록, 케어 기록의 실제
//     지표 계산) only turns on after an explicit, revocable opt-in.

const CONSENT_KEY = "memory-guard-observation-consent-v1";

export type ConsentState = {
  decided: boolean;
  observationConsent: boolean;
  decidedAt: number | null;
};

const DEFAULT_CONSENT: ConsentState = {
  decided: false,
  observationConsent: false,
  decidedAt: null,
};

export function getConsent(): ConsentState {
  if (typeof window === "undefined") return DEFAULT_CONSENT;
  try {
    const raw = window.localStorage.getItem(CONSENT_KEY);
    if (!raw) return DEFAULT_CONSENT;
    const parsed = JSON.parse(raw) as ConsentState;
    if (typeof parsed.observationConsent !== "boolean") return DEFAULT_CONSENT;
    return { ...DEFAULT_CONSENT, ...parsed };
  } catch {
    return DEFAULT_CONSENT;
  }
}

export function setConsent(observationConsent: boolean): ConsentState {
  const next: ConsentState = {
    decided: true,
    observationConsent,
    decidedAt: Date.now(),
  };
  if (typeof window !== "undefined") {
    window.localStorage.setItem(CONSENT_KEY, JSON.stringify(next));
  }
  return next;
}

export function clearConsent(): void {
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(CONSENT_KEY);
  }
}

export async function estimateStorageUsage(): Promise<{
  usageBytes: number;
  quotaBytes: number;
} | null> {
  if (typeof navigator === "undefined" || !navigator.storage?.estimate) {
    return null;
  }
  try {
    const estimate = await navigator.storage.estimate();
    return {
      usageBytes: estimate.usage ?? 0,
      quotaBytes: estimate.quota ?? 0,
    };
  } catch {
    return null;
  }
}
