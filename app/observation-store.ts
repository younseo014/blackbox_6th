import {
  DEFAULT_PROFILE,
  buildBaseline,
  type AnalysisFeedback,
  type BaselineSnapshot,
  type ObservationEpisode,
  type ObservationProfile,
} from "./observation-engine";

const DB_NAME = "memory-guard-observation-v1";
const DB_VERSION = 1;
const PROFILE_STORE = "profiles";
const EPISODE_STORE = "episodes";
const FEEDBACK_STORE = "feedback";

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

function openObservationDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(PROFILE_STORE)) {
        database.createObjectStore(PROFILE_STORE, { keyPath: "id" });
      }
      if (!database.objectStoreNames.contains(EPISODE_STORE)) {
        const store = database.createObjectStore(EPISODE_STORE, { keyPath: "id" });
        store.createIndex("recordedAt", "recordedAt", { unique: false });
        store.createIndex("sessionId", "sessionId", { unique: false });
      }
      if (!database.objectStoreNames.contains(FEEDBACK_STORE)) {
        const store = database.createObjectStore(FEEDBACK_STORE, { keyPath: "id" });
        store.createIndex("eventId", "eventId", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function getObservationProfile(): Promise<ObservationProfile> {
  const database = await openObservationDatabase();
  const transaction = database.transaction(PROFILE_STORE, "readonly");
  const record = await requestResult<ObservationProfile | undefined>(
    transaction.objectStore(PROFILE_STORE).get("primary"),
  );
  await transactionDone(transaction);
  database.close();
  if (record) return record;
  const now = Date.now();
  return { ...DEFAULT_PROFILE, learningStartedAt: now, updatedAt: now };
}

export async function saveObservationProfile(
  profile: ObservationProfile,
): Promise<ObservationProfile> {
  const next = { ...profile, updatedAt: Date.now() };
  const database = await openObservationDatabase();
  const transaction = database.transaction(PROFILE_STORE, "readwrite");
  transaction.objectStore(PROFILE_STORE).put(next);
  await transactionDone(transaction);
  database.close();
  return next;
}

export async function saveObservationEpisode(episode: ObservationEpisode) {
  const database = await openObservationDatabase();
  const transaction = database.transaction(EPISODE_STORE, "readwrite");
  transaction.objectStore(EPISODE_STORE).put(episode);
  await transactionDone(transaction);
  database.close();
  return episode;
}

export async function listObservationEpisodes(limit = 200): Promise<ObservationEpisode[]> {
  const database = await openObservationDatabase();
  const transaction = database.transaction(EPISODE_STORE, "readonly");
  const episodes = await requestResult<ObservationEpisode[]>(
    transaction.objectStore(EPISODE_STORE).getAll(),
  );
  await transactionDone(transaction);
  database.close();
  return episodes.sort((a, b) => b.recordedAt - a.recordedAt).slice(0, limit);
}

export async function getObservationBaseline(version = 1): Promise<BaselineSnapshot> {
  const episodes = await listObservationEpisodes(500);
  return buildBaseline(episodes, version);
}

export async function saveAnalysisFeedback(feedback: AnalysisFeedback) {
  const database = await openObservationDatabase();
  const transaction = database.transaction([FEEDBACK_STORE, EPISODE_STORE], "readwrite");
  transaction.objectStore(FEEDBACK_STORE).put(feedback);
  if (feedback.verdict === "false_positive") {
    const episodeStore = transaction.objectStore(EPISODE_STORE);
    const episodeRequest = episodeStore.get(feedback.eventId);
    episodeRequest.onsuccess = () => {
      const episode = episodeRequest.result as ObservationEpisode | undefined;
      if (!episode) return;
      episodeStore.put({
        ...episode,
        disposition: "excluded",
        dispositionReason: "사용자가 잘못된 감지로 확인해 분석 집계에서 제외했어요.",
      });
    };
  }
  await transactionDone(transaction);
  database.close();
  return feedback;
}

export async function getFeedbackForEvent(eventId: string): Promise<AnalysisFeedback | null> {
  const database = await openObservationDatabase();
  const transaction = database.transaction(FEEDBACK_STORE, "readonly");
  const records = await requestResult<AnalysisFeedback[]>(
    transaction.objectStore(FEEDBACK_STORE).index("eventId").getAll(eventId),
  );
  await transactionDone(transaction);
  database.close();
  return records.sort((a, b) => b.createdAt - a.createdAt)[0] ?? null;
}

export async function deleteAllObservationData() {
  const database = await openObservationDatabase();
  const transaction = database.transaction(
    [PROFILE_STORE, EPISODE_STORE, FEEDBACK_STORE],
    "readwrite",
  );
  transaction.objectStore(PROFILE_STORE).clear();
  transaction.objectStore(EPISODE_STORE).clear();
  transaction.objectStore(FEEDBACK_STORE).clear();
  await transactionDone(transaction);
  database.close();
}
