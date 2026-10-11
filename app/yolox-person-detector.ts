export const YOLOX_PERSON_THRESHOLD = 0.6;

type Pending = { resolve: (score: number) => void; reject: (error: Error) => void };

export class YoloxPersonDetector {
  private worker: Worker;
  private pending = new Map<number, Pending>();
  private nextId = 0;
  private closed = false;

  private constructor(worker: Worker) {
    this.worker = worker;
  }

  static async create() {
    const worker = new Worker(new URL("./yolox-person-worker.ts", import.meta.url), { type: "module" });
    const detector = new YoloxPersonDetector(worker);
    try {
      await new Promise<void>((resolve, reject) => {
        const onMessage = (event: MessageEvent) => {
          if (event.data?.type !== "ready" && event.data?.type !== "load-error") return;
          worker.removeEventListener("message", onMessage);
          if (event.data.type === "ready") resolve();
          else reject(new Error(event.data.error));
        };
        worker.addEventListener("message", onMessage);
        worker.addEventListener("error", (event) => reject(new Error(event.message)), { once: true });
        worker.postMessage({
          type: "load",
          modelUrl: new URL("/models/yolox_tiny.onnx", location.href).href,
          runtimeUrl: new URL("/ort/", location.href).href,
        });
      });
      worker.addEventListener("message", (event: MessageEvent) => detector.handleMessage(event));
      worker.addEventListener("error", (event) => detector.failAll(new Error(event.message)));
      return detector;
    } catch (error) {
      worker.terminate();
      throw error;
    }
  }

  private handleMessage(event: MessageEvent) {
    const { id, type, score, error } = event.data ?? {};
    const request = this.pending.get(id);
    if (!request) return;
    this.pending.delete(id);
    if (type === "result") request.resolve(score);
    else if (type === "result-error") request.reject(new Error(error));
  }

  private failAll(error: Error) {
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
  }

  async detect(video: HTMLVideoElement) {
    if (this.closed) throw new Error("YOLOX detector is closed.");
    const bitmap = await createImageBitmap(video);
    if (this.closed) {
      bitmap.close();
      throw new Error("YOLOX detector is closed.");
    }
    const id = ++this.nextId;
    return new Promise<number>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: "detect", id, bitmap }, [bitmap]);
    });
  }

  close() {
    this.closed = true;
    this.failAll(new Error("YOLOX detector was closed."));
    this.worker.terminate();
  }
}
