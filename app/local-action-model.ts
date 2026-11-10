import type { VisualActionSample } from "./local-action-context";

type Result = Omit<VisualActionSample, "timeMs" | "bodyObserved">;
export class LocalActionModel {
  private worker: Worker;
  private pending = new Map<number, { resolve: (result: Result) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  private closed = false;
  private nextId = 0;
  private constructor(worker: Worker) { this.worker = worker; }

  static async create(signal?: AbortSignal) {
    const worker = new Worker(new URL("./local-action-worker.ts", import.meta.url), { type: "module" });
    const model = new LocalActionModel(worker);
    try {
      await new Promise<void>((resolve, reject) => {
        const aborted = () => { clearTimeout(timer); worker.terminate(); reject(new Error("Local action model loading cancelled")); };
        const timer = setTimeout(() => { signal?.removeEventListener("abort", aborted); reject(new Error("Local action model loading timed out")); }, 120_000);
        if (signal?.aborted) { aborted(); return; }
        signal?.addEventListener("abort", aborted, { once: true });
        const loaded = (event: MessageEvent) => {
          if (!["ready", "load-error"].includes(event.data?.type)) return;
          clearTimeout(timer); worker.removeEventListener("message", loaded);
          signal?.removeEventListener("abort", aborted);
          if (event.data.type === "ready") resolve(); else reject(new Error(event.data.error));
        };
        worker.addEventListener("message", loaded);
        worker.addEventListener("error", event => { clearTimeout(timer); reject(new Error(event.message)); }, { once: true });
        worker.postMessage({ type: "load", modelUrl: new URL("/models/action-context/vision-int8.onnx", location.href).href,
          prototypesUrl: new URL("/models/action-context/prototypes.json", location.href).href, runtimeUrl: new URL("/ort/", location.href).href });
      });
      worker.addEventListener("message", (event: MessageEvent) => {
        const request = model.pending.get(event.data?.id);
        if (!request) return;
        model.pending.delete(event.data.id); clearTimeout(request.timer);
        if (event.data.type === "result") request.resolve(event.data.result);
        else request.reject(new Error(event.data.error));
      });
      worker.addEventListener("error", event => model.close(new Error(event.message)));
      return model;
    } catch (error) { worker.terminate(); throw error; }
  }

  async classify(video: HTMLVideoElement) {
    if (this.closed) throw new Error("Local action model is closed");
    const bitmap = await createImageBitmap(video);
    if (this.closed) { bitmap.close(); throw new Error("Local action model is closed"); }
    const id = ++this.nextId;
    return new Promise<Result>((resolve, reject) => {
      const timer = setTimeout(() => this.close(new Error("Local action inference timed out")), 20_000);
      this.pending.set(id, { resolve, reject, timer });
      this.worker.postMessage({ type: "classify", id, bitmap }, [bitmap]);
    });
  }

  close(error = new Error("Local action model closed")) {
    this.closed = true;
    for (const request of this.pending.values()) { clearTimeout(request.timer); request.reject(error); }
    this.pending.clear(); this.worker.terminate();
  }
}
