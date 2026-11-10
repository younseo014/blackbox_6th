/** Failed batches stay queued in order until the next explicit flush retries them. */
export class RecordingWriteQueue {
  private pending: Array<() => Promise<void>> = [];
  private running: Promise<void> | null = null;

  enqueue(write: () => Promise<void>) {
    this.pending.push(write);
    return this.flush();
  }

  flush(): Promise<void> {
    if (this.running) return this.running;
    this.running = Promise.resolve().then(async () => {
      while (this.pending.length) {
        await this.pending[0]();
        this.pending.shift();
      }
    }).finally(() => { this.running = null; });
    return this.running;
  }
}
