import test from "node:test";
import assert from "node:assert/strict";
import { RecordingWriteQueue } from "../app/recording-write-queue.ts";

test("a failed recording batch is retried before later batches without duplication", async () => {
  const queue = new RecordingWriteQueue();
  const saved: number[] = [];
  let attempts = 0;
  const first = queue.enqueue(async () => { if (++attempts === 1) throw new Error("disk full"); saved.push(1); });
  const second = queue.enqueue(async () => { saved.push(2); });
  await Promise.all([assert.rejects(first, /disk full/), assert.rejects(second, /disk full/)]);
  assert.deepEqual(saved, []);
  await queue.flush();
  assert.deepEqual(saved, [1, 2]);
  await queue.flush();
  assert.deepEqual(saved, [1, 2]);
});
