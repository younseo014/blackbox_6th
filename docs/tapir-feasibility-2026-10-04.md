# TAPIR point-tracking feasibility probe — 2026-10-04

## Decision

**Do not replace periodic pose inference with one initial detection plus indefinite point tracking.** Causal TAPIR followed visible surface points accurately in the public human clips overall, but tracking degraded after occlusion in the `judo` clip. On this Apple Silicon CPU, running it at the app's busiest three-camera scheduling rate is also outside the measured sequential compute budget. A hybrid with periodic pose re-detection remains a hypothesis worth testing; this probe does not establish an improvement in action detection.

## What was actually run

- Official `google-deepmind/tapnet` code at commit `730cda1c730877cfedbe01bf87fb1cadb78a565d`, using its **causal TAPIR** checkpoint. The checkpoint SHA-256 was `6e1083f074f840ca6dc9b72e7c895eae3f4d266644dbbc6d3c3a852192855012`.
- Official [TAP-Vid-DAVIS archive](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip), SHA-256 `60eb6239c57a9877900c269d14637c2b6eece8fa280f59e4592324c6392031b6`. The archive includes annotated videos; no study participant data was used.
- Three human-motion clips: `breakdance`, `judo`, and `dance-twirl`. First visible annotated surface points in frame 0 were passed as queries **once**. No new points were initialized later. Frames were resized to 256×256, as in the repository's TAP-Vid evaluation. The script is [benchmark-tapir-poc.py](../scripts/benchmark-tapir-poc.py).
- Apple Silicon macOS, JAX CPU device, 22 requested points. The videos supplied 21–22 visible points in the first frame. Times exclude loading the 1.6 GB dataset and checkpoint, but include JAX compilation separately.

| Clip | Frames | Points | Visible point error, median | Visible points within 8 px | Occlusion recall | Steady p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| breakdance | 84 | 21 | 1.64 px | 94.3% | 93.1% | 68.5 / 72.1 ms |
| judo | 34 | 22 | 1.40 px | 90.1% | 87.3% | 70.5 / 72.7 ms |
| dance-twirl | 90 | 22 | 1.52 px | 91.6% | 72.3% | 69.9 / 77.6 ms |

The denominator for “within 8 px” is all ground-truth visible point-frames. It is **not** pose-joint accuracy or action detection accuracy. In `judo`, the fraction within 8 px fell to **46.7% at frame 31** and **50.0% at frame 33**, after the first-frame-only query. The first prediction took about 1.4–1.5 seconds to compile; query initialization took about 0.4 seconds. Subsequent inference was about 70 ms/frame at 256×256.

The app schedules one primary camera at 10 frames/second and two recently active secondary cameras at about 6.7 frames/second each. At roughly 23.3 total frames/second, 23.3 × 70 ms ≈ **1.63 seconds of sequential TAPIR compute per wall-clock second**, before pose inference, hand inference, or UI work. This is an extrapolation, not a three-camera benchmark. Windows GPU performance is unmeasured.

## Reproduction

The public [TAPNet README](https://github.com/google-deepmind/tapnet) links the model code and [official checkpoint](https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy); the [TAP-Vid README](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md) links the data. Install `jax`, `dm-haiku`, `chex`, `optax`, `einops`, `einshape`, `dm-tree`, and `opencv-python` in an isolated Python environment. Extract `tapvid_davis/tapvid_davis.pkl` from the archive, then run `scripts/benchmark-tapir-poc.py` with `PYTHONPATH` set to the TAPNet clone and the `--dataset`, `--checkpoint`, `--clip`, `--frames`, and `--points` arguments.

For example, choose `--clip judo --frames 34 --points 22`. Repeat for the other clips. Timing may vary with hardware and background load; the image-point metrics are deterministic for the same code and checkpoint.

## Limits and next gate

TAP-Vid marks points on visible surfaces. It does not establish that a tracked point remains the same **anatomical joint**; the benchmark's [own README](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md) distinguishes clothes/skin points from intended joints. The app stores MediaPipe-layout body and hand joints and performs action detection on hand trajectories. The current normal capture path stores coordinates rather than raw video, so those saved sessions cannot be used to replay TAPIR over the original camera frames.

The result therefore supports only **point-tracking feasibility**. Before any app integration, compare the existing pose-only path with pose + TAPIR on the same public or consented developer pre-validation clips. Score anatomical joint error, left/right swaps, error after occlusion, `double_check`/`micro_delay` precision and recall, per-camera frame rate, and UI stalls. Keep periodic pose inference and a discard/re-initialize rule when joint geometry or visibility disagrees. Freeze no new thresholds until those measurements exist.
