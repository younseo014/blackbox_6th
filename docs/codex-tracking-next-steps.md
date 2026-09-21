# Codex prompt — 3-camera single-person tracking, remaining stages

Paste everything below the line into Codex as a single task. It assumes the
repository state produced by the previous session (see "What already exists").

---

You are continuing work on `memory-guard`, a Next.js + MediaPipe app that
tracks **exactly one person** across **three fixed webcams** for a research
study. Read `docs/tracking-eval.md` first — it is the audit, the baseline and
the threshold rationale, and it is the contract you are extending.

## Hard rules

- Preserve existing behaviour and backwards compatibility with stored sessions.
  Do not delete the MediaPipe pipeline; it is the known-good fallback.
- Do not weaken the privacy posture. In normal operation no video, audio or
  face features are stored. Training-data collection only behind an explicit
  developer mode with recorded consent.
- Study participant data is never used for evaluation or model improvement.
- Never present uncalibrated coordinates as 3D. `GlobalPersonTrack.calibrated`
  must stay `false` until a validated calibration produced the coordinates.
- Every new decision path gets unit tests. Run `npm run lint`, `npm test` and
  `npm run eval:tracking` before reporting done.
- Do not invent threshold values. Choose them from harness output and record
  the before/after numbers in `docs/tracking-eval.md`.
- If a criterion fails, report the failure and the bottleneck. Do not round it
  away or describe it as passing.

## What already exists (do not rebuild)

| File | Role |
| --- | --- |
| `app/global-person-track.ts` | Single global owner across the three cameras. Per-joint `observed`/`inferred`/`missing`, anatomy validation, temporal continuity gate, handoff reporting. |
| `app/pose-scheduler.ts` | One body inference per animation frame; per-camera cadence by tier (primary/secondary/alerted/idle); adaptive detector cadence. |
| `app/pose-model-adapter.ts` | `PoseBackend` interface + `loadPoseBackend()` with ordered fallback and an attempt log. **This is the seam you plug RTMW into.** |
| `app/tracking-eval.ts` | Replay evaluator + `checkAcceptance()` + Markdown report. |
| `app/tracking-eval-scenarios.ts` | Ten synthetic regression scenarios. |
| `scripts/run-tracking-eval.ts` | Runner; also loads `data/eval-scenarios/*.json`. |
| `app/pose-target-lock.ts`, `app/person-presence-gate.ts` | Per-camera lock and empty-room gate (pre-existing). |
| `app/page.tsx` | The loop; already calls the scheduler and `updateGlobalPersonTrack`. |

Baseline: all ten scenarios pass all six acceptance criteria with MediaPipe
Heavy. Fusion cost 0.003–0.02 ms/tick.

## Task 1 — RTMW/RTMPose backend in the browser (highest priority)

**Already done, do not redo:** the three official RTMW ONNX models are
downloaded with checksums recorded in `docs/model-provenance.md`; a working
reference implementation of preprocessing, SimCC decoding and the 133→33
keypoint mapping exists in `scripts/rtmw_backend.py`; the visibility
calibration is fitted and stored in `config/rtmw-visibility-calibration.json`;
and the A/B against MediaPipe on COCO-WholeBody is in `docs/tracking-eval.md`.

**The choice is made by the numbers: RTMW-l 384×288.** It is the only backend
with zero empty-frame false positives, which is the tightest acceptance budget.
RTMW-l 256×192 has marginally better retention but fails that criterion.

Your job is to port `scripts/rtmw_backend.py` to TypeScript as
`app/pose-backends/rtmw.ts` satisfying `PoseBackend`, and to verify the port
reproduces the Python numbers on the same images before trusting it.

1. **Runtime.** `onnxruntime-web`. WebGPU first, WASM fallback. Inference runs
   in a dedicated Web Worker — the UI thread must never block. Transfer frames
   with `ImageBitmap`/`OffscreenCanvas`, not base64.
2. **Preprocessing must match the Python reference exactly.** Aspect-preserving
   affine around the detector box with 1.25 padding, BGR→RGB, ImageNet mean/std,
   NCHW. A silent preprocessing mismatch produces plausible-looking garbage.
3. **Visibility.** Apply the fitted calibration from
   `config/rtmw-visibility-calibration.json`, keyed by model name. Never clamp
   the raw SimCC peak into [0,1] — that destroys the occlusion signal entirely
   and marks every hidden joint as confidently seen.
4. **Keypoint mapping.** RTMW WholeBody is 133 keypoints; the stored frame
   format and every analyser are defined against MediaPipe's 33-point layout
   (see `BODY_LANDMARK_NAMES` in `app/pose-store.ts`). Map inside the adapter.
   A keypoint RTMW does not provide must come back with visibility 0, never a
   fabricated coordinate.
5. **Failure handling.** Model fetch failure, WebGPU unavailable, worker crash,
   and shape mismatch must all fall back to MediaPipe via `loadPoseBackend`,
   and the `attempts` log must be written into the session record. Add unit
   tests for each of those four failures.
6. **Wire into `page.tsx`** behind the adapter; the three landmarker refs
   become three backend instances. Do not change the drawing or recording paths.

**Then benchmark before switching anything.** Add `scripts/benchmark-backends.ts`
driving a real browser (Playwright is already a devDependency) over a fixed
clip, reporting per-camera FPS, p50/p95 inference ms, UI long-task count,
GPU-process and renderer memory. Only pick a backend from those numbers, and
write the comparison table into `docs/tracking-eval.md`.

## Task 2 — Camera calibration and multi-view fusion

Implement `app/camera-calibration.ts` plus a developer-only calibration screen.

- Checkerboard or ArUco capture, per camera: lens distortion, intrinsics,
  extrinsics, floor plane, inter-camera overlap regions, timestamp sync offset.
- Persist to the existing store with a schema version; legacy sessions without
  calibration must still load.
- In `global-person-track.ts`, replace the torso-anchored similarity
  re-projection with the calibrated transform **only when a validated
  calibration exists**. Then, and only then, may a cross-camera joint be
  reported as `observed` and `calibrated` set true. Keep the 2D quality-based
  path as the fallback, unchanged.
- Add a geometric consistency gate: with calibration, reject a camera whose
  observation does not epipolar-agree with the current global track.
- Tests: calibrated fusion beats uncalibrated on a synthetic two-view scenario;
  a missing/stale calibration falls back without throwing; a corrupted
  calibration file is rejected rather than used.

## Task 3 — Hand ROI at full resolution

`app/page.tsx` already picks the single camera with the largest forearm
(`handTrackingDetailScore`, `MIN_HAND_FOREARM_PIXELS = 28`). Add, in a new
`app/hand-roi.ts`:

- Crop a wrist-centred ROI from the **source-resolution** frame and upscale it
  before running `HandLandmarker`, instead of running on the downscaled frame.
- Map ROI-local coordinates back to the original camera frame exactly; add a
  round-trip unit test (crop → detect → un-map must return the input).
- Below a minimum hand box size or pixel count, return `missing`. Do not
  synthesise fingers for a distant hand.
- Left/right assignment must use temporal continuity, not per-frame handedness
  alone, so the sides cannot swap between frames. Test a crossing-hands sequence.
- Discard hand results from any camera other than the selected one.

## Task 4 — Remaining evaluation data

`scripts/build_eval_scenarios.py` already converts a dataset into the
`EvalScenario` shape by running a real pose backend over its images. Extend it.

**Done:** COCO-WholeBody val (714 single-person + 375 people-free images),
scored against all four backends.

**Still needed:**

- **WILDTRACK** — downloading to `data/external/wildtrack/`. The only fixed-CCTV
  multi-camera data in the set, and it ships real camera calibration, so it
  serves Task 2 as well. It is crowded, which conflicts with the single-person
  assumption: use it to verify the system reports **at most one** global id in a
  crowd and never flips between people, and pick single-occupancy windows for
  retention scoring.
- **OCHuman** — blocked. Requires a form at
  https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman. A human must
  submit it; do not register on anyone's behalf.
- **H3WB** — annotations public, but images need a Human3.6M account and license
  acceptance. A human must obtain them.
- **CMU Panoptic** — hundreds of GB against ~55 GB free. Needs a selective
  subset or more disk.
- **Developer-recorded negatives** — empty store, chair, hanger, mannequin,
  furniture corners, reflected person, person on a monitor or poster, lighting
  changes, camera noise. Nothing substitutes for these: one people-free COCO
  image already fools three of the four backends.

## Task 4b — Fine-tuning (only if the numbers still demand it)

Do **not** start here. The evidence says the remaining gap is a domain gap —
web photography versus a fixed overhead camera — not an under-trained model.
Switching MediaPipe → RTMW already recovered distant 31%→63% and head-occluded
54%→83%, while our own tracking thresholds account for only 2 points of the
shortfall.

If fine-tuning is authorised it starts from the official pre-trained weights in
`docs/model-provenance.md`, never from scratch, never on participant data, and
not on this machine: the official RTMW recipe is 270 epochs on 8×A100, and
MMPose/mmcv needs a source build on macOS ARM with known MPS gaps. Use a Linux
GPU host and record the run in `docs/model-provenance.md`.

## Task 5 — 12-hour soak

- Add a soak mode to the runner that replays a long scenario and samples
  `performance.memory` / renderer RSS on an interval.
- Run three cameras for 12 hours against looped pre-validation footage.
- Report: memory growth, processing lag accumulation, dropped frames, empty-
  frame false positives per camera, and whether the run completed.
- Fix any leak before the study. Buffer flush paths in `page.tsx`
  (`flushPoseFrames`, `flushCameraHealth`) are the first place to look.

## Definition of done

Freeze model, thresholds and code version before the study, and report:

1. changed files
2. model source URLs, release tags, checksums, licenses
3. before/after comparison table on the *same* validation set
4. which of the nine acceptance criteria pass, with numbers
5. the failures and the single biggest bottleneck, stated plainly

The nine criteria: empty-frame false positives < 1/camera/12h; recognition
≤ 1s after entry; zero person→object switches; torso retention ≥ 95%; visible
major-joint detection ≥ 90%; at most one global person id at all times;
minimal handoff blind gaps; 12-hour run completes; UI never blocked by
inference.
