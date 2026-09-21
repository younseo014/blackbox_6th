# Model and dataset provenance

Every model file and dataset used for evaluation, with its source and checksum.
Nothing here may be used in the study unless it appears in this table.

## Pose models

| Model | File | sha256 | Source | License |
| --- | --- | --- | --- | --- |
| MediaPipe Pose Landmarker Heavy | `public/models/pose_landmarker_heavy.task` | `64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b` | Google MediaPipe Solutions | Apache-2.0 |
| MediaPipe Hand Landmarker | `public/models/hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | Google MediaPipe Solutions | Apache-2.0 |
| EfficientDet-Lite0 | `public/models/efficientdet_lite0.tflite` | `0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb` | Google MediaPipe Solutions | Apache-2.0 |
| RTMW-l WholeBody 384x288 (ONNX SDK) | `models/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip` | `a87e1af41a0a067776dba7d46e1c21c8f6e9f18e247e0e606718dd1f31e96ffd` | https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip | Apache-2.0 (MMPose) |
| RTMW-l WholeBody 256x192 (ONNX SDK) | `models/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip` | `1e3e77558dfc199129bfff1c583e51b4ee190914de6ae30688243c20163c148c` | https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip | Apache-2.0 (MMPose) |
| RTMW-m WholeBody 256x192 (ONNX SDK) | `models/rtmw/rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip` | `a5c2bcd897a654e99df2e04e2a24724aea79f668d4d2a44f9146063ddccc27a3` | https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip | Apache-2.0 (MMPose) |

These are the **official MMPose pre-trained weights**, exported to ONNX by the
MMPose team (`onnx_sdk` release). No model was trained from scratch, and no
weights were modified.

Note: the `pipeline.json` inside each SDK zip reports `image_size [192, 256]`
regardless of the actual variant. It is stale. `end2end.onnx`'s own input shape
is authoritative and is what `scripts/rtmw_backend.py` reads.

`models/rtmw/` and `data/external/` are git-ignored - they are large and are not
ours to redistribute. Re-obtain them from the URLs above and verify the sha256.

## Person detectors

| Model | File | sha256 | Source | License |
| --- | --- | --- | --- | --- |
| YOLOX-nano (HumanArt) | `models/yolox/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip` | `cd4fbef13d5325fb550e06ac70c8dc06b082f206ea24abc5bc4b1e60652a2596` | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip | Apache-2.0 (MMPose) |
| YOLOX-tiny (HumanArt) | `models/yolox/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip` | `36e09ca555916253fa1b5d51bec01e48e98e1f0af966c51ce3e844d8c8fc4cfc` | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip | Apache-2.0 (MMPose) |
| YOLOX-m (HumanArt) | `models/yolox/yolox_m_8xb8-300e_humanart-c2c7a14a.zip` | `a000224fd8ba283202bc62d4a5fcdfe353adb9f468777dbac1ea2ada2093adde` | https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip | Apache-2.0 (MMPose) |

As with RTMW, each zip's `pipeline.json` reports a stale input size (416) while
the exported graph is 640x640. `scripts/person_detector.py` reads the graph.
NMS is baked into these exports, so their outputs are final boxes.

## Fitted parameters

| Artifact | File | What it is |
| --- | --- | --- |
| RTMW visibility calibration | `config/rtmw-visibility-calibration.json` | 2-parameter logistic mapping RTMW's SimCC peak to P(keypoint visible), per model variant. Fitted by `scripts/fit_rtmw_visibility.py` on COCO-WholeBody val, split by image. |
| Tracking decision thresholds | `DEFAULT_TUNING` in `app/global-person-track.ts` | Limb-asymmetry floor swept end to end by `scripts/fit-tracking-thresholds.ts` on COCO-WholeBody val, split by image. |
| Detector operating point | `docs/tracking-eval/detector-sweep.json` | Recall / empty-frame false-positive curve for four detectors at twelve thresholds, from `scripts/sweep_detectors.py`. |

No pose or detector weights were modified. Everything fitted here is a decision
threshold or an output calibration, each with a held-out measurement recorded
in `docs/tracking-eval.md`.

## Datasets

| Dataset | Status | Size | Access |
| --- | --- | --- | --- |
| COCO-WholeBody v1.0 val | **downloaded, wired into evaluation** | 133 MB annotations | https://github.com/jin-s13/COCO-WholeBody (annotations via Google Drive, public) |
| COCO val2017 images | **downloaded** | 815 MB | http://images.cocodataset.org/zips/val2017.zip |
| COCO trainval2017 annotations | downloaded, unused so far | 200 MB | http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
| WILDTRACK | **download started** (6.8 GB, direct, no registration) | 6.8 GB | https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip |
| OCHuman | **blocked** | 667 MB | https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman - requires submitting a registration form. Creating accounts / submitting forms on the user's behalf is out of scope for an agent; a human must do this. |
| H3WB | **partially blocked** | - | Annotations are public on Google Drive, but the images come from Human3.6M, which requires a registered account and acceptance of its license agreement. A human must obtain them. |
| CMU Panoptic | **not attempted** | hundreds of GB | http://domedb.perception.cs.cmu.edu/ - exceeds the 59 GB free on this machine. Needs either a selective subset or more disk. |

`data/external/epfl/` (EPFL pose dataset) and `models/epfl-*.json`,
`models/lara-primitives-v1.json`, `models/utkinect-primitives-v1.json` predate
this work. They belong to the **motion classification** pipeline, not pose
estimation, and were not retrained here.

## Data policy

Study participant recordings are not used for evaluation, calibration or any
model fitting. Only public datasets and developer-recorded pre-validation clips
appear in `data/eval-scenarios/`, and every scenario file must declare a
`source` - the runner rejects one that does not.
