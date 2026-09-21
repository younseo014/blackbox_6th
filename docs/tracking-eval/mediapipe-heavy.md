# Tracking evaluation - mediapipe-heavy

Generated: 2026-09-20T06:20:59.691Z

## Acceptance criteria

| Scenario | Criterion | Target | Actual | Result |
| --- | --- | --- | --- | --- |
| empty-room | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| empty-room | Recognition after entry | <= 1000 ms | n/a ms | PASS |
| empty-room | Person-to-object target switches | 0 | 0 | PASS |
| empty-room | Torso tracking retention | >= 95% | 100.0% | PASS |
| empty-room | Visible major-joint detection rate | >= 90% | n/a | PASS |
| empty-room | Concurrent global person ids | <= 1 | 1 | PASS |
| decoy-furniture | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| decoy-furniture | Recognition after entry | <= 1000 ms | n/a ms | PASS |
| decoy-furniture | Person-to-object target switches | 0 | 0 | PASS |
| decoy-furniture | Torso tracking retention | >= 95% | 100.0% | PASS |
| decoy-furniture | Visible major-joint detection rate | >= 90% | n/a | PASS |
| decoy-furniture | Concurrent global person ids | <= 1 | 1 | PASS |
| entry-latency | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| entry-latency | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| entry-latency | Person-to-object target switches | 0 | 0 | PASS |
| entry-latency | Torso tracking retention | >= 95% | 98.0% | PASS |
| entry-latency | Visible major-joint detection rate | >= 90% | 100.0% | PASS |
| entry-latency | Concurrent global person ids | <= 1 | 1 | PASS |
| side-view | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| side-view | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| side-view | Person-to-object target switches | 0 | 0 | PASS |
| side-view | Torso tracking retention | >= 95% | 100.0% | PASS |
| side-view | Visible major-joint detection rate | >= 90% | 100.0% | PASS |
| side-view | Concurrent global person ids | <= 1 | 1 | PASS |
| back-view | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| back-view | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| back-view | Person-to-object target switches | 0 | 0 | PASS |
| back-view | Torso tracking retention | >= 95% | 100.0% | PASS |
| back-view | Visible major-joint detection rate | >= 90% | 100.0% | PASS |
| back-view | Concurrent global person ids | <= 1 | 1 | PASS |
| headless | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| headless | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| headless | Person-to-object target switches | 0 | 0 | PASS |
| headless | Torso tracking retention | >= 95% | 100.0% | PASS |
| headless | Visible major-joint detection rate | >= 90% | 100.0% | PASS |
| headless | Concurrent global person ids | <= 1 | 1 | PASS |
| upper-body-only | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| upper-body-only | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| upper-body-only | Person-to-object target switches | 0 | 0 | PASS |
| upper-body-only | Torso tracking retention | >= 95% | 100.0% | PASS |
| upper-body-only | Visible major-joint detection rate | >= 90% | 100.0% | PASS |
| upper-body-only | Concurrent global person ids | <= 1 | 1 | PASS |
| distant-small | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| distant-small | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| distant-small | Person-to-object target switches | 0 | 0 | PASS |
| distant-small | Torso tracking retention | >= 95% | 100.0% | PASS |
| distant-small | Visible major-joint detection rate | >= 90% | 100.0% | PASS |
| distant-small | Concurrent global person ids | <= 1 | 1 | PASS |
| camera-handoff | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| camera-handoff | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| camera-handoff | Person-to-object target switches | 0 | 0 | PASS |
| camera-handoff | Torso tracking retention | >= 95% | 100.0% | PASS |
| camera-handoff | Visible major-joint detection rate | >= 90% | 100.0% | PASS |
| camera-handoff | Concurrent global person ids | <= 1 | 1 | PASS |
| three-camera-overlap | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| three-camera-overlap | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| three-camera-overlap | Person-to-object target switches | 0 | 0 | PASS |
| three-camera-overlap | Torso tracking retention | >= 95% | 100.0% | PASS |
| three-camera-overlap | Visible major-joint detection rate | >= 90% | 100.0% | PASS |
| three-camera-overlap | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-edet035 | Empty-frame false positives | < 1 per camera per 12h | 1144.06 / 12h | **FAIL** |
| coco-wholebody-val-edet035 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-edet035 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-edet035 | Torso tracking retention | >= 95% | 80.4% | **FAIL** |
| coco-wholebody-val-edet035 | Visible major-joint detection rate | >= 90% | 73.0% | **FAIL** |
| coco-wholebody-val-edet035 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-edet050 | Empty-frame false positives | < 1 per camera per 12h | 476.69 / 12h | **FAIL** |
| coco-wholebody-val-edet050 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-edet050 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-edet050 | Torso tracking retention | >= 95% | 78.3% | **FAIL** |
| coco-wholebody-val-edet050 | Visible major-joint detection rate | >= 90% | 71.3% | **FAIL** |
| coco-wholebody-val-edet050 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-rtmw-l-256x192 | Empty-frame false positives | < 1 per camera per 12h | 264.50 / 12h | **FAIL** |
| coco-wholebody-val-rtmw-l-256x192 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-rtmw-l-256x192 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-rtmw-l-256x192 | Torso tracking retention | >= 95% | 90.6% | **FAIL** |
| coco-wholebody-val-rtmw-l-256x192 | Visible major-joint detection rate | >= 90% | 90.2% | PASS |
| coco-wholebody-val-rtmw-l-256x192 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet030 | Empty-frame false positives | < 1 per camera per 12h | 4576.22 / 12h | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-edet030 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet030 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet030 | Torso tracking retention | >= 95% | 95.5% | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet030 | Visible major-joint detection rate | >= 90% | 95.8% | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet030 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet035 | Empty-frame false positives | < 1 per camera per 12h | 3241.49 / 12h | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-edet035 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet035 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet035 | Torso tracking retention | >= 95% | 94.7% | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-edet035 | Visible major-joint detection rate | >= 90% | 95.0% | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet035 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet050 | Empty-frame false positives | < 1 per camera per 12h | 1144.06 / 12h | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-edet050 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet050 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet050 | Torso tracking retention | >= 95% | 90.8% | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-edet050 | Visible major-joint detection rate | >= 90% | 91.3% | PASS |
| coco-wholebody-val-rtmw-l-384x288-edet050 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | Empty-frame false positives | < 1 per camera per 12h | 1525.41 / 12h | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | Torso tracking retention | >= 95% | 95.8% | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | Visible major-joint detection rate | >= 90% | 96.1% | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | Empty-frame false positives | < 1 per camera per 12h | 1239.39 / 12h | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | Torso tracking retention | >= 95% | 94.7% | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | Visible major-joint detection rate | >= 90% | 95.2% | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | Empty-frame false positives | < 1 per camera per 12h | 762.70 / 12h | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | Torso tracking retention | >= 95% | 91.7% | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | Visible major-joint detection rate | >= 90% | 92.5% | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | Empty-frame false positives | < 1 per camera per 12h | 3336.83 / 12h | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | Torso tracking retention | >= 95% | 95.9% | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | Visible major-joint detection rate | >= 90% | 95.8% | PASS |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-rtmw-l-384x288 | Empty-frame false positives | < 1 per camera per 12h | 0.00 / 12h | PASS |
| coco-wholebody-val-rtmw-l-384x288 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-rtmw-l-384x288 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-rtmw-l-384x288 | Torso tracking retention | >= 95% | 90.6% | **FAIL** |
| coco-wholebody-val-rtmw-l-384x288 | Visible major-joint detection rate | >= 90% | 91.1% | PASS |
| coco-wholebody-val-rtmw-l-384x288 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val-rtmw-m-256x192 | Empty-frame false positives | < 1 per camera per 12h | 264.50 / 12h | **FAIL** |
| coco-wholebody-val-rtmw-m-256x192 | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val-rtmw-m-256x192 | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val-rtmw-m-256x192 | Torso tracking retention | >= 95% | 89.8% | **FAIL** |
| coco-wholebody-val-rtmw-m-256x192 | Visible major-joint detection rate | >= 90% | 89.8% | **FAIL** |
| coco-wholebody-val-rtmw-m-256x192 | Concurrent global person ids | <= 1 | 1 | PASS |
| coco-wholebody-val | Empty-frame false positives | < 1 per camera per 12h | 264.50 / 12h | **FAIL** |
| coco-wholebody-val | Recognition after entry | <= 1000 ms | 200 ms | PASS |
| coco-wholebody-val | Person-to-object target switches | 0 | 0 | PASS |
| coco-wholebody-val | Torso tracking retention | >= 95% | 78.3% | **FAIL** |
| coco-wholebody-val | Visible major-joint detection rate | >= 90% | 71.3% | **FAIL** |
| coco-wholebody-val | Concurrent global person ids | <= 1 | 1 | PASS |

## Per-scenario detail

| Scenario | Source | Ticks | Presence acc. | Retention | Handoffs | Worst handoff gap | Fusion ms/tick |
| --- | --- | --- | --- | --- | --- | --- | --- |
| empty-room | synthetic:developer-pre-validation | 300 | 100.0% | 100.0% | 0 | 0 ms | 0.000 |
| decoy-furniture | synthetic:developer-pre-validation | 200 | 100.0% | 100.0% | 0 | 0 ms | 0.010 |
| entry-latency | synthetic:developer-pre-validation | 120 | 98.3% | 98.0% | 0 | 0 ms | 0.017 |
| side-view | synthetic:developer-pre-validation | 60 | 96.7% | 100.0% | 0 | 0 ms | 0.000 |
| back-view | synthetic:developer-pre-validation | 60 | 96.7% | 100.0% | 0 | 0 ms | 0.017 |
| headless | synthetic:developer-pre-validation | 60 | 96.7% | 100.0% | 0 | 0 ms | 0.017 |
| upper-body-only | synthetic:developer-pre-validation | 60 | 96.7% | 100.0% | 0 | 0 ms | 0.017 |
| distant-small | synthetic:developer-pre-validation | 60 | 96.7% | 100.0% | 0 | 0 ms | 0.017 |
| camera-handoff | synthetic:developer-pre-validation | 150 | 98.7% | 100.0% | 2 | 100 ms | 0.087 |
| three-camera-overlap | synthetic:developer-pre-validation | 100 | 98.0% | 100.0% | 0 | 0 ms | 0.030 |
| coco-wholebody-val-edet035 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 18126 | 88.8% | 80.4% | 0 | 0 ms | 0.002 |
| coco-wholebody-val-edet050 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 18126 | 88.6% | 78.3% | 0 | 0 ms | 0.001 |
| coco-wholebody-val-rtmw-l-256x192 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 6534 | 74.0% | 90.6% | 0 | 0 ms | 0.003 |
| coco-wholebody-val-rtmw-l-384x288-edet030 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 18126 | 90.4% | 95.5% | 0 | 0 ms | 0.001 |
| coco-wholebody-val-rtmw-l-384x288-edet035 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 18126 | 90.5% | 94.7% | 0 | 0 ms | 0.001 |
| coco-wholebody-val-rtmw-l-384x288-edet050 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 18126 | 90.4% | 90.8% | 0 | 0 ms | 0.001 |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 18126 | 91.1% | 95.8% | 0 | 0 ms | 0.001 |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 18126 | 91.0% | 94.7% | 0 | 0 ms | 0.001 |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 18126 | 90.6% | 91.7% | 0 | 0 ms | 0.001 |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 18126 | 90.7% | 95.9% | 0 | 0 ms | 0.001 |
| coco-wholebody-val-rtmw-l-384x288 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 6534 | 74.0% | 90.6% | 0 | 0 ms | 0.002 |
| coco-wholebody-val-rtmw-m-256x192 | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 6534 | 73.6% | 89.8% | 0 | 0 ms | 0.002 |
| coco-wholebody-val | COCO-WholeBody v1.0 val (https://github.com/jin-s13/COCO-WholeBody, annotations sha256=f8272e9c12f3a424...) over COCO val2017 images (http://images.cocodataset.org/zips/val2017.zip) | 6534 | 68.6% | 78.3% | 0 | 0 ms | 0.002 |

## Conditions

| Scenario | Condition | Ticks | Presence acc. | Retention |
| --- | --- | --- | --- | --- |
| empty-room | empty | 300 | 100.0% | 100.0% |
| decoy-furniture | decoy_object | 200 | 100.0% | 100.0% |
| entry-latency | empty | 20 | 100.0% | 100.0% |
| entry-latency | front | 100 | 98.0% | 98.0% |
| side-view | side | 60 | 96.7% | 100.0% |
| back-view | back | 60 | 96.7% | 100.0% |
| headless | head_occluded | 60 | 96.7% | 100.0% |
| upper-body-only | partial_body | 60 | 96.7% | 100.0% |
| distant-small | distant | 60 | 96.7% | 100.0% |
| camera-handoff | handoff | 150 | 98.7% | 100.0% |
| three-camera-overlap | front | 100 | 98.0% | 100.0% |
| coco-wholebody-val-edet035 | front | 2730 | 59.8% | 89.7% |
| coco-wholebody-val-edet035 | partial_body | 546 | 54.9% | 82.4% |
| coco-wholebody-val-edet035 | back | 336 | 52.4% | 78.6% |
| coco-wholebody-val-edet035 | distant | 498 | 23.3% | 34.9% |
| coco-wholebody-val-edet035 | side | 30 | 66.7% | 100.0% |
| coco-wholebody-val-edet035 | head_occluded | 144 | 36.1% | 54.2% |
| coco-wholebody-val-edet035 | empty | 13842 | 99.7% | 100.0% |
| coco-wholebody-val-edet050 | front | 2730 | 58.6% | 87.9% |
| coco-wholebody-val-edet050 | partial_body | 546 | 53.5% | 80.2% |
| coco-wholebody-val-edet050 | back | 336 | 51.2% | 76.8% |
| coco-wholebody-val-edet050 | distant | 498 | 21.7% | 32.5% |
| coco-wholebody-val-edet050 | side | 30 | 40.0% | 60.0% |
| coco-wholebody-val-edet050 | head_occluded | 144 | 36.1% | 54.2% |
| coco-wholebody-val-edet050 | empty | 13842 | 99.9% | 100.0% |
| coco-wholebody-val-rtmw-l-256x192 | front | 2730 | 63.4% | 95.2% |
| coco-wholebody-val-rtmw-l-256x192 | partial_body | 546 | 61.5% | 92.3% |
| coco-wholebody-val-rtmw-l-256x192 | back | 336 | 59.5% | 89.3% |
| coco-wholebody-val-rtmw-l-256x192 | distant | 498 | 44.2% | 66.3% |
| coco-wholebody-val-rtmw-l-256x192 | side | 30 | 40.0% | 60.0% |
| coco-wholebody-val-rtmw-l-256x192 | head_occluded | 144 | 61.1% | 91.7% |
| coco-wholebody-val-rtmw-l-256x192 | empty | 2250 | 99.8% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-edet030 | front | 2730 | 65.5% | 98.2% |
| coco-wholebody-val-rtmw-l-384x288-edet030 | partial_body | 546 | 63.0% | 94.5% |
| coco-wholebody-val-rtmw-l-384x288-edet030 | back | 336 | 60.7% | 91.1% |
| coco-wholebody-val-rtmw-l-384x288-edet030 | distant | 498 | 55.4% | 83.1% |
| coco-wholebody-val-rtmw-l-384x288-edet030 | side | 30 | 66.7% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-edet030 | head_occluded | 144 | 66.7% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-edet030 | empty | 13842 | 98.6% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-edet035 | front | 2730 | 65.2% | 97.8% |
| coco-wholebody-val-rtmw-l-384x288-edet035 | partial_body | 546 | 62.3% | 93.4% |
| coco-wholebody-val-rtmw-l-384x288-edet035 | back | 336 | 60.7% | 91.1% |
| coco-wholebody-val-rtmw-l-384x288-edet035 | distant | 498 | 53.8% | 80.7% |
| coco-wholebody-val-rtmw-l-384x288-edet035 | side | 30 | 66.7% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-edet035 | head_occluded | 144 | 63.9% | 95.8% |
| coco-wholebody-val-rtmw-l-384x288-edet035 | empty | 13842 | 99.0% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-edet050 | front | 2730 | 63.6% | 95.4% |
| coco-wholebody-val-rtmw-l-384x288-edet050 | partial_body | 546 | 60.1% | 90.1% |
| coco-wholebody-val-rtmw-l-384x288-edet050 | back | 336 | 59.5% | 89.3% |
| coco-wholebody-val-rtmw-l-384x288-edet050 | distant | 498 | 45.8% | 68.7% |
| coco-wholebody-val-rtmw-l-384x288-edet050 | side | 30 | 40.0% | 60.0% |
| coco-wholebody-val-rtmw-l-384x288-edet050 | head_occluded | 144 | 61.1% | 91.7% |
| coco-wholebody-val-rtmw-l-384x288-edet050 | empty | 13842 | 99.7% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | front | 2730 | 65.1% | 97.6% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | partial_body | 546 | 61.5% | 92.3% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | back | 336 | 64.3% | 96.4% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | distant | 498 | 59.4% | 89.2% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | side | 30 | 66.7% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | head_occluded | 144 | 63.9% | 95.8% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm075 | empty | 13842 | 99.5% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | front | 2730 | 64.6% | 96.9% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | partial_body | 546 | 60.1% | 90.1% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | back | 336 | 64.3% | 96.4% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | distant | 498 | 57.0% | 85.5% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | side | 30 | 66.7% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | head_occluded | 144 | 63.9% | 95.8% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm080 | empty | 13842 | 99.6% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | front | 2730 | 63.7% | 95.6% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | partial_body | 546 | 57.1% | 85.7% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | back | 336 | 61.9% | 92.9% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | distant | 498 | 50.6% | 75.9% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | side | 30 | 66.7% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | head_occluded | 144 | 61.1% | 91.7% |
| coco-wholebody-val-rtmw-l-384x288-yoloxm085 | empty | 13842 | 99.8% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | front | 2730 | 65.6% | 98.5% |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | partial_body | 546 | 64.5% | 96.7% |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | back | 336 | 65.5% | 98.2% |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | distant | 498 | 54.6% | 81.9% |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | side | 30 | 66.7% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | head_occluded | 144 | 58.3% | 87.5% |
| coco-wholebody-val-rtmw-l-384x288-yoloxt050 | empty | 13842 | 99.0% | 100.0% |
| coco-wholebody-val-rtmw-l-384x288 | front | 2730 | 63.4% | 95.2% |
| coco-wholebody-val-rtmw-l-384x288 | partial_body | 546 | 60.1% | 90.1% |
| coco-wholebody-val-rtmw-l-384x288 | back | 336 | 59.5% | 89.3% |
| coco-wholebody-val-rtmw-l-384x288 | distant | 498 | 45.8% | 68.7% |
| coco-wholebody-val-rtmw-l-384x288 | side | 30 | 40.0% | 60.0% |
| coco-wholebody-val-rtmw-l-384x288 | head_occluded | 144 | 61.1% | 91.7% |
| coco-wholebody-val-rtmw-l-384x288 | empty | 2250 | 100.0% | 100.0% |
| coco-wholebody-val-rtmw-m-256x192 | front | 2730 | 63.0% | 94.5% |
| coco-wholebody-val-rtmw-m-256x192 | partial_body | 546 | 60.8% | 91.2% |
| coco-wholebody-val-rtmw-m-256x192 | back | 336 | 58.3% | 87.5% |
| coco-wholebody-val-rtmw-m-256x192 | distant | 498 | 43.4% | 65.1% |
| coco-wholebody-val-rtmw-m-256x192 | side | 30 | 40.0% | 60.0% |
| coco-wholebody-val-rtmw-m-256x192 | head_occluded | 144 | 61.1% | 91.7% |
| coco-wholebody-val-rtmw-m-256x192 | empty | 2250 | 99.8% | 100.0% |
| coco-wholebody-val | front | 2730 | 58.6% | 87.9% |
| coco-wholebody-val | partial_body | 546 | 53.5% | 80.2% |
| coco-wholebody-val | back | 336 | 51.2% | 76.8% |
| coco-wholebody-val | distant | 498 | 21.7% | 32.5% |
| coco-wholebody-val | side | 30 | 40.0% | 60.0% |
| coco-wholebody-val | head_occluded | 144 | 36.1% | 54.2% |
| coco-wholebody-val | empty | 2250 | 99.8% | 100.0% |

## Failures

- coco-wholebody-val-edet035: Empty-frame false positives (1144.06 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-edet035: Torso tracking retention (80.4%, target >= 95%)
- coco-wholebody-val-edet035: Visible major-joint detection rate (73.0%, target >= 90%)
- coco-wholebody-val-edet050: Empty-frame false positives (476.69 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-edet050: Torso tracking retention (78.3%, target >= 95%)
- coco-wholebody-val-edet050: Visible major-joint detection rate (71.3%, target >= 90%)
- coco-wholebody-val-rtmw-l-256x192: Empty-frame false positives (264.50 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-rtmw-l-256x192: Torso tracking retention (90.6%, target >= 95%)
- coco-wholebody-val-rtmw-l-384x288-edet030: Empty-frame false positives (4576.22 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-rtmw-l-384x288-edet035: Empty-frame false positives (3241.49 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-rtmw-l-384x288-edet035: Torso tracking retention (94.7%, target >= 95%)
- coco-wholebody-val-rtmw-l-384x288-edet050: Empty-frame false positives (1144.06 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-rtmw-l-384x288-edet050: Torso tracking retention (90.8%, target >= 95%)
- coco-wholebody-val-rtmw-l-384x288-yoloxm075: Empty-frame false positives (1525.41 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-rtmw-l-384x288-yoloxm080: Empty-frame false positives (1239.39 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-rtmw-l-384x288-yoloxm080: Torso tracking retention (94.7%, target >= 95%)
- coco-wholebody-val-rtmw-l-384x288-yoloxm085: Empty-frame false positives (762.70 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-rtmw-l-384x288-yoloxm085: Torso tracking retention (91.7%, target >= 95%)
- coco-wholebody-val-rtmw-l-384x288-yoloxt050: Empty-frame false positives (3336.83 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-rtmw-l-384x288: Torso tracking retention (90.6%, target >= 95%)
- coco-wholebody-val-rtmw-m-256x192: Empty-frame false positives (264.50 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val-rtmw-m-256x192: Torso tracking retention (89.8%, target >= 95%)
- coco-wholebody-val-rtmw-m-256x192: Visible major-joint detection rate (89.8%, target >= 90%)
- coco-wholebody-val: Empty-frame false positives (264.50 / 12h, target < 1 per camera per 12h)
- coco-wholebody-val: Torso tracking retention (78.3%, target >= 95%)
- coco-wholebody-val: Visible major-joint detection rate (71.3%, target >= 90%)

## Not measured here

- GPU utilisation and browser-process memory. The replay harness runs in Node and cannot read them; they need the in-browser instrumentation described in `docs/tracking-eval.md`.
- Pose-model inference cost. This harness measures the fusion layer only, so model A/B numbers must come from the browser benchmark.
- 12-hour figures are extrapolated from scenario duration unless the scenario itself is a soak run.
