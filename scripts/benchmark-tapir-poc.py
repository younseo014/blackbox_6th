"""Small, reproducible CPU probe for the official causal TAPIR checkpoint.

Run with PYTHONPATH pointing to a clone of google-deepmind/tapnet. This is an
evaluation aid, not an app backend: TAP-Vid labels surface points, not joints.
"""

import argparse
import json
import pickle
import time

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from tapnet.models import tapir_model
from tapnet.utils import model_utils


def evaluate(args):
    with open(args.dataset, "rb") as file:
        data = pickle.load(file)[args.clip]
    video = data["video"][: args.frames]
    ground = data["points"][:, : args.frames]
    occluded = data["occluded"][:, : args.frames]
    selected = np.flatnonzero(~occluded[:, 0])[: args.points]
    ground, occluded = ground[selected], occluded[selected]
    frames = np.stack([cv2.resize(frame, (256, 256)) for frame in video])
    query = np.stack(
        [
            np.zeros(len(selected), dtype=np.float32),
            ground[:, 0, 1] * 256,
            ground[:, 0, 0] * 256,
        ],
        axis=-1,
    )[None]

    checkpoint = np.load(args.checkpoint, allow_pickle=True).item()
    tapir = tapir_model.ParameterizedTAPIR(
        params=checkpoint["params"],
        state=checkpoint["state"],
        tapir_kwargs=dict(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False),
    )

    def init(frame, points):
        grids = tapir.get_feature_grids(frame, is_training=False)
        return tapir.get_query_features(
            frame, is_training=False, query_points=points, feature_grids=grids
        )

    def predict(frame, features, context):
        grids = tapir.get_feature_grids(frame, is_training=False)
        result = tapir.estimate_trajectories(
            frame.shape[-3:-1],
            is_training=False,
            feature_grids=grids,
            query_features=features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=context,
            get_causal_context=True,
        )
        context = result.pop("causal_context")
        return {key: val[-1] for key, val in result.items()}, context

    init, predict = jax.jit(init), jax.jit(predict)
    prepared = [model_utils.preprocess_frames(frame[None, None]) for frame in frames]
    started = time.perf_counter()
    features = init(prepared[0], jnp.asarray(query))
    jax.block_until_ready(features)
    init_ms = (time.perf_counter() - started) * 1000
    context = tapir.construct_initial_causal_state(
        len(selected), len(features.resolutions) - 1
    )

    times, errors, visible_correct, occluded_correct, last = [], [], [], [], []
    for index, frame in enumerate(prepared):
        started = time.perf_counter()
        result, context = predict(frame, features, context)
        jax.block_until_ready(result["tracks"])
        times.append((time.perf_counter() - started) * 1000)
        tracks = np.asarray(result["tracks"])[0, :, 0]
        predicted_visible = np.asarray(
            model_utils.postprocess_occlusions(
                result["occlusion"], result["expected_dist"]
            )
        )[0, :, 0]
        point_error = np.linalg.norm(tracks - ground[:, index] * 256, axis=-1)
        visible = ~occluded[:, index]
        errors.extend(point_error[visible].tolist())
        visible_correct.extend(predicted_visible[visible].tolist())
        occluded_correct.extend((~predicted_visible[~visible]).tolist())
        last.append(
            {
                "frame": index,
                "visible_points": int(visible.sum()),
                "within_8px_pct": round(float(np.mean(point_error[visible] < 8) * 100), 1)
                if np.any(visible)
                else None,
            }
        )

    return {
        "clip": args.clip,
        "frames": len(frames),
        "initially_visible_points": len(selected),
        "device": str(jax.devices()[0]),
        "resolution": "256x256",
        "initialization_ms": round(init_ms, 1),
        "first_prediction_ms": round(times[0], 1),
        "steady_p50_ms": round(float(np.median(times[1:])), 1),
        "steady_p95_ms": round(float(np.percentile(times[1:], 95)), 1),
        "visible_point_median_error_px": round(float(np.median(errors)), 2),
        "visible_point_within_8px_pct": round(float(np.mean(np.asarray(errors) < 8) * 100), 1),
        "visible_recall_pct": round(float(np.mean(visible_correct) * 100), 1),
        "occlusion_recall_pct": round(float(np.mean(occluded_correct) * 100), 1)
        if occluded_correct
        else None,
        "last_10_frames": last[-10:],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="TAP-Vid DAVIS .pkl")
    parser.add_argument("--checkpoint", required=True, help="Official causal_tapir_checkpoint.npy")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--points", type=int, default=22)
    print(json.dumps(evaluate(parser.parse_args()), indent=2))
