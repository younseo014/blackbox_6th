"""Prepare pinned MIT CLIP weights and fixed text prototypes; no participant data."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work/action-models/clip"
OUTPUT = ROOT / "public/models/action-context"
REVISION = "6b7c523703ae26e39be4b03774966f8571c7f523"


def main():
    definitions = json.loads((ROOT / "config/cafe-action-prompts.json").read_text())
    tokenizer = Tokenizer.from_file(str(SOURCE / "tokenizer.json"))
    tokenizer.enable_padding(pad_id=49407, pad_token="<|endoftext|>")
    tokenizer.enable_truncation(max_length=77)
    session = ort.InferenceSession(str(SOURCE / "onnx/text_model_quantized.onnx"), providers=["CPUExecutionProvider"])
    prototypes = []
    for definition in definitions:
        tokens = tokenizer.encode_batch(definition["prompts"])
        features = session.run(None, {"input_ids": np.asarray([item.ids for item in tokens], dtype=np.int64)})[0]
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        prototype = features.mean(axis=0)
        prototype /= np.linalg.norm(prototype)
        prototypes.append({**definition, "embedding": prototype.tolist()})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    weights = SOURCE / "onnx/vision_model_quantized.onnx"
    metadata = {
        "model": "clip-vit-base-patch32-int8",
        "revision": REVISION,
        "source": "https://huggingface.co/Xenova/clip-vit-base-patch32",
        "license": "MIT",
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "imageSize": 224,
        "normalization": json.loads((SOURCE / "preprocessor_config.json").read_text()),
        "prototypes": prototypes,
        "limitations": "Zero-shot image context, not a cafe-trained action or diagnostic model. Scores are not calibrated probabilities.",
    }
    (OUTPUT / "prototypes.json").write_text(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
    shutil.copyfile(weights, OUTPUT / "vision-int8.onnx")
    print(f"Prepared {len(prototypes)} prototypes, {weights.stat().st_size} model bytes, sha256={metadata['sha256']}")


if __name__ == "__main__":
    main()
