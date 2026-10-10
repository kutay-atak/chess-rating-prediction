"""Checkpoint -> rating_model.onnx + meta.json, which the web server loads (from model/ by default).

meta.json holds the vocabulary, the rating normalisation and the interval calibration for games
with and without engine evals (all ones until evaluate.py fits them on val games).

Usage: python export_onnx.py [checkpoints/best.pt] [--out model]
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch

from features import MAX_PLIES
from train import load_model

ONNX, META = "rating_model.onnx", "meta.json"


def export(checkpoint, out):
    ckpt = torch.load(checkpoint, weights_only=False)
    model = load_model(ckpt, dropout=0.0)
    model.eval()
    n_numeric, vocab_size = model.numeric.in_features, model.move.num_embeddings

    os.makedirs(out, exist_ok=True)
    onnx_path, meta_path = f"{out}/{ONNX}", f"{out}/{META}"
    tokens, numeric = torch.randint(3, vocab_size, (2, 7)), torch.randn(2, 7, n_numeric)
    torch.onnx.export(model, (tokens, numeric), onnx_path, input_names=["tokens", "numeric"], output_names=["out"],
                      dynamic_axes={"tokens": {0: "batch", 1: "plies"}, "numeric": {0: "batch", 1: "plies"},
                                    "out": {0: "batch", 1: "plies"}}, opset_version=17, dynamo=False)

    # the exported graph must match torch on a different shape than it was traced with
    tokens, numeric = torch.randint(3, vocab_size, (3, MAX_PLIES + 1)), torch.randn(3, MAX_PLIES + 1, n_numeric)
    with torch.no_grad():
        expected = model(tokens, numeric).numpy()
    got = ort.InferenceSession(onnx_path).run(None, {"tokens": tokens.numpy(), "numeric": numeric.numpy()})[0]
    np.testing.assert_allclose(got, expected, atol=1e-4)

    ones = [1.0] * (MAX_PLIES + 1)
    meta = dict(vocab=ckpt["vocab"], center=ckpt["center"], scale=ckpt["scale"], n_numeric=n_numeric,
                calibration={"evals": ones, "no_evals": ones}, step=ckpt["step"])
    if os.path.exists(meta_path):  # keep a calibration fitted earlier for this same checkpoint
        old = json.load(open(meta_path))
        if old.get("step") == meta["step"] and old.get("vocab") == meta["vocab"] and isinstance(old.get("calibration"), dict):
            meta["calibration"] = old["calibration"]
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"wrote {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB) and {meta_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="checkpoints/best.pt")
    ap.add_argument("--out", default="model")
    args = ap.parse_args()
    export(args.checkpoint, args.out)
