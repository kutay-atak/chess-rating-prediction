"""checkpoints/best.pt -> model/rating_model.onnx + model/meta.json, which the web server loads.

meta.json holds the vocabulary, the rating normalisation and the interval calibration (all ones
until evaluate.py fits it on val games).

Usage: python export_onnx.py
"""
import json
import os

import numpy as np
import onnxruntime as ort
import torch

from features import MAX_PLIES, N_NUMERIC
from train import CHECKPOINT, RatingModel

ONNX, META = "model/rating_model.onnx", "model/meta.json"


def main():
    ckpt = torch.load(CHECKPOINT, weights_only=False)
    model = RatingModel(**ckpt["config"], dropout=0.0)
    model.load_state_dict(ckpt["state"])
    model.eval()

    os.makedirs("model", exist_ok=True)
    tokens = torch.randint(3, ckpt["config"]["vocab_size"], (2, 7))
    numeric = torch.randn(2, 7, N_NUMERIC)
    torch.onnx.export(model, (tokens, numeric), ONNX, input_names=["tokens", "numeric"], output_names=["out"],
                      dynamic_axes={"tokens": {0: "batch", 1: "plies"}, "numeric": {0: "batch", 1: "plies"},
                                    "out": {0: "batch", 1: "plies"}}, opset_version=17, dynamo=False)

    # the exported graph must match torch on a different shape than it was traced with
    tokens = torch.randint(3, ckpt["config"]["vocab_size"], (3, MAX_PLIES + 1))
    numeric = torch.randn(3, MAX_PLIES + 1, N_NUMERIC)
    with torch.no_grad():
        expected = model(tokens, numeric).numpy()
    got = ort.InferenceSession(ONNX).run(None, {"tokens": tokens.numpy(), "numeric": numeric.numpy()})[0]
    np.testing.assert_allclose(got, expected, atol=1e-4)

    meta = dict(vocab=ckpt["vocab"], center=ckpt["center"], scale=ckpt["scale"],
                calibration=[1.0] * (MAX_PLIES + 1), step=ckpt["step"], val_nll=ckpt["val_nll"])
    if os.path.exists(META):  # keep a calibration fitted earlier for this same checkpoint
        old = json.load(open(META))
        if old.get("step") == meta["step"] and old.get("vocab") == meta["vocab"]:
            meta["calibration"] = old["calibration"]
    with open(META, "w") as f:
        json.dump(meta, f)
    print(f"wrote {ONNX} ({os.path.getsize(ONNX) / 1e6:.1f} MB) and {META}")


if __name__ == "__main__":
    main()
