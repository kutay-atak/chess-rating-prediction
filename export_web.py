"""Export the GRU in model.pt to web/model.json for the in-browser demo (web/index.html).

Usage: python export_web.py
"""
import json

import numpy as np
import torch

DIGITS = 5  # significant digits: shifts predictions by far less than a rating point and keeps the file near 3 MB


def export(path="model.pt", out="web/model.json"):
    saved = torch.load(path)
    weights = {k: np.asarray(v, dtype=float) for k, v in saved["state_dict"].items()}
    data = {"vocab": saved["vocab"], "rating_mean": saved["rating_mean"], "rating_sd": saved["rating_sd"],
            "conformal": {str(k): v for k, v in saved["conformal"].items()},
            "weights": {k: {"shape": list(v.shape), "data": [float(f"{x:.{DIGITS}g}") for x in v.ravel()]}
                        for k, v in weights.items()}}
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))


if __name__ == "__main__":
    export()
