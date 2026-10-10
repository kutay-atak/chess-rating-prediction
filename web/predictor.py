"""Runs the exported rating model on a game in progress."""
import json
from pathlib import Path

import onnxruntime as ort

from features import encode, to_ratings, vocab_index


class Predictor:
    def __init__(self, folder):
        folder = Path(folder)
        self.meta = json.loads((folder / "meta.json").read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(folder / "rating_model.onnx"), options)
        self.index = vocab_index(self.meta["vocab"])

    def __call__(self, sans, clocks, base, inc):
        """Estimates after each ply so far: [{ply, white: {rating, low, high}, black: {...}}, ...]."""
        tokens, numeric = encode(sans, clocks, base, inc, self.index)
        raw = self.session.run(None, {"tokens": tokens[None], "numeric": numeric[None]})[0][0]
        return to_ratings(raw, self.meta)
