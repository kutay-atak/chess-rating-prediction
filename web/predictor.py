"""Runs the exported rating model on a game in progress."""
import json
from pathlib import Path

import onnxruntime as ort

from features import EVAL_FEATURES, encode, to_ratings, vocab_index


class Predictor:
    def __init__(self, folder):
        folder = Path(folder)
        self.meta = json.loads((folder / "meta.json").read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(folder / "rating_model.onnx"), options)
        self.index = vocab_index(self.meta["vocab"])
        self.n_numeric = self.session.get_inputs()[1].shape[2]
        self.uses_evals = self.n_numeric > EVAL_FEATURES.start

    def __call__(self, sans, clocks, base, inc, evals=None):
        """Estimates after each ply so far: [{ply, white: {rating, low, high}, black: {...}}, ...].

        evals: engine eval after each ply, or None to estimate from moves and clocks alone."""
        tokens, numeric = encode(sans, clocks, base, inc, self.index, evals)
        numeric = numeric[:, :self.n_numeric]  # a model trained without evals takes fewer features
        raw = self.session.run(None, {"tokens": tokens[None], "numeric": numeric[None]})[0][0]
        return to_ratings(raw, self.meta, has_evals=evals is not None and self.uses_evals)
