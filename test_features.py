import numpy as np

from features import MAX_PLIES, START, UNK, build_vocab, encode, numeric_features, time_control, to_ratings, vocab_index


def test_time_control():
    assert time_control("180+2") == (180, 2)


def test_build_vocab_drops_rare_moves():
    assert build_vocab([["e4", "e5"], ["e4", "c5"]], min_count=2) == ["e4"]


def test_encode():
    index = vocab_index(["e4", "e5"])
    tokens, numeric = encode(["e4", "e5", "Qh5"], [180, 178, 175], 180, 2, index)
    assert tokens.tolist() == [START, index["e4"], index["e5"], UNK]
    assert numeric.shape == (4, 9) and numeric.dtype == np.float32
    assert numeric[0, 7] == 1 and numeric[1:, 7].sum() == 0  # start flag
    assert numeric[1:, 0].tolist() == [1, -1, 1]  # mover
    # time spent: first move per side counts from the base time; White's 2nd move 180 + 2 - 175 = 7s
    np.testing.assert_allclose(numeric[1:, 3], np.log1p([2, 4, 7]) / 3, rtol=1e-6)
    np.testing.assert_allclose(numeric[1:, 8], np.array([180, 180, 178]) / 260, rtol=1e-6)  # opponent's clock


def test_encode_truncates():
    tokens, numeric = encode(["e4"] * 400, [180.0] * 400, 180, 0, {})
    assert len(tokens) == len(numeric) == MAX_PLIES + 1


def test_numeric_features_never_negative_time_spent():
    numeric = numeric_features([180, 180, 181, 179], 180, 0)  # clock went up with no increment
    assert (numeric[:, 3] >= 0).all()


def test_to_ratings():
    raw = np.zeros((2, 5))
    raw[1] = [1, -1, np.log(0.5), 0, 0]
    meta = {"center": 1500, "scale": 300, "calibration": [1.0, 2.0]}
    out = to_ratings(raw, meta)
    assert out[0]["white"] == {"rating": 1500, "low": 1500 - 384, "high": 1500 + 384}
    assert out[1]["white"]["rating"] == 1800 and out[1]["black"]["rating"] == 1200
    assert out[1]["white"]["high"] - out[1]["white"]["rating"] == round(1.2816 * 2 * 150)


def test_numeric_features_finite_with_zero_base_time():
    assert np.isfinite(numeric_features([0, 0, 3, 4], 0, 5)).all()  # 0+5 is a blitz time control
