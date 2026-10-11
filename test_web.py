import json
import shutil
import subprocess

import numpy as np
import pytest

from demo import bands, load_model, read_pgn
from gru import ply_inputs

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node")


def node(script):
    return json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True,
                                     cwd="web").stdout)


def test_ply_inputs_match_python():
    clocks = [90, 180, 88, 177, 87, 175]  # White berserks in 3+2
    js = node(f"const m = require('./model.js'); console.log(JSON.stringify("
              f"{clocks}.map((_, i) => m.plyInput({clocks}, i, 180, 2))))")
    assert np.allclose(js, ply_inputs(clocks, 180, 2), atol=1e-6)


def test_bands_match_python_on_the_example_game():
    if not __import__("os").path.exists("web/model.json"):
        pytest.skip("run export_web.py first")
    _, sans, clocks, base, inc = read_pgn(open("example.pgn").read())
    js = np.array(node("const m = require('./model.js'); const fs = require('fs');"
                       "const g = m.parsePgn(fs.readFileSync('../example.pgn', 'utf8'));"
                       "const net = m.makeModel(JSON.parse(fs.readFileSync('model.json', 'utf8')));"
                       f"console.log(JSON.stringify([g.sans.length, net.run(g.sans, g.clocks, {base}, {inc})]))")[1])
    py = bands(*load_model(), sans, clocks, base, inc)
    assert js.shape == py.shape
    assert np.abs(js - py).max() < 0.5  # rating points, after rounding the weights to 5 digits
