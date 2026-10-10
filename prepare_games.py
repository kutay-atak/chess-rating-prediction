"""Lichess shards -> rated blitz games with ratings, SAN moves and clock after each move.

Input: Parquet shards of the Hugging Face dataset Lichess/standard-chess-games (rated games only):
  hf download Lichess/standard-chess-games --repo-type dataset \
      --include "data/year=2025/month=09/train-0000*" --local-dir lichess
Usage:
  python prepare_games.py OUT_DIR lichess/data/year=2025/month=09/*.parquet
"""
import os
import re
import sys

import pandas as pd

KEEP = ["White", "Black", "WhiteElo", "BlackElo", "ECO", "TimeControl"]
CLOCKED_MOVE = re.compile(r"(\S+) \{[^}]*\[%clk (\d+):(\d+):([\d.]+)\]")
NOT_A_MOVE = re.compile(r"\{[^}]*\}|\d+\.+|1-0|0-1|1/2-1/2|\*")  # comments, move numbers, results


def parse(movetext):
    """(SAN moves, seconds left after each move), or None unless every move has a clock."""
    pairs = CLOCKED_MOVE.findall(movetext)
    if not pairs or len(pairs) != len(NOT_A_MOVE.sub(" ", movetext).split()):
        return None
    return ([san.rstrip("?!") for san, *_ in pairs],
            [int(h) * 3600 + int(m) * 60 + float(s) for _, h, m, s in pairs])


def prepare(shard):
    df = pd.read_parquet(shard, columns=KEEP + ["WhiteTitle", "BlackTitle", "movetext"])
    tc = df.TimeControl.str.extract(r"^(\d+)\+(\d+)$").astype(float)
    blitz = (tc[0] + 40 * tc[1]).between(180, 479)  # Lichess's own blitz bounds
    bot = df[["WhiteTitle", "BlackTitle"]].isin(["BOT"]).any(axis=1)
    parsed = df.movetext[blitz & ~bot].map(parse).dropna()
    return df.loc[parsed.index, KEEP].assign(
        moves=[p[0] for p in parsed], clocks=[p[1] for p in parsed])


if __name__ == "__main__":
    out, *shards = sys.argv[1:]
    os.makedirs(out)  # refuses an existing folder, so old parts never mix in
    for i, shard in enumerate(shards):
        df = prepare(shard)
        df.to_parquet(f"{out}/part-{i:03d}.parquet", index=False)
        print(f"{shard}: {len(df)} games")
