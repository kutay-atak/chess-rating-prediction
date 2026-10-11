"""Lichess shards -> rated blitz games with ratings, SAN moves, and clock and eval after each move.

Input: Parquet shards of the Hugging Face dataset Lichess/standard-chess-games (rated games only),
e.g. as downloaded by download.py.
Usage:
  python prepare_games.py OUT_DIR lichess/data/year=2025/month=09/*.parquet
"""
import math
import os
import re
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEEP = ["White", "Black", "WhiteElo", "BlackElo", "ECO", "TimeControl"]
COMMENTED_MOVE = re.compile(r"(\S+) \{([^}]*)\}")
CLOCK = re.compile(r"\[%clk (\d+):(\d+):([\d.]+)\]")
EVAL = re.compile(r"\[%eval (#?)(-?[\d.]+)\]")
NOT_A_MOVE = re.compile(r"\{[^}]*\}|\d+\.+|1-0|0-1|1/2-1/2|\*")  # comments, move numbers, results


def parse(movetext):
    """(SAN moves, seconds left after each move, eval after each move), or None unless every move has a clock.

    Evals are from White's view in pawns, clipped to ±10 with mate = ±10, NaN where a move has none
    (most games were never analysed)."""
    moves = COMMENTED_MOVE.findall(movetext)
    if not moves or len(moves) != len(NOT_A_MOVE.sub(" ", movetext).split()):
        return None
    sans, clocks, evals = [], [], []
    for san, comment in moves:
        clock, ev = CLOCK.search(comment), EVAL.search(comment)
        if clock is None:
            return None
        h, m, s = clock.groups()
        sans.append(san.rstrip("?!"))
        clocks.append(int(h) * 3600 + int(m) * 60 + float(s))
        if ev is None:
            evals.append(math.nan)
        else:  # past ±10 pawns the game is decided, so mates and huge evals share the cap
            evals.append(math.copysign(10, float(ev[2])) if ev[1] else max(-10.0, min(10.0, float(ev[2]))))
    return sans, clocks, evals


def prepare_frame(df):
    tc = df.TimeControl.str.extract(r"^(\d+)\+(\d+)$").astype(float)
    blitz = (tc[0] + 40 * tc[1]).between(180, 479)  # Lichess's own blitz bounds
    bot = df[["WhiteTitle", "BlackTitle"]].isin(["BOT"]).any(axis=1)
    parsed = df.movetext[blitz & ~bot].map(parse).dropna()
    return df.loc[parsed.index, KEEP].assign(
        moves=[p[0] for p in parsed], clocks=[p[1] for p in parsed], evals=[p[2] for p in parsed])


def batches(shard, size=50_000):
    """Prepared games of a shard, 50k at a time: a whole shard as pandas runs out of memory."""
    columns = KEEP + ["WhiteTitle", "BlackTitle", "movetext"]
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=size, columns=columns):
        yield prepare_frame(batch.to_pandas())


def prepare(shard):
    return pd.concat(batches(shard), ignore_index=True)


def write(shard, path):
    """Prepare one shard into `path`, batch by batch; returns the number of games kept."""
    source = pq.read_schema(shard)
    lists = [("moves", pa.string()), ("clocks", pa.float64()), ("evals", pa.float64())]
    schema = pa.schema([source.field(c) for c in KEEP] + [pa.field(c, pa.list_(t)) for c, t in lists])
    n = 0
    with pq.ParquetWriter(path, schema) as writer:
        for df in batches(shard):
            if len(df):  # an empty frame has no list types to cast from
                writer.write_table(pa.Table.from_pandas(df, preserve_index=False).cast(schema))
                n += len(df)
    return n


if __name__ == "__main__":
    out, *shards = sys.argv[1:]
    os.makedirs(out)  # refuses an existing folder, so old parts never mix in
    for i, shard in enumerate(shards):
        print(f"{shard}: {write(shard, f'{out}/part-{i:03d}.parquet')} games", flush=True)
