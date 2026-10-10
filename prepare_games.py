"""Lichess shards -> rated blitz games with ratings, SAN moves and clock after each move.

Input: Parquet shards of the Hugging Face dataset Lichess/standard-chess-games (rated games only):
  hf download Lichess/standard-chess-games --repo-type dataset \
      --include "data/year=2025/month=09/train-0000*" --local-dir lichess
Usage:
  python prepare_games.py OUT_DIR lichess/data/year=2025/month=09/*.parquet
Shards are read and written in batches, so memory stays well under a shard's size.
"""
import os
import re
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEEP = ["White", "Black", "WhiteElo", "BlackElo", "ECO", "TimeControl"]
READ = KEEP + ["WhiteTitle", "BlackTitle", "movetext"]
CLOCKED_MOVE = re.compile(r"(\S+) \{[^}]*\[%clk (\d+):(\d+):([\d.]+)\]")
NOT_A_MOVE = re.compile(r"\{[^}]*\}|\d+\.+|1-0|0-1|1/2-1/2|\*")  # comments, move numbers, results


def parse(movetext):
    """(SAN moves, seconds left after each move), or None unless every move has a clock."""
    pairs = CLOCKED_MOVE.findall(movetext)
    if not pairs or len(pairs) != len(NOT_A_MOVE.sub(" ", movetext).split()):
        return None
    return ([san.rstrip("?!") for san, *_ in pairs],
            [int(h) * 3600 + int(m) * 60 + float(s) for _, h, m, s in pairs])


def prepare_frame(df):
    tc = df.TimeControl.str.extract(r"^(\d+)\+(\d+)$").astype(float)
    blitz = (tc[0] + 40 * tc[1]).between(180, 479)  # Lichess's own blitz bounds
    bot = df[["WhiteTitle", "BlackTitle"]].isin(["BOT"]).any(axis=1)
    parsed = df.movetext[blitz & ~bot].map(parse).dropna()
    return df.loc[parsed.index, KEEP].assign(
        moves=[p[0] for p in parsed], clocks=[p[1] for p in parsed])


def batches(shard, size=50_000):
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=size, columns=READ):
        yield prepare_frame(batch.to_pandas())


def prepare(shard):
    return pd.concat(batches(shard), ignore_index=True)


def write(shard, path):
    """Prepare one shard into `path` batch by batch; returns the number of games kept."""
    source = pq.read_schema(shard)
    schema = pa.schema([source.field(c) for c in KEEP] +
                       [pa.field("moves", pa.list_(pa.string())), pa.field("clocks", pa.list_(pa.float64()))])
    n = 0
    with pq.ParquetWriter(path, schema) as writer:
        for df in batches(shard):
            writer.write_table(pa.Table.from_pandas(df, preserve_index=False).cast(schema))
            n += len(df)
    return n


if __name__ == "__main__":
    out, *shards = sys.argv[1:]
    os.makedirs(out)  # refuses an existing folder, so old parts never mix in
    for i, shard in enumerate(shards):
        print(f"{shard}: {write(shard, f'{out}/part-{i:03d}.parquet')} games", flush=True)
