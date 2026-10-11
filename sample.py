"""Stratified sample of the prepared games, with a player-level train/val/test split.

Strata are 200-point bands of the two players' mean rating. Ratings pile up in the middle, so a
simple random sample would leave the tails, where models are most likely to be wrong, with few
games. Drawing up to 40 000 games per stratum covers every band; weight = N_h / n_h (games in the
stratum / games drawn) undoes the oversampling at evaluation.

The split (split.py) hashes each username to train/val/test. A side is scored only if its player
belongs to the game's split, so test results only ever come from players never seen in training.

Usage: python sample.py games    -> sample.parquet
"""
import glob
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from split import game_split, player_split

EDGES = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]  # open-ended first and last bands hold the thin tails
LABELS = ["<1000"] + [f"{a}-{b - 1}" for a, b in zip(EDGES, EDGES[1:])] + ["2400+"]
CAP = 40_000  # per stratum: a fraction of the dense middle bands, all of the thin tails
OUT = "sample.parquet"


def stratum(white_elo, black_elo):
    """Index into LABELS of each game's stratum, from the mean of the two ratings."""
    return np.searchsorted(EDGES, (np.asarray(white_elo) + np.asarray(black_elo)) / 2, side="right")


def draw(strata, cap=CAP, seed=0):
    """(drawn rows in ascending order, N_h, n_h): up to `cap` rows per stratum, without replacement."""
    rng = np.random.default_rng(seed)
    drawn, N, n = [], [], []
    for h in range(len(LABELS)):
        rows = np.flatnonzero(strata == h)
        drawn.append(rng.choice(rows, min(len(rows), cap), replace=False))
        N.append(len(rows))
        n.append(len(drawn[-1]))
    return np.sort(np.concatenate(drawn)), np.array(N), np.array(n)


def score_sides(white, black):
    """(split, score_white, score_black) per game. A side is scored only if its player belongs to the
    game's split: the train player in a test game is not, so test scores come from unseen players."""
    split = game_split(white, black)
    return split, white.map(player_split) == split, black.map(player_split) == split


def main(folder):
    shards = sorted(glob.glob(f"{folder}/part-*.parquet"))
    tables = [pq.read_table(p, columns=["White", "Black", "WhiteElo", "BlackElo"]).to_pandas() for p in shards]
    shard = np.concatenate([np.full(len(t), i) for i, t in enumerate(tables)])
    row = np.concatenate([np.arange(len(t)) for t in tables])
    games = pd.concat(tables, ignore_index=True)
    strata = stratum(games.WhiteElo, games.BlackElo)
    drawn, N, n = draw(strata)
    weight = N / np.maximum(n, 1)
    print(pd.DataFrame({"stratum": LABELS, "N_h": N, "n_h": n, "weight": weight.round(3)}).to_string(index=False))

    parts = [pq.read_table(path).take(row[drawn[shard[drawn] == i]]).to_pandas() for i, path in enumerate(shards)]
    sample = pd.concat(parts, ignore_index=True)  # same order as `drawn`: by shard, then by row
    sample.insert(0, "game_id", np.arange(len(sample)))
    sample["shard"] = shard[drawn]  # games/ part = downloaded shard, the cluster in the shard-level SE (evaluate.py)
    sample["row"] = row[drawn]  # row within the shard: with `shard`, the game's key in games/ (move_quality.py)
    sample["stratum"] = np.array(LABELS)[strata[drawn]]
    sample["weight"] = weight[strata[drawn]]
    sample["split"], sample["score_white"], sample["score_black"] = score_sides(sample.White, sample.Black)

    split = game_split(games.White, games.Black)
    train_players = set(games.White[split == "train"].str.lower()) | set(games.Black[split == "train"].str.lower())
    test = sample[sample.split == "test"]
    scored_test = set(test.White[test.score_white].str.lower()) | set(test.Black[test.score_black].str.lower())
    assert not scored_test & train_players, "a scored test player appears in a train game"
    print(f"{len(scored_test):,} scored test players, none of them in any of the {(split == 'train').sum():,} train games")
    print(sample.split.value_counts().to_string())

    sample.to_parquet(OUT, index=False)
    print(f"{len(sample):,} games -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
