"""Opening book: the average rating of the players who reached each position, from training games.

Positions are keyed by their Zobrist hash, so different move orders that reach the same position
count together. Only positions reached in at least --min-games games are kept.

Usage: python opening_book.py games2 [--plies 24] [--min-games 20]
Writes model/opening_book.npz: sorted keys, game count, mean White and Black rating, rating sd.
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

from performance import Book, position_keys
from split import game_split

OUT = "model/opening_book.npz"


def scan(args):
    """(keys, White rating, Black rating) for every position in the first `plies` of train games."""
    path, plies = args
    df = pq.read_table(path, columns=["White", "Black", "WhiteElo", "BlackElo", "moves"]).to_pandas()
    df = df[game_split(df.White, df.Black).to_numpy() == "train"]
    keys, white, black = [], [], []
    for moves, w, b in zip(df.moves, df.WhiteElo, df.BlackElo):
        k = position_keys(list(moves[:plies]))
        keys.append(k)
        white.append(np.full(len(k), w, dtype=np.int16))
        black.append(np.full(len(k), b, dtype=np.int16))
    return np.concatenate(keys), np.concatenate(white), np.concatenate(black)


def coverage(book, folder, plies=(6, 10, 16, 20, 24), games=20_000):
    """Share of test games whose position after each ply is in the book."""
    path = sorted(glob.glob(f"{folder}/*.parquet"))[-1]
    df = pq.read_table(path, columns=["White", "Black", "moves"]).to_pandas()
    df = df[game_split(df.White, df.Black).to_numpy() == "test"].head(games)
    hits = {p: [] for p in plies}
    for moves in df.moves:
        found = book.find(position_keys(list(moves[:max(plies)])))
        for p in plies:
            if p < len(found):
                hits[p].append(found[p] >= 0)
    return {p: float(np.mean(h)) for p, h in hits.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", help="folder of prepared parquet games")
    ap.add_argument("--plies", type=int, default=24)
    ap.add_argument("--min-games", type=int, default=20)
    args = ap.parse_args()

    shards = sorted(glob.glob(f"{args.games}/*.parquet"))
    with Pool(min(4, os.cpu_count())) as pool:
        parts = pool.map(scan, [(path, args.plies) for path in shards])
    keys = np.concatenate([p[0] for p in parts])
    white = np.concatenate([p[1] for p in parts]).astype(np.float64)
    black = np.concatenate([p[2] for p in parts]).astype(np.float64)
    del parts
    unique, inverse, n = np.unique(keys, return_inverse=True, return_counts=True)
    del keys
    w_mean = np.bincount(inverse, white) / n
    b_mean = np.bincount(inverse, black) / n
    # sd of both players' ratings around their own side's mean
    sd = np.sqrt((np.bincount(inverse, (white - w_mean[inverse]) ** 2) +
                  np.bincount(inverse, (black - b_mean[inverse]) ** 2)) / (2 * n))
    keep = n >= args.min_games
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, keys=unique[keep], n=n[keep].astype(np.int32), white=w_mean[keep].astype(np.float32),
                        black=b_mean[keep].astype(np.float32), sd=sd[keep].astype(np.float32))
    print(f"{len(unique):,} positions, {keep.sum():,} reached in >= {args.min_games} games -> {OUT} "
          f"({os.path.getsize(OUT) / 1e6:.1f} MB)")
    for p, share in coverage(Book(OUT), args.games).items():
        print(f"  test games whose position after move {p / 2:g} is in the book: {share:.1%}")


if __name__ == "__main__":
    main()
