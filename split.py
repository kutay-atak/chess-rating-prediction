"""Player-level train/val/test split.

Each player lands in one bucket by a stable hash of their (lowercased) username: 80% train, 10% val,
10% test. A game is test if either player is a test player, else val if either is a val player, else
train. So no val or test player is ever seen in training, and no game is thrown away.
"""
import hashlib

import pandas as pd

ORDER = ["train", "val", "test"]


def player_split(name):
    bucket = int(hashlib.md5(name.lower().encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def game_split(white, black):
    """Split of each game, given Series of White and Black usernames."""
    rank = {s: i for i, s in enumerate(ORDER)}
    worst = pd.concat([white.map(player_split).map(rank), black.map(player_split).map(rank)], axis=1).max(axis=1)
    return worst.map(dict(enumerate(ORDER)))
