"""Download 12 random September 2025 shards of Lichess/standard-chess-games into lichess/.

Shards are probably consecutive chunks of the month, so random ones, rather than the first ones,
make a cluster sample of the whole month.

Usage: python download.py
"""
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

REPO, MONTH, N_SHARDS = "Lichess/standard-chess-games", "data/year=2025/month=09/", 12


def pick(files, n=N_SHARDS, seed=0):
    shards = sorted(f for f in files if f.startswith(MONTH) and f.endswith(".parquet"))
    return sorted(np.random.default_rng(seed).choice(shards, n, replace=False))


if __name__ == "__main__":
    for shard in pick(list_repo_files(REPO, repo_type="dataset")):
        print(hf_hub_download(REPO, shard, repo_type="dataset", local_dir="lichess"), flush=True)
