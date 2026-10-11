"""Per player-game indicators of accuracy and time use, for the latent skill model (skill_sem.R).

From games in games/ that Lichess analysed: one random game per player (seed 0), and at most one
player per game, so the rows are independent. Indicators over that player's own moves (conventions
in CLAUDE.md): log average loss, mistake rate, blunder rate, mean and sd of log seconds spent, and
the share of moves made with under 10% of base time left; plus rating and TimeControl.

Usage: python sem_data.py   -> sem_data.csv
"""
import glob

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from features import own_moves

MIN_MOVES = 10  # rates over fewer moves with a loss are mostly noise
LOW_CLOCK = 0.1  # "under 10% of base time left", as PLAN.md sets
OUT = "sem_data.csv"


def indicators(clocks, evals, base, inc, side):
    """Indicators of one side's play, or None if it has fewer than MIN_MOVES moves with a loss."""
    spent, loss, left, _ = own_moves(np.asarray(clocks, float), np.asarray(evals, float), base, inc, side)
    loss = loss[~np.isnan(loss)]
    if len(loss) < MIN_MOVES:
        return None
    log_spent = np.log1p(spent)
    return {"log_avg_loss": np.log1p(loss.mean()),  # log1p: losses are skewed and can be 0
            "mistake_rate": np.mean(loss >= 1), "blunder_rate": np.mean(loss >= 3),
            "mean_log_spent": log_spent.mean(), "sd_log_spent": log_spent.std(ddof=1),
            "low_clock_share": np.mean(left < LOW_CLOCK * (base if base > 0 else 40 * inc))}  # 0+N: 40-move estimate


def one_per_player(cands, seed=0):
    """One random row per player, then at most one row per game, so no two rows share a player or a game."""
    shuffled = cands.sample(frac=1, random_state=seed)
    return shuffled.drop_duplicates("player").drop_duplicates("game").sort_values(["game", "side"])


def analysed(table):
    """Boolean per game: does it have any eval?"""
    flat = pc.list_flatten(table["evals"]).to_numpy(zero_copy_only=False)
    parents = pc.list_parent_indices(table["evals"]).to_numpy()
    return np.bincount(parents[~np.isnan(flat)], minlength=len(table)) > 0


def main():
    rows = []
    for shard, path in enumerate(sorted(glob.glob("games/part-*.parquet"))):
        table = pq.read_table(path, columns=["White", "Black", "WhiteElo", "BlackElo", "TimeControl", "clocks", "evals"])
        keep = np.flatnonzero(analysed(table))
        games = table.take(keep).to_pandas()
        for i, g in enumerate(games.itertuples(index=False)):
            base, inc = map(int, g.TimeControl.split("+"))
            for side, (player, elo) in enumerate([(g.White, g.WhiteElo), (g.Black, g.BlackElo)]):
                ind = indicators(g.clocks, g.evals, base, inc, side)
                if ind:
                    rows.append({"game": f"{shard}-{keep[i]}", "side": side, "player": player.lower(),
                                 "rating": elo / 100, "tc": g.TimeControl, **ind})  # hundreds: variances of similar size
        print(f"{path}: {len(keep):,} analysed games", flush=True)
    data = one_per_player(pd.DataFrame(rows))
    data.drop(columns=["game", "side", "player"]).to_csv(OUT, index=False)
    print(f"{len(data):,} player-games -> {OUT}; top time controls:\n{data.tc.value_counts().head(5).to_string()}")


if __name__ == "__main__":
    main()
