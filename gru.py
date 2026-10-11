"""GRU over the plies of a game: after every ply, a mean and sd for both players' ratings.

Inputs per ply (ply_inputs): the move as a SAN embedding; the mover's log seconds spent; their clock
left as a share of base; who moved; base and increment. No engine evals: Lichess has them only for
games someone had analysed after they ended, so they reveal a later event and never exist in a live game.
Output per ply: mean and log-variance of White's and Black's standardised rating. Loss: Gaussian
negative log-likelihood, weighted by N_h/n_h like every fit in baselines.py (the strata are bands of
the very ratings being predicted). Early stopping on the weighted val NLL. Intervals: split conformal
on val sides, scaled by the model's own sd (conformal()).

Usage: python gru.py [n_train_games]   (reads sample.parquet; writes model.pt and results/preds_gru.parquet)
"""
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

from baselines import weighted_quantile
from features import CHECKPOINTS, own_moves

N_INPUTS = 5
MAX_PLIES = 200  # 99.9% of games are shorter; the rest would only add padding
MIN_COUNT = 20  # rarer moves get noisy embeddings, so they share the unknown token
PAD, UNK = 0, 1
BATCH = 256
LR = 2e-3  # only a few epochs fit in the 30-minute budget, so a step above Adam's default 1e-3
MAX_EPOCHS = 7  # about 2.5 min each on 4 CPUs: the full run must stay under 30 minutes
PATIENCE = 2
CLIP = 500.0  # safety net only: on 20k train games the gradient norm had a 99th percentile of 356
LEVEL = 0.9  # 90% intervals
COLUMNS = ["game_id", "WhiteElo", "BlackElo", "TimeControl", "moves", "clocks", "weight", "split",
           "score_white", "score_black"]


def ply_inputs(clocks, base, inc):
    """[plies, N_INPUTS] float32; row i describes ply i and its mover (conventions in CLAUDE.md)."""
    clocks = np.asarray(clocks, dtype=float)
    x = np.zeros((len(clocks), N_INPUTS), dtype=np.float32)
    for side in (0, 1):
        spent, _, left, _ = own_moves(clocks, np.full(len(clocks), np.nan), base, inc, side)
        x[side::2, 0] = np.log1p(spent)  # log1p: 0-second moves are common
        x[side::2, 1] = np.clip(left / (base if base > 0 else 40 * inc), 0, 2)  # 0+N games: Lichess's 40-move estimate
        x[side::2, 2] = side
    x[:, 3], x[:, 4] = base / 180, inc / 5  # scaled to about 1
    return x


def vocab(move_lists, min_count=MIN_COUNT):
    """SAN moves seen at least min_count times; move i of the list gets id i + 2 (0 pad, 1 unknown)."""
    counts = Counter(m for moves in move_lists for m in moves)
    return sorted(m for m, c in counts.items() if c >= min_count)


def encode(moves, index):
    return np.array([index.get(m, UNK) for m in moves], dtype=np.int32)


def load(games, index):
    """(move ids, inputs, offsets): every game's first MAX_PLIES plies, flat; game i is offsets[i]:offsets[i+1]."""
    lengths = np.minimum(games.moves.map(len).to_numpy(), MAX_PLIES)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    moves, x = np.empty(offsets[-1], dtype=np.int32), np.empty((offsets[-1], N_INPUTS), dtype=np.float32)
    for i, g in enumerate(games[["moves", "clocks", "TimeControl"]].itertuples(index=False)):
        base, inc = map(int, g.TimeControl.split("+"))
        a, b = offsets[i], offsets[i + 1]
        moves[a:b] = encode(g.moves[:b - a], index)
        x[a:b] = ply_inputs(g.clocks[:b - a], base, inc)
    return moves, x, offsets


def batch(data, idx):
    """Games idx, right-padded: move ids [B, T], inputs [B, T, N_INPUTS], lengths [B]."""
    moves, x, offsets = data
    lengths = offsets[idx + 1] - offsets[idx]
    m = np.zeros((len(idx), lengths.max()), dtype=np.int64)
    xx = np.zeros((len(idx), lengths.max(), N_INPUTS), dtype=np.float32)
    for j, (i, n) in enumerate(zip(idx, lengths)):
        m[j, :n], xx[j, :n] = moves[offsets[i]:offsets[i] + n], x[offsets[i]:offsets[i] + n]
    return torch.from_numpy(m), torch.from_numpy(xx), torch.from_numpy(lengths)


def batches(lengths, rng=None, size=BATCH):
    """Positions into `lengths`, in batches of similar length (less padding), in random order when rng is given."""
    noise = rng.uniform(0, 8, len(lengths)) if rng is not None else 0  # jitter, so batches differ between epochs
    order = np.argsort(lengths + noise, kind="stable")
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    if rng is not None:
        rng.shuffle(chunks)
    return chunks


class GRU(torch.nn.Module):
    def __init__(self, n_moves, emb=64, hidden=128):
        super().__init__()
        self.emb = torch.nn.Embedding(n_moves, emb, padding_idx=PAD)
        self.gru = torch.nn.GRU(emb + N_INPUTS, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 4)

    def forward(self, moves, x):
        """Mean and log-variance of the standardised (White, Black) ratings after each ply: two [B, T, 2]."""
        out = self.head(self.gru(torch.cat([self.emb(moves), x], -1))[0])
        return out[..., [0, 2]], out[..., [1, 3]].clamp(-7, 3)  # sd between 0.03 and 4.5 rating sds


def nll(mean, logvar, y, keep, w):
    """Weighted Gaussian NLL (without the constant) summed over the kept (game, ply, side) cells, per game.

    Summed over plies rather than averaged per game: game length tracks rating, and the sum counts each
    game at every t it reaches, as evaluate.py does."""
    cell = 0.5 * (logvar + (y[:, None, :] - mean) ** 2 / logvar.exp()) * w[:, None, None]
    return cell[keep].sum() / len(w)  # boolean indexing, so padding can't leak a NaN into the sum


def run(model, data, chunks, y, scored, w, opt=None):
    """Mean per-game weighted NLL over the batches (games given as global indices); trains if opt is given."""
    total = games = 0
    for idx in chunks:
        moves, x, lengths = batch(data, idx)
        with torch.set_grad_enabled(opt is not None):
            mean, logvar = model(moves, x)
            keep = (torch.arange(moves.shape[1])[None, :] < lengths[:, None])[:, :, None] & scored[idx][:, None, :]
            loss = nll(mean, logvar, y[idx], keep, w[idx])
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        total, games = total + loss.item() * len(idx), games + len(idx)
    return total / games


def predict(model, data, idx, game_ids, m, sd):
    """Rows (game_id, side, t, mean, sd) in rating points at every checkpoint t each game in idx reaches."""
    out = []
    with torch.no_grad():
        for b in batches(np.diff(data[2])[idx]):
            moves, x, lengths = batch(data, idx[b])
            mean, logvar = model(moves, x)
            for t in CHECKPOINTS:
                ok = (lengths >= t).numpy()
                if not ok.any():
                    break  # no game in this batch reaches t, nor any later checkpoint
                mu = mean[ok, t - 1].numpy() * sd + m  # the output after ply t sits at index t - 1
                s = np.exp(logvar[ok, t - 1].numpy() / 2) * sd
                for side, name in enumerate(("white", "black")):
                    out.append(pd.DataFrame({"game_id": game_ids[idx[b][ok]], "side": name, "t": t,
                                             "mean": mu[:, side], "sd": s[:, side]}))
    return pd.concat(out, ignore_index=True)


def sides(games):
    """One row per side of each game: game_id, side, rating, scored, weight."""
    return pd.concat([pd.DataFrame({"game_id": games.game_id, "side": name, "rating": games[f"{name.title()}Elo"],
                                    "scored": games[f"score_{name}"], "weight": games.weight})
                      for name in ("white", "black")], ignore_index=True)


def conformal(preds, truth, level=LEVEL):
    """{t: q}: the weighted `level` quantile of |rating - mean| / sd over scored val sides at t, so that
    mean ± q·sd covers `level` of the month's sides. Scaled by the model's sd, so intervals stay wider
    where it is less sure; calibrated, because the Gaussian ±1.645 sd covered only 0.89."""
    d = preds.merge(truth[truth.scored], on=["game_id", "side"])
    score = ((d.rating - d["mean"]).abs() / d.sd).to_numpy()
    return {int(t): float(weighted_quantile(score[d.t == t], d.weight[d.t == t].to_numpy(), level))
            for t in sorted(d.t.unique())}


def main(limit=None):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    games = pd.read_parquet("sample.parquet", columns=COLUMNS)
    if limit:  # quick check: `limit` train games, val and test in proportion
        frac = limit / (games.split == "train").sum()
        games = games.groupby("split", group_keys=False).sample(frac=frac, random_state=0).reset_index(drop=True)
    split = games.split.to_numpy()
    train, val, test = (np.flatnonzero(split == s) for s in ("train", "val", "test"))

    start = time.time()
    words = vocab(games.moves.iloc[train])
    data = load(games, {m: i + 2 for i, m in enumerate(words)})
    lengths = np.diff(data[2])
    print(f"{len(games):,} games, {data[2][-1]:,} plies, {len(words):,} moves in the vocabulary "
          f"({time.time() - start:.0f} s)", flush=True)

    ratings = games[["WhiteElo", "BlackElo"]].to_numpy(dtype=float)
    weight = games.weight.to_numpy()
    # weighted, so the standardised targets are centred and scaled on the month, not on the oversampled tails
    m = np.average(ratings[train], weights=np.repeat(weight[train, None], 2, axis=1))
    sd = np.sqrt(np.average((ratings[train] - m) ** 2, weights=np.repeat(weight[train, None], 2, axis=1)))
    y = torch.tensor((ratings - m) / sd, dtype=torch.float32)
    scored = torch.tensor(games[["score_white", "score_black"]].to_numpy(dtype=bool))
    w = torch.tensor(weight / weight.mean(), dtype=torch.float32)

    model = GRU(len(words) + 2)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS, eta_min=LR / 20)  # big steps, then fine ones
    val_chunks = [val[b] for b in batches(lengths[val])]
    best, bad = np.inf, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()
        train_nll = run(model, data, [train[b] for b in batches(lengths[train], rng)], y, scored, w, opt)
        val_nll = run(model, data, val_chunks, y, scored, w)
        schedule.step()
        print(f"epoch {epoch}: train NLL {train_nll:.2f}, val NLL {val_nll:.2f} per game "
              f"({(time.time() - start) / 60:.1f} min)", flush=True)
        if val_nll < best:
            best, bad, state = val_nll, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad == PATIENCE:
                break
    model.load_state_dict(state)
    game_ids = games.game_id.to_numpy()
    q = conformal(predict(model, data, val, game_ids, m, sd), sides(games))
    torch.save({"state_dict": state, "vocab": words, "rating_mean": float(m), "rating_sd": float(sd),
                "conformal": q}, "model.pt")

    os.makedirs("results", exist_ok=True)
    preds = predict(model, data, test, game_ids, m, sd)
    half = preds.t.map(q) * preds.sd
    preds = preds.assign(lo=preds["mean"] - half, hi=preds["mean"] + half).drop(columns="sd")
    preds.to_parquet("results/preds_gru.parquet", index=False)
    print(f"best val NLL {best:.2f}; conformal multipliers {q}; {len(preds):,} test rows -> results/preds_gru.parquet",
          flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
