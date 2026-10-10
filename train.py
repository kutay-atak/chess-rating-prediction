"""Train the move-by-move rating model on prepared games (see prepare_games.py).

A small causal Transformer reads the start token and every half-move so far, and at each position
outputs a bivariate Gaussian over (White rating, Black rating): two means, two log-sigmas and a
correlation. Trained with the Gaussian negative log-likelihood over every position of every game,
so the same model gives the estimate before any move, after move 1, 2, ... and its uncertainty.

Engine evals exist only for games someone asked Lichess to analyse. Those games are mixed in with
their evals hidden in a share of them (--hide-evals), so one model works with and without an engine.
Analysed games are rarer at average ratings, so they are reweighted to the rating mix of other
games; otherwise "has evals" would itself shift the estimate.

Usage:
  python train.py games --games 3000000 --epochs 1                                # moves and clocks
  python train.py games --init checkpoints/v1.pt --games 600000 --analysed-games 1000000 --lr 3e-4
Writes --out (checkpoints/best.pt) whenever val MAE after move 20 improves: that early estimate is
the one we most want to be right. With analysed games, that's measured with evals shown.
"""
import argparse
import glob
import itertools
import math
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import EVAL_FEATURES, MAX_PLIES, N_NUMERIC, PAD, build_vocab, encode, time_control, vocab_index
from split import game_split

COLUMNS = ["White", "Black", "WhiteElo", "BlackElo", "TimeControl", "moves", "clocks", "evals"]
TARGET_MOVE = 20


def iter_games(folder, split, limit=None, seed=0, columns=COLUMNS, analysed=None):
    """One DataFrame per shard: up to `limit` games of one split in total, sampled evenly across shards.

    analysed=True/False keeps only games with/without engine evals. Filters with pyarrow before
    converting, so only the kept games become Python objects.
    """
    shards = sorted(glob.glob(f"{folder}/*.parquet"))
    per_shard = None if limit is None else math.ceil(limit / len(shards))
    rng = np.random.default_rng(seed)
    for path in shards:
        meta = pq.read_table(path, columns=["White", "Black", "evals"])
        names = meta.select(["White", "Black"]).to_pandas()
        keep = game_split(names.White, names.Black).to_numpy() == split
        if analysed is not None:
            keep &= meta["evals"].is_valid().to_numpy(zero_copy_only=False) == analysed
        keep = np.flatnonzero(keep)
        if per_shard is not None and len(keep) > per_shard:
            keep = np.sort(rng.choice(keep, per_shard, replace=False))
        yield pq.read_table(path, columns=columns).take(keep).to_pandas()


def load_games(folder, split, limit=None, seed=0, columns=COLUMNS, analysed=None):
    """iter_games as a single DataFrame, for val/test-sized sets."""
    df = pd.concat(iter_games(folder, split, limit, seed, columns, analysed), ignore_index=True)
    return df if limit is None else df.iloc[:limit]


class Encoded:
    """Games as flat arrays: tokens and numeric features end to end, plus per-game offsets."""

    def __init__(self, frames, index):
        """frames: a DataFrame of games, or an iterable of them (encoded one at a time to save memory)."""
        if isinstance(frames, pd.DataFrame):
            frames = [frames]
        tokens, numeric, lengths, ratings, tcs, analysed = [], [], [], [], [], []
        for df in frames:
            for sans, clocks, tc, evals in zip(df.moves, df.clocks, df.TimeControl, df.evals):
                t, f = encode(list(sans), list(clocks), *time_control(tc), index, evals)
                tokens.append(t.astype(np.int32))
                numeric.append(f.astype(np.float16))
                lengths.append(len(t))
                analysed.append(evals is not None)
            ratings.append(df[["WhiteElo", "BlackElo"]].to_numpy(np.float32))
            tcs.append(df.TimeControl.to_numpy())
        self.tokens = np.concatenate(tokens)
        self.numeric = np.concatenate(numeric)
        self.lengths = np.array(lengths)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.ratings = np.concatenate(ratings)
        self.time_control = np.concatenate(tcs)
        self.analysed = np.array(analysed)
        self.weights = np.ones(len(self.lengths), dtype=np.float32)

    def __len__(self):
        return len(self.lengths)

    def batch(self, ids, hide_evals=None):
        """Padded (tokens [B, T], numeric [B, T, F], mask [B, T]) for games `ids`, pads at the end.

        hide_evals: boolean per game; those games get their eval features zeroed.
        """
        T = self.lengths[ids].max()
        tokens = np.full((len(ids), T), PAD, dtype=np.int64)
        numeric = np.zeros((len(ids), T, N_NUMERIC), dtype=np.float32)
        for row, i in enumerate(ids):
            a, b = self.offsets[i], self.offsets[i + 1]
            tokens[row, :b - a] = self.tokens[a:b]
            numeric[row, :b - a] = self.numeric[a:b]
        if hide_evals is not None:
            numeric[hide_evals, :, EVAL_FEATURES] = 0
        return torch.from_numpy(tokens), torch.from_numpy(numeric), torch.from_numpy(tokens != PAD)

    def reweight(self):
        """Weight analysed games so their mean-rating histogram matches the other games'."""
        if not self.analysed.any() or self.analysed.all():
            return
        bins = np.clip(((self.ratings.mean(axis=1) - 400) // 100).astype(int), 0, 25)
        target = np.bincount(bins[~self.analysed], minlength=26) / (~self.analysed).sum()
        have = np.bincount(bins[self.analysed], minlength=26) / self.analysed.sum()
        w = np.where(self.analysed, np.clip(target / np.maximum(have, 1e-9), 0.1, 10)[bins], 1.0)
        w[self.analysed] /= w[self.analysed].mean()
        self.weights = w.astype(np.float32)


def batches(lengths, max_tokens, rng=None):
    """Lists of game ids whose padded size stays under max_tokens; similar lengths go together."""
    order = np.arange(len(lengths)) if rng is None else rng.permutation(len(lengths))
    out = []
    for c in range(0, len(order), 50_000):  # sort by length within large shuffled chunks
        chunk = order[c:c + 50_000]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        start = 0
        for end in range(1, len(chunk) + 1):
            if end == len(chunk) or (end + 1 - start) * lengths[chunk[end]] > max_tokens:
                out.append(chunk[start:end])
                start = end
    if rng is not None:
        rng.shuffle(out)
    return out


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class RatingModel(nn.Module):
    def __init__(self, vocab_size, d=128, layers=4, heads=4, dropout=0.1, n_numeric=N_NUMERIC):
        super().__init__()
        self.move = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_PLIES + 1, d)
        self.numeric = nn.Linear(n_numeric, d)
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 5)  # mu_w, mu_b, log_sd_w, log_sd_b, atanh-ish rho

    def forward(self, tokens, numeric):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.move(tokens) + self.pos(pos) + self.numeric(numeric)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def load_model(ckpt, n_numeric=None, dropout=0.1):
    """Model from a checkpoint dict. A larger n_numeric than the checkpoint's adds input features
    with zero weights, so the model starts out ignoring them."""
    have = ckpt["state"]["numeric.weight"].shape[1]
    config = {k: v for k, v in ckpt["config"].items() if k != "n_numeric"}
    model = RatingModel(**config, dropout=dropout, n_numeric=n_numeric or have)
    state = dict(ckpt["state"])
    if model.numeric.in_features > have:
        w = torch.zeros(model.numeric.weight.shape)
        w[:, :have] = state["numeric.weight"]
        state["numeric.weight"] = w
    model.load_state_dict(state)
    return model


def nll(out, y, mask, weights=None):
    """Mean bivariate Gaussian NLL over real positions. out [B, T, 5], y [B, 2] normalised ratings."""
    log_sd = out[..., 2:4]
    z = (y[:, None, :] - out[..., :2]) / log_sd.exp()
    rho = 0.99 * torch.tanh(out[..., 4])
    q = (z[..., 0] ** 2 - 2 * rho * z[..., 0] * z[..., 1] + z[..., 1] ** 2) / (1 - rho ** 2)
    loss = log_sd.sum(-1) + 0.5 * torch.log1p(-rho ** 2) + 0.5 * q
    w = mask if weights is None else mask * weights[:, None]
    return (loss * w).sum() / w.sum()


@torch.no_grad()
def validate(model, data, center, scale, max_tokens):
    """(val NLL, {move: MAE in rating points per player after that many full moves})."""
    model.eval()
    total, count = 0.0, 0
    moves = [0, 5, 10, TARGET_MOVE]
    err = {m: [] for m in moves}
    for ids in batches(data.lengths, max_tokens):
        tokens, numeric, mask = data.batch(ids)
        y = torch.from_numpy((data.ratings[ids] - center) / scale)
        out = model(tokens, numeric)
        total += nll(out, y, mask).item() * mask.sum().item()
        count += mask.sum().item()
        for m in moves:
            has = data.lengths[ids] > 2 * m  # position 2m is right after Black's m-th move
            if has.any():
                mu = out[torch.from_numpy(has), 2 * m, :2].numpy() * scale + center
                err[m].append(np.abs(mu - data.ratings[ids][has]).ravel())
    model.train()
    return total / count, {m: float(np.concatenate(e).mean()) for m, e in err.items() if e}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", help="folder of prepared parquet games")
    ap.add_argument("--games", dest="n_games", type=int, default=1_500_000, help="training games without evals")
    ap.add_argument("--analysed-games", type=int, default=0, help="training games with engine evals (at most)")
    ap.add_argument("--hide-evals", type=float, default=0.3, help="share of analysed games shown without evals")
    ap.add_argument("--val-games", type=int, default=20_000, help="per kind (with / without evals)")
    ap.add_argument("--init", help="checkpoint to start from")
    ap.add_argument("--out", default="checkpoints/best.pt")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-tokens", type=int, default=16_384, help="padded tokens per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    init = torch.load(args.init, weights_only=False) if args.init else None
    vocab = init["vocab"] if init else build_vocab(
        load_games(args.games, "train", 300_000, args.seed, columns=["moves"]).moves)
    index = vocab_index(vocab)
    with_evals = args.analysed_games > 0
    frames = iter_games(args.games, "train", args.n_games, args.seed, analysed=False if with_evals else None)
    if with_evals:
        frames = itertools.chain(frames, iter_games(args.games, "train", args.analysed_games, args.seed, analysed=True))
    train = Encoded(frames, index)
    train.reweight()
    vals = {"no evals": Encoded(load_games(args.games, "val", args.val_games, args.seed,
                                           analysed=False if with_evals else None), index)}
    if with_evals:
        vals["evals"] = Encoded(load_games(args.games, "val", args.val_games, args.seed, analysed=True), index)
    center = init["center"] if init else float(train.ratings.mean())
    scale = init["scale"] if init else float(train.ratings.std())
    print(f"{len(train):,} train games ({train.analysed.sum():,} with evals), {len(train.tokens):,} tokens, "
          f"val {', '.join(f'{len(v):,} {k}' for k, v in vals.items())}, vocab {len(vocab):,}, "
          f"ratings {center:.0f} ± {scale:.0f}, loaded in {time.time() - t0:.0f}s", flush=True)

    if init:
        model = load_model(init, N_NUMERIC)
        config = dict(init["config"], n_numeric=N_NUMERIC)
    else:
        config = dict(vocab_size=len(vocab) + 3, d=args.d, layers=args.layers, heads=args.heads, n_numeric=N_NUMERIC)
        model = RatingModel(**config)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(batches(train.lengths, args.max_tokens))
    total_steps = int(args.epochs * steps_per_epoch)
    warmup = min(500, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
    select = "evals" if with_evals else "no evals"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best, step, ema, tokens_seen, t0 = float("inf"), 0, None, 0, time.time()
    while step < total_steps:
        for ids in batches(train.lengths, args.max_tokens, rng):
            hide = train.analysed[ids] & (rng.random(len(ids)) < args.hide_evals)
            tokens, numeric, mask = train.batch(ids, hide)
            y = torch.from_numpy((train.ratings[ids] - center) / scale)
            loss = nll(model(tokens, numeric), y, mask, torch.from_numpy(train.weights[ids]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tokens_seen += int(mask.sum())
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % args.eval_every == 0 or step == total_steps:
                results = {k: validate(model, v, center, scale, args.max_tokens) for k, v in vals.items()}
                elapsed = time.time() - t0
                report = " | ".join(f"{k}: nll {n:.4f} MAE after move " + " ".join(f"{m}:{e:.0f}" for m, e in mae.items())
                                    for k, (n, mae) in results.items())
                print(f"step {step}/{total_steps} train {ema:.4f} | {report} | "
                      f"{tokens_seen / elapsed:,.0f} tok/s {elapsed / 60:.0f} min", flush=True)
                if results[select][1][TARGET_MOVE] < best:
                    best = results[select][1][TARGET_MOVE]
                    torch.save(dict(state=model.state_dict(), config=config, vocab=vocab, center=center, scale=scale,
                                    step=step, val_mae_move20=best), args.out)
            if step == total_steps:
                break
    print(f"best val MAE after move {TARGET_MOVE} ({select}): {best:.1f}, saved {args.out}")


if __name__ == "__main__":
    main()
