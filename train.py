"""Train the move-by-move rating model on prepared games (see prepare_games.py).

A small causal Transformer reads the start token and every half-move so far, and at each position
outputs a bivariate Gaussian over (White rating, Black rating): two means, two log-sigmas and a
correlation. Trained with the Gaussian negative log-likelihood over every position of every game,
so the same model gives the estimate before any move, after move 1, 2, ... and its uncertainty.

Usage:
  python train.py games --games 1500000 --epochs 2
Writes checkpoints/best.pt (weights, config, vocab, rating normalisation) whenever val NLL improves.
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import MAX_PLIES, N_NUMERIC, PAD, build_vocab, encode, time_control, vocab_index
from split import game_split

COLUMNS = ["White", "Black", "WhiteElo", "BlackElo", "TimeControl", "moves", "clocks"]
CHECKPOINT = "checkpoints/best.pt"


def iter_games(folder, split, limit=None, seed=0, columns=COLUMNS):
    """One DataFrame per shard: up to `limit` games of one split in total, sampled evenly across shards.

    Filters with pyarrow before converting, so only the kept games become Python objects.
    """
    shards = sorted(glob.glob(f"{folder}/*.parquet"))
    per_shard = None if limit is None else math.ceil(limit / len(shards))
    rng = np.random.default_rng(seed)
    for path in shards:
        names = pq.read_table(path, columns=["White", "Black"]).to_pandas()
        keep = np.flatnonzero(game_split(names.White, names.Black).to_numpy() == split)
        if per_shard is not None and len(keep) > per_shard:
            keep = np.sort(rng.choice(keep, per_shard, replace=False))
        yield pq.read_table(path, columns=columns).take(keep).to_pandas()


def load_games(folder, split, limit=None, seed=0, columns=COLUMNS):
    """iter_games as a single DataFrame, for val/test-sized sets."""
    df = pd.concat(iter_games(folder, split, limit, seed, columns), ignore_index=True)
    return df if limit is None else df.iloc[:limit]


class Encoded:
    """Games as flat arrays: tokens and numeric features end to end, plus per-game offsets."""

    def __init__(self, frames, index):
        """frames: a DataFrame of games, or an iterable of them (encoded one at a time to save memory)."""
        if isinstance(frames, pd.DataFrame):
            frames = [frames]
        tokens, numeric, lengths, ratings, tcs = [], [], [], [], []
        for df in frames:
            for sans, clocks, tc in zip(df.moves, df.clocks, df.TimeControl):
                t, f = encode(list(sans), list(clocks), *time_control(tc), index)
                tokens.append(t.astype(np.int32))
                numeric.append(f.astype(np.float16))
                lengths.append(len(t))
            ratings.append(df[["WhiteElo", "BlackElo"]].to_numpy(np.float32))
            tcs.append(df.TimeControl.to_numpy())
        self.tokens = np.concatenate(tokens)
        self.numeric = np.concatenate(numeric)
        self.lengths = np.array(lengths)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.ratings = np.concatenate(ratings)
        self.time_control = np.concatenate(tcs)

    def __len__(self):
        return len(self.lengths)

    def batch(self, ids):
        """Padded (tokens [B, T], numeric [B, T, F], mask [B, T]) for games `ids`, pads at the end."""
        T = self.lengths[ids].max()
        tokens = np.full((len(ids), T), PAD, dtype=np.int64)
        numeric = np.zeros((len(ids), T, N_NUMERIC), dtype=np.float32)
        for row, i in enumerate(ids):
            a, b = self.offsets[i], self.offsets[i + 1]
            tokens[row, :b - a] = self.tokens[a:b]
            numeric[row, :b - a] = self.numeric[a:b]
        return torch.from_numpy(tokens), torch.from_numpy(numeric), torch.from_numpy(tokens != PAD)


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
    def __init__(self, vocab_size, d=128, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.move = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_PLIES + 1, d)
        self.numeric = nn.Linear(N_NUMERIC, d)
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 5)  # mu_w, mu_b, log_sd_w, log_sd_b, atanh-ish rho

    def forward(self, tokens, numeric):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.move(tokens) + self.pos(pos) + self.numeric(numeric)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def nll(out, y, mask):
    """Mean bivariate Gaussian NLL over real positions. out [B, T, 5], y [B, 2] normalised ratings."""
    log_sd = out[..., 2:4]
    z = (y[:, None, :] - out[..., :2]) / log_sd.exp()
    rho = 0.99 * torch.tanh(out[..., 4])
    q = (z[..., 0] ** 2 - 2 * rho * z[..., 0] * z[..., 1] + z[..., 1] ** 2) / (1 - rho ** 2)
    loss = log_sd.sum(-1) + 0.5 * torch.log1p(-rho ** 2) + 0.5 * q
    return (loss * mask).sum() / mask.sum()


@torch.no_grad()
def validate(model, data, center, scale, max_tokens):
    """(val NLL, {ply: MAE in rating points per player}) over all of `data`."""
    model.eval()
    total, count = 0.0, 0
    plies = [0, 10, 20, 40, 80]
    err = {p: [] for p in plies}
    for ids in batches(data.lengths, max_tokens):
        tokens, numeric, mask = data.batch(ids)
        y = torch.from_numpy((data.ratings[ids] - center) / scale)
        out = model(tokens, numeric)
        total += nll(out, y, mask).item() * mask.sum().item()
        count += mask.sum().item()
        for p in plies:
            has = data.lengths[ids] > p
            if has.any():
                mu = out[torch.from_numpy(has), p, :2].numpy() * scale + center
                err[p].append(np.abs(mu - data.ratings[ids][has]).ravel())
    model.train()
    return total / count, {p: float(np.concatenate(e).mean()) for p, e in err.items() if e}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", help="folder of prepared parquet games")
    ap.add_argument("--games", dest="n_games", type=int, default=1_500_000, help="training games to use")
    ap.add_argument("--val-games", type=int, default=20_000)
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
    vocab = build_vocab(load_games(args.games, "train", 300_000, args.seed, columns=["moves"]).moves)
    index = vocab_index(vocab)
    train = Encoded(iter_games(args.games, "train", args.n_games, args.seed), index)
    val = Encoded(load_games(args.games, "val", args.val_games, args.seed), index)
    center = float(train.ratings.mean())
    scale = float(train.ratings.std())
    print(f"{len(train):,} train / {len(val):,} val games, {len(train.tokens):,} train tokens, "
          f"vocab {len(vocab):,}, ratings {center:.0f} ± {scale:.0f}, loaded in {time.time() - t0:.0f}s", flush=True)

    config = dict(vocab_size=len(vocab) + 3, d=args.d, layers=args.layers, heads=args.heads)
    model = RatingModel(**config)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(batches(train.lengths, args.max_tokens))
    total_steps = int(args.epochs * steps_per_epoch)
    warmup = min(500, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    best, step, ema, tokens_seen, t0 = float("inf"), 0, None, 0, time.time()
    while step < total_steps:
        for ids in batches(train.lengths, args.max_tokens, rng):
            tokens, numeric, mask = train.batch(ids)
            y = torch.from_numpy((train.ratings[ids] - center) / scale)
            loss = nll(model(tokens, numeric), y, mask)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tokens_seen += int(mask.sum())
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % args.eval_every == 0 or step == total_steps:
                val_nll, mae = validate(model, val, center, scale, args.max_tokens)
                elapsed = time.time() - t0
                print(f"step {step}/{total_steps} train {ema:.4f} val {val_nll:.4f} "
                      f"MAE by ply {' '.join(f'{p}:{m:.0f}' for p, m in mae.items())} "
                      f"{tokens_seen / elapsed:,.0f} tok/s {elapsed / 60:.0f} min", flush=True)
                if val_nll < best:
                    best = val_nll
                    torch.save(dict(state=model.state_dict(), config=config, vocab=vocab, center=center,
                                    scale=scale, step=step, val_nll=val_nll), CHECKPOINT)
            if step == total_steps:
                break
    print(f"best val NLL {best:.4f}, saved {CHECKPOINT}")


if __name__ == "__main__":
    main()
