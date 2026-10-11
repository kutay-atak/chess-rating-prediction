# Predicting chess ratings move by move

Predicts both players' Lichess blitz ratings after every move, with 90% intervals that should narrow as the game goes on. Ratings are Lichess Glicko-2 ratings, not FIDE Elo. [PLAN.md](PLAN.md) builds it in phases, each applying one course.

## Status

- [x] Phase 1 — Data, stratified sample, player split (Sampling Theory)
- [x] Phase 2 — Features, baselines, evaluation
- [x] Phase 3 — GRU sequence model (Data Mining and Neural Networks)
- [ ] Phase 4 — State-space filter (Advanced Time Series Analysis)
- [ ] Phase 5 — Latent skill model (Structural Equations)
- [ ] Phase 6 — Figures and demo app (Data Visualization in Data Science)

## Data

Rated games from the [Lichess open database](https://database.lichess.org) (CC0), via its Parquet version on Hugging Face, [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games).

- `download.py` fetches 12 random shards of September 2025 (seed 0). Shards are probably consecutive time chunks, so random ones make a cluster sample of the month.
- `prepare_games.py` keeps blitz games by Lichess's own definition (base time + 40 × increment between 180 and 479 seconds) and drops games with a bot or a move missing its clock. Per game: players, ratings, ECO, time control, SAN moves, the mover's clock after each move, and the engine eval after each move where Lichess analysed the game.
- `sample.py` draws up to 40 000 games per 200-point band of the players' mean rating, with weight N_h / n_h, and splits players into train/val/test (80/10/10) so test results come only from unseen players.

## Results

![Weighted MAE by plies seen](results/mae.png)

### Baselines

LightGBM on clock, time-control, opening and (where Lichess analysed the game) engine-loss features gets from 281 rating points of error at ply 2 (no better than the time control alone) to 256 by ply 20, then only to 252 by ply 80, while the `mean` and `timecontrol` baselines stay near 293 and 280. So most of what these per-side summaries can tell is known by move 10, and the 90% intervals are still about ±520 points wide ([coverage](results/coverage.png), [all numbers](results/metrics.csv)).

At larger t only longer games remain (about a third of test sides reach ply 80), so the curves mix "more information" with "different games": `mean` predicts the same number at every t, yet its error moves by 5 points.

- `features.py` summarises one side after t plies: own moves, log seconds spent per move (mean, sd, share under 2 s), clock left as a share of base, base, increment, berserk, first ECO letter and, when evals exist, average loss, mistakes (≥ 1 pawn) and blunders (≥ 3). ECO is hidden before ply 10, because Lichess sets it from moves played later.
- Only 9% of the month's games have evals (14% of the sample), and analysed games are not a random subset, so the model gets `has_evals` as a feature rather than treating missing evals as random.
- `baselines.py` weights every fit by N_h / n_h, since the strata are bands of the outcome itself. Intervals are split conformal: the weighted 90% quantile of absolute val residuals per t.
- `evaluate.py` reports plain `mae` and `coverage` over the sample, and `weighted_mae` / `weighted_coverage` for the month. The weighted versions use the stratified estimator over test games, with N_h(t) counted in `games/`. The plain ones look much worse (MAE 350–470, coverage 0.64–0.76) because the sample oversamples the rating tails. `se` treats test games as independent. `se_player` resamples whole test players within strata and is 1.2–1.6 times larger. The plot's bands use `se_player`.

### GRU

`gru.py` reads the game one ply at a time. Each ply's inputs:
- the move, as an embedding of the 3,126 SAN moves seen at least 20 times in train;
- the mover's log seconds spent and clock share;
- base and increment;
- the eval and loss, with missing flags.

After every ply it outputs a mean and log-variance for both players' standardised ratings. It is trained on the Gaussian NLL weighted by N_h / n_h, like the baselines. The model is one GRU layer of 128 units (279k parameters; `model.pt` is 1.2 MB). Training took 15 minutes on 4 CPUs, and early stopping on val NLL kept epoch 5 of 6.

| plies seen | lgbm MAE | GRU MAE | lgbm interval | GRU interval | GRU coverage |
|---|---|---|---|---|---|
| 2 | 281 | 276 | ±572 | ±564 | 0.90 |
| 10 | 263 | 225 | ±537 | ±460 | 0.89 |
| 20 | 256 | 206 | ±525 | ±421 | 0.89 |
| 80 | 252 | 187 | ±518 | ±377 | 0.89 |

MAE is the weighted MAE; its player-bootstrap SE stays under 1.2 points. Coverage is the weighted coverage of the 90% interval. lgbm's coverage is 0.90 at every t.

The GRU wins at every checkpoint, and the gap grows as the game goes on: 5 points at ply 2, 50 by ply 20 and 66 by ply 80. Its error keeps falling after ply 20, where lgbm's flattens.

What makes the difference is the moves themselves. lgbm only sees summaries, plus the ECO letter from ply 10. The GRU's advantage is just as large in games Lichess never analysed (unweighted MAE 257 vs 354 at ply 20), so it doesn't come from engine evals.

Its intervals are narrower too: ±377 vs ±518 at ply 80. But they are model-based Gaussian intervals, not conformal ones, and they cover slightly less than promised (0.89 from ply 10 on).

What I didn't do: no tuning, a single seed, and at most 6 epochs to keep the run under 30 minutes on CPU. Val NLL had already started rising at epoch 6, so the cap didn't cut training short.

## Run

```
pip install -r requirements.txt
python download.py
python prepare_games.py games lichess/data/year=2025/month=09/*.parquet
python sample.py games
python baselines.py
python gru.py 20000   # quick check on 20 000 train games that the loss falls
python gru.py
python evaluate.py
pytest
```
