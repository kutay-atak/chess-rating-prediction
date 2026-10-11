# Plan

One flagship project, built in phases. Each phase applies a course from this year.

| Phase | What | Course |
|---|---|---|
| 1 | Data, stratified sample, player split | Sampling Theory |
| 2 | Features, baselines, evaluation | backbone for everything after |
| 3 | GRU sequence model | Data Mining and Neural Networks |
| 4 | State-space filter | Advanced Time Series Analysis |
| 5 | Latent skill model | Structural Equations |
| 6 | Figures and demo app | Data Visualization in Data Science |

Advanced Time Series, Data Mining and Neural Networks, and Structural Equations run in the first semester, so Phases 3–5 can follow the lectures. Sampling Theory and Data Visualization come in the second semester, but Phases 1 and 6 only need their basics. Phase 5 depends only on Phase 1, so it can run alongside Phases 2–4.

## Decisions
- A GRU before any transformer: it's in the course, and one sequence model is enough until it plateaus.
- No Stockfish runs. Engine evals come only from games Lichess has already analysed.
- SEM in R with lavaan, the course tool. Python writes a CSV; R reads it.
- The demo is a Streamlit app, not a JavaScript front end.
- Every model outputs a mean and a 90% interval, and all are judged by `evaluate.py`.

## Phase 1 — Data and sampling design (Sampling Theory)
Goal: evaluation that generalises to a whole month of blitz games, and a split with no player leakage.

1. Replace the README's Status list with the six phases above.
2. `prepare_games.py`: add an `evals` column (see CLAUDE.md conventions; NaN where a move has no eval). Extend the tests with a comment holding an eval and a clock, one with a clock only, and a mate eval.
3. Pick 12 shards of `data/year=2025/month=09/` at random (seed 0; list them with `huggingface_hub.list_repo_files`), download them and run `prepare_games.py`. Random shards, not the first ones, because shards are probably consecutive time chunks: this makes them a cluster sample of the month.
4. `sample.py`:
   - stratum = 200-point band of the mean rating of the two players: under 1000, 1000–1199, …, 2200–2399, 2400 and up
   - N_h = games per stratum in `games/`; draw n_h = min(N_h, 40 000) per stratum (seed 0)
   - weight = N_h / n_h, so oversampled tails count less at evaluation
   - game_id = row number
5. Player split, also in `sample.py`: hash each username to train/val/test (80/10/10, stable across runs). A game is test if either player is a test player, else val if either player is a val player, else train. Add columns `split`, `score_white` and `score_black`: a side is scored only if its player belongs to the game's split, so test results only ever come from unseen players.

Done when: tests pass; `sample.py` prints N_h, n_h and the weight per stratum; an assert confirms no scored test player appears in any train game. Then run `sanity_check.ipynb` yourself: the "first clock == base time" share should be close to 1.

## Phase 2 — Features, baselines, evaluation
Goal: the yardstick every later model is judged against.

1. `features.py`: features for one side after t plies: own moves so far, mean and sd of log seconds spent, share of own moves under 2 seconds, clock left as a share of base time, base time, increment, first letter of the ECO code. When evals exist: average loss, mistakes (loss ≥ 1) and blunders (loss ≥ 3), plus a `has_evals` flag. Test it on a hand-made game.
2. `baselines.py`, writing one preds file per model:
   - `mean`: training mean rating
   - `timecontrol`: mean rating per TimeControl
   - `lgbm`: LightGBM on the features, one model across checkpoints with t as a feature
   - intervals for all three: split conformal per checkpoint, using absolute residuals on val
3. `evaluate.py`, on scored test sides, for each model and checkpoint: MAE; weighted MAE with standard error; 90% coverage; mean interval width; n. The weighted MAE is the stratified estimator over test games (unit = game, error = mean over its scored sides), with stratum sizes N_h(t) = games in `games/` with at least t plies, and the standard error from the stratified variance formula with finite-population correction. Writes `results/metrics.csv`, `results/mae.png` and `results/coverage.png`.

Done when: all three models are in `metrics.csv`, and the README shows the MAE plot with two sentences on what it shows. Note in the README that at larger t only longer games remain, so the curves mix "more information" with "different games".

Dependencies: lightgbm.

## Phase 3 — Sequence model (Data Mining and Neural Networks)
Goal: a GRU that predicts both ratings after every ply, with uncertainty.

- `gru.py`. Inputs per ply: SAN move embedding (vocabulary from train; rare moves map to an unknown token), log seconds spent, clock left as a share of base, side to move, base time and increment (scaled). No engine evals: they exist only for games someone had analysed after the game, so they reveal a later event, and a live game in the demo never has them.
- Output per ply: mean and log-variance of White's and Black's standardised rating. Loss: Gaussian negative log-likelihood summed over plies and sides. Early stopping on validation NLL.
- Train weighted by N_h/n_h, like the baselines: the strata are bands of the ratings being predicted, so an unweighted fit learns the sample's rating mix, not the month's.
- First train on 20 000 games to check the loss falls, then on everything. If the CPU is too slow, use a free Colab or Kaggle GPU.
- Save the weights to `model.pt` and keep it under 20 MB, since the demo loads it from the repo. Write preds at the checkpoints, with interval = mean ± q_t sd, q_t calibrated on val sides (split conformal on |error| / sd; the plain 1.645 covered only 0.89).

Done when: `gru` is in `metrics.csv` and the README compares it with `lgbm` honestly, win or lose.

Dependencies: torch.

## Phase 4 — State-space filter (Advanced Time Series Analysis)
Goal: a classical model of "the estimate sharpens as evidence arrives", to set against the GRU.

1. Per-move signal: LightGBM on single-move features (seconds spent, clock left, ply, loss if available) predicting the mover's rating, trained on a random 2 million moves. Its prediction is the observation z_t. Measurement variance σ²(t) comes from validation residuals, binned by ply.
2. `kalman_iid`: rating as a constant state, prior from the `timecontrol` baseline (mean and variance per TimeControl), one Kalman update per own move. This equals precision-weighted averaging.
3. `kalman_ar1`: the same with AR(1) measurement errors: augment the state with the error and estimate ρ from the autocorrelation of validation residuals within games.

Done when: both are in `metrics.csv`, and a coverage plot shows whether modelling the error correlation fixes the overconfidence expected from `kalman_iid`.

## Phase 5 — Latent skill (Structural Equations)
Goal: test whether skill splits into accuracy and time use, and whether that structure holds across time controls.

1. `sem_data.py`: from games in `games/` that have evals, one random game per player (seed 0) so observations are independent. Indicators per player-game: log average loss, mistake rate, blunder rate, mean log seconds per move, sd of log seconds per move, and the share of moves played with under 10% of base time left. Also the player's rating and the TimeControl. Writes `sem_data.csv`.
2. `skill_sem.R` with lavaan:
   - measurement model: Accuracy from the three loss indicators, TimeUse from the three time indicators
   - structural model: rating regressed on Accuracy and TimeUse
   - MLR estimator, since the rates are skewed
   - measurement invariance across the three most common TimeControls: configural, then metric, then scalar, compared with ΔCFI
   - writes `results/sem_fit.csv` and `results/sem_invariance.csv`
3. README: the model as a Mermaid path diagram (GitHub renders it), the fit indices, and a short interpretation.

Done when: both CSVs exist and the README section is written.

Dependencies: R with lavaan.

## Phase 6 — Figures and demo (Data Visualization in Data Science)
Goal: a page where anyone gets the idea in ten seconds.

1. Polish the MAE, coverage and interval-width figures for the README.
2. `app.py` (Streamlit): paste a Lichess game link, fetch the PGN from Lichess's game export API, reuse `parse()` from `prepare_games.py`, run the GRU, and show the board with a move slider (python-chess SVG) next to both players' rating bands over the game. Reveal the true ratings at the end. Reject non-blitz games with a message.
3. Deploy on Streamlit Community Cloud; add the link and a GIF to the README.

Done when: the deployed link works on a game played today.

Dependencies: streamlit, chess.
