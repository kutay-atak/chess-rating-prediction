# Predicting chess ratings move by move

Predicts both players' Lichess blitz ratings after every move, with 90% intervals that should narrow as the game goes on. Ratings are Lichess Glicko-2 ratings, not FIDE Elo. [PLAN.md](PLAN.md) builds it in phases, each applying one course.

## Status

- [x] Phase 1 — Data, stratified sample, player split (Sampling Theory)
- [ ] Phase 2 — Features, baselines, evaluation
- [ ] Phase 3 — GRU sequence model (Data Mining and Neural Networks)
- [ ] Phase 4 — State-space filter (Advanced Time Series Analysis)
- [ ] Phase 5 — Latent skill model (Structural Equations)
- [ ] Phase 6 — Figures and demo app (Data Visualization in Data Science)

## Data

Rated games from the [Lichess open database](https://database.lichess.org) (CC0), via its Parquet version on Hugging Face, [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games).

- `download.py` fetches 12 random shards of September 2025 (seed 0). Shards are probably consecutive time chunks, so random ones make a cluster sample of the month.
- `prepare_games.py` keeps blitz games by Lichess's own definition (base time + 40 × increment between 180 and 479 seconds) and drops games with a bot or a move missing its clock. Per game: players, ratings, ECO, time control, SAN moves, the mover's clock after each move, and the engine eval after each move where Lichess analysed the game.
- `sample.py` draws up to 40 000 games per 200-point band of the players' mean rating, with weight N_h / n_h, and splits players into train/val/test (80/10/10) so test results come only from unseen players.

## Run

```
pip install -r requirements.txt
python download.py
python prepare_games.py games lichess/data/year=2025/month=09/*.parquet
python sample.py games
pytest
```
