# Chess rating prediction

Predicts both players' Lichess blitz ratings after every move, with 90% intervals that should narrow as the game goes on. Portfolio project of a statistics master's student. PLAN.md builds it in phases, each applying one course.

## How to work
- Do one phase of PLAN.md at a time. Stop when its "Done when" checks pass, then say in one or two lines what you skipped or didn't check.
- Smallest change that fully solves the task. Scripts at the repo root, plain functions, no config files or frameworks.
- Add a dependency only when PLAN.md names it for the current phase, and add it to requirements.txt.
- New parsing or statistics logic gets a small pytest test in `test_<script>.py`.
- Every statistical choice (strata, weights, intervals, splits) gets a one-line comment saying why.
- Ask before downloading more than 3 GB or starting a job expected to run longer than 30 minutes.
- Never commit data: add each new data output to .gitignore. Commit code and small results (CSV, PNG) in `results/`.
- When a phase is done, tick it in the README's Status list.

## Data
- `lichess/`: raw Parquet shards of the Hugging Face dataset `Lichess/standard-chess-games` (rated games only).
- `games/part-*.parquet`: output of `prepare_games.py`, one file per shard. Columns: White, Black, WhiteElo, BlackElo, ECO, TimeControl, moves (SAN), clocks (seconds left after each move), evals (added in Phase 1).
- `sample.parquet`: output of `sample.py` (Phase 1).
- Ratings are Lichess Glicko-2 ratings, not FIDE Elo.

## Conventions
- t = plies seen so far. Evaluation checkpoints: t = 2, 5, 10, 20, 30, 40, 60, 80.
- Seconds spent on a move = the mover's previous clock − current clock + increment; on each side's first move, base − clock.
- Evals are from White's view in pawns, clipped to ±10, with mate = ±10. A move's loss = the drop in the mover's eval, in pawns (≥ 0). White's first move has no loss (there is no eval before it).
- Every model writes `results/preds_<model>.parquet` with columns game_id, side ("white"/"black"), t, mean, lo, hi (90% interval). `evaluate.py` scores all models the same way.

## Commands
- `pip install -r requirements.txt`
- `pytest`
- `python prepare_games.py games lichess/data/year=2025/month=09/*.parquet`
