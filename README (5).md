# Predicting chess ratings move by move

How quickly can a model tell how strong two players are? This project predicts both players' Lichess blitz ratings after every move of a game, with an uncertainty interval that should narrow as the game goes on.

Ratings are Lichess's Glicko-2 ratings, not FIDE Elo.

## Status

- [x] Data pipeline
- [ ] Sanity checks
- [ ] Player-level train/test split
- [ ] Baselines
- [ ] Sequence model with calibrated intervals

## Data

Rated games from the [Lichess open database](https://database.lichess.org) (CC0), via its Parquet version on Hugging Face: [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games).

`prepare_games.py` keeps blitz games by Lichess's own definition (base time + 40 × increment between 180 and 479 seconds). It drops games involving bot accounts or with a move missing its clock. For each game it stores the players, both ratings, ECO code, time control, the moves in SAN, and the seconds left on the mover's clock after each move.

## Run

```
pip install -r requirements.txt
hf download Lichess/standard-chess-games --repo-type dataset --include "data/year=2025/month=09/train-0000*" --local-dir lichess
python prepare_games.py games lichess/data/year=2025/month=09/*.parquet
pytest
```

Then open `sanity_check.ipynb`.
