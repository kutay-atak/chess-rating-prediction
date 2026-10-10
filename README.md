# Predicting chess ratings move by move

How quickly can a model tell how strong two players are? This project predicts both players' Lichess blitz ratings after every move of a game, with an uncertainty interval that should narrow as the game goes on. A small website lets two people play blitz and watch the estimates move.

Ratings are Lichess's Glicko-2 ratings, not FIDE Elo.

## Status

- [x] Data pipeline
- [ ] Sanity checks
- [x] Player-level train/val/test split
- [x] Baselines and sequence model with calibrated intervals (code; results below once trained)
- [x] Website: two-player blitz with live estimates

## Data

Rated games from the [Lichess open database](https://database.lichess.org) (CC0), via its Parquet version on Hugging Face: [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games).

`prepare_games.py` keeps blitz games by Lichess's own definition (base time + 40 × increment between 180 and 479 seconds). It drops games involving bot accounts or with a move missing its clock. For each game it stores the players, both ratings, ECO code, time control, the moves in SAN, and the seconds left on the mover's clock after each move.

`split.py` splits by player, not by game: each username hashes to train (80%), val (10%) or test (10%). A game is test if either player is a test player, else val if either is a val player, else train. Test players never appear in training.

## Model

- `features.py` turns a game into tokens: a start token (so there is an estimate before any move), then one per half-move with its SAN move and clock features (time left, time spent, opponent's clock, time control). Training and the website both use it.
- `train.py` trains a small causal Transformer (4 layers, d=128, ~1M parameters) that outputs, at every position, a bivariate Gaussian over the two ratings: two means, two standard deviations and their correlation. Matchmaking makes the two ratings strongly correlated, and the correlation term captures that. The loss is the negative log-likelihood at every half-move of every game.
- `export_onnx.py` writes `model/rating_model.onnx` and `model/meta.json` (vocabulary, rating scale, interval calibration). It checks the ONNX output against PyTorch.
- `evaluate.py` fits per-ply interval scales on val games so that 80% intervals hold 80% of true ratings. It then scores the model on test players against two baselines (the mean rating; the mean rating per time control). Results go to `reports/metrics.json` and `reports/mae_by_ply.png`.

Lichess pairs players of similar strength, so the model has rarely seen a big rating gap. When very different players meet, it tends to pull their estimates toward each other.

## Run

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
hf download Lichess/standard-chess-games --repo-type dataset --include "data/year=2025/month=09/train-0000*" --local-dir lichess
python prepare_games.py games lichess/data/year=2025/month=09/*.parquet
python train.py games            # CPU is fine: about 20k tokens/s on 4 cores
python export_onnx.py
python evaluate.py games
pytest
```

Open `sanity_check.ipynb` after `prepare_games.py`.

## Website

```
uvicorn web.server:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000. The site follows chess.com's live-game layout: board with player bars and clocks, move list, and a live rating chart. You can play three ways:

- **Start Game** pairs you with the next person who picks the same time control.
- **Play a Friend** gives you an invite link.
- **Pass and Play** puts both sides on one screen.

Only blitz time controls are offered (3|0, 3|2, 5|0, 5|3), because the model only knows blitz. The server (`web/`) checks moves with python-chess and keeps clocks the way Lichess records them, so the model sees the same features it was trained on. Each player's estimate (± half of the 80% range) is shown next to their name and updated after every move.

Each browser tab is its own player, so two tabs on one computer can play each other.

Credits: pieces by Colin M.L. Burnett (cburnett, BSD/GPL/GFDL); [chess.js](https://github.com/jhlywa/chess.js) (BSD-2-Clause) for client-side move hints. The layout imitates chess.com, but no chess.com assets, logo or name are used.
