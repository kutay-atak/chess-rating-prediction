# Game performance rating, move by move

A chess.com-style website where two people play blitz and see, after every move, each player's **game performance rating**: how strong that player has played *in this game so far*, on the Lichess blitz rating scale. It is not a prediction of anyone's long-term rating. Each player's number depends only on their own moves: hang your queen and your line drops; your opponent's doesn't move.

## Method

1. **Opening book** (`opening_book.py`). Every position in the first 12 moves of the training games is keyed by its Zobrist hash (`chess.polyglot.zobrist_hash`), so move orders don't matter. Per position: game count `n` and mean White / Black rating, kept if seen in at least 20 games.
2. **Move accuracy** (`performance.py`). Stockfish scores every move. From the winning chance a move gave away:
   - `win% = 100 / (1 + exp(-0.00368208 * cp))`
   - `accuracy = 103.1668 * exp(-0.04354 * win%_lost) - 3.1669`, clipped to 0..100

   A player's running game accuracy is the average of the mean and the harmonic mean (floor 10) of their move accuracies.
3. **Accuracy → rating table** (`fit_performance.py`). From Lichess games with engine evals (train players only): for (own moves made, accuracy bin), the median and 10th/90th percentile rating of players at that point. Monotone in accuracy; analysed games are weighted to the rating mix of all games.
4. **Blend** per player: `(n * book + 20 * accuracy_estimate) / (n + 20)`, using the position right after the player's own move. After leaving the book, `n` starts from at most 20 (the position now has fewer games than that) and halves with every own move, so the line slides instead of jumping. (Halving from the last book count instead would freeze the line for ~5 moves after leaving a popular position, hiding blunders.)

Ratings are Lichess Glicko-2 blitz ratings, not FIDE Elo.

## Run the website locally

The fitted files (`model/performance.json`, `model/opening_book.npz`) are committed, so you only need Python and Stockfish.

1. **Install Stockfish** (required for the estimate; without it you can still play, and the page says the estimate is unavailable):
   - Ubuntu/Debian: `sudo apt install stockfish`; macOS: `brew install stockfish`
   - or download the official binary from https://github.com/official-stockfish/Stockfish/releases and either put it on your `PATH` as `stockfish` or set `STOCKFISH=/path/to/stockfish`.
2. Install and start:
   ```
   python -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt
   uvicorn web.server:app --port 8000
   ```
3. Open http://localhost:8000.

Play three ways: **Start Game** (matchmaking per time control), **Play a Friend** (invite link), **Pass and Play** (both sides on one screen). Two browser tabs can play each other. Only blitz time controls are offered. The server checks moves with python-chess, keeps clocks the way Lichess does, and scores each move with Stockfish in the background (depth 14 by default; `STOCKFISH_DEPTH` changes it).

## Rebuild the fitted files

```
hf download Lichess/standard-chess-games --repo-type dataset --include "data/year=2025/month=09/train-0000*" --local-dir lichess
python prepare_games.py games lichess/data/year=2025/month=09/*.parquet
python opening_book.py games
python fit_performance.py games
pytest
```

`prepare_games.py` keeps blitz games (base + 40 × increment between 180 and 479 s, so 0+5 counts), drops bot games, and keeps Lichess's `%eval` where the game was analysed (about 9%). `split.py` splits by player, so test players never appear in the book or the table. Run one of these heavy steps at a time. `sanity_check.ipynb` looks at the prepared data.

Credits: data from the [Lichess open database](https://database.lichess.org) (CC0) via [Lichess/standard-chess-games](https://huggingface.co/datasets/Lichess/standard-chess-games); pieces by Colin M.L. Burnett (cburnett, BSD/GPL/GFDL); [chess.js](https://github.com/jhlywa/chess.js) (BSD-2-Clause) for client-side move hints. The layout imitates chess.com, but no chess.com assets, logo or name are used.
