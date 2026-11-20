import pandas as pd

from split import game_split, player_split


def test_player_split_is_stable_and_case_insensitive():
    assert player_split("Magnus") == player_split("magnus")
    counts = pd.Series([player_split(f"user{i}") for i in range(20_000)]).value_counts(normalize=True)
    assert abs(counts["train"] - 0.8) < 0.02 and abs(counts["test"] - 0.1) < 0.02


def test_val_and_test_players_never_reach_train():
    names = [f"user{i}" for i in range(2_000)]
    games = pd.DataFrame({"White": names, "Black": names[1:] + names[:1]})
    games["split"] = game_split(games.White, games.Black)
    assert set(games.split) == {"train", "val", "test"}
    for split in ["train", "val"]:
        players = set(games.White[games.split == split]) | set(games.Black[games.split == split])
        assert "test" not in {player_split(p) for p in players}
    train_players = set(games.White[games.split == "train"]) | set(games.Black[games.split == "train"])
    assert {player_split(p) for p in train_players} == {"train"}
