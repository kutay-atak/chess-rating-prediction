import pandas as pd

from prepare_games import parse, prepare

GAME = "1. e4 { [%clk 0:03:00] } 1... e5 { [%eval 0.2] [%clk 0:02:58.5] } 2. Qh5?! { [%clk 0:02:55] } 1-0"


def test_parse():
    assert parse(GAME) == (["e4", "e5", "Qh5"], [180, 178.5, 175])
    assert parse("1. e4 { [%clk 0:03:00] } 1... e5 2. Nf3 { [%clk 0:02:59] } 1-0") is None  # unclocked move
    assert parse("1. e4 e5 1/2-1/2") is None


def test_prepare(tmp_path):
    game = dict(White="a", Black="b", WhiteTitle=None, BlackTitle=None,
                WhiteElo=1500, BlackElo=1550, ECO="C20", movetext=GAME)
    pd.DataFrame([
        {**game, "TimeControl": "180+2"},                       # blitz: kept
        {**game, "TimeControl": "60+0"},                        # bullet
        {**game, "TimeControl": "-"},                           # correspondence
        {**game, "TimeControl": "300+0", "BlackTitle": "BOT"},  # bot
    ]).to_parquet(tmp_path / "shard.parquet")

    df = prepare(tmp_path / "shard.parquet")
    assert df.TimeControl.tolist() == ["180+2"]
    assert df.moves.iloc[0] == ["e4", "e5", "Qh5"]
    assert df.clocks.iloc[0] == [180, 178.5, 175]
