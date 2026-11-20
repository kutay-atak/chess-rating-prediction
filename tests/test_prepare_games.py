import math

import pandas as pd

from prepare_games import parse, prepare, write

GAME = "1. e4 { [%clk 0:03:00] } 1... e5 { [%eval 0.2] [%clk 0:02:58.5] } 2. Qh5?! { [%clk 0:02:55] } 1-0"


def test_parse():
    sans, clocks, evals = parse(GAME)
    assert sans == ["e4", "e5", "Qh5"] and clocks == [180, 178.5, 175]
    assert math.isnan(evals[0]) and evals[1] == 0.2 and math.isnan(evals[2])  # eval + clock, clock only
    assert parse("1. e4 { [%clk 0:03:00] } 1... e5 2. Nf3 { [%clk 0:02:59] } 1-0") is None  # unclocked move
    assert parse("1. e4 { [%eval 0.3] } 1... e5 { [%clk 0:03:00] } 1-0") is None  # comment without a clock
    assert parse("1. e4 e5 1/2-1/2") is None


def test_parse_mate_and_large_evals():
    game = "1. e4 { [%eval 14.3] [%clk 0:03:00] } 1... f6 { [%eval #3] [%clk 0:03:00] } 2. d4 { [%eval #-2] [%clk 0:02:59] } 1-0"
    assert parse(game)[2] == [10, 10, -10]


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
    assert df.evals.iloc[0][1] == 0.2 and math.isnan(df.evals.iloc[0][0])

    assert write(tmp_path / "shard.parquet", tmp_path / "part.parquet") == 1
    saved = pd.read_parquet(tmp_path / "part.parquet")
    assert saved.moves.iloc[0].tolist() == ["e4", "e5", "Qh5"] and saved.evals.iloc[0][1] == 0.2


def test_write_shard_with_no_blitz_games(tmp_path):
    game = dict(White="a", Black="b", WhiteTitle=None, BlackTitle=None, WhiteElo=1500, BlackElo=1550,
                ECO="C20", TimeControl="60+0", movetext=GAME)
    pd.DataFrame([game]).to_parquet(tmp_path / "shard.parquet")
    assert write(tmp_path / "shard.parquet", tmp_path / "part.parquet") == 0
    assert len(pd.read_parquet(tmp_path / "part.parquet")) == 0
