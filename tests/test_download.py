from download import pick


def test_pick_is_12_distinct_september_shards_and_fixed_by_the_seed():
    files = [f"data/year=2025/month=09/train-{i:05d}-of-00062.parquet" for i in range(62)]
    files += ["data/year=2025/month=08/train-00000-of-00060.parquet", "README.md"]
    shards = pick(files)
    assert len(set(shards)) == 12 and all(s.startswith("data/year=2025/month=09/") for s in shards)
    assert shards == pick(list(reversed(files)))
