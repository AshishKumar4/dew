"""A mutation score measures killed tests, not broken workers or missing runs."""

import json
from types import SimpleNamespace

import pytest

from tools.mutation import classify, summarize


@pytest.mark.parametrize("worker,test,output,expected", [
    ("normal", "killed", "1 failed", "killed"),
    ("normal", "survived", "1 passed", "survived"),
    ("normal", "killed", "timeout", "timeout"),
    ("exception", "incompetent", "launch failed", "worker-error"),
    ("normal", "incompetent", "launch failed", "incompetent"),
    ("no-test", None, None, "worker-error"),
])
def test_only_test_failures_count_as_killed(worker, test, output, expected):
    result = SimpleNamespace(worker_outcome=worker, test_outcome=test, output=output)
    assert classify(result) == expected
    assert classify(None) == "pending"


def test_shards_are_weighted_by_their_mutation_counts(tmp_path):
    for shard, counts in enumerate([{"killed": 4}, {"killed": 1, "survived": 2, "timeout": 1}]):
        directory = tmp_path / str(shard)
        directory.mkdir()
        (directory / "report.json").write_text(json.dumps({
            "population": 8, "shard": shard, "shards": 2, "counts": counts}))
    result = summarize(tmp_path)
    assert result["total"] == 8
    assert result["score"] == 5 / 7
    assert result["counts"] == {"killed": 5, "survived": 2, "timeout": 1}


@pytest.mark.parametrize("shards,populations,counts", [
    ([0], [8], [8]),
    ([0, 0], [8, 8], [4, 4]),
    ([0, 1], [8, 9], [4, 5]),
    ([0, 1], [8, 8], [4, 3]),
])
def test_an_incomplete_or_inconsistent_population_is_not_a_score(tmp_path, shards, populations, counts):
    for index, (shard, population, count) in enumerate(zip(shards, populations, counts, strict=True)):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "report.json").write_text(json.dumps({
            "population": population, "shard": shard, "shards": 2, "counts": {"killed": count}}))
    with pytest.raises(ValueError):
        summarize(tmp_path)
