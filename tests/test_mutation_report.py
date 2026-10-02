"""A mutation score measures killed tests, not broken workers or missing runs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mutation import Target, classify, score_counts, selected_target, source_digest, summarize


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


def test_pending_work_is_not_reported_as_a_completed_mutation_score():
    summary = score_counts({"killed": 3, "survived": 1, "pending": 2})
    assert summary["score"] is None
    assert summary["observed_score"] == 3 / 4
    assert summary["total"] == 6
    assert not summary["complete"]


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


def test_each_per_file_batch_stays_inside_its_requested_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "src" / "dew" / "schedules"
    source.mkdir(parents=True)
    filename = source / "cosine.py"
    filename.write_text("RATE = 1\n")
    target = Target("src/dew/schedules", ("tests/test_schedules.py",))
    selected = selected_target(target, filename)
    assert selected.path == "src/dew/schedules/cosine.py"
    assert selected.tests == target.tests
    outside = tmp_path / "outside.py"
    outside.write_text("RATE = 1\n")
    with pytest.raises(ValueError):
        selected_target(target, outside)
    with pytest.raises(ValueError):
        selected_target(target, Path("src/dew/schedules/missing.py"))


def test_session_digest_changes_with_source_content_or_population(tmp_path):
    first, second = (tmp_path / name for name in ("first.py", "second.py"))
    first.write_text("RATE = 1\n")
    second.write_text("RATE = 2\n")
    original = source_digest([first, second])
    assert original == source_digest([second, first])
    assert original != source_digest([first])
    second.write_text("RATE = 3\n")
    assert original != source_digest([first, second])
