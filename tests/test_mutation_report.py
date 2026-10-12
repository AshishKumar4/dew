"""A mutation score measures killed tests, not broken workers or missing runs."""

import json

from tools.mutation import source_digest, summarize


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


def test_session_digest_changes_with_source_content_or_population(tmp_path):
    first, second = (tmp_path / name for name in ("first.py", "second.py"))
    first.write_text("RATE = 1\n")
    second.write_text("RATE = 2\n")
    original = source_digest([first, second])
    assert original == source_digest([second, first])
    assert original != source_digest([first])
    second.write_text("RATE = 3\n")
    assert original != source_digest([first, second])
