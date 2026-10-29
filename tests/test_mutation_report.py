"""A mutation score measures killed tests, not broken workers or missing runs."""

import ast
import json
from pathlib import Path

import pytest

from tools.mutation import FILE_TESTS, TARGETS, Target, plan, run, source_digest, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_nightly_plan_keeps_the_rotation_and_covers_every_shard():
    for weekday, module in enumerate(("process", "transaction", "checkpoints", "process",
                                      "transaction", "checkpoints", "process")):
        result = plan(None, weekday)
        assert result["module"] == module
        assert result["matrix"]["shard"] == list(range(result["shards"]))


def test_requested_target_overrides_the_nightly_rotation():
    for target in TARGETS:
        assert plan(target, 0)["module"] == target


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


def defined(test: str) -> bool:
    """Whether `test`, a file or a `file::name` (a parametrized id's base), names a file of the
    repository and, with a name, a function or class its module defines."""
    path, _, name = test.partition("::")
    source = ROOT / path
    if not source.is_file():
        return False
    if not name:
        return True
    tree = ast.parse(source.read_text())
    return name.split("[")[0] in {node.name for node in tree.body
                                  if isinstance(node, ast.FunctionDef | ast.ClassDef)}


def test_every_selection_names_its_tests_and_a_deleted_test_is_refused_before_the_baseline(tmp_path):
    """Each target's tests and each file's name a test file the repository holds, and each
    `::name` a test its module defines; and a session whose selection names a deleted file stops
    before anything runs, naming it (`collected`, pytest's own collection). That every named file
    imports and collects is the suite's own collection of it, which this need not repeat: a
    collection of every selection's files took 123 s, past CI's group target (row 2150)."""
    selections = {test for selection in (*(target.tests for target in TARGETS.values()),
                                         *FILE_TESTS.values()) for test in selection}
    assert selections and [test for test in sorted(selections) if not defined(test)] == []
    gone = Target("src/dew/diffusion/schedules", ("tests/test_schedulers.py", "tests/test_deleted.py"))
    with pytest.raises(ValueError):
        run(gone, tmp_path / "session", 0, 1, 60.0)
    assert not (tmp_path / "session").exists()
