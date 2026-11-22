"""The test-types gate's reading of a change (tools/test_types.py): the test files a head touches since its
merge-base, a renamed one read against its old path, so a rename carries its findings and its own new ones
alone are new."""

from pathlib import Path

from tool_support import git

from tools.test_types import touched


def test_a_heads_test_files_are_read_against_their_paths_at_its_merge_base(tmp_path: Path):
    body = "def test_one():\n    assert sum(range(10)) == 45\n" * 6
    for name in ("test_changed", "test_renamed", "test_deleted", "test_kept"):
        (tmp_path / "tests").mkdir(exist_ok=True)
        (tmp_path / "tests" / f"{name}.py").write_text(f"# {name}\n{body}")
    (tmp_path / "tests/data.txt").write_text("rows\n")
    git(tmp_path, "init", "-q", "-b", "base")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "base")
    git(tmp_path, "checkout", "-q", "-b", "head")
    (tmp_path / "tests/test_changed.py").write_text("# changed\n" + body)
    git(tmp_path, "mv", "tests/test_renamed.py", "tests/test_moved.py")
    (tmp_path / "tests/test_moved.py").write_text(f"# test_renamed\n{body}x = 1\n")
    (tmp_path / "tests/test_deleted.py").unlink()
    (tmp_path / "tests/test_added.py").write_text("def test_new():\n    assert {1, 2} | {3} == {1, 2, 3}\n")
    (tmp_path / "tests/data.txt").write_text("rows, more\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "head")
    assert touched("base", cwd=tmp_path) == {"tests/test_changed.py": "tests/test_changed.py",
                                             "tests/test_moved.py": "tests/test_renamed.py",
                                             "tests/test_added.py": None}
