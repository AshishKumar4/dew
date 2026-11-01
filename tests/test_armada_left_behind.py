"""What a test file's run may leave for a later file of its task where no key holds it (followed.py's
`WROTE`): a write outside its own scratch, caches and record, by Python or by a tool it starts. A later file
that reads it keeps no key (ci.py's `left_behind`), so these decide which files of a task are reused."""

import os
import sys
import tempfile
import time

import pytest
from tool_support import load


@pytest.fixture
def followed(tmp_path, monkeypatch):
    """The recorder's module, in a file's process as ci.py's `alone` makes it: its scratch, caches, log,
    collection account and JUnit report its own."""
    module = load("followed", "tools/armada")
    own = tmp_path / "own"
    for name in ("tmp", "caches"):
        (own / name).mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(own / "tmp"))
    monkeypatch.setenv("TMPDIR", str(own / "tmp"))
    monkeypatch.setenv(module.PRIVATE, str(own / "caches"))
    monkeypatch.setenv("DEW_EXECUTED_LOG", str(own / "0.lifetime.log"))
    monkeypatch.setenv("DEW_COLLECTED", str(own / "0.collected.json"))
    monkeypatch.setenv("DEW_EXECUTED_REPORT", str(own / "0.junit.xml"))
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv(module.CHILDREN, raising=False)
    return module


def test_a_write_to_the_files_own_scratch_caches_and_record_leaves_nothing(followed, tmp_path):
    own = tmp_path / "own"
    for path in (own / "tmp" / "out.npy", own / "caches/dew/rung", own / "0.lifetime.log",
                 own / "0.lifetime.log.children/12.log", own / "0.junit.xml", own / "0.collected.json",
                 tmp_path / "checkout/src/dew/__pycache__/x.pyc", "/dev/null"):
        assert followed.written(str(path)) is None, path


def test_a_write_to_the_home_shared_scratch_or_a_hidden_directory_leaves_it_for_a_later_file(
        followed, tmp_path, monkeypatch):
    """Reviews of the reuse-keys head: /tmp/result, ~/.cache/x and .memo/result, each a later file may read
    where no key holds it; a file of the checkout a read keys (a fixture, a mark in its root) leaves nothing
    unkeyed, as the next checkout answers its read otherwise."""
    checkout = tmp_path / "checkout"
    monkeypatch.setenv("DEW_EXECUTED_ROOT", str(checkout))
    for path in (tmp_path / "shared/result", os.path.expanduser("~/.cache/x"), checkout / ".memo/result",
                 tmp_path / "own/next.lifetime.log"):
        assert followed.written(str(path)) == f"{followed.WROTE}{path}", path
    for path in (checkout / "tests/fixtures/x.json", checkout / "runs-a"):
        assert followed.written(str(path)) is None, path


@pytest.mark.parametrize(("argv", "in_checkout", "leaves"), [
    (["git", "status", "--short"], True, False),
    (["git", "-C", "{checkout}", "log", "-1"], False, False),
    (["git", "commit", "-qm", "x"], True, True),
    (["git", "-C", "{private}", "commit", "-qm", "x"], True, False),
    (["ptxas", "-arch=sm_90a", "{private}/k.ptx", "-o", "{private}/k.cubin"], True, False),
    (["ptxas", "-arch=sm_90a", "{private}/k.ptx"], True, True),
    (["uname", "-m"], True, False),
    (["file", "-b", "{checkout}/x"], True, False),
    (["rsync", "-a", "{private}/a", "{checkout}/b"], False, True),
], ids=["git-status", "git-log", "git-commit", "git-commit-private", "ptxas-out", "ptxas-cwd", "uname",
        "file", "rsync"])
def test_a_tool_leaves_what_it_writes_outside_the_files_own(followed, tmp_path, argv, in_checkout, leaves):
    """A tool a file may run and keep its key (followed.py's `TOOLS`) leaves something for a later file where
    what it writes is not the file's own (`outputs`): git's commands that change a repository but its own
    scratch's, ptxas's output or, with none named, its working directory, rsync's destination; uname, file
    and git's reads leave nothing."""
    checkout, private = tmp_path / "checkout", tmp_path / "own" / "tmp"
    checkout.mkdir()
    words = [word.format(checkout=checkout, private=private) for word in argv]
    directory = str(checkout if in_checkout else private)
    paths = followed.outputs(words[0], words, directory)
    assert paths is not None
    assert (not all(followed.own(os.path.abspath(path)) for path in paths)) is leaves


@pytest.mark.parametrize(("event", "args"), [
    ("os.mkdir", ("{path}", 0o777, -1)), ("os.remove", ("{path}", -1)),
    ("os.rename", ("{own}", "{path}", -1, -1)),
    ("os.symlink", ("{own}", "{path}", -1)), ("os.utime", ("{path}", None, None, -1)),
    ("shutil.rmtree", ("{path}", None)), ("os.truncate", ("{path}", 0))],
    ids=["mkdir", "remove", "rename", "symlink", "utime", "rmtree", "truncate"])
def test_each_change_the_audit_names_leaves_its_path(followed, tmp_path, event, args):
    """Each change event's path, as the audit names it (no directory descriptor as -1, or None), is the mark
    of what it leaves (`changed`)."""
    path, own = tmp_path / "checkout" / "made", tmp_path / "own" / "tmp" / "x"
    named = tuple(arg.format(path=path, own=own) if isinstance(arg, str) else arg for arg in args)
    assert followed.changed(event, named) == f"{followed.WROTE}{path}"


def test_a_directory_made_where_one_is_changes_nothing(followed, tmp_path):
    """`os.makedirs(..., exist_ok=True)` asks to make each directory of its path, which is no change where
    it is there already; a new one is."""
    there = tmp_path / "checkout"
    there.mkdir()
    assert followed.changed("os.mkdir", (str(there), 0o777, -1)) is None
    assert followed.changed("os.mkdir", (str(there / "new"), 0o777, -1)) == f"{followed.WROTE}{there / 'new'}"


def test_the_recorders_own_hook_beside_the_packages_is_its_own(followed):
    for name in (".dew_followed.pth.318", "dew_followed.pth"):
        assert followed.written(f"{sys.prefix}/lib/python3/site-packages/{name}") is None


def test_wandbs_service_may_write_anywhere(followed):
    assert followed.outputs("wandb-core", ["wandb-core", "--pid", "1"], "/") is None


@pytest.mark.parametrize("asked", ["written", "private", "native"])
def test_the_recorders_own_questions_of_a_path_are_no_runs_probes(followed, tmp_path, monkeypatch, asked):
    """The recorder asks of paths (`written`'s resolution, `private`'s, `native`'s of a program) under the
    probes it wraps for a run (`probed`): none of its own questions is the run's input, nor asks itself again
    without end (debugger's round 2: `leaving`'s realpath re-entered itself)."""
    root = f"{tmp_path / 'checkout'}{os.sep}"
    monkeypatch.setenv(followed.STARTED_AT, str(time.time() - 60))
    monkeypatch.setenv("DEW_EXECUTED_ROOT", root)
    monkeypatch.setitem(sys.modules, "jax", None)
    for name in followed.CACHES:
        monkeypatch.delenv(name, raising=False)
    notes = []
    with monkeypatch.context() as hooks:
        hooks.setattr(os, "lstat", followed.probed(os.lstat, notes.append, root))
        hooks.setattr(os, "stat", followed.probed(os.stat, notes.append, root))
        if asked == "written":
            assert followed.written(str(tmp_path / "checkout/x")) is not None
        elif asked == "private":
            followed.private()
        else:
            assert followed.native(["git"])
    assert notes == []
