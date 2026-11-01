"""Small reproductions of the reuse-keys classifier's self-observation."""

import os
import sys
import tempfile
import time

import pytest
from tool_support import load


@pytest.mark.parametrize("classifier", ["outside", "leaving", "private", "native"])
def test_classifier_metadata_probes_are_not_test_inputs(tmp_path, monkeypatch, classifier):
    followed = load("followed", "tools/armada")
    root = str(tmp_path / "checkout") + os.sep
    own = tmp_path / "private"
    foreign = tmp_path / "foreign"
    foreign.write_text("a different process's leaving\n")
    monkeypatch.setattr(tempfile, "tempdir", str(own))
    monkeypatch.setenv(followed.PRIVATE, str(own))
    monkeypatch.setenv("XDG_CACHE_HOME", str(own))
    monkeypatch.setenv(followed.STARTED_AT, str(time.time() - 60))
    monkeypatch.setenv("DEW_EXECUTED_ROOT", root)
    monkeypatch.setitem(sys.modules, "jax", None)
    for name in followed.CACHES:
        monkeypatch.delenv(name, raising=False)
    notes = []
    with monkeypatch.context() as hooks:
        hooks.setattr(os, "lstat", followed.probed(os.lstat, notes.append, root))
        if classifier in ("outside", "leaving"):
            assert getattr(followed, classifier)(str(foreign), root)
        elif classifier == "private":
            assert followed.private()
        else:
            assert followed.native(["git"])
    assert notes == []


@pytest.mark.parametrize("present", [False, True])
def test_only_the_callers_shared_probe_is_recorded(tmp_path, monkeypatch, present):
    followed = load("followed", "tools/armada")
    root = str(tmp_path / "checkout") + os.sep
    foreign = tmp_path / "foreign"
    if present:
        foreign.write_text("left by another process\n")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "private"))
    monkeypatch.setenv(followed.STARTED_AT, "0")
    notes = []
    with monkeypatch.context() as hooks:
        for owner, names in followed.PROBES:
            for name in names:
                hooks.setattr(owner, name, followed.probed(getattr(owner, name), notes.append, root))
        assert os.path.exists(foreign) is present
    assert notes == [followed.OUTSIDE + str(foreign)]
