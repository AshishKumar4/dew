"""tools/armada/reach.py: the rows a change reaches, as the test cache's own keys read it."""

import json
from pathlib import Path

from armada_support import apart, cached
from tool_support import run_from


def test_a_changes_reach_is_the_rows_whose_keys_it_changes(tmp_path, monkeypatch):
    """A change's reach (`reached`) is the rows whose key over their last run's inputs differs between its
    base and its checkout, as the cache reads it: one that executed a function it changed, not one that
    executed its untouched neighbour; one that read a data file it changed; one whose last run recorded no
    inputs, unknown and named apart; one whose record was not read, which runs (train 894's review: rows
    unread reached nothing); never a row the cache keeps; one moved by a module-level unit alone in a tier
    of its own (Main, 2026-11-15: a reach read off the plan's misses named 547 files for a one-space
    change, and the keys' own reach was the suite through module-level units every importer executes)."""
    apart(monkeypatch)
    with run_from("reach", "tools/armada") as reach:
        ci = reach.ci
    followed = ci.followed

    def tree(root: Path, body: str, data: str, top: str) -> None:
        (root / "lib").mkdir(parents=True)
        (root / "lib/x.py").write_text(f"def f():\n    return {body}\n\n\ndef g():\n    return 2\n")
        (root / "lib/z.py").write_text(f"Z = {top}\n")
        (root / "data.txt").write_text(data)

    base, head = tmp_path / "base", tmp_path / "head"
    tree(base, "1", "a\n", "1")
    tree(head, "10", "b\n", "2")
    monkeypatch.chdir(head)
    monkeypatch.setattr(ci, "shared", lambda python: "common")
    rows = [("tests/test_b.py", "1/2"), ("tests/test_b.py", "2/2"), ("tests/test_d.py", ""),
            ("tests/test_e.py", ""), ("tests/test_f.py", ""), ("tests/test_g.py", ""),
            ("tests/test_m.py", ""), ("tests/test_u.py", "")]
    # test_e's record holds no inputs (unknown); test_u's was not read (None), which runs.
    last = {("tests/test_e.py", ""): [], ("tests/test_u.py", ""): None,
            ("tests/test_b.py", "1/2"): [f"lib/x.py{followed.CODE}f"],
            ("tests/test_b.py", "2/2"): [f"lib/x.py{followed.CODE}g"],
            ("tests/test_d.py", ""): ["data.txt"],
            ("tests/test_f.py", ""): [f"lib/x.py{followed.CODE}f"],
            ("tests/test_g.py", ""): [f"lib/x.py{followed.CODE}g"],
            ("tests/test_m.py", ""): [f"lib/z.py{followed.CODE}<module>",
                                      f"lib/x.py{followed.CODE}g"]}
    found = reach.reached("3.12", rows, {("tests/test_f.py", "")}, last, base)
    # A module-level change alone (z.py's) is the second tier's.
    assert found == {"files": ["tests/test_b.py", "tests/test_d.py", "tests/test_u.py"],
                     "module": ["tests/test_m.py"], "unknown": ["tests/test_e.py"]}
    assert Path.cwd() == head


def test_a_rows_last_inputs_are_its_last_records_whatever_it_ran_on(tmp_path, monkeypatch, cache_worker):
    """`last_inputs` reads each row's last record, whatever shared inputs it ran on (its `read`, else its
    key's `depends`), [] where it holds none, and None where none was read: no read token, or no record there
    (train 894's review: rows read only on the head's own shared inputs left a change to them reaching
    nothing)."""
    apart(monkeypatch)
    with run_from("reach", "tools/armada") as reach:
        ci = reach.ci
    monkeypatch.chdir(tmp_path)
    pinned = {"task": {"cpu": {"vendor": "V", "family": 1, "model": 2}}}
    (tmp_path / ".armada.json").write_text(json.dumps(pinned))
    rows = [("tests/test_a.py", ""), ("tests/test_b.py", "1/2"), ("tests/test_c.py", ""),
            ("tests/test_d.py", "")]
    assert reach.last_inputs("3.12", rows) == dict.fromkeys(rows)
    url, stored, _ = cache_worker
    cached(monkeypatch, url)
    monkeypatch.setenv(ci.READ_TOKEN, "token")
    bodies = ({"read": ["x.py"], "depends": None}, {"depends": ["y.py"]}, {"depends": None})
    for (name, group), body in zip(rows[:3], bodies, strict=False):
        place = "/" + ci.records.record_key("3.12", ci.row("3.12", name, group), "V-1-2")
        stored[place] = json.dumps(body).encode()
    assert reach.last_inputs("3.12", rows) == {rows[0]: ["x.py"], rows[1]: ["y.py"], rows[2]: [],
                                               rows[3]: None}
