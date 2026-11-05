"""What the test cache's keys stand for: each input of a row's key (ci.py's `shared` and `cache_key`, what
`record` keeps, .armada.json's environment key) changed through the real flow on small checkouts, and the
rows that must and must not run again. The harness the tests run is the repository's own tools (`loaded`) or
a copy of them an edit changes (`copied_tools`)."""

import json
import re
from pathlib import Path

from armada_support import cached_checkout, commit, copied_tools, green_runs
from tool_support import load

ROOT = Path(__file__).resolve().parents[1]


def install_inputs(root: Path, script: str) -> set[str]:
    """The files of the checkout at `root` that `script` (its path there) names in its commands, and those
    the shell scripts among them name in turn: a limited check of what an install runs or reads (the words of
    each line that is no comment, a command substitution's and a quote's taken apart), not a shell
    interpreter, so a path a variable builds (`locks/$name.txt`) is none of them."""
    found, pending = {script}, [script]
    while pending:
        lines = [line for line in (root / pending.pop()).read_text().splitlines()
                 if not line.lstrip().startswith("#")]
        words = re.findall(r"[\w./-]+", "\n".join(lines))
        files = {word for word in words if not word.startswith(("/", "..")) and (root / word).is_file()}
        pending.extend(sorted(name for name in files - found if name.endswith(".sh")))
        found |= files
    return found


def test_the_environments_key_names_every_file_its_install_script_runs_or_reads(tmp_path, monkeypatch):
    """armada rebuilds the environment where the files its key names change (.armada.json), so each file
    tools/armada/install.sh runs or reads is among them (row 3303: tools/fetched_fixtures.py, whose fetch
    placed release files elsewhere, the classifiers of pyproject.toml that set which Pythons get an
    environment, and the .python-version the repository's own is linked to), but ci.py, which install.sh runs
    for those classifiers (`pythons`): its code is the plan's, which no environment is rebuilt for."""
    environment = json.loads((ROOT / ".armada.json").read_text())["environment"]
    # What armada hashes: the files its globs match in the commit, and the recipe's setup and install scripts.
    keyed = {path.relative_to(ROOT).as_posix() for glob in environment["key"] for path in ROOT.glob(glob)}
    keyed |= {environment["setup"], environment["install"]}
    runs = install_inputs(ROOT, environment["install"])
    reader = "tools/armada/ci.py"
    assert reader in runs
    assert sorted(runs - keyed - {reader}) == []
    # The classifiers are what it prints: a version added to them is one more environment, so pyproject.toml,
    # which holds them, is keyed.
    before = copied_tools(tmp_path, monkeypatch).PYTHONS
    pyproject = tmp_path / "pyproject.toml"
    classifier = '"Programming Language :: Python :: 3.14",'
    assert classifier in pyproject.read_text()
    pyproject.write_text(pyproject.read_text().replace(
        classifier, classifier + '\n    "Programming Language :: Python :: 3.15",'))
    after = load("ci", "tools/armada", tmp_path).PYTHONS
    assert after == (*before, "3.15")
    assert "pyproject.toml" in keyed


def test_a_harness_edit_of_comments_and_docstrings_reruns_only_what_observed_the_edited_text(
        tmp_path, monkeypatch, cache_worker):
    """The recorder, session and plugins under tools/armada are in every key by their code, not their bytes,
    and the cache stays on for a change of them that is no code (row 2730: train 350's docstring edits to
    followed.py and child_groups.py turned the cache off and re-keyed every row). What a test observes of the
    harness is its own input all the same: a text it reads is its bytes, a function's `__code__` it takes its
    unit with its docstring. A change to code reruns every row and turns the cache off."""
    ci = copied_tools(tmp_path, monkeypatch)
    observers = {
        "plain": "",
        "reads": "    assert 'def read(' in Path('tools/armada/followed.py').read_text()\n",
        "code": "    import followed\n    assert followed.read.__code__.co_argcount == 3\n"}
    for name, body in observers.items():
        (tmp_path / f"tests/test_{name}.py").write_text(
            f"from pathlib import Path\ndef test_{name}():\n"
            f"    with Path('runs-{name}').open('a') as runs: runs.write('x')\n{body}")
    verdict = cached_checkout(ci, tmp_path, monkeypatch, cache_worker)
    base = commit(tmp_path, "base")
    names = tuple(observers)

    def runs(*more: str) -> list[int]:
        return green_runs(ci, monkeypatch, tmp_path, verdict, ci.policy(base), names)

    assert runs() == [1, 1, 1]
    assert runs() == [1, 1, 1]
    followed = tmp_path / "tools/armada/followed.py"
    # A comment, and a module's docstring in the recorder, the plugin, the session and the process group's.
    followed.write_text(followed.read_text() + "\n# a note on the recorder\n")
    for name in ("session", "child_groups", "lifetime", "executed"):
        path = tmp_path / f"tools/armada/{name}.py"
        path.write_text(path.read_text().replace('"""', '"""A note on it. ', 1))
    commit(tmp_path, "comments and docstrings")
    assert ci.policy(base) == "on"
    assert runs() == [1, 2, 1], "what read the text reruns, nothing else"
    # A docstring of the function whose code one file took.
    signature = "def read(event: str, args: tuple, root: str) -> str | None:\n    \"\"\""
    assert signature in followed.read_text()
    followed.write_text(followed.read_text().replace(signature, signature + "A note on reading. ", 1))
    commit(tmp_path, "a function's docstring")
    assert ci.policy(base) == "on"
    assert runs() == [1, 3, 2], "the text, and the code that took the function's"
    session = tmp_path / "tools/armada/session.py"
    session.write_text(session.read_text().replace("GRACE = 60.0", "GRACE = 61.0"))
    commit(tmp_path, "code")
    assert ci.policy(base) == "off"
    assert runs() == [2, 4, 3]
