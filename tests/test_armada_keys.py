"""What the test cache's keys stand for: each input of a row's key (ci.py's `shared` and `cache_key`, what
`record` keeps, .armada.json's environment key) changed through the real flow on small checkouts, and the
rows that must and must not run again. The harness the tests run is the repository's own tools (`loaded`) or
a copy of them an edit changes (`copied_tools`)."""

import json
import re
from pathlib import Path

from armada_support import copied_tools
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
