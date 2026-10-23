"""An API rename keeps the landing page's recordings: site/scripts/redigest.py, on a copy of the site in a git
repository of its own, writes the renamed programs' digests and leaves every other byte of their recordings as
it was, and writes nothing when a change is more than a renaming or a recording was already of other code."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COPIED = ("site/scripts/redigest.py", "site/snippets/cells.py", "site/snippets/cells.json",
          "site/snippets/framework.py", "site/src/data/hero.py", "site/src/data/text.py",
          "site/src/data/sampler.py", "site/src/data/capture.json")
DIGEST = re.compile(rb"[0-9a-f]{64}")


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """The page's sources and recordings in a repository whose `recorded` branch is what they ran; the lm
    cell's capture written with CRLF line ends."""
    for relative in COPIED:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, tmp_path / relative)
    shutil.copytree(REPO / "site/public/examples", tmp_path / "site/public/examples")
    lm = tmp_path / "site/public/examples/framework/lm/capture.json"
    lm.write_bytes(lm.read_bytes().replace(b"\n", b"\r\n"))
    git(tmp_path, "init", "-q", "-b", "recorded")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "recorded")
    return tmp_path


def redigest(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(root / "site/scripts/redigest.py"), "--base", "recorded"],
                          capture_output=True, text=True, timeout=60)


def recordings(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted((root / "site").rglob("*"))
            if path.is_file() and ("public" in path.parts or path.name == "capture.json")}


def program_digest(root: Path, name: str) -> str:
    """The sha256 the page checks program `name`'s recording against, from the copy's own cells.py."""
    code = ("import sys; sys.path.insert(0, 'site/snippets'); import cells; "
            f"print(cells.digest(cells.programs()[{name!r}]))")
    return subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


def edit(root: Path, relative: str, old: str, new: str, count: int = -1) -> None:
    text = (root / relative).read_text()
    assert old in text
    (root / relative).write_text(text.replace(old, new, count))


def test_a_rename_rewrites_the_digests_of_the_programs_it_reaches_and_keeps_every_other_byte(site):
    before = recordings(site)
    edit(site, "site/snippets/framework.py", "OptimConfig(learning_rate=", "OptimConfig(rate=")
    edit(site, "site/src/data/hero.py", "OptimConfig(schedule=", "OptimConfig(rate=")
    # An API's keyword renamed while the variable of the same spelling keeps its name.
    edit(site, "site/src/data/hero.py", "steps=steps,", "num_steps=steps,")
    assert redigest(site).returncode == 0
    after = recordings(site)
    changed = {path for path in before if before[path] != after[path]}
    assert {"site/src/data/capture.json", "site/public/examples/framework/lm/capture.json"} <= changed
    for path in changed:
        assert DIGEST.sub(b"", before[path]) == DIGEST.sub(b"", after[path])
    lm = json.loads(after["site/public/examples/framework/lm/capture.json"])
    assert lm["script_sha256"] == program_digest(site, "lm")
    assert after["site/public/examples/framework/lm/capture.json"].count(b"\r\n") > 1
    hero = json.loads(after["site/src/data/capture.json"])["meta"]
    assert (hero["cell_sha256"], hero["script_sha256"]) == (program_digest(site, "hero"),
                                                            program_digest(site, "hero.py"))
    # Run again, it finds every recording of the program the page shows.
    assert redigest(site).returncode == 0 and recordings(site) == after


FRAMEWORK, TRAINS = "site/snippets/framework.py", "trainer = Trainer(LMObjective("


@pytest.mark.parametrize(("path", "old", "new"), [
    (FRAMEWORK, "OptimConfig(learning_rate=1e-3)", "OptimConfig(learning_rate=2e-3)"),
    (FRAMEWORK, "OptimConfig(learning_rate=1e-3)", "OptimConfig(learning_rate=1e-3, clip=1.0)"),
    (FRAMEWORK, f"{TRAINS}model,", f"{TRAINS}net,"),
    (FRAMEWORK, f"{TRAINS}model,", f"{TRAINS}data,"),
    ("site/src/data/text.py", "model", "__debug__"),
], ids=["a value", "an argument", "a variable renamed in one place", "a variable renamed to another's name",
        "a rename that does not compile"])
def test_a_change_that_is_more_than_a_renaming_writes_nothing(site, path, old, new):
    before = recordings(site)
    edit(site, path, old, new, -1 if path.endswith("text.py") else 1)
    assert redigest(site).returncode == 1
    assert recordings(site) == before


def test_two_variables_renamed_to_one_fresh_name_writes_nothing(site):
    """The hero's training and held-out pipelines both become `batches`: a name neither had, which now means
    one where it meant two."""
    before = recordings(site)
    hero = "site/src/data/hero.py"
    edit(site, hero, "data = train.", "batches = train.")
    edit(site, hero, "held = val.", "batches = val.")
    edit(site, hero, "fit(data, validation=held,", "fit(batches, validation=batches,")
    assert redigest(site).returncode == 1
    assert recordings(site) == before


def recorded(root: Path, source: str) -> None:
    """`source` as the text program the base recorded: text.py, its capture naming its digest."""
    (root / "site/src/data/text.py").write_text(source)
    capture = root / "site/public/examples/text/capture.json"
    named = json.loads(capture.read_bytes())["script_sha256"]
    capture.write_bytes(capture.read_bytes().replace(named.encode(),
                                                     hashlib.sha256(source.encode()).hexdigest().encode()))
    git(root, "commit", "-qam", "a text program recorded")


@pytest.mark.parametrize(("before", "after", "kept"), [
    ("import math as math\nprint(math.pi)\n", "import math as m\nprint(m.pi)\n", True),
    ("import math\nprint(math.pi)\n", "import math as m\nprint(m.pi)\n", True),
    ("import math\nimport json\nprint(math.pi, json)\n",
     "import math as json\nimport json\nprint(json.pi, json)\n", False),
    ("from json import dumps\nprint(dumps(1))\n", "from json import dumps as d\nprint(d(1))\n", True),
    ("import os.path\nprint(os.__name__)\n", "import os.path as m\nprint(m.__name__)\n", False),
    ("import os.path as m\nprint(m.__name__)\n", "import os.path\nprint(os.__name__)\n", False),
    ("a, b = 1, 2\nprint(a, b)\n", "b, a = 1, 2\nprint(b, a)\n", True),
    ("value = 9\ndef f(unused=0):\n    return value\nprint(f())\n",
     "value = 9\ndef f(value=0):\n    return value\nprint(f())\n", False),
    ("x = 1\nprint(f'{x!r}')\n", "x = 1\nprint(f'{x!s}')\n", False),
    ("x = 1\nprint(f'{x=}')\n", "x = 1\nprint(f'{ x = }')\n", False),
    ("print(dict(steps=3))\nsteps = 1\n", "print(dict(num_steps=3))\nsteps = 1\n", True),
], ids=["an alias renamed", "an alias added", "an alias that takes another import's name",
        "an imported name given an alias", "an alias added to a dotted import", "an alias taken from one",
        "two variables swapped",
        "a parameter renamed to another variable's name", "an f-string's conversion", "an f-string's label",
        "a keyword renamed beside a variable of its spelling"])
def test_the_programs_parsed_roles_decide_what_is_a_rename(site, before, after, kept):
    """A rename is read in the parsed program: an import's name and its binding apart, a parameter a
    variable, an f-string's conversion and literal text what it prints."""
    recorded(site, before)
    was = recordings(site)
    (site / "site/src/data/text.py").write_text(after)
    assert redigest(site).returncode == (0 if kept else 1)
    changed = {path for path in was if was[path] != recordings(site)[path]}
    assert changed == ({"site/public/examples/text/capture.json"} if kept else set())


def test_a_refusal_found_after_a_good_rename_writes_nothing(site):
    """The hero's cell digest written otherwise than the tool would write it: the rename of its script and
    cell is found good, the cell's digest cannot be placed, and neither digest is written."""
    capture = site / "site/src/data/capture.json"
    capture.write_bytes(capture.read_bytes().replace(b'"cell_sha256": "', b'"cell_sha256":"'))
    git(site, "commit", "-qam", "a capture written by hand")
    before = recordings(site)
    edit(site, "site/src/data/hero.py", "OptimConfig(schedule=", "OptimConfig(rate=")
    assert redigest(site).returncode == 1
    assert recordings(site) == before


def test_a_recording_already_of_other_code_writes_nothing(site):
    stale = site / "site/public/examples/framework/jepa/capture.json"
    capture = json.loads(stale.read_text())
    stale.write_text(stale.read_text().replace(capture["script_sha256"], "0" * 64))
    git(site, "commit", "-qam", "a stale recording")
    before = recordings(site)
    edit(site, "site/snippets/framework.py", "OptimConfig(learning_rate=", "OptimConfig(rate=")
    assert redigest(site).returncode == 1
    assert recordings(site) == before
