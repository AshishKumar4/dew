"""site/scripts/digest.py, on a copy of the site: a recorded program's capture takes the digest of the program
as the page now shows it, and nothing else in it changes, byte for byte; the other recordings stay as they
were, and a name no recording has is refused."""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COPIED = ("site/snippets/cells.py", "site/snippets/cells.json",
          "site/snippets/framework.py", "site/src/data/hero.py", "site/src/data/text.py",
          "site/src/data/sampler.py", "site/src/data/capture.json")
DIGEST = re.compile(rb"[0-9a-f]{64}")


@pytest.fixture
def site(tmp_path: Path) -> Path:
    for relative in COPIED:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, tmp_path / relative)
    shutil.copytree(REPO / "site/public/examples", tmp_path / "site/public/examples")
    return tmp_path


def digest(root: Path, *names: str) -> subprocess.CompletedProcess:
    """site/scripts/digest.py run on the copy, as its own: it reads and writes the site beside it."""
    (root / "site/scripts").mkdir(exist_ok=True)
    shutil.copyfile(REPO / "site/scripts/digest.py", root / "site/scripts/digest.py")
    return subprocess.run([sys.executable, str(root / "site/scripts/digest.py"), *names], capture_output=True,
                          text=True, timeout=60)


def captures(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted((root / "site").rglob("capture.json"))}


def program(root: Path, name: str) -> str:
    """Program `name` as the copy's own cells.py gives it."""
    code = ("import json, sys; sys.path.insert(0, 'site/snippets'); import cells; "
            f"print(json.dumps(cells.programs()[{name!r}]))")
    return json.loads(subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True,
                                     text=True).stdout)


def test_a_recording_takes_its_programs_new_digest_and_keeps_every_other_byte(site):
    framework = site / "site/snippets/framework.py"
    framework.write_text(framework.read_text() + "\n# a line no cell holds\n")
    hero = site / "site/src/data/hero.py"
    hero.write_text(hero.read_text() + "\n# a comment the hero's cell shows\n")
    before = captures(site)
    assert digest(site, "lm", "hero", "hero.py").returncode == 0
    after = captures(site)
    changed = {path for path in before if before[path] != after[path]}
    assert changed == {"site/src/data/capture.json"}, "the lm cell did not change, so neither its digest"
    for path in before:
        assert DIGEST.sub(b"", before[path]) == DIGEST.sub(b"", after[path])
    meta = json.loads(after["site/src/data/capture.json"])["meta"]
    assert (meta["cell_sha256"], meta["script_sha256"]) == tuple(
        hashlib.sha256(program(site, name).encode()).hexdigest() for name in ("hero", "hero.py"))


def test_a_name_no_recording_has_is_refused_and_nothing_written(site):
    before = captures(site)
    assert digest(site, "lm", "absent").returncode == 2
    assert captures(site) == before
