"""An API rename keeps the landing page's recordings: site/scripts/redigest.py, on a copy of the site in a git
repository of its own, writes the renamed programs' digests and leaves every other byte of their recordings as
it was, and writes nothing when a change is more than a renaming or a recording was already of other code."""

import http.server
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COPIED = ("site/scripts/redigest.py", "site/snippets/cells.py", "site/snippets/cells.json",
          "site/snippets/framework.py", "site/src/data/hero.py", "site/src/data/text.py",
          "site/src/data/sampler.py", "site/src/data/capture.json", "site/live/container/text-to-image")
DIGEST = re.compile(rb"[0-9a-f]{64}")


def load_redigest():
    spec = importlib.util.spec_from_file_location("redigest", REPO / "site/scripts/redigest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


redigest_module = load_redigest()


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"})


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


def redigest(root: Path, hub: str = "http://127.0.0.1:9") -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(root / "site/scripts/redigest.py"), "--base", "recorded"],
                          capture_output=True, text=True, timeout=60,
                          env={"HF_ENDPOINT": hub, "PATH": "/usr/bin:/bin"})


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
    ("site/src/data/hero.py", "held = val.pipe", "data = val.pipe"),
    ("site/src/data/text.py", "model", "__debug__"),
], ids=["a value", "an argument", "a variable renamed in one place", "a variable renamed to another's name",
        "two variables become one", "a rename that does not compile"])
def test_a_change_that_is_more_than_a_renaming_writes_nothing(site, path, old, new):
    before = recordings(site)
    edit(site, path, old, new, -1 if path.endswith("text.py") or "held" in old else 1)
    if "held" in old:
        edit(site, path, "validation=held", "validation=data")
    assert redigest(site).returncode == 1
    assert recordings(site) == before


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


@pytest.mark.parametrize(("before", "after", "renames"), [
    ("import math\nprint(math.pi)\n", "import math as m\nprint(m.pi)\n", None),
    ("import math as math\nprint(math.pi)\n", "import math as m\nprint(m.pi)\n", "math -> m"),
    ("from dew import Old\nOld()\n", "from dew import New\nNew()\n", "Old -> New"),
    ("a, b = 1, 2\nprint(a, b)\n", "b, a = 1, 2\nprint(b, a)\n", "a -> b, b -> a"),
    ("value = 9\ndef f(unused=0):\n    return value\nprint(f())\n",
     "value = 9\ndef f(value=0):\n    return value\nprint(f())\n", None),
    ("x = 1\nprint(f'{x!r}')\n", "x = 1\nprint(f'{x!s}')\n", None),
    ("x = 1\nprint(f'{x=}')\n", "x = 1\nprint(f'{ x = }')\n", None),
    ("f(steps=3)\nsteps = 1\n", "f(num_steps=3)\nsteps = 1\n", "steps -> num_steps"),
], ids=["an alias added", "an alias renamed", "an imported name renamed", "two variables swapped",
        "a parameter renamed to another variable's name", "an f-string's conversion", "an f-string's label",
        "a keyword renamed beside a variable of its spelling"])
def test_the_programs_parsed_roles_decide_what_is_a_rename(before, after, renames):
    """A rename is read in the parsed program: an import's name and its binding apart, a parameter a
    variable, an f-string's conversion and literal text its output; None where it is no rename."""
    if renames is None:
        with pytest.raises(ValueError):
            redigest_module.renamed(before, after, ("", ""))
    else:
        assert redigest_module.renamed(before, after, ("", "")).printed() == renames


@pytest.fixture
def hub():
    """The Hub's tree listing for the pinned model's revisions, as `trees` (a revision to its files) holds
    them; its endpoint."""
    trees: dict[str, list[dict]] = {}

    class Tree(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            revision = self.path.split("/tree/", 1)[1].split("?", 1)[0]
            body = json.dumps(trees[revision]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Tree)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", trees
    server.shutdown()


def listing(weights: str, record: str) -> list[dict]:
    """A revision's files: a weight object (an LFS file) and a run's record (a plain one)."""
    return [{"type": "file", "path": "1350000/d/432fa0667", "oid": "a", "size": 9, "lfs": {"oid": weights}},
            {"type": "file", "path": "run.json", "oid": record, "size": 10}]


@pytest.mark.parametrize(("weights", "kept"), [("same", 0), ("other", 1)],
                         ids=["same weights", "other weights"])
def test_a_pin_moved_to_the_same_weights_keeps_the_sampler_and_to_other_weights_records_again(
        site, hub, weights, kept):
    endpoint, trees = hub
    owner = site / "site/live/container/text-to-image"
    repository, old = owner.read_text().strip().split("@")
    new = "1" * 40
    trees[old], trees[new] = listing("same", "record"), listing(weights, "rewritten record")
    before = recordings(site)
    owner.write_text(f"{repository}@{new}\n")
    edit(site, "site/src/data/sampler.py", old, new)
    assert redigest(site, endpoint).returncode == kept
    after = recordings(site)
    sampler = "site/public/examples/sampler/capture.json"
    assert {path for path in before if before[path] != after[path]} == ({sampler} if kept == 0 else set())
