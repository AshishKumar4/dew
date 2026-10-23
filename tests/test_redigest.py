"""An API rename keeps the landing page's recordings: site/scripts/redigest.py, on a copy of the site in a git
repository of its own, writes the renamed programs' digests and leaves their outputs as they were, and writes
nothing when a change is more than a renaming or a recording was already of other code."""

import http.server
import json
import os
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


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"})


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """The page's sources and recordings in a repository whose `recorded` branch is what they ran."""
    for relative in COPIED:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, tmp_path / relative)
    shutil.copytree(REPO / "site/public/examples", tmp_path / "site/public/examples")
    git(tmp_path, "init", "-q", "-b", "recorded")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "recorded")
    return tmp_path


def redigest(root: Path, hub: str = "http://127.0.0.1:9") -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(root / "site/scripts/redigest.py"), "--base", "recorded"],
                          capture_output=True, text=True, timeout=60, env={**os.environ, "HF_ENDPOINT": hub})


def recordings(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted((root / "site").rglob("*"))
            if path.is_file() and ("public" in path.parts or path.name == "capture.json")}


def program_digest(root: Path, cell: str) -> str:
    """The sha256 the page checks a cell's recording against, from the copy's own cells.py."""
    code = ("import sys; sys.path.insert(0, 'site/snippets'); import cells; "
            f"print(cells.digest(cells.cell({cell!r})))")
    return subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


def rename(root: Path, relative: str, old: str, new: str) -> None:
    (root / relative).write_text((root / relative).read_text().replace(old, new))


def test_a_rename_rewrites_the_digests_of_the_programs_it_reaches_and_keeps_their_outputs(site):
    before = recordings(site)
    rename(site, "site/snippets/framework.py", "OptimConfig(learning_rate=", "OptimConfig(rate=")
    rename(site, "site/src/data/hero.py", "OptimConfig(schedule=", "OptimConfig(rate=")
    assert redigest(site).returncode == 0
    after = recordings(site)
    changed = {path for path in before if before[path] != after[path]}
    assert changed and all(path.endswith("capture.json") for path in changed)
    for path in changed:
        # Only digests differ: a cell's one, the hero's two (its script's and its cell's).
        lines = zip(before[path].decode().splitlines(), after[path].decode().splitlines(), strict=True)
        assert all(re.fullmatch(r'\s*"(script|cell)_sha256": "[0-9a-f]{64}",?', now)
                   for was, now in lines if was != now)
    lm = json.loads((site / "site/public/examples/framework/lm/capture.json").read_text())
    assert lm["script_sha256"] == program_digest(site, "lm")
    assert "site/src/data/capture.json" in changed
    # Run again, it finds every recording of the program the page shows.
    assert redigest(site).returncode == 0 and recordings(site) == after


@pytest.mark.parametrize(("old", "new"), [
    ("OptimConfig(learning_rate=1e-3)", "OptimConfig(learning_rate=2e-3)"),
    ("OptimConfig(learning_rate=1e-3)", "OptimConfig(learning_rate=1e-3, clip=1.0)"),
    ("trainer = Trainer(LMObjective(model,", "trainer = Trainer(LMObjective(net,"),
    ("trainer = Trainer(LMObjective(model,", "trainer = Trainer(LMObjective(data,"),
], ids=["a value", "an argument", "a variable renamed in one place", "a variable renamed to another's name"])
def test_a_change_that_is_more_than_a_renaming_writes_nothing(site, old, new):
    before = recordings(site)
    # The last two rename the lm cell's `model` where it trains: to a new name, which leaves it two names
    # for one, and to `data`, another variable's.
    text = (site / "site/snippets/framework.py").read_text()
    (site / "site/snippets/framework.py").write_text(text.replace(old, new, 1))
    assert redigest(site).returncode == 1
    assert recordings(site) == before


def test_a_recording_already_of_other_code_writes_nothing(site):
    stale = site / "site/public/examples/framework/lm/capture.json"
    capture = json.loads(stale.read_text())
    stale.write_text(stale.read_text().replace(capture["script_sha256"], "0" * 64))
    git(site, "commit", "-qam", "a stale recording")
    before = recordings(site)
    rename(site, "site/snippets/framework.py", "OptimConfig(learning_rate=", "OptimConfig(rate=")
    assert redigest(site).returncode == 1
    assert recordings(site) == before


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
    rename(site, "site/src/data/sampler.py", old, new)
    assert redigest(site, endpoint).returncode == kept
    after = recordings(site)
    sampler = "site/public/examples/sampler/capture.json"
    assert {path for path in before if before[path] != after[path]} == ({sampler} if kept == 0 else set())
