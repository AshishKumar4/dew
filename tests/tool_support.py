"""Source-file loading for tools exercised by several test suites, git in a test's scratch repository, and a
node session's side of a measurement window."""

import importlib.util
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def load(name: str, directory: str = "tools") -> ModuleType:
    """The repository's `directory/name.py` as the module `name_under_test`,
    registered before it runs so its dataclasses resolve their module."""
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / directory / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def run_from(name: str, directory: str) -> Iterator[ModuleType]:
    """`directory/name.py` loaded as `load` loads it, with its directory first on sys.path while held: a
    script run from its directory, which imports its siblings."""
    path = str(REPO_ROOT / directory)
    sys.path.insert(0, path)
    try:
        yield load(name, directory)
    finally:
        sys.path.remove(path)


def git(root: Path, *arguments: str) -> str:
    """Git in the scratch repository `root`, its author and committer named, its output returned; a
    failure raises."""
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}).stdout


def linted(root: Path, files: dict[str, str]) -> list[tuple[str, int, str]]:
    """`files` written into a sparx checkout at `root`, each finding tools/lint_slop.py reports in it as
    (path, line, code)."""
    lint = load("lint_slop")
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    modules = list(lint.collect(["src/sparx", "tests", "tools"], root, "sparx"))
    return sorted((finding.path, finding.line, finding.code)
                  for module in modules for finding in lint.check(module))


def lint_module(lint: ModuleType, root: Path, parts: tuple[str, ...], body: str):
    """`body` written at `parts` under `root`, a checkout of that one file, as `lint` (tools/lint_slop.py)
    collects it."""
    import textwrap

    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    [module] = lint.collect(["/".join(parts)], root, "dew")
    return module


def lint_codes(lint: ModuleType, root: Path, parts: tuple[str, ...], body: str) -> list[str]:
    """The codes `lint` reports for `body` written at `parts` under `root` (`lint_module`)."""
    return [finding.code for finding in lint.check(lint_module(lint, root, parts, body))]


class StandInSession:
    """A node session's side of a measurement window, as a job meets it: `$SESSION` (the directory holding
    `wake`, the FIFO a session reads) and `$OUT` (its job's directory, where the job writes `window` and the
    session answers `window.granted`). A thread plays the session and reads each wake. A request is granted
    once `grants` is set, or answered in its words by `refusal`. A window its job ended is acknowledged (its
    grant removed) once `acks` is set, after `on_release` has run. Both events are set to begin with; a test
    clears one to hold the session back. `asked` is each request's text, `granted` and `released` when each
    happened."""

    def __init__(self, root: Path):
        self.session = root / "session"
        self.out = self.session / "out"
        self.out.mkdir(parents=True)
        os.mkfifo(self.session / "wake")
        # Opened to read and write, as a session opens it: a write never waits for a reader.
        self.wake = os.open(self.session / "wake", os.O_RDWR)
        self.asked: list[str] = []
        self.granted: list[float] = []
        self.released: list[float] = []
        self.grants, self.acks = threading.Event(), threading.Event()
        self.grants.set()
        self.acks.set()
        self.refusal: str | None = None
        self.on_release: Callable[[], None] | None = None
        self.closing = False
        self.thread = threading.Thread(target=self.serve, daemon=True)

    @property
    def environment(self) -> dict[str, str]:
        """What a job's process is given of its session."""
        return {"SESSION": str(self.session), "OUT": str(self.out)}

    def __enter__(self) -> "StandInSession":
        self.thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.closing = True
        self.grants.set()
        self.acks.set()
        os.write(self.wake, b"x")
        self.thread.join(10)
        os.close(self.wake)

    def serve(self) -> None:
        request, granted = self.out / "window", self.out / "window.granted"
        while os.read(self.wake, 1) and not self.closing:
            if request.exists() and not granted.exists():
                self.asked.append(request.read_text())
                if self.refusal is not None:
                    request.unlink()
                    written = self.out / ".window.refused"
                    written.write_text(self.refusal)
                    written.replace(self.out / "window.refused")
                    continue
                self.grants.wait()
                self.granted.append(time.time())
                granted.touch()
            elif granted.exists() and not request.exists():
                self.released.append(time.time())
                if self.on_release is not None:
                    self.on_release()
                self.acks.wait()
                granted.unlink()
