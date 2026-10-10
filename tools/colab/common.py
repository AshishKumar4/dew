"""Private state and process primitives shared by the Colab controller and VM."""

import contextlib
import fcntl
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE = Path(os.environ.get("DEW_COLAB_STATE", Path.home() / ".cache/dew"))
PART = 50 * 1024 * 1024


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid session/attempt name")
    return value


@contextlib.contextmanager
def lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def redact(text: str) -> str:
    return re.sub(r"(colab-runtime-proxy-token=)[^&\s]+", r"\1[REDACTED]", text)


def command(args: Sequence[str], **kwargs) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(redact(f"{list(args)[:3]}: {result.stdout[-500:]} {result.stderr[-500:]}"))
    return result


def git(repo: Path, *args: str) -> str:
    return command(["git", "-C", str(repo), *args], timeout=300).stdout.strip()


def driver_sha() -> str:
    return git(REPO, "rev-parse", "HEAD")
