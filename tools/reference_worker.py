"""The protocol a recorder and a reference's worker speak (tools/dreamerv3_worker.py, tools/kimi_worker.py,
tools/maxtext_worker.py, tools/mamba_selective_reference.py, tools/fla_gdn_reference.py,
tools/remote_code_reference.py, tools/llada_moe_reference.py), the worker run in the reference's own
environment, which holds no Dew.

A request is a dict of arrays and an answer is one, as the worker's `run` gives it: a caller's options (a
dtype, a site) ride among the arrays as 0-d arrays. A worker's first line says it is ready, which carrier
its arrays travel by and what it has to tell the recorder of how it started (`serve`):

- `paths`: the recorder writes the request to a file and sends the line `IN OUT`, the worker writes its
  answer to OUT and replies with the line `done`. IN holds cases, each `<case>.<input>` and
  `<case>.param.<path>` per parameter by its path in the reference's own layout; OUT gets each case's
  outputs `<case>.reference.<leaf>` and `<case>.truth.<leaf>`, the reference run in float32 and in float64
  (`answered`).
- `frames`: both ways, an 8-byte little-endian size and an npz of that many bytes (`send`, `received`).

A worker's end of the pipe is its stdout, which carries nothing but the protocol: a library's print, a
native write to the descriptor and a child that inherited it all reach stderr, as a line a library printed
was once read as a request's answer and a recorder waited on a live process for 4.5 hours (`reserved`).
A recorder starts a worker through `session`, which runs the worker's script under this file, so that is
done before the script imports anything, and which owns the worker's process group, its pipes and its
scratch files until its block ends, however it ends. Ending a worker is killing its group: it holds
nothing to save, and nothing waits on a live process.
"""

from __future__ import annotations

import contextlib
import functools
import io
import os
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np

type Applied = Callable[[dict[str, object], dict[str, np.ndarray]], Mapping[str, object]]
"""The reference's run of a case: its parameters and inputs to its outputs by leaf."""


def answered(cases: Mapping[str, np.ndarray],
             applying: Callable[[str], contextlib.AbstractContextManager[Applied]]) -> dict[str, np.ndarray]:
    """Every case's outputs in float32 (`reference`) and float64 (`truth`): JAX's float64 enabled for the
    float64 run alone, each run inside `applying(dtype)`, which yields the reference's run at that dtype,
    every case's parameters given it at that dtype."""
    import jax
    import jax.numpy as jnp

    names = sorted({key.split(".", 1)[0] for key in cases})
    out = {}
    for dtype, kind, wide in (("float32", "reference", False), ("float64", "truth", True)):
        jax.config.update("jax_enable_x64", wide)
        with applying(dtype) as apply:
            for name in names:
                prefix, given = f"{name}.param.", f"{name}."
                params = {key.removeprefix(prefix): jnp.asarray(value, dtype) for key, value in cases.items()
                          if key.startswith(prefix)}
                case = {key.removeprefix(given): value for key, value in cases.items()
                        if key.startswith(given) and not key.startswith(prefix)}
                for leaf, value in apply(params, case).items():
                    out[f"{name}.{kind}.{leaf}"] = np.asarray(value, dtype)
    return out


def paired(reference: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A worker's answer from its reference run's leaves and its float64 truth's: `reference.<leaf>` and
    `truth.<leaf>` (`arms` reads them)."""
    return {**{f"reference.{leaf}": value for leaf, value in reference.items()},
            **{f"truth.{leaf}": value for leaf, value in truth.items()}}


def arms(answer: Mapping[str, np.ndarray], prefix: str = "") -> tuple[dict[str, np.ndarray],
                                                                      dict[str, np.ndarray]]:
    """The reference's leaves and the truth's of a worker's answer, each by its leaf: the arrays named
    `<prefix>reference.<leaf>` and `<prefix>truth.<leaf>` (`paired`, `answered`'s `<case>.`)."""
    def arm(kind: str) -> dict[str, np.ndarray]:
        head = f"{prefix}{kind}."
        return {leaf.removeprefix(head): value for leaf, value in answer.items() if leaf.startswith(head)}

    return arm("reference"), arm("truth")


def send(stream: BinaryIO, arrays: Mapping[str, np.ndarray]) -> None:
    """`arrays` as one frame to the other end of a pipe (`received`)."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    stream.write(struct.pack("<Q", buffer.getbuffer().nbytes))
    stream.write(buffer.getvalue())
    stream.flush()


def received(stream: BinaryIO) -> dict[str, np.ndarray] | None:
    """The arrays of the next frame the other end sent (`send`), or None where it closed the pipe between
    frames. A frame cut short, in its size or its bytes, is an EOFError."""
    header = stream.read(8)
    if not header:
        return None
    if len(header) < 8:
        raise EOFError(f"the pipe closed {len(header)} bytes into a frame's 8-byte size")
    (size,) = struct.unpack("<Q", header)
    payload = stream.read(size)
    if len(payload) < size:
        raise EOFError(f"the pipe closed {len(payload)} bytes into a frame of {size}")
    with np.load(io.BytesIO(payload)) as arrays:
        return dict(arrays)


@functools.cache
def reserved() -> BinaryIO:
    """The worker's end of the protocol: its own copy of fd 1, which stdout then no longer is. Fd 1 becomes
    stderr, so a Python print, a native library's write and a child that inherited fd 1 all reach the
    recorder's stderr, and the pipe carries what `serve` writes alone. Once for the process, as a second
    copy would be a copy of stderr: `session` has a worker's script run under this file, which calls it
    before the script imports anything, and `serve` calls it again."""
    answers = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return answers


def serve(run: Callable[[dict[str, np.ndarray]], Mapping[str, np.ndarray]], ready: str = "", *,
          frames: bool = False) -> None:
    """The worker's side: the line `ready` (carrier and `ready`, the text the recorder reads as the worker's
    startup, a line of its own), then each request answered by `run` over its arrays, until the recorder
    closes the pipe. A `run` that raises ends the worker, its traceback on stderr: what a failed request
    left in the libraries is unknown, and no later request should run on it."""
    answers = reserved()
    requests = sys.stdin.buffer
    answers.write(f"ready {'frames' if frames else 'paths'} {ready}\n".encode())
    answers.flush()
    if frames:
        while (request := received(requests)) is not None:
            send(answers, run(request))
        return
    for line in requests:
        given, out = line.decode().split()
        with np.load(given) as cases:
            np.savez(out, **run(dict(cases)))
        answers.write(b"done\n")
        answers.flush()


def bootstrap(script: str, arguments: Sequence[str]) -> None:
    """The worker `script` run as `__main__` with `arguments`, its stdout reserved before it imports a
    library (`reserved`)."""
    reserved()
    sys.argv[:] = [script, *arguments]
    runpy.run_path(script, run_name="__main__")


class Session:
    """A recorder's end of a worker's protocol (`session`): `ask` sends a request and returns its answer."""

    def __init__(self, process: subprocess.Popen[bytes], script: str, directory: Path):
        self.process, self.script, self.directory = process, Path(script).name, directory
        line = process.stdout.readline().decode().strip()
        words = line.split(maxsplit=2)
        if words[:1] != ["ready"] or words[1:2] not in (["paths"], ["frames"]):
            raise self._ended("did not say it was ready", line)
        self.frames = words[1] == "frames"
        self.started = words[2] if len(words) > 2 else ""
        """The worker's account of how it started (`serve`'s `ready`)."""

    def _ended(self, what: str, said: str = "") -> RuntimeError:
        """The error for a worker that `what`, with its exit status where it is known by now and the line it
        said where it said one."""
        code = self.process.poll()
        return RuntimeError(f"the worker {self.script} {what}" + (f", saying {said!r}" if said else "")
                            + ("" if code is None else f" (exit {code})"))

    def ask(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The worker's answer to a request of `arrays`. A worker that ended on it, or answered it wrongly,
        is an error."""
        try:
            answer = self._framed(arrays) if self.frames else self._by_paths(arrays)
        except (BrokenPipeError, EOFError) as error:
            raise self._ended("ended on a request") from error
        if answer is None:
            raise self._ended("ended on a request")
        return answer

    def _framed(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        send(self.process.stdin, arrays)
        return received(self.process.stdout)

    def _by_paths(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        given, out = self.directory / "in.npz", self.directory / "out.npz"
        np.savez(given, **arrays)
        self.process.stdin.write(f"{given} {out}\n".encode())
        self.process.stdin.flush()
        reply = self.process.stdout.readline()
        if not reply:
            return None
        if reply != b"done\n":
            raise self._ended("answered a request with something else than done", reply.decode().strip())
        with np.load(out) as answer:
            return dict(answer)


@contextlib.contextmanager
def session(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> Iterator[Session]:
    """`command`, `[python, script, *arguments]`, started as a worker (`serve`): its script run under this
    file (`bootstrap`) as a group its recorder owns (dew._cleanup's `owned_group`), whose lifeline kills it
    if the recorder dies without unwinding, and `ask`ed through the `Session` the block holds. `env` is its
    environment, the recorder's where none is given, but for TMPDIR: the worker's temporary files are the
    session's, in a directory removed after the worker's group is stopped. However the block ends, an error
    among them, the group is stopped and reaped, its pipes closed and the directory removed, each step
    attempted whichever fail, the block's own error the one raised."""
    from dew._cleanup import attempted, owned_group

    python, script, *arguments = command
    root = Path(tempfile.mkdtemp(prefix="reference-"))
    try:
        (root / "io").mkdir()
        (root / "tmp").mkdir()
        environment = {**(os.environ if env is None else env), "TMPDIR": str(root / "tmp")}
        started = [python, __file__, script, *arguments]
        with owned_group(started, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=environment) as process:
            yield Session(process, script, root / "io")
    finally:
        attempted([("remove the session's scratch directory", lambda: shutil.rmtree(root))],
                  primary=sys.exception())


__all__ = ["Applied", "Session", "answered", "arms", "paired", "received", "reserved", "send", "serve",
           "session"]


if __name__ == "__main__":
    # Imported under its own name, so the worker's script, which imports `reference_worker`, shares this
    # module's one reserved channel.
    import reference_worker

    reference_worker.bootstrap(sys.argv[1], sys.argv[2:])
