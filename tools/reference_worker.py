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
import json
import os
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from pathlib import Path
from typing import IO, BinaryIO

import numpy as np

type Applied = Callable[[Mapping[str, object], dict[str, np.ndarray]], Mapping[str, object]]
"""The reference's run of a case: its parameters and inputs to its outputs by leaf."""


def draws(cases: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """A request's draws, each its inputs by name: the arrays `<draw>.<input>`."""
    found: dict[str, dict[str, np.ndarray]] = {}
    for key, value in cases.items():
        name, _, given = key.partition(".")
        found.setdefault(name, {})[given] = value
    return dict(sorted(found.items()))


def drawn(cases: Mapping[str, np.ndarray],
          run: Callable[[dict[str, np.ndarray]], tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]]
          ) -> dict[str, np.ndarray]:
    """A torch worker's answer: each draw's (`draws`) reference leaves and float64 truth's, `run(inputs)`,
    keyed under the draw (`paired`)."""
    return {f"{name}.{key}": value for name, inputs in draws(cases).items()
            for key, value in paired(*run(inputs)).items()}


def answered(cases: Mapping[str, np.ndarray],
             applying: Callable[[str], contextlib.AbstractContextManager[Applied]]) -> dict[str, np.ndarray]:
    """Every case's outputs in float32 (`reference`) and float64 (`truth`): JAX's float64 enabled for the
    float64 run alone, each run inside `applying(dtype)`, which yields the reference's run at that dtype,
    every case's parameters given it at that dtype."""
    import jax
    import jax.numpy as jnp

    out = {}
    for dtype, kind, wide in (("float32", "reference", False), ("float64", "truth", True)):
        jax.config.update("jax_enable_x64", wide)
        with applying(dtype) as apply:
            for name, inputs in draws(cases).items():
                params = {key.removeprefix("param."): jnp.asarray(value, dtype)
                          for key, value in inputs.items() if key.startswith("param.")}
                case = {key: value for key, value in inputs.items() if not key.startswith("param.")}
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


FRAME = struct.Struct("<Q")
"""A frame's size, its header: what `send` writes before the frame's bytes and `received` reads first."""


def send(stream: IO[bytes], arrays: Mapping[str, np.ndarray]) -> None:
    """`arrays` as one frame to the other end of a pipe (`received`)."""
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    stream.write(FRAME.pack(buffer.getbuffer().nbytes))
    stream.write(buffer.getvalue())
    stream.flush()


def received(stream: IO[bytes]) -> dict[str, np.ndarray] | None:
    """The arrays of the next frame the other end sent (`send`), or None where it closed the pipe between
    frames. A frame cut short, in its size or its bytes, is an EOFError."""
    header = stream.read(FRAME.size)
    if not header:
        return None
    if len(header) < FRAME.size:
        raise EOFError(f"the pipe closed {len(header)} bytes into a frame's {FRAME.size}-byte size")
    (size,) = FRAME.unpack(header)
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
            np.savez(out, allow_pickle=False, **run(dict(cases)))
        answers.write(b"done\n")
        answers.flush()


def bootstrap(script: str, arguments: Sequence[str]) -> None:
    """The worker `script` run as `__main__` with `arguments`, its stdout reserved before it imports a
    library (`reserved`)."""
    reserved()
    sys.argv[:] = [script, *arguments]
    runpy.run_path(script, run_name="__main__")


def merged_units(units: Sequence[Path], fixture: Path, license: str, base: Path | None = None) -> Path:
    """A site recording's units (each recorder's `write_seeded`) joined over the `fixture`'s recording as
    pinned (`base`, `pinned_base`; none for a first recording) and saved as the fixture under `license`
    (fetched_fixtures' `save`), their arrays `merged_arrays`'. Merged from its units alone, a recording of a
    few sites dropped every other site the fixture held (M-JS-233: 22 of 93 kept)."""
    from fetched_fixtures import save

    stored = merged_arrays(units, base)
    reference = json.loads(str(stored["provenance"]))["reference"]
    return save(fixture, stored, reference=reference, license=license)


def merged_arrays(units: Sequence[Path], base: Path | None = None) -> dict[str, np.ndarray]:
    """`merged_units`' arrays: the units' (their keys disjoint: a key recorded twice is refused) over
    `base`'s, a site the units record replacing every `<site>.*` key of the base's, a site they do not
    keeping its keys and the device it was recorded on. Their provenance the units' own, which must agree
    but for the device, and the base's too (else refused: its sites cannot sit beside theirs); where the
    sites ran on more than one device, it names each device's sites (`devices`), as a reference's library
    may tune each device apart (fla autotunes its configs per device)."""
    stored: dict[str, np.ndarray] = {}
    devices: dict[str, list[str]] = {}
    common = None

    def placed(site: str, device: str) -> None:
        if site not in devices.setdefault(device, []):
            devices[device].append(site)

    for unit in units:
        with np.load(unit, allow_pickle=False) as part:
            recorded = json.loads(str(part["provenance"]))
            device = recorded.pop("device")
            if common is not None and recorded != common:
                raise ValueError(f"{unit} was recorded under another provenance: {recorded}")
            common = recorded
            for key, value in part.items():
                if key == "provenance":
                    continue
                if key in stored:
                    raise ValueError(f"{unit} records {key} again")
                stored[key] = value
                placed(key.split(".", 1)[0], device)
    if common is None:
        raise ValueError("a fixture merges at least one unit")
    if base is not None:
        kept, placed_on = kept_sites(base, {key.split(".", 1)[0] for key in stored}, common)
        stored.update(kept)
        for site, device in placed_on.items():
            placed(site, device)
    named = {"device": next(iter(devices))} if len(devices) == 1 else {"device": " and ".join(devices),
                                                                       "devices": devices}
    stored["provenance"] = np.asarray(json.dumps({**common, **named}))
    return stored


def kept_sites(base: Path, redone: Collection[str],
               common: Mapping[str, object]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """What a recording of the sites `redone` keeps of its pinned recording `base`: the keys of every other
    site, in their order, and each kept site's device. Refused where a site is kept and `base` was recorded
    under another provenance than `common` (the new recording's, its device aside): its sites cannot sit
    beside the new ones. A recording of every site again keeps none, whatever `base`'s provenance (row
    12337: a whole re-recording under a new release was refused). The one overlay of a pinned recording, a
    merge of units' (`merged_arrays`) and a recorder's own (tools/mamba2_reference.py's `write_seeded`)."""
    with np.load(base, allow_pickle=False) as part:
        recorded = json.loads(str(part["provenance"]))
        named, device = recorded.pop("devices", None), recorded.pop("device")
        names = [key for key in part.files if key != "provenance" and key.split(".", 1)[0] not in redone]
        if names and recorded != common:
            raise ValueError(f"{base} was recorded under another provenance ({recorded}) than its sites "
                             f"recorded again ({common}): record every site of it again")
        # Each kept key read alone, once its keeping is allowed: a site recorded again is never decompressed.
        kept = {key: part[key] for key in names}
    owner = {site: name for name, sites in named.items() for site in sites} if named else {}
    sites = dict.fromkeys(key.split(".", 1)[0] for key in kept)
    return kept, {site: owner[site] if named else device for site in sites}


def pinned_base(fixture: Path) -> Path | None:
    """`fixture`'s recording as it is pinned now, which a merge of a few sites' units keeps the others of:
    its file in the local store checked against its pin's digest, or the file committed at its place
    (fetched_fixtures' `local`, which refuses a pinned file not fetched); None where there is none (a
    first recording)."""
    from fetched_fixtures import local

    found = local(fixture)
    return found if found.is_file() else None


def recording_command(script: str, sites: str, *, serve: Callable[[], None], write_seeded: Callable,
                      fixture: Path, license: str) -> None:
    """A site recorder's command line (tools/fla_gdn_reference.py, tools/mamba3_reference.py): `--serve`, the
    reference's side (`serve`); `--list`, its site names a line each (the module `sites`' `sites()`);
    `--merge <units>`, the `fixture` from the recording's units over its pinned recording's other sites
    under its reference's `license` (`merged_units`, `pinned_base`); else `<python> <out> <site>...`, those
    sites' unit written to `out` (`write_seeded`) against `script`'s worker run in `<python>` (`session`)."""
    if sys.argv[1] == "--serve":
        serve()
    elif sys.argv[1] == "--list":
        import importlib

        print("\n".join(importlib.import_module(sites).sites()), flush=True)
    elif sys.argv[1] == "--merge":
        written = merged_units([Path(unit) for unit in sys.argv[2:]], fixture, license, pinned_base(fixture))
        print(f"wrote {fixture.name}: {written} ({written.stat().st_size:,} bytes)", flush=True)
    else:
        with session([sys.argv[1], script, "--serve"]) as worker:
            write_seeded(worker, sys.argv[3:], Path(sys.argv[2]))


class Session:
    """A recorder's end of a worker's protocol (`session`): `ask` sends a request and returns its answer."""

    def __init__(self, process: subprocess.Popen[bytes], script: str, directory: Path):
        self.process, self.script, self.directory = process, Path(script).name, directory
        if process.stdin is None or process.stdout is None:
            raise ValueError(f"{self.script}'s worker was started without its stdin and stdout piped")
        self.stdin, self.stdout = process.stdin, process.stdout
        line = self.stdout.readline().decode().strip()
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
        send(self.stdin, arrays)
        return received(self.stdout)

    def _by_paths(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        given, out = self.directory / "in.npz", self.directory / "out.npz"
        np.savez(given, allow_pickle=False, **arrays)
        self.stdin.write(f"{given} {out}\n".encode())
        self.stdin.flush()
        reply = self.stdout.readline()
        if not reply:
            return None
        if reply != b"done\n":
            raise self._ended("answered a request with something else than done", reply.decode().strip())
        with np.load(out) as answer:
            return dict(answer)


SOURCE = Path(__file__).resolve().parents[1] / "src"
"""This checkout's src, the one place a worker reads `_dew_torch` from (the float64 twin's arithmetic,
which a reference's own environment, holding no Dew, does not install): `session` puts it on every
worker's path, as a reference tool run by hand is given it (`PYTHONPATH=src`)."""


@contextlib.contextmanager
def session(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> Iterator[Session]:
    """`command`, `[python, script, *arguments]`, started as a worker (`serve`): its script run under this
    file (`bootstrap`) as a group its recorder owns (dew._cleanup's `owned_group`), whose lifeline kills it
    if the recorder dies without unwinding, and `ask`ed through the `Session` the block holds. `env` is its
    environment, the recorder's where none is given, but for TMPDIR and this checkout's src ahead on its
    PYTHONPATH (`SOURCE`): the worker's temporary files are the session's, in a directory removed after the
    worker's group is stopped. However the block ends, an error among them, the group is stopped and
    reaped, its pipes closed and the directory removed, each step attempted whichever fail, the block's
    own error the one raised."""
    from dew._cleanup import attempted, owned_group

    python, script, *arguments = command
    root = Path(tempfile.mkdtemp(prefix="reference-"))
    try:
        (root / "io").mkdir()
        (root / "tmp").mkdir()
        held = os.environ if env is None else env
        path = os.pathsep.join(filter(None, (str(SOURCE), held.get("PYTHONPATH"))))
        environment = {**held, "TMPDIR": str(root / "tmp"), "PYTHONPATH": path}
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
