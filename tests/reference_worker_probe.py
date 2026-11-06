"""A worker for tests/test_reference_worker.py, run as a reference's worker is (`reference_worker.session`),
in this environment: numpy and, for `applied`, JAX. Its first argument says which worker it is.

`paths` and `frames` serve each request by its `mode` (`probed`), over the carrier they are named for;
`fragments` answers frames in pieces. A
library's output reaches stdout at import, as torch's or tilelang's has, before the worker serves anything.
`applied` is the `answered` run of a product in float32 and float64, tests/reference_probe_site.py's site.
The rest speak the protocol wrongly or stop speaking it, on the first line or the first request."""

import io
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import reference_worker

# What a library prints while it imports, by Python and by a native write: neither is the protocol's.
print("a library's import-time print", flush=True)
os.write(1, b"a native import-time write\nready paths\n")


def log(text: str) -> None:
    """`text` in the file the test names, which says which process answered."""
    with Path(os.environ["REFERENCE_PROBE_LOG"]).open("a") as written:
        written.write(text + "\n")


def probed(request: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The answer to a request, by its `mode`: its arrays as the reference and the truth (`echo`) after
    the library's prints (`noisy`), the worker's and its descendant's processes and its temporary
    directory (`facts`), its stdout closed (`stdout`), or its end, by its own exit, an error or a signal, or
    by closing the pipe it answers by and living on."""
    mode = str(request.pop("mode"))
    if mode == "noisy":
        print("done", flush=True)
        os.write(1, b"done\n")
        os.write(1, struct.pack("<Q", 5) + b"hello")
        sys.stdout.write("a partial line")
        sys.stdout.flush()
    elif mode == "facts":
        descendant = subprocess.Popen(["sleep", "600"])
        created = Path(tempfile.gettempdir()) / "made-by-the-worker"
        created.write_text("a file a library made in its temporary directory")
        return {"worker": np.asarray(os.getpid()), "descendant": np.asarray(descendant.pid),
                "temporary": np.asarray(tempfile.gettempdir()), "created": np.asarray(str(created))}
    elif mode == "stdout":
        sys.stdout.close()
    elif mode == "exit":
        os._exit(3)
    elif mode == "raise":
        raise RuntimeError("the worker's request failed")
    elif mode == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
    elif mode == "close":
        reference_worker.reserved().close()
        time.sleep(600)
    return reference_worker.paired(request, request)


def applied() -> None:
    """`answered` over a product of a case's `x` and its parameter `w`, whose run the first call of each
    dtype builds."""
    import contextlib

    import jax.numpy as jnp

    @contextlib.contextmanager
    def applying(dtype: str):
        def apply(params, case):
            return {"y": jnp.dot(jnp.asarray(case["x"], dtype), params["w"], precision="highest")}

        yield apply

    reference_worker.serve(lambda cases: reference_worker.answered(cases, applying))


def main() -> None:
    kind = sys.argv[1]
    if kind in ("paths", "frames"):
        reference_worker.serve(probed, "probe 1", frames=kind == "frames")
    elif kind == "applied":
        applied()
    elif kind == "stops-at-start":
        os._exit(7)
    elif kind == "silent-at-start":
        time.sleep(600)
    elif kind in ("unknown-carrier", "no-ready"):
        answers = reference_worker.reserved()
        answers.write(b"ready tcp\n" if kind == "unknown-carrier" else b"hello\n")
        answers.flush()
        time.sleep(600)
    elif kind == "wrong-answer":
        answers = reference_worker.reserved()
        answers.write(b"ready paths\n")
        answers.flush()
        sys.stdin.buffer.readline()
        answers.write(b"nothing\n")
        answers.flush()
        time.sleep(600)
    elif kind == "fragments":
        answers = reference_worker.reserved()
        answers.write(b"ready frames\n")
        answers.flush()
        while (request := reference_worker.received(sys.stdin.buffer)) is not None:
            frame = io.BytesIO()
            reference_worker.send(frame, reference_worker.paired(request, request))
            # Each piece reaches the recorder alone, a frame never whole in one read.
            for start in range(0, frame.tell(), 7):
                answers.write(frame.getvalue()[start:start + 7])
                answers.flush()
                time.sleep(0.001)
    elif kind == "cut-frame":
        answers = reference_worker.reserved()
        answers.write(b"ready frames\n")
        answers.flush()
        reference_worker.received(sys.stdin.buffer)
        answers.write(struct.pack("<Q", 100) + b"ten bytes!")
        answers.flush()
        os._exit(0)
    else:
        raise SystemExit(f"no probe is named {kind}")


if __name__ == "__main__":
    if "REFERENCE_PROBE_LOG" in os.environ:
        log(f"{sys.argv[1]} {os.getpid()}")
    main()
