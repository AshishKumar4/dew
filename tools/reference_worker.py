"""The protocol a reference's worker answers tools/venv_reference.py by (tools/dreamerv3_worker.py,
tools/maxtext_worker.py), run in the reference's own environment.

A worker reads `IN OUT` pairs of paths on stdin and answers each with a line `done` on stdout (`serve`).
IN holds cases, each `<case>.<input>` and `<case>.param.<path>` per parameter by its path in the
reference's own layout. OUT gets each case's outputs `<case>.reference.<leaf>` and `<case>.truth.<leaf>`,
the reference run in float32 and in float64 (`answered`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager

import numpy as np

type Applied = Callable[[dict[str, object], dict[str, np.ndarray]], Mapping[str, object]]
"""The reference's run of a case: its parameters and inputs to its outputs by leaf."""


def answered(cases: Mapping[str, np.ndarray],
             applying: Callable[[str], AbstractContextManager[Applied]]) -> dict[str, np.ndarray]:
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


def serve(run: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]) -> None:
    """Each line on stdin an `IN OUT` pair of paths, answered by `run` over IN's cases and a line `done`
    on stdout."""
    for line in sys.stdin:
        given, out = line.split()
        with np.load(given) as cases:
            np.savez(out, **run(dict(cases)))
        print("done", flush=True)


__all__ = ["Applied", "answered", "serve"]
