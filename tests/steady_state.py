"""A hot path's steady state: nothing compiles, nothing crosses unasked.

Once a loop has compiled its programs, every further iteration over inputs
of the same shapes should reuse them, and the only data that moves between
host and device is the data a caller asked for by name: a batch placed with
`jax.device_put`, a logged value read with `jax.device_get`. A float() of a
loss, a numpy array handed to a jitted function, or a fresh jnp.zeros in the
loop is an implicit transfer, and a device-to-host one also waits on every
step dispatched before it, so the host stops running ahead of the device.

`steady_state` checks both over a block. It counts the programs JAX traces
and compiles inside it (`dew.telemetry.instrumentation.Compiles`, over
jax.monitoring's `/jax/core/compile` events; a persistent-cache hit still
counts as a compile), and runs it under
`jax.transfer_guard("disallow")`, which refuses implicit transfers and lets
explicit ones through.

JAX's guard refuses an implicit device-to-host read where the read copies,
on a GPU; on the CPU backend an array reaches numpy as a view, with no copy
for upstream's guard to refuse (jax-ml/jax#16602). Dew's JAX build holds a
CPU array's read to the guard too (constraints.txt: its host value, its copy
to host and its buffer, as an accelerator's copy is held), so a block's
`float(x)` fails on the CPU suite as it does on a GPU.

A checkpoint's save passes the guard: Orbax moves a GPU array to pinned host
memory with an explicit `jax.device_put` and reads that (a save under
`jax.transfer_guard_device_to_host("disallow")` passes on an RTX 4080).
"""

import contextlib
from collections.abc import Iterator, Sequence

import jax

from dew.telemetry.instrumentation import Compiles


@contextlib.contextmanager
def guarded(allow: Sequence[str] = ()) -> Iterator[None]:
    """Refuse every implicit host<->device transfer in the block:
    `jax.transfer_guard("disallow")`, with the directions `allow` names, of
    "host_to_device", "device_to_device" and "device_to_host", let through: a
    request path whose own inputs are the point of the transfer allows the
    first and is held to reads."""
    with contextlib.ExitStack() as levels:
        levels.enter_context(jax.transfer_guard("disallow"))
        for direction in allow:
            levels.enter_context(getattr(jax, f"transfer_guard_{direction}")("allow"))
        yield


@contextlib.contextmanager
def steady_state(compiles: int = 0, allow: Sequence[str] = ()) -> Iterator[None]:
    """Run the block as a hot path's steady state: `guarded`, and tracing
    and compiling no program, or exactly `compiles` where the block brings a
    new shape bucket. Fails naming each program and where it was asked for."""
    with Compiles(None) as heard, guarded(allow):
        yield
    seen = heard.programs
    unexpected = seen if compiles == 0 else [program for program in seen if program.kind == "compiled"]
    assert len(unexpected) == compiles, (
        f"the steady state {'traced or ' if compiles == 0 else ''}compiled {len(unexpected)} "
        f"program(s), {compiles} expected:\n" + "\n".join(
            f"  {program.kind} {program.name} at {program.where}" for program in seen))
