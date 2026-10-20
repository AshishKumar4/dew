"""A hot path's steady state: nothing compiles, nothing crosses unasked.

Once a loop has compiled its programs, every further iteration over inputs
of the same shapes should reuse them, and the only data that moves between
host and device is the data a caller asked for by name: a batch placed with
`jax.device_put`, a logged value read with `jax.device_get`. A float() of a
loss, a numpy array handed to a jitted function, or a fresh jnp.zeros in the
loop is an implicit transfer, and a device-to-host one also waits on every
step dispatched before it, so the host stops running ahead of the device.

`steady_state` checks both over a block. It counts the programs JAX traces
and compiles inside it (jax.monitoring's `/jax/core/compile` events; a
persistent-cache hit still counts as a compile), and runs it under
`jax.transfer_guard("disallow")`, which refuses implicit transfers and lets
explicit ones through.

JAX's guard refuses an implicit device-to-host read only where the read
copies, on a GPU. On the CPU backend an array reaches numpy as a view, with
no copy for the guard to refuse, so `float(x)` under the guard passes there
(jax 0.11.2: float, int, np.asarray and item all pass on CPU; jax-ml/jax#16602).
A block's reads are therefore held on a GPU, where the suite's GPU lane runs
these tests; on a CPU its host-to-device and device-to-device transfers are.
Holding CPU reads too belongs in JAX's guard (ledger 801), not in a second
guard over JAX's private array class.

A checkpoint's save passes the guard: Orbax moves a GPU array to pinned host
memory with an explicit `jax.device_put` and reads that (a save under
`jax.transfer_guard_device_to_host("disallow")` passes on an RTX 4080).
"""

import contextlib
import traceback
from collections.abc import Iterator, Sequence

import jax

_COMPILE_EVENTS = {"/jax/core/compile/jaxpr_trace_duration": "traced",
                   "/jax/core/compile/backend_compile_duration": "compiled"}
"""The compile events `jax.monitoring` records a duration for, by what each counts."""


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
    seen: list[tuple[str, str, str]] = []

    def listen(event: str, seconds: float, **kwargs) -> None:
        if event in _COMPILE_EVENTS:
            frame = next((frame for frame in reversed(traceback.extract_stack()[:-1])
                          if "/jax/" not in frame.filename and "/jaxlib/" not in frame.filename
                          and frame.filename != __file__ and "contextlib" not in frame.filename), None)
            where = "?" if frame is None else f"{frame.filename}:{frame.lineno}"
            seen.append((_COMPILE_EVENTS[event], str(kwargs.get("fun_name")), where))

    jax.monitoring.register_event_duration_secs_listener(listen)
    try:
        with guarded(allow):
            yield
    finally:
        jax.monitoring.unregister_event_duration_listener(listen)
    compiled = [entry for entry in seen if entry[0] == "compiled"]
    unexpected = seen if compiles == 0 else compiled
    assert len(unexpected) == compiles, (
        f"the steady state {'traced or ' if compiles == 0 else ''}compiled {len(unexpected)} "
        f"program(s), {compiles} expected:\n" + "\n".join(
            f"  {kind} {name} at {where}" for kind, name, where in seen))
