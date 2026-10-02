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

On the CPU backend JAX's own guard never sees a device-to-host transfer: a
CPU array reaches numpy as a view, without a copy for the guard to refuse.
`float(x)` under `jax.transfer_guard("disallow")` passes on CPU and raises
on a GPU (measured on jax 0.11.2: float, int, np.asarray and item all pass
on CPU). So for the block's duration `guarded` holds CPU arrays to the same
rule, at the one place every implicit read goes through, `jax.Array`'s host
value, reading the guard level the calling thread set. A test that passes
on CPU then also passes the native guard on a GPU.
"""

import contextlib
import traceback
from collections.abc import Iterator

import jax
from jax._src import array as jax_array, dispatch
from jax._src.lib import guard_lib

_COMPILE_EVENTS = {dispatch.JAXPR_TRACE_EVENT: "traced", dispatch.BACKEND_COMPILE_EVENT: "compiled"}
_REFUSED = {guard_lib.TransferGuardLevel.DISALLOW, guard_lib.TransferGuardLevel.DISALLOW_EXPLICIT}
_host_value = jax_array.ArrayImpl._value
_guarding = 0


def _guarded_value(array):
    """`jax.Array._value`, refusing an implicit read where the calling
    thread's device-to-host guard refuses one, as a GPU's would."""
    state = guard_lib.thread_local_state()
    level = state.device_to_host or guard_lib.global_state().device_to_host
    explicit = state.explicit_device_get
    if (array._npy_value is None and level in _REFUSED
            and (not explicit or level == guard_lib.TransferGuardLevel.DISALLOW_EXPLICIT)):
        raise AssertionError(
            f"Disallowed device-to-host transfer: aval={array.aval}; read it with "
            f"jax.device_get at the cadence that needs it")
    return _host_value.fget(array)


@contextlib.contextmanager
def guarded() -> Iterator[None]:
    """Refuse every implicit host<->device transfer in the block, on any
    backend: `jax.transfer_guard("disallow")`, with CPU arrays' reads to
    the host held to it too."""
    global _guarding
    if _guarding == 0:
        jax_array.ArrayImpl._value = property(_guarded_value)
    _guarding += 1
    try:
        with jax.transfer_guard("disallow"):
            yield
    finally:
        _guarding -= 1
        if _guarding == 0:
            jax_array.ArrayImpl._value = _host_value


@contextlib.contextmanager
def steady_state(compiles: int = 0) -> Iterator[None]:
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
        with guarded():
            yield
    finally:
        jax.monitoring.unregister_event_duration_listener(listen)
    compiled = [entry for entry in seen if entry[0] == "compiled"]
    unexpected = seen if compiles == 0 else compiled
    assert len(unexpected) == compiles, (
        f"the steady state {'traced or ' if compiles == 0 else ''}compiled {len(unexpected)} "
        f"program(s), {compiles} expected:\n" + "\n".join(
            f"  {kind} {name} at {where}" for kind, name, where in seen))
