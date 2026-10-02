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
rule, reading the guard level the calling thread set, at the two places an
implicit read goes through: `jax.Array`'s host value, which float(), int(),
item() and `__array__` read, and its buffer, which np.asarray takes first
and, refused, falls back from to `__array__`. A test that passes on CPU
then also passes the native guard on a GPU.

The one read let through is a checkpoint's. Orbax moves a GPU array to
pinned host memory with an explicit `jax.device_put` and reads that with
np.asarray, which the native guard allows (a save under
`jax.transfer_guard_device_to_host("disallow")` passes on an RTX 4080); a
CPU device has no pinned memory, and the same np.asarray reads the array
itself (`transfer_arrays_to_host` in orbax's replica_slices.py).
"""

import contextlib
import sys
import traceback
from collections.abc import Iterator, Sequence

import jax
from jax._src import array as jax_array, dispatch
from jax._src.lib import guard_lib

_COMPILE_EVENTS = {dispatch.JAXPR_TRACE_EVENT: "traced", dispatch.BACKEND_COMPILE_EVENT: "compiled"}
_REFUSED = {guard_lib.TransferGuardLevel.DISALLOW, guard_lib.TransferGuardLevel.DISALLOW_EXPLICIT}
_CHECKPOINT_COPY = "orbax/checkpoint/_src/serialization/replica_slices.py"
_host_value = jax_array.ArrayImpl._value
_buffer = jax_array.ArrayImpl.__buffer__
_guarding = 0


def _refused(array) -> bool:
    """Whether the calling thread's device-to-host guard refuses reading
    `array` here, as a GPU's would."""
    state = guard_lib.thread_local_state()
    level = state.device_to_host or guard_lib.global_state().device_to_host
    explicit = state.explicit_device_get
    return (array._npy_value is None and level in _REFUSED
            and (not explicit or level == guard_lib.TransferGuardLevel.DISALLOW_EXPLICIT)
            and not _checkpoint_copy())


def _checkpoint_copy() -> bool:
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename.endswith(_CHECKPOINT_COPY):
            return True
        frame = frame.f_back
    return False


def _guarded_value(array):
    if _refused(array):
        raise AssertionError(
            f"Disallowed device-to-host transfer: aval={array.aval}; read it with "
            f"jax.device_get at the cadence that needs it")
    return _host_value.fget(array)


def _guarded_buffer(array, flags):
    if _refused(array):
        raise BufferError("a guarded read goes through jax.Array._value")
    return _buffer(array, flags)


@contextlib.contextmanager
def guarded(allow: Sequence[str] = ()) -> Iterator[None]:
    """Refuse every implicit host<->device transfer in the block, on any
    backend: `jax.transfer_guard("disallow")`, with CPU arrays' reads to
    the host held to it too. `allow` names the directions, of
    "host_to_device" and "device_to_device", a block moves by design: a
    request path whose own inputs are the point of the transfer allows the
    first and is held to reads."""
    global _guarding
    if _guarding == 0:
        jax_array.ArrayImpl._value = property(_guarded_value)
        jax_array.ArrayImpl.__buffer__ = _guarded_buffer
    _guarding += 1
    try:
        with contextlib.ExitStack() as levels:
            levels.enter_context(jax.transfer_guard("disallow"))
            for direction in allow:
                levels.enter_context(getattr(jax, f"transfer_guard_{direction}")("allow"))
            yield
    finally:
        _guarding -= 1
        if _guarding == 0:
            jax_array.ArrayImpl._value = _host_value
            jax_array.ArrayImpl.__buffer__ = _buffer


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
