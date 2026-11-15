"""A launch's body as every Mosaic GPU family's kernel builds it (`mosaic.kernel`, `Family.kernel`): run in
Pallas's interpreter or natively (`interpreted`, and `interpreting` while its body traces, which every
family's products read), a dependent launch waiting for the one before it where the GPU has grid dependency
control (`waits`, `dependent`), with its L2 prefetches issued before that wait (`prefetched`)."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator

import jax
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation


def interpreted(interpret) -> plgpu.InterpretGPUParams | None:
    """A launch's interpreter: `interpret`'s own parameters where it gives them (a read past an operand
    poisoned, `grouped_matmul.INTERPRETED`), the default where it is True, none where False."""
    if isinstance(interpret, plgpu.InterpretGPUParams):
        return interpret
    return plgpu.InterpretGPUParams() if interpret else None


def waits(interpret) -> bool:
    """Whether a launch runs as a dependent launch, waiting for the one before it (`dependent`): natively, on
    a GPU with grid dependency control (`generation.Architecture.dependent_launch`, sm_90 on)."""
    return not interpreted(interpret) and generation.gpu_target().architecture.dependent_launch


def dependent(entry: Callable, interpret, *, prefetched: Callable | None = None) -> Callable:
    """`entry` as a dependent launch (programmatic dependent launch, PTX's `griddepcontrol`): it waits for the
    launch before it in its stream to finish and its writes to land (`wait`) before any memory access of its
    own, so it is ordered against any launch as without it, and once its own work is done it lets the next
    such launch start (`launch_dependents`), whose launch and setup then overlap this one's last programs'
    tails. Triggered at the work's end, not its start: a trigger at entry let a dependent's programs take the
    multiprocessors this launch's later waves needed, each waiting there on this launch to finish (an H100
    SXM's recurrence launches, wall time per call: Qwen3.5-27B's GDN forward and Nemotron-H 30B's SSD
    training 1.0 and 1.1% above no dependent launch at entry, within 0.5% of it at the end, pdl-sxm at
    b2b872c91; CUTLASS triggers after its mainloop for the same reason). Mosaic marks the launch
    (`mosaic_gpu.uses_pdl`) and the runtime launches it with programmatic stream serialization. Interpreted,
    or on a GPU without grid dependency control (`generation.Architecture.dependent_launch`, sm_90 on),
    `entry` as it is. Its body traces knowing whether it runs interpreted (`interpreting`), as every family's
    products read it.

    With `prefetched`, `prefetched(*refs)`'s views of any operands each prefetched into L2 before the wait
    (`plgpu.async_prefetch`): a prefetch reads nothing into the program, and L2 is where every write
    lands, so a view the launch before it still writes is correct to prefetch, at worst early."""
    interpreting_ = bool(interpreted(interpret))
    waiting = waits(interpret)
    if not waiting and not interpreting_:
        return entry

    def body(*refs):
        with launched(interpreting_):
            if waiting:
                for view in () if prefetched is None else prefetched(*refs):
                    plgpu.async_prefetch(view)
                plgpu.griddepcontrol_wait()
            entry(*refs)
            if waiting:
                plgpu.griddepcontrol_launch_dependents()

    return body


_INTERPRETING: contextvars.ContextVar[bool | None] = contextvars.ContextVar("interpreting", default=None)
"""Whether the launch whose body is tracing runs in Pallas's interpreter (`launched`, which every launch's
body passes through, `dew.nn.kernels.launches`; `dependent` too); None outside any launch's body."""


def interpreting() -> bool:
    """Whether the launch whose program is tracing runs in Pallas's interpreter: a family's products issue
    no instruction the interpreter lacks there (mma.sync's sums, `mosaic_sm80.Mma.products`), whatever a
    caller states (row 5209's A100 backward: products called with no interpret flag issued mma.sync's)."""
    return bool(_INTERPRETING.get())


@contextlib.contextmanager
def launched(interpreting_: bool) -> Iterator[None]:
    """Within the block, a launch's body traces (`dew.nn.kernels.launches`), run in Pallas's interpreter or
    not (`interpreting`, `subnormals`)."""
    token = _INTERPRETING.set(interpreting_)
    try:
        yield
    finally:
        _INTERPRETING.reset(token)


def subnormals[T](compute: Callable[..., T]) -> T:
    """`compute(flushing=)`, `flushing` whether the arithmetic it traces flushes a subnormal operand and
    result (XLA's CPU, per primitive, row 4023), so it keeps them itself: within a launch's body, whether
    the launch runs in Pallas's interpreter (on XLA's CPU); outside any, XLA's CPU's program
    (`jax.lax.platform_dependent`): the one owner of the choice (row 6302's finish, train 866's review)."""
    within = _INTERPRETING.get()
    if within is not None:
        return compute(flushing=within)
    return jax.lax.platform_dependent(cpu=lambda: compute(flushing=True),
                                      default=lambda: compute(flushing=False))


RESIDENT: contextvars.ContextVar[int] = contextvars.ContextVar("resident", default=1)
"""The programs a multiprocessor holds at once of the launch being traced (`kernel`'s `resident`): its roles
raise their registers to that residency's share (`roles`)."""

MOVES: contextvars.ContextVar[bool] = contextvars.ContextVar("moves", default=True)
"""Whether the launch being traced moves registers between its roles (`resources.RegisterModel.moves`): its
schedule raises and lowers them (`handoffs.Schedule.run`) where its consumers' need passes the program's
entry share, and leaves every warpgroup at that share otherwise."""
