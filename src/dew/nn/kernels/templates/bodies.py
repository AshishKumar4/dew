"""A launch's body as every Mosaic GPU family's kernel builds it (`mosaic.kernel`, `Family.kernel`): run in
Pallas's interpreter or natively (`interpreted`, and `interpreting` while its body traces, which every
family's products read), a dependent launch waiting for the one before it where the GPU has grid dependency
control (`waits`, `dependent`), with its L2 prefetches issued before that wait (`prefetched`)."""

from __future__ import annotations

import contextvars
from collections.abc import Callable

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
        token = _INTERPRETING.set(interpreting_)
        try:
            if waiting:
                for view in () if prefetched is None else prefetched(*refs):
                    plgpu.async_prefetch(view)
                plgpu.griddepcontrol_wait()
            entry(*refs)
            if waiting:
                plgpu.griddepcontrol_launch_dependents()
        finally:
            _INTERPRETING.reset(token)

    return body


_INTERPRETING: contextvars.ContextVar[bool] = contextvars.ContextVar("interpreting", default=False)
"""Whether the launch whose body is tracing runs in Pallas's interpreter (`dependent`, which every launch's
body passes through)."""


def interpreting() -> bool:
    """Whether the launch whose program is tracing runs in Pallas's interpreter: a family's products issue
    no instruction the interpreter lacks there (mma.sync's sums, `mosaic_sm80.Mma.products`), whatever a
    caller states (row 5209's A100 backward: products called with no interpret flag issued mma.sync's)."""
    return _INTERPRETING.get()


RESIDENT: contextvars.ContextVar[int] = contextvars.ContextVar("resident", default=1)
"""The programs a multiprocessor holds at once of the launch being traced (`kernel`'s `resident`): its roles
raise their registers to that residency's share (`roles`)."""

MOVES: contextvars.ContextVar[bool] = contextvars.ContextVar("moves", default=True)
"""Whether the launch being traced moves registers between its roles (`resources.RegisterModel.moves`): its
schedule raises and lowers them (`handoffs.Schedule.run`) where its consumers' need passes the program's
entry share, and leaves every warpgroup at that share otherwise."""
