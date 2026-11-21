"""A launch's body as every Mosaic GPU family's kernel builds it (`mosaic.kernel`, `Family.kernel`): run in
Pallas's interpreter or natively (`interpreted`; `launches.interpreting` while its body traces, which every
family's products read), a dependent launch waiting for the one before it where the GPU has grid dependency
control (`waits`, `dependent`), with its L2 prefetches issued before that wait (`prefetched`)."""

from __future__ import annotations

import contextvars
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation, launches
from dew.nn.kernels.templates import rings


def interpreted(interpret) -> plgpu.InterpretGPUParams | None:
    """A launch's interpreter: `interpret`'s own parameters where it gives them (a read past an operand
    poisoned, `grouped_launch.INTERPRETED`), the default where it is True, none where False."""
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
    `entry` as it is.

    With `prefetched`, `prefetched(*refs)`'s views of any operands each prefetched into L2 before the wait
    (`rings.prefetch`): a prefetch reads nothing into the program, and L2 is where every write
    lands, so a view the launch before it still writes is correct to prefetch, at worst early."""
    if not waits(interpret):
        return entry

    def body(*refs):
        for view in () if prefetched is None else prefetched(*refs):
            rings.prefetch(view)
        plgpu.griddepcontrol_wait()
        entry(*refs)
        plgpu.griddepcontrol_launch_dependents()

    return body


def subnormals[T](compute: Callable[..., T], dtype) -> T:
    """`compute(flushing=)`, `flushing` whether the arithmetic it traces in `dtype` flushes a subnormal
    operand and result, so it keeps them itself: fp32's on XLA's CPU (per primitive, row 4023), within a
    launch's body whether the launch runs in Pallas's interpreter (`launches.interpreting`), outside any
    XLA's CPU's program (`jax.lax.platform_dependent`); a wider dtype's never (a float64 truth's). The one
    owner of the choice (row 6302's finish, train 866's and 870's reviews)."""
    if jnp.dtype(dtype) != jnp.dtype(jnp.float32):
        return compute(flushing=False)
    within = launches.interpreting()
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
