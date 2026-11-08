"""A launch's body as every Mosaic GPU family's kernel builds it (`mosaic.kernel`, `Family.kernel`): run in
Pallas's interpreter or natively (`interpreted`, and `interpreting` while its body traces, which every
family's products read), a dependent launch waiting for the one before it where the GPU has grid dependency
control (`waits`, `dependent`), with what it reads before that wait: its settled operands (`ahead`,
`SETTLED`, which the compile point checks) and its L2 prefetches (`prefetched`)."""

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


def dependent(entry: Callable, interpret, *, ahead: Callable | None = None, settled: tuple[int, ...] = (),
              scratch: int = 0, prefetched: Callable | None = None) -> Callable:
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

    With `ahead`, a program first runs it, once, on the refs of its `settled` operands (indices among the
    launch's operands) and its `scratch` refs (the body's last), before its wait, and then `entry` on every
    ref and what `ahead` returned, where it returned anything: the reads of operands no launch before it
    writes (a decode step's weights, loaded once a run) issued under the previous launch's tail, which its
    other operands' and its outputs' refs, not given to `ahead`, cannot be. The launch declares `settled` on
    its call (`SETTLED`), and the compile point refuses a program where one is not an input it holds whole
    (`lowered.unsettled`).

    With `prefetched`, `prefetched(*refs)`'s views of any operands each prefetched into L2 before the wait
    (`plgpu.async_prefetch`): a prefetch reads nothing into the program, and L2 is where every write
    lands, so a view the launch before it still writes is correct to prefetch, at worst early."""
    interpreting_ = bool(interpreted(interpret))
    waiting = waits(interpret)
    if ahead is None and not waiting and not interpreting_:
        return entry

    def body(*refs):
        token = _INTERPRETING.set(interpreting_)
        try:
            own = refs[len(refs) - scratch:]
            held = None if ahead is None else ahead(*(refs[index] for index in settled), *own)
            if waiting:
                for view in () if prefetched is None else prefetched(*refs):
                    plgpu.async_prefetch(view)
                plgpu.griddepcontrol_wait()
            if held is None:
                entry(*refs)
            else:
                entry(*refs, held)
            if waiting:
                plgpu.griddepcontrol_launch_dependents()
        finally:
            _INTERPRETING.reset(token)

    return body


SETTLED = "dew_settled"
"""The frontend attribute a launch's call declares the operands its programs read ahead of their wait under
(`dependent`'s `settled`), their indices joined by commas: each must be an input of the program that no
operation of it produces (`lowered.unsettled`)."""


_INTERPRETING: contextvars.ContextVar[bool] = contextvars.ContextVar("interpreting", default=False)
"""Whether the launch whose body is tracing runs in Pallas's interpreter (`dependent`, which every launch's
body passes through)."""


def interpreting() -> bool:
    """Whether the launch whose program is tracing runs in Pallas's interpreter: a family's products issue
    no instruction the interpreter lacks there (mma.sync's sums, `mosaic_sm80.Mma.products`), whatever a
    caller states (row 5209's A100 backward: products called with no interpret flag issued mma.sync's)."""
    return _INTERPRETING.get()
