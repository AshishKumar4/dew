"""The in-context memory mixer kind (`dew.nn.memory`), configured as the memory itself."""

from __future__ import annotations

import dataclasses
import functools

from dew.nn.memory import InContextMemory
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.registry import rebuilt


@dataclasses.dataclass(frozen=True)
class InContextMemoryMixer(MixerBase):
    """A layer kind whose mixer is `memory`, an in-context memory (`InContextMemory`) as a record holds
    it: its network, objective, target, gates, chunk and views, every field it computes by. The
    layer's dtype and matmul precision are the decoder's."""

    keeps_cache = True

    memory: InContextMemory

    def fork_refusal(self) -> str | None:
        """Why the layer reads no forks (`MixerBase.fork_refusal`), None where it does: a memory on the
        shared recurrence (`InContextMemory._recurrence`) continues a fork's document from the state its
        parent's tokens leave, through the forks' table (`dew.nn.linear.forked_scan`, as lightning's), and
        from the share of W_0 they kept; the general path steps each document's weights from its own
        start, its chunks counted per document, which no fork continues. A call's wrappers that take a
        recurrent memory's heads off the recurrence (a quantized or nonlinear product) are refused where
        the call reads forks (`InContextMemory.__call__`)."""
        if self.memory._recurrence() is not None:
            return None
        return ("an in-context memory off its shared recurrence continues no fork's document from its "
                "parent: its general path steps each document's weights from its own start")

    def wrapped_fork_refusal(self) -> str | None:
        """Why the layer reads no forks under a wrapper of the model's products
        (`MixerBase.wrapped_fork_refusal`): a quantized kernel or a rounded input takes the memory's heads
        off its shared recurrence (`InContextMemory._route`), the one path of it that reads them."""
        return ("a wrapper's products (a quantization's) take an in-context memory's heads off its shared "
                "recurrence, the one path of it that reads forks")

    def marks(self) -> tuple[str, ...]:
        """The residuals of every path the memory's configuration can take (`InContextMemory.residuals`):
        the kind cannot see the wrappers a model computes under, which decide whether a linear memory
        runs the recurrence or its general path, so it names both paths' residuals."""
        return self.memory.residuals

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError("memory requires causal=True; its in-context steps read a token's prefix alone")
        return functools.partial(type(self.memory), **{**rebuilt(dict, self.memory), "dtype": ctx.dtype,
                                                       "precision": ctx.precision})
