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
        """Why the layer reads no forks (`MixerBase.fork_refusal`): the memory's state carries through a
        row's chunks in place order, and a fork's document continues its parent's state, which the row's
        order does not hand it: its parent's tokens sit earlier in the row beside other documents'."""
        return ("an in-context memory carries its state through a row's chunks in place order, which does "
                "not hand a fork's document its parent's state")

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
