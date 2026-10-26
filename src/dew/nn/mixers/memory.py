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

    def marks(self) -> tuple[str, ...]:
        """The residuals the memory's path marks (`InContextMemory.residuals`)."""
        return self.memory.residuals

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError("memory requires causal=True; its in-context steps read a token's prefix alone")
        return functools.partial(type(self.memory), **{**rebuilt(dict, self.memory), "dtype": ctx.dtype,
                                                       "precision": ctx.precision})
