"""The in-context memory mixer kind (`dew.nn.memory`), by fields a record holds."""

from __future__ import annotations

import dataclasses
import functools
from typing import Literal

from dew.nn.mixer_base import MixerBase, MixerContext


@dataclasses.dataclass(frozen=True)
class InContextMemoryMixer(MixerBase):
    """A layer kind whose mixer is an in-context memory (`InContextMemory`), its configuration flat.

    `network` is the inner network: `linear` (`LinearMemory` at the head width), `residual`
    (`ResidualMemory`: `hidden`, `activation`, `norm`) or `mlp` (`MLPMemory`: `widths`, the last the
    head width where omitted). `objective` with `p` and `delta` is its loss, `target` its target; the
    step size is `learning_rate_scale * sigmoid(.)` from its bias, or `learning_rate` itself where
    given; `momentum` and `retention` add those gates from their biases. `heads` memories of
    `head_dim` (the layer's where None) run side by side, every `chunk` tokens updating each.
    """

    keeps_cache = True

    network: Literal["linear", "residual", "mlp"] = "residual"
    hidden: int | None = None
    widths: tuple[int, ...] = ()
    activation: Literal["gelu", "silu", "relu"] = "gelu"
    norm: bool = True
    objective: Literal["dot", "l2", "lp", "huber"] = "l2"
    p: float = 3.0
    delta: float = 1.0
    target: Literal["given", "self_generated"] = "given"
    learning_rate: float | None = None
    learning_rate_scale: float = 1.0
    learning_rate_bias: float = 0.0
    momentum: bool = False
    momentum_bias: float = 0.0
    retention: bool = False
    retention_bias: float = -2.0
    heads: int | None = None
    head_dim: int | None = None
    chunk: int = 16
    normalize: bool = False

    def build(self, ctx: MixerContext):
        from dew.nn.memory import Gate, InContextMemory, LinearMemory, MLPMemory, Objective, ResidualMemory

        heads = self.heads or ctx.num_heads
        width = self.head_dim or ctx.head_dim
        network = {
            "linear": functools.partial(LinearMemory, features=width),
            "residual": functools.partial(ResidualMemory, hidden=self.hidden, activation=self.activation,
                                          norm=self.norm),
            "mlp": functools.partial(MLPMemory, widths=self.widths if self.widths and self.widths[-1] == width
                                     else (*self.widths, width), activation=self.activation),
        }[self.network]
        rate = (Gate(value=self.learning_rate) if self.learning_rate is not None
                else Gate(scale=self.learning_rate_scale, bias=self.learning_rate_bias))
        return functools.partial(
            InContextMemory, network=network, heads=heads, head_features=width,
            objective=Objective(self.objective, p=self.p, delta=self.delta), target=self.target,
            learning_rate=rate, momentum=Gate(bias=self.momentum_bias) if self.momentum else None,
            retention=Gate(bias=self.retention_bias) if self.retention else None, chunk=self.chunk,
            normalize=self.normalize, dtype=ctx.dtype, precision=ctx.precision)
