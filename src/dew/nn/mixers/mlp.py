"""Tokenwise MLP blocks for decoders whose layer pattern separates MLP and attention."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING

from flax import linen as nn

from dew.nn.mixer_base import MixerBase, MixerContext, mixers
from dew.nn.moe import SparseMLP

if TYPE_CHECKING:
    from dew.nn.backbones.decoder_block import GatedMLP, Mixture


class _MLP(nn.Module):
    """Share the feed-forward's scope so the mixer boundary adds no parameter nesting."""

    feedforward: GatedMLP | SparseMLP

    def setup(self):
        nn.share_scope(self, self.feedforward)

    def __call__(self, x, decode: bool = False, positions=None, segment_ids=None, kv_store=None,
                 attention_metadata=None):
        del decode, positions, segment_ids, kv_store, attention_metadata
        return self.feedforward(x)


@mixers("mlp")
@dataclasses.dataclass(frozen=True)
class MLPMixer(MixerBase):
    """An MLP in the mixer's slot, with no attention or recurrent state.

    Nemotron-H gives its MLP a pre-norm and residual of its own. Reusing
    `GatedMLP` or `SparseMLP` keeps the projection names and placement.
    `mixture` makes the slot routed, with the same factory a decoder's
    second feed-forward uses.
    """

    intermediate_size: int
    activation: str = "relu2"
    use_bias: bool = False
    mixture: Mixture | None = None

    def __post_init__(self):
        if isinstance(self.mixture, Mapping):
            from dew.nn.backbones.decoder_block import Mixture
            from dew.registry import from_record

            object.__setattr__(self, 'mixture', from_record(Mixture, self.mixture))

    def build(self, ctx: MixerContext):
        # decoder_block imports the mixer registry while its classes are being defined.
        from dew.nn.attention import RMSNorm
        from dew.nn.backbones.decoder_block import GatedMLP

        dense = functools.partial(
            GatedMLP, out_features=ctx.emb_features,
            activation=self.activation, use_bias=self.use_bias, init_std=ctx.init_std,
            output_init_std=ctx.output_init_std, dtype=ctx.dtype, precision=ctx.precision, parent=None)
        feedforward = (dense(hidden_features=self.intermediate_size) if self.mixture is None else
                       self.mixture.build(out_features=ctx.emb_features,
                                          hidden_features=self.intermediate_size,
                                          activation=self.activation, dense=dense,
                                          norm=functools.partial(
                                              RMSNorm, epsilon=ctx.norm_eps, scale_offset=ctx.scale_offset,
                                              scale_after_cast=ctx.scale_after_cast, dtype=ctx.dtype),
                                          init_std=ctx.init_std, output_init_std=ctx.output_init_std,
                                          dtype=ctx.dtype, precision=ctx.precision)(parent=None))
        return functools.partial(_MLP, feedforward=feedforward)
