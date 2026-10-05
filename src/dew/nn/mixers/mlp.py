"""Tokenwise MLP blocks for decoders whose layer pattern separates MLP and attention."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

from flax import linen as nn

from dew.nn.mixer_base import MixerBase, MixerContext, mixers

if TYPE_CHECKING:
    from dew.nn.backbones.decoder_block import GatedMLP


class _MLP(nn.Module):
    """Share the feed-forward's scope so the mixer boundary adds no parameter nesting."""

    feedforward: GatedMLP

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
    `GatedMLP` keeps its two projection names and tensor-axis placement.
    """

    intermediate_size: int
    activation: str = "relu2"
    use_bias: bool = False

    def build(self, ctx: MixerContext):
        # decoder_block imports the mixer registry while its classes are being defined.
        from dew.nn.backbones.decoder_block import GatedMLP

        feedforward = GatedMLP(
            hidden_features=self.intermediate_size, out_features=ctx.emb_features,
            activation=self.activation, use_bias=self.use_bias, init_std=ctx.init_std,
            output_init_std=ctx.output_init_std, dtype=ctx.dtype, precision=ctx.precision, parent=None)
        return functools.partial(_MLP, feedforward=feedforward)
