"""Tokenwise MLP blocks for decoders whose layer pattern separates MLP and attention."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, ClassVar

from flax import linen as nn

from dew.nn.flops import counted_flops, parameter_flops
from dew.nn.inputs import PLAIN_CALL, TOKENWISE, MixerCall, routing_blocks
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.protocols import ProjectionGroup, Routes, RoutesExperts, declared_groups

if TYPE_CHECKING:
    from dew.nn.backbones.decoder_block import Mixture


class _MLP(nn.Module):
    """Share the feed-forward's scope so the mixer boundary adds no parameter nesting."""

    reads: ClassVar[frozenset[str]] = TOKENWISE
    """Tokenwise: no position, document, cache or mask changes what an MLP computes."""

    feedforward: nn.Module

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """The same MLP work when it occupies a mixer's slot."""
        return counted_flops(self.feedforward, lengths)

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """Its feed-forward's learned work, as in the feed-forward's own slot."""
        return parameter_flops(self.feedforward, lengths)

    def setup(self):
        nn.share_scope(self, self.feedforward)

    @property
    def routes_experts(self) -> bool:
        """Whether its feed-forward routes over experts, and so takes the
        layer's replayed ones (`RoutesExperts`)."""
        return isinstance(self.feedforward, RoutesExperts) and self.feedforward.routes_experts

    def __call__(self, x, call: MixerCall = PLAIN_CALL, routes: Routes | None = None):
        if isinstance(self.feedforward, RoutesExperts) and self.feedforward.routes_experts:
            blocks = routing_blocks(self.feedforward, call.metadata, x.shape)
            return self.feedforward(x, routes=routes, **blocks)
        return self.feedforward(x)

    def projection_groups(self) -> tuple[ProjectionGroup, ...]:
        """The packed groups its feed-forward declares (`ProjectionSites`), in the scope they share."""
        return declared_groups(self.feedforward)


@dataclasses.dataclass(frozen=True)
class MLPMixer(MixerBase):
    """An MLP in the mixer's slot, with no attention or recurrent state.

    Nemotron-H gives its MLP a pre-norm and residual of its own. Reusing
    `GatedMLP` or `SparseMLP` keeps the projection names and placement.
    `mixture` makes the slot routed, with the same factory a decoder's
    second feed-forward uses.
    """

    residuals = ('gate_proj', 'up_proj', 'down_proj')

    # Tokenwise, with no cache for a mixed call to misread or a loop's pass to rewrite.
    keeps_cache = False
    mixed_step = True

    intermediate_size: int
    activation: str = "relu2"
    use_bias: bool = False
    mixture: Mixture | None = None

    def __post_init__(self):
        from dew.nn.backbones.decoder_block import Mixture
        from dew.registry import from_record

        if isinstance(self.mixture, Mapping):
            object.__setattr__(self, 'mixture', from_record(Mixture, self.mixture))
        mixture = self.mixture
        if mixture is not None and not isinstance(mixture, Mixture):
            raise ValueError('MLPMixer mixture is a Mixture, its record, or None')
        if mixture is not None and (mixture.layers is not None or mixture.hash_layers is not None
                                    or mixture.media_bias or mixture.beside_dense):
            raise ValueError('MLPMixer uses the layer-kind pattern; mixture layers, hash_layers, '
                             'media_bias and a branch beside a dense feed-forward need a decoder '
                             'feed-forward slot')

    def generation_refusal(self) -> str | None:
        """Its mixture's (`Mixture.generation_refusal`), where the slot is routed."""
        return None if self.mixture is None else self.mixture.generation_refusal()

    def build(self, ctx: MixerContext):
        return functools.partial(_MLP, feedforward=self.feedforward(ctx)(parent=None))

    def feedforward(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        """The factory of the MLP this slot holds, dense or routed, which a
        decoder block's feed-forward slot takes as it is (a prediction
        depth's, `CausalTransformer.mtp_feedforward_type`)."""
        # decoder_block imports the mixer registry while its classes are being defined.
        from dew.nn.mlp import GatedMLP

        dense = functools.partial(
            GatedMLP, out_features=ctx.emb_features,
            activation=self.activation, use_bias=self.use_bias, ngpt=ctx.ngpt, init_std=ctx.init_std,
            output_init_std=ctx.output_init_std, dtype=ctx.dtype, precision=ctx.precision)
        if self.mixture is None:
            return functools.partial(dense, hidden_features=self.intermediate_size)
        return self.mixture.build(out_features=ctx.emb_features, hidden_features=self.intermediate_size,
                                  activation=self.activation, dense=functools.partial(dense, parent=None),
                                  norm_eps=ctx.norm_eps, scale_offset=ctx.scale_offset,
                                  scale_after_cast=ctx.scale_after_cast, init_std=ctx.init_std,
                                  output_init_std=ctx.output_init_std, dtype=ctx.dtype,
                                  precision=ctx.precision)
