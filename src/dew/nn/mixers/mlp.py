"""Tokenwise MLP blocks for decoders whose layer pattern separates MLP and attention."""

import dataclasses
import functools

from dew.nn.mixer_base import MixerBase, MixerContext, mixers


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

        return functools.partial(
            GatedMLP, hidden_features=self.intermediate_size, out_features=ctx.emb_features,
            activation=self.activation, use_bias=self.use_bias, init_std=ctx.init_std,
            output_init_std=ctx.output_init_std, dtype=ctx.dtype, precision=ctx.precision)
