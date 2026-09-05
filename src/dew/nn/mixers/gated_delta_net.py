"""The gated delta net mixer kind, from the reference's own field names."""

from __future__ import annotations

import dataclasses
import functools

from dew.nn.mixers import MixerBase, MixerContext, mixers


@mixers("gated_delta_net")
@dataclasses.dataclass(frozen=True)
class GatedDeltaNetMixer(MixerBase):
    """The Qwen3.5 family's linear-attention layer, by the config's field names.

    `linear_num_key_heads`/`linear_num_value_heads` with their head dims are
    the mixer's own geometry: value heads may outnumber key heads, and one
    key head serves `num_v // num_k` value heads, which is the reference's
    `repeat_interleave`. `linear_conv_kernel_dim` is the depthwise short
    conv's window. `output_gate_type` is the activation the gated norm
    applies to its gate, silu in qwen3_5 (`Qwen3_5RMSNormGated.activation`,
    modeling_qwen3_5.py:173) and `output_gate_type or hidden_act` in
    qwen4_exp (modeling_qwen4_exp.py:438). The chunked/recurrent trade
    (chunk size 64, the reference's default) is a property of the
    implementation, not the config, so it is not a field here.

    This kind ignores the context's attention geometry (num_kv_heads,
    head_dim, the window, partial rotary, KV sharing, the output gate): a
    linear-attention layer has no keys to cache, no rope and no window,
    which is the whole point of the family.
    """

    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = 'silu'

    def build(self, ctx: MixerContext):
        from dew.nn.linear import GatedDeltaNet, CHUNK_SIZE

        return functools.partial(
            GatedDeltaNet,
            emb_features=ctx.emb_features,
            num_k_heads=self.linear_num_key_heads,
            num_v_heads=self.linear_num_value_heads,
            head_k_dim=self.linear_key_head_dim,
            head_v_dim=self.linear_value_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
            max_seq_len=ctx.max_seq_len,
            chunk_size=CHUNK_SIZE,
            norm_eps=ctx.norm_eps,
            gate_activation=self.output_gate_type,
            dtype=ctx.dtype,
            precision=ctx.precision)
