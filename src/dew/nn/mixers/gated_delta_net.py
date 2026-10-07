"""The gated delta net mixer kind, from the reference's own field names."""

from __future__ import annotations

import dataclasses

from dew.nn.mixer_base import MixerBase, MixerContext


@dataclasses.dataclass(frozen=True)
class GatedDeltaNetMixer(MixerBase):
    """The Qwen3.5 family's linear-attention layer, by the config's field names.

    One key head serves `num_v // num_k` value heads (`repeat_interleave`).
    `output_gate_type` is the gated norm's activation: silu in qwen3_5
    (modeling_qwen3_5.py:173), `output_gate_type or hidden_act` in qwen4_exp
    (modeling_qwen4_exp.py:438). `fused_in_proj` is Qwen3-Next's two fused
    input leaves (modeling_qwen3_next.py:540-586). The chunk size, 64, belongs
    to the implementation. The context's attention geometry is not read: the
    layer has no keys to cache, no rope and no window.
    """

    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = 'silu'
    fused_in_proj: bool = False

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError("gated_delta_net requires causal=True; its recurrence has no bidirectional mode")
        from dew.nn.linear import CHUNK_SIZE, GatedDeltaNet

        return self.factory(
            GatedDeltaNet, ctx, context=("emb_features", "norm_eps", "dtype", "precision"),
            kind=("fused_in_proj",), num_k_heads=self.linear_num_key_heads,
            num_v_heads=self.linear_num_value_heads, head_k_dim=self.linear_key_head_dim,
            head_v_dim=self.linear_value_head_dim, conv_kernel=self.linear_conv_kernel_dim,
            chunk_size=CHUNK_SIZE, gate_activation=self.output_gate_type)
