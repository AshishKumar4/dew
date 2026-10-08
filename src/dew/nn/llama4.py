"""Llama 4's text attention: iRoPE, chunked local layers, temperature tuning.

`Llama4TextAttention` (modeling_llama4.py) is the shared attention under
four of the reference's config fields. Local layers rotate adjacent pairs,
L2-normalise queries and keys with no scale (`use_qk_norm`) and attend
inside chunks of `attention_chunk_size`; global layers carry no positions at
all (`no_rope_layers`) and instead scale each query by a logarithm of its
position (`attn_temperature_tuning`, arXiv 2501.19399). The routed experts
scale each token's input by its routing weight, which `dew.nn.moe.ExpertMLP`
does under `scale_inputs`.
"""

import dataclasses
import functools
from collections.abc import Callable

from flax import linen as nn

from dew.nn.kv_cache import KVCache
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.mixers.attention import CausalSelfAttention


@dataclasses.dataclass(frozen=True)
class Llama4Mixer(MixerBase):
    """The `llama4` kind: `Llama4TextAttention` under the reference's names.

    `use_rope` is the layer's `no_rope_layers` entry, so a config names the
    local kind with the chunk (`LayerKind.chunk`, the config's
    `attention_chunk_size`) and the global kind without rope; the other
    fields are the config's. The head geometry, rope base and its llama3
    ramp, bias, norm epsilon, chunk and kernel choice come from the backbone
    through the context.
    """

    use_rope: bool = True
    use_qk_norm: bool = True
    attn_temperature_tuning: bool = True
    floor_scale: float = 8192.0
    attn_scale: float = 0.1

    def build(self, ctx: MixerContext) -> Callable[..., nn.Module]:
        unsupported = {
            "qk_norm": ctx.qk_norm,
            "v_norm": ctx.v_norm,
            "k_eq_v": ctx.k_eq_v,
            "kv_shared": ctx.kv_shared,
            "sliding_window": ctx.sliding_window,
            "attention_scale": ctx.attention_scale,
            "attention_sinks": ctx.attention_sinks,
            "yarn": ctx.yarn,
            "partial_rotary_factor": ctx.partial_rotary_factor,
            "output_gate": ctx.output_gate,
            "scale_offset": ctx.scale_offset,
            "kv_cache": ctx.kv_cache != KVCache(),
        }
        asked = sorted(name for name, value in unsupported.items() if value)
        if asked:
            raise ValueError(
                f"the llama4 mixer has no {', '.join(asked)}: it norms queries "
                "and keys with its own scale-free L2 norm under use_qk_norm, "
                "scales by 1/sqrt(head_dim), rotates whole interleaved pairs "
                "and attends by chunk rather than by window")
        if ctx.attention_chunk is not None and ctx.attention_chunk < 1:
            raise ValueError(f"attention_chunk_size is a positive chunk length, got {ctx.attention_chunk}; "
                             "None attends the whole sequence")
        if ctx.attention_chunk is not None and not self.use_rope:
            raise ValueError("Llama 4 chunks its rotated local layers only; a global layer without rope "
                             "attends the whole sequence")
        return functools.partial(
            CausalSelfAttention,
            emb_features=ctx.emb_features,
            num_heads=ctx.num_heads,
            num_kv_heads=ctx.num_kv_heads,
            head_dim=ctx.head_dim,
            max_seq_len=ctx.max_seq_len,
            causal=ctx.causal,
            rope_theta=ctx.rope_theta,
            rope_scaling=ctx.rope_scaling,
            nope=not self.use_rope,
            rotary_pairs='adjacent',
            qk_norm=self.use_rope and self.use_qk_norm,
            qk_norm_weight=False,
            attention_chunk=ctx.attention_chunk,
            temperature_tuning=(None if self.use_rope or not self.attn_temperature_tuning
                                else (self.floor_scale, self.attn_scale)),
            norm_eps=ctx.norm_eps,
            attention_bias=ctx.attention_bias,
            dtype=ctx.dtype,
            precision=ctx.precision,
            attention_impl=ctx.attention_impl,
            force_fp32_for_softmax=ctx.force_fp32_for_softmax)


def default_no_rope_layers(num_layers: int, interval: int) -> tuple[int, ...]:
    """`Llama4TextConfig`'s pattern: every `interval`th layer carries no rope."""
    return tuple(int((index + 1) % interval != 0) for index in range(num_layers))


def rope_layer_types(no_rope_layers) -> tuple[str, ...]:
    """The config's layer pattern: rotated layers chunk, the rest attend whole."""
    return tuple('chunked_attention' if rope else 'full_attention' for rope in no_rope_layers)


