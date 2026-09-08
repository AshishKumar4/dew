#!/usr/bin/env python3
"""Write the Hugging Face fixtures tests/test_hf_decoders.py checks against.

Everything here runs under torch and transformers, which dew does not depend
on, so this is the only place where the reference implementation is executed.
The fixtures it writes are what CI compares against.

Set up the venv and run it:

    uv venv /tmp/hfref --python 3.12
    uv pip install --python /tmp/hfref/bin/python torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /tmp/hfref/bin/python transformers safetensors \
        sentencepiece numpy
    /tmp/hfref/bin/python tools/hf_reference.py

What lands in tests/fixtures/hf:
- <family>-tiny/ for qwen3, gemma, gemma2, gemma3, llama, llama31, mistral,
  mixtral, qwen2, qwen3-moe, olmo3, olmo3-yarn, deepseek-v3, deepseek-v32,
  llada and dream: a random-weight checkpoint in the HF layout (config.json +
  model.safetensors), the 2 x 12 token ids it was run on, and the fp32
  logits of the reference model in eval mode with eager attention. Small
  enough to live in git. Each tiny config turns on what its family adds
  (its docstring says which dial and why the size was chosen). The DeepSeek
  pair is one dense layer over one MoE layer (`first_k_dense_replace` 1)
  with a shared expert, the released YaRN spelling, q and kv LoRA, and, on
  V3.2, the sparse indexer; their routers' balancing bias is scattered too,
  since a checkpoint carries it and a fixture at its zeros would not tell a
  load that reads it from one that drops it.
  olmo3-yarn carries the released 7B rope: its rope_scaling record on the
  full-attention layers alone and plain rope on the sliding ones, so the
  fixture is the per-kind frequency table rather than one model-wide ramp.
  The llada and dream tinies carry no transformers class (both ship remote
  code), so their logits come from the small torch port beside them, which
  follows the released block line for line and runs at fp32.
  siglip-tiny/ and llama4-vision-tiny/ hold a tiny trunk with its projector:
  model.safetensors and projector.safetensors under the reference tensor
  names, config.json with the tower config, projector.json with the knobs the
  tower config leaves out, fixed pixels, and the fp32 trunk and projector
  outputs. The SigLIP tower is two layers of width 32 over a 2x2 grid pooled
  to one soft token of width 16; the Llama 4 trunk is two layers of width 32
  shuffled to one token of width 64 and mapped to width 32.
  gemma4-vision-tiny/ holds the Gemma 4 trunk with its embedder: the same
  file layout, patches and (x, y) position ids as the processor emits them
  for a 4x4 grid, pooled to four soft tokens of width 32. qwen35-vision-tiny/
  holds the Qwen 3.5 trunk with its merger split off under the released
  merger.* names: the same layout, a 32-pixel square image patchifying into
  a 4x4 grid over the 8x8 learned position table, merged to four soft tokens
  of width 64.
  gemma3-tiny-mm/, llama4-tiny-mm/, gemma4-tiny-mm/ and qwen35-tiny-mm/ hold
  a wrapper fixture each: the vision and projector halves under the released
  model.* nesting beside a tiny decoder half, the wrapper config, fixed
  pixels with one image mark per soft token in the input ids, and the fp32
  wrapper logits. The Gemma 4 pixels are patches with positions; the Qwen
  wrapper reference runs on pre-merged embeddings, since its image-grid
  positions are outside what Dew models.
- qwen3-0.6b/: no weights. tensors.json is the tensor table of the real
  checkpoint straight from the hub metadata API, so a test can check the
  parameter tree without downloading 1.5 GB. prompt.json holds a 48 token
  prompt and reference.npz the top 32 logits per position of the real
  weights in fp32, which the network test compares against.
- One directory per released config the translation is tested on
  (gemma3-1b, gemma-2b, gemma-2-2b, mistral-7b-v0.3, mixtral-8x7b,
  qwen2-0.5b, qwen3-30b-a3b, olmo-3-7b, llama-3.1-8b, llada-8b, dream-7b):
  config.json and the repo it came from in source.json, no weights. Google's
  and Meta's gated repos come from unsloth's mirrors, minus the mirror's marker keys.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
from huggingface_hub import get_safetensors_metadata, hf_hub_download

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM,
    Gemma2Config, Gemma2ForCausalLM, Gemma3Config, Gemma3ForCausalLM, Gemma3TextConfig,
    GemmaConfig, GemmaForCausalLM, LlamaConfig, LlamaForCausalLM,
    Qwen3Config, Qwen3ForCausalLM, MistralConfig, MistralForCausalLM, PreTrainedModel,
    MixtralConfig, MixtralForCausalLM, Qwen2Config, Qwen2ForCausalLM,
    Qwen3MoeConfig, Qwen3MoeForCausalLM, Olmo3Config, Olmo3ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from transformers.models.deepseek_v32.configuration_deepseek_v32 import (
    DeepseekV32Config,
)
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32ForCausalLM,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration, Gemma4MultimodalEmbedder, Gemma4VisionModel,
)
from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig
from transformers.models.llama4.modeling_llama4 import (
    Llama4MultiModalProjector, Llama4VisionModel,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration, Qwen3_5VisionModel,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
REAL_MODEL = "Qwen/Qwen3-0.6B"
# google/gemma-3-1b-pt is gated; this mirror carries the identical config
GEMMA_MIRROR = "unsloth/gemma-3-1b-pt"
PROMPT = (
    "The Cascade Range runs from northern California through Oregon and "
    "Washington into British Columbia, and its volcanoes include Mount Rainier, "
    "Mount Hood and Mount St. Helens, which erupted in 1980. The tallest of "
    "them is the"
)
PROMPT_TOKENS = 48
TOP_K = 32
BATCH, LENGTH = 2, 12


def tiny_qwen3() -> Qwen3ForCausalLM:
    config = Qwen3Config.from_dict(dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
        tie_word_embeddings=True, rope_theta=1e6, max_position_embeddings=64,
        rms_norm_eps=1e-6, attention_bias=False, hidden_act="silu"))
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config)


def tiny_llama() -> LlamaForCausalLM:
    """Untied head and biased projections: the two switches Qwen3 leaves off.

    Llama applies config.attention_bias to all four projections, which is
    what CausalSelfAttention's one flag means, so a biased fixture is the
    test that the bias path loads.
    """
    config = LlamaConfig.from_dict(dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
        tie_word_embeddings=False, rope_theta=5e5, max_position_embeddings=64,
        rms_norm_eps=1e-5, attention_bias=True, mlp_bias=False, hidden_act="silu"))
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def tiny_qwen2() -> Qwen2ForCausalLM:
    """Biased q/k/v projections with a bias-free o_proj, and a sliding window
    from the second layer on (use_sliding_window with max_window_layers)."""
    config = Qwen2Config.from_dict(dict(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128, vocab_size=256,
        use_sliding_window=True, sliding_window=4, max_window_layers=1,
        max_position_embeddings=64, rope_theta=1e6, tie_word_embeddings=True))
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config)


# Llama 3.1's released ramp: factor 8 off 8192 positions, smoothing between
# wavelengths 8192 / 4 and 8192 / 1. The tiny fixture keeps the ramp and
# shrinks the pretraining context so that, at head_dim 16 and base 5e5, the
# smoothing band holds frequencies of the tiny table; at the released
# context it would lie beyond all of them.
LLAMA3_ROPE = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
               "high_freq_factor": 4.0, "original_max_position_embeddings": 64}


def tiny_llama31() -> LlamaForCausalLM:
    """The llama-tiny shape under Llama 3.1's rope_scaling, so a load that
    applied plain rope at rope_theta fails the parity."""
    config = LlamaConfig.from_dict(dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
        tie_word_embeddings=False, rope_theta=5e5, max_position_embeddings=512,
        rope_scaling=dict(LLAMA3_ROPE), rms_norm_eps=1e-5, hidden_act="silu"))
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def tiny_mixtral() -> MixtralForCausalLM:
    config = MixtralConfig.from_dict(dict(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=48, vocab_size=128,
        num_local_experts=4, num_experts_per_tok=2, sliding_window=4,
        max_position_embeddings=64))
    torch.manual_seed(0)
    return MixtralForCausalLM(config)


def tiny_mistral() -> MistralForCausalLM:
    config = MistralConfig.from_dict(dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
        sliding_window=4, max_position_embeddings=64, rope_theta=10000.0))
    torch.manual_seed(0)
    return MistralForCausalLM(config)


def tiny_qwen3_moe() -> Qwen3MoeForCausalLM:
    """Three layers: the first dense by mlp_only_layers, the second routed
    and the third dense by decoder_sparse_step 2, so both dials pick a
    layer. norm_topk_prob stays at the release's False, where the top-k
    softmax weights are used as they are, and the routed width differs
    from the dense one so the two cannot be confused."""
    config = Qwen3MoeConfig.from_dict(dict(
        hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=48,
        moe_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
        decoder_sparse_step=2, mlp_only_layers=[0], norm_topk_prob=False,
        vocab_size=128, max_position_embeddings=64, rope_theta=1e6))
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(config)


def tiny_olmo3() -> Olmo3ForCausalLM:
    """Four layers, so the reference's own 3:1 sliding-to-full pattern picks
    one full layer; the q/k norms over the whole projection with grouped
    heads (so a per-head norm cannot pass) and the post-norm block."""
    config = Olmo3Config.from_dict(dict(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128, vocab_size=256,
        sliding_window=4, max_position_embeddings=64, rope_theta=5e5,
        rms_norm_eps=1e-6))
    torch.manual_seed(0)
    return Olmo3ForCausalLM(config)


def tiny_olmo3_yarn() -> Olmo3ForCausalLM:
    """allenai/Olmo-3-1025-7B's rope at toy width: its rope_scaling record
    field for field (yarn, factor 8 off 8192 pretraining positions, the
    betas and the explicit attention_factor) and its 3:1 pattern, on
    head_dim 16 instead of 128.

    Olmo3Config moves a flat rope_scaling onto the full-attention entry
    (configuration_olmo3.py:110-113) and leaves the sliding layers at
    rope_theta, so this fixture is the per-kind rotary: the frequency table
    of the one full layer is YaRN's, the three sliding layers' is plain.
    At this head dim the correction range truncates to (2, 5), so of the 8
    rotated pairs two extrapolate, two ride the linear ramp and four
    interpolate at 1/8 - a ramp that neither plain rope nor a whole-model
    YaRN reproduces.
    """
    config = Olmo3Config.from_dict(dict(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=96, vocab_size=128,
        sliding_window=4, max_position_embeddings=65536, rope_theta=5e5,
        rms_norm_eps=1e-6, eos_token_id=127, pad_token_id=1,
        rope_scaling=dict(rope_type="yarn", factor=8.0, beta_fast=32, beta_slow=1,
                          original_max_position_embeddings=8192,
                          attention_factor=1.2079441541679836)))
    torch.manual_seed(0)
    return Olmo3ForCausalLM(config)


def tiny_gemma() -> GemmaForCausalLM:
    """Gemma 1 at toy width, with hidden_act 'gelu' as the released config
    spells it: transformers 5.16.1 computes that as the erf gelu
    (modeling_gemma.py:93), which is 1.7e-03 away from the tanh form on
    these weights, so the fixture tells the two apart."""
    config = GemmaConfig.from_dict(dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=1, head_dim=16, intermediate_size=128, vocab_size=256,
        hidden_act="gelu", max_position_embeddings=64))
    torch.manual_seed(0)
    return GemmaForCausalLM(config)


def tiny_gemma2() -> Gemma2ForCausalLM:
    """The Gemma 3 shape without q/k norms, alternating sliding and full
    layers, and an attention softcap of 5: at the release's 50 the cap moves
    these logits by 1.9e-02, at 5 by 1.5, so a load that dropped the cap
    fails the parity by a wide margin instead of a narrow one."""
    config = Gemma2Config.from_dict(dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, intermediate_size=128, vocab_size=256,
        query_pre_attn_scalar=16, sliding_window=4, final_logit_softcapping=30.0,
        attn_logit_softcapping=5.0, max_position_embeddings=64, rope_theta=1e4))
    torch.manual_seed(0)
    return Gemma2ForCausalLM(config)


def tiny_gemma3() -> Gemma3ForCausalLM:
    config = Gemma3TextConfig.from_dict(dict(
        hidden_size=64, num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=1, head_dim=32,
        query_pre_attn_scalar=16, sliding_window=4, intermediate_size=128,
        vocab_size=256, tie_word_embeddings=True, rope_theta=1e6,
        rope_local_base_freq=1e4, final_logit_softcapping=30.0,
        max_position_embeddings=64, rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh"))
    torch.manual_seed(0)
    return Gemma3ForCausalLM(config)


# The released rope spelling on both DeepSeek checkpoints: `rope_scaling`
# with `type` yarn, factor 40 off 4096 base positions, mscale on every dim.
DEEPSEEK_YARN = {
    "type": "yarn", "factor": 40.0, "beta_fast": 32, "beta_slow": 1,
    "mscale": 1.0, "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
}
# n_group 4 over 8 experts with topk_group 2 reaches four experts, which is
# the top_k, so the group limit decides which experts are chosen, beyond
# their order.
DEEPSEEK_TINY = dict(
    vocab_size=256, hidden_size=32, intermediate_size=48,
    moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8,
    routed_scaling_factor=2.5, q_lora_rank=8, kv_lora_rank=8,
    qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, n_group=4,
    topk_group=2, num_experts_per_tok=4, first_k_dense_replace=1,
    norm_topk_prob=True, hidden_act="silu", max_position_embeddings=64,
    rms_norm_eps=1e-6, tie_word_embeddings=False, rope_theta=10000.0,
    rope_scaling=dict(DEEPSEEK_YARN), attention_bias=False)


def tiny_deepseek_v3() -> DeepseekV3ForCausalLM:
    config = DeepseekV3Config.from_dict(dict(DEEPSEEK_TINY, rope_interleave=True))
    torch.manual_seed(0)
    return DeepseekV3ForCausalLM(config)


# The v32 fixture's weights come from this seed, not the family's 1234. The
# indexer scores a key at zero whenever every head's query-key agreement is
# negative (the relu), and torch.topk and jax.lax.top_k break an exact tie
# at the top-k boundary differently, so a fixture with a tie on any row
# compares two selections, not two implementations. At two heads a quarter
# of the keys score zero and no seed in 3000 clears every row by more than
# 3.6e-3; at eight heads seed 202 keeps the fourth and fifth scores of
# every row of both layers at least 0.0217 apart.
DEEPSEEK_V32_SEED = 202


def tiny_deepseek_v32() -> DeepseekV32ForCausalLM:
    """The V3 shape with the sparse indexer: eight heads of width 16 over
    the rope width of 8, keeping four of the twelve keys."""
    config = DeepseekV32Config.from_dict(dict(
        DEEPSEEK_TINY, index_topk=4, index_n_heads=8, index_head_dim=16))
    torch.manual_seed(0)
    return DeepseekV32ForCausalLM(config)


def _diffusion_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm without bias, the norm both diffusion decoders use."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _diffusion_rope(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half rope tables, the convention LLaDA and Dream share with Llama."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.einsum("i,j->ij", torch.arange(seq_len).float(), inv_freq)
    both = torch.cat((freqs, freqs), dim=-1)
    return both.cos(), both.sin()


def _diffusion_rotate(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _diffusion_attend(x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      o: torch.Tensor, qb, kb, vb, ob,
                      cos: torch.Tensor, sin: torch.Tensor, heads: int) -> torch.Tensor:
    """Full attention over the sequence, one head width shared by both models."""
    batch, seq, _ = x.shape
    head_dim = q.shape[-1] // heads
    queries = (x @ q.t() + (0 if qb is None else qb)).view(batch, seq, heads, head_dim)
    keys = (x @ k.t() + (0 if kb is None else kb)).view(batch, seq, -1, head_dim)
    values = (x @ v.t() + (0 if vb is None else vb)).view(batch, seq, -1, head_dim)
    table = cos.view(1, seq, 1, head_dim)
    turn = sin.view(1, seq, 1, head_dim)
    queries = queries * table + _diffusion_rotate(queries) * turn
    keys = keys * table + _diffusion_rotate(keys) * turn
    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    if keys.shape[1] != heads:
        repeat = heads // keys.shape[1]
        keys = keys.repeat_interleave(repeat, dim=1)
        values = values.repeat_interleave(repeat, dim=1)
    attended = torch.nn.functional.scaled_dot_product_attention(
        queries, keys, values, is_causal=False)
    return (attended.transpose(1, 2).reshape(batch, seq, heads * head_dim) @ o.t()
            + (0 if ob is None else ob))


class _RmsWeight(torch.nn.Module):
    """A bare RMS scale, named by its holder to match the checkpoint."""

    def __init__(self, width: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))


class LladaTinyBlock(torch.nn.Module):
    """One LLaDA llama block under the released tensor names."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, intermediate: int):
        super().__init__()
        self.attn_norm = _RmsWeight(hidden)
        self.q_proj = torch.nn.Linear(hidden, heads * hidden // heads, bias=False)
        self.k_proj = torch.nn.Linear(hidden, kv_heads * hidden // heads, bias=False)
        self.v_proj = torch.nn.Linear(hidden, kv_heads * hidden // heads, bias=False)
        self.attn_out = torch.nn.Linear(heads * hidden // heads, hidden, bias=False)
        self.ff_norm = _RmsWeight(hidden)
        self.ff_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.ff_out = torch.nn.Linear(intermediate, hidden, bias=False)
        self.heads = heads

    def forward(self, x, cos, sin):
        normed = _diffusion_rms(x, self.attn_norm.weight, 1e-5)
        x = x + _diffusion_attend(
            normed, self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
            self.attn_out.weight, None, None, None, None, cos, sin, self.heads)
        normed = _diffusion_rms(x, self.ff_norm.weight, 1e-5)
        return x + self.ff_out(torch.nn.functional.silu(self.ff_proj(normed))
                               * self.up_proj(normed))


class _LladaTransformer(torch.nn.Module):
    """The transformer holder, so the checkpoint keys read model.transformer.*."""

    def __init__(self, hidden: int, layers: int, heads: int, intermediate: int,
                 vocab: int):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab, hidden)
        self.blocks = torch.nn.ModuleList(
            [LladaTinyBlock(hidden, heads, heads, intermediate) for _ in range(layers)])
        self.ln_f = _RmsWeight(hidden)
        self.ff_out = torch.nn.Linear(hidden, vocab, bias=False)


class _LladaModel(torch.nn.Module):
    """The model holder, so the checkpoint keys read model.*."""

    def __init__(self, hidden: int, layers: int, heads: int, intermediate: int,
                 vocab: int):
        super().__init__()
        self.transformer = _LladaTransformer(hidden, layers, heads, intermediate, vocab)


class LladaTiny(torch.nn.Module):
    """LLaDA-8B's computation at toy width, under its OLMo-style names."""

    def __init__(self, hidden=32, layers=2, heads=4, intermediate=64, vocab=128,
                 theta=500000.0):
        super().__init__()
        self.model = _LladaModel(hidden, layers, heads, intermediate, vocab)
        self.theta = theta
        self.head_dim = hidden // heads

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        cos, sin = _diffusion_rope(ids.shape[1], self.head_dim, self.theta)
        x = self.model.transformer.wte(ids)
        for block in self.model.transformer.blocks:
            x = block(x, cos, sin)
        return self.model.transformer.ff_out(
            _diffusion_rms(x, self.model.transformer.ln_f.weight, 1e-5))


class _DreamAttention(torch.nn.Module):
    """The attention holder, so the checkpoint keys read self_attn.*."""

    def __init__(self, hidden: int, heads: int, kv_heads: int):
        super().__init__()
        head_dim = hidden // heads
        self.q_proj = torch.nn.Linear(hidden, heads * head_dim, bias=True)
        self.k_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=True)
        self.v_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=True)
        self.o_proj = torch.nn.Linear(heads * head_dim, hidden, bias=False)


class _DreamMlp(torch.nn.Module):
    """The MLP holder, so the checkpoint keys read mlp.*."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)


class _DreamModel(torch.nn.Module):
    """The model holder, so the checkpoint keys read model.*."""

    def __init__(self, hidden: int, layers: int, heads: int, kv_heads: int,
                 intermediate: int, vocab: int):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.layers = torch.nn.ModuleList(
            [DreamTinyLayer(hidden, heads, kv_heads, intermediate) for _ in range(layers)])
        self.norm = _RmsWeight(hidden)


class DreamTinyLayer(torch.nn.Module):
    """One Dream decoder layer under the qwen2 tensor names."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, intermediate: int):
        super().__init__()
        self.input_layernorm = _RmsWeight(hidden)
        self.self_attn = _DreamAttention(hidden, heads, kv_heads)
        self.post_attention_layernorm = _RmsWeight(hidden)
        self.mlp = _DreamMlp(hidden, intermediate)
        self.heads = heads

    def forward(self, x, cos, sin):
        normed = _diffusion_rms(x, self.input_layernorm.weight, 1e-6)
        x = x + _diffusion_attend(
            normed, self.self_attn.q_proj.weight, self.self_attn.k_proj.weight,
            self.self_attn.v_proj.weight, self.self_attn.o_proj.weight,
            self.self_attn.q_proj.bias, self.self_attn.k_proj.bias,
            self.self_attn.v_proj.bias, None, cos, sin, self.heads)
        normed = _diffusion_rms(x, self.post_attention_layernorm.weight, 1e-6)
        return x + self.mlp.down_proj(torch.nn.functional.silu(self.mlp.gate_proj(normed))
                                      * self.mlp.up_proj(normed))


class DreamTiny(torch.nn.Module):
    """Dream-v0's computation at toy width, under its qwen2 names."""

    def __init__(self, hidden=32, layers=2, heads=4, kv_heads=2, intermediate=64,
                 vocab=128, theta=1000000.0):
        super().__init__()
        self.model = _DreamModel(hidden, layers, heads, kv_heads, intermediate, vocab)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.theta = theta
        self.head_dim = hidden // heads

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        cos, sin = _diffusion_rope(ids.shape[1], self.head_dim, self.theta)
        x = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            x = layer(x, cos, sin)
        return self.lm_head(_diffusion_rms(x, self.model.norm.weight, 1e-6))


LLADA_TINY_CONFIG = {
    "model_type": "llada", "architectures": ["LLaDAModelLM"],
    "d_model": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 4,
    "mlp_hidden_size": 64, "embedding_size": 128, "vocab_size": 128,
    "max_sequence_length": 64, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
    "mask_token_id": 120, "weight_tying": False, "include_bias": False,
    "include_qkv_bias": False, "activation_type": "silu", "block_type": "llama",
    "layer_norm_type": "rms", "layer_norm_with_affine": True,
    "attention_layer_norm": False, "bias_for_layer_norm": False,
    "rope": True, "rope_full_precision": True,
}

DREAM_TINY_CONFIG = {
    "model_type": "Dream", "architectures": ["DreamModel"],
    "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "intermediate_size": 64, "vocab_size": 128,
    "max_position_embeddings": 64, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
    "mask_token_id": 120, "tie_word_embeddings": False, "hidden_act": "silu",
    "use_mrope": False, "use_sliding_window": False, "sliding_window": None,
    "rope_scaling": None,
}


def write_diffusion_tiny(name: str, model: torch.nn.Module, config: dict,
                         seed: int = 1234) -> None:
    """A diffusion tiny fixture: the torch reference's weights under the
    released tensor names, its config and its fp32 logits on fixed ids."""
    from safetensors.torch import save_file

    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)
    scatter_weights(model, seed)
    model = model.float().eval()
    save_file(model.state_dict(), directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    ids = np.random.RandomState(7).randint(
        0, config["vocab_size"], (BATCH, LENGTH)).astype(np.int64)
    np.save(directory / "input_ids.npy", ids.astype(np.int32))
    with torch.no_grad():
        logits = model(torch.from_numpy(ids)).to(torch.float32).numpy()
    np.save(directory / "logits.npy", logits)
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


def siglip_tiny_system(seed: int = 1234, text_width: int = 16, mm_tokens: int = 1):
    """A tiny SigLIP tower and Gemma projector with scattered weights.

    Returns the torch modules with fp32 reference outputs on fixed pixels, and
    the configs that describe them. The tower is two layers of width 32 over a
    2x2 patch grid pooled to one soft token; both writers below share this
    system so the plain and wrapper fixtures agree.
    """

    vconf = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, image_size=28, patch_size=14, num_channels=3,
        hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6)
    torch.manual_seed(seed)
    tower = SiglipVisionModel(vconf)
    scatter_weights(tower, seed)
    tower = tower.float().eval()
    pixels = np.random.RandomState(11).rand(BATCH, 3, 28, 28).astype(np.float32)
    with torch.no_grad():
        last = tower(pixel_values=torch.from_numpy(pixels),
                     return_dict=True).last_hidden_state.to(torch.float32).numpy()
    gconf = Gemma3Config(
        text_config=Gemma3TextConfig(hidden_size=text_width),
        vision_config=vconf, mm_tokens_per_image=mm_tokens)
    projector = Gemma3MultiModalProjector(gconf)
    scatter_weights(projector, seed + 1)
    projector = projector.float().eval()
    with torch.no_grad():
        soft = projector(torch.from_numpy(last)).to(torch.float32).numpy()
    return {"tower": tower, "projector": projector, "vconf": vconf,
            "pixels": pixels, "last": last, "soft": soft,
            "text_width": text_width, "mm_tokens": mm_tokens}


def llama4_vision_tiny_system(seed: int = 1234, text_width: int = 32):
    """A tiny Llama 4 vision trunk and outer projector, same sharing deal."""
    from types import SimpleNamespace

    vconf = Llama4VisionConfig(
        hidden_size=32, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, image_size=28, patch_size=14, num_channels=3,
        norm_eps=1e-5, hidden_act="gelu",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        pixel_shuffle_ratio=0.5, projector_input_dim=64, projector_output_dim=64,
        vision_output_dim=64, vision_feature_select_strategy="default",
        attention_dropout=0.0, projector_dropout=0.0)
    torch.manual_seed(seed)
    tower = Llama4VisionModel(vconf)
    scatter_weights(tower, seed)
    tower = tower.float().eval()
    pixels = np.random.RandomState(11).rand(BATCH, 3, 28, 28).astype(np.float32)
    with torch.no_grad():
        last = tower(pixel_values=torch.from_numpy(pixels),
                     return_dict=True).last_hidden_state.to(torch.float32).numpy()
    projector = Llama4MultiModalProjector(SimpleNamespace(
        vision_config=vconf, text_config=SimpleNamespace(hidden_size=text_width)))
    scatter_weights(projector, seed + 1)
    projector = projector.float().eval()
    with torch.no_grad():
        soft = projector(torch.from_numpy(last)).to(torch.float32).numpy()
    return {"tower": tower, "projector": projector, "vconf": vconf,
            "pixels": pixels, "last": last, "soft": soft, "text_width": text_width}


def write_siglip_tiny() -> None:
    """The SigLIP trunk and Gemma projector as Dew reads them: bare tensor
    names, the tower config, the projector's two knobs, fixed pixels and both
    fp32 reference outputs."""
    from safetensors.torch import save_file

    system = siglip_tiny_system()
    directory = FIXTURES / "siglip-tiny"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(system["tower"].state_dict(), directory / "model.safetensors")
    save_file(system["projector"].state_dict(), directory / "projector.safetensors")
    (directory / "config.json").write_text(
        json.dumps(system["vconf"].to_dict(), indent=1) + "\n")
    (directory / "projector.json").write_text(json.dumps(
        {"text_width": system["text_width"],
         "mm_tokens_per_image": system["mm_tokens"]}, indent=1) + "\n")
    np.save(directory / "pixels.npy", system["pixels"])
    np.save(directory / "tower_ref.npy", system["last"])
    np.save(directory / "projector_ref.npy", system["soft"])
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


def write_llama4_vision_tiny() -> None:
    """The Llama 4 trunk and outer projector, same layout."""
    from safetensors.torch import save_file

    system = llama4_vision_tiny_system()
    directory = FIXTURES / "llama4-vision-tiny"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(system["tower"].state_dict(), directory / "model.safetensors")
    save_file(system["projector"].state_dict(), directory / "projector.safetensors")
    (directory / "config.json").write_text(
        json.dumps(system["vconf"].to_dict(), indent=1) + "\n")
    (directory / "projector.json").write_text(json.dumps(
        {"text_width": system["text_width"]}, indent=1) + "\n")
    np.save(directory / "pixels.npy", system["pixels"])
    np.save(directory / "tower_ref.npy", system["last"])
    np.save(directory / "projector_ref.npy", system["soft"])
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


def write_gemma3_mm_tiny() -> None:
    """A Gemma 3 wrapper fixture: the SigLIP and projector halves under the
    released model.* prefixes beside a tiny decoder half, with the wrapper
    config and both vision reference outputs.

    The decoder half reuses gemma3-tiny's weights under model.language_model.*,
    and the tied head rides top-level as the released layout carries it. The
    tower and projector come from the shared SigLIP system at a fresh seed.
    """
    from safetensors.torch import load_file, save_file

    system = siglip_tiny_system(seed=4321, text_width=64)
    text = load_file(str(FIXTURES / "gemma3-tiny" / "model.safetensors"))
    text_config = json.loads((FIXTURES / "gemma3-tiny" / "config.json").read_text())
    merged = {}
    for name, tensor in system["tower"].state_dict().items():
        merged[f"model.vision_tower.{name}"] = tensor
    for name, tensor in system["projector"].state_dict().items():
        merged[f"model.multi_modal_projector.{name}"] = tensor
    for name, tensor in text.items():
        tail = name[6:] if name.startswith("model.") else name
        merged[f"model.language_model.{tail}"] = tensor
    merged["lm_head.weight"] = text["model.embed_tokens.weight"].clone()
    directory = FIXTURES / "gemma3-tiny-mm"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(merged, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({
        "model_type": "gemma3",
        "text_config": text_config,
        "vision_config": system["vconf"].to_dict(),
        "mm_tokens_per_image": system["mm_tokens"],
        "boi_token_index": 200, "eoi_token_index": 201,
        "image_token_index": 202}, indent=1) + "\n")
    np.save(directory / "pixels.npy", system["pixels"])
    np.save(directory / "tower_ref.npy", system["last"])
    ids = np.array([[2, 5, 202, 7, 9], [202, 3, 4, 5, 6]], np.int32)
    with torch.no_grad():
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3ForConditionalGeneration,
        )
        wrapper = Gemma3ForConditionalGeneration(Gemma3Config(
            text_config={k: v for k, v in text_config.items()
                         if k not in ("architectures", "model_type")},
            vision_config=system["vconf"], mm_tokens_per_image=system["mm_tokens"],
            boi_token_index=200, eoi_token_index=201, image_token_index=202))
        wrapper.load_state_dict(
            {k: v for k, v in merged.items()}, strict=True)
        wrapper = wrapper.float().eval()
        logits = wrapper(
            input_ids=torch.from_numpy(ids), pixel_values=torch.from_numpy(
                system["pixels"]), use_cache=False).logits.to(torch.float32).numpy()
    np.save(directory / "input_ids.npy", ids)
    np.save(directory / "wrapper_ref.npy", logits)
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


def write_llama4_mm_tiny() -> None:
    """A Llama 4 wrapper fixture: vision halves and decoder half under the
    released nesting with no model prefix, with the wrapper config, both
    vision references and the wrapper logits."""
    from safetensors.torch import load_file, save_file

    system = llama4_vision_tiny_system(seed=4321)
    text = load_file(str(FIXTURES / "llama4-tiny" / "model.safetensors"))
    text_config = json.loads((FIXTURES / "llama4-tiny" / "config.json").read_text())
    merged = {}
    for name, tensor in system["tower"].state_dict().items():
        merged[f"vision_model.{name}"] = tensor
    for name, tensor in system["projector"].state_dict().items():
        merged[f"multi_modal_projector.{name}"] = tensor
    for name, tensor in text.items():
        merged[f"language_model.{name}"] = tensor
    directory = FIXTURES / "llama4-tiny-mm"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(merged, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({
        "model_type": "llama4",
        "text_config": text_config,
        "vision_config": system["vconf"].to_dict(),
        "boi_token_index": 90, "eoi_token_index": 91,
        "image_token_index": 92}, indent=1) + "\n")
    np.save(directory / "pixels.npy", system["pixels"])
    np.save(directory / "tower_ref.npy", system["last"])
    np.save(directory / "projector_ref.npy", system["soft"])
    ids = np.array([[2, 5, 92, 7, 9], [92, 3, 4, 5, 6]], np.int32)
    with torch.no_grad():
        from transformers.models.llama4.configuration_llama4 import Llama4Config
        from transformers.models.llama4.modeling_llama4 import (
            Llama4ForConditionalGeneration,
        )
        wrapper = Llama4ForConditionalGeneration(Llama4Config(
            text_config={k: v for k, v in text_config.items()
                         if k not in ("architectures", "model_type")},
            vision_config=system["vconf"],
            boi_token_index=90, eoi_token_index=91, image_token_index=92))
        wrapper.load_state_dict(merged, strict=True)
        wrapper = wrapper.float().eval()
        logits = wrapper(
            input_ids=torch.from_numpy(ids), pixel_values=torch.from_numpy(
                system["pixels"]), use_cache=False).logits.to(torch.float32).numpy()
    np.save(directory / "input_ids.npy", ids)
    np.save(directory / "wrapper_ref.npy", logits)
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


G4V_TEXT_WIDTH = 32
G4V_IMAGE = 60


def gemma4_vision_tiny_config() -> Gemma4VisionConfig:
    """Two layers of width 32 over a 4x4 patch grid pooled by 2 into four
    soft tokens, with one grouped-query repeat and the released
    standardization on."""
    return Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6,
        patch_size=8, pooling_kernel_size=2, position_embedding_size=64,
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
        standardize=True, use_clipped_linears=False, attention_bias=False,
        attention_dropout=0.0)


def gemma4_positions(grid: int, batch: int) -> np.ndarray:
    """The processor's (x, y) patch ids for a square grid, row-major."""
    rows = np.arange(grid).reshape(-1, 1).repeat(grid, axis=1).reshape(-1)
    cols = np.arange(grid).reshape(1, -1).repeat(grid, axis=0).reshape(-1)
    one = np.stack([cols, rows], axis=-1).astype(np.int64)
    return np.stack([one] * batch, axis=0)


def gemma4_vision_tiny_system(seed: int = 1234):
    """A tiny Gemma 4 vision trunk and multimodal embedder with scattered
    weights, on patchified pixels as the processor emits them.

    Returns the torch modules with fp32 reference outputs on fixed patches,
    and the configs that describe them. The 32-pixel image patchifies into
    a 4x4 grid pooled to four soft tokens. The standardization buffers ride
    the state dict, so they are scattered with the weights.
    """
    from types import SimpleNamespace

    vconf = gemma4_vision_tiny_config()
    torch.manual_seed(seed)
    tower = Gemma4VisionModel(vconf)
    scatter_weights(tower, seed)
    with torch.no_grad():
        tower.std_bias.copy_(torch.randn(32) * 0.5)
        tower.std_scale.copy_(1.0 + torch.randn(32) * 0.05)
    tower = tower.float().eval()
    pixels = np.random.RandomState(11).rand(BATCH, 16, 192).astype(np.float32)
    positions = gemma4_positions(4, BATCH)
    with torch.no_grad():
        last = tower(pixel_values=torch.from_numpy(pixels),
                     pixel_position_ids=torch.from_numpy(positions),
                     return_dict=True).last_hidden_state.to(torch.float32).numpy()
    # The trunk strips padding with a boolean mask, which flattens the batch;
    # the fixture has no padding, so the reshape back is the same tokens.
    last = last.reshape(BATCH, -1, vconf.hidden_size)
    projector = Gemma4MultimodalEmbedder(
        vconf, Gemma4TextConfig(hidden_size=G4V_TEXT_WIDTH))
    scatter_weights(projector, seed + 1)
    projector = projector.float().eval()
    with torch.no_grad():
        soft = projector(torch.from_numpy(last)).to(torch.float32).numpy()
    return {"tower": tower, "projector": projector, "vconf": vconf,
            "pixels": pixels, "positions": positions, "last": last,
            "soft": soft}


def write_gemma4_vision_tiny() -> None:
    """The Gemma 4 trunk and embedder as Dew reads them: bare tensor names,
    the tower config, the text width, fixed patches and positions, and both
    fp32 reference outputs."""
    from safetensors.torch import save_file

    system = gemma4_vision_tiny_system()
    directory = FIXTURES / "gemma4-vision-tiny"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(system["tower"].state_dict(), directory / "model.safetensors")
    save_file(system["projector"].state_dict(), directory / "projector.safetensors")
    (directory / "config.json").write_text(
        json.dumps(system["vconf"].to_dict(), indent=1) + "\n")
    (directory / "projector.json").write_text(json.dumps(
        {"text_width": G4V_TEXT_WIDTH}, indent=1) + "\n")
    np.save(directory / "pixels.npy", system["pixels"])
    np.save(directory / "positions.npy", system["positions"])
    np.save(directory / "tower_ref.npy", system["last"])
    np.save(directory / "projector_ref.npy", system["soft"])
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


QWEN_VISION_TEXT_WIDTH = 64
QWEN_VISION_IMAGE = 200


def qwen35_vision_tiny_config() -> Qwen3_5VisionConfig:
    """Two layers of width 32 over a 4x4 patch grid merged by 2 into four
    soft tokens, with the two-frame temporal patch the still-image processor
    fills by repeating the frame."""
    return Qwen3_5VisionConfig(
        depth=2, hidden_size=32, hidden_act="gelu_pytorch_tanh",
        intermediate_size=64, num_heads=4, in_channels=3, patch_size=8,
        spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
        num_position_embeddings=64)


def qwen35_patchify(images: np.ndarray, patch_size: int, merge_size: int,
                    temporal_patch_size: int) -> np.ndarray:
    """Still images into flat tokens the way the Qwen processor lays them:
    spatial patches in merge-block order, each frame repeated along time
    (image_processing_qwen2_vl.py, patchify)."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    processor = Qwen2VLImageProcessor()
    patches, _, _ = processor.patchify(torch.from_numpy(images), patch_size, merge_size, temporal_patch_size)
    return patches.reshape(-1, patches.shape[-1]).numpy()


def qwen35_vision_tiny_system(seed: int = 1234):
    """A tiny Qwen 3.5 vision trunk and merger with scattered weights.

    Returns the torch modules with fp32 reference outputs on a fixed 32-pixel
    image, and the configs that describe them. The image patchifies into a
    4x4 grid over an 8x8 learned position table, so the interpolation path
    runs, and merges by 2 into four soft tokens.
    """
    vconf = qwen35_vision_tiny_config()
    torch.manual_seed(seed)
    tower = Qwen3_5VisionModel(vconf)
    scatter_weights(tower, seed)
    tower = tower.float().eval()
    pixels = np.random.RandomState(11).rand(BATCH, 3, 32, 32).astype(np.float32)
    patch_size = vconf.patch_size
    temporal_patch_size = vconf.temporal_patch_size
    assert isinstance(patch_size, int) and isinstance(temporal_patch_size, int)
    features = qwen35_patchify(pixels, patch_size, vconf.spatial_merge_size,
                               temporal_patch_size)
    grid = torch.tensor([[1, 4, 4]] * BATCH)
    with torch.no_grad():
        output = tower(hidden_states=torch.from_numpy(features), grid_thw=grid,
                       return_dict=True)
        last = output.last_hidden_state.to(torch.float32).numpy().reshape(
            BATCH, -1, vconf.hidden_size)
        soft = output.pooler_output.to(torch.float32).numpy().reshape(
            BATCH, -1, vconf.out_hidden_size)
    return {"tower": tower, "vconf": vconf, "pixels": pixels,
            "last": last, "soft": soft}


def write_qwen35_vision_tiny() -> None:
    """The Qwen 3.5 trunk and merger as Dew reads them: bare tensor names
    with the merger under its released prefix, the tower config, the fixed
    square image, and both fp32 reference outputs."""
    from safetensors.torch import save_file

    system = qwen35_vision_tiny_system()
    directory = FIXTURES / "qwen35-vision-tiny"
    directory.mkdir(parents=True, exist_ok=True)
    tower_tensors = {name: tensor for name, tensor in
                     system["tower"].state_dict().items()
                     if not name.startswith("merger.")}
    merger_tensors = {name: tensor for name, tensor in
                      system["tower"].state_dict().items()
                      if name.startswith("merger.")}
    save_file(tower_tensors, directory / "model.safetensors")
    save_file(merger_tensors, directory / "projector.safetensors")
    (directory / "config.json").write_text(
        json.dumps(system["vconf"].to_dict(), indent=1) + "\n")
    (directory / "projector.json").write_text(json.dumps(
        {"text_width": QWEN_VISION_TEXT_WIDTH}, indent=1) + "\n")
    np.save(directory / "pixels.npy", system["pixels"])
    np.save(directory / "tower_ref.npy", system["last"])
    np.save(directory / "projector_ref.npy", system["soft"])
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


GEMMA4_MM_TEXT: Dict[str, Any] = dict(
    vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"],
    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    hidden_activation="gelu_pytorch_tanh", attention_k_eq_v=True,
    sliding_window=4, hidden_size_per_layer_input=0, num_kv_shared_layers=0,
    per_layer_config={"2": {"head_dim": 16, "num_key_value_heads": 1}},
    max_position_embeddings=64, rms_norm_eps=1e-6,
    final_logit_softcapping=30.0, tie_word_embeddings=True,
    bos_token_id=2, eos_token_id=1, pad_token_id=0,
    use_bidirectional_attention="vision", use_double_wide_mlp=False,
    rope_parameters={
        "full_attention": {"rope_type": "proportional", "rope_theta": 1e6,
                           "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}})

QWEN35_MM_TEXT: Dict[str, Any] = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, hidden_act="silu", max_position_embeddings=64,
    rms_norm_eps=1e-6, tie_word_embeddings=True,
    linear_conv_kernel_dim=4, linear_key_head_dim=12, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    rope_parameters={"rope_type": "default", "rope_theta": 1000000.0,
                     "partial_rotary_factor": 0.25,
                     "mrope_interleaved": True, "mrope_section": [2, 1, 1]})


def write_gemma4_mm_tiny() -> None:
    """A Gemma 4 wrapper fixture: the vision and embedder halves under the
    released model.* prefixes beside a tiny dense decoder half, with the
    wrapper config, patches and positions, and the wrapper logits.

    Each row marks four image positions, the pooled count of the 4x4 patch
    grid, and the reference runs without multimodal masks, so the text side
    stays causal the way Dew's decoder runs it.
    """
    from safetensors.torch import save_file

    system = gemma4_vision_tiny_system(seed=4321)
    torch.manual_seed(4321)
    text = Gemma4ForConditionalGeneration(Gemma4Config(
        text_config=Gemma4TextConfig(**GEMMA4_MM_TEXT),
        vision_config=system["vconf"], audio_config=None)).model.language_model
    scatter_weights(text, seed=4321)
    merged = {}
    for name, tensor in system["tower"].state_dict().items():
        merged[f"model.vision_tower.{name}"] = tensor
    for name, tensor in system["projector"].state_dict().items():
        merged[f"model.embed_vision.{name}"] = tensor
    for name, tensor in text.state_dict().items():
        if name == "lm_head.weight":
            continue
        merged[f"model.language_model.{name}"] = tensor
    merged["lm_head.weight"] = text.state_dict()["embed_tokens.weight"].clone()
    directory = FIXTURES / "gemma4-tiny-mm"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(merged, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({
        "model_type": "gemma4",
        "text_config": dict(GEMMA4_MM_TEXT, model_type="gemma4_text"),
        "vision_config": system["vconf"].to_dict(),
        "audio_config": None,
        "vision_soft_tokens_per_image": 4,
        "boi_token_id": 61, "eoi_token_id": 62,
        "image_token_id": G4V_IMAGE}, indent=1) + "\n")
    np.save(directory / "pixels.npy", system["pixels"])
    np.save(directory / "positions.npy", system["positions"])
    np.save(directory / "tower_ref.npy", system["last"])
    np.save(directory / "projector_ref.npy", system["soft"])
    ids = np.array([[60, 60, 60, 60, 9], [2, 60, 60, 60, 60]], np.int32)
    with torch.no_grad():
        wrapper = Gemma4ForConditionalGeneration(Gemma4Config(
            text_config=dict(GEMMA4_MM_TEXT, model_type="gemma4_text"),
            vision_config=system["vconf"], audio_config=None,
            image_token_id=G4V_IMAGE, boi_token_id=61, eoi_token_id=62))
        wrapper.load_state_dict(merged, strict=True)
        wrapper = wrapper.float().eval()
        logits = wrapper(
            input_ids=torch.from_numpy(ids),
            pixel_values=torch.from_numpy(system["pixels"]),
            image_position_ids=torch.from_numpy(system["positions"]),
            use_cache=False).logits.to(torch.float32).numpy()
    np.save(directory / "input_ids.npy", ids)
    np.save(directory / "wrapper_ref.npy", logits)
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


def write_qwen35_mm_tiny() -> None:
    """A Qwen 3.5 wrapper fixture: the vision trunk and merger beside a tiny
    hybrid decoder half under the released model.* nesting, with the wrapper
    config, the square image, and the wrapper logits.

    Each row marks four image positions, the merged count of the 4x4 patch
    grid. The reference's image-grid positions are outside what Dew models,
    so the wrapper reference runs on pre-merged embeddings with no vision
    inputs, which scores with plain sequential positions on both sides.
    """
    from safetensors.torch import load_file, save_file

    system = qwen35_vision_tiny_system(seed=4321)
    torch.manual_seed(4321)
    text = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**QWEN35_MM_TEXT)).model
    scatter_weights(text, seed=4321)
    merged = {}
    for name, tensor in system["tower"].state_dict().items():
        merged[f"model.visual.{name}"] = tensor
    for name, tensor in text.state_dict().items():
        if name == "lm_head.weight":
            continue
        merged[f"model.language_model.{name}"] = tensor
    merged["lm_head.weight"] = text.state_dict()["embed_tokens.weight"].clone()
    directory = FIXTURES / "qwen35-tiny-mm"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(merged, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": dict(QWEN35_MM_TEXT, model_type="qwen3_5_text"),
        "vision_config": system["vconf"].to_dict(),
        "image_token_id": QWEN_VISION_IMAGE, "video_token_id": 201,
        "vision_start_token_id": 202, "vision_end_token_id": 203},
        indent=1) + "\n")
    np.save(directory / "pixels.npy", system["pixels"])
    np.save(directory / "tower_ref.npy", system["last"])
    np.save(directory / "projector_ref.npy", system["soft"])
    ids = np.array([[2, 200, 200, 200, 200], [200, 200, 200, 200, 9]], np.int32)
    with torch.no_grad():
        wrapper = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
            text_config=dict(QWEN35_MM_TEXT, model_type="qwen3_5_text"),
            vision_config=system["vconf"].to_dict(),
            image_token_id=QWEN_VISION_IMAGE, video_token_id=201,
            vision_start_token_id=202, vision_end_token_id=203))
        wrapper.load_state_dict(merged, strict=True)
        wrapper = wrapper.float().eval()
        embeds = wrapper.model.get_input_embeddings()(torch.from_numpy(ids))
        image = torch.from_numpy(system["soft"]).to(embeds.dtype)
        mask = (torch.from_numpy(ids) == QWEN_VISION_IMAGE).unsqueeze(-1).expand_as(embeds)
        fused = embeds.masked_scatter(mask, image)
        logits = wrapper(inputs_embeds=fused, use_cache=False).logits.to(
            torch.float32).numpy()
    np.save(directory / "input_ids.npy", ids)
    np.save(directory / "wrapper_ref.npy", logits)
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


def write_diffusion_sc_tiny() -> None:
    """The self-conditioning MLP alone: its weights, narrow config, fixed
    inputs and the fp32 reference output."""
    from safetensors.torch import save_file
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaTextConfig,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaSelfConditioning,
    )

    config = DiffusionGemmaTextConfig(
        hidden_size=32, intermediate_size=64, hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6)
    module = DiffusionGemmaSelfConditioning(config)
    scatter_weights(module, seed=4321)
    module = module.float().eval()
    rng = np.random.RandomState(11)
    embeds = rng.rand(BATCH, 4, 32).astype(np.float32)
    signal = rng.rand(BATCH, 4, 32).astype(np.float32)
    with torch.no_grad():
        ref = module(torch.from_numpy(embeds),
                     torch.from_numpy(signal)).to(torch.float32).numpy()
    directory = FIXTURES / "diffusion-gemma-sc-tiny"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(module.state_dict(), directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(
        {"hidden_size": 32, "intermediate_size": 64,
         "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6},
        indent=1) + "\n")
    np.save(directory / "inputs.npy", embeds)
    np.save(directory / "signal.npy", signal)
    np.save(directory / "ref.npy", ref)
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


DIFFUSION_DENOISER_TEXT = {
    "model_type": "diffusion_gemma_text",
    "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 64,
    "vocab_size": 64, "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
    "hidden_activation": "gelu_pytorch_tanh", "tie_word_embeddings": False,
    "attention_bias": False, "sliding_window": 8,
    "layer_types": ["full_attention", "full_attention"],
    "rope_parameters": {
        "full_attention": {"rope_type": "proportional",
                           "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
    "num_experts": 4, "top_k_experts": 2, "moe_intermediate_size": 16,
    "global_head_dim": 8, "num_global_key_value_heads": 2,
}


def write_diffusion_denoiser_tiny() -> None:
    """One DiffusionGemma denoise step: the text weights under the released
    encoder/decoder prefixes, the text config, a fixed prompt, canvas and
    previous logits, and the fp32 reference logits with and without
    self-conditioning.

    The reference ties its encoder and decoder text weights, so the fixture
    copies the encoder's text weights over the decoder's after scattering;
    the dew tree holds them once. The tied check in the loader refuses a
    checkpoint that disagrees there.
    """
    from safetensors.torch import save_file
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaConfig, DiffusionGemmaTextConfig,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaForBlockDiffusion,
    )
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    text = DiffusionGemmaTextConfig(**{
        key: value for key, value in DIFFUSION_DENOISER_TEXT.items()
        if key != "model_type"})
    vision = Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16, patch_size=8,
        pooling_kernel_size=2, position_embedding_size=64)
    model = DiffusionGemmaForBlockDiffusion(DiffusionGemmaConfig(
        text_config=text, vision_config=vision, canvas_length=4,
        tie_word_embeddings=False))
    state = model.state_dict()
    for name in list(state):
        if name.startswith("model.decoder.layers.") or name in (
                "model.decoder.embed_tokens.weight", "model.decoder.norm.weight"):
            counterpart = name.replace("model.decoder.", "model.encoder.language_model.", 1)
            state[name].copy_(state[counterpart])
    model.load_state_dict(state)
    model = model.float().eval()
    saved = {name: tensor for name, tensor in model.state_dict().items()
             if name.startswith("model.encoder.language_model.")
             or name.startswith("model.decoder.") or name == "lm_head.weight"}
    directory = FIXTURES / "diffusion-gemma-denoise-tiny"
    directory.mkdir(parents=True, exist_ok=True)
    save_file(saved, directory / "model.safetensors")
    (directory / "config.json").write_text(
        json.dumps(DIFFUSION_DENOISER_TEXT, indent=1) + "\n")
    prompt = np.array([[2, 5, 7, 9]], np.int32)
    canvas = np.array([[3, 4, 5, 6]], np.int32)
    prev = np.random.RandomState(3).randn(1, 4, 64).astype(np.float32)
    with torch.no_grad():
        bare = model(input_ids=torch.from_numpy(prompt),
                     decoder_input_ids=torch.from_numpy(canvas)
                     ).logits.to(torch.float32).numpy()
        conditioned = model(
            input_ids=torch.from_numpy(prompt),
            decoder_input_ids=torch.from_numpy(canvas),
            self_conditioning_logits=torch.from_numpy(prev)
            ).logits.to(torch.float32).numpy()
    np.save(directory / "prompt.npy", prompt)
    np.save(directory / "canvas.npy", canvas)
    np.save(directory / "prev_logits.npy", prev)
    np.save(directory / "ref_bare.npy", bare)
    np.save(directory / "ref_conditioned.npy", conditioned)
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")



def scatter_weights(model: torch.nn.Module, seed: int = 1234) -> None:
    """Random weights with something in every tensor.

    A freshly constructed model leaves the RMSNorm scales at their identity
    value, and a fixture whose norms are all ones or all zeros would pass a
    parity test that had the (1 + w) offset backwards.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            noise = torch.randn(tensor.shape, generator=generator) * 0.05
            tensor.copy_(tensor + noise if "norm" in name or "layernorm" in name
                         else noise * 4.0)
        # DeepSeek's balancing bias is a buffer the checkpoint carries, and
        # the reference selects on it: nonzero, or the load path that reads
        # it would agree with one that drops it.
        for name, tensor in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                tensor.copy_(torch.linspace(-0.4, 0.4, tensor.shape[0]))


def reference_logits(model: PreTrainedModel, ids: np.ndarray) -> np.ndarray:
    model.eval()
    model.set_attn_implementation("eager")
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids), use_cache=False)
    return out.logits.to(torch.float32).numpy()


def write_tiny(name: str, model: PreTrainedModel, seed: int = 1234) -> None:
    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)
    scatter_weights(model, seed)
    model = model.float()
    model.save_pretrained(directory, safe_serialization=True)

    ids = np.random.RandomState(7).randint(
        0, model.config.vocab_size, (BATCH, LENGTH)).astype(np.int32)
    np.save(directory / "input_ids.npy", ids)
    np.save(directory / "logits.npy", reference_logits(model, ids))
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, {sorted(p.name for p in directory.iterdir())}")


def write_tensor_table(directory: Path) -> None:
    """The real checkpoint's config, tensor names, shapes and dtypes.

    No weights: the hub serves this table without them, so a test can hold
    the parameter tree of a 1.5 GB checkpoint to account.
    """
    metadata = get_safetensors_metadata(REAL_MODEL)
    tensors = {}
    for file_metadata in metadata.files_metadata.values():
        for tensor_name, info in file_metadata.tensors.items():
            tensors[tensor_name] = {"shape": list(info.shape), "dtype": info.dtype}
    payload = {"repo": REAL_MODEL, "tensors": dict(sorted(tensors.items()))}
    (directory / "tensors.json").write_text(json.dumps(payload, indent=1) + "\n")

    config = hf_hub_download(REAL_MODEL, "config.json")
    (directory / "config.json").write_text(Path(config).read_text())
    print(f"{directory / 'tensors.json'}: {len(tensors)} tensors, with config.json")


def write_real_reference(directory: Path) -> None:
    """A prompt, and what the real weights predict for every position of it."""
    tokenizer = AutoTokenizer.from_pretrained(REAL_MODEL)
    ids = tokenizer(PROMPT, return_tensors="np")["input_ids"][:, :PROMPT_TOKENS]
    if ids.shape[1] != PROMPT_TOKENS:
        raise SystemExit(
            f"the prompt tokenizes to {ids.shape[1]} ids, not {PROMPT_TOKENS}")
    (directory / "prompt.json").write_text(json.dumps({
        "repo": REAL_MODEL, "prompt": PROMPT,
        "input_ids": ids[0].tolist(),
    }, indent=1) + "\n")

    model = AutoModelForCausalLM.from_pretrained(REAL_MODEL, dtype=torch.float32)
    logits = reference_logits(model, ids.astype(np.int64))[0]
    order = np.argsort(-logits, axis=-1)[:, :TOP_K]
    np.savez(
        directory / "reference.npz",
        top_ids=order.astype(np.int32),
        top_logits=np.take_along_axis(logits, order, axis=-1).astype(np.float32),
        argmax=np.argmax(logits, axis=-1).astype(np.int32))
    print(f"{directory / 'reference.npz'}: {logits.shape[0]} positions, "
          f"top {TOP_K}, argmax[:8]={np.argmax(logits, axis=-1)[:8].tolist()}")


def write_released_config(name: str, repo: str) -> None:
    """The real config.json of a released checkpoint, and the repo it came
    from in source.json. Only the config is downloaded, never the weights.

    Google's Gemma repos answer 401 without an accepted licence, so those
    come from unsloth's mirrors, which carry the identical config plus
    marker keys of their own (unsloth_fixed, unsloth_version); the markers
    are dropped so the fixture is the released config alone.
    """
    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    for key in [key for key in config if key.startswith("unsloth")]:
        del config[key]
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    (directory / "source.json").write_text(json.dumps({"repo": repo}) + "\n")
    layers = config.get('num_hidden_layers', config.get('n_layers', config.get('num_layers')))
    print(f"{directory / 'config.json'}: {repo}, "
          f"{layers} layers, {len(config)} fields")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-real", action="store_true",
                        help="only the tiny fixtures, no 1.5 GB download")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    write_tiny("qwen3-tiny", tiny_qwen3())
    write_tiny("gemma3-tiny", tiny_gemma3())
    write_tiny("gemma-tiny", tiny_gemma())
    write_tiny("gemma2-tiny", tiny_gemma2())
    write_tiny("llama-tiny", tiny_llama())
    write_tiny("llama31-tiny", tiny_llama31())
    write_released_config("llama-3.1-8b", "unsloth/Llama-3.1-8B")
    write_tiny("mistral-tiny", tiny_mistral())
    write_tiny("mixtral-tiny", tiny_mixtral())
    write_tiny("qwen2-tiny", tiny_qwen2())
    write_tiny("qwen3-moe-tiny", tiny_qwen3_moe())
    write_tiny("olmo3-tiny", tiny_olmo3())
    write_tiny("olmo3-yarn-tiny", tiny_olmo3_yarn())
    write_released_config("olmo-3-7b", "allenai/Olmo-3-1025-7B")
    write_released_config("qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B")
    write_released_config("qwen2-0.5b", "Qwen/Qwen2-0.5B")
    write_released_config("mixtral-8x7b", "mistralai/Mixtral-8x7B-v0.1")
    write_released_config("mistral-7b-v0.3", "mistralai/Mistral-7B-v0.3")
    write_tiny("deepseek-v3-tiny", tiny_deepseek_v3())
    write_tiny("deepseek-v32-tiny", tiny_deepseek_v32(), seed=DEEPSEEK_V32_SEED)
    write_diffusion_tiny("llada-tiny", LladaTiny(), LLADA_TINY_CONFIG)
    write_diffusion_tiny("dream-tiny", DreamTiny(), DREAM_TINY_CONFIG)
    write_siglip_tiny()
    write_llama4_vision_tiny()
    write_gemma4_vision_tiny()
    write_qwen35_vision_tiny()
    write_gemma3_mm_tiny()
    write_llama4_mm_tiny()
    write_gemma4_mm_tiny()
    write_qwen35_mm_tiny()
    write_diffusion_sc_tiny()
    write_diffusion_denoiser_tiny()
    write_released_config("diffusiongemma-26b", "google/diffusiongemma-26B-A4B-it")
    write_released_config("llada-8b", "GSAI-ML/LLaDA-8B-Base")
    write_released_config("dream-7b", "Dream-org/Dream-v0-Base-7B")

    real = FIXTURES / "qwen3-0.6b"
    real.mkdir(parents=True, exist_ok=True)
    write_tensor_table(real)
    if not args.skip_real:
        write_real_reference(real)

    write_released_config("gemma3-1b", GEMMA_MIRROR)
    write_released_config("gemma-2b", "unsloth/gemma-2b")
    write_released_config("gemma-2-2b", "unsloth/gemma-2-2b")


if __name__ == "__main__":
    main()
