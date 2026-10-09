#!/usr/bin/env python3
"""Write the Hugging Face fixtures tests/test_hf_decoders.py checks against.

Everything here runs the transformers reference under torch, which dew does
not depend on. The fixtures it writes are what CI compares against. The
weight scattering and the reference run come from dew.interop.verify, which
runs the same recipe at load time for an unregistered model_type, so dew and
torch share one venv:

    uv venv /tmp/hfref --python 3.12
    uv pip install --python /tmp/hfref/bin/python torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /tmp/hfref/bin/python -e '.[torch]' safetensors \
        sentencepiece
    /tmp/hfref/bin/python tools/hf_reference.py

What lands in tests/fixtures/hf:
- <family>-tiny/ for qwen3, gemma, gemma2, gemma3, llama, llama31, mistral,
  mixtral, qwen2, qwen3-moe, olmo3, olmo3-yarn, deepseek-v3 and deepseek-v32:
  a random-weight checkpoint in the HF layout (config.json +
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
  code); tools/remote_code_reference.py writes their logits with the
  released modeling files at a pinned commit.
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
  qwen2-0.5b, qwen3-30b-a3b, olmo-3-7b, llama-3.1-8b, llada-8b, dream-7b,
  and kimi-k3-dspark, the speculative drafter the qwen3 family refuses):
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
    AutoModelForCausalLM, AutoTokenizer, BloomConfig, BloomForCausalLM, DeepseekV3Config, DeepseekV3ForCausalLM,
    Gemma2Config, Gemma2ForCausalLM, Gemma3Config, Gemma3ForCausalLM, Gemma3TextConfig,
    GemmaConfig, GemmaForCausalLM, GPTNeoConfig, GPTNeoForCausalLM, LlamaConfig, LlamaForCausalLM,
    PhiConfig, PhiForCausalLM, Phi3Config, Phi3ForCausalLM,
    FalconConfig, FalconForCausalLM,
    GPTJConfig, GPTJForCausalLM, GPTBigCodeConfig, GPTBigCodeForCausalLM, Starcoder2Config,
    Starcoder2ForCausalLM,
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
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration, Qwen3_5VisionModel,
)

from dew.interop.verify import BATCH, probe_ids, reference_logits, scatter_weights

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
NEMOTRON_H_CONFIGS = (
    ("nemotron-h-4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"),
    ("nemotron-h-30b-a3b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"),
)
BLOOM_CONFIG = ('bloom-560m', 'bigscience/bloom-560m', 'ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971')
BLOOMZ_CONFIG = ('bloomz-560m', 'bigscience/bloomz-560m', 'a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05')


def tiny_bloom() -> BloomForCausalLM:
    """Three heads exercise the non-power-of-two ALiBi slopes and fused qkv."""
    torch.manual_seed(0)
    return BloomForCausalLM(BloomConfig(
        vocab_size=64, hidden_size=24, n_layer=2, n_head=3,
        layer_norm_epsilon=3e-5, hidden_dropout=0., attention_dropout=0.,
        bos_token_id=1, eos_token_id=None, pad_token_id=0))


GPT_NEO_CONFIG = ('gpt-neo-125m', 'EleutherAI/gpt-neo-125m', '21def0189f5705e2521767faed922f1f15e7d7db')


def tiny_gpt_neo() -> GPTNeoForCausalLM:
    """Local/global attention without a logit scale, learned positions and an output bias."""
    torch.manual_seed(0)
    return GPTNeoForCausalLM(GPTNeoConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_layers=2, num_heads=4,
        attention_types=[[['global', 'local'], 1]], window_size=4, max_position_embeddings=48,
        layer_norm_epsilon=3e-5, resid_dropout=0., embed_dropout=0., attention_dropout=0.,
        bos_token_id=1, eos_token_id=None, pad_token_id=0))


PHI_CONFIG = ('phi-2', 'microsoft/phi-2', '810d367871c1d460086d9f82db8696f2e0a0fcd0')


def tiny_phi() -> PhiForCausalLM:
    """Biased one-norm parallel branches, partial rotary, GQA and an affine head."""
    torch.manual_seed(0)
    return PhiForCausalLM(PhiConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=48,
        layer_norm_eps=3e-5, resid_pdrop=0., embd_pdrop=0., attention_dropout=0.,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': .5},
        bos_token_id=1, eos_token_id=None, pad_token_id=0))


FALCON_CONFIG = ('falcon-7b', 'tiiuae/falcon-7b', 'ec89142b67d748a1865ea4451372db8313ada0d8')


def tiny_falcon(multi_query: bool = True) -> FalconForCausalLM:
    """A biased one-norm parallel block, with contiguous MQA or head-interleaved MHA."""
    torch.manual_seed(0)
    return FalconForCausalLM(FalconConfig(
        vocab_size=64, hidden_size=24, ffn_hidden_size=48, num_hidden_layers=2,
        num_attention_heads=3, multi_query=multi_query, bias=True, parallel_attn=True,
        max_position_embeddings=48, layer_norm_epsilon=3e-5,
        hidden_dropout=0., attention_dropout=0.,
        bos_token_id=1, eos_token_id=None, pad_token_id=0))


GPTJ_CONFIG = ('gpt-j-6b', 'EleutherAI/gpt-j-6b', '47e169305d2e8376be1d31e765533382721b2cc1')
GPT_BIGCODE_CONFIG = ('gpt-bigcode-santacoder', 'bigcode/gpt_bigcode-santacoder',
                      '291931872cae83498cf984b16319f47f5e9e7a07')
STARCODER2_CONFIG = ('starcoder2-3b', 'bigcode/starcoder2-3b', '733247c55e3f73af49ce8e9c7949bf14af205928')


def tiny_gpt_bigcode(multi_query: bool = True) -> GPTBigCodeForCausalLM:
    """GPT-2's block under torch Linears, its fused c_attn one key and value
    head wide (multi-query) or every head's q, k and v side by side."""
    torch.manual_seed(0)
    return GPTBigCodeForCausalLM(GPTBigCodeConfig(
        vocab_size=64, n_embd=24, n_inner=48, n_layer=2, n_head=3, n_positions=48, multi_query=multi_query,
        layer_norm_epsilon=3e-5, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
        bos_token_id=1, eos_token_id=None, pad_token_id=0))


def tiny_starcoder2() -> Starcoder2ForCausalLM:
    """Grouped-query rotary attention under a window shorter than the
    sequence, biased LayerNorms and an ungated biased feed-forward."""
    torch.manual_seed(0)
    return Starcoder2ForCausalLM(Starcoder2Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=48, sliding_window=5, norm_epsilon=3e-5, use_bias=True,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0}, residual_dropout=0.,
        embedding_dropout=0., attention_dropout=0., bos_token_id=1, eos_token_id=None, pad_token_id=0))


def tiny_gptj() -> GPTJForCausalLM:
    """Interleaved partial rotary, one-norm parallel residual and an affine head."""
    torch.manual_seed(0)
    return GPTJForCausalLM(GPTJConfig(
        vocab_size=64, n_embd=32, n_inner=48, n_layer=2, n_head=4, rotary_dim=4, n_positions=48,
        layer_norm_epsilon=3e-5, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
        bos_token_id=1, eos_token_id=None, pad_token_id=0))


PHI3_CONFIGS = (
    ('phi3-mini-4k', 'microsoft/Phi-3-mini-4k-instruct', 'f39ac1d28e925b323eae81227eaba4464caced4e'),
    ('phi4-mini', 'microsoft/Phi-4-mini-instruct', 'cfbefacb99257ffa30c83adab238a50856ac3083'),
)


def tiny_phi3() -> Phi3ForCausalLM:
    """Unequal fused qkv widths, a window and two nontrivial LongRoPE tables."""
    torch.manual_seed(0)
    return Phi3ForCausalLM(Phi3Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=48,
        original_max_position_embeddings=8, sliding_window=5, rms_norm_eps=3e-5,
        rope_parameters={'rope_type': 'longrope', 'rope_theta': 10000.,
                         'short_factor': [1.1, 1.3, 1.5, 1.7], 'long_factor': [2., 3., 4., 5.],
                         'original_max_position_embeddings': 8, 'factor': 6.},
        resid_pdrop=0., embd_pdrop=0., attention_dropout=0.,
        bos_token_id=1, eos_token_id=None, pad_token_id=0))


def write_classic_tiny(name: str, model: PreTrainedModel) -> None:
    """Same-weight fp32/float64 logits, left-padding and greedy continuation.

    Phi-3's crossing oracle is its uncached full-prefix forward. Its pinned
    generate drops the prefix at that crossing (transformers issue #49334),
    so that output is recorded separately from the intended continuation.
    """
    from unittest.mock import patch

    from diffusers_wan_reference import float64

    model.config._attn_implementation = 'eager'
    write_tiny(name, model)
    directory = FIXTURES / name
    ids = np.load(directory / 'input_ids.npy')
    mask = np.ones_like(ids, bool)
    mask[1, :3] = False
    padded = np.where(mask, ids, 0)
    positions = np.maximum(np.cumsum(mask, axis=-1) - 1, 0)
    positioned = ({} if model.config.model_type == 'bloom' else
                  {'position_ids': torch.from_numpy(positions).long()})
    np.save(directory / 'padded_ids.npy', padded)
    np.save(directory / 'attention_mask.npy', mask)
    phi3 = model.config.model_type == 'phi3'
    model.eval()
    with torch.no_grad():
        generated = model.generate(torch.from_numpy(ids[:, :4]).long(), do_sample=False,
                                   max_new_tokens=6, eos_token_id=None, pad_token_id=0).numpy()
        logits = model(torch.from_numpy(padded).long(), attention_mask=torch.from_numpy(mask),
                       use_cache=False, **positioned).logits.numpy()
        if phi3:
            np.save(directory / 'transformers_generated.npy', generated)
            generated = ids[:, :4].copy()
            scores = []
            for _ in range(6):
                score = model(torch.from_numpy(generated).long(), use_cache=False).logits[:, -1].numpy()
                scores.append(score)
                generated = np.concatenate((generated, score.argmax(-1)[:, None]), axis=1)
            np.save(directory / 'generation_logits.npy', np.stack(scores, axis=1))
            np.save(directory / 'short_logits.npy',
                    model(torch.from_numpy(ids[:, :4]).long(), use_cache=False).logits.numpy())
    np.save(directory / 'generated.npy', generated)
    np.save(directory / 'padded_logits.npy', logits)
    affine = model.config.model_type in ('phi', 'gptj')
    if affine:
        full = torch.from_numpy(np.load(directory / 'logits.npy'))
        losses = torch.nn.functional.cross_entropy(
            full[:, :-1].transpose(1, 2), torch.from_numpy(ids[:, 1:]).long(), reduction='none')
        np.save(directory / 'losses.npy', losses.numpy())
    tensor, full = torch.tensor, torch.full

    def wide_tensor(*args, dtype=None, **kwargs):
        return tensor(*args, dtype=torch.float64 if dtype == torch.float32 else dtype, **kwargs)

    def wide_full(*args, dtype=None, **kwargs):
        return full(*args, dtype=torch.float64 if dtype == torch.float32 else dtype, **kwargs)

    with (float64(), patch.object(torch, 'tensor', wide_tensor),
          patch.object(torch, 'full', wide_full), torch.no_grad()):
        # Rebuild frequency/sinusoidal buffers in float64 rather than
        # widening an already-rounded fp32 table. Parameters remain the
        # same checkpoint values, loaded into the wider reference model.
        truth_model = AutoModelForCausalLM.from_config(
            model.config, dtype=torch.float64, attn_implementation='eager').eval()
        truth_model.load_state_dict(model.state_dict())
        truth = truth_model(torch.from_numpy(ids).long(), use_cache=False).logits.double().numpy()
        if affine:
            losses = torch.nn.functional.cross_entropy(
                torch.from_numpy(truth[:, :-1]).transpose(1, 2),
                torch.from_numpy(ids[:, 1:]).long(), reduction='none')
            np.save(directory / 'losses_f64.npy', losses.numpy())
        padded_truth = truth_model(
            torch.from_numpy(padded).long(), attention_mask=torch.from_numpy(mask),
            use_cache=False, **positioned).logits.double().numpy()
        if phi3:
            np.save(directory / 'short_logits_f64.npy',
                    truth_model(torch.from_numpy(ids[:, :4]).long(), use_cache=False).logits.double().numpy())
            scores = [truth_model(torch.from_numpy(generated[:, :4 + step]).long(),
                                  use_cache=False).logits[:, -1].double().numpy() for step in range(6)]
            np.save(directory / 'generation_logits_f64.npy', np.stack(scores, axis=1))
    np.save(directory / 'logits_f64.npy', truth)
    np.save(directory / 'padded_logits_f64.npy', padded_truth)


MINIMAX_M2_CONFIGS = (
    ('minimax-m2', 'MiniMaxAI/MiniMax-M2', '757303d492a50514c312788b5247a4f696a4c6a3'),
    ('minimax-m2.5', 'MiniMaxAI/MiniMax-M2.5', 'f710177d938eff80b684d42c5aa84b382612f21f'),
    ('minimax-m2.7', 'MiniMaxAI/MiniMax-M2.7', 'd494266a4affc0d2995ba1fa35c8481cbd84294b'),
)


def write_minimax_m2() -> None:
    """The fixed MiniMax rotary port, saved under the released per-expert names."""
    from diffusers_wan_reference import float64
    from safetensors.torch import save_file
    import transformers
    from transformers import MiniMaxM2Config, MiniMaxM2ForCausalLM

    if transformers.__version__ != '5.18.0':
        raise ValueError('MiniMax-M2 requires transformers 5.18.0 (rotary fix PR 48486)')
    torch.manual_seed(0)
    model = MiniMaxM2ForCausalLM(MiniMaxM2Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=24, num_local_experts=4,
        num_experts_per_tok=2, rotary_dim=8, max_position_embeddings=64,
        rope_theta=5000000.0, rms_norm_eps=1e-6, tie_word_embeddings=False,
        bos_token_id=1, eos_token_id=2))
    model.set_experts_implementation('eager')
    write_tiny('minimax-m2-tiny', model, padded=True)
    directory = FIXTURES / 'minimax-m2-tiny'
    ids = torch.from_numpy(np.load(directory / 'input_ids.npy')).long()
    with float64(), torch.no_grad():
        truth = model.double()(input_ids=ids, use_cache=False).logits.numpy()
    np.save(directory / 'logits_f64.npy', truth)
    weights = {}
    for name, value in model.float().state_dict().items():
        if '.mlp.experts.' in name:
            stem, projection = name.split('.mlp.experts.')
            parts = value.chunk(2, dim=1) if projection == 'gate_up_proj' else (value,)
            names = ('w1', 'w3') if projection == 'gate_up_proj' else ('w2',)
            for part, target in zip(parts, names, strict=True):
                for index, expert in enumerate(part):
                    weights[f'{stem}.block_sparse_moe.experts.{index}.{target}.weight'] = expert.contiguous()
        else:
            weights[name.replace('.mlp.', '.block_sparse_moe.')] = value.contiguous()
    save_file(weights, str(directory / 'model.safetensors'))
    (directory / 'model.safetensors').chmod(0o644)
    write_minimax_m2_fp8(model.config.to_dict(), weights, ids)
    for name, repo, revision in MINIMAX_M2_CONFIGS:
        write_released_config(name, repo, revision)


def write_minimax_m2_fp8(config: dict, weights: dict, ids: torch.Tensor) -> None:
    """Run Transformers' CPU dequantization on the released FP8 storage layout."""
    from diffusers_wan_reference import float64
    from safetensors.torch import save_file
    from transformers import MiniMaxM2ForCausalLM, FineGrainedFP8Config

    directory = FIXTURES / 'minimax-m2-fp8-tiny'
    directory.mkdir(parents=True, exist_ok=True)
    quantization = {'quant_method': 'fp8', 'fmt': 'float8_e4m3fn',
                    'activation_scheme': 'dynamic', 'weight_block_size': [128, 128],
                    'modules_to_not_convert': ['gate', 'e_score_correction_bias', 'lm_head']}
    packed = {}
    for name, value in weights.items():
        if value.ndim != 2 or name in ('model.embed_tokens.weight', 'lm_head.weight') or '.gate.' in name:
            packed[name] = value
            continue
        # Toy matrices occupy one partial 128x128 tile. Scales differ across
        # projections and experts, and the reference dequantizes the same codes.
        scale = value.abs().max() / torch.finfo(torch.float8_e4m3fn).max
        packed[name] = (value / scale).to(torch.float8_e4m3fn)
        packed[name.removesuffix('.weight') + '.weight_scale_inv'] = scale.reshape(1, 1)
    save_file(packed, str(directory / 'model.safetensors'))
    (directory / 'config.json').write_text(json.dumps({**config, 'quantization_config': quantization}, indent=1))
    model = MiniMaxM2ForCausalLM.from_pretrained(
        str(directory), dtype=torch.float32, quantization_config=FineGrainedFP8Config(dequantize=True))
    model.set_experts_implementation('eager')
    model.set_attn_implementation('eager')
    model.eval()
    np.save(directory / 'input_ids.npy', ids.numpy().astype(np.int32))
    with torch.no_grad():
        np.save(directory / 'logits.npy', model(input_ids=ids, use_cache=False).logits.numpy())
    with float64(), torch.no_grad():
        np.save(directory / 'logits_f64.npy', model.double()(input_ids=ids, use_cache=False).logits.numpy())
    (directory / 'model.safetensors').chmod(0o644)


def write_qwen2_moe() -> None:
    """Three mixed-window layers, two routed layers and a gated shared expert."""
    from diffusers_wan_reference import float64
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    torch.manual_seed(0)
    model = Qwen2MoeForCausalLM(Qwen2MoeConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=48, moe_intermediate_size=16,
        shared_expert_intermediate_size=24, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=False, decoder_sparse_step=1, mlp_only_layers=[1],
        qkv_bias=True, use_sliding_window=True, sliding_window=4, max_window_layers=3,
        max_position_embeddings=64, rope_theta=1000000.0, tie_word_embeddings=False))
    model.set_experts_implementation('eager')
    write_tiny('qwen2-moe-tiny', model, padded=True)
    directory = FIXTURES / 'qwen2-moe-tiny'
    ids = torch.from_numpy(np.load(directory / 'input_ids.npy')).long()
    with float64(), torch.no_grad():
        np.save(directory / 'logits_f64.npy', model.double()(input_ids=ids, use_cache=False).logits.numpy())
    (directory / 'model.safetensors').chmod(0o644)
    write_released_config('qwen1.5-moe-a2.7b', 'Qwen/Qwen1.5-MoE-A2.7B',
                          '1a758c50ecb6350748b9ce0a99d2352fd9fc11c9')


def write_granitemoe() -> None:
    """Granite's four non-unit multipliers and the released packed expert layout."""
    from diffusers_wan_reference import float64
    from safetensors.torch import save_file
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

    torch.manual_seed(0)
    model = GraniteMoeForCausalLM(GraniteMoeConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=24, num_local_experts=4,
        num_experts_per_tok=2, max_position_embeddings=64, rope_theta=10000.0,
        embedding_multiplier=12.0, residual_multiplier=0.22, logits_scaling=6.0,
        attention_multiplier=0.015625, tie_word_embeddings=True, bos_token_id=0, eos_token_id=0))
    model.set_experts_implementation('eager')
    write_tiny('granitemoe-tiny', model, padded=True)
    directory = FIXTURES / 'granitemoe-tiny'
    ids = torch.from_numpy(np.load(directory / 'input_ids.npy')).long()
    with float64(), torch.no_grad():
        np.save(directory / 'logits_f64.npy', model.double()(input_ids=ids, use_cache=False).logits.numpy())
    weights = {}
    for name, value in model.float().state_dict().items():
        name = name.replace('.block_sparse_moe.experts.gate_up_proj', '.block_sparse_moe.input_linear.weight')
        name = name.replace('.block_sparse_moe.experts.down_proj', '.block_sparse_moe.output_linear.weight')
        name = name.replace('.block_sparse_moe.router.weight', '.block_sparse_moe.router.layer.weight')
        if name != 'lm_head.weight':
            weights[name] = value.contiguous()
    save_file(weights, str(directory / 'model.safetensors'))
    (directory / 'model.safetensors').chmod(0o644)
    write_released_config('powermoe-3b', 'ibm-research/PowerMoE-3b',
                          '13fcb5a98001438bed01cf1ac4b423751dc4c2ea')


MODERNBERT_CONFIGS = (
    ('modernbert-base', 'answerdotai/ModernBERT-base', '8949b909ec900327062f0ebf497f51aef5e6f0c8',
     'config.json'),
    ('laya-encoder', 'convaiinnovations/laya', '7b928d828b7b0e022f929d9bd2e44165aa270148',
     'encoder/config.json'),
)


def write_modernbert_tiny(name: str = 'modernbert-tiny') -> None:
    """ModernBertForMaskedLM's logits and ModernBertModel's states, fp32 and float64.

    Four layers give the global, local, local, global pattern, and a local
    span of 6 (three keys either side) is shorter than the 12-token probe, so
    a window applied on one side only, or not at all, shows in the states.
    Beside the plain rows: right-padded rows with their validity mask, and
    rows packing two documents, whose reference is each document run alone
    at positions from zero.
    """
    from diffusers_wan_reference import float64
    from transformers import ModernBertConfig, ModernBertForMaskedLM

    torch.manual_seed(0)
    model = ModernBertForMaskedLM(ModernBertConfig(
        vocab_size=64, hidden_size=32, intermediate_size=24, num_hidden_layers=4,
        num_attention_heads=2, global_attn_every_n_layers=3, local_attention=6,
        global_rope_theta=160000.0, local_rope_theta=10000.0, norm_eps=3e-5,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        cls_token_id=1, sep_token_id=2))
    model.config._attn_implementation = 'eager'
    write_tiny(name, model)
    directory = FIXTURES / name
    ids = np.load(directory / 'input_ids.npy')
    mask = np.ones_like(ids, bool)
    mask[1, -3:] = False
    padded = np.where(mask, ids, 0)
    documents = np.where(np.arange(ids.shape[1]) < 5, 1, 2)[None].repeat(ids.shape[0], 0)
    np.save(directory / 'padded_ids.npy', padded)
    np.save(directory / 'attention_mask.npy', mask)
    np.save(directory / 'segment_ids.npy', documents.astype(np.int32))

    def states(dtype_scope):
        with dtype_scope, torch.no_grad():
            encoder = model.model
            plain = encoder(torch.from_numpy(ids).long()).last_hidden_state
            masked = encoder(torch.from_numpy(padded).long(),
                             attention_mask=torch.from_numpy(mask)).last_hidden_state
            packed = torch.cat([encoder(torch.from_numpy(ids[:, :5]).long()).last_hidden_state,
                                encoder(torch.from_numpy(ids[:, 5:]).long()).last_hidden_state], dim=1)
            logits = model(torch.from_numpy(ids).long()).logits
        return {'hidden': plain, 'padded_hidden': masked, 'packed_hidden': packed, 'logits': logits}

    import contextlib

    model.eval()
    for key, value in states(contextlib.nullcontext()).items():
        np.save(directory / f'{key}.npy', value.float().numpy())
    # The reference's own bf16 run, which a bf16 forward is held to, on a
    # copy: a round trip through bf16 would round the float64 run's weights.
    import copy

    with torch.no_grad():
        narrow = copy.deepcopy(model).to(torch.bfloat16)
        np.save(directory / 'hidden_bf16.npy',
                narrow.model(torch.from_numpy(ids).long()).last_hidden_state.float().numpy())
    with float64():
        model.double()
        # The inverse frequencies are buffers built at construction, in float32.
        rotary = model.model.rotary_emb
        for layer_type in rotary.layer_types:
            inverse, _ = rotary.compute_default_rope_parameters(model.config, layer_type=layer_type)
            setattr(rotary, f'{layer_type}_inv_freq', inverse.double())
        truths = states(contextlib.nullcontext())
    for key, value in truths.items():
        np.save(directory / f'{key}_f64.npy', value.double().numpy())


def tiny_qwen3() -> Qwen3ForCausalLM:
    config = Qwen3Config.from_dict(dict(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
        tie_word_embeddings=True, rope_theta=1e6, max_position_embeddings=64,
        rms_norm_eps=1e-6, attention_bias=False, hidden_act="silu"))
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config)


def write_nemotron_h(*, moe: bool = False, latent: bool = False) -> None:
    """Two grouped SSD blocks, two dense or routed ReLU² blocks and positionless GQA.

    Four-token chunks cross two boundaries on the twelve-token probe. The
    dt floor is active, the SSD projections have biases, and expand does
    not determine the inner width. Float64 widens the reference's explicit
    fp32 casts in the SSD, norms, softmax and head as well as its parameters.
    MoE uses nonzero selection bias, two routing groups and an ungated shared
    expert; the latent case leaves top-k weights unnormalized.
    """
    from diffusers_wan_reference import float64
    from unittest.mock import patch
    import transformers

    torch.manual_seed(0)
    name = "nemotron-h-moe-latent-tiny" if latent else "nemotron-h-moe-tiny" if moe else "nemotron-h-tiny"
    config = NemotronHConfig(
        vocab_size=64, hidden_size=16,
        layers_block_type=(["linear_attention", "moe", "full_attention", "moe", "linear_attention"] if moe else
                          ["linear_attention", "mlp", "full_attention", "linear_attention", "mlp"]),
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, intermediate_size=32,
        mamba_num_heads=4, mamba_head_dim=8, ssm_state_size=4, n_groups=2,
        conv_kernel=4, chunk_size=4, time_step_min=0.7, expand=3, use_bias=True,
        mlp_hidden_act="relu2", layer_norm_epsilon=3e-5, max_position_embeddings=64,
        tie_word_embeddings=False, use_mamba_kernels=False)
    if moe:
        config.n_routed_experts, config.num_experts_per_tok = 8, 2
        config.moe_intermediate_size, config.moe_shared_expert_intermediate_size = 16, 24
        config.n_group, config.topk_group, config.routed_scaling_factor = 2, 1, 2.5
        config.moe_latent_size = 8 if latent else None
        config.norm_topk_prob = not latent
        config._experts_implementation = "eager"
    model = NemotronHForCausalLM(config)
    write_tiny(name, model, seed=2024)
    directory = FIXTURES / name
    ids = torch.from_numpy(np.load(directory / "input_ids.npy")).long()
    type_ = torch.Tensor.type

    def wide_type(tensor, dtype=None, **kwargs):
        # Nemotron's router pins .type(torch.float32), beyond the shared cast widener.
        return type_(tensor, torch.float64 if dtype == torch.float32 else dtype, **kwargs)

    with float64(), patch.object(torch.Tensor, "type", wide_type), torch.no_grad():
        truth = model.double()(input_ids=ids, use_cache=False).logits.double().numpy()
    np.save(directory / "logits_f64.npy", truth)
    (directory / "source.json").write_text(json.dumps({
        "transformers": {"version": transformers.__version__,
                         "revision": "93c8b7b485963a10800c91f55304db6be211c2bd"},
        "seed": 2024,
    }, indent=1) + "\n")
    (directory / "model.safetensors").chmod(0o644)


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
    np.save(directory / "projector_ref.npy", system["soft"])
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


def gemma4_vision_tiny_config(head_dim: int = 8) -> Gemma4VisionConfig:
    """Two layers of width 32 over a 4x4 patch grid pooled by 2 into four
    soft tokens, with one grouped-query repeat and the released
    standardization on."""
    return Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
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


def gemma4_vision_tiny_system(seed: int = 1234, *, head_dim: int = 8, device: str = 'cpu'):
    """A tiny Gemma 4 vision trunk and multimodal embedder with scattered
    weights, on patchified pixels as the processor emits them.

    Returns the torch modules with fp32 reference outputs on fixed patches,
    and the configs that describe them. The 32-pixel image patchifies into
    a 4x4 grid pooled to four soft tokens. The standardization buffers ride
    the state dict, so they are scattered with the weights.
    """
    from types import SimpleNamespace

    vconf = gemma4_vision_tiny_config(head_dim)
    torch.manual_seed(seed)
    tower = Gemma4VisionModel(vconf)
    scatter_weights(tower, seed)
    with torch.no_grad():
        tower.std_bias.copy_(torch.randn(32) * 0.5)
        tower.std_scale.copy_(1.0 + torch.randn(32) * 0.05)
    tower = tower.float().eval().to(device)
    pixels = np.random.RandomState(11).rand(BATCH, 16, 192).astype(np.float32)
    positions = gemma4_positions(4, BATCH)
    with torch.no_grad():
        last = tower(pixel_values=torch.from_numpy(pixels).to(device),
                     pixel_position_ids=torch.from_numpy(positions).to(device),
                     return_dict=True).last_hidden_state.to(torch.float32).cpu().numpy()
    # The trunk strips padding with a boolean mask, which flattens the batch;
    # the fixture has no padding, so the reshape back is the same tokens.
    last = last.reshape(BATCH, -1, vconf.hidden_size)
    projector = Gemma4MultimodalEmbedder(
        vconf, Gemma4TextConfig(hidden_size=G4V_TEXT_WIDTH))
    scatter_weights(projector, seed + 1)
    projector = projector.float().eval().to(device)
    with torch.no_grad():
        soft = projector(torch.from_numpy(last).to(device)).to(torch.float32).cpu().numpy()
    return {"tower": tower, "projector": projector, "vconf": vconf,
            "pixels": pixels, "positions": positions, "last": last,
            "soft": soft}


def write_gemma4_vision_tiny(*, head_dim: int = 8, device: str = 'cpu',
                           name: str = 'gemma4-vision-tiny') -> None:
    """The Gemma 4 trunk and embedder as Dew reads them: bare tensor names,
    the tower config, the text width, fixed patches and positions, and both
    fp32 reference outputs."""
    from safetensors.torch import save_file

    system = gemma4_vision_tiny_system(head_dim=head_dim, device=device)
    directory = FIXTURES / name
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


def write_diffusion_window_tiny() -> None:
    """One DiffusionGemma denoise step past its sliding window: a sliding then
    a full layer at a window of 4, a prompt of 6 and a canvas of 6, so the
    decoder's sliding layer reads only the last 3 cached prompt keys while
    every canvas key reads every other (modeling_diffusion_gemma.py:1399-1401).
    The fp32 and float64 reference logits, the latter for the float64 rule.
    """
    from diffusers_wan_reference import float64
    from safetensors.torch import save_file
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaConfig, DiffusionGemmaTextConfig,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaForBlockDiffusion,
    )
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    fields = {**DIFFUSION_DENOISER_TEXT, "sliding_window": 4,
              "layer_types": ["sliding_attention", "full_attention"]}
    text = DiffusionGemmaTextConfig(**{key: value for key, value in fields.items() if key != "model_type"})
    vision = Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16, patch_size=8,
        pooling_kernel_size=2, position_embedding_size=64)
    torch.manual_seed(0)
    model = DiffusionGemmaForBlockDiffusion(DiffusionGemmaConfig(
        text_config=text, vision_config=vision, canvas_length=6, tie_word_embeddings=False))
    state = model.state_dict()
    for name in list(state):
        if name.startswith("model.decoder.layers.") or name in (
                "model.decoder.embed_tokens.weight", "model.decoder.norm.weight"):
            state[name].copy_(state[name.replace("model.decoder.", "model.encoder.language_model.", 1)])
    model.load_state_dict(state)
    model = model.float().eval()
    # The experts' loop, since the grouped matmul has no float64 kernel.
    for module in model.modules():
        if hasattr(module, "experts") and hasattr(module.experts, "config"):
            module.experts.config._experts_implementation = "eager"
    directory = FIXTURES / "diffusion-gemma-window-tiny"
    directory.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor for name, tensor in model.state_dict().items()
               if name.startswith(("model.encoder.language_model.", "model.decoder.")) or name == "lm_head.weight"},
              directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(fields, indent=1) + "\n")
    prompt = np.array([[2, 5, 7, 9, 11, 13]], np.int32)
    canvas = np.array([[3, 4, 5, 6, 8, 10]], np.int32)

    def logits():
        with torch.no_grad():
            return model(input_ids=torch.from_numpy(prompt).long(),
                         decoder_input_ids=torch.from_numpy(canvas).long()).logits

    np.save(directory / "prompt.npy", prompt)
    np.save(directory / "canvas.npy", canvas)
    np.save(directory / "logits.npy", logits().to(torch.float32).numpy())
    with float64():
        model.double()
        np.save(directory / "logits_f64.npy", logits().double().numpy())


def write_tiny(name: str, model: PreTrainedModel, seed: int = 1234, *, padded: bool = False) -> None:
    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)
    scatter_weights(model, seed)
    model = model.float()
    model.save_pretrained(directory, safe_serialization=True)

    ids = probe_ids(model.config.vocab_size)
    np.save(directory / "input_ids.npy", ids)
    np.save(directory / "logits.npy", reference_logits(model, ids))
    if padded:
        from copy import deepcopy
        from diffusers_wan_reference import float64

        tokens = torch.from_numpy(ids.astype(np.int64))
        valid = torch.ones_like(tokens)
        tokens[1, :4] = 0
        valid[1, :4] = 0
        positions = (valid.cumsum(-1) - 1).clamp(min=0)
        inputs = {'input_ids': tokens, 'attention_mask': valid, 'position_ids': positions, 'use_cache': False}
        with torch.no_grad():
            reference = model(**inputs).logits.numpy()
        with float64(), torch.no_grad():
            truth = deepcopy(model).double()(**inputs).logits.numpy()
        np.savez(directory / 'padded.npz', input_ids=tokens.numpy().astype(np.int32),
                 attention_mask=valid.numpy().astype(bool), position_ids=positions.numpy().astype(np.int32),
                 logits=reference, logits_f64=truth)
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


def write_released_config(name: str, repo: str, revision: str | None = None,
                          filename: str = "config.json") -> None:
    """The real config.json of a released checkpoint, and the repo it came
    from in source.json. Only the config is downloaded, never the weights.

    Google's Gemma repos answer 401 without an accepted licence, so those
    come from unsloth's mirrors, which carry the identical config plus
    marker keys of their own (unsloth_fixed, unsloth_version); the markers
    are dropped so the fixture is the released config alone.
    """
    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(hf_hub_download(repo, filename, revision=revision)).read_text())
    for key in [key for key in config if key.startswith("unsloth")]:
        del config[key]
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    source = {"repo": repo, **({"revision": revision} if revision is not None else {}),
              **({"filename": filename} if filename != "config.json" else {})}
    (directory / "source.json").write_text(json.dumps(source) + "\n")
    layers = config.get('num_hidden_layers', config.get('n_layers', config.get('num_layers')))
    print(f"{directory / 'config.json'}: {repo}, "
          f"{layers} layers, {len(config)} fields")


def write_classic_family(family: str) -> None:
    """One classic decoder's tiny fixtures and its pinned released configs."""
    if family == 'bloom':
        write_classic_tiny('bloom-tiny', tiny_bloom())
        write_released_config(*BLOOM_CONFIG)
        write_released_config(*BLOOMZ_CONFIG)
        return
    if family == 'gpt_neo':
        write_classic_tiny('gpt-neo-tiny', tiny_gpt_neo())
        write_released_config(*GPT_NEO_CONFIG)
        return
    if family == 'phi':
        write_classic_tiny('phi-tiny', tiny_phi())
        write_released_config(*PHI_CONFIG)
        return
    if family == 'falcon':
        write_classic_tiny('falcon-tiny', tiny_falcon())
        write_classic_tiny('falcon-mha-tiny', tiny_falcon(multi_query=False))
        write_released_config(*FALCON_CONFIG)
        return
    if family == 'gptj':
        write_classic_tiny('gptj-tiny', tiny_gptj())
        write_released_config(*GPTJ_CONFIG)
        return
    if family == 'gpt_bigcode':
        write_classic_tiny('gpt-bigcode-tiny', tiny_gpt_bigcode())
        write_classic_tiny('gpt-bigcode-mha-tiny', tiny_gpt_bigcode(multi_query=False))
        write_released_config(*GPT_BIGCODE_CONFIG)
        return
    if family == 'starcoder2':
        write_classic_tiny('starcoder2-tiny', tiny_starcoder2())
        write_released_config(*STARCODER2_CONFIG)
        return
    if family == 'phi3':
        write_classic_tiny('phi3-tiny', tiny_phi3())
        for config in PHI3_CONFIGS:
            write_released_config(*config)
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-real", action="store_true",
                        help="only the tiny fixtures, no 1.5 GB download")
    parser.add_argument("--nemotron-h-only", action="store_true",
                        help="only the Nemotron-H tiny fixture and two pinned released configs")
    parser.add_argument('--classic-family', choices=('bloom', 'gpt_neo', 'phi', 'phi3', 'falcon', 'gptj',
                                                     'gpt_bigcode', 'starcoder2'),
                        help='only this classic decoder fixture and its pinned released configs')
    parser.add_argument('--minimax-m2-only', action='store_true',
                        help='only MiniMax-M2 with transformers 5.18.0 and its pinned configs')
    parser.add_argument('--qwen2-moe-only', action='store_true',
                        help='only Qwen2-MoE and its pinned released config')
    parser.add_argument('--granitemoe-only', action='store_true',
                        help='only Granite MoE and its pinned PowerMoE config')
    parser.add_argument('--encoder-family', choices=('modernbert',),
                        help='only this encoder fixture and its pinned released configs')
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.classic_family is not None:
        write_classic_family(args.classic_family)
        return
    if args.minimax_m2_only:
        write_minimax_m2()
        return
    if args.qwen2_moe_only:
        write_qwen2_moe()
        return
    if args.granitemoe_only:
        write_granitemoe()
        return
    if args.encoder_family == 'modernbert':
        write_modernbert_tiny()
        for config in MODERNBERT_CONFIGS:
            write_released_config(*config)
        return
    write_nemotron_h()
    write_nemotron_h(moe=True)
    write_nemotron_h(moe=True, latent=True)
    for name, repo, revision in NEMOTRON_H_CONFIGS:
        write_released_config(name, repo, revision)
    if args.nemotron_h_only:
        return
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
    write_released_config("kimi-k3-dspark", "RadixArk/Kimi-K3-DSpark")
    write_released_config("qwen2-0.5b", "Qwen/Qwen2-0.5B")
    write_released_config("mixtral-8x7b", "mistralai/Mixtral-8x7B-v0.1")
    write_released_config("mistral-7b-v0.3", "mistralai/Mistral-7B-v0.3")
    write_tiny("deepseek-v3-tiny", tiny_deepseek_v3())
    write_tiny("deepseek-v32-tiny", tiny_deepseek_v32(), seed=DEEPSEEK_V32_SEED)
    write_siglip_tiny()
    write_llama4_vision_tiny()
    write_gemma4_vision_tiny()
    write_gemma4_vision_tiny(head_dim=16, name="gemma4-vision-wide-tiny")
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
