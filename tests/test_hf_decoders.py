"""Hugging Face decoder checkpoints, loaded into CausalTransformer and back out.

The claim these tests defend is parity, not plausibility: the same weights and
the same token ids through the reference implementation and through dew have to
produce the same logits. tools/hf_reference.py writes the fixtures under torch
and transformers (the reference), including two tiny random-weight checkpoints
whose logits are committed, so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- gemma-tiny  : max |logit difference| 4.77e-06, tolerance 1e-4, logits up to
  7.0; Gemma 1 with hidden_act 'gelu', the erf form.
- gemma2-tiny : max |logit difference| 4.05e-06, tolerance 1e-4, logits up to
  10; alternating sliding and full layers, the sandwich norms without q/k
  norms, and the attention softcap at 5, which moves the logits by 1.45.
- mistral-tiny: max |logit difference| 6.44e-06, tolerance 1e-4.
- qwen2-tiny  : max |logit difference| 8.39e-06, tolerance 1e-4, logits up to
  6.7; biased q/k/v over a bias-free o_proj, with a window from layer 1 on.
- mixtral-tiny: max |logit difference| 2.32e-06, tolerance 1e-4, logits up
  to 4.4 in magnitude; the released per-expert w1/w2/w3 tensors stack.
- qwen3-moe-tiny: max |logit difference| 2.86e-06, tolerance 1e-4, logits up
  to 4.1; one routed layer of three between two dense ones, with
  norm_topk_prob off (renormalising moves the logits by 0.38).
- olmo3-tiny  : max |logit difference| 4.77e-06, tolerance 1e-4, logits up to
  6.2; the post-norm block, q/k norms over the whole projection and three
  sliding layers to one full (per-head norms of the same scale miss by 0.1).
- olmo3-yarn-tiny: max |logit difference| 4.53e-06, tolerance 1e-4, logits
  up to 5.5; allenai/Olmo-3-1025-7B's rope_scaling record (yarn, factor 8
  off 8192 pretraining positions, the explicit attention_factor) on the one
  full-attention layer of its 3:1 pattern, the three sliding layers plain
  at rope_theta. Plain rope everywhere misses by 0.35, the same YaRN on
  every layer by 1.70, and an attention_factor of 1 by 0.35.
- llama31-tiny: max |logit difference| 9.89e-06, tolerance 1e-4, logits up to
  6.4; Llama 3.1's rope_scaling (factor 8 over 64 pretraining positions, so
  the ramp moves 7 of 8 pairs), and plain rope on the same weights misses
  by 4.4.
- qwen3-tiny  : max |logit difference| 8.3e-06, tolerance 1e-4
- gemma3-tiny : max |logit difference| 3.3e-06, tolerance 1e-4
- llama-tiny  : max |logit difference| 6.1e-06, tolerance 1e-4 (untied head,
  biased projections)
- Qwen3-0.6B  : max |top-32 logit difference| 1.4e-04, mean 1.2e-05, tolerance
  5e-3, and the argmax of all 48 positions equal. The larger residue is 28
  layers of a real checkpoint accumulating fp32 rounding, not a different
  computation.
- gemma4-ple  : max |logit difference| 4.9e-07, tolerance 1e-5
- gemma4-kvshare: max |logit difference| 8.6e-07, tolerance 1e-5
- gemma4-e2b  : max |logit difference| 1.4e-06, tolerance 1e-5. An E2B-shaped
  tiny config with every Gemma 4 gap at once: partial rotary, mixed head
  dims, double-wide MLP, KV sharing, per-layer inputs and the values norm.
  The logit cap is absent because the text path never reads it.
- deepseek-v3-tiny : max |logit difference| 4.4e-06, tolerance 1e-4. MLA
  with q and kv LoRA, the released YaRN spelling, a dense layer over a
  routed one with a shared expert, the group limit and the balancing bias.
- deepseek-v32-tiny: max |logit difference| 3.6e-06, tolerance 1e-4. The
  same over the sparse indexer; the dense mixer on the same weights differs
  from the fixture by 3.8, so the fixture is the sparse model.
- qwen35-tiny : max |logit difference| 9.1e-05, tolerance 5e-4, on logits of
  magnitude 6.8. Three gated delta net layers and one gated attention layer
  with a sliced quarter-head rope; the delta net's own numbers are in
  tests/test_linear_attention.py.
- qwen3-next-tiny: max |logit difference| 2.3e-05 on the trunk and 1.8e-05
  on the MTP layer, on logits of magnitude 6.6, tolerance 1e-4 scaled by
  that magnitude. The Qwen3.5 hybrid with the delta net's projections fused
  and grouped by key head, and every layer routing softmax top-2 over eight
  experts beside a sigmoid-gated shared expert.
- gpt-oss-tiny: max |logit difference| 2.5e-06, tolerance 1e-4. Sink
  attention on a sliding and a full layer, YaRN over grouped-query heads,
  the biased router and the clamped interleaved experts; the block's own
  numbers are in tests/test_attention_sinks.py and tests/test_gpt_oss.py.
- deepseek-v2-tiny: max |logit difference| 2.3e-06, tolerance 1e-4. MLA
  without a query LoRA, V2's softmax router under group_limited_greedy and
  no renormalisation; the router's numbers are in tests/test_deepseek_v2.py.
- glm4-moe-tiny: max |logit difference| 3.3e-06 on the trunk and 3.2e-06 on
  the MTP depth, tolerance 1e-4. Biased q/k/v, a half rotary, DeepSeek V3
  routing with a shared expert, one MTP depth composed as the engines run it.
- llama4-tiny : max |logit difference| 3.5e-06, tolerance 1e-4. Chunked
  rotated local layers around a global layer with temperature tuning, the
  routed layers scaling their inputs; the mixer's numbers are in
  tests/test_llama4.py.
- gemma4-moe-tiny: max |logit difference| 4.9e-06, tolerance 1e-4. The
  routed branch beside every layer's dense MLP, the global layer reading its
  values off one key/value head of 16, and the per-layer scalars; the
  branch's numbers are in tests/test_gemma4_moe.py.
- gemma3n-tiny: max |logit difference| 4.2e-06, tolerance 1e-4. Three
  copies of the residual stream under AltUp, the LAuReL block, gaussian
  top-k on the first two layers, widths of 48 and 64, per-layer inputs and
  one sharing layer; the blocks' numbers are in tests/test_gemma3n.py.
- deepseek-v4-tiny: max |logit difference| 9.8e-06, tolerance 1e-4. mHC's
  two residual streams over all three attention kinds (sliding, compressed
  sparse with its lightning indexer, heavily compressed), the grouped
  output projection and per-head sinks, three hash-routed layers over
  three top-k ones and the gate clamp at 2.0; rolling the hash table moves
  the logits by more than 1e-2 and dropping the clamp by more than 1.
- nemotron-h-tiny uses the float64 rule. Native CPU RMS distances are
  2.64e-7 for Dew and 2.19e-7 for transformers (ratio 1.20), with max
  fp32 logit difference 1.25e-6 and equal argmax. Grouped gated SSD norms,
  a dt floor, biased SSD projections, positionless GQA and ReLU² MLPs
  live in independent pre-norm residual blocks.
- nemotron-h-moe-tiny and its latent variant use the same float64 rule.
  Native CPU ratios are 0.967 and 1.215, max fp32 logit difference 1.43e-6
  for each, with equal argmax. They carry ungated routed and shared experts,
  nonzero selection bias, grouped sigmoid routing and optional latent projections.
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import flat_tree
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.decoder_parts import DrafterRefused
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.gemma3n import AltUp
from dew.nn.hyper_connections import HyperConnections
from dew.nn.mixers import AttentionMixer
from dew.nn.mixers.mamba2 import Mamba2Mixer
from dew.nn.mixers.mlp import MLPMixer
from dew.nn.moe import Situ
from dew.registry import models

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
# The committed byte-level BPE an export can name without a download.
TOKENIZER = Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-tools"
TINY = ("qwen3-tiny", "gemma3-tiny", "llama-tiny", "mistral-tiny", "qwen2-tiny",
        "gemma-tiny", "gemma2-tiny", "olmo3-tiny", "olmo3-yarn-tiny",
        "llama31-tiny", "gpt2-tiny", "opt-tiny", "gpt-neox-tiny",
        "bloom-tiny", "gpt-neo-tiny", "phi-tiny", "falcon-tiny", "falcon-mha-tiny", "gptj-tiny", "phi3-tiny",
        "gpt-bigcode-tiny", "starcoder2-tiny", "stablelm-tiny", "stablelm-parallel-tiny", "cohere-tiny",
        "cohere2-tiny")
# A multi-head GPTBigCode's record is GPT-2's, so a weight translation from
# the record alone reads GPT-2's Conv1D layout; the loader names the source's
# family, and tests/test_hf_bigcode_starcoder2.py holds it to the reference.
CLASSIC = ('bloom-tiny', 'gpt-neo-tiny', 'phi-tiny', 'falcon-tiny', 'falcon-mha-tiny', 'gptj-tiny',
           'phi3-tiny', 'gpt-bigcode-tiny', 'starcoder2-tiny', 'stablelm-tiny', 'stablelm-parallel-tiny',
           'cohere-tiny', 'cohere2-tiny')
DEEPSEEK = ("deepseek-v3-tiny", "deepseek-v32-tiny")
ROUTED = (*DEEPSEEK, "kimi-k2-tiny", "mixtral-tiny", "qwen3-moe-tiny")
HYBRID = ("nemotron-h-tiny", "nemotron-h-moe-tiny", "nemotron-h-moe-latent-tiny")
GEMMA4_MOE = FIXTURES / "gemma4-moe-tiny"
REAL = FIXTURES / "qwen3-0.6b"


def scaled_difference(ours, theirs) -> float:
    return float(np.max(np.abs(ours - theirs)) / np.max(np.abs(theirs)))


def fixture_config(name):
    return json.loads((FIXTURES / name / "config.json").read_text())


# A sparse record uses its native owner to read every supplied field and to
# reject stale names. Copies retain that owner and the wire's explicit keys.
@pytest.mark.parametrize("fixture, path, value", [
    ("qwen3-tiny", (), CausalTransformer),
    ("gemma3-tiny", ("kinds", "sliding_attention"), LayerKind),
    ("qwen3-moe-tiny", ("mixture",), Mixture),
    ("gemma3n-tiny", ("altup",), AltUp),
    ("glm5-next-tiny", ("hyper_connections",), HyperConnections),
    ("kimi-k3-source", ("attention_residuals",), AttentionResiduals),
    ("kimi-k3-source", ("mlp",), Situ),
])
def test_a_config_record_uses_its_native_owner_after_a_copy(fixture, path, value):
    from dew.interop.config_records import NativeFields

    record = translate_config(fixture_config(fixture))
    for name in path:
        record = record[name]
    assert isinstance(record, NativeFields)
    copied = record.copy()
    assert type(copied.value) is value
    assert copied == record
    assert copied.value == value(**record)
    copied["not_a_native_field"] = None
    with pytest.raises(ValueError, match="not_a_native_field"):
        _ = copied.value
    assert "not_a_native_field" not in record


def test_registered_family_alias_preserves_its_source_when_exported(tmp_path, monkeypatch):
    from shutil import copytree

    from dew.interop import hf_decoders

    alias = "dream_registered_alias"
    family = hf_decoders.families()["dream"]
    monkeypatch.setitem(hf_decoders.families(), alias,
                        dataclasses.replace(family, model_types=(*family.model_types, alias)))
    source = copytree(FIXTURES / "dream-tiny", tmp_path / "source")
    config = fixture_config("dream-tiny")
    config["model_type"] = alias
    (source / "config.json").write_text(json.dumps(config))
    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    destination = tmp_path / "export"
    loaded.save(destination)
    assert json.loads((destination / "config.json").read_text()) == config
    restored = Pretrained.load(destination, dtype="float32", attention_impl="reference")
    ids = np.load(source / "input_ids.npy")
    np.testing.assert_array_equal(loaded.model.apply(loaded.variables, ids),
                                  restored.model.apply(restored.variables, ids))


def fp32_decoder(directory, **kwargs):
    """The fixture as a model plus variables, in fp32 on the reference kernel."""
    pretrained = Pretrained.load(str(directory), dtype='float32',
                                 attention_impl='reference', **kwargs)
    return pretrained.model, pretrained.variables


def test_llama_checkpoint_with_training_metadata_keeps_reference_logits(tmp_path):
    from shutil import copytree

    original = FIXTURES / "llama-tiny"
    directory = tmp_path / "checkpoint"
    copytree(original, directory)
    config = fixture_config("llama-tiny")
    config.update(is_llama_config=True, rope_interleaved=False)
    (directory / "config.json").write_text(json.dumps(config))
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    ids = np.load(original / "input_ids.npy")
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    expected = np.load(original / "logits.npy")
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-4, rtol=0)


def test_qwen3_config_translates_field_by_field():
    config = translate_config(fixture_config("qwen3-tiny"))

    assert config == {
        'vocab_size': 256, 'emb_features': 64, 'num_layers': 2, 'num_heads': 4,
        'num_kv_heads': 2, 'head_dim': 16, 'mlp': 'swiglu', 'mlp_features': 128,
        'max_seq_len': 64, 'rope_theta': 1e6,
        'layer_types': ('full_attention', 'full_attention'), 'kinds': {},
        'norm_eps': 1e-6, 'scale_after_cast': True, 'qk_norm': True, 'attention_bias': False,
        'tie_embeddings': True,
    }


def test_gemma3_config_carries_the_gemma_switches():
    config = translate_config(fixture_config("gemma3-tiny"))

    assert config['sandwich_norms'] and config['scale_offset']
    assert config['embedding_scale'] and config['mlp'] == 'geglu'
    assert config['qk_norm'] and config['num_kv_heads'] == 1
    # query_pre_attn_scalar 16, not the head_dim of 32
    assert config['attention_scale'] == pytest.approx(0.25)
    assert config['final_logit_softcap'] == 30.0
    assert config['layer_types'] == ('sliding_attention', 'full_attention')
    # rope_local_base_freq and the window belong to the sliding kind,
    # rope_theta to the model the full layers take it from
    assert config['rope_theta'] == 1e6
    assert config['kinds'] == {'sliding_attention': {'window': 4, 'rope_theta': 1e4}}


def test_a_multimodal_gemma3_config_is_refused():
    """Only a text decoder maps, and no published multimodal Gemma 3 has a
    text_config that would: gemma-3-4b, 12b and 27b all carry rope_scaling
    {'rope_type': 'linear', 'factor': 8}, which the field map refuses. So the
    refusal names the model_type; the text half of a checkpoint whose vision
    tower nothing here runs stays unloaded."""
    wrapped = {'model_type': 'gemma3', 'text_config': fixture_config("gemma3-tiny"),
               'vision_config': {'hidden_size': 8}, 'mm_tokens_per_image': 256,
               'boi_token_index': 255999, 'eoi_token_index': 256000,
               'image_token_index': 262144}

    with pytest.raises(ValueError, match="model_type 'gemma3'"):
        translate_config(wrapped)


def test_a_speculative_drafter_is_refused_by_design_naming_what_it_reads():
    """RadixArk/Kimi-K3-DSpark ships a SpecForge drafter under model_type
    qwen3. It reads Kimi K3's states after five of its 93 layers and has no
    embedding or head of its own, so the refusal names the drafter, its
    target layers and SGLang, where its draft arithmetic lives. Without
    `num_target_layers` the same fields are an ordinary qwen3 config that
    fails on what it cannot express."""
    config = fixture_config("kimi-k3-dspark")

    with pytest.raises(DrafterRefused, match=r"\['DSparkDraftModel'\].* 93-layer target's hidden "
                       r"states after its layers \[7, 23, 51, 67, 83\].*SGLang"):
        translate_config(config)
    del config["num_target_layers"]
    with pytest.raises(ValueError, match="config fields") as refused:
        translate_config(config)
    assert not isinstance(refused.value, DrafterRefused)


def test_released_mistral_v03_config_translates_every_computational_field():
    # v0.3 deliberately disables the window; the tiny fixture exercises it.
    assert translate_config(fixture_config("mistral-7b-v0.3")) == {
        'vocab_size': 32768, 'emb_features': 4096, 'num_layers': 32,
        'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'mlp': 'swiglu',
        'mlp_features': 14336, 'max_seq_len': 8192, 'rope_theta': 1e6,
        'layer_types': ('full_attention',) * 32, 'kinds': {},
        'norm_eps': 1e-5, 'scale_after_cast': True, 'qk_norm': False,
        'attention_bias': False, 'tie_embeddings': False,
    }


def test_released_qwen2_0_5b_config_translates_every_computational_field():
    """Qwen2-0.5B ships use_sliding_window false with a sliding_window of
    131072, so every layer attends the whole sequence."""
    assert translate_config(fixture_config("qwen2-0.5b")) == {
        'vocab_size': 151936, 'emb_features': 896, 'num_layers': 24,
        'num_heads': 14, 'num_kv_heads': 2, 'head_dim': 64, 'mlp': 'swiglu',
        'mlp_features': 4864, 'max_seq_len': 8192, 'rope_theta': 1e6,
        'layer_types': ('full_attention',) * 24, 'kinds': {},
        'norm_eps': 1e-6, 'scale_after_cast': True, 'qk_norm': False,
        'attention_bias': True, 'o_proj_bias': False, 'tie_embeddings': True,
    }


def test_qwen2_loads_its_projection_biases_and_no_o_proj_bias():
    """The split dial in the loaded tree: zeroing the q/k/v biases moves the
    reference logits, and o_proj has no bias leaf to zero."""
    model, variables = fp32_decoder(FIXTURES / 'qwen2-tiny')
    ids = np.load(FIXTURES / 'qwen2-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'qwen2-tiny' / 'logits.npy')
    attention = variables['params']['layers_0']['self_attn']
    assert 'bias' not in attention['o_proj']
    unbiased = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf) if path[-1].key == 'bias' else leaf,
        variables)
    assert np.max(np.abs(np.asarray(model.apply(unbiased, ids)) - reference)) > 0.1


def test_released_mixtral_8x7b_config_translates_every_computational_field():
    config = translate_config(fixture_config("mixtral-8x7b"))
    assert config['mixture'] == {'experts': 8, 'top_k': 2}
    assert config['layer_types'] == ('full_attention',) * 32
    assert (config['emb_features'], config['mlp_features'], config['num_kv_heads']) == (4096, 14336, 8)


def test_mixtral_experts_stack_in_checkpoint_order():
    model, variables = fp32_decoder(FIXTURES / 'mixtral-tiny')
    ids = np.load(FIXTURES / 'mixtral-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'mixtral-tiny' / 'logits.npy')
    params = variables['params']
    experts = params['layers_0']['mlp']['experts']
    swapped = {**experts, 'gate_proj': {'kernel': experts['gate_proj']['kernel'][::-1]}}
    shuffled = {**variables, 'params': {**params, 'layers_0': {**params['layers_0'], 'mlp': {
        **params['layers_0']['mlp'], 'experts': swapped}}}}
    assert np.max(np.abs(np.asarray(model.apply(shuffled, ids)) - reference)) > 0.1


def test_released_qwen3_30b_a3b_config_translates_every_computational_field():
    """Qwen/Qwen3-30B-A3B spells its expert count num_experts (the released
    key; num_local_experts is the alias transformers writes back), routes
    every layer (decoder_sparse_step 1, mlp_only_layers []) and
    renormalises the top-8 softmax weights."""
    config = translate_config(fixture_config("qwen3-30b-a3b"))
    assert config['mixture'] == {'experts': 128, 'top_k': 8, 'layers': tuple(range(48)),
                                 'norm_topk_prob': True, 'expert_features': 768}
    assert config['qk_norm'] and config['layer_types'] == ('full_attention',) * 48
    assert (config['emb_features'], config['mlp_features'], config['head_dim']) == (2048, 6144, 128)


def test_qwen3_moe_picks_its_sparse_layers_like_the_reference():
    """decoder_sparse_step counts layers from one and mlp_only_layers takes
    layers back out (modeling_qwen3_moe.py:309-313): the tiny fixture's
    three layers leave only the second routed, and the loaded tree has the
    experts there and a dense MLP on the other two. A configuration that
    routes nothing is a dense qwen3 model and refuses."""
    config = translate_config(fixture_config("qwen3-moe-tiny"))
    assert config['mixture']['layers'] == (1,)
    _, variables = fp32_decoder(FIXTURES / 'qwen3-moe-tiny')
    mlps = {layer: sorted(block['mlp']) for layer, block in variables['params'].items()
            if layer.startswith('layers_')}
    assert mlps == {'layers_0': ['down_proj', 'gate_proj', 'up_proj'],
                    'layers_1': ['experts', 'gate'],
                    'layers_2': ['down_proj', 'gate_proj', 'up_proj']}
    with pytest.raises(ValueError, match="mlp_only_layers with decoder_sparse_step"):
        translate_config({**fixture_config("qwen3-moe-tiny"), 'mlp_only_layers': [0, 1, 2]})


def test_norm_topk_prob_off_keeps_the_raw_softmax_weights():
    """The fixture ships norm_topk_prob false, so a token's two weights are
    the softmax values themselves; renormalising them to sum to one, which
    Mixtral always does, moves the logits by 0.38 against the reference."""
    model, variables = fp32_decoder(FIXTURES / 'qwen3-moe-tiny')
    ids = np.load(FIXTURES / 'qwen3-moe-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'qwen3-moe-tiny' / 'logits.npy')
    assert model.mixture is not None and not model.mixture.norm_topk_prob
    renormalised = model.clone(mixture=dataclasses.replace(model.mixture, norm_topk_prob=True))
    assert np.max(np.abs(np.asarray(renormalised.apply(variables, ids)) - reference)) > 0.1


def test_olmo3_config_translates_to_the_post_norm_block():
    """The tiny fixture's config: three sliding layers to one full at one
    base, no pre-norms with the output pair on, and q/k norms over the
    whole projection. A config without layer_types takes the reference's
    own 3:1 pattern (configuration_olmo3.py:96-98)."""
    config = translate_config(fixture_config("olmo3-tiny"))
    assert config['layer_types'] == ('sliding_attention',) * 3 + ('full_attention',)
    assert config['sandwich_norms'] and not config['pre_norms']
    assert config['qk_norm'] and config['qk_norm_scope'] == 'projection'
    assert not config['scale_after_cast'] and config['rope_theta'] == 5e5
    assert config['kinds'] == {'sliding_attention': {'window': 4}}

    from transformers import Olmo3Config

    bare = {**fixture_config("olmo3-tiny"), 'num_hidden_layers': 6}
    del bare['layer_types']
    reference = Olmo3Config(**{**bare, 'layer_types': None}).layer_types
    assert reference is not None
    assert translate_config(bare)['layer_types'] == tuple(reference)


def test_olmo3_norms_the_whole_projection_and_no_input():
    """The loaded tree carries a q_norm of heads * head_dim and no pre-norm
    leaves. Normed per head instead, with each head taking its slice of the
    same scale (what a loader that split the projection's norm across heads
    would compute), the logits leave the reference by more than 0.1: the
    RMS over 16 dims is not the RMS over 64, so the scope is load-bearing."""
    model, variables = fp32_decoder(FIXTURES / 'olmo3-tiny')
    ids = np.load(FIXTURES / 'olmo3-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'olmo3-tiny' / 'logits.npy')
    attention = variables['params']['layers_0']['self_attn']
    assert attention['q_norm']['scale'].shape == (model.num_heads * model.features_per_head,)
    assert attention['k_norm']['scale'].shape == (model.kv_heads * model.features_per_head,)
    assert 'input_layernorm' not in variables['params']['layers_0']

    per_head = model.clone(qk_norm_scope='head')
    width = model.features_per_head
    sliced = jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf[:width] if path[-2].key in ('q_norm', 'k_norm') else leaf,
        variables)
    difference = np.max(np.abs(np.asarray(per_head.apply(sliced, ids)) - reference))
    assert difference > 0.1


RELEASED_OLMO3_YARN = {
    'rope_type': 'yarn', 'rope_theta': 5e5, 'factor': 8.0,
    'original_max_position_embeddings': 8192, 'beta_fast': 32.0,
    'beta_slow': 1.0, 'mscale': None, 'mscale_all_dim': None,
    'truncate': True, 'attention_factor': 1.2079441541679836,
}


def test_the_released_olmo_3_7b_config_puts_its_yarn_on_the_full_layers():
    """allenai/Olmo-3-1025-7B carries a rope_scaling of type yarn that the
    reference applies to its full-attention layers alone
    (configuration_olmo3.py:110-113), so it lands on that kind and the
    sliding layers keep rotating plainly at rope_theta. Everything else
    translates field for field."""
    config = translate_config(fixture_config("olmo-3-7b"))
    assert config == {
        'vocab_size': 100278, 'emb_features': 4096, 'num_layers': 32,
        'num_heads': 32, 'num_kv_heads': 32, 'head_dim': 128, 'mlp': 'swiglu',
        'mlp_features': 11008, 'max_seq_len': 8192, 'rope_theta': 5e5,
        'layer_types': (('sliding_attention',) * 3 + ('full_attention',)) * 8,
        'kinds': {'sliding_attention': {'window': 4096},
                  'full_attention': {'yarn': RELEASED_OLMO3_YARN}},
        'norm_eps': 1e-6, 'scale_after_cast': False, 'qk_norm': True,
        'attention_bias': False, 'tie_embeddings': False,
        'sandwich_norms': True, 'pre_norms': False, 'qk_norm_scope': 'projection',
    }


def test_the_released_olmo_3_7b_yarn_frequencies_are_the_references():
    """The rotary rule, on the release's own geometry and without its
    weights: `Olmo3RotaryEmbedding` calls `ROPE_INIT_FUNCTIONS['yarn']` for
    a kind whose entry names a type (modeling_olmo3.py:277-291) and scales
    that kind's cos/sin by the returned factor, while the kinds at
    'default' keep the plain table. Both halves are checked against the
    reference's own functions at head_dim 128 and base 5e5: the YaRN table,
    the attention factor, and that the reference leaves the sliding entry
    plain, with 46 of 64 pairs moved."""
    from transformers import Olmo3Config
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from dew.nn.rope import YarnScaling, inverse_frequencies, yarn_attention_factor, yarn_inv_freq

    released = fixture_config("olmo-3-7b")
    reference = Olmo3Config.from_dict(released)
    ropes = reference.to_dict()["rope_parameters"]
    assert ropes['sliding_attention']['rope_type'] == 'default'
    assert ropes['full_attention']['rope_type'] == 'yarn'

    expected, attention_factor = ROPE_INIT_FUNCTIONS['yarn'](
        reference, None, layer_type='full_attention')
    record = translate_config(released)['kinds']['full_attention']['yarn']
    scaling = YarnScaling(**record)
    scaled = yarn_inv_freq(128, 5e5, scaling, dtype=np.float32)
    plain = 1.0 / (5e5 ** (np.arange(0, 128, 2, dtype=np.float32) / 128))

    # The tables are equal only where torch's float32 pow rounds 5e5 ** x as
    # Dew's correctly rounded pow does; the rest of the table is the same
    # float32 arithmetic. 11 of these 64 powers lie within 0.1 ulp of a
    # rounding midpoint, so the precondition is checked, not assumed: a lane
    # whose torch pow rounds one differently fails here, naming torch's pow,
    # and there the tables may differ by up to 2 ulps.
    import torch

    exponents = torch.arange(0, 128, 2, dtype=torch.int64).float() / 128
    torch_powers = (1.0 / 5e5 ** exponents).numpy()
    rounded = inverse_frequencies(5e5, 128, dtype=np.float32)
    assert np.array_equal(torch_powers, rounded), (
        "torch's float32 pow rounds 5e5 ** x differently from the correctly rounded value at "
        f"{np.flatnonzero(torch_powers != rounded).tolist()}; the YaRN tables then differ by up to 2 ulps")
    assert isinstance(scaled, np.ndarray)
    np.testing.assert_array_equal(scaled, expected.numpy())
    assert yarn_attention_factor(scaling) == pytest.approx(attention_factor)
    assert np.sum(scaled != plain) >= 32


def test_the_olmo3_yarn_scales_the_full_layers_and_no_other():
    """The fixture's config is the release's rope at toy width: the yarn on
    the one full-attention layer of its 3:1 pattern, the three sliding
    layers plain. Three loads that read the same record and use it
    differently leave the reference by more than any tolerance here: plain
    rope on every layer by 0.35, the same yarn on every layer by 1.70, and
    the yarn frequencies rotated at unit amplitude, without the record's
    attention_factor, by 0.35. So the fixture holds the placement, the
    frequency ramp and the amplitude to account, not the presence of a ramp.

    The record's explicit attention_factor is the value
    `_compute_yarn_parameters` derives for factor 8 anyway
    (0.1 * ln(8) + 1), so dropping the field alone changes nothing; what
    the reference would disagree with is not applying the amplitude.
    """
    model, variables = fp32_decoder(FIXTURES / 'olmo3-yarn-tiny')
    ids = np.load(FIXTURES / 'olmo3-yarn-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'olmo3-yarn-tiny' / 'logits.npy')
    kinds = dict(model.kinds or {})
    yarn = kinds['full_attention'].yarn
    assert yarn is not None and kinds['sliding_attention'].yarn is None

    def moved(**replaced):
        clone = model.clone(kinds={**kinds, **replaced})
        return np.max(np.abs(np.asarray(clone.apply(variables, ids)) - reference))

    plain = moved(full_attention=dataclasses.replace(kinds['full_attention'], yarn=None))
    everywhere = moved(
        sliding_attention=dataclasses.replace(kinds['sliding_attention'], yarn=yarn))
    unscaled = moved(full_attention=dataclasses.replace(
        kinds['full_attention'],
        yarn=dataclasses.replace(yarn, attention_factor=1.0)))
    assert plain > 0.1 and everywhere > 0.1 and unscaled > 0.1


def test_mistral_window_changes_the_reference_logits():
    model, variables = fp32_decoder(FIXTURES / 'mistral-tiny')
    ids = np.load(FIXTURES / 'mistral-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'mistral-tiny' / 'logits.npy')
    unwindowed = model.clone(kinds={}, layer_types=('full_attention',) * model.num_layers)
    difference = np.max(np.abs(np.asarray(unwindowed.apply(variables, ids)) - reference))
    assert difference > 0.1


def test_the_real_gemma3_1b_config_translates():
    """google/gemma-3-1b-pt is gated, so the fixture is the identical config
    from a mirror. Nothing in it is beyond the field map."""
    config = translate_config(fixture_config("gemma3-1b"))

    assert config['emb_features'] == 1152 and config['num_layers'] == 26
    assert (config['num_heads'], config['num_kv_heads']) == (4, 1)
    assert config['head_dim'] == 256 and config['mlp_features'] == 6912
    # the config names no tie_word_embeddings, and Gemma3TextConfig ties by
    # default where Qwen and Llama do not
    assert config['vocab_size'] == 262144 and config['tie_embeddings'] is True
    assert config['attention_scale'] == pytest.approx(256 ** -0.5)
    assert config['rope_theta'] == 1e6
    assert config['kinds'] == {'sliding_attention': {'window': 512, 'rope_theta': 1e4}}
    assert config['sandwich_norms'] and config['scale_offset']
    # sliding_window_pattern 6: five sliding layers, then a full one
    assert config['layer_types'][:6] == (
        'sliding_attention',) * 5 + ('full_attention',)
    assert config['layer_types'].count('full_attention') == 4
    # the cache is clamped, the config asks for 32768
    assert config['max_seq_len'] == 8192


@pytest.mark.parametrize("field, value, message", [
    ('model_type', 'mamba', "model_type 'mamba'"),
    ('use_bidirectional_attention', True, "use_bidirectional_attention"),
    ('hidden_activation', 'relu', "hidden_act 'relu'"),
    ('rope_parameters', {'rope_type': 'linear', 'factor': 8.0, 'rope_theta': 1e6},
     "rope_type 'linear'"),
    ('sliding_window', None, "sliding_window is not set"),
])
def test_a_config_field_with_no_counterpart_is_refused(field, value, message):
    config = {**fixture_config("gemma3-tiny"), field: value}
    with pytest.raises(ValueError, match=message):
        translate_config(config)


def test_an_attention_softcap_maps_where_the_reference_applies_it():
    """Gemma 2 squashes its attention logits (modeling_gemma2.py:282); Gemma 3
    reads the same field into its attention and never passes it on
    (modeling_gemma3.py:334, :370-379), so there it changes nothing and maps
    to nothing. Llama's reference has no such field at all."""
    assert translate_config(fixture_config("gemma2-tiny"))['attn_logit_softcap'] == 5.0
    ignored = translate_config({**fixture_config("gemma3-tiny"), 'attn_logit_softcapping': 50.0})
    assert 'attn_logit_softcap' not in ignored
    with pytest.raises(ValueError, match="attn_logit_softcapping"):
        translate_config({**fixture_config("llama-tiny"), 'attn_logit_softcapping': 50.0})


def test_the_released_gemma_2b_config_translates_field_by_field():
    """unsloth/gemma-2b carries google/gemma-2b's config: hidden_act 'gelu',
    which transformers 5.16.1 runs as the erf gelu (modeling_gemma.py:93),
    the (1 + w) norms, sqrt(d) embeddings and a tied head."""
    assert translate_config(fixture_config("gemma-2b")) == {
        'vocab_size': 256000, 'emb_features': 2048, 'num_layers': 18,
        'num_heads': 8, 'num_kv_heads': 1, 'head_dim': 256, 'mlp': 'geglu_exact',
        'mlp_features': 16384, 'max_seq_len': 8192, 'rope_theta': 1e4,
        'layer_types': ('full_attention',) * 18, 'kinds': {},
        'norm_eps': 1e-6, 'scale_after_cast': False, 'qk_norm': False,
        'attention_bias': False, 'tie_embeddings': True,
        'scale_offset': True, 'embedding_scale': True,
    }


def test_the_released_gemma_2_2b_config_translates_field_by_field():
    """unsloth/gemma-2-2b carries google/gemma-2-2b's config: 26 layers
    alternating sliding and full at one rope base, query_pre_attn_scalar
    256 on head_dim 256, both softcaps, and the sandwich norms."""
    config = translate_config(fixture_config("gemma-2-2b"))
    assert config == {
        'vocab_size': 256000, 'emb_features': 2304, 'num_layers': 26,
        'num_heads': 8, 'num_kv_heads': 4, 'head_dim': 256, 'mlp': 'geglu',
        'mlp_features': 9216, 'max_seq_len': 8192, 'rope_theta': 1e4,
        'layer_types': ('sliding_attention', 'full_attention') * 13,
        'kinds': {'sliding_attention': {'window': 4096}},
        'norm_eps': 1e-6, 'scale_after_cast': False, 'qk_norm': False,
        'attention_bias': False, 'tie_embeddings': True,
        'scale_offset': True, 'embedding_scale': True, 'sandwich_norms': True,
        'attention_scale': 256 ** -0.5, 'final_logit_softcap': 30.0,
        'attn_logit_softcap': 50.0,
    }


def test_a_gemma2_config_without_layer_types_alternates_like_the_reference():
    """Gemma2Config fills the pattern itself (configuration_gemma2.py:95-98);
    the expected pattern comes from the reference class."""
    from transformers import Gemma2Config

    config = {**fixture_config("gemma2-tiny"), "num_hidden_layers": 5}
    del config["layer_types"]
    reference = Gemma2Config(**{**config, "layer_types": None}).layer_types
    assert reference is not None
    assert translate_config(config)["layer_types"] == tuple(reference)
    assert translate_config(config)["layer_types"][-1] == "sliding_attention"


def test_dropping_the_attention_softcap_breaks_gemma2_parity():
    """The fixture caps at 5 so the tanh moves the logits by 1.45; a load
    that read the cap and applied none would pass no tolerance below that."""
    model, variables = fp32_decoder(FIXTURES / 'gemma2-tiny')
    ids = np.load(FIXTURES / 'gemma2-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'gemma2-tiny' / 'logits.npy')
    uncapped = model.clone(attn_logit_softcap=None)
    assert np.max(np.abs(np.asarray(uncapped.apply(variables, ids)) - reference)) > 1.0


def test_the_erf_gelu_is_not_the_tanh_gelu_on_gemma():
    """gemma-tiny names hidden_act 'gelu'; run through the tanh approximation
    instead, its logits drift 1.7e-03 from the reference, above the 1e-4
    parity tolerance, so the two activations are two mlp values."""
    model, variables = fp32_decoder(FIXTURES / 'gemma-tiny')
    ids = np.load(FIXTURES / 'gemma-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'gemma-tiny' / 'logits.npy')
    approximate = model.clone(mlp='geglu')
    difference = np.max(np.abs(np.asarray(approximate.apply(variables, ids)) - reference))
    assert 1e-4 < difference < 1e-2


def test_a_rope_scaling_spelled_the_old_way_is_refused():
    """transformers reads rope_type from the older 'type' key too
    (modeling_rope_utils.py:785, 839), so a config that spells its Yarn
    scaling that way scales there and must not load here as plain rope. A
    bare factor names no type at all and is still a scaling."""
    yarn = {**fixture_config("llama-tiny"),
            'rope_scaling': {'type': 'yarn', 'factor': 4.0,
                             'original_max_position_embeddings': 32}}
    with pytest.raises(ValueError, match="rope_type 'yarn'"):
        translate_config(yarn)

    factor_only = {**fixture_config("llama-tiny"), 'rope_scaling': {'factor': 8.0}}
    with pytest.raises(ValueError, match=r"rope_scaling scaling fields \['factor'\]"):
        translate_config(factor_only)


@pytest.mark.parametrize("head_dim, theta, ramp", [
    (16, 5e5, {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
               'original_max_position_embeddings': 64}),
    (128, 5e5, {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
                'original_max_position_embeddings': 8192}),
])
def test_the_llama3_ramp_matches_the_reference_frequencies(head_dim, theta, ramp):
    """RopeScaling.apply against transformers' _compute_llama3_parameters,
    the reference's own function, on the tiny fixture's geometry and on
    Llama-3.1-8B's (head_dim 128, base 5e5, factor 8 off 8192). Observed
    difference 0.0 on both; the ramp moves 7 of the tiny table's 8 pairs
    and 35 of the release's 64."""
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from dew.nn.rope import RopeScaling

    config = LlamaConfig.from_dict({
        "hidden_size": head_dim * 4, "num_attention_heads": 4, "head_dim": head_dim,
        "rope_theta": theta, "rope_scaling": {'rope_type': 'llama3', **ramp},
        "max_position_embeddings": 8 * ramp['original_max_position_embeddings']})
    reference, attention_factor = ROPE_INIT_FUNCTIONS['llama3'](config, 'cpu')
    assert attention_factor == 1.0
    plain = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    scaled = np.asarray(RopeScaling(**ramp).apply(jnp.asarray(plain)))
    assert np.max(np.abs(scaled - reference.numpy())) < 1e-7
    assert np.sum(scaled != plain) >= head_dim // 4


def test_a_llama3_rope_scaling_translates_and_loads():
    """The tiny Llama 3.1 fixture: rope_scaling under the reference's names
    on the backbone, and the same weights under plain rope at rope_theta
    miss the reference by 4.4."""
    config = translate_config(fixture_config("llama31-tiny"))
    assert config['rope_scaling'] == {
        'rope_type': 'llama3', 'factor': 8.0, 'low_freq_factor': 1.0,
        'high_freq_factor': 4.0, 'original_max_position_embeddings': 64}
    model, variables = fp32_decoder(FIXTURES / 'llama31-tiny')
    ids = np.load(FIXTURES / 'llama31-tiny' / 'input_ids.npy')
    reference = np.load(FIXTURES / 'llama31-tiny' / 'logits.npy')
    plain = model.clone(rope_scaling=None)
    assert np.max(np.abs(np.asarray(plain.apply(variables, ids)) - reference)) > 1.0


def test_the_released_llama_3_1_8b_config_translates_field_by_field():
    """unsloth/Llama-3.1-8B carries meta-llama/Llama-3.1-8B's config."""
    assert translate_config(fixture_config("llama-3.1-8b")) == {
        'vocab_size': 128256, 'emb_features': 4096, 'num_layers': 32,
        'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'mlp': 'swiglu',
        'mlp_features': 14336, 'max_seq_len': 8192, 'rope_theta': 5e5,
        'layer_types': ('full_attention',) * 32, 'kinds': {},
        'norm_eps': 1e-5, 'scale_after_cast': True, 'qk_norm': False,
        'attention_bias': False, 'tie_embeddings': False,
        'rope_scaling': {'rope_type': 'llama3', 'factor': 8.0, 'low_freq_factor': 1.0,
                         'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192},
    }


def test_a_ramp_the_reference_puts_on_one_kind_lands_on_that_kind():
    """OLMo 3 moves a flat rope_scaling onto its full-attention entry
    (configuration_olmo3.py:110-113) and leaves the sliding layers plain, and
    a nested rope_parameters may state the same directly. Both land on the
    full kind, not the model, since a kind's None rides the model's ramp and
    could not turn one off. A ramp with a field missing raises a ValueError
    naming the field."""
    ramp = {'rope_type': 'llama3', 'factor': 8.0, 'low_freq_factor': 1.0,
            'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192}
    flat = {**fixture_config("olmo-3-7b"), 'rope_scaling': ramp}
    config = translate_config(flat)
    assert 'rope_scaling' not in config
    assert config['kinds'] == {'sliding_attention': {'window': 4096},
                               'full_attention': {'rope_scaling': ramp}}
    from transformers import Olmo3Config

    parameters = Olmo3Config.from_dict(flat).to_dict()['rope_parameters']
    assert parameters['sliding_attention']['rope_type'] == 'default'
    assert parameters['full_attention']['rope_type'] == 'llama3'

    nested = {**fixture_config("olmo3-tiny"), 'rope_parameters': {
        'full_attention': {**ramp, 'rope_theta': 5e5},
        'sliding_attention': {'rope_type': 'default', 'rope_theta': 5e5}}}
    assert translate_config(nested)['kinds']['full_attention'] == {'rope_scaling': ramp}

    with pytest.raises(ValueError, match="missing \\['high_freq_factor'\\]"):
        translate_config({**flat, 'rope_scaling': {
            k: v for k, v in ramp.items() if k != 'high_freq_factor'}})


@pytest.mark.parametrize("name", TINY + ROUTED + HYBRID)
def test_translated_weights_are_exactly_the_models_variables(name, rng):
    """Same collections, same paths, same shapes, same dtypes as a freshly
    initialised model: `params` for every family, and the `moe` collection
    DeepSeek's routers keep their bias in."""
    config = translate_config(fixture_config(name))
    built = {**config, 'dtype': 'float32', 'attention_impl': 'reference'}
    model = models.build('causal_transformer', **built)
    initialised = flat_tree(model.init(rng, jnp.zeros((1, 4), jnp.int32)))

    from dew.interop.sources import load_shards
    loaded = flat_tree(translate_weights(load_shards(FIXTURES / name), config))

    assert set(loaded) == set(initialised)
    for path, leaf in loaded.items():
        assert leaf.shape == initialised[path].shape, path
        assert leaf.dtype == jnp.float32, path


@pytest.mark.parametrize("name", TINY + ROUTED + HYBRID)
def test_fp32_logits_match_the_reference_implementation(name):
    """The parity claim: transformers' logits, our logits, same weights."""
    directory = FIXTURES / name
    model, variables = fp32_decoder(directory)
    ids = np.load(directory / "input_ids.npy")
    reference = np.load(directory / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))
    if name in HYBRID + CLASSIC:
        assert_as_exact_as_the_reference(logits, reference, np.load(directory / "logits_f64.npy"),
                                         f"{name} logits")
    else:
        difference = float(np.max(np.abs(logits - reference)))
        assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_nemotron_h_reads_independent_blocks_and_grouped_ssd():
    """The SSD width and dt floor are the reference's, independently of expand and time_step_limit."""
    config = translate_config(fixture_config("nemotron-h-tiny"))
    model = config.value
    assert model.per_layer_types == ("linear_attention", "mlp", "full_attention", "linear_attention", "mlp")
    assert model.mlp_features == 0 and model.qk_norm is False
    assert model.kind_of("linear_attention").mixer == Mamba2Mixer(
        num_heads=4, head_dim=8, state_size=4, n_groups=2, norm_groups=2,
        conv_kernel=4, chunk_size=4, use_bias=True, time_step_limit=(0.7, float("inf")))
    assert model.kind_of("full_attention").mixer == AttentionMixer(nope=True)
    assert model.kind_of("mlp").mixer == MLPMixer(intermediate_size=32)


def test_nemotron_h_weight_names_invert():
    from dew.interop.families.nemotron_h import export_path, weight_path
    from dew.interop.sources import load_shards

    config = translate_config(fixture_config("nemotron-h-tiny"))
    for name in load_shards(FIXTURES / "nemotron-h-tiny"):
        path = weight_path(name, config)
        assert export_path(".".join(path[1:]), config) == name
    assert weight_path("lm_head.weight", {"tie_embeddings": True}) is None
    assert export_path("lm_head.kernel", {"tie_embeddings": True}) is None
    with pytest.raises(ValueError, match="no place"):
        weight_path("backbone.layers.0.mixer.rotary_emb.inv_freq", config)
    with pytest.raises(ValueError, match="not a Nemotron-H"):
        export_path("layers_0.mlp.gate_proj.kernel", config)


def test_nemotron_h_does_not_claim_another_mamba2_hybrid():
    from dew.interop.families.nemotron_h import NEMOTRON_H as family

    config = translate_config(fixture_config("nemotron-h-tiny"))
    assert family.matches(config.value)
    assert not family.matches(CausalTransformer(vocab_size=64, emb_features=16, num_heads=2,
                                               mlp_features=0, qk_norm=False,
                                               layer_types=("mamba", "full_attention"),
                                               kinds={"mamba": LayerKind(mixer=Mamba2Mixer(n_groups=1))}))


def test_nemotron_h_legacy_patterns_and_mamba_aliases_read_as_the_modern_config():
    modern = fixture_config("nemotron-h-tiny")
    legacy = {**modern, "hybrid_override_pattern": "M-*M-", "num_hidden_layers": 999}
    del legacy["layers_block_type"]
    for key, alias in (("n_groups", "mamba_n_groups"), ("conv_kernel", "mamba_d_conv"),
                       ("chunk_size", "mamba_chunk_size"), ("use_conv_bias", "mamba_conv_bias"),
                       ("time_step_min", "mamba_dt_min")):
        legacy[alias] = legacy.pop(key)
    assert translate_config(legacy) == translate_config(modern)
    assert translate_config({**modern, "hybrid_override_pattern": "E"}) == translate_config(modern)
    legacy["layer_types"] = ["mamba", "mlp", "attention", "mamba", "mlp"]
    assert translate_config(legacy) == translate_config(modern)


@pytest.mark.parametrize("legacy, read", [
    ("mamba_num_groups", "n_groups"),
    ("mamba_state_dim", "ssm_state_size"),
    ("num_query_groups", "num_key_value_heads"),
    ("rms_norm_eps", "layer_norm_epsilon"),
    ("norm_eps", "layer_norm_epsilon"),
])
def test_nemotron_h_legacy_geometry_must_repeat_the_field_the_reference_reads(legacy, read):
    """Released duplicates agree; a contradiction cannot silently name two geometries."""
    modern = fixture_config("nemotron-h-tiny")
    repeated = {**modern, legacy: modern[read]}
    assert translate_config(repeated) == translate_config(modern)
    with pytest.raises(ValueError, match=legacy):
        translate_config({**repeated, legacy: modern[read] + 1})


@pytest.mark.parametrize("changes, message", [
    ({"num_nextn_predict_layers": 1}, "multi-token prediction"),
    ({"mlp_hidden_act": "silu"}, "mlp_hidden_act"),
    ({"mamba_hidden_act": "relu"}, "mamba_hidden_act"),
    ({"layers_block_type": ["unknown"]}, "layers_block_type"),
    ({"layers_block_type": None, "hybrid_override_pattern": "MX"}, "unknown block kinds"),
    ({"layers_block_type": []}, "layers_block_type"),
])
def test_nemotron_h_refuses_a_block_it_cannot_compute(changes, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("nemotron-h-tiny"), **changes})


def _assert_nemotron_h_released_config(config, name):
    if name == "nemotron-h-30b-a3b":
        model = translate_config(config).value
        assert model.num_layers == 52 and model.emb_features == 2688
        assert model.per_layer_types.count("moe") == 23
        mixer = model.kind_of("moe").mixer
        assert isinstance(mixer, MLPMixer) and mixer.activation == "relu2"
        assert mixer.mixture == Mixture(
            experts=128, top_k=6, score_function="sigmoid", scaling=2.5, bias=True,
            expert_features=1856, shared_features=3712)
        return
    model = translate_config(config).value
    assert model.num_layers == 42 and model.emb_features == 3136 and model.max_seq_len == 262144
    assert model.num_heads == 40 and model.kv_heads == 8 and model.features_per_head == 128
    assert model.per_layer_types.count("linear_attention") == 21
    assert model.per_layer_types.count("full_attention") == 4
    assert model.per_layer_types.count("mlp") == 17
    assert model.kind_of("linear_attention").mixer == Mamba2Mixer(
        num_heads=96, head_dim=80, state_size=128, n_groups=8, norm_groups=8,
        conv_kernel=4, chunk_size=256, time_step_limit=(0.001, float("inf")))
    assert model.kind_of("mlp").mixer == MLPMixer(intermediate_size=12544)


@pytest.mark.parametrize("name", ("nemotron-h-4b", "nemotron-h-30b-a3b"))
def test_nemotron_h_released_configs_translate(name):
    _assert_nemotron_h_released_config(fixture_config(name), name)


@pytest.mark.parametrize("name", HYBRID[1:])
def test_nemotron_h_routed_mixer_rebuilds_from_its_run_record(name):
    from dew.registry import mixers

    mixer = translate_config(fixture_config(name)).value.kind_of("moe").mixer
    assert isinstance(mixer, MLPMixer) and mixer.mixture is not None
    fields = dataclasses.asdict(mixer)
    assert mixers.from_record({"class": "mlp", "fields": fields}) == mixer


@pytest.mark.parametrize("fields", [{"layers": (1,)}, {"hash_layers": (1,)}, {"media_bias": True}])
def test_nemotron_h_mixer_refuses_controls_owned_by_the_decoder_feedforward(fields):
    with pytest.raises(ValueError, match="decoder feed-forward slot"):
        MLPMixer(intermediate_size=16, mixture=Mixture(experts=8, **fields))


def test_nemotron_h_zero_shared_width_uses_the_references_dense_width():
    config = {**fixture_config("nemotron-h-moe-tiny"), "moe_shared_expert_intermediate_size": 0}
    mixer = translate_config(config).value.kind_of("moe").mixer
    assert isinstance(mixer, MLPMixer) and mixer.mixture is not None
    assert mixer.mixture.shared_features == config["intermediate_size"]


@pytest.mark.parametrize("name", HYBRID[1:])
def test_nemotron_h_fused_experts_export_the_source_layout(name, tmp_path):
    """The native 3-D layout and transformers' per-expert save layout compute the same model."""
    from safetensors.numpy import save_file

    from dew.interop.sources import load_shards

    tensors = load_shards(FIXTURES / name)
    packed = {key: value for key, value in tensors.items() if ".experts." not in key}
    for layer in (1, 3):
        for projection in ("up_proj", "down_proj"):
            stem = f"backbone.layers.{layer}.mixer.experts"
            packed[f"{stem}.{projection}"] = np.stack(
                [tensors[f"{stem}.{index}.{projection}.weight"] for index in range(8)])
    source = tmp_path / "packed"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(fixture_config(name)))
    save_file(packed, source / "model.safetensors")
    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    expected = Pretrained.load(FIXTURES / name, dtype="float32", attention_impl="reference")
    for path, tensor in flat_tree(expected.variables).items():
        np.testing.assert_array_equal(tensor, flat_tree(loaded.variables)[path])
    destination = tmp_path / "export"
    loaded.save(destination)
    exported = load_shards(destination)
    assert exported.keys() == packed.keys()
    for key, tensor in packed.items():
        np.testing.assert_array_equal(tensor, exported[key])
    ids = np.load(FIXTURES / name / "input_ids.npy")
    assert_as_exact_as_the_reference(loaded.model.apply(loaded.variables, ids),
                                     np.load(FIXTURES / name / "logits.npy"),
                                     np.load(FIXTURES / name / "logits_f64.npy"), f"{name} packed logits")


@pytest.mark.network
@pytest.mark.parametrize("name", ("nemotron-h-4b", "nemotron-h-30b-a3b"))
def test_nemotron_h_pinned_hub_configs_read_as_the_committed_releases(name):
    from huggingface_hub import hf_hub_download

    source = json.loads((FIXTURES / name / "source.json").read_text())
    config = json.loads(Path(hf_hub_download(source["repo"], "config.json",
                                            revision=source["revision"])).read_text())
    assert config == fixture_config(name)
    _assert_nemotron_h_released_config(config, name)


@pytest.mark.parametrize("name", HYBRID)
def test_nemotron_h_export_preserves_the_source_config_weights_and_logits(name, tmp_path):
    """The loaded hybrid exports in its source layout, as Mamba-2 and Qwen3-Next do."""
    from dew.interop.sources import load_shards

    directory = FIXTURES / name
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    destination = tmp_path / "nemotron-h"
    loaded.save(destination)
    assert json.loads((destination / "config.json").read_text()) == fixture_config(name)
    source, exported = load_shards(directory), load_shards(destination)
    assert source.keys() == exported.keys()
    for name, tensor in source.items():
        np.testing.assert_array_equal(tensor, exported[name])
    restored = Pretrained.load(destination, dtype="float32", attention_impl="reference")
    ids = np.load(directory / "input_ids.npy")
    np.testing.assert_array_equal(loaded.model.apply(loaded.variables, ids),
                                  restored.model.apply(restored.variables, ids))
    assert_as_exact_as_the_reference(transformers_logits(destination, ids, tmp_path),
                                     np.load(directory / "logits.npy"),
                                     np.load(directory / "logits_f64.npy"), "Nemotron-H export logits")


def test_the_bf16_gemma_forward_still_tracks_the_reference():
    """gemma3-tiny is the fixture with an attention scale (query_pre_attn_scalar
    16 on head_dim 32), and bf16 is the recipe's default compute dtype. Against
    the fp32 reference logits the observed difference is 5.7e-02 on the
    reference kernel, tolerance 1e-01; dropping the scale moves them by 1.06."""
    directory = FIXTURES / "gemma3-tiny"
    pretrained = Pretrained.load(str(directory), dtype='bfloat16',
                                 attention_impl='reference')
    model, variables = pretrained.model, pretrained.variables
    reference = np.load(directory / "logits.npy")

    logits = np.asarray(model.apply(
        variables, jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)),
        np.float32)

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-1, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_the_tied_head_is_the_embedding_and_untied_heads_load():
    """A tied checkpoint carries lm_head as a copy; the tree keeps one leaf."""
    config = translate_config(fixture_config("qwen3-tiny"))
    embedding = np.arange(256 * 64, dtype=np.float32).reshape(256, 64)
    tensors = {'model.embed_tokens.weight': embedding,
               'lm_head.weight': embedding.copy()}

    tied = translate_weights(tensors, config)['params']
    assert 'lm_head' not in tied
    assert np.array_equal(tied['embed_tokens']['embedding'], embedding)

    untied = translate_weights(tensors, {**config, 'tie_embeddings': False})['params']
    assert np.array_equal(untied['lm_head']['kernel'], embedding.T)

    # Qwen3-0.6B really does ship both, byte for byte identical; a head that
    # is not that copy means the checkpoint is not the tied model it says
    different = {**tensors, 'lm_head.weight': embedding + 1.0}
    with pytest.raises(ValueError, match="not the embedding it claims to copy"):
        translate_weights(different, config)


def test_an_unfamiliar_tensor_name_is_refused():
    config = translate_config(fixture_config("qwen3-tiny"))
    with pytest.raises(ValueError, match="unknown tensor name"):
        translate_weights({'model.layers.0.self_attn.rotary_emb.inv_freq':
                           np.zeros((8,), np.float32)}, config)


@pytest.mark.parametrize("name", TINY)
def test_export_round_trips_the_weights_and_the_config(name, tmp_path):
    model, variables = fp32_decoder(FIXTURES / name)
    export = tmp_path / name

    PretrainedDecoder.from_model(model, variables, tokenizer="byte").save(export)
    again, reloaded = fp32_decoder(export)

    # against the fixture's config, not the exported one read twice: a field
    # the export changes and the model does not read back (the context length,
    # the hidden_act spelling) shows up here
    assert (translate_config(json.loads((export / "config.json").read_text()))
            == translate_config(fixture_config(name)))
    assert again == model, "the exported config rebuilds a different model"
    for path, leaf in flat_tree(reloaded['params']).items():
        assert np.array_equal(np.asarray(leaf),
                              np.asarray(flat_tree(variables['params'])[path])), path
    generation = json.loads((export / "generation_config.json").read_text())
    # Dew's byte vocabulary is no HF tokenizer and has no files to write, so
    # the name it was exported with is the whole record of it.
    assert generation['tokenizer_name'] == "byte"
    assert not (export / "tokenizer_config.json").exists()


def test_an_export_carries_the_tokenizer_it_names(tmp_path):
    """A named tokenizer writes its own files into the export directory.

    What makes the directory a checkpoint rather than weights: transformers'
    AutoTokenizer, llama.cpp's converter and everything built on it look for
    tokenizer_config.json beside the weights, and a `tokenizer_name` string
    is not a tokenizer. The name is resolved through the loader a training
    run uses, from local files only, so an export copies what the host
    already has and reaches nothing; the files are the whole record, and
    the name, a path on this machine, is written nowhere.
    """
    from transformers import AutoTokenizer

    model, variables = fp32_decoder(FIXTURES / "llama-tiny")
    export = tmp_path / "named"

    PretrainedDecoder.from_model(model, variables, tokenizer=str(TOKENIZER)).save(export)

    assert (export / "tokenizer_config.json").exists()
    written = AutoTokenizer.from_pretrained(str(export), local_files_only=True)
    expected = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    assert written.get_vocab() == expected.get_vocab()
    assert written.encode("The trainer") == expected.encode("The trainer")
    assert "tokenizer_name" not in json.loads((export / "generation_config.json").read_text())


def test_an_export_with_a_local_tokenizer_is_the_directory_alone(tmp_path):
    """The tokenizer is named by a path that is gone once the export is
    written, as it is on any other machine. No file of the export holds
    that path, and no JSON string is an absolute path of two or more parts
    (a vocabulary's "/" is a token, not a path); a re-export of a source whose
    generation config carried a `tokenizer_name` path drops it, and a
    fresh process loads the export's tokenizer from the directory alone,
    as transformers' AutoTokenizer does and as `Pretrained.load` does."""
    import shutil

    from transformers import AutoTokenizer

    model, variables = fp32_decoder(FIXTURES / "llama-tiny")
    source = tmp_path / "elsewhere" / "tokenizer"
    shutil.copytree(TOKENIZER, source)
    export = tmp_path / "export"
    PretrainedDecoder.from_model(model, variables, tokenizer=str(source)).save(export)
    shutil.rmtree(tmp_path / "elsewhere")

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    def assert_portable(directory):
        for path in directory.iterdir():
            assert str(tmp_path).encode() not in path.read_bytes(), path.name
            if path.suffix == ".json":
                absolute = [text for text in strings(json.loads(path.read_text()))
                            if os.path.isabs(text) and len(Path(text).parts) > 2]
                assert absolute == [], (path.name, absolute)

    assert_portable(export)
    generation = json.loads((export / "generation_config.json").read_text())
    (export / "generation_config.json").write_text(json.dumps({**generation, "tokenizer_name": str(source)}))
    again = tmp_path / "again"
    Pretrained.load(export, dtype="float32", attention_impl="reference").save(again)
    assert_portable(again)

    expected = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True).encode("The trainer")
    probe = (
        "import sys, json, numpy as np\n"
        "from transformers import AutoTokenizer\n"
        "from dew.interop import Pretrained\n"
        "export = sys.argv[1]\n"
        "ids = AutoTokenizer.from_pretrained(export, local_files_only=True).encode('The trainer')\n"
        "bundle = Pretrained.load(export, dtype='float32', attention_impl='reference')\n"
        "print(json.dumps({'ids': ids, 'processor': bundle.processor is not None,\n"
        "                  'decoded': bundle.processor.decode(np.asarray([ids]))[0]}))\n")
    result = subprocess.run([sys.executable, "-c", probe, str(again)], capture_output=True, text=True,
                            env={**os.environ, "HF_HUB_OFFLINE": "1"}, check=True)
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded["ids"] == expected and loaded["processor"]
    assert "The trainer" in loaded["decoded"]


def test_a_tokenizer_object_is_exported_without_being_named(tmp_path):
    """An export given the tokenizer itself writes its files and records no
    name: an HF tokenizer knows its vocabulary, not which hub repo or run a
    caller means by it, and the directory it lands in is the answer to that.
    """
    from transformers import AutoTokenizer

    model, variables = fp32_decoder(FIXTURES / "llama-tiny")
    export = tmp_path / "object"

    PretrainedDecoder.from_model(model, variables, tokenizer=AutoTokenizer.from_pretrained(
                                str(TOKENIZER), local_files_only=True)).save(export)

    assert (export / "tokenizer_config.json").exists()
    assert "tokenizer_name" not in json.loads(
        (export / "generation_config.json").read_text())


def biased_qwen3(rng):
    """The qwen3-tiny shape with the q/k/v/o biases its own config leaves off.

    nn.Dense starts a bias at zero, and zeros would let a mismapped bias name
    through the round-trip, so every bias gets its own draw.
    """
    config = {**translate_config(fixture_config("qwen3-tiny")), 'attention_bias': True}
    built = {**config, 'dtype': 'float32', 'attention_impl': 'reference'}
    model = models.build('causal_transformer', **built)
    leaves, structure = jax.tree_util.tree_flatten_with_path(
        model.init(rng, jnp.zeros((1, 4), jnp.int32)))
    return model, jax.tree_util.tree_unflatten(structure, [
        jax.random.normal(jax.random.fold_in(rng, index), leaf.shape, leaf.dtype) * 0.2
        if path[-1].key == 'bias' else leaf
        for index, (path, leaf) in enumerate(leaves)])


def test_a_biased_qwen3_round_trips_through_an_export(tmp_path, rng):
    """attention_bias is one flag for all four projections, the flag
    Qwen3Attention builds from config.attention_bias (modeling_qwen3.py:225-236)
    and Gemma3Attention from the same field (modeling_gemma3.py:322-333), so a
    biased qk_norm model exports."""
    model, variables = biased_qwen3(rng)
    export = tmp_path / "biased"

    PretrainedDecoder.from_model(model, variables).save(export)
    again, reloaded = fp32_decoder(export)

    assert json.loads((export / "config.json").read_text())['attention_bias'] is True
    assert again == model, "the exported config rebuilds a different model"
    for path, leaf in flat_tree(reloaded['params']).items():
        assert np.array_equal(np.asarray(leaf),
                              np.asarray(flat_tree(variables['params'])[path])), path


def test_the_real_checkpoints_tensor_table_matches_the_built_tree(rng):
    """No weights: the 311 names and shapes of Qwen3-0.6B against our tree.

    This is the check that a config translation and a key map fit a
    checkpoint nobody wants to download in CI.
    """
    from dew.interop.decoder_parts import dew_path

    table = json.loads((REAL / "tensors.json").read_text())
    config = translate_config(json.loads((REAL / "config.json").read_text()))
    built = {**config, 'dtype': 'float32', 'attention_impl': 'reference'}
    model = models.build('causal_transformer', **built)

    expected = {}
    for name, info in table['tensors'].items():
        path = dew_path(name, config)
        if path is None:
            continue  # the tied lm_head copy
        shape = tuple(info['shape'])
        expected['.'.join(path)] = (tuple(reversed(shape))
                                    if path[-1] == 'kernel' else shape)

    tree = jax.eval_shape(lambda: model.init(rng, jnp.zeros((1, 4), jnp.int32)))
    initialised = {path: leaf.shape for path, leaf in flat_tree(tree).items()}

    assert initialised == expected
    assert len(expected) == len(table['tensors']) - 1, "the tied head was not skipped"


def qwen3_is_available() -> bool:
    if os.environ.get("DEW_NETWORK_TESTS") == "1":
        return True
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache("Qwen/Qwen3-0.6B", "model.safetensors")
    return isinstance(cached, str)


@pytest.mark.network
@pytest.mark.skipif(not qwen3_is_available(),
                    reason="Qwen3-0.6B is neither cached nor DEW_NETWORK_TESTS=1")
def test_qwen3_0_6b_matches_the_reference_on_the_real_weights():
    """The real thing: 28 layers of Qwen3-0.6B, fp32, against torch's top 32."""
    prompt = json.loads((REAL / "prompt.json").read_text())
    reference = np.load(REAL / "reference.npz")
    ids = np.asarray(prompt['input_ids'], np.int32)[None]

    pretrained = Pretrained.load(prompt['repo'], dtype='float32',
                                 attention_impl='reference',
                                 max_seq_len=int(ids.shape[1]))
    model, variables = pretrained.model, pretrained.variables
    logits = np.asarray(model.apply(variables, jnp.asarray(ids)), np.float32)[0]

    assert np.array_equal(np.argmax(logits, axis=-1), reference['argmax'])
    ours = np.take_along_axis(logits, reference['top_ids'], axis=-1)
    difference = float(np.max(np.abs(ours - reference['top_logits'])))
    assert difference < 5e-3, f"max |top-32 logit difference| {difference:.3e}"


def transformers_logits(export, ids, tmp_path):
    """What transformers computes for `ids` on the checkpoint at `export`."""
    script = """
import sys
import numpy as np, torch
from transformers import AutoModelForCausalLM
directory, ids_path, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
model.eval()
model.set_attn_implementation("eager")
with torch.no_grad():
    logits = model(input_ids=torch.from_numpy(np.load(ids_path).astype(np.int64))).logits
np.save(out, logits.to(torch.float32).numpy())
"""
    ids_path, out = tmp_path / "ids.npy", tmp_path / "theirs.npy"
    np.save(ids_path, ids)
    subprocess.run([sys.executable, "-c", script, str(export), str(ids_path), str(out)],
                   check=True, capture_output=True)
    return np.load(out)


def test_our_export_loads_in_transformers_with_the_same_logits(tmp_path):
    """The export is a real HF checkpoint: transformers reads it and agrees."""
    model, variables = fp32_decoder(FIXTURES / "qwen3-tiny")
    export = tmp_path / "exported"
    PretrainedDecoder.from_model(model, variables).save(export)

    ids = np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy")
    ours = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(transformers_logits(export, ids, tmp_path) - ours)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"


def test_a_biased_qwen3_export_carries_its_biases_into_transformers(tmp_path, rng):
    """Qwen3Attention builds q, k, v and o with bias=config.attention_bias
    (modeling_qwen3.py:225-236), so the reference applies the biases where dew
    does and a biased export is a checkpoint it reads."""
    model, variables = biased_qwen3(rng)
    export = tmp_path / "biased"
    PretrainedDecoder.from_model(model, variables).save(export)

    ids = np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy")
    ours = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(transformers_logits(export, ids, tmp_path) - ours)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"


MIXED = ("sliding_attention", "full_attention")


@pytest.mark.parametrize(
    "fixture, changes, model_type",
    [
        # Llama's block with a window on some layers: LlamaConfig has no window,
        # MinistralConfig names one per layer.
        (
            "llama31-tiny",
            {"layer_types": MIXED, "kinds": {"sliding_attention": LayerKind(window=3)}},
            "ministral",
        ),
        # Gemma3TextConfig rotates the sliding layers of a config that states only
        # rope_theta at its own 10000, not at rope_theta.
        (
            "gemma3-tiny",
            {"layer_types": MIXED, "kinds": {"sliding_attention": LayerKind(window=3)}},
            "gemma3_text",
        ),
        # Olmo3Config moves a flat rope_theta onto the full layers alone.
        ("olmo3-tiny", {"rope_theta": 500.0}, "olmo3"),
    ],
)
def test_an_export_transformers_reads_computes_what_dew_computes(tmp_path, fixture, changes, model_type):
    model, variables = fp32_decoder(FIXTURES / fixture)
    model = model.clone(**changes)
    PretrainedDecoder.from_model(model, variables).save(tmp_path / "exported")
    assert json.loads((tmp_path / "exported" / "config.json").read_text())["model_type"] == model_type
    ids = np.random.default_rng(0).integers(3, model.vocab_size, (2, 12))
    ours = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))
    difference = float(np.max(np.abs(transformers_logits(tmp_path / "exported", ids, tmp_path) - ours)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"


@pytest.mark.parametrize("fixture, changes, lost", [
    ("llama-tiny", {"final_logit_softcap": 1.5}, "final_logit_softcap"),
    ("qwen3-tiny", {"causal": False}, "causal"),
    ("gemma-tiny", {"attention_scale": 0.9}, "attention_scale"),
    ("qwen3-tiny", {"swiglu_limit": 0.5}, "swiglu_limit"),
])
def test_an_export_no_family_carries_is_refused_naming_what_it_would_lose(tmp_path, fixture, changes, lost):
    model, variables = fp32_decoder(FIXTURES / fixture)
    with pytest.raises(ValueError, match=lost):
        PretrainedDecoder.from_model(model.clone(**changes), variables).save(str(tmp_path))
    assert not (tmp_path / "config.json").exists()





def _diffusion_fp32(name):
    """A tiny diffusion fixture as a model plus variables, mask id stored on
    the model the way the backbone holds it."""
    from safetensors.numpy import load_file

    directory = FIXTURES / name
    config = translate_config(fixture_config(name))
    assert config["mask_token_id"] == 120
    model = models.build("causal_transformer", config, dtype="float32", attention_impl="reference")
    assert model.mask_token_id == 120
    variables = translate_weights(load_file(str(directory / "model.safetensors")), config)
    return (model, variables, np.load(directory / "input_ids.npy"),
            np.load(directory / "logits.npy"), np.load(directory / "logits_f64.npy"))


def test_llada_logits_match_the_released_model_code():
    """fp32 parity with LLaDA's own modeling_llada.py (GSAI-ML/LLaDA-8B-Base
    at 0f2787f, run by tools/remote_code_reference.py under transformers
    4.46.3) on a tiny random-weight checkpoint, held to the float64 rule:
    Dew's RMS error from the release's float64 logits at most twice the
    release's own fp32 error. A causal model on the same weights misses by
    1.5, so the fixture exercises full attention."""
    model, variables, ids, reference, truth = _diffusion_fp32("llada-tiny")
    assert_as_exact_as_the_reference(model.apply(variables, ids), reference, truth, "llada-tiny logits")
    causal = model.clone(causal=True)
    assert np.max(np.abs(np.asarray(causal.apply(variables, ids)) - reference)) > 1.0


@pytest.fixture(scope='module')
def glm5_next_source():
    return Pretrained.load(FIXTURES / 'glm5-next-tiny', dtype='float32', attention_impl='reference')


def test_glm5_next_translates_the_released_text_config_and_refuses_the_wrapper():
    wrapper = fixture_config('glm-5.3-flash')
    config = translate_config(wrapper['text_config'])
    assert config['num_layers'] == 45
    assert config['layer_types'].count('linear_attention') == 34
    assert config['layer_types'].count('full_attention') == 11
    assert config['mixture']['experts'] == 288 and config['mixture']['top_k'] == 8
    assert config['mixture']['layers'] == tuple(range(3, 45))
    assert config['hyper_connections'] == {'hc_mult': 4, 'hc_eps': 1e-6,
                                           'hc_sinkhorn_iters': 20, 'head': 'mean'}
    assert config["index_share_for_mtp_iteration"]
    with pytest.raises(ValueError, match='text_config'):
        translate_config(wrapper)


def test_glm5_next_logits_and_prediction_depth_match_reference(glm5_next_source):
    source = glm5_next_source
    ids = jnp.asarray(np.load(source.source / 'input_ids.npy'), jnp.int32)
    logits = np.asarray(source.model.apply(source.variables, ids))
    expected = np.load(source.source / 'logits.npy')
    assert scaled_difference(logits, expected) < 1e-4
    np.testing.assert_array_equal(logits.argmax(-1), expected.argmax(-1))
    hidden = source.model.apply(source.variables, ids, method=source.model.hidden_states)
    mtp = source.model.apply(source.variables, hidden, ids, method=source.model.mtp_logits)[0]
    assert scaled_difference(np.asarray(mtp), np.load(source.source / 'mtp_logits.npy')) < 1e-4


def test_glm5_prediction_index_reuse_does_not_change_forward_or_sft(glm5_next_source):
    source = glm5_next_source
    ids = jnp.asarray(np.load(source.source / "input_ids.npy")[:, :6], jnp.int32)
    enabled = source.model
    disabled = enabled.clone(index_share_for_mtp_iteration=False)

    def loss(model, variables):
        variables = {**source.variables, "params": variables}
        hidden = model.apply(variables, ids, method="hidden_states")
        predicted = model.apply(variables, hidden, ids, train=True, method="mtp_logits")[0]
        return jnp.mean(predicted ** 2)

    expected, expected_grad = jax.jit(jax.value_and_grad(lambda p: loss(disabled, p)))(
        source.variables["params"]
    )
    actual, actual_grad = jax.jit(jax.value_and_grad(lambda p: loss(enabled, p)))(source.variables["params"])
    np.testing.assert_array_equal(actual, expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    np.testing.assert_array_equal(enabled.apply(source.variables, ids), disabled.apply(source.variables, ids))


def test_glm5_prediction_index_reuse_preserves_public_padded_greedy_generation(glm5_next_source):
    from dew.nn.inputs import ModelInputs
    from dew.sampling import Sampling, Speculative

    source = glm5_next_source
    tokens = np.load(source.source / "input_ids.npy")[[0, 1, 0], :5].copy()
    valid = np.arange(5)[None, :] >= np.asarray([0, 2, 4])[:, None]
    tokens[~valid] = 0
    inputs = ModelInputs(jnp.asarray(tokens), {"attention_mask": jnp.asarray(valid)})
    task = source.text_generation(sampling=Sampling(temperature=0))
    ordinary = task(inputs, max_new_tokens=7, key=0).host()
    speculative = task(inputs, max_new_tokens=7, key=0, strategy=Speculative(block=3)).host()
    np.testing.assert_array_equal(speculative.tokens, ordinary.tokens)
    np.testing.assert_array_equal(speculative.lengths, ordinary.lengths)
    np.testing.assert_array_equal(speculative.terminated, ordinary.terminated)


def test_glm5_next_prefill_and_token_steps_match_parallel(glm5_next_source):
    source = glm5_next_source
    model, variables = source.model, source.variables
    ids = jnp.asarray(np.load(source.source / 'input_ids.npy'), jnp.int32)
    full = np.asarray(model.apply(variables, ids))
    state = model.apply(variables, ids.shape[0], method='init_cache', mutable=['cache'])[1]
    out, state = model.apply({**variables, **state}, ids[:, :4], decode=True, mutable=['cache'])
    pieces = [np.asarray(out)]
    for index in range(4, ids.shape[1]):
        out, state = model.apply(
            {**variables, **state}, ids[:, index : index + 1], decode=True, mutable=["cache"]
        )
        pieces.append(np.asarray(out))
    actual = np.concatenate(pieces, axis=1)
    assert scaled_difference(actual, full) < 1e-4
    np.testing.assert_array_equal(actual.argmax(-1), full.argmax(-1))


def test_glm5_next_mapping_preserves_every_leaf_and_dynamic_stream_mixing(glm5_next_source):
    source = glm5_next_source
    ids = jnp.asarray(np.load(source.source / 'input_ids.npy'), jnp.int32)
    expected = flat_tree(jax.eval_shape(source.model.init, jax.random.key(0), ids))
    loaded = flat_tree(source.variables)
    assert {name: leaf.shape for name, leaf in loaded.items()} == {
        name: leaf.shape for name, leaf in expected.items()}
    changed = jax.tree.map(lambda x: x, source.variables)
    changed['params']['layers_0']['attn_hc']['scale'] = jnp.zeros(3)
    actual = np.asarray(source.model.apply(changed, ids))
    assert scaled_difference(actual, np.load(source.source / 'logits.npy')) > 1e-4


@pytest.mark.parametrize('field,value', [
    ('mhc', False), ('mla_use_nope', False), ('qk_rope_head_dim', 2),
    ('indexer_types', ['full', 'full', 'full', 'shared', 'full']),
    ('moe_router_dtype', 'bfloat16'), ('scoring_func', 'softmax'),
    ('index_topk', 3), ('num_nextn_predict_layers', 2),
    ('attention_dropout', 0.1),
])
def test_glm5_next_refuses_unimplemented_variants_by_name(field, value):
    with pytest.raises(ValueError, match=field):
        translate_config({**fixture_config('glm5-next-tiny'), field: value})


def test_glm5_next_empty_nested_config_still_applies_safe_gate():
    config = fixture_config('glm5-next-tiny')
    del config['linear_attn_config']
    config['linear_lower_bound'] = None
    def mixer(record):
        return translate_config(record)['kinds']['linear_attention']['mixer']['fields']

    absent = mixer(config)
    empty = mixer({**config, 'linear_attn_config': {}})
    disabled = mixer({**config, 'linear_attn_config': {'safe_gate': False}})
    assert absent['linear_lower_bound'] is None
    assert empty['linear_lower_bound'] == -5.0
    assert disabled['linear_lower_bound'] is None


def test_glm5_next_null_kv_head_count_uses_query_heads():
    config = fixture_config('glm5-next-tiny')
    assert translate_config({**config, 'num_key_value_heads': None}) == translate_config(config)


def test_glm5_next_refuses_disagreeing_layer_schedules():
    config = fixture_config('glm5-next-tiny')
    linear = {**config['linear_attn_config'], 'kda_layers': [0, 1]}
    with pytest.raises(ValueError, match=r"linear_attn_config.kda_layers"):
        translate_config({**config, 'linear_attn_config': linear})


def test_glm5_mlp_schedule_and_router_normalization_follow_source_config():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    config = {**fixture_config("glm5-next-tiny"), "first_k_dense_replace": 1,
              "mlp_layer_types": ["dense", "sparse", "dense", "sparse", "sparse"],
              "norm_topk_prob": False}
    reference = Glm5NextTextConfig.from_dict(config)
    assert reference.mlp_layer_types is not None
    native = translate_config(config)["mixture"]
    assert native["layers"] == tuple(index for index, kind in enumerate(reference.mlp_layer_types)
                                      if kind == "sparse") == (1, 3, 4)
    assert native["norm_topk_prob"] is reference.norm_topk_prob is False


@pytest.mark.parametrize("field,value", [
    ("layer_scalar", "trainable"), ("scale_offset", True),
    ("final_logit_softcap", 20.0), ("num_nextn_predict_layers", 2),
    ("hyper_connections", {"hc_mult": 4, "head": "weighted"}),
])
def test_standalone_glm5_refuses_computation_without_a_source_inverse(
    field, value, glm5_next_source, tmp_path
):
    with pytest.raises(ValueError, match=field):
        PretrainedDecoder.from_model(glm5_next_source.model.clone(**{field: value}),
                                     glm5_next_source.variables).save(tmp_path)
    assert not (tmp_path / "config.json").exists()


def test_dream_logits_match_the_released_model_code():
    """fp32 parity with Dream's own modeling_dream.py (Dream-org/Dream-v0-Base-7B
    at 6572adb, run by tools/remote_code_reference.py under transformers
    4.46.3) on a tiny random-weight checkpoint, held to the float64 rule. A
    causal model on the same weights misses by 4.9, so the fixture exercises
    full attention."""
    model, variables, ids, reference, truth = _diffusion_fp32("dream-tiny")
    assert_as_exact_as_the_reference(model.apply(variables, ids), reference, truth, "dream-tiny logits")
    causal = model.clone(causal=True)
    assert np.max(np.abs(np.asarray(causal.apply(variables, ids)) - reference)) > 1.0


@pytest.mark.parametrize("mode", ["frozen", "trainable"])
def test_scalar_mode_survives_scanning_and_rematerialized_backward(mode):
    """Frozen HF buffers and trainable Google scalars retain each view's math."""
    from safetensors.numpy import load_file
    base, _ = fp32_decoder(GEMMA4_MOE)
    fields = {**translate_config(fixture_config("gemma4-moe-tiny")), "layer_scalar": mode}
    variables = translate_weights(load_file(str(GEMMA4_MOE / "model.safetensors")), fields)
    plain = base.clone(layer_scalar=mode)
    scanned = plain.clone(scan_layers=True, remat="full")
    ids = jnp.asarray(np.load(GEMMA4_MOE / "input_ids.npy"), jnp.int32)

    def loss(model, params):
        return jnp.mean(model.apply({**variables, "params": params}, ids) ** 2)

    expected, expected_grad = jax.jit(jax.value_and_grad(lambda params: loss(plain, params)))(
        variables["params"]
    )
    value, gradient = jax.jit(jax.value_and_grad(lambda params: loss(scanned, params)))(variables["params"])
    np.testing.assert_allclose(value, expected, atol=1e-5, rtol=0)
    for actual, wanted in zip(jax.tree.leaves(gradient), jax.tree.leaves(expected_grad), strict=True):
        np.testing.assert_allclose(actual, wanted, atol=1e-4, rtol=1e-5)
