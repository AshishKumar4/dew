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
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.nn.backbones.layer_plan import LayerKind
from dew.registry import models

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
# The committed byte-level BPE an export can name without a download.
TOKENIZER = Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-tools"
TINY = ("qwen3-tiny", "gemma3-tiny", "llama-tiny", "mistral-tiny", "qwen2-tiny",
        "gemma-tiny", "gemma2-tiny", "olmo3-tiny", "olmo3-yarn-tiny",
        "llama31-tiny", "gpt2-tiny", "opt-tiny", "gpt-neox-tiny",
        "bloom-tiny", "gpt-neo-tiny", "phi-tiny", "falcon-tiny", "falcon-mha-tiny", "gptj-tiny", "phi3-tiny",
        "gpt-bigcode-tiny", "starcoder2-tiny")
# A multi-head GPTBigCode's record is GPT-2's, so a weight translation from
# the record alone reads GPT-2's Conv1D layout; the loader names the source's
# family, and tests/test_hf_bigcode_starcoder2.py holds it to the reference.
CLASSIC = ('bloom-tiny', 'gpt-neo-tiny', 'phi-tiny', 'falcon-tiny', 'falcon-mha-tiny', 'gptj-tiny',
           'phi3-tiny', 'gpt-bigcode-tiny', 'starcoder2-tiny')
DEEPSEEK = ("deepseek-v3-tiny", "deepseek-v32-tiny")
ROUTED = (*DEEPSEEK, "kimi-k2-tiny", "mixtral-tiny", "qwen3-moe-tiny", "mimo-v2-flash-tiny")
HYBRID = ("nemotron-h-tiny", "nemotron-h-moe-tiny", "nemotron-h-moe-latent-tiny")
GEMMA4_MOE = FIXTURES / "gemma4-moe-tiny"
REAL = FIXTURES / "qwen3-0.6b"


def scaled_difference(ours, theirs) -> float:
    return float(np.max(np.abs(ours - theirs)) / np.max(np.abs(theirs)))


def fixture_config(name):
    return json.loads((FIXTURES / name / "config.json").read_text())


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


def _respelled(name, tmp_path, edit):
    """The fixture's checkpoint under the config `edit` returns from its own."""
    from shutil import copytree

    directory = copytree(FIXTURES / name, tmp_path / name)
    config = edit(fixture_config(name))
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def _reference_logits(name, directory):
    """The reference logits the fixture shipped, beside what `directory` loads to."""
    model, variables = fp32_decoder(directory)
    ids = np.load(FIXTURES / name / "input_ids.npy")
    return np.asarray(model.apply(variables, ids)), np.load(FIXTURES / name / "logits.npy")


def _without(*fields):
    return lambda config: {key: value for key, value in config.items() if key not in fields}


@pytest.mark.parametrize("name, edit", [
    # Gemma3TextConfig: GELU, global base 1e6 and local 1e4 by default.
    ("gemma3-tiny", _without("rope_parameters", "hidden_activation", "attn_logit_softcapping")),
    # Gemma2Config: GELU and a final cap of 30 by default.
    ("gemma2-tiny", _without("hidden_activation", "final_logit_softcapping")),
    # Olmo3Config: a record per kind at its default_theta of 500000.
    ("olmo3-tiny", _without("rope_parameters")),
    # A flat llama3 record whose base sits beside it, in either field.
    ("llama31-tiny", lambda config: {**config, "rope_theta": config["rope_parameters"]["rope_theta"],
                                     "rope_parameters": {key: value for key, value
                                                         in config["rope_parameters"].items()
                                                         if key != "rope_theta"}}),
    ("llama31-tiny", lambda config: {**_without("rope_parameters")(config),
                                     "rope_theta": config["rope_parameters"]["rope_theta"],
                                     "rope_scaling": {key: value for key, value
                                                      in config["rope_parameters"].items()
                                                      if key != "rope_theta"}}),
    # OLMo 3's class folds a legacy rope_scaling into the full layers' record.
    ("olmo3-yarn-tiny", lambda config: {
        **config,
        "rope_parameters": {**config["rope_parameters"],
                            "full_attention": {"rope_type": "default", "rope_theta": 500000.0}},
        "rope_scaling": {key: value for key, value in config["rope_parameters"]["full_attention"].items()
                         if key != "rope_theta"}}),
])
def test_a_source_spelled_another_way_computes_the_reference_logits(name, edit, tmp_path):
    """A source that leaves its config class's defaults unstated, or spells
    its rope another way the class reads, loads to the model the fully
    spelled fixture is, and computes the reference's logits."""
    ours, theirs = _reference_logits(name, _respelled(name, tmp_path, edit))
    assert np.max(np.abs(ours - theirs)) < 1e-4


def test_a_plain_rope_scaling_clears_the_ramp_rope_parameters_names(tmp_path):
    """The mixin replaces rope_parameters with a stated rope_scaling whole,
    so an explicit plain rope_scaling over a llama3 record is plain rope at
    the default base, as that spelling stated alone is."""
    plain = {"rope_type": "default"}
    conflicting = _respelled("llama31-tiny", tmp_path / "conflicting",
                             lambda config: {**config, "rope_scaling": plain})
    alone = _respelled("llama31-tiny", tmp_path / "alone",
                       lambda config: {**config, "rope_parameters": {**plain, "rope_theta": 10000.0}})
    ids = np.load(FIXTURES / "llama31-tiny" / "input_ids.npy")
    logits = [np.asarray(model.apply(variables, ids))
              for model, variables in (fp32_decoder(conflicting), fp32_decoder(alone))]
    np.testing.assert_array_equal(*logits)
    assert np.max(np.abs(logits[0] - np.load(FIXTURES / "llama31-tiny" / "logits.npy"))) > 1e-3


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


@pytest.mark.parametrize("name", [name for name in TINY + ROUTED + HYBRID if name not in (
    'bloom-tiny',
    'falcon-mha-tiny',
    'falcon-tiny',
    'gpt-neo-tiny',
    'gpt-neox-tiny',
    'gpt2-tiny',
    'gptj-tiny',
    'opt-tiny',
    'phi-tiny',
    'phi3-tiny',
)])
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


@pytest.mark.parametrize("name", [name for name in TINY if name not in (
    'bloom-tiny',
    'falcon-mha-tiny',
    'falcon-tiny',
    'gpt-bigcode-tiny',
    'gpt-neo-tiny',
    'gpt-neox-tiny',
    'gpt2-tiny',
    'gptj-tiny',
    'opt-tiny',
    'phi-tiny',
    'phi3-tiny',
    'starcoder2-tiny',
)])
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

    written = AutoTokenizer.from_pretrained(str(again), local_files_only=True)
    reference_tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    assert written.get_vocab() == reference_tokenizer.get_vocab()
    assert written.encode("The trainer") == reference_tokenizer.encode("The trainer")
    assert "tokenizer_name" not in json.loads((again / "generation_config.json").read_text())


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

    again, reloaded = fp32_decoder(export)
    assert again == model
    for path, leaf in flat_tree(reloaded["params"]).items():
        assert np.array_equal(np.asarray(leaf), np.asarray(flat_tree(variables["params"])[path])), path


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
