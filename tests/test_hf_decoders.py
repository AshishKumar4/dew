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
from model_support import flat_tree, llada_draw, llada_fields
from reference_error import (
    as_given,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    declared_seeds,
    recording,
)

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


def recorded_logit_difference(model, variables, directory) -> float:
    """`model`'s logits over a fixture's ids (`directory`/input_ids.npy) against the logits its
    reference recorded (`directory`/logits.npy): the same argmax everywhere, and their largest
    difference, which each caller holds to its own bound."""
    ids = np.load(directory / "input_ids.npy")
    reference = np.load(directory / "logits.npy")
    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))
    return float(np.max(np.abs(logits - reference)))


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


def test_a_llama_release_repeating_its_fields_under_megatrons_names_loads_as_it_states_them(tmp_path):
    """Nanbeige4.3's Llama configs repeat five fields under Megatron's names (ffn_hidden_size,
    num_query_groups, rotary_base, kv_channels, seq_length) beside the ones LlamaConfig reads, and
    state its embedding and residual dropout, which LlamaModel never applies: the release loads as
    the model its own fields state, the same model and variables bit for bit, and a repeat that
    states another value than its field is refused by name (ledger 1883)."""
    from shutil import copytree

    original = FIXTURES / "llama-tiny"
    plain = Pretrained.load(original, dtype="float32")
    config = fixture_config("llama-tiny")
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    theta = config["rope_parameters"]["rope_theta"]
    repeats = {"ffn_hidden_size": config["intermediate_size"],
               "num_query_groups": config["num_key_value_heads"], "rotary_base": theta,
               "kv_channels": head_dim, "seq_length": config["max_position_embeddings"]}
    directory = tmp_path / "checkpoint"
    copytree(original, directory)
    stated = {**config, "head_dim": head_dim, **repeats, "embd_pdrop": 0.0, "resid_pdrop": 0.0}
    (directory / "config.json").write_text(json.dumps(stated))
    loaded = Pretrained.load(directory, dtype="float32")
    assert loaded.model == plain.model
    for got, want in zip(jax.tree.leaves(loaded.variables), jax.tree.leaves(plain.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    (directory / "config.json").write_text(json.dumps({**stated, "rotary_base": 2 * theta}))
    with pytest.raises(ValueError):
        Pretrained.load(directory, dtype="float32")


@pytest.mark.parametrize("name, head", [("llama-tiny", "causal"), ("nemotron-h-tiny", "causal"),
                                        ("modernbert-tiny", "masked")])
def test_a_biased_mlp_loads_and_exports_where_the_reference_reads_mlp_bias(name, head, tmp_path):
    """Llama's MLP, Nemotron-H's MLP layers and ModernBERT's GLU carry biases where their config
    states mlp_bias, which each reference reads (modeling_llama.py's LlamaMLP, NemotronHMLP,
    ModernBertMLP's packed Wi): a checkpoint with random biases loads to logits as exact as the
    reference's float32 ones, both from its float64 ones, and saves, from its source and (but for
    Nemotron-H, whose hybrid pattern only its source writes) from the model and its variables
    alone, to a checkpoint that reloads to the same logits."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM

    from tools.torch_precision import float64

    auto = AutoModelForCausalLM if head == "causal" else AutoModelForMaskedLM
    config = AutoConfig.from_pretrained(FIXTURES / name)
    config.mlp_bias = True
    reference, loading = auto.from_pretrained(FIXTURES / name, config=config, attn_implementation="eager",
                                             output_loading_info=True)
    added = set(loading["missing_keys"])
    assert added and all(key.endswith("bias") for key in added)
    draws = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for key, parameter in reference.named_parameters():
            if key in added:
                parameter.copy_(0.1 * torch.randn(parameter.shape, generator=draws))
    source = tmp_path / "source"
    reference.save_pretrained(source)
    ids = np.load(FIXTURES / name / "input_ids.npy")

    def theirs(dtype):
        with float64() if dtype == torch.float64 else torch.no_grad():
            model = auto.from_config(config, dtype=dtype, attn_implementation="eager").eval()
            model.load_state_dict(reference.state_dict())
            with torch.no_grad():
                return model(torch.from_numpy(ids).long()).logits.double().numpy()

    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    ours = np.asarray(loaded.model.apply(loaded.variables, ids), np.float64)
    assert_as_exact_as_the_reference(ours, theirs(torch.float32), theirs(torch.float64),
                                     f"{name} logits with MLP biases")
    loaded.save(tmp_path / "export")
    exports = [tmp_path / "export"]
    if name == "nemotron-h-tiny":
        # Nemotron-H writes its hybrid pattern only from its source (`preserve_source_layout`):
        # written from the model alone, the export refuses rather than drop the pattern.
        with pytest.raises(ValueError):
            PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / "from_model")
    else:
        # Written from the model and its variables alone too, as a trained run exports.
        PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / "from_model")
        exports.append(tmp_path / "from_model")
    for export in exports:
        again = Pretrained.load(export, dtype="float32", attention_impl="reference")
        reloaded = np.asarray(again.model.apply(again.variables, ids))
        np.testing.assert_array_equal(reloaded, ours.astype(np.float32))

@pytest.mark.gpu(reason="TF32 is a GPU's: a CPU multiplies float32 operands in float32 at any precision")
def test_a_float32_decoder_multiplies_float32_products_on_a_gpu():
    """A float32 decoder that states no precision multiplies float32 products, as torch's float32
    matmuls do by default (`dew.nn.precision.float32_products`): its logits on a GPU are as exact
    as transformers' float32 ones, both from transformers' float64 ones, under JAX's own default
    precision (the test lane sets 'highest' for every product that states none), which multiplies
    float32 in TF32 there, 2.9e-4 from float64 against 2.9e-7."""
    import contextlib

    import torch
    from transformers import AutoModelForCausalLM

    from tools.torch_precision import float64

    directory = FIXTURES / "qwen3-tiny"
    ids = np.load(directory / "input_ids.npy")
    loaded = Pretrained.load(str(directory), dtype="float32", attention_impl="xla")
    with jax.default_matmul_precision(None):
        ours = np.asarray(jax.jit(loaded.model.apply)(loaded.variables, jnp.asarray(ids)), np.float64)

    def theirs(dtype):
        with float64() if dtype == torch.float64 else contextlib.nullcontext(), torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(directory, attn_implementation="eager")
            return model.to(dtype).eval()(torch.from_numpy(ids).long()).logits.double().numpy()

    assert_as_exact_as_the_reference(ours, theirs(torch.float32), theirs(torch.float64), "float32 logits")


@pytest.mark.parametrize(("name", "stale"), [
    ("qwen2-tiny", {"use_mrope": True}),
    ("mamba2-tiny", {"rms_norm": False, "intermediate_size": 3, "norm_before_gate": True}),
    ("nemotron-h-tiny", {"mamba_num_groups": 7, "rms_norm_eps": 0.5}),
    ("qwen35-tiny", {"attn_output_gate": False}),
    ("granitemoe-tiny", {"router_jitter_noise": 0.5, "activation_function": "gelu"}),
    ("modernbert-tiny", {"is_causal": True, "causal_mask": True}),
])
def test_a_field_the_reference_does_not_read_leaves_its_logits_whatever_it_holds(tmp_path, name, stale):
    """A released config may carry fields its reference neither declares nor reads (a multimodal
    rotary flag, mamba_ssm's own sizes, an older name for a field it reads under another, an output
    gate the attention always builds, a router jitter or a direction the reference never applies), and
    the reference computes the same model whatever they hold: its logits under them are its logits
    without them, bit for bit. The checkpoint loads to the same model too, however the fields
    disagree with the ones the reference reads: its logits under them are the bits it computes
    without them, the same program on the same values, which the fixture's parity test holds to the
    reference's."""
    from shutil import copytree

    import torch
    import transformers

    directory = tmp_path / "checkpoint"
    copytree(FIXTURES / name, directory)
    config = {**fixture_config(name), **stale}
    (directory / "config.json").write_text(json.dumps(config))
    ids = np.load(FIXTURES / name / "input_ids.npy")

    def theirs(source):
        reference = getattr(transformers, config["architectures"][0]).from_pretrained(
            source, attn_implementation="eager").eval()
        with torch.no_grad():
            return reference(torch.from_numpy(ids).long()).logits.numpy()

    def ours(source):
        loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
        return np.asarray(loaded.model.apply(loaded.variables, ids))

    np.testing.assert_array_equal(theirs(directory), theirs(FIXTURES / name))
    np.testing.assert_array_equal(ours(directory), ours(FIXTURES / name))


@pytest.mark.parametrize("family", ["qwen3_moe", "deepseek_v3"])
def test_an_expert_schedule_that_routes_no_layer_loads_as_the_dense_model_the_reference_builds(
        tmp_path, family):
    """A Qwen3-MoE whose mlp_only_layers cover every layer, or a DeepSeek-V3 whose dense layers are
    all of them (with no expert count at all), is a dense model to the reference, which builds a
    plain MLP on each layer; it loads as that model, to the reference's logits."""
    import torch
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    if family == "qwen3_moe":
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=4,
            num_experts_per_tok=2, mlp_only_layers=[0, 1], tie_word_embeddings=False))
    else:
        model = DeepseekV3ForCausalLM(DeepseekV3Config(
            vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=None,
            kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, first_k_dense_replace=2,
            tie_word_embeddings=False))
    model.eval()
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    if family == "deepseek_v3":
        del config["n_routed_experts"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    ids = np.arange(12, dtype=np.int32).reshape(1, 12) % 64
    with torch.no_grad():
        expected = model(torch.tensor(ids, dtype=torch.long)).logits.float().numpy()
    loaded = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    assert loaded.model.mixture is None
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)


def test_an_olmo3_model_with_a_llama3_ramp_on_one_kind_writes_back_what_transformers_reads(tmp_path):
    """OLMo3 rotates each layer kind through its own rope_parameters entry, so a Llama-3 ramp on
    the full layers beside plain sliding ones loads, writes back in that spelling, and
    transformers' Olmo3 reads the export to the logits Dew computes."""
    from shutil import copytree

    import torch
    from transformers import Olmo3ForCausalLM

    source = tmp_path / "source"
    copytree(FIXTURES / "olmo3-tiny", source)
    config = fixture_config("olmo3-tiny")
    config["rope_parameters"]["full_attention"] = {
        "rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 16}
    (source / "config.json").write_text(json.dumps(config))
    ids = np.load(FIXTURES / "olmo3-tiny" / "input_ids.npy")
    from dew.interop import PretrainedDecoder

    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    expected = np.asarray(loaded.model.apply(loaded.variables, ids))
    # The native writer, which spells the config from the model, not the source's own file.
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path / "export")
    reloaded = Pretrained.load(tmp_path / "export", dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)), expected)
    reference = Olmo3ForCausalLM.from_pretrained(tmp_path / "export", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        logits = reference(torch.tensor(ids, dtype=torch.long)).logits.numpy()
    np.testing.assert_allclose(logits, expected, atol=1e-4, rtol=0)


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
    record = translate_config(released)['kinds']['full_attention']['rope_scaling']
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


_YARN_SOURCES = (
    ("deepseek-v2-tiny", 10000.0),
    ("deepseek-v3-tiny", 10000.0),
    ("deepseek-v32-tiny", 10000.0),
    ("gpt-oss-tiny", 150000.0),
)


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
    # Gemma4TextConfig installs proportional full-layer rotary at 1e6 and plain
    # sliding rotary at 1e4 for an absent or null record, and GELU by default.
    ("gemma4-moe-tiny", _without("rope_parameters", "hidden_activation")),
    ("gemma4-moe-tiny", lambda config: {**config, "rope_parameters": None}),
    # Phi3Config replaces rope_parameters with a legacy rope_scaling whole, and
    # its own original_max_position_embeddings replaces the record's.
    ("phi3-tiny", lambda config: {**config, "rope_parameters": {"rope_type": "default", "rope_theta": 500.0},
                                  "rope_scaling": {**config["rope_parameters"],
                                                   "original_max_position_embeddings": 16}}),
    # OLMo 3's class folds a legacy rope_scaling into the full layers' record.
    ("olmo3-yarn-tiny", lambda config: {
        **config,
        "rope_parameters": {**config["rope_parameters"],
                            "full_attention": {"rope_type": "default", "rope_theta": 500000.0}},
        "rope_scaling": {key: value for key, value in config["rope_parameters"]["full_attention"].items()
                         if key != "rope_theta"}}),
    # DeepSeek and GPT OSS read a legacy YaRN entry at the stated flat base.
    *[(name, lambda config: {
        **_without("rope_parameters")(config),
        "rope_theta": config["rope_parameters"]["rope_theta"],
        "rope_scaling": {**_without("rope_theta", "rope_type", "type")(config["rope_parameters"]),
                         "type": "yarn"}})
      for name, _ in _YARN_SOURCES],
    # An unstated base takes the family's own default, not another family's.
    *[(name, lambda config: {**config,
                            "rope_parameters": _without("rope_theta")(config["rope_parameters"])})
      for name, _ in _YARN_SOURCES],
    ("gpt-oss-tiny", lambda config: {
        **config, "rope_parameters": _without("factor")(config["rope_parameters"])}),
    # The Nemotron-H releases that carry their own remote code name the head width attention_head_dim.
    ("nemotron-h-tiny", lambda config: {**_without("head_dim")(config),
                                        "attention_head_dim": config["head_dim"]}),
    ("llada-tiny", _without("n_kv_heads", "head_dim")),
    # Glm4MoeConfig fills an absent factor with half the head, the fixture's own.
    ("glm4-moe-tiny", lambda config: {**_without("partial_rotary_factor")(config),
                                      "rope_parameters": _without("partial_rotary_factor")(
                                          config["rope_parameters"])}),
    # MiMoV2FlashConfig installs per-kind bases 5e6 and 1e4 at a factor of 0.334 for an absent record.
    ("mimo-v2-flash-tiny", _without("rope_parameters")),
    ("llada-tiny", lambda config: {
        **_without("n_layers", "activation_type")(config),
        "num_layers": config["n_layers"], "hidden_activation": config["activation_type"]}),
])
def test_a_source_spelled_another_way_computes_the_reference_logits(name, edit, tmp_path):
    """A source that leaves its config class's defaults unstated, or spells
    its rope another way the class reads, loads to the model the fully
    spelled fixture is, and computes the reference's logits."""
    ours, theirs = _reference_logits(name, _respelled(name, tmp_path, edit))
    assert np.max(np.abs(ours - theirs)) < 1e-4


@pytest.mark.parametrize("name, edit", [
    # LLaDA-MoE's config classes default the rotary's base and factor to -1, which names no rotary.
    *[(name, _without(field)) for name in ("llada-moe-tiny", "llada-moe-v2-tiny")
      for field in ("rope_theta", "partial_rotary_factor")],
    # MiMoV2FlashRotaryEmbedding indexes each per-kind record's base; its class fills none.
    ("mimo-v2-flash-tiny", lambda config: {**config, "rope_parameters": {
        kind: _without("rope_theta")(entry) for kind, entry in config["rope_parameters"].items()}}),
])
def test_a_config_whose_class_names_no_value_for_an_unstated_field_is_refused(name, edit, tmp_path):
    """A field the config class has no working default for, left unstated, is refused rather than
    read at another family's default."""
    with pytest.raises(ValueError):
        fp32_decoder(_respelled(name, tmp_path, edit))


def test_a_dense_llada_config_without_a_base_rotates_at_its_classs_10000(tmp_path):
    """LLaDA's ModelConfig, which its config class starts from, rotates at 10000; the release states
    its own 500000. Without a base the fixture computes what it computes at 10000."""
    ids = np.load(FIXTURES / "llada-tiny" / "input_ids.npy")
    unstated, stated = (fp32_decoder(_respelled("llada-tiny", tmp_path / where, edit)) for where, edit in (
        ("unstated", _without("rope_theta")), ("stated", lambda config: {**config, "rope_theta": 10000.0})))
    np.testing.assert_array_equal(np.asarray(unstated[0].apply(unstated[1], ids)),
                                  np.asarray(stated[0].apply(stated[1], ids)))


def test_a_plain_rotary_gpt_oss_exports_a_plain_record_that_reloads_to_its_logits(tmp_path):
    """GptOssConfig reads an omitted rope record as YaRN; a GPT OSS loaded with a plain one exports a
    plain one, which reloads to the logits it computed."""
    directory = _respelled("gpt-oss-tiny", tmp_path / "plain", lambda config: {
        **config, "rope_parameters": {"rope_type": "default",
                                      "rope_theta": config["rope_parameters"]["rope_theta"]}})
    ids = np.load(FIXTURES / "gpt-oss-tiny" / "input_ids.npy")
    model, variables = fp32_decoder(directory)
    PretrainedDecoder.from_model(model, variables).save(tmp_path / "export")
    restored, restored_variables = fp32_decoder(tmp_path / "export")
    np.testing.assert_array_equal(np.asarray(restored.apply(restored_variables, ids)),
                                  np.asarray(model.apply(variables, ids)))


CLASS_DEFAULTED = ("deepseek-v2-tiny", "deepseek-v3-tiny", "gpt-oss-tiny", "mistral-tiny", "mixtral-tiny",
                   "ministral-tiny", "llama4-tiny", "modernbert-tiny", "qwen2-tiny", "qwen2-moe-tiny",
                   "qwen3-tiny", "qwen3-moe-tiny", "gemma2-tiny", "gemma3-tiny", "olmo3-tiny", "phi3-tiny",
                   "glm4-moe-tiny", "muse-glimmer-text-tiny")
"""Fixtures of families whose transformers config class states their defaults."""
_IDENTITY = frozenset({"model_type", "architectures", "transformers_version", "dtype", "torch_dtype",
                       "auto_map", "_name_or_path"})


def _class_fills(name):
    """For each field of the fixture's transformers config class, what the class fills it with where
    the fixture's config leaves it out, everything else as stated: a fixed default, or one derived
    from the other fields (a layer pattern from the layer count). Each is a JSON value a file could
    state."""
    from huggingface_hub.errors import StrictDataclassClassValidationError
    from transformers import CONFIG_MAPPING

    config = fixture_config(name)
    cls = CONFIG_MAPPING[config["model_type"]]
    stated = {field: value for field, value in config.items() if field not in _IDENTITY}
    fills = {}
    for field in cls().to_dict():
        if field in _IDENTITY or field.startswith("_"):
            continue
        try:
            filled = cls(**{key: value for key, value in stated.items() if key != field}).to_dict().get(field)
        except (ValueError, StrictDataclassClassValidationError):
            continue  # the class refuses the config without it: there is no fill to read
        if json.loads(json.dumps(filled, default=lambda value: None)) == filled:
            fills[field] = filled
    return fills


def _translated(config):
    """The decoder a config translates to, or the refusal's kind where the translation refuses it or
    finds a field it requires missing."""
    try:
        return translate_config(config)
    except (ValueError, KeyError) as refused:
        return type(refused)


@pytest.mark.parametrize("name", CLASS_DEFAULTED)
def test_a_field_left_out_reads_as_its_config_class_fills_it(name):
    """Each field of the family's config class, once stated as the class fills it and once left
    out of the fixture's config: where the field left out translates at all, it translates to the
    decoder the stated default does, because the family fills an omitted field as its class does,
    never at another family's default. A field the family requires is refused when left out, never
    read as something else. Over the fixture's own weights, one decoder computes one set of logits
    (`test_a_source_spelled_another_way_computes_the_reference_logits`)."""
    config = fixture_config(name)
    differing = []
    for field, filled in _class_fills(name).items():
        left = _translated(_without(field)(config))
        if not isinstance(left, type) and left != _translated({**config, field: filled}):
            differing.append(field)
    assert not differing, f"{name} reads {differing} otherwise than its class where they are left out"


@pytest.mark.parametrize("name, theta", [
    ("llama31-tiny", 10000.0),
    *_YARN_SOURCES,
])
def test_a_plain_rope_scaling_clears_the_ramp_rope_parameters_names(tmp_path, name, theta):
    """The mixin replaces rope_parameters with a stated rope_scaling whole,
    so an explicit plain rope_scaling over a ramp is plain rope at the
    family's default base, as that spelling stated alone is. A family
    whose mixer reads one rope entry refuses per-kind records through the
    same loader, and GPT-OSS, whose mixer reads YaRN alone, other ramps."""
    plain = {"rope_type": "default"}
    conflicting = _respelled(name, tmp_path / "conflicting",
                             lambda config: {**config, "rope_scaling": plain})
    alone = _respelled(name, tmp_path / "alone",
                       lambda config: {**config, "rope_parameters": {**plain, "rope_theta": theta}})
    ids = np.load(FIXTURES / name / "input_ids.npy")
    logits = [np.asarray(model.apply(variables, ids))
              for model, variables in (fp32_decoder(conflicting), fp32_decoder(alone))]
    np.testing.assert_array_equal(*logits)
    reference = np.load(FIXTURES / name / "logits.npy")
    assert not np.array_equal(logits[0], reference)
    if name == "llama31-tiny":
        assert np.max(np.abs(logits[0] - reference)) > 1e-3
    else:
        llama3 = {"rope_type": "llama3", "factor": 4.0, "low_freq_factor": 1.0,
                  "high_freq_factor": 4.0, "original_max_position_embeddings": 16}
        # DeepSeek's latent attention takes Llama 3.1's ramp beside YaRN's; GPT-OSS's reads YaRN alone.
        for index, entry in enumerate((*(llama3,) * name.startswith("gpt-oss"),
                                       {"full_attention": plain, "sliding_attention": plain})):
            unsupported = _respelled(name, tmp_path / f"unsupported-{index}",
                                      lambda config, entry=entry: {**config, "rope_scaling": entry})
            with pytest.raises(ValueError):
                fp32_decoder(unsupported)


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


@pytest.mark.network
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


def assert_transformers_reads_the_export(model, variables, export, tmp_path) -> None:
    """`model`'s export to `export` is a checkpoint transformers reads, computing Dew's logits over the
    Qwen3 fixture's ids."""
    PretrainedDecoder.from_model(model, variables).save(export)
    ids = np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy")
    ours = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))
    difference = float(np.max(np.abs(transformers_logits(export, ids, tmp_path) - ours)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"


def test_our_export_loads_in_transformers_with_the_same_logits(tmp_path):
    """The export is a real HF checkpoint: transformers reads it and agrees."""
    model, variables = fp32_decoder(FIXTURES / "qwen3-tiny")
    assert_transformers_reads_the_export(model, variables, tmp_path / "exported", tmp_path)


def test_a_biased_qwen3_export_carries_its_biases_into_transformers(tmp_path, rng):
    """Qwen3Attention builds q, k, v and o with bias=config.attention_bias
    (modeling_qwen3.py:225-236), so the reference applies the biases where dew
    does and a biased export is a checkpoint it reads."""
    model, variables = biased_qwen3(rng)
    export = tmp_path / "biased"
    assert_transformers_reads_the_export(model, variables, export, tmp_path)

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


@pytest.mark.parametrize("name", ["llada-tiny", "dream-tiny"])
def test_the_diffusion_fixtures_exercise_full_attention(name):
    """The release's own logits over the fixture's tokens (LLaDA's modeling_llada.py at 0f2787f, Dream's
    modeling_dream.py at 6572adb, run by tools/remote_code_reference.py under transformers 4.46.3), which
    tests/test_remote_code_releases.py holds Dew to over seeds, are a full-attention model's: a causal
    model on the same weights misses them by more than 1 (1.5 for LLaDA, 4.9 for Dream)."""
    model, variables, ids, reference, _ = _diffusion_fp32(name)
    causal = model.clone(causal=True)
    assert np.max(np.abs(np.asarray(causal.apply(variables, ids)) - reference)) > 1.0


LLADA_MOE = ('llada-moe-tiny', 'llada-moe-v2-tiny', 'llada2-tiny', 'llada21-tiny', 'llada22-tiny')


@pytest.fixture(scope='module', params=LLADA_MOE)
def llada_moe_source(request):
    """A LLaDA-MoE release's synthetic checkpoint, its forward's fields (`llada_fields`), one of its
    recording's draws for a single forward, and its recording of the release over seeds
    (tools/llada_moe_reference.py, `record_case`)."""
    name = request.param
    source = Pretrained.load(FIXTURES / name, dtype='float32', attention_impl='reference')
    path = FIXTURES / name / 'released.npz'
    return name, source, llada_draw(0, source.model.vocab_size)['ids'], llada_fields(name, 2, 16), (
        recording(path), declared_seeds(path)[name])


def assert_released_logits(name, source, fields, recorded, forward, label):
    """`forward(ids)` as exact as the release's fp32 forward over its recording's seeds (check 1), both
    measured from the release run in float64, held through Dew's own float64 forward to the recorded
    truth's sketch (`indexed_recording`)."""
    from reference_error import indexed_recording, widened

    stored, seeds = recorded
    wide_model, wide_variables = source.model.clone(dtype=jnp.float64), widened(source.variables)
    wide = jax.jit(lambda ids: {'logits': wide_model.apply(wide_variables, ids, **fields)})
    draw, reference, truth = indexed_recording(
        stored, name, seeds, inputs_of=lambda seed: llada_draw(seed, source.model.vocab_size),
        truth_fn=lambda inputs: wide(inputs['ids']))
    assert_as_exact(lambda index, inputs: {'logits': forward(inputs['ids'])}, reference, draw,
                    f'{name} {label}', seeds=seeds, truth_fn=truth)


def test_llada_moe_logits_match_the_released_model_code(llada_moe_source):
    """The pinned release's own fp32 forward, including its router and block mask, over its
    recording's seeds of two 16-token rows, whole blocks of 2.2's routing block."""
    name, source, _, fields, recorded = llada_moe_source
    forward = jax.jit(lambda tokens: source.model.apply(source.variables, tokens, **fields))
    assert_released_logits(name, source, fields, recorded, forward, 'released logits')


@pytest.mark.mesh(devices=4)
def test_llada_moe_expert_and_sequence_shards_match_released_logits(llada_moe_source):
    from dew.mesh import mesh_axes
    from dew.training import Layout, MeshSpec

    name, _, _, fields, recorded = llada_moe_source
    mesh = MeshSpec(expert=2, sequence=2).build(jax.devices()[:4])
    source = Pretrained.load(FIXTURES / name, dtype='float32', attention_impl='reference',
                              mesh=mesh, layout=Layout(min_shard=1, tolerance=1.0))
    kernels = [value for path, value in flat_tree(source.variables['params']).items()
               if '.experts.' in path]
    assert all(any('expert' in mesh_axes(axis) for axis in value.sharding.spec) for value in kernels)
    with jax.set_mesh(mesh):
        forward = jax.jit(lambda tokens: source.model.apply(source.variables, tokens, **fields))
        assert_released_logits(name, source, fields, recorded, forward, 'sharded released logits')


def test_llada_moe_routing_limits_change_the_forward(llada_moe_source):
    """The expert bias, group limit and 2.2 block shortlist each bind on real hidden states."""
    from dataclasses import replace

    name, source, ids, fields, recorded = llada_moe_source
    model, variables = source.model, source.variables
    mixture = model.mixture

    def chosen(run_model, run_variables):
        _, state = run_model.apply(run_variables, ids, **fields, mutable=['router'])
        return {layer: np.sort(np.asarray(values['mlp']['gate']['indices'][0]), axis=-1)
                for layer, values in state['router'].items()}

    original = chosen(model, variables)

    def differs(run_model, run_variables):
        changed = chosen(run_model, run_variables)
        return any(not np.array_equal(original[layer], changed[layer]) for layer in original)

    if mixture.bias:
        unbiased = {**variables, 'moe': jax.tree.map(jnp.zeros_like, variables['moe'])}
        assert differs(model, unbiased)
    if mixture.groups > 1:
        unlimited = model.clone(mixture=replace(mixture, groups=1, groups_per_token=1))
        assert differs(unlimited, variables)
    if mixture.routing_block is not None:
        unlimited = model.clone(mixture=replace(mixture, block_experts=mixture.experts))
        assert differs(unlimited, variables)
    # Every fixture also distinguishes a causal decoder from the release's attention.
    causal = model.clone(causal=True)
    forward = jax.jit(lambda tokens: causal.apply(variables, tokens))
    with pytest.raises(AssertionError):
        assert_released_logits(name, source, fields, recorded, forward, 'causal negative control')


def llada22_routes(source, tokens, **fields) -> dict:
    """Each routed layer's chosen experts `[B, S, top_k]`, sorted, for `tokens` under `fields`."""
    _, state = source.model.apply(source.variables, jnp.asarray(tokens, jnp.int32), **fields,
                                  mutable=['router'])
    return {layer: np.sort(np.asarray(values['mlp']['gate']['indices'][0]), axis=-1)
            for layer, values in state['router'].items()}


def test_llada22_block_routing_is_each_requests_own():
    """LLaDA2.2 routes each block of `block_size` places together. The release cuts its blocks from
    the batch flattened into one stream, so there a row's experts depend on the rows beside it; Dew
    numbers each request's blocks over its own real tokens (`numbered_blocks`), the release's blocks
    wherever a row is whole blocks (the recorded draws). Two requests 6 and 9 tokens long, neither a
    whole number of 4-token blocks, batched with left padding, route every real token to the experts
    it routes to alone: their blocks do not meet, and the padding votes for none. (The recorded draws
    are two 16-token rows, whole blocks, where Dew's routing is the release's.) Eight requests of
    5 to 15 tokens make the stream's blocks straddle rows and padding throughout."""
    source = Pretrained.load(FIXTURES / 'llada22-tiny', dtype='float32', attention_impl='reference')
    assert source.model.mixture.routing_block == 4
    vocabulary = source.model.vocab_size
    requests = [llada_draw(seed, vocabulary)['ids'][seed % 2, :length]
                for seed, length in enumerate((6, 9, 5, 11, 7, 13, 10, 15), start=1)]
    width = max(len(request) for request in requests)
    padded = np.stack([np.pad(request, (width - len(request), 0)) for request in requests])
    valid = np.stack([np.arange(width) >= width - len(request) for request in requests])
    batched = llada22_routes(source, padded, attention_mask=jnp.asarray(valid))
    assert batched
    for row, request in enumerate(requests):
        alone = llada22_routes(source, request[None])
        for layer, chosen in alone.items():
            np.testing.assert_array_equal(batched[layer][row, width - len(request):], chosen[0],
                                          err_msg=f'request {row}, {layer}')


def test_a_block_router_refuses_a_cached_decode_by_name():
    """A block's shortlist is chosen from every token in it, which a cached decode step holding one
    token of its block cannot see, so a causal decoder with a block router refuses autoregressive
    generation by name rather than route its steps unlike its full forward."""
    from dataclasses import replace

    from dew.sampling.text import generate

    source = Pretrained.load(FIXTURES / 'llada22-tiny', dtype='float32', attention_impl='reference')
    causal = source.model.clone(causal=True)
    unblocked = causal.clone(mixture=replace(causal.mixture, routing_block=None, block_experts=None))
    assert unblocked.generation_refusal() is None and causal.generation_refusal() is not None
    with pytest.raises(ValueError):
        generate(causal, source.variables, jnp.ones((1, 4), jnp.int32), 2, key=0)


def test_llada22_block_routing_is_each_packed_documents_own():
    """Two documents packed in one row, 6 and 7 tokens, route every token to the experts it routes to
    alone: each document's blocks are counted from its own first token, so neither a block nor its
    shortlist spans the boundary."""
    source = Pretrained.load(FIXTURES / 'llada22-tiny', dtype='float32', attention_impl='reference')
    vocabulary = source.model.vocab_size
    documents = [llada_draw(3, vocabulary)['ids'][0, :6], llada_draw(4, vocabulary)['ids'][1, :7]]
    packed = np.concatenate(documents)[None]
    segments = np.concatenate([np.full(len(document), index + 1) for index, document in enumerate(documents)])
    together = llada22_routes(source, packed, segment_ids=jnp.asarray(segments[None], jnp.int32))
    start = 0
    for index, document in enumerate(documents):
        alone = llada22_routes(source, document[None])
        for layer, chosen in alone.items():
            np.testing.assert_array_equal(together[layer][0, start:start + len(document)], chosen[0],
                                          err_msg=f'document {index}, {layer}')
        start += len(document)


@pytest.mark.parametrize('derived', [False, True], ids=['source-layout', 'native-layout'])
def test_llada_moe_trained_exports_reload_and_run(llada_moe_source, tmp_path, derived):
    name, source, ids, fields, _ = llada_moe_source
    model, variables = source.model, source.variables
    draws = np.stack([llada_draw(seed, model.vocab_size)['ids'] for seed in range(40)])
    loss, gradients = jax.value_and_grad(lambda params: jnp.mean(model.apply(
        {**variables, 'params': params}, ids, **fields) ** 2))(variables['params'])
    assert np.isfinite(loss) and all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients))
    trained = {**variables, 'params': jax.tree.map(lambda p, g: p - 0.01 * g,
                                                 variables['params'], gradients)}
    before = model.apply(variables, ids, **fields)
    after = model.apply(trained, ids, **fields)
    with pytest.raises(AssertionError):
        assert_as_exact(lambda seed: after, lambda seed: before,
                        lambda seed: (np.asarray(seed, np.int32),), name + ' update negative control',
                        truth_fn=lambda seed: np.asarray(before, np.float64), seeds=8)
    if derived:
        PretrainedDecoder.from_model(model, trained).save(tmp_path)
    else:
        source.save(tmp_path, variables=trained, max_shard_size='30KB')
    reloaded = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    actual = jax.jit(lambda tokens: reloaded.model.apply(reloaded.variables, tokens, **fields))
    expected = jax.jit(lambda tokens: model.apply(trained, tokens, **fields))
    # Saving and loading is an identity, so the reference is exact at every draw.
    assert_as_exact(as_given(lambda seed: actual(draws[int(seed)])),
                    as_given(lambda seed: expected(draws[int(seed)])),
                    lambda seed: (np.asarray(seed, np.int32),), name + ' trained export', seeds=8,
                    truth_fn=lambda seed: np.asarray(expected(draws[int(seed)]), np.float64))
    prompt = ids[:1, :4]
    generated = reloaded.text_generation()(prompt, 4, key=7, steps=3).host().tokens
    np.testing.assert_array_equal(generated[:, :4], prompt)
    assert generated.shape == (1, 8) and np.all(generated[:, 4:] != model.mask_token_id)


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


def assert_padded_speculation_is_greedy(source) -> None:
    """Three left-padded rows of a source's recorded ids (0, 2 and 4 pads) generate the same seven
    greedy tokens, lengths and terminations under speculation (`Speculative(block=3)`) as without."""
    from dew.nn.inputs import ModelInputs
    from dew.sampling import Sampling, Speculative

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


def test_glm5_prediction_index_reuse_preserves_public_padded_greedy_generation(glm5_next_source):
    assert_padded_speculation_is_greedy(glm5_next_source)


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


def nemotron_step_source(directory, *, seed: int = 7):
    """The latent-MoE fixture with Nemotron 3 Ultra's prediction step, one
    depth of an attention block and a routed latent block
    (mtp_layers_block_type [attention, moe]), its tensors drawn under the
    released names `mtp.layers.{0,1}.*` (vLLM nemotron_h_mtp.py), written to
    `directory`; and the reference: transformers' own blocks composed as
    vLLM composes the step, at `dtype`."""
    import torch
    from safetensors.numpy import load_file, save_file
    from transformers.models.nemotron_h import modeling_nemotron_h as reference
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    source = FIXTURES / "nemotron-h-moe-latent-tiny"
    hf = {**fixture_config("nemotron-h-moe-latent-tiny"), "num_nextn_predict_layers": 1,
          "mtp_layers_block_type": ["attention", "moe"]}
    trunk = hf["layers_block_type"]
    # The step's blocks are built as layers past the trunk, as vLLM builds them.
    extended = NemotronHConfig.from_dict({**NemotronHConfig.from_pretrained(str(source)).to_dict(),
                                          "layers_block_type": [*trunk, "full_attention", "moe"]})
    extended._attn_implementation, extended._experts_implementation = "eager", "eager"
    torch.manual_seed(seed)
    step = torch.nn.ModuleDict({
        "enorm": reference.NemotronHRMSNorm(hf["hidden_size"], eps=hf["layer_norm_epsilon"]),
        "hnorm": reference.NemotronHRMSNorm(hf["hidden_size"], eps=hf["layer_norm_epsilon"]),
        "eh_proj": torch.nn.Linear(2 * hf["hidden_size"], hf["hidden_size"], bias=False),
        "attention": reference.NemotronHBlock(extended, layer_idx=len(trunk)),
        "moe": reference.NemotronHBlock(extended, layer_idx=len(trunk) + 1),
        "final_layernorm": reference.NemotronHRMSNorm(hf["hidden_size"], eps=hf["layer_norm_epsilon"]),
    })
    with torch.no_grad():
        for parameter in step.parameters():
            parameter.normal_(0.0, 0.3)
        bias = step["moe"].mixer.gate.e_score_correction_bias
        bias.copy_(torch.linspace(-0.4, 0.4, hf["n_routed_experts"]))
    names = {"enorm.weight": "mtp.layers.0.enorm.weight", "hnorm.weight": "mtp.layers.0.hnorm.weight",
             "eh_proj.weight": "mtp.layers.0.eh_proj.weight",
             "final_layernorm.weight": "mtp.layers.1.final_layernorm.weight"}
    tensors = load_file(str(source / "model.safetensors"))
    for name, tensor in step.state_dict().items():
        block, _, rest = name.partition(".")
        stored = tensor.to(torch.float32).numpy()
        if name in names:
            tensors[names[name]] = stored
        elif rest.startswith("mixer.experts."):
            # The release stores one tensor per expert.
            projection = rest.removeprefix("mixer.experts.")
            for expert, matrix in enumerate(stored):
                tensors[f"mtp.layers.1.mixer.experts.{expert}.{projection}.weight"] = matrix
        else:
            tensors[f"mtp.layers.{0 if block == 'attention' else 1}.{rest}"] = stored
    directory.mkdir()
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(hf))

    def logits(ids, dtype):
        model = reference.NemotronHForCausalLM.from_pretrained(str(source), dtype=dtype).eval()
        # The eager experts run in float64, where torch's grouped matmul does not.
        model.config._experts_implementation = "eager"
        depth = step.to(dtype)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
            fused = depth["eh_proj"](torch.cat([depth["enorm"](model.model.embeddings(ids[:, 1:])),
                                                depth["hnorm"](hidden)], dim=-1))
            positions = torch.arange(fused.shape[1])[None]
            mask = reference.create_causal_mask(config=extended, inputs_embeds=fused, attention_mask=None,
                                                past_key_values=None, position_ids=positions)
            out = depth["moe"](depth["attention"](fused, attention_mask=mask, position_ids=positions))
            return model.lm_head(depth["final_layernorm"](out)).double().numpy()
    return logits


def test_a_nemotron_h_prediction_step_loads_computes_the_reference_and_trains(tmp_path):
    """Nemotron 3 Ultra's prediction step over the latent-MoE fixture loads as
    `mtp_0`: the attention block's mixer and pre-norm, the routed block in
    its feed-forward slot, the step's input maps and final norm. Its logits
    are what transformers' own blocks give composed as vLLM composes the
    step, within the reference's own distance from float64; and the LM
    objective's `mtp_weight` trains the step's weights."""
    from unittest.mock import patch

    import torch

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from tools.torch_precision import float64

    reference = nemotron_step_source(tmp_path / "ultra-step")
    model, variables = fp32_decoder(tmp_path / "ultra-step")
    depth = variables["params"]["mtp_0"]
    assert set(depth) == {"enorm", "hnorm", "eh_proj", "block", "final_norm"}
    assert {"input_layernorm", "self_attn", "post_attention_layernorm", "mlp"} <= set(depth["block"])
    assert depth["block"]["mlp"]["experts"]["up_proj"]["kernel"].shape[0] == 8
    assert "mtp_0" in variables["moe"]

    ids = np.load(FIXTURES / "nemotron-h-moe-latent-tiny" / "input_ids.npy")
    hidden = model.apply(variables, jnp.asarray(ids, jnp.int32), method=CausalTransformer.hidden_states)
    logits = np.asarray(model.apply(variables, hidden, jnp.asarray(ids, jnp.int32),
                                    method=CausalTransformer.mtp_logits)[0])
    tokens = torch.from_numpy(ids.astype(np.int64))
    fp32 = reference(tokens, torch.float32)
    widen = torch.Tensor.type

    def wide_type(tensor, dtype=None, **kwargs):
        # Nemotron's router pins .type(torch.float32), beyond the shared cast widener.
        return widen(tensor, torch.float64 if dtype == torch.float32 else dtype, **kwargs)

    with float64(), patch.object(torch.Tensor, "type", wide_type):
        truth = reference(tokens, torch.float64)
    assert_as_exact_as_the_reference(logits, fp32, truth, "nemotron-h prediction step logits")
    assert np.array_equal(np.argmax(logits, -1), np.argmax(fp32, -1))

    import optax

    from dew.objectives.lm import LMObjective
    from dew.training import Layout, Trainer

    objective = LMObjective(model, ids.shape[1] - 1, mtp_weight=0.5, variables=variables)
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0), checkpoints=None, tracker=None)
    state, _, _ = trainer.place()
    before = jax.device_get(state.variables["params"]["mtp_0"])
    batch = {"text": jnp.asarray(np.tile(ids, (max(1, jax.device_count() // len(ids)), 1)), jnp.int32)}
    trained, *_ = trainer.compile(state, batch)(state, batch)
    moved = jax.tree.map(lambda old, new: bool(np.any(old != np.asarray(new))), before,
                         jax.device_get(trained.variables["params"]["mtp_0"]))
    assert all(jax.tree.leaves(moved)), moved


def test_a_nemotron_h_load_keeps_its_prediction_steps_whole(tmp_path):
    """A Nemotron-H prediction step is its blocks together, an attention block and a routed one under
    `mtp.layers.{0,1}.*` (`step_depth`): `prediction_depths=1` keeps a one-step source whole, and of a
    two-step source, its second step the first's blocks again under `mtp.layers.{2,3}.*`, keeps the
    first step alone, as the one-step source loads; 0 keeps the trunk."""
    from safetensors.numpy import load_file, save_file

    def load(directory, **options):
        return Pretrained.load(str(directory), dtype="float32", attention_impl="reference", **options)

    def same(loaded, other) -> bool:
        ours, theirs = flat_tree(loaded.variables), flat_tree(other.variables)
        return ours.keys() == theirs.keys() and all(np.array_equal(ours[key], theirs[key]) for key in ours)

    nemotron_step_source(tmp_path / "one")
    one = load(tmp_path / "one")
    assert one.model.num_nextn_predict_layers == 1
    assert same(load(tmp_path / "one", prediction_depths=1), one)
    two = tmp_path / "two"
    two.mkdir()
    tensors = load_file(str(tmp_path / "one" / "model.safetensors"))
    for name in [name for name in tensors if name.startswith("mtp.layers.")]:
        index, rest = name.removeprefix("mtp.layers.").split(".", 1)
        tensors[f"mtp.layers.{int(index) + 2}.{rest}"] = tensors[name]
    save_file(tensors, str(two / "model.safetensors"))
    config = json.loads((tmp_path / "one" / "config.json").read_text())
    (two / "config.json").write_text(json.dumps({**config, "num_nextn_predict_layers": 2}))
    first = load(two, prediction_depths=1)
    assert first.model == one.model and same(first, one)
    trunk = load(two, prediction_depths=0)
    assert trunk.model.num_nextn_predict_layers == 0
    assert not any("mtp_" in key for key in flat_tree(trunk.variables))


@pytest.mark.parametrize("step", [
    {"mtp_layers_block_type": ["attention", "moe"]},
    {"mtp_layers_block_type": None, "mtp_hybrid_override_pattern": "*E"},
    {"mtp_layers_block_type": ["full_attention", "moe"], "mtp_hybrid_override_pattern": "*"},
    {"mtp_layers_block_type": ["full_attention"], "mtp_hybrid_override_pattern": "*E"},
])
def test_nemotron_h_reads_the_prediction_step_transformers_reads(step):
    """The step's blocks are the ones NemotronHConfig resolves from the list
    and the legacy pattern, which overrides the list where the list is its
    default."""
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    config = {**fixture_config("nemotron-h-moe-tiny"), "num_nextn_predict_layers": 1, **step}
    model = translate_config(config).value
    reference = NemotronHConfig.from_dict({**NemotronHConfig.from_pretrained(
        str(FIXTURES / "nemotron-h-moe-tiny")).to_dict(), **step, "num_nextn_predict_layers": 1})
    blocks = (model.mtp_layer_type, *((model.mtp_feedforward_type,) if model.mtp_feedforward_type else ()))
    assert list(blocks) == list(reference.mtp_layers_block_type)


@pytest.mark.parametrize("name", ["nemotron-h-tiny", "nemotron-h-moe-tiny"])
def test_a_nemotron_h_trunk_whose_config_names_a_step_loads_as_the_trunk(name, tmp_path):
    """transformers saves the trunk alone whatever depth the config names:
    such a checkpoint, the dense one too, whose default step names a routed
    kind its trunk lacks, loads the trunk it holds and computes its logits."""
    from shutil import copytree

    directory = tmp_path / name
    copytree(FIXTURES / name, directory)
    stated = {**fixture_config(name), "num_nextn_predict_layers": 1}
    (directory / "config.json").write_text(json.dumps(stated))
    model, variables = fp32_decoder(directory)
    assert model.num_nextn_predict_layers == 0 and "mtp_0" not in variables["params"]
    ids = jnp.asarray(np.load(FIXTURES / name / "input_ids.npy"), jnp.int32)
    plain, held = fp32_decoder(FIXTURES / name)
    np.testing.assert_array_equal(np.asarray(model.apply(variables, ids)), np.asarray(plain.apply(held, ids)))


def test_a_bundle_made_from_a_model_carries_its_record_as_a_loaded_one_does():
    """`PretrainedDecoder.from_model`'s model_config is the decoder's record, the form a load's
    takes: written as JSON and read back, it builds the same decoder."""
    loaded = Pretrained.load(str(FIXTURES / "qwen3-tiny"), dtype="float32", attention_impl="reference")
    bundle = PretrainedDecoder.from_model(loaded.model, loaded.variables)
    assert models.build("causal_transformer", **json.loads(json.dumps(bundle.model_config))) == loaded.model


def test_a_float64_decoder_saves_and_reloads_in_float64(tmp_path):
    """A decoder computing in float64 end to end, as Dory's and nGPT's runs do: its bundle and a
    run's record of it name float64, and the checkpoint it saves loads back in float64 to the same
    weights and logits."""
    from dew.registry import ModelConfig, to_record

    with jax.enable_x64(new_val=True):
        loaded = Pretrained.load(str(FIXTURES / "qwen3-tiny"), dtype="float64", param_dtype="float64",
                                 attention_impl="reference")
        bundle = PretrainedDecoder.from_model(loaded.model, loaded.variables)
        record = json.loads(json.dumps(bundle.model_config))
        assert models.build("causal_transformer", **record) == loaded.model
        # A run records its model as every run does (`ModelConfig.from_model`), and rebuilds it.
        run = json.loads(json.dumps(to_record(ModelConfig.from_model(loaded.model), ModelConfig)))
        assert ModelConfig.from_dict(run).build() == loaded.model
        bundle.save(tmp_path / "float64")
        again = Pretrained.load(str(tmp_path / "float64"), dtype="float64", param_dtype="float64",
                                attention_impl="reference")
        pairs = zip(jax.tree.leaves(again.variables), jax.tree.leaves(loaded.variables), strict=True)
        for ours, theirs in pairs:
            np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
        ids = jnp.asarray(np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy"))
        np.testing.assert_array_equal(np.asarray(again.model.apply(again.variables, ids)),
                                      np.asarray(loaded.model.apply(loaded.variables, ids)))


@pytest.mark.parametrize("fixture", ["qwen2-tiny", "qwen3-tiny"])
def test_a_qwen_config_without_a_window_bound_windows_the_layers_transformers_windows(fixture):
    """A 32-layer Qwen config that turns the window on but states neither `layer_types` nor
    `max_window_layers` windows the layers its own transformers class does (the last four, from the
    class's default bound of 28), so the model reads what transformers' model reads."""
    from transformers import AutoConfig

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.registry import from_record

    config = json.loads((FIXTURES / fixture / "config.json").read_text())
    unbound = ("layer_types", "max_window_layers")
    stated = {name: value for name, value in config.items() if name not in unbound}
    config = {**stated, "num_hidden_layers": 32, "use_sliding_window": True, "sliding_window": 4096}
    model = from_record(CausalTransformer, translate_config(config))
    expected = AutoConfig.for_model(config["model_type"], **{
        name: value for name, value in config.items() if name != "model_type"}).layer_types
    windows = tuple("sliding_attention" if model.kind_of(kind).window else "full_attention"
                    for kind in model.per_layer_types)
    assert windows == tuple(expected)
    assert windows.count("sliding_attention") == 4


def test_a_config_of_a_type_transformers_lacks_reads_its_layers_by_its_familys_convention():
    """Dream (no transformers class) reads by Qwen2's convention, whose class takes a null head_dim
    as Dream's own config does: the record is the one without it."""
    config = json.loads((FIXTURES / "dream-tiny" / "config.json").read_text())
    assert translate_config({**config, "head_dim": None}) == translate_config(
        {name: value for name, value in config.items() if name != "head_dim"})
