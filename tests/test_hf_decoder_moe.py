"""Hugging Face decoders with routed experts or latent attention: GPT OSS,
DeepSeek V2/V4, Kimi K2 and K2.5, GLM 4.5 and GLM-5, Llama 4 and Gemma 4's
routed size, each against transformers' logits (see test_hf_decoders).
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from model_support import flat_tree
from reference_error import FACTOR, assert_as_exact_as_the_reference, distance
from test_hf_decoders import (
    GEMMA4_MOE,
    assert_padded_speculation_is_greedy,
    fixture_config,
    fp32_decoder,
    recorded_logit_difference,
)

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.codecs import dequantize_mxfp4, quantize_mxfp4
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.nn.backbones.causal_transformer import CausalTransformer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

ROUTED = {'qwen2-moe-tiny': 'Qwen2-MoE', 'granitemoe-tiny': 'Granite MoE', 'minimax-m2-tiny': 'MiniMax-M2'}
"""Routed decoders whose recorded fixtures carry the reference's float32 and float64 logits, whole,
padded and cached: Qwen2-MoE's fused experts and shared gate, Granite MoE's packed experts, MiniMax-M2's
router state, recorded from the corrected reference."""


@pytest.mark.parametrize('name', (*ROUTED, 'minimax-m2-fp8-tiny'))
def test_a_routed_decoders_logits_match_the_reference(name):
    directory = FIXTURES / name
    model, variables = fp32_decoder(directory)
    reference = np.load(directory / 'logits.npy')
    logits = np.asarray(model.apply(variables, np.load(directory / 'input_ids.npy')))
    truth = np.load(directory / 'logits_f64.npy')
    assert_as_exact_as_the_reference(logits, reference, truth, f'{name} logits')
    np.testing.assert_array_equal(logits.argmax(-1), reference.argmax(-1))


@pytest.mark.parametrize('name', ROUTED)
def test_a_routed_decoders_prefill_and_token_steps_match_the_reference(name):
    directory = FIXTURES / name
    model, variables = fp32_decoder(directory)
    ids = np.load(directory / 'input_ids.npy')
    state = model.apply(variables, ids.shape[0], method='init_cache', mutable=['cache'])[1]
    out, state = model.apply({**variables, **state}, ids[:, :4], decode=True, mutable=['cache'])
    pieces = [np.asarray(out)]
    for index in range(4, ids.shape[1]):
        out, state = model.apply({**variables, **state}, ids[:, index:index + 1],
                                 decode=True, mutable=['cache'])
        pieces.append(np.asarray(out))
    assert_as_exact_as_the_reference(np.concatenate(pieces, axis=1), np.load(directory / 'logits.npy'),
                                     np.load(directory / 'logits_f64.npy'), f'{ROUTED[name]} cache')


@pytest.mark.parametrize('name', ROUTED)
def test_a_routed_decoders_padded_routing_matches_the_reference(name):
    model, variables = fp32_decoder(FIXTURES / name)
    arrays = np.load(FIXTURES / name / 'padded.npz')
    logits = np.asarray(model.apply(
        variables, arrays['input_ids'], positions=arrays['position_ids'],
        attention_mask=arrays['attention_mask']))
    valid = arrays['attention_mask']
    assert_as_exact_as_the_reference(logits[valid], arrays['logits'][valid],
                                     arrays['logits_f64'][valid], f'{ROUTED[name]} padded')


@pytest.mark.parametrize('name', ROUTED)
def test_a_routed_decoders_export_reloads_its_weights_and_source_config(name, tmp_path):
    """The export reloads to the same weights and writes the source's config, Qwen2-MoE's fused experts
    and shared gate, Granite MoE's packed experts and MiniMax-M2's router state included."""
    loaded = Pretrained.load(str(FIXTURES / name), dtype='float32', attention_impl='reference')
    loaded.save(tmp_path / 'export')
    again = Pretrained.load(str(tmp_path / 'export'), dtype='float32', attention_impl='reference')
    for path, leaf in flat_tree(loaded.variables).items():
        np.testing.assert_array_equal(leaf, flat_tree(again.variables)[path])
    assert json.loads((tmp_path / 'export/config.json').read_text()) == fixture_config(name)


def test_granitemoe_export_writes_packed_experts_and_ties_its_head(tmp_path):
    """Granite MoE's export writes its experts packed and its head tied, as its source does."""
    from safetensors.numpy import load_file

    Pretrained.load(str(FIXTURES / 'granitemoe-tiny'), dtype='float32', attention_impl='reference').save(
        tmp_path / 'export')
    tensors = load_file(str(tmp_path / 'export/model.safetensors'))
    assert 'model.layers.0.block_sparse_moe.input_linear.weight' in tensors
    assert 'lm_head.weight' not in tensors


# --------------------------------------------------------------------------
# GPT OSS: attention sinks, biased interleaved experts, YaRN over GQA
# --------------------------------------------------------------------------

GPT_OSS = FIXTURES / "gpt-oss-tiny"


def test_gpt_oss_logits_match_the_reference_implementation():
    """fp32 parity through xla sink attention: tolerance 1e-4, observed
    max |logit difference| 2.5e-06 with identical argmax."""
    pretrained = Pretrained.load(str(GPT_OSS), dtype="float32", attention_impl="xla")
    model, variables = pretrained.model, pretrained.variables
    difference = recorded_logit_difference(model, variables, GPT_OSS)
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    flat = flat_tree(variables["params"])
    assert flat["layers_0.self_attn.sinks"].shape == (4,)
    assert flat["layers_1.mlp.experts.gate_up_proj_bias"].shape == (4, 128)


def test_gpt_oss_export_round_trips_sinks_and_fused_experts(tmp_path):
    pretrained = Pretrained.load(str(GPT_OSS), dtype="float32", attention_impl="xla")
    model, variables = pretrained.model, pretrained.variables
    export = tmp_path / "gpt-oss"
    PretrainedDecoder.from_model(model, variables).save(export)
    round_trip = Pretrained.load(str(export), dtype="float32", attention_impl="xla")
    again, reloaded = round_trip.model, round_trip.variables
    assert again == model
    for path, leaf in flat_tree(reloaded["params"]).items():
        assert np.array_equal(np.asarray(leaf), np.asarray(flat_tree(variables["params"])[path])), path


def test_an_mxfp4_gpt_oss_checkpoint_loads_through_the_dequantization(tmp_path):
    """The released layout: uint8 blocks and scales in place of the expert
    matrices. Loading them must give the logits of the fixture whose
    experts are those exact bf16 values, so the packed checkpoint is
    written from the fixture's own weights."""
    from safetensors.numpy import load_file, save_file

    tensors = load_file(str(GPT_OSS / "model.safetensors"))
    packed = {}
    for name, tensor in tensors.items():
        if not (name.endswith(("gate_up_proj", "down_proj"))):
            packed[name] = tensor
            continue
        blocks, scales = (np.asarray(part) for part in quantize_mxfp4(jnp.asarray(tensor)))
        packed[name + "_blocks"], packed[name + "_scales"] = blocks, scales
        tensors[name] = np.asarray(dequantize_mxfp4(jnp.asarray(blocks), jnp.asarray(scales))
                                   .astype(jnp.float32))
    directory = tmp_path / "packed"
    directory.mkdir()
    save_file(packed, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(
        {**fixture_config("gpt-oss-tiny"), "quantization_config": {"quant_method": "mxfp4"}}))

    pretrained = Pretrained.load(str(directory), dtype="float32", attention_impl="xla")
    _model, variables = pretrained.model, pretrained.variables
    expected = translate_weights(tensors, translate_config(fixture_config("gpt-oss-tiny")))
    for path, leaf in flat_tree(variables["params"]).items():
        assert np.array_equal(np.asarray(leaf), flat_tree(expected["params"])[path]), path


GPT_OSS_20B = ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee")
"""The released GPT OSS 20B and the commit both sides read."""


@pytest.mark.network
def test_gpt_oss_20b_matches_transformers_on_the_real_weights():
    """The released MXFP4 checkpoint through the loader against transformers'
    own dequantized bf16 forward at the same prompt. Tolerance on the top-32
    logits 5e-2 for bf16 accumulation over 24 layers; the argmax must agree
    everywhere. The largest observed difference is not recorded here: the
    test needs the download. The weights stay in the bf16 the MXFP4 blocks
    decode to (`param_dtype="auto"`), as the reference holds them: 42 GB on
    the device, where FP32 master weights would take 84."""
    import torch
    from transformers import AutoTokenizer, GptOssForCausalLM

    repo, revision = GPT_OSS_20B
    prompt = "The Cascade Range runs from northern California through Oregon"
    ids = AutoTokenizer.from_pretrained(repo, revision=revision)(
        prompt, return_tensors="np")["input_ids"].astype(np.int32)
    pretrained = Pretrained.load(repo, dtype="bfloat16", param_dtype="auto", attention_impl="xla",
                                 max_seq_len=int(ids.shape[1]), revision=revision)
    model, variables = pretrained.model, pretrained.variables
    logits = np.asarray(model.apply(variables, jnp.asarray(ids)), np.float32)[0]

    reference = GptOssForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, revision=revision)
    reference.eval()
    reference.set_attn_implementation("eager")
    with torch.no_grad():
        theirs = reference(input_ids=torch.from_numpy(ids.astype(np.int64))).logits[0].float().numpy()

    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(theirs, axis=-1))
    top_ids = np.argsort(-theirs, axis=-1)[:, :32]
    difference = float(np.max(np.abs(np.take_along_axis(logits, top_ids, axis=-1)
                                     - np.take_along_axis(theirs, top_ids, axis=-1))))
    assert difference < 5e-2, f"max |top-32 logit difference| {difference:.3e}"



# --------------------------------------------------------------------------
# DeepSeek V2 and Kimi K2: the V2 router under MLA, and Kimi's own release
# --------------------------------------------------------------------------

DEEPSEEK_V2 = FIXTURES / "deepseek-v2-tiny"


def test_deepseek_v2_logits_match_the_reference_implementation():
    """fp32 parity: tolerance 1e-4, observed max |logit difference| 2.3e-06
    with identical argmax."""
    model, variables = fp32_decoder(DEEPSEEK_V2)
    difference = recorded_logit_difference(model, variables, DEEPSEEK_V2)
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert "moe" not in variables, "V2 keeps no balancing bias"


# --------------------------------------------------------------------------
# Kimi K2.5: a vision wrapper whose decoder is Kimi K2's, text half only
# --------------------------------------------------------------------------

KIMI_K25 = FIXTURES / "kimi-k25-tiny"


def test_a_kimi_k25_export_writes_the_source_names_and_the_vision_bytes(tmp_path):
    """The whole checkpoint comes back out: the decoder from the parameter
    tree under the release's names, the tower and the projector from the
    bytes they were retained as, and the source's own config beside them."""
    loaded = Pretrained.load(str(KIMI_K25), dtype="float32", attention_impl="reference")
    from dew.interop.sources import load_shards

    destination = tmp_path / "export"
    loaded.save(destination)
    exported = load_shards(destination)
    source = load_shards(KIMI_K25)

    assert set(exported) == set(source)
    for name, tensor in source.items():
        np.testing.assert_array_equal(exported[name], tensor, err_msg=name)
    assert (json.loads((destination / "config.json").read_text())
            == fixture_config("kimi-k25-tiny"))
    again = Pretrained.load(str(destination), dtype="float32", attention_impl="reference")
    ids = jnp.asarray(np.load(KIMI_K25 / "input_ids.npy"), jnp.int32)
    np.testing.assert_array_equal(np.asarray(again.model.apply(again.variables, ids)),
                                  np.asarray(loaded.model.apply(loaded.variables, ids)))

    for name, tensor in loaded.retained_tensors.items():
        np.testing.assert_array_equal(tensor, source[name])


# --------------------------------------------------------------------------
# GLM 4.5: a half rotary over biased GQA, DeepSeek V3 routing, an MTP depth
# --------------------------------------------------------------------------

GLM4_MOE = FIXTURES / "glm4-moe-tiny"


def test_the_glm4_moe_mtp_depth_matches_the_reference_composition():
    """The checkpoint's model.layers.2.* depth loads as mtp_0 with the
    trunk's routing and computes what the engines running the released
    weights compute (tools/hf_reference_b.py Glm4MoeMTP): tolerance 1e-4,
    observed max |logit difference| 3.2e-06 with identical argmax. Swapping
    the two norms, the order today's from-scratch code had, disagrees."""

    variables, depth, depth_logits = mtp_depth(GLM4_MOE, experts=(8, 32, 16))
    assert "mtp_0" in variables["moe"]

    swapped = {**variables, "params": {**variables["params"], "mtp_0": {
        **depth, "enorm": depth["hnorm"], "hnorm": depth["enorm"]}}}
    other = depth_logits(swapped)
    assert float(np.max(np.abs(other - np.load(GLM4_MOE / "mtp_logits.npy")))) > 1e-2


def mtp_depth(directory, *, experts):
    """A checkpoint's MTP depth, loaded as `mtp_0` with its routed experts' `experts` shape, computing
    the reference composition's recorded logits over the fixture's ids (mtp_logits.npy): its variables,
    the depth's parameters and the depth's logits under any variables."""
    model, variables = fp32_decoder(directory)
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    depth = variables["params"]["mtp_0"]
    assert set(depth) == {"enorm", "hnorm", "eh_proj", "block", "final_norm"}
    assert depth["block"]["mlp"]["experts"]["gate_proj"]["kernel"].shape == experts

    def depth_logits(held):
        hidden = model.apply(held, ids, method=CausalTransformer.hidden_states)
        return np.asarray(model.apply(held, hidden, ids, method=CausalTransformer.mtp_logits)[0])

    logits, reference = depth_logits(variables), np.load(directory / "mtp_logits.npy")
    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |mtp logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))
    return variables, depth, depth_logits


# --------------------------------------------------------------------------
# GLM-5 (glm_moe_dsa): V3.2's sparse MLA with an interleaved indexer,
# IndexShare layers and the MTP depth
# --------------------------------------------------------------------------

GLM_MOE_DSA = FIXTURES / "glm-moe-dsa-tiny"


FP8_TENSORS = {'quant_method': 'compressed-tensors', 'format': 'float-quantized', 'ignore': [],
               'config_groups': {'group_0': {'targets': ['Linear'], 'weights': {
                   'num_bits': 8, 'type': 'float', 'strategy': 'tensor', 'symmetric': True,
                   'dynamic': False}}}}
"""Per-tensor E4M3 weights as compressed-tensors describes them."""


@pytest.mark.parametrize(("name", "quantized"),
                         [("kimi-k25-tiny", ".mlp.experts."), ("gpt-oss-tiny", ".self_attn.")])
def test_a_family_reads_quantized_weights_through_the_shared_codec(tmp_path, name, quantized):
    """Kimi K2.5 describes its text weights' storage under text_config, and a GPT OSS source may
    store its attention in another format than the released MXFP4 experts; the shared codec reads
    either description, so the source loads as its weights decoded do."""
    from dew.interop import codecs
    from dew.interop.safetensors_io import read_weights, save_hf_layout

    stored, dense = {}, {}
    for key, value in read_weights(FIXTURES / name).items():
        if quantized in key and key.endswith('.weight') and value.ndim == 2:
            # One E4M3 scale over the whole tensor, as compressed-tensors calibrates it.
            scale = np.asarray(np.abs(value).max() / 448, np.float32)
            codec = codecs.source_quantization({'quantization_config': FP8_TENSORS},
                                               grid={key.removesuffix('weight') + 'weight_scale': scale})
            assert codec is not None
            encoded = dict(codec.encode(key, value))
            stored.update(encoded)
            dense[key] = codec.decode(encoded, key)
        else:
            stored[key] = dense[key] = value
    config = fixture_config(name)
    described = ({**config, 'text_config': {**config['text_config'], 'quantization_config': FP8_TENSORS}}
                 if 'text_config' in config else {**config, 'quantization_config': FP8_TENSORS})
    save_hf_layout(stored, described, tmp_path / 'quantized')
    save_hf_layout(dense, config, tmp_path / 'dense')
    ids = jnp.asarray(np.load(FIXTURES / name / "input_ids.npy"), jnp.int32)
    loads = (fp32_decoder(tmp_path / 'quantized'), fp32_decoder(tmp_path / 'dense'))
    logits = [np.asarray(model.apply(variables, ids)) for model, variables in loads]
    np.testing.assert_array_equal(*logits)


def test_kimi_k25_text_only_placeholders_match_reference():
    import torch

    from tools.decoder_export_reference import Case, reference_model

    directory = FIXTURES / "kimi-k25-tiny"
    source = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    ids = np.load(directory / "input_ids.npy").copy()
    config = fixture_config("kimi-k25-tiny")
    ids[0, 0] = config["image_token_id"]
    ids[1, 2] = config["video_token_id"]
    reference, _ = reference_model(
        Case("kimi_k25", "kimi-k25-tiny", reference_class="Kimi_K25ForConditionalGeneration"),
        directory)
    reference.eval()
    reference.set_attn_implementation("eager")
    with torch.no_grad():
        expected = reference(input_ids=torch.from_numpy(ids.astype(np.int64)), use_cache=False).logits.numpy()
    actual = np.asarray(source.model.apply(source.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))


def test_the_glm_moe_dsa_mtp_depth_matches_the_reference_composition():
    """The checkpoint's model.layers.8.* depth loads as mtp_0 with its own
    indexer and the trunk's routing, and computes what the engines running
    the released weights compute (tools/hf_reference_b.py GlmMTP over a
    GlmMoeDsaDecoderLayer): tolerance 1e-4, observed max |logit difference|
    6.7e-06 with identical argmax."""

    mtp_depth(GLM_MOE_DSA, experts=(8, 32, 12))


def test_glm_moe_dsa_decodes_what_it_computes_in_one_pass():
    """Cached decode over the sparse layers, the sharing ones included:
    the prompt then one token reaches the one-pass logits of that token,
    with the sharing layers' cache holding no indexer keys."""

    model, variables = fp32_decoder(GLM_MOE_DSA)
    ids = jnp.asarray(np.load(GLM_MOE_DSA / "input_ids.npy"), jnp.int32)
    full = np.asarray(model.apply(variables, ids))
    cache = model.apply(variables, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
    state = {**variables, "cache": cache}
    _, updated = model.apply(state, ids[:, :-1], decode=True, mutable=["cache"])
    step, _ = model.apply({**state, "cache": updated["cache"]}, ids[:, -1:], decode=True,
                          mutable=["cache"])
    assert "cached_index" in updated["cache"]["layers_6"]["self_attn"]
    assert "cached_index" not in updated["cache"]["layers_7"]["self_attn"]
    difference = float(np.max(np.abs(np.asarray(step)[:, 0] - full[:, -1])))
    assert difference < 1e-4, f"decode differs from the one-pass logits by {difference:.3e}"


# --------------------------------------------------------------------------
# DeepSeek V4: mHC's residual streams over the sliding, compressed sparse
# and heavily compressed kinds, with hash-routed first layers
# --------------------------------------------------------------------------

DEEPSEEK_V4 = FIXTURES / "deepseek-v4-tiny"


@pytest.mark.parametrize("directory", [KIMI_K25, GLM4_MOE, GLM_MOE_DSA, DEEPSEEK_V4],
                         ids=["kimi-k25", "glm4-moe", "glm-moe-dsa", "deepseek-v4"])
def test_a_moe_decoders_logits_match_the_reference_implementation(directory):
    """fp32 parity against each release's reference, the complete vocabulary at every input position
    within 1e-4 with identical argmax: Kimi K2.5's text half on input_ids alone; GLM-4 MoE's trunk
    (observed 3.3e-06); GLM MoE DSA's over four selecting layers and four that share the previous
    full layer's top-k (observed 2.7e-05); DeepSeek V4's over all three attention kinds and both
    routers, on logits of magnitude 4.4."""
    model, variables = fp32_decoder(directory)
    difference = recorded_logit_difference(model, variables, directory)
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"


def test_deepseek_v4_draws_each_output_group_at_its_own_fan_in():
    """A from-scratch DeepSeek V4's grouped output projection (`o_a_proj`, one `[in, o_lora_rank]`
    block per head group) starts each group at lecun's variance over that group's inputs, as
    `DeepseekV4GroupedLinear`'s per-group blocks are each a projection of their own: at eight groups
    of 128 inputs, each group's 8,192 entries pass the chi-square test of variance 1 / 128 at one in a
    billion each side, where counting the groups as inputs starts them at 1 / 1024."""
    import jax
    from scipy import stats

    from dew.registry import models

    config = translate_config({**fixture_config("deepseek-v4-tiny"), "num_attention_heads": 16,
                               "head_dim": 64, "o_groups": 8, "o_lora_rank": 64})
    model = models.build("causal_transformer", config)
    params = jax.jit(model.init)(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))["params"]
    kernels = [leaf for path, leaf in flat_tree(params).items() if "o_a_proj" in path]
    assert kernels and all(kernel.shape == (8, 128, 64) for kernel in kernels)
    low, high = stats.chi2.ppf([1e-9, 1 - 1e-9], 128 * 64 - 1)
    for kernel in kernels:
        statistics = (128 * 64 - 1) * 128 * np.var(np.asarray(kernel, np.float64), axis=(1, 2), ddof=1)
        assert np.all((low < statistics) & (statistics < high)), (statistics, low, high)


def test_deepseek_v4_a_left_padded_row_pools_its_compressed_windows_over_its_own_tokens():
    """Uncached, rows padded on the left by three tokens pool the HCA windows and the CSA
    windows its lightning indexer scores over their real tokens at their ranks, as a cache appends
    them: the real tokens' logits lie as close to the reference's as the unpadded rows' do, within
    FACTOR."""
    model, variables = fp32_decoder(DEEPSEEK_V4)
    ids = jnp.asarray(np.load(DEEPSEEK_V4 / "input_ids.npy"), jnp.int32)
    reference = np.load(DEEPSEEK_V4 / "logits.npy")
    pad = 3
    valid = jnp.broadcast_to(jnp.arange(ids.shape[1] + pad) >= pad, (ids.shape[0], ids.shape[1] + pad))
    padded = model.apply(variables, jnp.pad(ids, ((0, 0), (pad, 0))), attention_mask=valid)
    alone = model.apply(variables, ids)
    assert distance(padded[:, pad:], reference) <= FACTOR * distance(alone, reference)


@pytest.mark.parametrize('saved_spelling', [False, True], ids=['released', 'hf_saved'])
def test_deepseek_v4_tied_source_names_roundtrip(tmp_path, saved_spelling):
    from dew.interop.safetensors_io import save_hf_layout
    from dew.interop.sources import load_shards

    tensors = load_shards(DEEPSEEK_V4)
    tensors['head.weight'] = tensors['embed.weight'].copy()
    if saved_spelling:
        # HF's reverse converter prefixes the trunk and misses the anchored
        # embedding/head renames. Keep the released prediction payload too.
        renamed = {}
        for name, tensor in tensors.items():
            if name == 'embed.weight':
                name = 'model.embed_tokens.weight'
            elif name == 'norm.weight':
                name = 'model.norm.weight'
            elif name.startswith('hc_head_'):
                name = 'model.hc_head.hc_' + name.removeprefix('hc_head_')
            elif name.startswith('layers.'):
                name = 'model.' + name.replace('.attn.kv_norm.', '.attn.norm.')
            renamed[name] = tensor
        tensors = renamed
    config = fixture_config('deepseek-v4-tiny')
    config['tie_word_embeddings'] = True
    directory = tmp_path / 'source'
    save_hf_layout(tensors, config, directory)
    source = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    export = tmp_path / 'export'
    source.save(export)
    emitted = load_shards(export)
    assert set(emitted) == set(tensors)
    for name, tensor in tensors.items():
        np.testing.assert_array_equal(emitted[name], tensor, err_msg=name)
    reloaded = Pretrained.load(export, dtype='float32', attention_impl='reference')
    ids = np.load(DEEPSEEK_V4 / 'input_ids.npy')
    np.testing.assert_array_equal(source.model.apply(source.variables, ids),
                                  reloaded.model.apply(reloaded.variables, ids))
    save_hf_layout({**tensors, 'head.weight': tensors['head.weight'] + 1}, config, directory)
    with pytest.raises(ValueError):
        Pretrained.load(directory, dtype='float32', attention_impl='reference')


def test_a_v4_grouped_output_projection_under_the_reference_models_names_loads_as_released(tmp_path):
    """The reference model's own state dict names the grouped output
    projection and its query `model.layers.N.self_attn.o_a_proj` and
    `q_b_proj`. Stored `[groups * rank, in]` under either spelling, the
    projection is regrouped by its query's width, and the model computes
    the released checkpoint's logits."""
    from dew.interop.safetensors_io import save_hf_layout
    from dew.interop.sources import load_shards

    spellings = {'.attn.wo_a.weight': '.self_attn.o_a_proj.weight',
                 '.attn.wq_b.weight': '.self_attn.q_b_proj.weight'}
    renamed = {}
    for name, tensor in load_shards(DEEPSEEK_V4).items():
        theirs = next((suffix for suffix in spellings if name.endswith(suffix)), None)
        if name.startswith('layers.') and theirs is not None:
            name = 'model.' + name.removesuffix(theirs) + spellings[theirs]
        renamed[name] = tensor
    assert sum(name.endswith('.self_attn.o_a_proj.weight') for name in renamed) > 0
    save_hf_layout(renamed, fixture_config('deepseek-v4-tiny'), tmp_path)
    module = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    released = Pretrained.load(DEEPSEEK_V4, dtype='float32', attention_impl='reference')
    ids = np.load(DEEPSEEK_V4 / 'input_ids.npy')
    np.testing.assert_array_equal(module.model.apply(module.variables, ids),
                                  released.model.apply(released.variables, ids))


def test_deepseek_v4_public_speculation_preserves_padded_rows():
    # Three rows differ from the two residual streams: a row mask must not
    # accidentally broadcast along the stream axis during predictor reseeding.
    source = Pretrained.load(DEEPSEEK_V4, dtype='float32', attention_impl='reference')
    assert_padded_speculation_is_greedy(source)


def test_deepseek_v4_cached_chunks_match_transformers():
    import torch

    from tools.decoder_export_reference import Case, reference_model

    source = Pretrained.load(DEEPSEEK_V4, dtype='float32', attention_impl='reference')
    ids = np.load(DEEPSEEK_V4 / 'input_ids.npy')
    cache = source.model.apply(source.variables, ids.shape[0], method=source.model.init_cache,
                               mutable=['cache'])[1]['cache']
    reference, _ = reference_model(Case('deepseek_v4', 'deepseek-v4-tiny'), DEEPSEEK_V4)
    reference.eval()
    reference.set_attn_implementation('eager')
    past, begin = None, 0
    # End calls both inside and on compressor boundaries; the third call
    # closes several windows and must retain the preceding CSA Ca series.
    for size in (3, 1, 5, 3):
        tokens = ids[:, begin:begin + size]
        actual, updated = source.model.apply(
            {**source.variables, 'cache': cache}, jnp.asarray(tokens),
            decode=True, mutable=['cache'])
        cache = updated['cache']
        with torch.no_grad():
            output = reference(input_ids=torch.from_numpy(tokens.astype(np.int64)),
                               past_key_values=past, use_cache=True)
        past = output.past_key_values
        np.testing.assert_allclose(actual, output.logits.numpy(), atol=1e-4, rtol=0)
        begin += size


def test_deepseek_v4_public_prediction_states_preserve_raw_streams():
    source = Pretrained.load(DEEPSEEK_V4, dtype='float32', attention_impl='reference')
    ids = np.load(DEEPSEEK_V4 / 'input_ids.npy')
    model, variables = source.model, source.variables
    states, logits = model.apply(variables, ids, method=model.states_and_logits)
    states = jnp.asarray(states)
    assert states.shape == (*ids.shape, 2, model.emb_features)
    normalized, expected = model.apply(variables, ids, method=model.hidden_and_mtp_inputs)
    np.testing.assert_array_equal(states, expected)
    assert jnp.asarray(normalized).shape == (*ids.shape, model.emb_features)
    doubled = {**variables, 'params': {**variables['params'], 'norm': {
        **variables['params']['norm'], 'scale': variables['params']['norm']['scale'] * 2}}}
    raw, louder = model.apply(doubled, ids, method=model.states_and_logits)
    np.testing.assert_array_equal(raw, states)
    np.testing.assert_allclose(jnp.asarray(louder), 2 * jnp.asarray(logits), atol=1e-4, rtol=0)


def test_deepseek_v4_mtp_matches_released_raw_stream_composition():
    source = Pretrained.load(DEEPSEEK_V4, dtype='float32', attention_impl='reference')
    ids = np.load(DEEPSEEK_V4 / 'input_ids.npy')
    hidden, streams = source.model.apply(source.variables, ids, method=source.model.hidden_and_mtp_inputs)
    streams = jnp.asarray(streams)
    actual = source.model.apply(source.variables, streams, ids, method=source.model.mtp_logits)[0]
    expected = np.load(DEEPSEEK_V4 / 'mtp_logits.npy')
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    cache = source.model.apply(source.variables, ids.shape[0], method=source.model.init_mtp_cache,
                               mutable=['cache'])[1]['cache']
    start = 0
    for size in (3, 1, 5, 2):
        (logits, prediction), updated = source.model.apply(
            {**source.variables, 'cache': cache}, streams[:, start:start + size],
            ids[:, start + 1:start + size + 1], method=source.model.mtp_step,
            decode=True, mutable=['cache'])
        np.testing.assert_allclose(logits, expected[:, start:start + size], atol=1e-4, rtol=0)
        assert prediction.shape == streams[:, start:start + size].shape
        cache = updated['cache']
        start += size
    with pytest.raises(ValueError):
        source.model.apply(source.variables, hidden, ids, method=source.model.mtp_logits)


# --------------------------------------------------------------------------
# Llama 4 (text): iRoPE with chunked local layers, input-scaled experts
# --------------------------------------------------------------------------

LLAMA4 = FIXTURES / "llama4-tiny"


def test_llama4_logits_match_the_reference_implementation():
    """fp32 parity: tolerance 1e-4, observed max |logit difference| 3.5e-06
    with identical argmax. The fused expert kernels arrive split in place."""
    model, variables = fp32_decoder(LLAMA4)
    difference = recorded_logit_difference(model, variables, LLAMA4)
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    experts = variables["params"]["layers_1"]["mlp"]["experts"]
    assert experts["gate_proj"]["kernel"].shape == (4, 32, 48)
    assert experts["down_proj"]["kernel"].shape == (4, 48, 32)
    assert "shared_experts" in variables["params"]["layers_1"]["mlp"]
    assert "experts" not in variables["params"]["layers_0"]["mlp"]


# ---------------------------------------------------------------------------
# Gemma 4's routed size: the parallel experts, keys as values, layer scalars
# ---------------------------------------------------------------------------



def test_gemma4_moe_logits_match_the_reference_implementation():
    """fp32 parity: tolerance 1e-4, observed max |logit difference| 4.9e-06
    with identical argmax. The fused expert kernels arrive split in place,
    the global layer holds no v_proj, and every layer holds its scalar."""
    model, variables = fp32_decoder(GEMMA4_MOE)
    difference = recorded_logit_difference(model, variables, GEMMA4_MOE)
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    params = variables["params"]
    experts = params["layers_0"]["moe"]["experts"]
    assert experts["gate_proj"]["kernel"].shape == (4, 32, 16)
    assert experts["down_proj"]["kernel"].shape == (4, 16, 32)
    assert params["layers_0"]["moe"]["router"]["per_expert_scale"].shape == (4,)
    assert "v_proj" not in params["layers_2"]["self_attn"]
    assert params["layers_2"]["self_attn"]["k_proj"]["kernel"].shape == (32, 16)
