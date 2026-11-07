"""Hugging Face decoders with routed experts or latent attention: GPT OSS,
DeepSeek V2/V4, Kimi K2 and K2.5, GLM 4.5 and GLM-5, Llama 4 and Gemma 4's
routed size, each against transformers' logits (see test_hf_decoders).
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from interop_support import (
    clearing_selections,
    fixture_ids,
    token_draw,
    transformers_arms,
    transformers_model,
)
from model_support import decode_steps, flat_tree, prediction_logits, stepped_outputs
from one_draw import assert_as_exact_as_the_reference
from reference_error import (
    FACTOR,
    as_given,
    assert_as_exact,
    declared_seeds,
    distance,
    indexed_recording,
    recording,
    widened,
    without_ynn,
)
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
"""Routed decoders with padded fixtures: Qwen2-MoE's fused experts and shared gate, Granite MoE's packed
experts, MiniMax-M2's router state, recorded from the corrected reference."""

RECORDED = ('minimax-m2-tiny', 'minimax-m2-fp8-tiny')
"""The routed decoders whose reference is transformers 5.18.0, past the suite's release (MiniMax-M2's
rotary fix, tools/reference_releases.py), so the test cannot run it: each one's logits recorded over
check 1's seeds in its recorded_logits.npz (tools/hf_reference.py `--minimax-m2-recording`), whose
declared count it reads. transformers 5.17.0 reads every other one as its recorder did, in the test."""

ROUTED_SEEDS = 8
"""Each in-test routed decoder's declared draws of ids against transformers' logits (check 1), whole,
prefilled and stepped through the cache, and left-padded, sized by `seeds_for` paired with Dew's logits
on the CPU."""


def assert_routed(name: str, ours, label: str, leaf: str = 'logits', *, wide=None, prefill: int | None = None,
                  **fields) -> None:
    """Check 1 of `ours(model, variables)`, Dew's fp32 function of a draw's ids (`fixture_ids`), against
    transformers' logits over them, the padded rows' `fields` beside them (their `attention_mask`, which
    pads the ids with 0, and `position_ids`) at the mask's valid places: run in the test from its float64
    twin, or for a `RECORDED` decoder its recording's `leaf`, the truth Dew's `wide` (by default `ours`)
    run in float64 and held to the recording's sketch (`indexed_recording`). With `prefill`, `ours`
    decodes through its cache and transformers' logits are read at the shapes its steps compute
    (`stepped_outputs`: the prefill's from one forward over it, each later id's from one over the ids up
    to it; ledger 1543), the float64 truth one forward over the whole row."""
    import torch

    directory = FIXTURES / name
    model, variables = fp32_decoder(directory)
    draw = fixture_ids(directory)
    if name in RECORDED:
        path = directory / 'recorded_logits.npz'
        seeds = declared_seeds(path)['logits']
        # A cached decode's reference is the recording's at the shapes its steps compute.
        leaf = leaf if prefill is None else 'stepped_logits'
        fp32 = ours(model, variables)
        wide = (wide or ours)(model.clone(dtype=jnp.float64), widened(variables))
        recorded, reference, truth = indexed_recording(
            recording(path), 'logits', seeds, leaves=(leaf,), truth_fn=lambda inputs: {leaf: wide(
                inputs['input_ids'])}, inputs_of=lambda seed: {'input_ids': draw(seed)[0]})
        assert_as_exact(lambda index, inputs: {leaf: fp32(inputs['input_ids'])}, reference, recorded, label,
                        seeds=seeds, truth_fn=truth)
        return
    tensors = {key: torch.from_numpy(value) for key, value in fields.items()}
    valid = tensors.get('attention_mask')

    def read(model, ids):
        if valid is None:
            return model(ids, use_cache=False).logits
        return model(torch.where(valid, ids, 0), **tensors, use_cache=False).logits[valid]

    def stepped(model, ids):
        return torch.from_numpy(np.asarray(stepped_outputs(
            lambda length: read(model, ids[:, :length]).numpy(), ids.shape[1], prefill)))

    reference, _ = transformers_arms(transformers_model(directory), read if prefill is None else stepped)
    _, truth = transformers_arms(transformers_model(directory), read)
    assert_as_exact(ours(model, variables), reference, draw, label, seeds=ROUTED_SEEDS, truth_fn=truth)


def whole(model, variables):
    """`model`'s logits over a draw's ids, for `assert_routed`."""
    return lambda ids: model.apply(variables, ids)


@pytest.mark.parametrize('name', (*ROUTED, 'minimax-m2-fp8-tiny'))
def test_a_routed_decoders_logits_match_the_reference(name):
    """Dew's logits are on par with transformers', from float64, over drawn ids (check 1), with the
    recorded reference's argmax on the fixture's ids."""
    assert_routed(name, whole, f'{name} logits')
    directory = FIXTURES / name
    model, variables = fp32_decoder(directory)
    logits = np.asarray(model.apply(variables, np.load(directory / 'input_ids.npy')))
    np.testing.assert_array_equal(logits.argmax(-1), np.load(directory / 'logits.npy').argmax(-1))


@pytest.mark.parametrize('name', ROUTED)
def test_a_routed_decoders_prefill_and_token_steps_match_the_reference(name):
    """Four ids prefilled into the cache and the rest stepped one at a time give logits on par with
    transformers' uncached forward, from float64, over drawn ids (check 1)."""
    def stepped(model, variables):
        def run(ids):
            state = model.apply(variables, ids.shape[0], method='init_cache', mutable=['cache'])[1]
            out, state = model.apply({**variables, **state}, ids[:, :4], decode=True, mutable=['cache'])
            pieces = [out]
            for index in range(4, ids.shape[1]):
                out, state = model.apply({**variables, **state}, ids[:, index:index + 1],
                                         decode=True, mutable=['cache'])
                pieces.append(out)
            return jnp.concatenate(pieces, axis=1)
        return run

    # The truth is the whole forward's in float64: a recording holds no cache walk of its own.
    assert_routed(name, stepped, f'{ROUTED[name]} cache', wide=whole, prefill=4)


@pytest.mark.parametrize('name', ROUTED)
def test_a_routed_decoders_padded_routing_matches_the_reference(name):
    """Left-padded rows under their mask and positions route as transformers' do: their logits are on par
    with its, from float64, over drawn ids (check 1)."""
    with np.load(FIXTURES / name / 'padded.npz') as arrays:
        valid, positions = arrays['attention_mask'], arrays['position_ids']

    def padded(model, variables):
        def run(ids):
            rows = jnp.where(valid, ids, 0)
            return model.apply(variables, rows, positions=positions, attention_mask=valid)[valid]
        return run

    assert_routed(name, padded, f'{ROUTED[name]} padded', 'padded_logits', attention_mask=valid,
                  position_ids=positions)


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


@pytest.mark.network
def test_gpt_oss_20b_matches_transformers_on_the_real_weights():
    """The released MXFP4 checkpoint through the loader, its weights in the bf16 the blocks decode to
    (`param_dtype="auto"`, 42 GB on the device, where FP32 master weights would take 84), on the prompt
    the recording read: Dew's bf16 logits as near the float64 truth as transformers' own bf16 forward
    on a GPU, its experts dequantized as Dew's are (the one-draw rule, ledger 1719). Both are recorded
    by tools/gpt_oss_20b_reference.py: the truth holds the released weights in float64, 168 GB, which
    no test process holds."""
    from tools.fetched_fixtures import local
    from tools.gpt_oss_20b_reference import FIXTURE, RELEASED

    with np.load(local(FIXTURE)) as recorded:
        ids, reference, truth = (recorded[name] for name in ("input_ids", "reference", "truth"))
    repo, revision = RELEASED
    pretrained = Pretrained.load(repo, dtype="bfloat16", param_dtype="auto", attention_impl="xla",
                                 max_seq_len=int(ids.shape[1]), revision=revision)
    logits = np.asarray(pretrained.model.apply(pretrained.variables, jnp.asarray(ids)), np.float32)[0]
    assert_as_exact_as_the_reference(logits, reference, truth, "GPT OSS 20B logits")


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
        return np.asarray(prediction_logits(model, held, ids))

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


def test_the_glm_moe_dsa_references_selections_are_its_function_not_a_tie_rules():
    """transformers' GLM-5 indexers select every query's keys, on the fixture's ids and on prompts drawn
    from its tokens, with no tie at their top-k cut (`dsa_support.tied_cuts`), so a check of Dew against
    them compares the two implementations' selections rather than two tie rules: Dew chooses the lower
    key index among equal scores, torch's topk no order. Its eight indexer heads put an exact 0 score at
    2^-8 a key; at four, 26 to 30 queries a prompt were cut among ties (row 4703)."""
    import torch
    from dsa_support import indexer_rankings, tied_cuts
    from transformers import AutoModelForCausalLM
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer

    reference = AutoModelForCausalLM.from_pretrained(GLM_MOE_DSA, dtype=torch.float32,
                                                     local_files_only=True).eval()
    ids = np.load(GLM_MOE_DSA / "input_ids.npy")
    prompts = [ids, *(np.random.default_rng([241, seed]).choice(np.unique(ids), size=ids.shape)
                      for seed in range(16))]
    with indexer_rankings(reference, GlmMoeDsaIndexer) as taken, torch.no_grad():
        for prompt in prompts:
            reference(input_ids=torch.from_numpy(prompt.astype(np.int64)))
    assert taken
    assert tied_cuts(taken) == 0


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


KIMI_K25_MEDIA = {"image": ((1, 6, 4), (3,)), "video": ((2, 4, 4), (2,))}
"""Each Kimi K2.5 media site's item grid `(t, h, w)` in 2-pixel patches and where its placeholder run
starts: an image of 6 merge blocks, and a two-frame video whose frames pool into 4 (`sd2_tpool`)."""

KIMI_K25_MEDIA_SEEDS = 8
"""The media sites' declared check-1 seeds, sized by `seeds_for` on armada paired with Dew's arm: 8 for
each site (22 s the image's, 14 s the video's)."""


def kimi_k25_media_case(site):
    """Check 1's arms for Kimi K2.5 whole over a row a drawn image, or a drawn video's frames, fill (row
    855): Dew's bundle (`Pretrained.load`: MoonViT, its patch merger, the decoder), its processor reading
    the fields transformers' processor gives (`Processor.from_hf`: the placeholder run as long as the
    item's merged blocks, its patches `[N, 3, 2, 2]` with their grid), and transformers' own
    Kimi_K25ForConditionalGeneration over the same checkpoint, in float32 and as its float64 twin
    (`transformers_arms`); each draw's tokens redrawn until the decoder's routing clears its margin."""
    import selection_margins
    import torch
    from transformers import Kimi_K25ForConditionalGeneration

    from dew.nn.inputs import ModelInputs

    grid, starts = KIMI_K25_MEDIA[site]
    config = fixture_config("kimi-k25-tiny")
    token = config["image_token_id" if site == "image" else "video_token_id"]
    count = grid[1] * grid[2] // 4
    ids_shape = np.load(KIMI_K25 / "input_ids.npy")[:1].shape
    pixels, grids = ("pixel_values", "image_grid_thw") if site == "image" else (
        "pixel_values_videos", "video_grid_thw")

    def draw(seed):
        rng = np.random.default_rng([0x855, len(site), seed])
        ids = rng.integers(0, min(config["image_token_id"], config["video_token_id"]) - 8, ids_shape)
        for start in starts:
            ids[0, start:start + count] = token
        patches = rng.normal(size=(int(np.prod(grid)), 3, 2, 2)).astype(np.float32)
        return ids.astype(np.int32), patches

    source = Pretrained.load(KIMI_K25, dtype="float32", attention_impl="reference")
    model, variables = source.model, source.variables

    def batch(ids, patches):
        fields = source.processor.from_hf({"input_ids": ids, pixels: patches, grids: np.asarray([grid])})
        inputs = ModelInputs(jnp.asarray(ids), fields.token_fields, fields.conditioning)
        return inputs.tokens, inputs.kwargs()

    @as_given
    def dew(ids, patches):
        tokens, fields = batch(ids, patches)
        return np.asarray(model.apply(variables, tokens, **fields))

    network = Kimi_K25ForConditionalGeneration.from_pretrained(
        str(KIMI_K25), dtype=torch.float32, local_files_only=True, experts_implementation="eager")
    network.set_attn_implementation("eager")

    def logits(network, ids, patches):
        dtype = next(network.parameters()).dtype
        if site == "image":
            return network(input_ids=ids, **{pixels: patches.to(dtype), grids: torch.tensor([grid])}).logits
        # transformers 5.17 splits a video's projected tokens by t·h·w/4 where its tower pools the frames (row
        # 4368) to h·w/4 (`get_image_features`), so its forward refuses a clip of more than one frame: its own
        # tower, projector and decoder, placed as its forward places them (`Kimi_K25Model.forward`: the
        # placeholders read token 0, then the projected tokens scattered over them).
        inner, placed = network.model, ids == token
        embeds = inner.get_input_embeddings()(ids.masked_fill(placed, 0))
        tower = inner.vision_tower(patches.to(dtype), grid_thw=torch.tensor([grid])).pooler_output
        projected = inner.mm_projector(tower).squeeze(1).to(embeds.dtype)
        embeds = embeds.masked_scatter(placed[..., None], projected)
        return network(inputs_embeds=embeds).logits

    reference, truth = transformers_arms(network.eval(), logits)
    # The decoder's routing over the row with its media, as the compared runs route it.
    captured = selection_margins.capture(model, lambda module, tokens, fields: module(tokens, **fields))
    decoder = translate_config(config)

    def clears(drawn):
        selections = captured(variables, *batch(*drawn))
        return (selection_margins.margin(decoder, selections, f"kimi-k25 {site}", stable=False)
                > selection_margins.MARGIN)
    return dew, reference, selection_margins.clearing(draw, clears), truth


@pytest.mark.parametrize("site", sorted(KIMI_K25_MEDIA))
def test_kimi_k25_reads_its_media_whole_on_par_with_transformers(site):
    """Kimi K2.5 loads whole (row 855): its MoonViT, attending over an item's every frame, its frames pooled,
    its patch merger and the decoder compute transformers' logits over a row a drawn image or a drawn
    video fills, on par with transformers' own model from its float64 twin (check 1)."""
    dew, reference, draw, truth = kimi_k25_media_case(site)
    assert_as_exact(dew, reference, draw, f"kimi-k25 {site}", seeds=KIMI_K25_MEDIA_SEEDS, truth_fn=truth)


@pytest.mark.parametrize("kind", ["image", "video"])
def test_kimi_k25_reads_raw_media_through_its_processor_as_its_fields(kind):
    """Kimi K2.5's raw media reach the model through the checkpoint's own processor (transformers'
    Kimi_K25Processor, its tiny patches and merge): Dew's call (`Processor.__call__`) gives the inputs its
    processed fields give (`Processor.from_hf`), whose logits the media sites hold by check 1, and the
    model's logits over them are those inputs' bit for bit (row 855's review)."""
    from transformers import AutoProcessor

    from dew.nn.inputs import ModelInputs

    source = Pretrained.load(KIMI_K25, dtype="float32", attention_impl="reference")
    rng = np.random.default_rng(855)
    picture = (rng.random((12, 8, 3)) * 255).astype(np.uint8)
    # A clip of two frames at two a second, which its processor samples by its own fps.
    media = {"images": [picture]} if kind == "image" else {
        "videos": [np.stack([picture, picture[::-1]])],
        "video_metadata": [{"total_num_frames": 2, "fps": 2.0, "duration": 1.0}]}
    text = "w5 " + ("<|media_pad|>" if kind == "image" else "<|kimi_k25_video_placeholder|>") + " w7"
    called = source.processor(text, **media)
    fields = AutoProcessor.from_pretrained(str(KIMI_K25), local_files_only=True)(
        text=text, return_tensors="np", **media)
    given = source.processor.from_hf(dict(fields))

    def logits(inputs: ModelInputs):
        return np.asarray(source.model.apply(source.variables, inputs.tokens, **inputs.kwargs()))

    np.testing.assert_array_equal(np.asarray(called.tokens), np.asarray(given.tokens))
    np.testing.assert_array_equal(logits(called), logits(given))


def test_kimi_k25_reads_raw_images_through_its_releases_own_processor(tmp_path):
    """Kimi K2.5's release ships its own processor (KimiK25Processor and its vision processor at 4d01dfe,
    the source's code), whose public call takes `medias` and keeps one placeholder an image (row 4367): a
    load that trusts the source's code reads raw images through it (`Processor.media`), over a row of one
    image and a row of two of other sizes, as the release lays them out: each row's tokens the release's,
    each placeholder expanded to as many as the release's own count of its image's tokens
    (`media_tokens_calculator`), and the images' patches and grids its own, in order. A chat over an image
    part, under the release's own template, reads it as the call over the text the template writes does.
    The model reads them: its logits are finite and another image changes them."""
    import json
    import shutil

    from PIL import Image
    from tool_support import run_from
    from transformers import AutoProcessor

    from dew.nn.inputs import pad_token_rows

    source = tmp_path / "source"
    shutil.copytree(KIMI_K25, source)
    # The release's own processor in place of transformers' (its processor_config.json), at the fixture's
    # tiny patches, merge and patch budget.
    tiny = json.loads((source / "processor_config.json").read_text())["image_processor"]
    (source / "processor_config.json").unlink()
    with run_from("remote_code_reference", "tools") as reference:
        files = reference.PROCESSORS["kimi-k25-tiny"][2]
        for file in files:
            shutil.copy(reference.release_file("kimi-k25-tiny", file), source / file)
    released = json.loads((source / "preprocessor_config.json").read_text())
    side = tiny["size"]["max_height"] // tiny["patch_size"]
    released["media_proc_cfg"].update(patch_size=tiny["patch_size"], merge_kernel_size=tiny["merge_size"],
                                      in_patch_limit=tiny["max_patches"], in_patch_limit_each_frame=None,
                                      patch_limit_on_one_side=side)
    (source / "preprocessor_config.json").write_text(json.dumps(released))
    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference", trust_remote_code=True)
    release = AutoProcessor.from_pretrained(str(source), trust_remote_code=True, local_files_only=True)
    assert type(release).__name__ == "KimiK25Processor"
    rng = np.random.default_rng(4367)
    images = [(rng.random(shape) * 255).astype(np.uint8) for shape in ((12, 8, 3), (8, 16, 3), (10, 10, 3))]
    rows = ["w5 <|media_pad|> w7", "<|media_pad|> w2 w3 <|media_pad|>"]
    called = loaded.processor(rows, images=images)
    image_id = fixture_config("kimi-k25-tiny")["image_token_id"]
    expanded, pixels, grids = [], [], []
    # The release's call reads PIL images (its `_to_pil`), as Dew hands it the arrays.
    pictures = [Image.fromarray(image) for image in images]
    for row, taken in zip(rows, ([pictures[0]], pictures[1:]), strict=True):
        theirs = release(medias=[{"type": "image", "image": image} for image in taken], text=row,
                         return_tensors="np")
        ids = np.asarray(theirs["input_ids"]).reshape(-1)
        repeats = np.ones(ids.shape, np.int64)
        repeats[ids == image_id] = [
            release.media_processor.media_tokens_calculator({"type": "image", "image": image})
            for image in taken]
        expanded.append(np.repeat(ids, repeats))
        pixels.append(np.asarray(theirs["pixel_values"]))
        grids.append(np.asarray(theirs["grid_thws"]))
    tokenizer = release.tokenizer
    padded, fields = pad_token_rows(expanded, pad_id=tokenizer.pad_token_id,
                                    padding_side=tokenizer.padding_side)
    given = loaded.processor.from_hf({"input_ids": padded, **fields, "pixel_values": np.concatenate(pixels),
                                      "image_grid_thw": np.concatenate(grids)})
    np.testing.assert_array_equal(np.asarray(called.tokens), np.asarray(given.tokens))
    for kind, ours, theirs in (("token", called.token_fields, given.token_fields),
                               ("conditioning", called.conditioning, given.conditioning)):
        assert sorted(ours) == sorted(theirs), kind
        for name in theirs:
            np.testing.assert_array_equal(np.asarray(ours[name]), np.asarray(theirs[name]), err_msg=name)

    # Dew's spelling of an image part and the release's two (its template's and its call's `image_url`).
    for part in ({"type": "image", "image": images[1]}, {"type": "image_url", "image_url": pictures[1]},
                 {"type": "image", "image_url": images[1]}):
        messages = [{"role": "user", "content": [part, {"type": "text", "text": "w5"}]}]
        chatted = loaded.processor.chat(messages)
        written = loaded.processor.reference.apply_chat_template(messages, tokenize=False,
                                                                 add_generation_prompt=True)
        direct = loaded.processor(written, images=[images[1]])
        np.testing.assert_array_equal(np.asarray(chatted.tokens), np.asarray(direct.tokens))
        np.testing.assert_array_equal(np.asarray(chatted.conditioning["pixel_values"]),
                                      np.asarray(direct.conditioning["pixel_values"]))
        assert np.count_nonzero(np.asarray(chatted.tokens) == image_id) == \
            release.media_processor.media_tokens_calculator({"type": "image", "image": pictures[1]})

    def logits(images):
        inputs = loaded.processor(rows, images=images)
        return np.asarray(loaded.model.apply(loaded.variables, inputs.tokens, **inputs.kwargs()))

    first = logits(images)
    assert np.isfinite(first).all()
    assert not np.array_equal(first, logits([images[0][::-1], *images[1:]]))


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
    init = jax.jit(lambda *args: nn.unbox(model.init(*args)))
    params = init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))["params"]
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


V4_CHUNKS = (3, 1, 5, 3)
"""The cached calls' widths: they end both inside and on compressor boundaries, and the third closes
several windows and must retain the preceding CSA Ca series."""
V4_CACHED_SEEDS = 16
"""DeepSeek V4's cached chunks' declared check-1 seeds against transformers' cached chunks, sized by
`seeds_for` on armada."""
V4_TRIES = 256
"""Draws a seed takes to clear V4's selections against transformers' (`clearing_selections`): its
indexer scores a key every head disagrees with zero, a tie torch's `topk` breaks in no stated order,
which 1 in 18 of the fixture's two-row draws keeps clear of (22 of 400 on armada), so a seed takes
under 2e-6 chance of none in this many."""


def test_deepseek_v4_cached_chunks_match_transformers():
    """Calls of `V4_CHUNKS` tokens through the decode cache on par with transformers' calls of the same
    widths through its own cache, from its float64 twin's (check 1, ledger 2739), over drawn rows whose
    indexer top-k and routing clear `selection_margins.MARGIN` with no tie at a budget
    (`clearing_selections`, `stable=False`): a row whose indexer budget ends in keys scored zero is one
    torch's own cached and uncached calls disagree on by up to 0.3, which its `topk` breaks in no
    stated order."""
    import torch

    from tools.decoder_export_reference import Case, reference_model

    source = Pretrained.load(DEEPSEEK_V4, dtype='float32', attention_impl='reference')
    model, variables = source.model, source.variables
    rows = np.load(DEEPSEEK_V4 / 'input_ids.npy').shape[0]
    reference, _ = reference_model(Case('deepseek_v4', 'deepseek-v4-tiny'), DEEPSEEK_V4)
    reference.eval()
    reference.set_attn_implementation('eager')

    made, step = decode_steps(model, without_ynn)

    @as_given
    def cached(ids):
        cache, begin, logits = made(variables, rows), 0, []
        for size in V4_CHUNKS:
            out, cache = step(variables, cache, ids[:, begin:begin + size])
            logits.append(out)
            begin += size
        return jnp.concatenate(logits, axis=1)

    def chunked(network, ids):
        past, begin, logits = None, 0, []
        for size in V4_CHUNKS:
            output = network(input_ids=ids[:, begin:begin + size], past_key_values=past, use_cache=True)
            past, begin = output.past_key_values, begin + size
            logits.append(output.logits)
        return torch.cat(logits, 1)

    run, truth = transformers_arms(reference, chunked)
    draw = clearing_selections(source, token_draw(0x2739, model.vocab_size, (rows, sum(V4_CHUNKS))),
                               "deepseek_v4 cached chunks", stable=False, tries=V4_TRIES)
    assert_as_exact(cached, run, draw, "deepseek_v4 cached chunks", seeds=V4_CACHED_SEEDS, truth_fn=truth)


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
