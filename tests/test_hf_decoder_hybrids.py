"""Hugging Face decoders whose layers are not all attention: Gemma 4's per-layer
inputs and shared KV, Qwen3.5's and Qwen3-Next's gated delta nets, and Gemma
3n's AltUp copies, each against transformers' logits (see test_hf_decoders).
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from interop_support import assert_decodes_as_it_scores
from model_support import flat_tree, prediction_logits
from test_hf_decoders import fixture_config, fp32_decoder, recorded_logit_difference, scaled_difference

from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config, translate_weights

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
DECODE_SEEDS = {"qwen35": 8, "qwen3_next": 8, "gemma3n": 32}
"""Each checkpoint's declared check-1 seeds for its cached decode against its parallel forward, sized by
`seeds_for` on armada."""

# --------------------------------------------------------------------------
# Gemma 4 gaps: per-layer input embeddings and cross-layer KV sharing
# --------------------------------------------------------------------------

GEMMA4 = ("gemma4-ple", "gemma4-kvshare")


@pytest.mark.parametrize("name", (*GEMMA4, "gemma4-e2b"))
def test_gemma4_logits_match_the_reference_implementation(name):
    """Full-model parity, fully live on both branches. Largest observed max
    |logit difference| on CPU: gemma4-ple 4.9e-07, gemma4-kvshare 8.6e-07,
    gemma4-e2b 1.4e-06."""
    directory = FIXTURES / name
    model, variables = fp32_decoder(directory)
    difference = recorded_logit_difference(model, variables, directory)
    assert difference < 1e-5, f"max |logit difference| {difference:.3e}"


def test_a_kv_shared_layers_stored_keys_and_values_load_unused_and_save_back(tmp_path):
    """google/gemma-4-E4B-it stores its kv-shared layers' own k_proj, v_proj and k_norm (layers 24-41),
    which transformers builds and never computes with, as the layer reads its provider's keys and
    values: a load leaves them unplaced, building the variables a release without them builds, and a
    save writes them back as stored (ledger 2010)."""
    import shutil

    from safetensors import numpy as safetensors_numpy

    from dew.interop import Pretrained
    from dew.interop.safetensors_io import read_weights

    directory = tmp_path / "gemma4-kvshare"
    shutil.copytree(FIXTURES / "gemma4-kvshare", directory)
    stored = safetensors_numpy.load_file(directory / "model.safetensors")
    config = json.loads((directory / "config.json").read_text())
    layers, shared = config["num_hidden_layers"], config["num_kv_shared_layers"]
    rng = np.random.default_rng(0)
    unused = {}
    for layer in range(layers - shared, layers):
        for part in ("k_proj", "v_proj", "k_norm"):
            # The layer of the same type before the shared run stores the shapes this one's would have.
            like = stored[f"model.layers.{layer - shared}.self_attn.{part}.weight"]
            unused[f"model.layers.{layer}.self_attn.{part}.weight"] = rng.standard_normal(like.shape).astype(
                like.dtype)
    safetensors_numpy.save_file({**stored, **unused}, directory / "model.safetensors",
                                metadata={"format": "pt"})
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    alone = Pretrained.load(FIXTURES / "gemma4-kvshare", dtype="float32", attention_impl="reference")
    assert jax.tree.structure(loaded.variables) == jax.tree.structure(alone.variables)
    for ours, theirs in zip(jax.tree.leaves(loaded.variables), jax.tree.leaves(alone.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    loaded.save(tmp_path / "saved")
    saved = read_weights(tmp_path / "saved")
    for name, tensor in unused.items():
        np.testing.assert_array_equal(saved[name], tensor)


@pytest.mark.network
def test_gemma4_e4b_builds_from_its_released_tensors():
    """google/gemma-4-E4B-it as released (@ee0ef60), read from its safetensors header alone
    (`weights=False`), its kv-shared layers' unused keys, values and key norms among its tensors,
    builds its model: the release's own tensor list, which no fixture holds (ledger 2010)."""
    from dew.interop import Pretrained

    loaded = Pretrained.load("google/gemma-4-E4B-it", revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
                             weights=False)
    shapes = jax.tree.leaves(loaded.variables)
    assert shapes and all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in shapes)


def test_the_router_bias_lands_in_the_moe_collection():
    """DeepSeek's `e_score_correction_bias` is router state a training step
    moves, not a weight, and `Router` keeps it in the `moe` collection. The
    checkpoint names it `model.layers.N.mlp.gate.e_score_correction_bias`,
    six dot-separated parts, one fewer than the per-expert tensors the map
    also reads under `mlp`; a map that counts it as seven falls through to
    the params map and refuses the name. Its leaf is the checkpoint's tensor
    and the loaded model selects on it: zeroing the bias moves the logits by
    2.1 on deepseek-v3-tiny.
    """
    from dew.interop.decoder_parts import dew_path
    from dew.interop.sources import load_shards

    directory = FIXTURES / "deepseek-v3-tiny"
    config = translate_config(fixture_config("deepseek-v3-tiny"))
    tensors = load_shards(directory)
    name = 'model.layers.1.mlp.gate.e_score_correction_bias'
    assert dew_path(name, config) == (
        'moe', 'layers_1', 'mlp', 'gate', 'e_score_correction_bias')

    variables = translate_weights(tensors, config)
    assert set(flat_tree(variables['moe'])) == {
        'layers_1.mlp.gate.e_score_correction_bias'}
    assert np.array_equal(
        variables['moe']['layers_1']['mlp']['gate']['e_score_correction_bias'],
        tensors[name])
    assert not [path for path in flat_tree(variables['params'])
                if path.endswith('e_score_correction_bias')]

    model, loaded = fp32_decoder(directory)
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)
    zeroed = {**loaded, 'moe': jax.tree.map(jnp.zeros_like, loaded['moe'])}
    moved = float(np.max(np.abs(np.asarray(model.apply(loaded, ids))
                                - np.asarray(model.apply(zeroed, ids)))))
    assert moved > 1.0, moved


# --------------------------------------------------------------------------
# Qwen3.5: gated delta net layers, a gated attention, a sliced partial rotary
# --------------------------------------------------------------------------

QWEN35_REAL = FIXTURES / "qwen35-0.8b"


def qwen35_real_config():
    """The released Qwen/Qwen3.5-0.8B config's text decoder; the repo's own
    config.json is the multimodal wrapper around it."""
    return json.loads((QWEN35_REAL / "config.json").read_text())["text_config"]


def test_the_real_qwen35_rotary_rotates_the_dims_the_reference_rotates():
    """The partial rotary convention, pinned to the reference's numbers: on
    the real config's head_dim 256 and factor 0.25 the reference builds a
    64-dim rope with exponents over 64 (Qwen3_5TextRotaryEmbedding.
    compute_default_rope_parameters, modeling_qwen3_5.py:117-124), 32
    inverse frequencies with no zero tail. Gemma 4's proportional reading
    of the same factor puts the exponents over 256, whose frequencies
    differ from these by up to 4.7e-01, so a translation that guessed the
    convention would rotate every position by different angles. Largest
    observed cosine difference at 5 positions 6.3e-08."""
    from dew.nn.rope import rotary_freqs

    config = translate_config(qwen35_real_config())
    inv_freq = np.load(QWEN35_REAL / "inv_freq.npy")
    assert inv_freq.shape == (32,)
    rot_dim = int(config["head_dim"] * config["partial_rotary_factor"])
    assert rot_dim == 64

    positions = np.arange(5)
    cos, _ = rotary_freqs(jnp.asarray(positions), config["head_dim"], config["rope_theta"],
                          rot_dim=rot_dim, partial_rotary_type=config["partial_rotary_type"],
                          dtype=np.float32)
    assert cos.shape == (5, 32)
    assert float(np.max(np.abs(np.asarray(cos) - np.cos(positions[:, None] * inv_freq[None])))) < 1e-6

    proportional, _ = rotary_freqs(jnp.asarray(positions), config["head_dim"], config["rope_theta"],
                                   rot_dim=rot_dim, partial_rotary_type="proportional", dtype=np.float32)
    assert proportional.shape == (5, 128)
    assert not np.allclose(np.asarray(proportional)[:, :32],
                           np.cos(positions[:, None] * inv_freq[None]), atol=1e-2)


def test_qwen35_logits_match_the_reference_implementation():
    """Full-model parity on the tiny hybrid: the delta net layers, the gated
    attention with its sliced quarter-head rope (the reference applies its
    interleaved mRoPE to text-only positions, which is this rope exactly),
    the (1 + w) norms and the tied head. Largest observed max |logit
    difference| 9.1e-05 on logits of magnitude 6.8, tolerance 5e-4, every
    argmax equal. Translating the rope as Gemma 4's proportional convention
    instead moves the logits by 7.8e-01."""
    directory = FIXTURES / "qwen35-tiny"
    model, variables = fp32_decoder(directory)
    difference = recorded_logit_difference(model, variables, directory)
    assert difference < 5e-4, f"max |logit difference| {difference:.3e}"


@pytest.mark.parametrize("depth", [0, 1], ids=["trunk", "prediction-layer"])
def test_a_qwen35_model_written_without_its_source_states_its_hybrid_fields_and_reloads(depth, tmp_path):
    """A Qwen3.5 model written from the model alone (`PretrainedDecoder.from_model`, a trained or scratch
    one holding no source config) states its delta net's geometry, its rotary's fraction and its
    prediction depth, which the generic export writes none of, and writes its delta net's tensors and
    its prediction layer's under their own names (`mtp.*`); so it reloads to its own variables, logits
    and prediction logits, bit for bit (ledger 4533). The prediction layer's case is a scratch model of
    the fixture's geometry with one, drawn whole."""
    from dew.interop.pretrained import PretrainedDecoder

    loaded = Pretrained.load(FIXTURES / "qwen35-tiny", dtype="float32", attention_impl="reference",
                             text_only=True)
    ids = jnp.arange(1, 9, dtype=jnp.int32)[None]
    model, variables = loaded.model, loaded.variables
    if depth:
        model = model.clone(num_nextn_predict_layers=depth)
        variables = nn.unbox(model.init(jax.random.key(4533), ids))
    PretrainedDecoder.from_model(model, variables).save(tmp_path / "export")
    restored = Pretrained.load(tmp_path / "export", dtype="float32", attention_impl="reference")
    assert flat_tree(restored.variables).keys() == flat_tree(variables).keys()
    for path, leaf in flat_tree(variables).items():
        np.testing.assert_array_equal(np.asarray(flat_tree(restored.variables)[path]), np.asarray(leaf))
    np.testing.assert_array_equal(np.asarray(restored.model.apply(restored.variables, ids)),
                                  np.asarray(model.apply(variables, ids)))
    if depth:
        np.testing.assert_array_equal(np.asarray(prediction_logits(restored.model, restored.variables, ids)),
                                      np.asarray(prediction_logits(model, variables, ids)))


def test_a_qwen35_checkpoint_decodes_as_it_scores_in_parallel():
    """The loaded weights through the cache: a prefill and single-token steps on par with the parallel
    forward at each step's width from float64 over drawn rows (check 1, ledger 2739)."""
    assert_decodes_as_it_scores(Pretrained.load(FIXTURES / "qwen35-tiny", dtype="float32", max_seq_len=16,
                                                attention_impl="reference"),
                                "qwen35 decode", seeds=DECODE_SEEDS["qwen35"], prefill=4, length=12,
                                salt=0x2739)


# --------------------------------------------------------------------------
# Qwen3-Next: the Qwen3.5 hybrid with fused delta net projections and MoE
# --------------------------------------------------------------------------

QWEN3_NEXT = FIXTURES / "qwen3-next-tiny"
def test_qwen3_next_logits_match_the_reference_implementation():
    """Full-model parity on the tiny hybrid: three fused delta net layers,
    the gated attention, and softmax top-2 routing with the gated shared
    expert on every layer. Largest observed max |logit difference| 2.3e-05
    on logits of magnitude 6.6, every argmax equal."""
    model, variables = fp32_decoder(QWEN3_NEXT)
    ids = np.load(QWEN3_NEXT / "input_ids.npy")
    reference = np.load(QWEN3_NEXT / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    assert scaled_difference(logits, reference) < 1e-4
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


QWEN3_NEXT_MTP_SEEDS = 256
"""The prediction depth's declared check-1 seeds against the published composition, sized by `seeds_for`
on armada paired with Dew's arm (`dew_fn`, 256 in 45 s): the reference compared with itself does not
resolve PARITY within 512 seeds."""


def qwen3_next_mtp_case():
    """Check 1's arms for Qwen3-Next's prediction depth (row 3912): Dew's first depth's logits over a
    draw of the fixture's rows (`prediction_logits`), and the published composition's, vLLM's
    Qwen3NextMultiTokenPredictor over transformers' own decoder layer (tools/qwen_mtp_reference.py's
    `QwenMTP`) over the actual transformers model's final states and embeddings, loaded from the
    checkpoint in float32 and as its float64 twin (`transformers_arms`); each draw's tokens redrawn until
    every routing the trunk and the depth take clears `selection_margins.MARGIN`."""
    import selection_margins
    import torch
    from interop_support import transformers_arms
    from reference_error import as_given
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    from tools.qwen_mtp_reference import QwenMTP, published_mtp_logits
    from tools.torch_precision import float64

    config = AutoConfig.from_pretrained(QWEN3_NEXT)
    stored = load_file(str(QWEN3_NEXT / "model.safetensors"))

    def experts(prefix):
        # The checkpoint's experts one by one, as transformers fuses them on load: each expert's gate
        # rows over its up rows, then its down map.
        each = range(config.num_experts)
        return {f"{prefix}.gate_up_proj": torch.stack([torch.cat([stored[f"{prefix}.{e}.gate_proj.weight"],
                                                                  stored[f"{prefix}.{e}.up_proj.weight"]])
                                                       for e in each]),
                f"{prefix}.down_proj": torch.stack([stored[f"{prefix}.{e}.down_proj.weight"] for e in each])}

    def built(dtype):
        # The eager experts: transformers' grouped product takes no float64 operands. Its own load fuses
        # the trunk's experts; the depth, which it does not know, takes the same fusion, held to the
        # trunk's as transformers loaded it.
        network = AutoModelForCausalLM.from_pretrained(QWEN3_NEXT, dtype=dtype, attn_implementation="eager",
                                                       experts_implementation="eager")
        loaded = network.state_dict()
        for name, fused in experts("model.layers.0.mlp.experts").items():
            assert torch.equal(loaded[name], fused.to(dtype)), name
        held = {name.removeprefix("mtp."): tensor for name, tensor in stored.items()
                if name.startswith("mtp.") and ".mlp.experts." not in name}
        held.update({name.removeprefix("mtp."): tensor
                     for name, tensor in experts("mtp.layers.0.mlp.experts").items()})
        # The model's own config, which carries the experts' implementation its load was given.
        network.mtp = QwenMTP(network.config)
        network.mtp.load_state_dict(held, strict=True)
        return network.to(dtype).eval()

    def predicted(network, ids):
        _, logits, admitted, _ = published_mtp_logits(network, {"input_ids": ids,
                                                                "attention_mask": torch.ones_like(ids)})
        return logits[admitted]

    with float64():
        twin = built(torch.float64)
    reference, truth = transformers_arms(built(torch.float32), predicted, twin=twin)
    model, variables = fp32_decoder(QWEN3_NEXT)
    compiled = jax.jit(lambda ids: prediction_logits(model, variables, ids))

    @as_given
    def dew(ids):
        logits = np.asarray(compiled(jnp.asarray(ids, jnp.int32)))
        return logits.reshape(-1, logits.shape[-1])

    shape = np.load(QWEN3_NEXT / "input_ids.npy").shape

    def draw(seed):
        return (np.random.default_rng([0x3912, seed]).integers(0, config.vocab_size, shape),)

    fields = translate_config(json.loads((QWEN3_NEXT / "config.json").read_text()))
    # The trunk's routing and the depth's, over the draw the depth's logits are judged on.
    captured = selection_margins.capture(
        model, lambda module, ids: module.mtp_logits(module.hidden_and_mtp_inputs(ids)[1], ids)[0])

    def clears(drawn):
        selections = captured(variables, jnp.asarray(drawn[0], jnp.int32))
        return selection_margins.margin(fields, selections, "prediction depth", stable=False) > \
            selection_margins.MARGIN
    return dew, reference, selection_margins.clearing(draw, clears), truth


def test_qwen3_next_prediction_layer_matches_the_published_composition():
    """The mtp.* tensors compose as vLLM's Qwen3NextMultiTokenPredictor does over the trunk's final states
    and the next tokens' embeddings: Dew's first prediction depth's logits on par with the published
    composition's over the actual transformers model at every position, from its float64 twin's (check
    1, row 3912, ledger 2739)."""
    from reference_error import assert_as_exact

    dew, reference, draw, truth = qwen3_next_mtp_case()
    assert_as_exact(dew, reference, draw, "qwen3-next prediction depth", seeds=QWEN3_NEXT_MTP_SEEDS,
                    truth_fn=truth)


def test_the_fused_delta_net_projection_is_read_by_key_head_group():
    """`fix_query_key_value_ordering` (modeling_qwen3_next.py:558-586) splits
    each key head's row group into its q, k and the v and z of the value
    heads it serves, so the fused kernel is not q|k|v|z stacked whole. The
    split-projection model on the regrouped rows must compute the same
    layer; on the rows stacked whole it must not."""
    from dew.nn.linear import GatedDeltaNet

    def net(fused: bool) -> GatedDeltaNet:
        return GatedDeltaNet(emb_features=16, num_k_heads=2, num_v_heads=4, head_k_dim=4,
                             head_v_dim=6, fused_in_proj=fused)

    fused, split = net(fused=True), net(fused=False)
    x = jax.random.normal(jax.random.key(0), (1, 5, 16))
    variables = nn.unbox(fused.init(jax.random.key(1), x))
    params = dict(variables["params"])
    qkvz_kernel = np.asarray(params.pop("in_proj_qkvz")["kernel"])
    ba_kernel = np.asarray(params.pop("in_proj_ba")["kernel"])
    q, k, v, z = np.split(qkvz_kernel.reshape(16, 2, 2 * 4 + 2 * 2 * 6), [4, 8, 8 + 12], axis=-1)
    ba = ba_kernel.reshape(16, 2, 4)
    regrouped = {
        **params,
        "in_proj_qkv": {
            "kernel": np.concatenate([q.reshape(16, -1), k.reshape(16, -1), v.reshape(16, -1)], -1)
        },
        "in_proj_z": {"kernel": z.reshape(16, -1)},
        "in_proj_b": {"kernel": ba[..., :2].reshape(16, -1)},
        "in_proj_a": {"kernel": ba[..., 2:].reshape(16, -1)},
    }
    whole = {**regrouped,
             "in_proj_qkv": {"kernel": qkvz_kernel[:, :2 * 8 + 24]},
             "in_proj_z": {"kernel": qkvz_kernel[:, 2 * 8 + 24:]}}

    wanted = np.asarray(fused.apply(variables, x))
    np.testing.assert_allclose(np.asarray(split.apply({"params": regrouped}, x)), wanted, atol=1e-6, rtol=0)
    assert float(np.abs(np.asarray(split.apply({"params": whole}, x)) - wanted).max()) > 1e-2


def test_a_qwen3_next_checkpoint_decodes_as_it_scores_in_parallel():
    """The fused delta net's conv and recurrent states through the cache: a prefill and single-token
    steps on par with the parallel forward at each step's width from float64 over drawn rows that clear
    its routing (check 1, ledger 2739)."""
    assert_decodes_as_it_scores(Pretrained.load(QWEN3_NEXT, dtype="float32", max_seq_len=16,
                                                attention_impl="reference"),
                                "qwen3_next decode", seeds=DECODE_SEEDS["qwen3_next"], prefill=4, length=12,
                                salt=0x2739)


# ---------------------------------------------------------------------------
# Gemma 3n: AltUp's residual copies, LAuReL, activation sparsity, per-layer widths
# ---------------------------------------------------------------------------

GEMMA3N = FIXTURES / "gemma3n-tiny"


def test_gemma3n_logits_match_the_reference_implementation():
    """fp32 parity: tolerance 1e-4, observed max |logit difference| 4.2e-06
    with identical argmax. The copies' projections, each layer's AltUp and
    LAuReL leaves and the sharing layer's missing K/V are what the tree
    holds."""
    model, variables = fp32_decoder(GEMMA3N)
    difference = recorded_logit_difference(model, variables, GEMMA3N)
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    params = variables["params"]
    assert {name for name in params if name.startswith("altup")} == {
        "altup_projections_0", "altup_projections_1",
        "altup_unembed_projections_0", "altup_unembed_projections_1"}
    assert params["layers_0"]["altup"]["prediction_coefs"]["kernel"].shape == (3, 9)
    assert params["layers_0"]["laurel"]["linear_left"]["kernel"].shape == (32, 8)
    assert params["layers_2"]["mlp"]["gate_proj"]["kernel"].shape == (32, 64)
    assert "k_proj" not in params["layers_3"]["self_attn"]


def test_gemma3n_decodes_through_the_cache_as_it_scores_in_parallel():
    """The stream of copies rides the decode path: single-token steps through the KV cache on par with
    the whole sequence at each step's width from float64 over drawn rows that clear its gaussian top-k
    (check 1, ledger 2739)."""
    assert_decodes_as_it_scores(Pretrained.load(GEMMA3N, dtype="float32", max_seq_len=16,
                                                attention_impl="reference"),
                                "gemma3n decode", seeds=DECODE_SEEDS["gemma3n"], prefill=1, length=12,
                                salt=0x2739, rows=1)
