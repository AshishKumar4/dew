"""DeepSeek-V4.1-Flash against the release's own inference code.

deepseek-v41-tiny comes from tools/deepseek_v41_reference.py: the release's
inference/model.py at dba1be0a (DeepSeek-V4.1-Flash) run in fp32 over the
torch stand-ins for its kernels, whose quantizers match the tilelang kernels
bit for bit on bf16 input (`--check-kernels`). The architecture outputs are
taken with the quantizers off and repeated with them on (`qat_*`). Every
quantizer and top-k call of the reference's forward is recorded with its
margins: Dew's quantizers are held bit for bit to each recorded call, and
Dew's own calls may round or select differently only where the recorded
margin sits within the fp32 noise between the two (source.json's `noise`).

TOLERANCE is twice the largest distance from the reference that
`tools/deepseek_v41_numerics.py residuals` measures on CPU and on GPU. Its
`fp64` mode runs both sides widened to fp64, where they agree to 1e-13 on
every output, so each distance is fp32 rounding; one SGD step amplifies it,
as the reference's own fp32 update already lies 1.3e-4 from its fp64 one.
"""

import dataclasses
import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import load_pretrained
from dew.interop.hf_decoders import _FAMILIES, _flatten, translate_config
from dew.nn.engram import Engram
from dew.nn.fake_quant import fake_quant_fp4, fake_quant_fp8
from dew.nn.inputs import ModelInputs
from dew.registry import models, with_precision
from dew.sampling import Sampling, generate
from tools.deepseek_v41_numerics import (
    cached_run,
    captured,
    input_noise,
    loss_and_gradient,
    row_noise,
    selections,
    stepped,
    unquantized,
)

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "deepseek-v41-tiny"
RELEASED = ROOT / "deepseek-v41-flash"
TOLERANCE = {
    "logits": 1e-5, "loss": 4e-6, "updated_logits": 4e-4, "prompt_logits": 1e-5,
    "decode_logits": 1e-5, "draft_logits": 1e-5, "draft_confidence": 1.6e-5,
    "qat_logits": 8e-6, "qat_loss": 1e-6, "qat_updated_logits": 1.5e-3, "qat_prompt_logits": 2.5e-5,
    "qat_decode_logits": 7e-6, "qat_draft_logits": 4e-6, "qat_draft_confidence": 6.2e-6,
}


@pytest.fixture(scope="module", autouse=True)
def fp32_matmuls():
    """The reference multiplies in fp32, where a GPU's default is TF32."""
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(scope="module")
def source():
    loaded = load_pretrained(TINY, dtype="float32", attention_impl="reference")
    return loaded, unquantized(loaded.model), np.load(TINY / "reference.npz")


def close(actual, reference, name: str):
    """`actual` within TOLERANCE[name] of the reference's `name`, its argmax
    exact."""
    actual = np.asarray(actual)
    np.testing.assert_allclose(actual, reference[name], atol=TOLERANCE[name], rtol=0, err_msg=name)
    np.testing.assert_array_equal(actual.argmax(-1), reference[name].argmax(-1), err_msg=name)


def test_the_quantizers_round_as_the_release_kernels():
    """FP8 over 32 channels under power-of-two scales, FP4 over 16 under E4M3
    scales and over 32 under power-of-two scales, bit for bit against the
    kernels' torch stand-ins (signed zeros included), ties, all-zero blocks
    and saturation included, compiled, on whichever backend runs the test."""
    torch = pytest.importorskip("torch")
    from tools import deepseek_v41_kernels as kernels

    rng = np.random.default_rng(0)
    blocks = [rng.standard_normal((64, 64)) * scale * rng.random((64, 1)) * 3
              for scale in (1e-3, 0.3, 3.0, 40.0, 1e4)]
    ties = np.tile(np.array([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 6]), 4)
    ties[0] = 6.0  # the block's amax, so its scale is exactly one
    x = np.concatenate([*blocks, np.stack([ties, -ties])]).astype(np.float32)
    x[3, :32] = 0
    for jax_quant, torch_quant in (
            (lambda v: fake_quant_fp8(v, 32), lambda v: kernels.act_quant(v, 32, "ue8m0", None, inplace=True)),
            (lambda v: fake_quant_fp4(v, 16, True),
             lambda v: kernels.fp4_act_quant(v, 16, inplace=True, scale_dtype=torch.float8_e4m3fn)),
            (lambda v: fake_quant_fp4(v, 32, False), lambda v: kernels.fp4_act_quant(v, 32, inplace=True))):
        expected = torch_quant(torch.from_numpy(x.copy())).numpy()
        # under jit, where XLA GPU would delete a convert-pair rounding
        actual = np.asarray(jax.jit(jax_quant)(jnp.asarray(x)))
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


QUANTIZERS = {"window": lambda v: fake_quant_fp8(v, 32), "entries": lambda v: fake_quant_fp4(v, 16, True),
              "index": lambda v: fake_quant_fp4(v, 32, False)}


def test_the_quantizers_round_every_call_the_reference_made(source):
    """Every quantizer call of the reference's quantization-aware forward,
    fed the input it recorded, gives the output it recorded bit for bit:
    the rounding is held to the release on the values the model meets,
    whatever margins the fixture's seed left them."""
    _, _, reference = source
    for site, quantize in QUANTIZERS.items():
        actual = np.asarray(jax.jit(quantize)(jnp.asarray(reference[f"qat_{site}_in"])))
        np.testing.assert_array_equal(actual.view(np.uint32), reference[f"qat_{site}_out"].view(np.uint32),
                                      err_msg=site)


def test_every_rounding_and_selection_meets_the_reference_within_the_noise(source):
    """Dew's quantization-aware and plain forwards hand every quantizer and
    top-k call the input the reference recorded for it, to within the fp32
    noise between the two (source.json's `noise`, which
    tools/deepseek_v41_numerics.py measures), and a rounded value or a pick
    differs from the recorded one only where the reference had it within
    that noise of where it changes. No seed's margins enter."""
    loaded, plain, reference = source
    noise = json.loads((TINY / "source.json").read_text())["noise"]
    ids = jnp.asarray(reference["input_ids"])
    for prefix, model in (("qat_", loaded.model), ("", plain)):
        record = captured(model, loaded.variables, ids)
        assert set(record.blocks) == ({*QUANTIZERS} if prefix else set())
        for site, blocks in record.blocks.items():
            ours = np.concatenate(blocks)
            assert np.max(input_noise(ours, reference[f"{prefix}{site}_in"])) <= noise[site], site
            rounded = np.asarray(jax.jit(QUANTIZERS[site])(jnp.asarray(ours)))
            explained = ((reference[f"{prefix}{site}_margin"] < noise[site])
                         | (reference[f"{prefix}{site}_scale_margin"][:, None] < noise[site]))
            assert not np.any((rounded != reference[f"{prefix}{site}_out"]) & ~explained), site
        ours, our_picks, theirs, their_picks = selections(record, reference, prefix)
        assert np.max(row_noise(ours, theirs, reference[f"{prefix}selection_scale"])) <= noise["selection"]
        moved = np.any(our_picks != their_picks, -1)
        assert not np.any(moved & (reference[f"{prefix}selection_margin"] >= noise["selection"]))


def test_the_quantizers_pass_their_gradient_straight_through():
    x = jnp.linspace(-3.0, 3.0, 64).reshape(2, 32)
    for quant in (lambda v: fake_quant_fp8(v, 32), lambda v: fake_quant_fp4(v, 16, True)):
        np.testing.assert_array_equal(jax.grad(lambda v: jnp.sum(quant(v) * v))(x),
                                      quant(x) + x)
    # bf16 input whose E4M3 scale saturates: the forward value is the
    # rounded one exactly, not a bf16 re-rounding of the correction
    big = jnp.asarray([[1e4] + [3.0] * 15], jnp.bfloat16)
    rounded = fake_quant_fp4(big.astype(jnp.float32), 16, True).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(fake_quant_fp4(big, 16, True)), np.asarray(rounded))


def test_the_engram_hash_is_the_releases_int64_arithmetic():
    """At the release's sizes (a compressed vocabulary of 99092, buckets
    above 16M, 8 heads over 2- to 4-grams) the limb arithmetic equals
    int64 multiply, XOR and modulo, whose products reach 2**63."""
    released = json.loads((RELEASED / "config.json").read_text())["text_config"]
    spec = Engram(layer_ids=released["engram_layer_ids"], num_embeddings=released["engram_num_embeddings"],
                  max_ngram_size=released["engram_max_ngram_size"], vocab_size=released["engram_vocab_size"],
                  n_heads=released["engram_n_heads"], head_dim=released["engram_head_dim"],
                  compressed_vocab_size=released["engram_compressed_vocab_size"])
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, spec.compressed_vocab_size, (3, 40, spec.max_ngram_size))
    tokens[0, 0] = spec.compressed_vocab_size - 1
    blocked = rng.random(tokens.shape) < 0.1
    pad = 7
    ids = np.asarray(spec.hash_ids(jnp.asarray(tokens, jnp.int32), jnp.asarray(blocked), pad))
    values = np.where(blocked, pad, tokens).astype(np.int64)
    products = values[:, :, None, :] * spec.multipliers  # [B, S, layers, n]
    primes = np.asarray(spec.primes, np.int64)
    offsets = np.concatenate([np.zeros((len(spec.layer_ids), 1), np.int64),
                              np.cumsum(primes.reshape(len(spec.layer_ids), -1), axis=1)[:, :-1]], axis=1)
    rolling, expected = products[..., 0], []
    for size in range(1, spec.max_ngram_size):
        rolling = np.bitwise_xor(rolling, products[..., size])
        expected.append(rolling[..., None] % primes[:, size - 1])
    np.testing.assert_array_equal(ids, np.concatenate(expected, -1) + offsets)


@pytest.fixture(scope="module")
def released():
    """The released config, its fields, the model they build and its tree's shapes."""
    config = json.loads((RELEASED / "config.json").read_text())
    fields = translate_config(config)
    model = models.build("causal_transformer", **fields)
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))
    return config, fields, model, shapes


def test_every_released_tensor_lands_on_one_leaf_of_the_released_tree(released):
    """The pinned weight index's 96085 tensors, their FP8/FP4 `.scale`
    partners aside, map onto the tree the released config builds, and
    together they cover it; the vision tower, its aligner, the image span
    embeddings and the routers' image-token bias are the vision half, which
    the text model retains by name."""
    config, fields, model, shapes = released
    family = _FAMILIES["deepseek_v41"]
    text = config["text_config"]
    names = []
    for name in json.loads((RELEASED / "tensor_names.json").read_text())["names"]:
        if ".experts.K." in name:
            experts = (text["dspark_n_routed_experts"] if name.startswith("mtp.")
                       else text["n_routed_experts"])
            names += [name.replace(".K.", f".{index}.") for index in range(experts)]
        else:
            names.append(name)
    assert len(names) == len(set(names))
    placeholders = {name: np.zeros((2, 4, 2) if name.endswith("wo_a.weight") else (1,)) for name in names}
    placeholders.update({name.removesuffix("wo_a.weight") + "wq_b.weight": np.zeros((8, 1))
                         for name in names if name.endswith("wo_a.weight")})
    prepared = family.prepare_weights(placeholders)
    paths = {name: family.weight_path(name, fields) for name in prepared}
    retained = {name for name, path in paths.items() if path is None}
    assert retained and all(name.startswith(("vision.", "aligner.", "image_")) or name.endswith("bias_vl")
                            for name in retained)
    bound = [path for path in paths.values() if path is not None]
    assert len(set(bound)) == len(bound)
    # One tensor per expert stacks into one leaf per projection.
    bound = {tuple(part for index, part in enumerate(path)
                   if not (index and path[index - 1] == 'experts' and part.isdigit()))
             for path in bound}
    tree = {tuple(name.split(".")) for name in _flatten(dict(shapes))}
    tree.discard(("constants", "engram_hashes", "token_map"))
    assert bound == tree
    assert model.num_layers == 40 and model.dspark.stages == 3 and model.mixture.experts == 384


def test_every_matrix_of_the_released_tree_shards_by_a_declared_rule(released):
    """No weight of two or more axes is left to the shape heuristic unasked:
    the engram tables alone hold 384M rows a layer, which shard as a
    vocabulary's do."""
    from dew.nn.sharding import declared_axes, is_heuristic

    *_, shapes = released
    uncovered = [jax.tree_util.keystr(path) for path, leaf in jax.tree_util.tree_flatten_with_path(shapes)[0]
                 if leaf.ndim >= 2 and declared_axes(path, leaf.ndim) is None and not is_heuristic(path)]
    assert uncovered == []
    engram = [declared_axes(path, leaf.ndim) for path, leaf in jax.tree_util.tree_flatten_with_path(shapes)[0]
              if jax.tree_util.keystr(path).endswith("['engram']['embed']['embedding']")]
    assert engram == [("vocab", None)] * 2


def test_the_forward_matches_the_reference(source):
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    close(jax.jit(plain.apply)(loaded.variables, ids), reference, "logits")


def test_the_quantization_aware_forward_matches_the_reference(source):
    loaded, _, reference = source
    ids = jnp.asarray(reference["input_ids"])
    close(jax.jit(loaded.model.apply)(loaded.variables, ids), reference, "qat_logits")


def test_the_quantization_aware_loss_and_gradient_match_the_reference(source):
    """The loss through every quantizer, and one SGD step along its gradient,
    which passes each quantizer straight through, read back through the
    forward without quantization, so the step compares the gradient alone."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    value, gradient = loss_and_gradient(loaded.model, loaded.variables, ids)
    np.testing.assert_allclose(value, reference["qat_loss"], atol=TOLERANCE["qat_loss"], rtol=0)
    variables = stepped(loaded.variables, gradient, reference["learning_rate"])
    close(jax.jit(plain.apply)(variables, ids), reference, "qat_updated_logits")


def test_the_candidate_pool_decides_the_logits(source):
    """Dropping the pool's restriction on the Reindex layers moves the logits
    past the tolerance, so the fixture exercises the hierarchical indexer."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    kinds = dict(plain.kinds)
    kinds["csa2_ratio_1_reindex"] = dataclasses.replace(
        kinds["csa2_ratio_1_reindex"], mixer=dataclasses.replace(
            kinds["csa2_ratio_1_reindex"].mixer, candidates=None, candidate_blocks=None,
            candidate_block_size=None))
    unpooled = plain.clone(kinds=flax.core.freeze(kinds))
    moved = np.max(np.abs(np.asarray(unpooled.apply(loaded.variables, ids)) - reference["logits"]))
    assert moved > TOLERANCE["logits"]


def test_the_update_exports_and_decodes_as_the_reference(source, tmp_path):
    """Loss, one SGD step with every gradient (the indexer's are zero, as the
    reference's selection passes none), the source-layout export read back
    bit for bit, and greedy decoding after the fixture's prompt."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    value, gradient = loss_and_gradient(plain, loaded.variables, ids)
    np.testing.assert_allclose(value, reference["loss"], atol=TOLERANCE["loss"], rtol=0)
    indexer = [np.max(np.abs(leaf)) for path, leaf in _flatten(gradient).items() if ".indexer." in f".{path}."]
    assert indexer and max(indexer) == 0
    variables = stepped(loaded.variables, gradient, reference["learning_rate"])
    loaded.save(tmp_path, variables=variables)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    held, again = _flatten(variables), _flatten(restored.variables)
    assert held.keys() == again.keys()
    for name, leaf in again.items():
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(held[name]), err_msg=name)
    close(unquantized(restored.model).apply(restored.variables, ids), reference, "updated_logits")
    prompt = int(reference["decode_prompt"])
    generated = generate(plain, loaded.variables, ModelInputs(ids[:, :prompt]),
                         reference["generated"].shape[1], key=jax.random.key(1),
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens)[:, -reference["generated"].shape[1]:],
                                  reference["generated"])


def cached_matches(model, variables, reference, prefix: str):
    ids = jnp.asarray(reference["input_ids"])
    prompt_logits, steps, (draft_ids, draft_logits, confidence) = cached_run(
        model, variables, ids, int(reference["decode_prompt"]))
    close(prompt_logits, reference, f"{prefix}prompt_logits")
    close(steps, reference, f"{prefix}decode_logits")
    close(draft_logits, reference, f"{prefix}draft_logits")
    name = f"{prefix}draft_confidence"
    np.testing.assert_allclose(confidence, reference[name], atol=TOLERANCE[name], rtol=0)
    np.testing.assert_array_equal(draft_ids, reference[f"{prefix}draft_ids"])


def test_the_decode_cache_and_the_drafter_match_the_reference(source):
    """The cached prefill and teacher-forced steps cover the window ring,
    ratio-2 windows closing across calls, the shared entries, index keys,
    selections and candidate pool, and the engram history; DSpark's windows
    seed on the prompt and draft after every step."""
    loaded, plain, reference = source
    cached_matches(plain, loaded.variables, reference, "")


def test_the_quantization_aware_cache_and_drafter_match_the_reference(source):
    """The same cached run with the cache rounded as the release stores it:
    each window key, entry and index key rounded where the cache takes it,
    DSpark's window keys included."""
    loaded, _, reference = source
    cached_matches(loaded.model, loaded.variables, reference, "qat_")


def test_left_padding_changes_no_row(source):
    """Rows of different lengths in one batch, the shorter padded on the
    left: engram's look-back packs each row's valid tokens (its argsort
    path), and CSA2's windows, entries and index keys follow each row's own
    positions. The full row still decodes the reference's greedy ids, and
    the padded one what it decodes alone."""
    loaded, plain, reference = source
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])
    steps, pad = reference["generated"].shape[1], 3

    def greedy(inputs):
        return np.asarray(generate(plain, loaded.variables, inputs, steps, key=jax.random.key(1),
                                   sampling=Sampling(temperature=0)).tokens)[:, -steps:]

    tokens = jnp.stack([ids[0, :prompt], jnp.pad(ids[1, :prompt - pad], (pad, 0))])
    valid = jnp.arange(prompt)[None] >= jnp.asarray([[0], [pad]])
    batched = greedy(ModelInputs(tokens, {"attention_mask": valid}))
    np.testing.assert_array_equal(batched[0], reference["generated"][0])
    np.testing.assert_array_equal(batched[1], greedy(ModelInputs(ids[1:, :prompt - pad]))[0])


def test_consecutive_reindex_layers_publish_their_selections_under_scan_layers():
    """Two Reindex layers of one kind in a row stay two runs of one under
    scan_layers, so the Reuse layer after them attends the second one's
    selection, as it does unrolled; scanned together, their publications
    would stay inside the loop."""
    config = json.loads((TINY / "config.json").read_text())
    text = config["text_config"]
    config = {**config, "text_config": {**text, "index_source_layer_ids": [2, 4, 6, 7, 8, 10]}}
    fields = with_precision("causal_transformer", translate_config(config), dtype="float32",
                            attention_impl="reference")
    ids = jnp.asarray(np.load(TINY / "reference.npz")["input_ids"])
    unrolled, scanned = (models.build("causal_transformer", **{**fields, "scan_layers": scan})
                         for scan in (False, True))
    variables = unrolled.init(jax.random.key(0), ids)
    np.testing.assert_allclose(scanned.apply(variables, ids), unrolled.apply(variables, ids),
                               atol=TOLERANCE["logits"], rtol=0)


def test_a_config_without_swiglu_limit_clamps_nothing():
    """The release's ModelArgs default is 0.0, no clamp (v41 model.py:70)."""
    config = json.loads((TINY / "config.json").read_text())
    text = {key: value for key, value in config["text_config"].items() if key != "swiglu_limit"}
    assert translate_config({**config, "text_config": text})["swiglu_limit"] is None
