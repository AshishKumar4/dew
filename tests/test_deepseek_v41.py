"""DeepSeek-V4.1-Flash against the release's own inference code.

deepseek-v41-tiny comes from tools/deepseek_v41_reference.py: the release's
inference/model.py at dba1be0a (DeepSeek-V4.1-Flash) run in fp32 over the
torch stand-ins for its kernels. kernels.npz holds the release's own tilelang
kernels' outputs (`--kernels`, on a GPU), which Dew's quantizers and the
stand-ins are held to bit for bit on bf16 input. The architecture outputs are
taken with the quantizers off and repeated with them on (`qat_*`), and every
quantizer and top-k call of both forwards is recorded. reference_f64.npz is
the same run widened to fp64 (`--fp64`), the truth Dew and the reference are
both measured from by tests/reference_error.py's rule: Dew's RMS distance
from it at most FACTOR times the reference's own, for every output and for
the inputs of every quantizer and top-k call. Dew's quantizers are held bit
for bit to each recorded call.

A value within fp32 noise of where a rounding or a pick changes can go
either way in two fp32 runs, and every seed leaves some (an L4 rounded one
window key of the cached prefill the other way at the seed a CPU and an RTX
4080 had passed). So each run compared here takes every rounding and pick
from its own float64 twin (`decided`), the way the exact value goes, which
is the way the reference went (`--fp64` refuses a fixture it did not):
the rule then measures fp32 rounding alone, whatever the seed. Each twin is
held within float64 rounding of the truth (`twin_close`), which a single
rounding or pick taken otherwise than the truth's would move it far past.
`tools/deepseek_v41_numerics.py residuals` reports every ratio and each
twin's distance from the truth.
"""

import dataclasses
import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict
from reference_error import FACTOR, distance

from dew.batch import coordinates
from dew.interop import Pretrained
from dew.interop.hf_decoders import families, translate_config, translate_wrapper_config
from dew.nn.engram import Engram
from dew.nn.fake_quant import E2M1, E2M1_MIDPOINTS, fake_quant_fp4, fake_quant_fp8
from dew.nn.inputs import ModelInputs
from dew.nn.kv_cache import KVCache
from dew.nn.rope import LongRopeScaling, RopeScaling
from dew.registry import models
from dew.sampling import Sample, Sampling, Speculative, generate
from tools.deepseek_v41_numerics import QUANTIZERS, unquantized, vision_bundle, vision_run

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "deepseek-v41-tiny"
RELEASED = ROOT / "deepseek-v41-flash"
TRUTH = np.load(TINY / "reference_f64.npz")
KERNELS = TINY / "kernels.npz"
# kernels.npz's quantizer outputs by the Dew quantizer each is.
KERNEL_QUANTIZERS = {
    "fp8_32_ue8m0": lambda v: fake_quant_fp8(v, 32),
    "fp4_16_e4m3": lambda v: fake_quant_fp4(v, 16, e4m3_scale=True),
    "fp4_32_e8m0": lambda v: fake_quant_fp4(v, 32, e4m3_scale=False),
}
@pytest.fixture(scope="module", autouse=True)
def fp32_matmuls():
    """The reference multiplies in fp32, where a GPU's default is TF32."""
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(scope="module")
def source():
    loaded = Pretrained.load(TINY, dtype="float32", attention_impl="reference")
    return loaded, unquantized(loaded.model), np.load(TINY / "reference.npz")


def rounding(name: str) -> float:
    """The reference's own RMS distance from the float64 truth for `name`:
    the fp32 rounding two runs of one computation may each carry FACTOR
    times of."""
    return distance(np.load(TINY / "vision.npz")[name] if name.startswith("vision_")
                    else np.load(TINY / "reference.npz")[name], TRUTH[name])


def kernel_outputs() -> dict[str, np.ndarray]:
    with np.load(KERNELS) as loaded:
        return dict(loaded)


def test_the_quantizers_round_as_the_release_kernels():
    """FP8 over 32 channels under power-of-two scales, FP4 over 16 under E4M3
    scales and over 32 under power-of-two scales, bit for bit against the
    release's tilelang kernels on bf16 rows over nine magnitude decades (an
    all-zero block and saturating E4M3 scales among them) and on every E2M1
    tie, compiled, on whichever backend runs the test."""
    kernels = kernel_outputs()
    for label in ("quant", "ties"):
        x = jnp.asarray(kernels[f"{label}/input"].view(ml_dtypes.bfloat16))
        for name, quantize in KERNEL_QUANTIZERS.items():
            # under jit, where XLA GPU would delete a convert-pair rounding
            actual = np.asarray(jax.jit(quantize)(x)).view(np.int16)
            np.testing.assert_array_equal(actual, kernels[f"{label}/{name}"], err_msg=f"{label} {name}")


def test_the_quantizers_round_fp32_input_as_the_stand_ins(source):
    """The kernels take bf16 only, so on fp32 input the stand-ins, held to
    the kernels on bf16 above, define the rounding: Dew's quantizers match
    them bit for bit (signed zeros included), ties, all-zero blocks and
    saturation included, compiled."""
    import torch

    from tools import deepseek_v41_kernels as kernels

    rng = np.random.default_rng(0)
    blocks = [rng.standard_normal((64, 64)) * scale * rng.random((64, 1)) * 3
              for scale in (1e-3, 0.3, 3.0, 40.0, 1e4)]
    grid = np.sort(np.concatenate([E2M1[:8], E2M1_MIDPOINTS]))
    ties = np.tile(np.append(grid, 6.0), 4)
    ties[0] = 6.0  # the block's amax, so its scale is exactly one
    x = np.concatenate([*blocks, np.stack([ties, -ties])]).astype(np.float32)
    x[3, :32] = 0
    for jax_quant, torch_quant in (
        (lambda v: fake_quant_fp8(v, 32), lambda v: kernels.act_quant(v, 32, "ue8m0", None, inplace=True)),
        (
            lambda v: fake_quant_fp4(v, 16, e4m3_scale=True),
            lambda v: kernels.fp4_act_quant(v, 16, inplace=True, scale_dtype=torch.float8_e4m3fn),
        ),
        (
            lambda v: fake_quant_fp4(v, 32, e4m3_scale=False),
            lambda v: kernels.fp4_act_quant(v, 32, inplace=True),
        ),
    ):
        expected = torch_quant(torch.from_numpy(x.copy())).numpy()
        actual = np.asarray(jax.jit(jax_quant)(jnp.asarray(x)))
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    _, _, reference = source
    for site, quantize in QUANTIZERS.items():
        actual = np.asarray(jax.jit(quantize)(jnp.asarray(reference[f"qat_{site}_in"])))
        np.testing.assert_array_equal(actual.view(np.uint32),
                                      reference[f"qat_{site}_out"].view(np.uint32), err_msg=site)


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
    """The released bundle's config, its wrapper record, the model it builds
    and its tree's shapes."""
    from dew.interop.pretrained import _wrapper_model

    config = json.loads((RELEASED / "config.json").read_text())
    record = translate_wrapper_config(config)
    model = _wrapper_model(
        config, record, models.build("causal_transformer", record["text"]), dtype="float32"
    )
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))
    return config, record, model, shapes


def test_the_candidate_pool_decides_the_logits(source):
    """Dropping the pool's restriction on the Reindex layers moves the logits
    further from the float64 truth than rounding can, so the fixture
    exercises the hierarchical indexer."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    kinds = dict(plain.kinds)
    kinds["csa2_ratio_1_reindex"] = dataclasses.replace(
        kinds["csa2_ratio_1_reindex"], mixer=dataclasses.replace(
            kinds["csa2_ratio_1_reindex"].mixer, candidates=None, candidate_blocks=None,
            candidate_block_size=None))
    unpooled = plain.clone(kinds=flax.core.freeze(kinds))
    assert distance(unpooled.apply(loaded.variables, ids), TRUTH["logits"]) > FACTOR * rounding("logits")


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


@pytest.mark.parametrize("given", ["attention_mask", "coordinates"])
def test_an_uncached_left_padded_row_pools_its_windows_over_its_own_tokens(source, given):
    """Uncached, rows padded on the left by three tokens, a count no compression rate divides,
    pool their compressed windows (CSA2's entries and index keys, HCA's) and hash their n-grams
    (Engram) over their real tokens at their ranks, as a cache appends them, whether the padding
    comes as a raw mask or as the tokens' coordinates alone: the real tokens' logits lie within
    FACTOR of the reference's rounding of the float64 truth of the rows unpadded."""
    loaded, plain, reference = source
    ids, pad = jnp.asarray(reference["input_ids"]), 3
    valid = jnp.broadcast_to(jnp.arange(ids.shape[1] + pad) >= pad, (ids.shape[0], ids.shape[1] + pad))
    padding = {"attention_mask": valid} if given == "attention_mask" else {
        "coordinates": coordinates(valid=valid)}
    logits = plain.apply(loaded.variables, jnp.pad(ids, ((0, 0), (pad, 0))), **padding)
    assert distance(logits[:, pad:], TRUTH["logits"]) <= FACTOR * rounding("logits")


def test_an_interior_gap_leaves_every_window_on_its_rows_real_tokens(source):
    """Uncached, three padding slots inside each row leave each real token's sliding window counting
    its row's real tokens (`counted_places`), as a cache's slots count them, and its compressed windows
    and n-grams on them: the real tokens' logits lie within FACTOR of the reference's rounding of the
    float64 truth of the rows without the gap, where a window counted by index let the gap crowd out
    the keys before it (audit NN-02)."""
    loaded, plain, reference = source
    ids, at, gap = jnp.asarray(reference["input_ids"]), 5, 3
    tokens = jnp.concatenate([ids[:, :at], jnp.zeros((ids.shape[0], gap), ids.dtype), ids[:, at:]], axis=1)
    slots = jnp.arange(tokens.shape[1])
    real = (slots < at) | (slots >= at + gap)
    logits = plain.apply(loaded.variables, tokens, attention_mask=jnp.broadcast_to(real, tokens.shape))
    assert distance(logits[:, real], TRUTH["logits"]) <= FACTOR * rounding("logits")


SHIFT = 8
"""How far the shifted rows' given positions sit past their indices: a multiple of every compression
rate, so a query that read entries by its given place would see SHIFT // rate windows of later tokens."""


def test_shifted_positions_compute_the_unshifted_rows_logits(source):
    """Uncached, positions given SHIFT past the tokens' indices: every rotation is relative, each
    compressed entry is visible by its window's tokens' ranks and rotates at its window's first given
    place, where its queries rotate (`window_origins`), so the logits lie within FACTOR of the
    reference's rounding of the float64 truth of the rows at their indices; entries visible by the
    given places showed each query windows of later tokens, and entries rotated at zero-based
    origins sat SHIFT away from their queries (audit NN-03)."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    positions = jnp.broadcast_to(jnp.arange(ids.shape[1]) + SHIFT, ids.shape)
    logits = plain.apply(loaded.variables, ids, positions=positions)
    assert distance(logits, TRUTH["logits"]) <= FACTOR * rounding("logits")


def test_a_shifted_prompt_decodes_the_unshifted_rows_logits(source):
    """Cached, a prompt at positions SHIFT past its indices, then the rest of each row one token at a
    time: each compressed entry rotates at its window's first given place, which the cache carries
    across the calls a window spans (`append_windows`), so every step's logits lie within FACTOR of
    the reference's rounding of the float64 truth of the rows at their indices, where entries rotated
    at their slots sat SHIFT away from their queries (audit NN-03)."""
    from dew.sampling.text import decode_ops, prefill_state

    loaded, plain, reference = source
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])
    ops = decode_ops(plain, loaded.variables, 0, None)
    positions = jnp.broadcast_to(jnp.arange(prompt) + SHIFT, (ids.shape[0], prompt))
    inputs = ModelInputs(ids[:, :prompt], {"positions": positions})
    state, _ = prefill_state(plain, loaded.variables, inputs, ops)
    steps = [np.asarray(state.logits)]
    for column in range(prompt, ids.shape[1]):
        state = ops.advance(state, ids[:, column], jnp.ones(ids.shape[0], bool))
        steps.append(np.asarray(state.logits))
    assert distance(np.stack(steps, axis=1), TRUTH["logits"][:, prompt - 1:]) <= FACTOR * rounding("logits")


def test_a_llama3_ramp_reaches_every_layer_and_longrope_is_refused(source):
    """V4's layers take the model's one rope scaling (`MixerContext.rotary_scaling`), each kind's own
    in its place where the kind gives one (`kind_of`), as the compressed kinds give their YaRN ramp: a
    Llama 3.1 ramp moves the logits from the unscaled decoder's further than rounding can, where the
    layers forwarded YaRN alone and rotated at the plain frequencies. LongRoPE, whose factors a call's
    reach picks and which the compressed entries do not see, is refused by name (audit NN-04)."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    ramp = RopeScaling(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                       original_max_position_embeddings=4)
    scaled, unscaled = (plain.clone(yarn=None, rope_scaling=scaling).apply(loaded.variables, ids)
                        for scaling in (ramp, None))
    assert distance(scaled, unscaled) > FACTOR * rounding("logits")
    longrope = LongRopeScaling(short_factor=(1.0,) * 4, long_factor=(2.0,) * 4,
                               original_max_position_embeddings=4)
    with pytest.raises(ValueError):
        plain.clone(yarn=None, rope_scaling=longrope).apply(loaded.variables, ids)


def test_an_uncached_draft_reads_a_padded_context_at_its_real_positions(source):
    """Uncached, DSpark drafts after a context whose padding `valid` marks, after it in one row and
    before it in the other, as after the context alone: its real positions keep their ranks, the
    padding is no key, and the block follows the last real position. The two drafts' logits, each
    within FACTOR of the reference's rounding, lie within twice that of each other."""
    loaded, plain, reference = source
    ids, prompt, pad = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"]), 3
    _, sown = plain.apply(loaded.variables, ids[:, :prompt], mutable=["prediction_inputs"])
    context = plain.apply(loaded.variables, sown["prediction_inputs"], method="draft_context")
    drawn = ids[:, prompt]
    _, alone, _ = plain.apply(loaded.variables, context, drawn, decode=False, method="draft")
    padded = jnp.stack([jnp.pad(context[0], ((0, pad), (0, 0))), jnp.pad(context[1], ((pad, 0), (0, 0)))])
    places = jnp.arange(prompt + pad)
    valid = jnp.stack([places < prompt, places >= pad])
    _, drafted, _ = plain.apply(loaded.variables, padded, drawn, decode=False, valid=valid, method="draft")
    assert distance(drafted, alone) <= 2 * FACTOR * rounding("draft_logits")


def test_speculative_decoding_drafts_with_dspark_and_emits_the_greedy_walk(source):
    """At zero temperature every rejected draft is replaced by the target's
    own token, so Speculative decoding with DSpark's blocks emits the greedy
    walk, a left-padded row included."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])[:, :int(reference["decode_prompt"])]
    valid = jnp.ones(ids.shape, bool).at[1, :3].set(False)
    inputs = ModelInputs(jnp.where(valid, ids, 0), {"attention_mask": valid})
    walked, speculated = (generate(plain, loaded.variables, inputs, 8, key=jax.random.key(0),
                                   sampling=Sampling(temperature=0), strategy=strategy)
                          for strategy in (Sample(), Speculative(block=3)))
    np.testing.assert_array_equal(np.asarray(speculated.tokens), np.asarray(walked.tokens))
    np.testing.assert_array_equal(np.asarray(speculated.lengths), 8)


def test_the_decode_ops_draft_what_dspark_drafts_over_the_history(source):
    """After a prefill and a verified stretch whose rows keep different
    counts, the block the decode ops draft is the one the drafter drafts
    uncached over each row's real history: the prompt's and the stretch's
    context reach its windows at their own positions. Two runs each within
    FACTOR of the reference's rounding lie within twice that of each other."""
    from dew.sampling.text import decode_ops, prefill_state

    loaded, plain, reference = source
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])
    ops = decode_ops(plain, loaded.variables, 0, 0)
    assert ops.record is not None and ops.draft is not None and ops.verify is not None
    state, _ = prefill_state(plain, loaded.variables, ModelInputs(ids[:, :prompt]), ops)
    kept = jnp.asarray([3, 1])
    keep = jnp.arange(3)[None, :] < kept[:, None]
    state, _, context = ops.verify(state, ids[:, prompt:prompt + 3], keep)
    assert context is not None
    state = ops.record(state, context, keep)
    drawn, scored = ids[:, prompt + 3], []

    def choose(index, logits):
        scored.append(logits)
        return jnp.argmax(logits, axis=-1)

    ops.draft(state, drawn, choose)
    for row in range(2):
        history = ids[row:row + 1, :prompt + int(kept[row])]
        _, sown = plain.apply(loaded.variables, history, mutable=["prediction_inputs"])
        whole = plain.apply(loaded.variables, sown["prediction_inputs"], method="draft_context")
        _, logits, _ = plain.apply(loaded.variables, whole, drawn[row:row + 1], decode=False,
                                   method="draft")
        assert distance(jnp.stack(scored, 1)[row], logits[0]) <= 2 * FACTOR * rounding("draft_logits")


@pytest.mark.parametrize("quantized", [None, "int8", "float8_e4m3fn"])
def test_a_paged_window_cache_decodes_what_the_compact_cache_decodes(source, quantized):
    """The tiny V4.1 checkpoint's window cache takes the decoder's layout: paged, it draws the compact
    cache's greedy tokens at the same format, full precision and quantized, its compressors' entries,
    buffers and index keys kept compact beside it."""
    loaded, plain, reference = source
    prompts = jnp.asarray(reference["input_ids"])[:, :7]

    def decoded(layout: KVCache) -> np.ndarray:
        return np.asarray(generate(plain.with_kv_cache(layout), loaded.variables, prompts, 9,
                                   key=jax.random.key(1), sampling=Sampling(temperature=0)).tokens)

    np.testing.assert_array_equal(decoded(KVCache(quantized=quantized, page_size=4)),
                                  decoded(KVCache(quantized=quantized)))


@pytest.mark.parametrize("quantized", [None, "int8", "float8_e4m3fn"])
def test_a_paged_drafting_cache_drafts_what_the_compact_cache_drafts(source, quantized):
    """The DSpark drafter's window cache takes the decoder's layout: after a prefill and a verified
    stretch recorded into it, a paged cache drafts the compact cache's logits at the same format, bit
    for bit, its context written as the stage's keys and values and read back in their basis."""
    from dew.sampling.text import decode_ops, prefill_state

    loaded, plain, reference = source
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])

    def drafted(layout: KVCache) -> np.ndarray:
        model = plain.with_kv_cache(layout)
        ops = decode_ops(model, loaded.variables, 0, 0)
        state, _ = prefill_state(model, loaded.variables, ModelInputs(ids[:, :prompt]), ops)
        keep = jnp.arange(3)[None, :] < jnp.asarray([3, 1])[:, None]
        state, _, context = ops.verify(state, ids[:, prompt:prompt + 3], keep)
        state = ops.record(state, context, keep)
        scored = []

        def choose(index, logits):
            scored.append(logits)
            return jnp.argmax(logits, axis=-1)

        ops.draft(state, ids[:, prompt + 3], choose)
        return np.asarray(jnp.stack(scored, 1))

    np.testing.assert_array_equal(drafted(KVCache(quantized=quantized, page_size=4)),
                                  drafted(KVCache(quantized=quantized)))


def test_consecutive_reindex_layers_publish_their_selections_under_scan_layers():
    """Two Reindex layers of one kind in a row stay two runs of one under
    scan_layers, so the Reuse layer after them attends the second one's
    selection, as it does unrolled, to within twice the fixture logits' fp32
    rounding; scanned together, their publications would stay inside the
    loop."""
    config = json.loads((TINY / "config.json").read_text())
    text = config["text_config"]
    config = {**config, "text_config": {**text, "index_source_layer_ids": [2, 4, 6, 7, 8, 10]}}
    fields = {**translate_config(config), "dtype": "float32", "attention_impl": "reference"}
    ids = jnp.asarray(np.load(TINY / "reference.npz")["input_ids"])
    unrolled, scanned = (models.build("causal_transformer", **{**fields, "scan_layers": scan})
                         for scan in (False, True))
    variables = unrolled.init(jax.random.key(0), ids)
    assert distance(scanned.apply(variables, ids), unrolled.apply(variables, ids)) <= 2 * FACTOR * rounding(
        "logits"
    )


def test_a_reindex_layer_reads_its_full_layers_publications_with_no_reuse_layer():
    """A stack whose CSA2 layers are a Full layer and the Reindex layers after it,
    with no Reuse layer and so no shared keys and values, still threads the store
    the Full layer publishes its entries in: the Reindex layers read them, unrolled
    and scanned alike, to within twice the fixture logits' fp32 rounding."""
    config = json.loads((TINY / "config.json").read_text())
    text = config["text_config"]
    ratios = text["compress_ratios"][:4] + [2] * 2 + text["compress_ratios"][text["num_hidden_layers"]:]
    config = {**config, "text_config": {
        **text, "num_hidden_layers": 6, "compress_ratios": ratios, "kv_source_layer_ids": [2],
        "index_source_layer_ids": [2, 3, 4, 5], "candidate_source_layer_id": 2,
        "engram_layer_ids": [1], "engram_num_embeddings": text["engram_num_embeddings"][:1],
        "dspark_target_layer_ids": [3, 4, 5]}}
    fields = {**translate_config(config), "dtype": "float32", "attention_impl": "reference"}
    assert not fields.get("kv_shared_layers")
    ids = jnp.asarray(np.load(TINY / "reference.npz")["input_ids"])
    unrolled, scanned = (models.build("causal_transformer", **{**fields, "scan_layers": scan})
                         for scan in (False, True))
    variables = unrolled.init(jax.random.key(0), ids)
    assert distance(scanned.apply(variables, ids), unrolled.apply(variables, ids)) <= 2 * FACTOR * rounding(
        "logits"
    )


def test_a_family_that_reads_its_media_bundle_plugs_in_through_its_entry(monkeypatch):
    """The wrapper dispatch and the routing of a bundle's tensors read the
    family's own entry, so a bundled family under another name loads with
    no branch of its own: its reader takes its config and its projector
    names route to the projector, the decoder's tensors unprefixed."""
    import dataclasses

    from dew.interop.families.deepseek_v41 import DEEPSEEK_V41
    from dew.interop.hf_decoders import wrapper_route

    read = []

    def reader(hf_config, used):
        read.append(hf_config["model_type"])
        return {**DEEPSEEK_V41.wrapper(dict(hf_config, model_type="deepseek_v41"), used),
                "model_type": "bundle_probe"}

    probe = dataclasses.replace(DEEPSEEK_V41, model_types=("bundle_probe",), wrapper=reader,
                                wrapper_projector_names=("probe_span",))
    monkeypatch.setitem(families(), "bundle_probe", probe)
    config = json.loads((RELEASED / "config.json").read_text())
    record = translate_wrapper_config({**config, "model_type": "bundle_probe"})
    assert read == ["bundle_probe"] and record["model_type"] == "bundle_probe"
    assert wrapper_route("probe_span", record) == ("projector", "probe_span")
    assert wrapper_route("image_newline", record) == ("language_model", "image_newline")
    assert wrapper_route("embed.weight", record) == ("language_model", "embed.weight")


def test_the_image_bias_and_the_dead_image_positions_decide_the_prefill(tmp_path):
    """Routing the image span by the text bias moves the prefill's logits
    further from the float64 truth than rounding can, and the image
    positions change the n-gram ids of the text after them, so the fixture
    exercises both."""
    reference = np.load(TINY / "vision.npz")
    loaded = Pretrained.load(vision_bundle(tmp_path), dtype="float32", attention_impl="reference")
    routers = flatten_dict(loaded.variables["moe"])
    routers.update({path: routers[(*path[:-1], "e_score_correction_bias")]
                    for path in routers if path[-1] == "media_bias"})
    _, moved, _ = vision_run(loaded.model, {**loaded.variables, "moe": unflatten_dict(routers)}, reference)
    assert distance(moved, TRUTH["vision_prompt_logits"]) > FACTOR * rounding("vision_prompt_logits")
    ids = jnp.asarray(reference["input_ids"])
    media = jnp.asarray(reference["token_types"] >= 0)
    language = {collection: tree["language_model"] for collection, tree in loaded.variables.items()}
    hashed = [np.asarray(loaded.model.language_model.apply(
        language, ids, jnp.ones(ids.shape, bool), jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape),
        decode=False, media=mask,
        method=lambda module, tokens, valid, positions, *, decode, media: module.engram_hashes(
            tokens, valid, positions, decode=decode, media=media))) for mask in (media, None)]
    after = int(np.flatnonzero(reference["token_types"][0] >= 0)[-1]) + 1
    assert np.any(hashed[0][:, after] != hashed[1][:, after])
