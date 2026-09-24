"""DeepSeek-V4.1-Flash against the release's own inference code.

deepseek-v41-tiny comes from tools/deepseek_v41_reference.py: the release's
inference/model.py at dba1be0a (DeepSeek-V4.1-Flash) run in fp32 over the
torch stand-ins for its kernels, whose quantizers match the tilelang kernels
bit for bit on bf16 input (`--check-kernels`). The architecture outputs are
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
import numpy as np
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict
from reference_error import FACTOR, assert_as_exact_as_the_reference, distance

from dew.interop import load_pretrained
from dew.interop.hf_decoders import _FAMILIES, _wrapper_sources, translate_config, translate_wrapper_config
from dew.nn.engram import Engram
from dew.nn.fake_quant import fake_quant_fp4, fake_quant_fp8
from dew.nn.inputs import ModelInputs
from dew.registry import models, with_precision
from dew.sampling import Sample, Sampling, Speculative, generate
from tools.deepseek_v41_numerics import (
    QUANTIZERS,
    cached_run,
    decided,
    forward,
    loss_and_gradient,
    scores,
    stepped,
    unquantized,
    updated,
    vision_bundle,
    vision_run,
)

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "deepseek-v41-tiny"
RELEASED = ROOT / "deepseek-v41-flash"
TRUTH = np.load(TINY / "reference_f64.npz")
# float64 rounds 2**-29 times as finely as float32, so the reference's
# arithmetic run in float64 lies that much closer to the truth than in fp32;
# a float64 twin and the truth are two such runs, each carrying that rounding.
TWIN = 2 * float(np.finfo(np.float64).eps / np.finfo(np.float32).eps)


@pytest.fixture(scope="module", autouse=True)
def fp32_matmuls():
    """The reference multiplies in fp32, where a GPU's default is TF32."""
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture(scope="module")
def source():
    loaded = load_pretrained(TINY, dtype="float32", attention_impl="reference")
    return loaded, unquantized(loaded.model), np.load(TINY / "reference.npz")


@pytest.fixture(scope="module")
def forwards(source):
    """The plain and the quantization-aware forward (`decided`), each with
    the quantizer inputs and top-k rows it met."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    return {prefix: decided(lambda model, variables: forward(model, variables, ids), model, loaded.variables)
            for prefix, model in (("", plain), ("qat_", loaded.model))}


def close(actual, reference, name: str):
    """`actual` as exact as the reference's `name`, both measured from the
    float64 truth, its argmax the reference's."""
    actual = np.asarray(actual)
    assert_as_exact_as_the_reference(actual, reference[name], TRUTH[name], name)
    np.testing.assert_array_equal(actual.argmax(-1), reference[name].argmax(-1), err_msg=name)


def twin_close(twin, name: str):
    """A float64 twin (`decided`) within FACTOR times the reference's own
    rounding, scaled to two float64 runs (TWIN), of the truth: the reference's
    arithmetic in float64, deciding every rounding and pick as the truth
    does, since one taken otherwise moves an output by fp32-sized steps."""
    apart = distance(twin, TRUTH[name])
    assert apart <= FACTOR * TWIN * rounding(name), f"{name}: the float64 twin is {apart:.3e} from the truth"


def loss_close(value, prefix: str):
    """The loss within reach of the logits' rule: it is the mean of each
    position's cross entropy, whose gradient in that position's logits has
    norm at most sqrt(2), so logits RMS `r` from the float64 truth move it at
    most sqrt(2 V) r, and the rule holds the logits to FACTOR times the
    reference's own `r`, in units of it that the loss's own dtype sets (TWIN
    for a float64 twin's loss); one spacing of the loss, in its own dtype,
    covers its own sum."""
    value = np.asarray(value)
    unit = TWIN if value.dtype == np.float64 else 1.0
    logits = np.load(TINY / "reference.npz")[f"{prefix}logits"]
    bound = np.sqrt(2 * logits.shape[-1]) * FACTOR * rounding(f"{prefix}logits") * unit + np.spacing(value)
    assert abs(float(value) - float(TRUTH[f"{prefix}loss"])) <= bound, f"{prefix}loss"


def rounding(name: str) -> float:
    """The reference's own RMS distance from the float64 truth for `name`:
    the fp32 rounding two runs of one computation may each carry FACTOR
    times of."""
    return distance(np.load(TINY / "vision.npz")[name] if name.startswith("vision_")
                    else np.load(TINY / "reference.npz")[name], TRUTH[name])


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


def test_every_quantizer_and_top_k_input_is_as_exact_as_the_reference(source, forwards):
    """Dew's quantization-aware and plain forwards hand every quantizer and
    top-k call its input as exactly as the reference does, both measured
    from the float64 truth, calling each as often and on as many blocks."""
    _, _, reference = source
    for prefix, (_, _, record) in forwards.items():
        assert set(record.blocks) == ({*QUANTIZERS} if prefix else set())
        for site, blocks in record.blocks.items():
            name = f"{prefix}{site}_in"
            ours = np.concatenate(blocks)
            assert ours.shape == reference[name].shape, name
            assert_as_exact_as_the_reference(ours, reference[name], TRUTH[name], name)
        assert_as_exact_as_the_reference(*scores(record, reference, TRUTH, prefix), f"{prefix}selection_rows")


def test_the_quantizers_pass_their_gradient_straight_through():
    x = jnp.linspace(-3.0, 3.0, 64).reshape(2, 32)
    for quant in (lambda v: fake_quant_fp8(v, 32), lambda v: fake_quant_fp4(v, 16, True)):
        np.testing.assert_array_equal(jax.grad(lambda v: jnp.sum(quant(v) * v))(x),
                                      quant(x) + x)


def test_a_value_far_past_the_clamp_reads_back_as_what_it_rounds_to():
    """A block led by 1e5 or 1e10 saturates its E4M3 scale at 448, so the lead
    stores the largest code, 6 * 448 = 2688, and the 3.0s store 0; compiled,
    that is the forward value in either dtype. `x + (rounded - x)` rounds
    the correction when x is far from what it rounds to: summed in bf16 it
    gives 2560 for 1e5, summed in fp32 3072 for 1e10."""
    quantize = jax.jit(lambda v: fake_quant_fp4(v, 16, True))
    for dtype in (jnp.bfloat16, jnp.float32):
        big = jnp.asarray([[lead] + [3.0] * 15 for lead in (1e5, 1e10)], dtype)
        expected = np.zeros(big.shape, np.float32)
        expected[:, 0] = 6 * 448
        np.testing.assert_array_equal(np.asarray(quantize(big), np.float32), expected,
                                      err_msg=jnp.dtype(dtype).name)


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
    model = _wrapper_model(config, record, models.build("causal_transformer", record["text"]), dtype="float32")
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))
    return config, record, model, shapes


def test_every_released_tensor_lands_on_one_leaf_of_the_released_tree(released):
    """The pinned weight index's 96085 tensors, their FP8/FP4 `.scale`
    partners aside, map onto the tree the released bundle builds, and
    together they cover it: the decoder with its DSpark stages and every
    router's image bias, the ViT, and the aligner with the image span's
    vectors."""
    from dew.nn.vision import deepseek_v41_vision_path, projector_weight_path

    config, record, model, shapes = released
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
    sources, _ = _wrapper_sources(names, lambda name: np.zeros(1), record)
    decoder = sources["language_model"]
    placeholders = {name: np.zeros((2, 4, 2) if name.endswith("wo_a.weight") else (1,)) for name in decoder}
    placeholders.update({name.removesuffix("wo_a.weight") + "wq_b.weight": np.zeros((8, 1))
                         for name in decoder if name.endswith("wo_a.weight")})
    paths = [family.weight_path(name, record["text"]) for name in family.prepare_weights(placeholders)]
    assert None not in paths
    bound = [(path[0], "language_model", *path[1:]) for path in paths if path is not None]
    bound += [("params", "tower", *deepseek_v41_vision_path(name)) for name in sources["tower"]]
    bound += [("params", "projector", *projector_weight_path("deepseek_v41", name))
              for name in sources["projector"]]
    assert len(set(bound)) == len(bound)
    # One tensor per expert stacks into one leaf per projection.
    bound = {tuple(part for index, part in enumerate(path)
                   if not (index and path[index - 1] == 'experts' and part.isdigit()))
             for path in bound}
    tree = set(flatten_dict(dict(shapes)))
    tree.discard(("constants", "language_model", "engram_hashes", "token_map"))
    assert bound == tree
    language = model.language_model
    assert language.num_layers == 40 and language.dspark.stages == 3 and language.mixture.experts == 384


def test_every_matrix_of_the_released_decoder_shards_by_a_declared_rule(released):
    """No decoder weight of two or more axes is left to the shape heuristic
    unasked: the engram tables alone hold 384M rows a layer, which shard as
    a vocabulary's do. The ViT's, like every tower's, are the heuristic's."""
    from dew.nn.sharding import declared_axes, is_heuristic

    *_, shapes = released
    leaves = [(jax.tree_util.keystr(path), path, leaf)
              for path, leaf in jax.tree_util.tree_flatten_with_path(shapes)[0]]
    uncovered = [name for name, path, leaf in leaves if "['language_model']" in name and leaf.ndim >= 2
                 and declared_axes(path, leaf.ndim) is None and not is_heuristic(path)]
    assert uncovered == []
    engram = [declared_axes(path, leaf.ndim) for name, path, leaf in leaves
              if name.endswith("['engram']['embed']['embedding']")]
    assert engram == [("vocab", None)] * 2


def test_the_forward_matches_the_reference(source, forwards):
    given, twin, _ = forwards[""]
    close(given, source[2], "logits")
    twin_close(twin, "logits")


def test_the_quantization_aware_forward_matches_the_reference(source, forwards):
    given, twin, _ = forwards["qat_"]
    close(given, source[2], "qat_logits")
    twin_close(twin, "qat_logits")


def test_the_quantization_aware_loss_and_gradient_match_the_reference(source):
    """The loss through every quantizer, and one SGD step along its gradient,
    which passes each quantizer straight through, read back through the
    forward without quantization, so the step compares the gradient alone."""
    loaded, _, reference = source
    ids = jnp.asarray(reference["input_ids"])
    (value, logits), (twin_value, twin_logits), _ = decided(
        lambda model, variables: updated(model, variables, ids, reference["learning_rate"]),
        loaded.model, loaded.variables)
    loss_close(value, "qat_")
    loss_close(twin_value, "qat_")
    close(logits, reference, "qat_updated_logits")
    twin_close(twin_logits, "qat_updated_logits")


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


def test_the_update_exports_and_decodes_as_the_reference(source, tmp_path):
    """Loss, one SGD step with every gradient (the indexer's are zero, as the
    reference's selection passes none), the source-layout export read back
    bit for bit, and greedy decoding after the fixture's prompt."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    (value, gradient), (twin_value, _), _ = decided(
        lambda model, variables: loss_and_gradient(model, variables, ids), plain, loaded.variables)
    loss_close(value, "")
    loss_close(twin_value, "")
    indexer = [np.max(np.abs(leaf)) for path, leaf in flatten_dict(gradient, sep=".").items() if ".indexer." in f".{path}."]
    assert indexer and max(indexer) == 0
    variables = stepped(loaded.variables, gradient, reference["learning_rate"])
    loaded.save(tmp_path, variables=variables)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    held, again = flatten_dict(variables, sep="."), flatten_dict(dict(restored.variables), sep=".")
    assert held.keys() == again.keys()
    for name, leaf in again.items():
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(held[name]), err_msg=name)
    # The float64 twin of this forward starts from the fp32 step, which the
    # truth does not take, so only its decisions count here.
    close(decided(lambda model, variables: forward(model, variables, ids), unquantized(restored.model),
                  restored.variables)[0], reference, "updated_logits")
    prompt = int(reference["decode_prompt"])
    generated = generate(plain, loaded.variables, ModelInputs(ids[:, :prompt]),
                         reference["generated"].shape[1], key=jax.random.key(1),
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens)[:, -reference["generated"].shape[1]:],
                                  reference["generated"])


def cached_matches(model, variables, reference, prefix: str):
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])
    (prompt_logits, steps, draft_ids, draft_logits, confidence), twin, _ = decided(
        lambda model, variables: cached_run(model, variables, ids, prompt), model, variables)
    close(prompt_logits, reference, f"{prefix}prompt_logits")
    close(steps, reference, f"{prefix}decode_logits")
    close(draft_logits, reference, f"{prefix}draft_logits")
    name = f"{prefix}draft_confidence"
    assert_as_exact_as_the_reference(confidence, reference[name], TRUTH[name], name)
    np.testing.assert_array_equal(draft_ids, reference[f"{prefix}draft_ids"])
    wide_prompt, wide_steps, _, wide_draft, wide_confidence = twin
    for name, wide in (("prompt_logits", wide_prompt), ("decode_logits", wide_steps),
                       ("draft_logits", wide_draft), ("draft_confidence", wide_confidence)):
        twin_close(wide, prefix + name)


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
    from dew.sampling.text import _operations, _prefill

    loaded, plain, reference = source
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])
    ops = _operations(plain, loaded.variables, 0, 0)
    assert ops.record is not None and ops.draft is not None and ops.verify is not None
    state, _ = _prefill(plain, loaded.variables, ModelInputs(ids[:, :prompt]), ops)
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


def test_consecutive_reindex_layers_publish_their_selections_under_scan_layers():
    """Two Reindex layers of one kind in a row stay two runs of one under
    scan_layers, so the Reuse layer after them attends the second one's
    selection, as it does unrolled, to within twice the fixture logits' fp32
    rounding; scanned together, their publications would stay inside the
    loop."""
    config = json.loads((TINY / "config.json").read_text())
    text = config["text_config"]
    config = {**config, "text_config": {**text, "index_source_layer_ids": [2, 4, 6, 7, 8, 10]}}
    fields = with_precision("causal_transformer", translate_config(config), dtype="float32",
                            attention_impl="reference")
    ids = jnp.asarray(np.load(TINY / "reference.npz")["input_ids"])
    unrolled, scanned = (models.build("causal_transformer", **{**fields, "scan_layers": scan})
                         for scan in (False, True))
    variables = unrolled.init(jax.random.key(0), ids)
    assert distance(scanned.apply(variables, ids), unrolled.apply(variables, ids)) <= 2 * FACTOR * rounding("logits")


def test_a_config_without_swiglu_limit_clamps_nothing():
    """The release's ModelArgs default is 0.0, no clamp (v41 model.py:70)."""
    config = json.loads((TINY / "config.json").read_text())
    text = {key: value for key, value in config["text_config"].items() if key != "swiglu_limit"}
    assert translate_config({**config, "text_config": text})["swiglu_limit"] is None


def test_the_vision_half_matches_the_reference(tmp_path):
    """The bundle's vision half: the ViT's patches, blocks and 2D rotary, the
    aligner over a patch grid it pads to whole squares, the span's learned
    vectors, every router's image bias and the engram's dead image positions,
    through a prefill with the image and text steps after it, against the
    release's Transformer.forward with the quantizers off."""
    reference = np.load(TINY / "vision.npz")
    loaded = load_pretrained(vision_bundle(tmp_path), dtype="float32", attention_impl="reference")
    (span, prompt_logits, steps), twin, _ = decided(
        lambda model, variables: vision_run(model, variables, reference), loaded.model, loaded.variables)
    assert_as_exact_as_the_reference(span, reference["vision_span"], TRUTH["vision_span"], "vision_span")
    close(prompt_logits, reference, "vision_prompt_logits")
    close(steps, reference, "vision_decode_logits")
    for name, wide in zip(("vision_span", "vision_prompt_logits", "vision_decode_logits"), twin, strict=True):
        twin_close(wide, name)


def test_the_image_bias_and_the_dead_image_positions_decide_the_prefill(tmp_path):
    """Routing the image span by the text bias moves the prefill's logits
    further from the float64 truth than rounding can, and the image
    positions change the n-gram ids of the text after them, so the fixture
    exercises both."""
    reference = np.load(TINY / "vision.npz")
    loaded = load_pretrained(vision_bundle(tmp_path), dtype="float32", attention_impl="reference")
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
        False, mask, method=lambda module, *inputs: module.engram_hashes(*inputs))) for mask in (media, None)]
    after = int(np.flatnonzero(reference["token_types"][0] >= 0)[-1]) + 1
    assert np.any(hashed[0][:, after] != hashed[1][:, after])
