"""DeepSeek-V4.1-Flash against the release's own inference code.

deepseek-v41-tiny comes from tools/deepseek_v41_reference.py, the release's inference/model.py at
dba1be0a (DeepSeek-V4.1-Flash). Its model-level claims are held to the release as released (row 354,
`RELEASE`): nothing of its code replaced, over the weights it holds, in bf16 over its own tilelang kernels
and activation quantizers on an H100, recorded over check 1's seeds with its float64 truth (the same code
over the stand-ins for its kernels, widened), over drawn rows whose every selection both float64 runs clear
`selection_margins.MARGIN` by: every position's logits, a cached walk's past its prompt, DSpark's drafts
after a step that closes every compressed group, and an image prompt's. Dew's variants of a row (an interior
gap, shifted positions, left padding, a shifted cached walk) are held to them, and a sensitivity (the
candidate pool, the image bias) fails them. kernels.npz holds the release's tilelang kernels' outputs
(`--kernels`, on a GPU), which Dew's quantizers and the stand-ins are held to bit for bit on bf16 input;
reference.npz keeps the fixture's prompt rows and every quantizer call's input blocks, which Dew's
quantizers are held to the stand-ins on.
"""

import dataclasses
import functools
import json
from collections.abc import Mapping
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict
from model_support import decode_through
from quantized_basis import dequantized_in_own_basis
from reference_error import GateFailure, Inconclusive, as_given, assert_as_exact, declared_seeds

from dew.batch import coordinates
from dew.interop import Pretrained
from dew.interop.hf_decoders import families, translate_config, translate_wrapper_config
from dew.nn.deepseek_v4 import DRAFT_CONTEXT, DeepseekV4Attention, DSparkAttention
from dew.nn.engram import Engram
from dew.nn.fake_quant import E2M1, E2M1_MIDPOINTS, fake_quant_fp4, fake_quant_fp8
from dew.nn.inputs import MixerCall, ModelInputs
from dew.nn.kv_cache import KVCache
from dew.nn.precision import at_least_fp32
from dew.nn.rope import LongRopeScaling, RopeScaling, rotary_freqs
from dew.nn.sharding import down_projection
from dew.nn.visibility import visible
from dew.registry import models
from dew.sampling import Sample, Sampling, Speculative, generate
from tools.deepseek_v41_numerics import QUANTIZERS, as_released, unquantized, vision_bundle

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "deepseek-v41-tiny"
RELEASED = ROOT / "deepseek-v41-flash"
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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_the_quantizers_pass_their_gradient_straight_through_and_round_their_value(dtype):
    """Each quantizer's derivative is the identity (straight through), the estimator a quantization-aware
    run trains through, which the release, an inference program, states none of (row 354); and its value
    is what it rounds to however far a block's lead sits past its scale's clamp, compiled in either dtype:
    a block led by 1e5 or 1e10 saturates its E4M3 scale at 448, so the lead stores the largest code,
    6 * 448, and the 3.0s store 0, where `x + (rounded - x)` rounds its correction far from x."""
    x = jnp.linspace(-3.0, 3.0, 64, dtype=dtype).reshape(2, 32)
    for quant in (lambda v: fake_quant_fp8(v, 32), lambda v: fake_quant_fp4(v, 16, e4m3_scale=True)):
        np.testing.assert_array_equal(jax.grad(lambda v, quant=quant: jnp.sum(quant(v) * v))(x),
                                      quant(x) + x)
    big = jnp.asarray([[lead] + [3.0] * 15 for lead in (1e5, 1e10)], dtype)
    expected = np.zeros(big.shape, np.float32)
    expected[:, 0] = 6 * 448
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda v: fake_quant_fp4(v, 16, e4m3_scale=True))(big),
                                             np.float32), expected)


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


def test_a_packed_documents_engram_lookback_stops_at_its_start_whatever_its_positions():
    """Each token hashes with its own document's predecessors alone (`EngramHashes`, review of ledger
    1122): a packed row's second document hashes as it does alone, though its logical positions run on
    from 10, which a lookback bounded by positions read as a predecessor of the first document's."""
    from dew.nn.engram import EngramHashes

    spec = Engram(layer_ids=(0,), num_embeddings=(88,), max_ngram_size=3, vocab_size=16, n_heads=2,
                  head_dim=4, compressed_vocab_size=16)
    hashes = EngramHashes(spec, 16)
    packed = jnp.asarray([[3, 5, 7, 9, 11]], jnp.int32)
    laid = coordinates(packed.shape, segments=jnp.asarray([[1, 1, 2, 2, 2]]),
                       positions=jnp.asarray([[0, 1, 10, 11, 12]]), max_segments=2)
    variables = hashes.init(jax.random.key(0), packed, laid, decode=False)
    together = hashes.apply(variables, packed, laid, decode=False)
    alone = hashes.apply(variables, packed[:, 2:], None, decode=False)
    np.testing.assert_array_equal(together[:, 2:], alone)


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


def test_left_padding_changes_no_row(source):
    """Rows of different lengths in one batch, the shorter padded on the
    left: engram's look-back packs each row's valid tokens (its argsort
    path), and CSA2's windows, entries and index keys follow each row's own
    positions. Each row decodes what it decodes alone, the full one and the
    padded one."""
    loaded, plain, reference = source
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])
    steps, pad = ids.shape[1] - prompt, 3

    def greedy(inputs):
        return np.asarray(generate(plain, loaded.variables, inputs, steps, key=jax.random.key(1),
                                   sampling=Sampling(temperature=0)).tokens)[:, -steps:]

    tokens = jnp.stack([ids[0, :prompt], jnp.pad(ids[1, :prompt - pad], (pad, 0))])
    valid = jnp.arange(prompt)[None] >= jnp.asarray([[0], [pad]])
    batched = greedy(ModelInputs(tokens, {"attention_mask": valid}))
    np.testing.assert_array_equal(batched[0], greedy(ModelInputs(ids[:1, :prompt]))[0])
    np.testing.assert_array_equal(batched[1], greedy(ModelInputs(ids[1:, :prompt - pad]))[0])


RELEASE = TINY / "release.npz"
"""The release as released (row 354): its inference/model.py at the pinned revision, nothing of it
replaced, over the weights it holds (release.safetensors), in bf16 over its own tilelang kernels and
activation quantizers on an H100, its logits at every position (each prefix's prefill, the paper's
arithmetic: a cached step of the release that closes no ratio-2 group reads another layer's index keys,
which Dew does not reproduce), from position zero and from a walk's prompt on, and an image prompt's,
and their float64 truth (the same code over the stand-ins kernels.npz holds to the kernels, widened),
over drawn rows whose every selection both float64 runs clear `selection_margins.MARGIN` by, summarized
over check 1's seeds, beside the rows and prompts each seed drew (release_rows.npz;
tools/deepseek_v41_reference.py --release)."""


@pytest.fixture(scope="module")
def as_run(tmp_path_factory):
    """The fixture as the release runs it: bf16, over the weights the release holds (`as_released`), its
    activation quantizers on (the family's `kv_qat`, as the release rounds every forward)."""
    return Pretrained.load(as_released(tmp_path_factory.mktemp("released")), dtype="bfloat16",
                           param_dtype="auto", attention_impl="reference")


@pytest.fixture(scope="module")
def recorded_rows(as_run):
    """The rows the recording drew (release_rows.npz), each of whose selections, the release's float64 run
    cleared when it was recorded, Dew's float64 forward clears `selection_margins.MARGIN` by too: so
    Dew's truth and the release's select alike, whichever positions a variant of the forward computes
    them at (a gap, shifted positions, a cached walk), each of whose selections is one of the forward's.
    The bf16 runs select as their own rounding takes them, the release's and Dew's alike, which check 1
    weighs."""
    import selection_margins
    from model_support import float64_twin

    with np.load(TINY / "release_rows.npz") as stored:
        rows = stored["rows"]
    selecting = jax.jit(functools.partial(selection_margins.capturing, float64_twin(as_run.model)))
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                            if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf,
                            as_run.variables)
    for index, row in enumerate(rows):
        with jax.enable_x64(new_val=True):
            _, captured = selecting(wide, jnp.asarray(row))
        smallest = selection_margins.margin(as_run.model_config, captured, f"recorded row {index}")
        assert smallest > selection_margins.MARGIN, (
            f"Dew's selections over recorded row {index} clear {smallest:.3g}, not MARGIN")
    return rows


def held_to_the_release(as_run, rows, site: str, start: int = 0):
    """A recorded site's draw of `rows`, the release's evidence and truth over its declared seeds, Dew's
    forward in float64 past `start` its truth, held to the release's (`recorded_site`), and the count."""
    from model_support import float64_twin, recorded_site

    wide = jax.jit(float64_twin(as_run.model).apply)

    def twin(inputs):
        with jax.enable_x64(new_val=True):
            held = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                                if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf,
                                as_run.variables)
            return {"logits": wide(held, jnp.asarray(inputs["input_ids"]))[:, start:]}

    return recorded_site(RELEASE, site, lambda seed: {"input_ids": rows[int(seed)]}, twin)


def test_an_interior_gap_leaves_every_window_on_its_rows_real_tokens(as_run, recorded_rows):
    """Uncached, three padding slots inside each row leave each real token's sliding window counting
    its row's real tokens (`counted_places`), as a cache's slots count them, and its compressed windows
    and n-grams on them: over drawn rows the release selects over as it would without the gap, the real
    tokens' logits are on par with the release's of the rows without it from its float64 run's (check 1,
    the recording's), where a window counted by index let the gap crowd out the keys before it (audit
    NN-02)."""
    at, gap = 5, 3
    draw, reference, truth, seeds = held_to_the_release(as_run, recorded_rows,
                                                         "deepseek v41 release rows")

    @jax.jit
    def gapped(ids):
        filler = jnp.zeros((ids.shape[0], gap), ids.dtype)
        tokens = jnp.concatenate([ids[:, :at], filler, ids[:, at:]], axis=1)
        slots = jnp.arange(tokens.shape[1])
        real = (slots < at) | (slots >= at + gap)
        logits = as_run.model.apply(as_run.variables, tokens,
                                      attention_mask=jnp.broadcast_to(real, tokens.shape))
        return jnp.concatenate([logits[:, :at], logits[:, at + gap:]], axis=1)

    assert_as_exact(as_given(lambda index, inputs: {"logits": gapped(jnp.asarray(inputs["input_ids"]))}),
                    reference, draw, "an interior gap", seeds=seeds, truth_fn=truth)


SHIFT = 8
"""How far the shifted rows' given positions sit past their indices: a multiple of every compression
rate, so a query that read entries by its given place would see SHIFT // rate windows of later tokens."""


def test_shifted_positions_compute_the_unshifted_rows_logits(as_run, recorded_rows):
    """Uncached, positions given SHIFT past the tokens' indices: every rotation is relative, each
    compressed entry is visible by its window's tokens' ranks and rotates at its window's first given
    place, where its queries rotate (`window_origins`), so over drawn rows the logits are on par with
    the release's of the rows at their indices from its float64 run's (check 1, the recording's);
    entries visible by the given places showed each query windows of later tokens, and entries rotated
    at zero-based origins sat SHIFT away from their queries (audit NN-03)."""
    draw, reference, truth, seeds = held_to_the_release(as_run, recorded_rows,
                                                         "deepseek v41 release rows")

    @jax.jit
    def shifted(ids):
        return as_run.model.apply(as_run.variables, ids,
                                    positions=jnp.broadcast_to(jnp.arange(ids.shape[1]) + SHIFT, ids.shape))

    assert_as_exact(as_given(lambda index, inputs: {"logits": shifted(jnp.asarray(inputs["input_ids"]))}),
                    reference, draw, "shifted positions", seeds=seeds, truth_fn=truth)


def test_a_shifted_prompt_decodes_the_unshifted_rows_logits(as_run, recorded_rows):
    """Cached, a prompt at positions SHIFT past its indices, then the rest of each row one token at a
    time: each compressed entry rotates at its window's first given place, which the cache carries
    across the calls a window spans (`append_windows`), so over drawn rows every step's logits are on
    par with the release's of the rows at their indices from its float64 run's (check 1, the
    recording's), where entries rotated at their slots sat SHIFT away from their queries (audit NN-03).
    The walk is one compiled program, its steps scanned."""
    from dew.sampling.text import decode_ops, prefill_state

    prompt = int(np.load(TINY / "reference.npz")["decode_prompt"])
    draw, recorded, truth, seeds = held_to_the_release(as_run, recorded_rows, "deepseek v41 release walk",
                                                       prompt - 1)

    @jax.jit
    def walked(ids):
        ops = decode_ops(as_run.model, as_run.variables, 0, None)
        positions = jnp.broadcast_to(jnp.arange(prompt) + SHIFT, (ids.shape[0], prompt))
        prompted = ModelInputs(ids[:, :prompt], {"positions": positions})
        state, _ = prefill_state(as_run.model, as_run.variables, prompted, ops)

        def step(state, column):
            state = ops.advance(state, column, jnp.ones(ids.shape[0], bool))
            return state, state.logits

        _, steps = jax.lax.scan(step, state, ids[:, prompt:].T)
        return jnp.concatenate([state.logits[:, None], jnp.moveaxis(steps, 0, 1)], axis=1)

    assert_as_exact(as_given(lambda index, inputs: {"logits": walked(jnp.asarray(inputs["input_ids"]))}),
                    recorded, draw, "a shifted cached walk", seeds=seeds, truth_fn=truth)


def test_the_candidate_pool_decides_the_logits(as_run, recorded_rows):
    """Dropping the pool's restriction on the Reindex layers fails the release's check (check 1, a decided
    failure, not Inconclusive), so the fixture exercises the hierarchical indexer."""
    model = as_run.model
    kinds = dict(model.kinds)
    kinds["csa2_ratio_1_reindex"] = dataclasses.replace(
        kinds["csa2_ratio_1_reindex"], mixer=dataclasses.replace(
            kinds["csa2_ratio_1_reindex"].mixer, candidates=None, candidate_blocks=None,
            candidate_block_size=None))
    unpooled = jax.jit(model.clone(kinds=flax.core.freeze(kinds)).apply)
    draw, reference, truth, seeds = held_to_the_release(as_run, recorded_rows, "deepseek v41 release rows")
    with pytest.raises(GateFailure) as failed:
        assert_as_exact(as_given(lambda index, inputs: {"logits": unpooled(as_run.variables,
                                                                           jnp.asarray(inputs["input_ids"]))}),
                        reference, draw, "without the candidate pool", seeds=seeds, truth_fn=truth)
    assert not isinstance(failed.value, Inconclusive)


@pytest.mark.parametrize("given", ["attention_mask", "coordinates"])
def test_an_uncached_left_padded_row_pools_its_windows_over_its_own_tokens(as_run, recorded_rows, given):
    """Uncached, rows padded on the left by three tokens, a count no compression rate divides, pool their
    compressed windows (CSA2's entries and index keys, HCA's) and hash their n-grams (Engram) over their
    real tokens at their ranks, as a cache appends them, whether the padding comes as a raw mask or as the
    tokens' coordinates alone: over drawn rows, the real tokens' logits are on par with the release's of
    the rows unpadded from its float64 run's (check 1, the recording's)."""
    pad = 3
    draw, reference, truth, seeds = held_to_the_release(as_run, recorded_rows, "deepseek v41 release rows")

    @jax.jit
    def padded(ids):
        valid = jnp.broadcast_to(jnp.arange(ids.shape[1] + pad) >= pad, (ids.shape[0], ids.shape[1] + pad))
        padding = {"attention_mask": valid} if given == "attention_mask" else {
            "coordinates": coordinates(valid=valid)}
        return as_run.model.apply(as_run.variables, jnp.pad(ids, ((0, 0), (pad, 0))), **padding)[:, pad:]

    assert_as_exact(as_given(lambda index, inputs: {"logits": padded(jnp.asarray(inputs["input_ids"]))}),
                    reference, draw, f"left padded by {given}", seeds=seeds, truth_fn=truth)


DRAFTED = (7, 9, 11, 13)
"""The prompts after whose cached step the release drafted (the tool's `DRAFTED`)."""


def test_dsparks_draft_is_the_releases(as_run, recorded_rows):
    """DSpark's draft from each row's next token after a history of each length of DRAFTED plus one, its
    block's logits (the Markov bias included) and confidence, uncached, on par with the release's after
    that history's prefill and one cached step (each closing every ratio-2 group, so the release computes
    the paper's) from its float64 run's (check 1, the recording's: row 354), over rows whose trunk's,
    drafter's and drafts' choices both float64 runs clear `selection_margins.MARGIN` by."""
    from model_support import float64_twin, recorded_site

    def drafted(model, variables, ids):
        logits, confidence = [], []
        for prompt in DRAFTED:
            _, trunk = model.apply(variables, ids[:, :prompt + 1], mutable=["prediction_inputs"])
            whole = model.apply(variables, trunk["prediction_inputs"], method="draft_context")
            _, block, sure = model.apply(variables, whole, ids[:, prompt + 1],
                                         method=lambda module, whole, token: module.draft(whole, token,
                                                                                          decode=False))
            logits.append(block)
            confidence.append(sure)
        return {"logits": jnp.stack(logits, 1), "confidence": jnp.stack(confidence, 1)}

    run = jax.jit(functools.partial(drafted, as_run.model))
    wide = jax.jit(functools.partial(drafted, float64_twin(as_run.model)))

    def twin(inputs):
        with jax.enable_x64(new_val=True):
            held = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                                if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf,
                                as_run.variables)
            return wide(held, jnp.asarray(inputs["input_ids"]))

    draw, reference, truth, seeds = recorded_site(RELEASE, "deepseek v41 release draft",
                                                  lambda seed: {"input_ids": recorded_rows[int(seed)]}, twin)
    assert_as_exact(as_given(lambda index, inputs: run(as_run.variables, jnp.asarray(inputs["input_ids"]))),
                    reference, draw, "dspark's draft", seeds=seeds, truth_fn=truth)


def held_to_the_release_images(model, variables, label: str, arm=None) -> None:
    """`model` over `variables` (the vision bundle's), or over `arm`'s where given, on par with the
    release's image prompts (check 1, the recording's 'deepseek v41 release vision'): every position's
    logits from the prompt's last on, one forward with the image, the truth its float64 twin over
    `variables`, held to the recording's sketch."""
    from model_support import float64_twin
    from reference_error import indexed_recording, recording

    with np.load(TINY / "release_rows.npz") as stored:
        prompts = {name: stored[f"vision_{name}"] for name in ("pixels", "input_ids", "token_types")}
    # Every prompt's image span the same, GRID's at IMAGE_START_AT: the logits from the token after it.
    prompt = int(np.flatnonzero(prompts["token_types"][0, 0] >= 0)[-1]) + 2

    def logits(model, variables, inputs):
        media = jnp.asarray(inputs["token_types"]) >= 0
        return {"logits": model.apply(
            variables, jnp.asarray(inputs["input_ids"]),
            image_indices=jnp.where(media, jnp.cumsum(media, -1) - 1, -1),
            conditioning={"pixel_values": jnp.asarray(inputs["pixels"])[None, None]})[:, prompt - 1:]}

    wide = float64_twin(model)

    def truth(inputs):
        with jax.enable_x64(new_val=True):
            held = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                                if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf, variables)
            return logits(wide, held, inputs)

    site = "deepseek v41 release vision"
    seeds = declared_seeds(RELEASE)[site]
    draw, reference, truth = indexed_recording(
        recording(RELEASE), site, seeds, truth_fn=truth,
        inputs_of=lambda seed: {name: value[int(seed)] for name, value in prompts.items()})
    held = variables if arm is None else arm
    assert_as_exact(as_given(lambda index, inputs: logits(model, held, inputs)), reference, draw, label,
                    seeds=seeds, truth_fn=truth)


def test_an_image_prompt_computes_the_releases_logits(tmp_path):
    """The vision half as the release runs it (its ViT, aligner, span vectors and every router's image
    bias, bf16 with its activation quantizers): over drawn image prompts, every position's logits from
    the prompt's last on, one forward with the image, on par with the release's from its float64 run's
    (check 1, the recording's: each a prefix's prefill with the image, row 354)."""
    loaded = Pretrained.load(as_released(tmp_path, vision=True), dtype="bfloat16", param_dtype="auto",
                             attention_impl="reference")
    held_to_the_release_images(loaded.model, loaded.variables, "an image prompt")


def test_a_llama3_ramp_reaches_every_layer_and_longrope_is_refused(source):
    """V4's layers take the model's one rope scaling (`MixerContext.rope_scaling`), each kind's own
    in its place where the kind gives one (`kind_of`), as the compressed kinds give their YaRN ramp: a
    Llama 3.1 ramp makes the decoder compute otherwise than the unscaled one, where the layers
    forwarded YaRN alone, rotated at the plain frequencies and left the logits bit for bit the same.
    LongRoPE, whose factors a call's reach picks and which the compressed entries do not see, is
    refused by name (audit NN-04)."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    ramp = RopeScaling(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                       original_max_position_embeddings=4)
    scaled, unscaled = (plain.clone(rope_scaling=scaling).apply(loaded.variables, ids)
                        for scaling in (ramp, None))
    assert not np.array_equal(scaled, unscaled)
    longrope = LongRopeScaling(short_factor=(1.0,) * 4, long_factor=(2.0,) * 4,
                               original_max_position_embeddings=4)
    with pytest.raises(ValueError):
        plain.clone(rope_scaling=longrope).apply(loaded.variables, ids)


def test_a_prefilled_decode_cache_holds_no_payload_twice(source):
    """After a prompt's prefill for drafting, the target's and the drafter's decode caches hold no
    payload twice: V4's window head, which serves as its keys and its values, and the drafter's context
    are each kept once (`KVStore`'s `keys_as_values`), where both were allocated and written and every
    window layer held each of its tokens' heads twice (ledger 1921)."""
    from dew.sampling.text import decode_ops, prefill_state

    loaded, plain, reference = source
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])
    ops = decode_ops(plain, loaded.variables, 0, 1)
    state, _ = prefill_state(plain, loaded.variables, ModelInputs(ids[:, :prompt]), ops)

    def twice(node):
        if not isinstance(node, Mapping):
            return []
        payloads = [leaf for leaf in node.values() if not isinstance(leaf, Mapping) and np.ndim(leaf) >= 3]
        found = [index for index, leaf in enumerate(payloads)
                 if any(np.array_equal(leaf, other) and np.any(leaf) for other in payloads[:index])]
        return found + [held for child in node.values() for held in twice(child)]

    assert not twice(jax.device_get(state.cache))


PADDED_DRAFT_SEEDS = 8
"""Check 1's declared seeds for a padded context's draft against the unpadded one's, sized by `seeds_for`
paired on armada."""


def test_an_uncached_draft_reads_a_padded_context_at_its_real_positions(as_run, recorded_rows):
    """Uncached, DSpark drafts after a context whose padding `valid` marks, after it in one row and
    before it in the other, as after the context alone: its real positions keep their ranks, the
    padding is no key, and the block follows the last real position. Over drawn rows, the padded draft's
    logits are on par with the unpadded one's from its float64 run's (check 1). Padding a layout marks as
    document 0 instead (`real_slots`) drafts as its validity does, bit for bit, and a context of two
    documents a row is refused (review of ledger 1122)."""
    from model_support import float64_twin

    model, variables = as_run.model, as_run.variables
    prompt, pad = DRAFTED[0], 3
    places = jnp.arange(prompt + pad)
    valid = jnp.stack([places < prompt, places >= pad])

    def context(model, variables, ids):
        _, sown = model.apply(variables, ids[:, :prompt], mutable=["prediction_inputs"])
        return model.apply(variables, sown["prediction_inputs"], method="draft_context")

    def padded(context):
        return jnp.stack([jnp.pad(context[0], ((0, pad), (0, 0))), jnp.pad(context[1], ((pad, 0), (0, 0)))])

    def alone(model, variables, ids):
        return {"logits": model.apply(variables, context(model, variables, ids), ids[:, prompt], decode=False,
                                      method="draft")[1]}

    by_validity = coordinates(valid.shape, valid=valid)

    @jax.jit
    def drafted(ids, laid):
        return model.apply(variables, padded(context(model, variables, ids)), ids[:, prompt], decode=False,
                           coordinates=laid, method="draft")[1]

    wide = jax.jit(functools.partial(alone, float64_twin(model)))

    @as_given
    def truth(ids):
        with jax.enable_x64(new_val=True):
            held = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                                if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf, variables)
            return wide(held, jnp.asarray(ids))

    import selection_margins
    from interop_support import token_draw

    twin = float64_twin(model)

    @jax.jit
    def chosen(variables, ids):
        """The float64 unpadded draft's block and every selection its trunk and its drafter make."""
        _, trunk = selection_margins.capturing(twin, variables, ids[:, :prompt],
                                               mutable=["prediction_inputs"])
        whole = twin.apply(variables, trunk["prediction_inputs"], method="draft_context")
        (_, block, _), drafter = selection_margins.capturing(
            twin, variables, whole, ids[:, prompt],
            method=lambda module, whole, token: module.draft(whole, token, decode=False))
        return block, (trunk, drafter)

    def clears(inputs):
        """The float64 draft's choices, its block's greedy tokens and every selection of its trunk and
        drafter, clear of MARGIN, so both arms choose alike."""
        with jax.enable_x64(new_val=True):
            held = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                                if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf, variables)
            block, captured = chosen(held, jnp.asarray(inputs[0]))
        smallest = float(np.min(selection_margins.top_margin(np.asarray(block)[..., :-1, :], 1)))
        for selections in captured:
            smallest = min(smallest, selection_margins.margin(as_run.model_config, selections,
                                                              "a padded context's draft"))
        return smallest > selection_margins.MARGIN

    draw = selection_margins.clearing(token_draw(0x1122, model.vocab_size, recorded_rows.shape[1:]), clears)

    assert_as_exact(as_given(lambda ids: {"logits": drafted(jnp.asarray(ids), by_validity)}),
                    jax.jit(functools.partial(alone, model, variables)), draw, "a padded context's draft",
                    seeds=PADDED_DRAFT_SEEDS, truth_fn=truth)
    ids = jnp.asarray(recorded_rows[0])
    documents = coordinates(valid.shape, segments=valid.astype(jnp.int32), max_segments=1)
    np.testing.assert_array_equal(drafted(ids, documents), drafted(ids, by_validity))
    two = coordinates(valid.shape, segments=1 + jnp.broadcast_to(places >= prompt // 2, valid.shape))
    with pytest.raises(ValueError):
        drafted(ids, two)


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


DRAFT_SEEDS = 8
"""The cached draft's declared check-1 seeds against the uncached one, sized by `seeds_for` on armada."""


def test_the_decode_ops_draft_what_dspark_drafts_over_the_history(source):
    """After a prefill and a verified stretch whose rows keep different
    counts, the block the decode ops draft is the one the drafter drafts
    uncached over each row's real history: the prompt's and the stretch's
    context reach its windows at their own positions. Over drawn rows whose
    drafted tokens clear their greedy choice, and whose trunk's and drafter's
    selections (its indexer top-k, candidate pool and routing) clear theirs,
    by `selection_margins.MARGIN` in the float64 run, so every run chooses
    alike, the cached block's logits are on par with the uncached ones from
    the float64 drafter's (check 1, ledger 2739). Each side is one compiled
    program, where run op by op they took two minutes of the test on armada
    (3820); the float64 run that clears a draw is its truth, one program, not
    a float32 one to clear it beside a float64 one to judge it (ledger 4374)."""
    import selection_margins
    from arm_arithmetic import Truth, widened
    from model_support import float64_twin

    from dew.sampling.text import decode_ops, prefill_state

    loaded, plain, reference = source
    shape, prompt = reference["input_ids"].shape, int(reference["decode_prompt"])
    counts = (3, 1)
    kept = jnp.asarray(counts)

    def drafted(variables, ids):
        drawn = ids[:, prompt + 3]
        ops = decode_ops(plain, variables, 0, 0)
        assert ops.record is not None and ops.draft is not None and ops.verify is not None
        state, _ = prefill_state(plain, variables, ModelInputs(ids[:, :prompt]), ops)
        keep = jnp.arange(3)[None, :] < kept[:, None]
        state, _, context = ops.verify(state, ids[:, prompt:prompt + 3], keep)
        assert context is not None
        state = ops.record(state, context, keep)
        scored = []

        def choose(index, logits):
            scored.append(logits)
            return jnp.argmax(logits, axis=-1)

        ops.draft(state, drawn, choose)
        return jnp.stack(scored, 1)

    def histories(ids):
        """Each row's real history and the token its draft starts from."""
        return [(ids[row:row + 1, :prompt + counts[row]], ids[row:row + 1, prompt + 3])
                for row in range(ids.shape[0])]

    def uncached(model, apply=None):
        """The drafter's block over each row's real history, uncached; by `apply`, `model.apply` or a
        capture of it (`selection_margins.capturing`), with what each call captured."""
        apply = apply or model.apply

        def drafted_alone(variables, ids):
            rows, captured = [], []
            for history, token in histories(ids):
                _, trunk = apply(variables, history, mutable=["prediction_inputs"])
                whole = model.apply(variables, trunk["prediction_inputs"], method="draft_context")
                block, drafter = apply(variables, whole, token, method=lambda module, whole, token: (
                    module.draft(whole, token, decode=False)), mutable=[])
                rows.append(block[1][0])
                captured += [trunk, drafter]
            return jnp.stack(rows), captured
        return drafted_alone

    def logits(drafted_alone):
        return lambda variables, ids: drafted_alone(variables, ids)[0]

    # One float64 program for the uncached block and every selection it makes, the trunk's and the
    # drafter's, as the gate runs a truth (`Truth`): a draw's selections clear in it, and the block it
    # computes is the draw's truth.
    wide = float64_twin(plain)
    selecting = Truth(uncached(wide, functools.partial(selection_margins.capturing, wide)))
    truths: dict[bytes, jax.Array] = {}

    def clears(inputs):
        with jax.enable_x64(new_val=True):
            block, captured = selecting(*widened(inputs))
        truths[np.asarray(inputs[1]).tobytes()] = block
        smallest = float(np.min(selection_margins.top_margin(np.asarray(block), 1)))
        for selections in captured:
            smallest = min(smallest, selection_margins.margin(loaded.model_config, selections,
                                                              "dspark draft"))
        return smallest > selection_margins.MARGIN

    variables = jax.tree.map(jnp.asarray, loaded.variables)

    def draw(seed):
        ids = np.random.default_rng([0x2739, seed]).integers(0, plain.vocab_size, shape).astype(np.int32)
        return variables, jnp.asarray(ids)

    @as_given
    def truth(variables, ids):
        """The float64 block the draw cleared over (`clears`)."""
        return truths[np.asarray(ids).tobytes()]

    assert_as_exact(drafted, logits(uncached(plain)), selection_margins.clearing(draw, clears),
                    "dspark draft", seeds=DRAFT_SEEDS, truth_fn=truth)


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
    for bit, both drafted in one program, its context written as the stage's keys and values and read
    back in their basis. Op by op the two drafts took 110 s of the test (ledger 3903)."""
    from dew.sampling.text import decode_ops, prefill_state

    loaded, plain, reference = source
    ids, prompt = jnp.asarray(reference["input_ids"]), int(reference["decode_prompt"])

    def drafted(layout: KVCache, variables, ids):
        model = plain.with_kv_cache(layout)
        ops = decode_ops(model, variables, 0, 0)
        state, _ = prefill_state(model, variables, ModelInputs(ids[:, :prompt]), ops)
        keep = jnp.arange(3)[None, :] < jnp.asarray([3, 1])[:, None]
        state, _, context = ops.verify(state, ids[:, prompt:prompt + 3], keep)
        state = ops.record(state, context, keep)
        scored = []

        def choose(index, logits):
            scored.append(logits)
            return jnp.argmax(logits, axis=-1)

        ops.draft(state, ids[:, prompt + 3], choose)
        return jnp.stack(scored, 1)

    np.testing.assert_array_equal(*jax.jit(lambda variables, ids: [
        drafted(layout, variables, ids)
        for layout in (KVCache(quantized=quantized, page_size=4), KVCache(quantized=quantized))])(
            loaded.variables, ids))


def uncached_window_steps(module, variables, hidden, quantized: str):
    """A sliding V4 layer's per-step outputs over `hidden` with no cache: every position's
    projections computed once, the shared window head dequantized in its own basis as the keys
    and, where a rotated layout keeps it twice, unrotated as the values (a float8 layout stores
    it once, its values its keys), and each step's query attending its window of the slots up
    to its own in a cache-shaped row, so a cached decode is held to a computation that never
    enters the stored basis."""
    def steps(m, x):
        batch, length, _ = x.shape
        capacity = m.max_seq_len
        positions = jnp.broadcast_to(jnp.arange(length)[None], (batch, length))
        cos, sin = rotary_freqs(positions, m.rope_dim, m.rope_theta, rope_scaling=m.rope_scaling,
                                dtype=at_least_fp32(x.dtype))
        place = down_projection(x, m.q_lora_rank + m.head_dim)
        query, _ = m._queries(x, cos, sin, place)
        head = m._window_keys(x, cos, sin, place)
        keys = dequantized_in_own_basis(head, quantized, keys=True)
        values = keys if quantized != "int8" else dequantized_in_own_basis(head, quantized, keys=False)
        slots = jnp.arange(capacity)

        def slotted(values):
            """`values` [B, length, D] in `capacity` slots, zero past them, as an unwritten
            cache slot reads."""
            return jnp.pad(values, ((0, 0), (0, capacity - length), (0, 0)))

        keys, values = slotted(keys), slotted(values)
        outputs = []
        for step in range(length):
            written = (slots <= step)[None, :, None]
            allowed = jnp.broadcast_to(
                ((slots <= step) & (step - slots < m.sliding_window))[None, None], (batch, 1, capacity))
            outputs.append(m._attend(query[:, step:step + 1], jnp.where(written, keys, 0),
                                     jnp.where(written, values, 0), allowed,
                                     cos[:, step:step + 1], sin[:, step:step + 1]))
        return jnp.concatenate(outputs, axis=1)

    return module.apply(variables, hidden, method=lambda m, x: steps(m, x))


@pytest.mark.parametrize("quantized", ["int8", "float8_e4m3fn"])
def test_a_quantized_window_cache_decodes_what_dequantizing_in_the_own_basis_computes(quantized):
    """A sliding V4 layer's quantized window cache, its shared head stored as keys and values
    and its keys read back plain, held to no cache at all: each step as the layer computes it
    over the head dequantized in its own basis (`dequantized_in_own_basis`, plain jnp from the
    layout's rule), bit for bit. Dropping the plain read-back leaves the int8 keys in their
    rotated basis and moves every step off this computation, where a paged-against-compact
    comparison at one format reads them back plain on both arms."""
    module = DeepseekV4Attention(emb_features=32, num_heads=2, head_dim=16, q_lora_rank=8, o_groups=1,
                                 o_lora_rank=8, rope_dim=8, sliding_window=3, max_seq_len=16,
                                 kv_cache=KVCache(quantized=quantized))
    hidden = jax.random.normal(jax.random.key(7), (2, 6, 32))
    variables = {"params": module.init(jax.random.key(3), hidden[:, :1],
                                       MixerCall(decode=True))["params"]}
    decoded = np.asarray(jax.jit(
        lambda variables, hidden: decode_through(module, variables, hidden, prefill=1)[0])(variables, hidden))
    own = np.asarray(jax.jit(
        lambda variables, hidden: uncached_window_steps(module, variables, hidden, quantized))(
        variables, hidden))
    np.testing.assert_array_equal(decoded, own)


DRAFTING = DSparkAttention(emb_features=32, num_heads=2, head_dim=16, q_lora_rank=8, o_groups=1,
                           o_lora_rank=8, rope_dim=8, sliding_window=4, max_seq_len=16)
"""A small DSpark stage's attention, its window shorter than the context it drafts after."""


def cached_draft(module, variables, context, block):
    """A DSpark stage's draft of `block` after `context`, cached: the context written by an empty block,
    then the block drafted after it with none."""
    writing = MixerCall(decode=True, kv_store={DRAFT_CONTEXT: context})
    # The call that allocates the cache writes nothing (`cache_positions`), as a decoder's first does.
    state = {**module.init(jax.random.key(0), block, writing), "params": variables["params"]}
    _, written = module.apply(state, block[:, :0], writing, mutable=["cache"])
    return module.apply({**state, "cache": written["cache"]}, block, MixerCall(decode=True))


def uncached_own_basis_draft(module, variables, context, block, quantized: str):
    """The draft as the cached stage computes it, its context's shared head never entering the stored
    basis: the head projected at the context's slots and dequantized in its own basis as the keys and,
    where a rotated layout keeps it twice, unrotated as the values (`dequantized_in_own_basis`), in a
    cache-shaped row, and the block attending its window of them and all of itself."""
    def drafted(m, context, x):
        batch, written = context.shape[:2]
        slots = jnp.broadcast_to(jnp.arange(written)[None], (batch, written))
        freqs = rotary_freqs(slots, m.rope_dim, m.rope_theta, rope_scaling=m.rope_scaling,
                             dtype=at_least_fp32(context.dtype))
        head = m._window_keys(context, *freqs, down_projection(context, m.head_dim))
        keys = dequantized_in_own_basis(head, quantized, keys=True)
        values = keys if quantized != "int8" else dequantized_in_own_basis(head, quantized, keys=False)
        padding = ((0, 0), (0, m.max_seq_len - written), (0, 0))
        keys, values = jnp.pad(keys, padding), jnp.pad(values, padding)
        last = jnp.full((batch,), written - 1)
        places = jnp.arange(m.max_seq_len)[None]
        length = x.shape[1]
        positions = last[:, None] + 1 + jnp.arange(length)
        cos, sin = rotary_freqs(positions, m.rope_dim, m.rope_theta, rope_scaling=m.rope_scaling,
                                dtype=at_least_fp32(x.dtype))
        place = down_projection(x, m.q_lora_rank + m.head_dim)
        query, _ = m._queries(x, cos, sin, place)
        window = visible(last[:, None], places, causal=True, window=m.sliding_window)
        allowed = jnp.concatenate([jnp.broadcast_to(window[:, None], (batch, length, m.max_seq_len)),
                                   jnp.ones((batch, length, length), bool)], axis=-1)
        own = m._window_keys(x, cos, sin, place)
        return m._attend(query, jnp.concatenate([keys, own], axis=1),
                         jnp.concatenate([values, own], axis=1), allowed, cos, sin)

    return module.apply(variables, context, block, method=drafted)


@pytest.mark.parametrize("quantized", ["int8", "float8_e4m3fn"])
def test_a_quantized_drafting_cache_drafts_what_dequantizing_in_the_own_basis_computes(quantized):
    """A DSpark stage's quantized window cache, the target's context stored as its shared head and its
    keys read back plain (`held.plain`), held to no cache at all: the block drafted over the context's
    head dequantized in its own basis (`uncached_own_basis_draft`), bit for bit. Dropping the plain
    read-back leaves the int8 keys in their rotated basis and moves the draft off this computation, where
    a paged-against-compact comparison at one format reads them back plain on both arms."""
    module = DRAFTING.clone(kv_cache=KVCache(quantized=quantized))
    context = jax.random.normal(jax.random.key(7), (2, 6, 32))
    block = jax.random.normal(jax.random.key(8), (2, 3, 32))
    variables = {"params": module.init(jax.random.key(3), block,
                                       MixerCall(decode=True, kv_store={DRAFT_CONTEXT: context}))["params"]}
    decoded = jax.jit(lambda variables, context, block: cached_draft(module, variables, context, block))
    own = jax.jit(lambda variables, context, block: uncached_own_basis_draft(
        module, variables, context, block, quantized))
    np.testing.assert_array_equal(np.asarray(decoded(variables, context, block)),
                                  np.asarray(own(variables, context, block)))


SCANNED_SEEDS = 8
"""Check 1's declared seeds for a stack scanned against itself unrolled, sized by `seeds_for` paired on
armada."""


def assert_scans_as_unrolled(fields: dict, label: str) -> None:
    """A stack built from `fields` scanned (`scan_layers`) on par with it unrolled (check 1), over drawn
    weights and ids whose every selection clears `selection_margins.MARGIN` in the unrolled float64 run (so
    every run selects alike), the truth that run."""
    import selection_margins
    from model_support import float64_twin, model_draw

    unrolled, scanned = (models.build("causal_transformer", **{**fields, "scan_layers": scan})
                         for scan in (False, True))
    shape = np.load(TINY / "reference.npz")["input_ids"].shape
    wide = float64_twin(unrolled)
    capture = selection_margins.capture(wide)

    def clears(inputs):
        variables, ids = inputs
        with jax.enable_x64(new_val=True):
            captured = capture(jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables), ids)
        return selection_margins.margin(fields, captured, label) > selection_margins.MARGIN

    draw = selection_margins.clearing(model_draw(unrolled, shape, unrolled.vocab_size), clears)
    assert_as_exact(scanned.apply, unrolled.apply, draw, label, seeds=SCANNED_SEEDS, truth_fn=wide.apply)


def test_consecutive_reindex_layers_publish_their_selections_under_scan_layers():
    """Two Reindex layers of one kind in a row stay two runs of one under
    scan_layers, so the Reuse layer after them attends the second one's
    selection, as it does unrolled (check 1, `assert_scans_as_unrolled`);
    scanned together, their publications would stay inside the loop."""
    config = json.loads((TINY / "config.json").read_text())
    text = config["text_config"]
    config = {**config, "text_config": {**text, "index_source_layer_ids": [2, 4, 6, 7, 8, 10]}}
    fields = {**translate_config(config), "dtype": "float32", "attention_impl": "reference"}
    assert_scans_as_unrolled(fields, "consecutive reindex layers scanned")


def test_a_reindex_layer_reads_its_full_layers_publications_with_no_reuse_layer():
    """A stack whose CSA2 layers are a Full layer and the Reindex layers after it,
    with no Reuse layer and so no shared keys and values, still threads the store
    the Full layer publishes its entries in: the Reindex layers read them, unrolled
    and scanned alike (check 1, `assert_scans_as_unrolled`)."""
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
    assert_scans_as_unrolled(fields, "reindex layers with no reuse layer scanned")


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


def test_a_bundle_loaded_text_only_is_its_text_checkpoint(tmp_path):
    """The bundle with its ViT, loaded `text_only`, reads as the family's text checkpoint: the model the
    text-only checkpoint builds, its variables leaf for leaf that checkpoint's, so neither the tower, the
    aligner nor the routers' image-span bias is read, and its logits bit for bit."""
    text = Pretrained.load(vision_bundle(tmp_path), dtype="float32", attention_impl="reference",
                           text_only=True)
    plain = Pretrained.load(TINY, dtype="float32", attention_impl="reference")
    assert text.model == plain.model
    assert jax.tree.structure(text.variables) == jax.tree.structure(plain.variables)
    jax.tree.map(np.testing.assert_array_equal, text.variables, plain.variables)
    ids = jnp.asarray(np.load(TINY / "reference.npz")["input_ids"])
    np.testing.assert_array_equal(np.asarray(text.model.apply(text.variables, ids)),
                                  np.asarray(plain.model.apply(plain.variables, ids)))


def quantized_as_released(directory: Path) -> Path:
    """The checkpoint at `directory` stored as the release stores its decoder: its quantization_config's
    `.scale` format (FP8 blocks of 32 under E8M0 scales, routed experts as FP4 pairs), over every
    projection the blocks cover, the rest as they are."""
    from safetensors.numpy import load_file

    from dew.interop.codecs import deepseek_v4
    from dew.interop.safetensors_io import write_file

    released = json.loads((RELEASED / "config.json").read_text())["quantization_config"]
    codec = deepseek_v4(32, fp4_experts=True)
    stored = {}
    for name, tensor in load_file(directory / "model.safetensors").items():
        covered = (name.endswith(".weight") and tensor.ndim == 2
                   and not name.startswith(("embed", "head", "vision"))
                   and tensor.shape[0] % 32 == 0 and tensor.shape[1] % 32 == 0)
        stored.update(codec.encode(name, tensor) if covered else {name: tensor})
    write_file(stored, directory / "model.safetensors", {"format": "pt"})
    config = json.loads((directory / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps({**config, "quantization_config": released}))
    return directory


def test_a_quantized_bundle_loaded_text_only_is_its_quantized_text_checkpoint(tmp_path):
    """A bundle stored as the release stores its decoder (FP8 and FP4 under its quantization_config),
    loaded `text_only`, is the quantized text-only checkpoint loaded: its codec reads the decoder's names
    unchanged, so its variables are that checkpoint's leaf for leaf."""
    import shutil

    (tmp_path / "bundle").mkdir()
    bundle = quantized_as_released(vision_bundle(tmp_path / "bundle"))
    text = tmp_path / "text"
    shutil.copytree(TINY, text, ignore=shutil.ignore_patterns("vision*"))
    quantized_as_released(text)
    taken = Pretrained.load(bundle, dtype="float32", attention_impl="reference", text_only=True)
    plain = Pretrained.load(text, dtype="float32", attention_impl="reference")
    assert taken.model == plain.model
    jax.tree.map(np.testing.assert_array_equal, taken.variables, plain.variables)


def test_the_image_bias_and_the_dead_image_positions_decide_the_prefill(tmp_path):
    """Routing the image span by the text bias fails the release's image prompts (check 1, a decided
    failure, not Inconclusive), and the image positions change the n-gram ids of the text after them, so
    the fixture exercises both."""
    reference = np.load(TINY / "vision.npz")
    loaded = Pretrained.load(as_released(tmp_path, vision=True), dtype="bfloat16", param_dtype="auto",
                             attention_impl="reference")
    routers = flatten_dict(loaded.variables["moe"])
    routers.update({path: routers[(*path[:-1], "e_score_correction_bias")]
                    for path in routers if path[-1] == "media_bias"})
    with pytest.raises(GateFailure) as failed:
        held_to_the_release_images(loaded.model, loaded.variables, "the image span routed by the text bias",
                                   arm={**loaded.variables, "moe": unflatten_dict(routers)})
    assert not isinstance(failed.value, Inconclusive)
    ids = jnp.asarray(reference["input_ids"])
    media = jnp.asarray(reference["token_types"] >= 0)
    language = {collection: tree["language_model"] for collection, tree in loaded.variables.items()}
    laid = coordinates(ids.shape, valid=jnp.ones(ids.shape, bool),
                       positions=jnp.broadcast_to(jnp.arange(ids.shape[1]), ids.shape))
    hashed = [np.asarray(loaded.model.language_model.apply(
        language, ids, laid, decode=False, media=mask,
        method=lambda module, tokens, laid, *, decode, media: module.engram_hashes(
            tokens, laid, decode=decode, media=media))) for mask in (media, None)]
    after = int(np.flatnonzero(reference["token_types"][0] >= 0)[-1]) + 1
    assert np.any(hashed[0][:, after] != hashed[1][:, after])
