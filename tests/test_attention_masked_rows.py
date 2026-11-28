"""A query row that sees no key reads nothing, on every attention implementation.

Its output is exactly zero and so is its query's gradient, and it adds
nothing to any key's or value's gradient. A kernel left to such a row
softmaxes its masked logits into the mean of the values, and Splash's
backward recomputes that uniform weight into its query's and its keys'
gradients. Two ways a row sees no key: an explicit mask with an all-false
row, and a packed row's padding (segment id 0), which `document_mask` keeps
from every key. Each implementation is held to the float64 attention of
tests/attention_oracle.py, which defines such a row as zero: on par (check 1)
with the reference attention at the drawn dtype, fp32 and the GPU kernels'
bf16, every entry sized by its float64 terms (`attention_on_par`). A row
whose own key is cut but whose block runs on reads its block's later keys
(`block_groups`). A row that sees keys but weighs none of them (each logit -inf, a bias's or a fill's) of a
call that hides no key reads NaN, its derivative NaN, as jax.nn's softmax reads it, on the reference path
and on every template mainloop alike; a masked call's reads zero (row 14116).
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tier_wire
from arm_arithmetic import drawn_on_the_host
from attention_oracle import attention_on_par, exact_value_and_grads, kept, packed_ids, qkv, value_and_grads
from reference_error import FEWEST_SEEDS

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_portable,
    attention_triton,
)
from dew.nn.kernels.templates.hooks import Hook, Side
from dew.nn.precision import FP32_SUMS


def masked():
    """Rows 10-19 and 200-209 keep no key; rows 130 on keep only the first 128 keys."""
    mask = np.ones((1, 1, 256, 256), bool)
    mask[..., 130:, 128:] = False
    mask[..., 10:20, :] = False
    mask[..., 200:210, :] = False
    return {"mask": mask}, np.r_[10:20, 200:210]


def padded():
    """Two packed rows of 256 that end in padding."""
    return {"causal": True, "segment_ids": packed_ids((100, 84, 50), (30, 190, 10))}, None


CASES = {"mask": masked, "padding": padded}
CPU = [("reference", "mask"), ("xla", "mask"), ("tpu", "mask"),
       ("reference", "padding"), ("xla", "padding"), ("tpu", "padding")]
GPU = [("cudnn", "mask"), ("cudnn", "padding"), ("flash", "padding")]


def dead_rows(structure, dead):
    """`[B, S]` true where a query row keeps no key."""
    if dead is not None:
        rows = np.zeros((2, 256), bool)
        rows[:, dead] = True
        return rows
    return np.asarray(structure["segment_ids"]) == 0


def run(implementation, case, dtype):
    structure, dead = CASES[case]()
    query, key, value = qkv((2, 256, 4, 64), dtype)
    extra = {"max_segments": 4} if implementation == "flash" else {}
    found = value_and_grads(implementation, query, key, value, **structure, **extra)
    truth = exact_value_and_grads(query, key, value, **structure)
    rows = dead_rows(structure, dead)
    assert rows.any()
    for name, array in zip(("output", "query gradient"), found[:2], strict=True):
        assert not np.any(array[rows]), f"{implementation} {case}: a row seeing no key has a nonzero {name}"
    return query, key, value, structure, found, truth


SEEDS = dict.fromkeys(CPU, 8)
"""Each CPU implementation's declared draws against the fp32 reference attention (check 1), by its case:
sized by `seeds_for` paired on armada."""


@pytest.mark.parametrize("implementation, case", CPU, ids=[f"{i}-{c}" for i, c in CPU])
def test_a_row_that_sees_no_key_reads_nothing(implementation, case):
    """Each CPU implementation: a row that sees no key has a zero output and query gradient, and its output
    and gradients are on par (check 1) with the fp32 reference attention's from float64 over drawn
    operands and cotangents."""
    structure = run(implementation, case, jnp.float32)[3]
    _on_par(implementation, case, structure, jnp.float32, SEEDS[implementation, case])


def _on_par(implementation: str, case: str, structure: dict, dtype, seeds: int) -> None:
    """`implementation`'s output and gradients over `structure` on par (check 1) with the reference
    attention's at the drawn `dtype`, from float64, over drawn `dtype` operands and cotangents."""
    shape = (2, 256, 4, 64)
    extra = {"max_segments": 4} if implementation == "flash" else {}

    def drawn(seed):
        cotangent = jax.random.normal(jax.random.fold_in(jax.random.key(93), seed), shape, dtype)
        return (*qkv(shape, dtype, seed=seed), cotangent)

    def draw(seed):
        return drawn_on_the_host(drawn, seed)

    def attended(implementation, **options):
        def attend(q, k, v):
            return scaled_dot_product_attention(q, k, v, implementation=implementation, **structure,
                                                **options)
        return attend

    def reference(*drawn):
        # The 'tpu' row interprets the TPU kernel on this CPU: its reference sums as the target does (fp32),
        # not as the CPU's policy does.
        with jax.default_matmul_precision(FP32_SUMS) if implementation == "tpu" else contextlib.nullcontext():
            return attended("reference", precision=jax.lax.Precision.HIGHEST)(*drawn)

    # The truth attention written out over the keys each query keeps, every entry sized by its float64 terms
    # (`attention_on_par`): a document's first query keeps one key, its query gradient held to delta's
    # rounding against dP, not to its own zero.
    attention_on_par(attended(implementation, **extra), reference, draw, f"{implementation} {case}",
                     keep=kept(256, 256, **structure), seeds=seeds)


@pytest.mark.parametrize("implementation, case", GPU, ids=[f"{i}-{c}" for i, c in GPU])
def a_row_that_sees_no_key_reads_nothing_on_the_fused_gpu_kernels(implementation, case,
                                                                         without_deterministic_ops):
    """Each fused GPU kernel, bf16: a row that sees no key has a zero output and query gradient, and its
    output and gradients are on par (check 1) with the bf16 reference attention's from float64; a GPU tier
    part's prepare compiles its sites alone, its run checking the dead rows too."""
    _on_par(implementation, case, CASES[case]()[0], jnp.bfloat16, FEWEST_SEEDS)
    if not tier_wire.preparing():
        run(implementation, case, jnp.bfloat16)


@pytest.mark.parametrize("implementation", ["reference", "xla", "template"])
def test_a_row_its_key_starts_empty_reads_its_blocks_later_keys(implementation):
    """A causal row whose own key the key starts cut sees none of its own, and its whole block's later keys
    revive it, on every implementation: in the first row a two-token block over starts of 1, whose first
    query reads the second key alone; in the second, starts of 2 under a block of tokens 1 and 2, whose
    first query reads key 2 alone, and a first query no block holds, which reads nothing. A row reading one
    key is that key's value exactly."""
    query, key, value = qkv((2, 4, 2, 8), jnp.float32)
    groups = jnp.asarray([[0, 0, -1, -1], [-1, 1, 1, -1]], jnp.int32)
    out = np.asarray(scaled_dot_product_attention(query, key, value, causal=True,
                                                  key_value_seq_starts=jnp.asarray([1, 2], jnp.int32),
                                                  block_groups=groups, implementation=implementation))
    np.testing.assert_array_equal(out[0, 0], np.asarray(value)[0, 1])
    np.testing.assert_array_equal(out[1, 1], np.asarray(value)[1, 2])
    np.testing.assert_array_equal(out[1, 0], 0)


@pytest.mark.parametrize("implementation", ["reference", "xla", "template"])
def test_a_row_whose_window_holds_no_valid_key_reads_nothing(implementation):
    """Key validity can empty a causal window: a row whose window of two holds only invalid keys reads
    exactly zero on every implementation, and a row beside it that holds one reads that key's value."""
    query, key, value = qkv((1, 6, 2, 8), jnp.float32)
    valid = jnp.asarray([[1, 1, 0, 0, 1, 0]], bool)
    out = np.asarray(scaled_dot_product_attention(query, key, value, causal=True, sliding_window=2,
                                                  key_valid=valid, implementation=implementation))
    np.testing.assert_array_equal(out[0, 3], 0)
    np.testing.assert_array_equal(out[0, 2], np.asarray(value)[0, 1])
    np.testing.assert_array_equal(out[0, 5], np.asarray(value)[0, 4])


def test_a_lone_grouped_query_keeps_its_key_validity():
    """A decode step's one query of two heads over one key head, its keys bounded by their lengths and one
    of the two invalid: the invalid key's value takes no weight (train 217's K1: the grouped shortcut, which
    carries key lengths alone, read it)."""
    query = jnp.ones((1, 1, 2, 16), jnp.float32)
    key = jnp.ones((1, 2, 1, 16), jnp.float32)
    value = jnp.stack([jnp.zeros((1, 16)), jnp.full((1, 16), 10.0)])[None]
    out = scaled_dot_product_attention(query, key, value, key_value_seq_lengths=jnp.asarray([2]),
                                       key_valid=jnp.asarray([[True, False]]))
    np.testing.assert_array_equal(np.asarray(out), 0.0)


def test_a_row_its_counted_places_bring_in_reach_reads_its_key():
    """Two queries at counted places 0 and 0 over one key at place 0 in a window of one: each sees the key,
    so neither row comes back zero (train 217's K2: the rows' trace-time reach read their indices, not
    their places)."""
    query = jnp.ones((1, 2, 1, 16), jnp.float32)
    key = jnp.ones((1, 1, 1, 16), jnp.float32)
    value = jnp.full((1, 1, 1, 16), 10.0)
    out = scaled_dot_product_attention(query, key, value, sliding_window=1,
                                       places=(jnp.asarray([[0, 0]]), jnp.asarray([[0]])))
    np.testing.assert_array_equal(np.asarray(out), 10.0)


MASSLESS = (1, 40, 4, 2, 64, 64)
"""A call's rows, queries, query heads, key heads, width and keys: two query heads a key head, the queries
past none of the mainloops' tiles, so each pads them."""


def _massless_operands():
    """Zero queries and keys, values of one, and a bias of query head 1's row 3 at -inf on every key (a row
    of no mass) and query head 2's row 5 on every other key: every weight a power of two, so each mainloop's
    sums are exact, its outputs 1 or NaN, its gradients 0, multiples of 1/64 or NaN."""
    batch, queries, heads, kv_heads, width, keys = MASSLESS
    bias = np.zeros((batch, heads, queries, keys), np.float32)
    bias[0, 1, 3] = -np.inf
    bias[0, 2, 5, ::2] = -np.inf
    return (jnp.zeros((batch, queries, heads, width), jnp.bfloat16),
            jnp.zeros((batch, keys, kv_heads, width), jnp.bfloat16),
            jnp.ones((batch, keys, kv_heads, width), jnp.bfloat16), jnp.asarray(bias))


HALF = jnp.asarray([MASSLESS[-1] // 2], jnp.int32)
"""A masked call's key lengths: each row's first half of the keys, a power of two of them."""


def _reference_massless(masked: bool):
    """The reference path's output and gradients (query, key, value, bias) of `_massless_operands` under a
    cotangent of ones: jax.nn's softmax op for op unmasked, over `HALF` of the keys where `masked`."""
    def attend(query, key, value, bias):
        return scaled_dot_product_attention(query, key, value, bias=bias,
                                            key_value_seq_lengths=HALF if masked else None,
                                            implementation="reference")

    out, pull = jax.vjp(attend, *_massless_operands())
    return out, *pull(jnp.ones_like(out))


def _template_massless(mainloop: str, masked: bool):
    """`mainloop`'s output and gradients of `_massless_operands`, the bias a tile parameter of its score hook:
    the plain-JAX mainloop's, or the Pallas-Triton one's or Mosaic's interpreted as the named GPU runs it."""
    query, key, value, bias = _massless_operands()
    batch, queries, _, _, _, keys = MASSLESS
    spans = template.spans(queries, keys, batch, lengths=HALF if masked else None)
    hooks = template.Hooks(score=Hook(template.Biased(), (Side(bias, template.Kind.TILE),)))
    dout = jnp.ones(query.shape, query.dtype)
    if mainloop == "portable":
        tiles = attention_portable.Tiles(32, 32, 32)
        out = attention_portable.xla_mainloop(tiles, query, key, value, spans, hooks)
        grads = attention_portable.backward(tiles, query, key, value, spans, hooks, dout)
    elif mainloop == "triton":
        with generation.assumed("sm90", interpreted=True):
            out = attention_triton.forward(query, key, value, spans, hooks, interpret=True)
            grads = attention_triton.backward(query, key, value, spans, hooks, dout, interpret=True)
    else:
        with generation.assumed(mainloop.split()[1], interpreted=True):
            out, *residuals = attention_mosaic.forward(query, key, value, spans, hooks,
                                                       residuals=template.Kept.SPLIT, interpret=True)
            grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                              interpret=True)
    return out, grads.query, grads.key, grads.value, grads.score_params[0]


MAINLOOPS = ("portable", "triton", "mosaic sm90", "mosaic sm80")


@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("mainloop", MAINLOOPS)
def test_a_row_of_no_mass_reads_as_the_reference_path_on_every_mainloop(mainloop, masked):
    """A row whose bias is -inf at every key it sees, beside a padded call's rows and a row half of whose
    keys a bias hides, on each template mainloop, as the reference path reads it: an unmasked call's row of
    no mass reads NaN, its query's gradient NaN and its key head's every key and value gradient NaN, as
    jax.nn's softmax gives them; a masked call's (key lengths) reads zero, every gradient finite. Every other
    entry is the reference path's bit for bit. The template read zero for the unmasked call's row, so a
    written biased attention recognized onto it changed what the program computed (row 14116)."""
    found = _template_massless(mainloop, masked)
    expected = _reference_massless(masked)
    row = np.asarray(expected[0], np.float32)[0, 3, 1]
    assert (row == 0).all() if masked else np.isnan(row).all()
    for name, mine, theirs in zip(("output", "query", "key", "value", "bias"), found, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(mine, np.float32), np.asarray(theirs, np.float32),
                                      err_msg=f"{mainloop} {name}")


@pytest.mark.parametrize("implementation", ["reference", "template"])
def test_a_filled_row_that_sees_no_key_reads_nan(implementation):
    """A packed call under a mask fill of -inf (`mask_fill`, a written program's `where(mask, logits, -inf)`),
    its last query a padding row that sees no key: that row's every logit is the fill, so it reads NaN as
    jax.nn's softmax does, on the reference path and on the template (`template.Filled`, whose call hides no
    key); its documents' rows read their keys' mean (row 14116)."""
    query, key, value = (jnp.zeros((1, 4, 1, 8), jnp.float32), jnp.zeros((1, 4, 1, 8), jnp.float32),
                         jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 1, 8))
    ids = jnp.asarray([[1, 1, 2, 0]], jnp.int32)
    out = np.asarray(scaled_dot_product_attention(query, key, value, segment_ids=ids, mask_fill=-np.inf,
                                                  implementation=implementation))
    rows = np.asarray(value)[0]
    assert np.isnan(out[0, 3]).all()
    np.testing.assert_array_equal(out[0, :3], np.stack([(rows[0] + rows[1]) / 2] * 2 + [rows[2]]))
