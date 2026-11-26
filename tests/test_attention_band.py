"""A band bias read natively by the attention template (row 13138, kernel-templates.md row 58): Inkling 2's
input-conditioned relative bias held as a band of keys about each query (`dew.nn.attention.BandBias`), read a
tile at a time at its key tile's column (`templates.Kind.BAND`), never written out over the call's keys."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import attention_terms, kept
from reference_error import assert_as_exact, with_gradients

from dew.nn.attention import BandBias, scaled_dot_product_attention
from dew.nn.kernels import generation
from dew.nn.kernels.templates import band

HEADS, KV_HEADS, WIDTH = 4, 2, 32
SEEDS = 8
"""Check 1's declared seeds for a band bias on the template."""

QUERIES = 1600
"""A call's queries and keys: past two band groups (`band.BAND_GROUP`), so three origins."""

CASES = {"causal": {"causal": True, "extent": 300}, "sliding": {"causal": True, "window": 300, "extent": 300},
         "global": {"causal": True, "extent": 1024}, "full": {"causal": False, "extent": 300},
         "rows-apart": {"causal": True, "extent": 300, "offset": (0, 2048), "shared": True}}
"""Each layer a band serves: a causal one whose band is shorter than its keys (Inkling 2's global layers'),
a sliding one whose window is its band (its sliding layers'), one at Inkling 2's extent, a full one whose
keys past each query its band holds none of, and one whose rows share the band's values at origins apart
(two chunks' places, 0 and 2048: a row's values shared and its origins its own, head review)."""


def _banded(dense: np.ndarray, extent: int, offset: np.ndarray, rows: int) -> np.ndarray:
    """A `[B, H, Q, K]` array's entries at each query's band columns (`band.band_origins` from each row's
    `offset`), zeros past the keys, summed over the rows where the band's values are `rows` (one, shared):
    the band's shape of a dense bias's terms."""
    queries, keys = dense.shape[-2:]
    origins = np.asarray(band.band_origins(np.asarray(offset, np.int32), queries, extent))
    at = origins[:, None, :, None] + np.arange(band.band_width(extent))
    inside = (at >= 0) & (at < keys)
    read = np.take_along_axis(dense, np.clip(at, 0, keys - 1), axis=-1)
    read = np.where(inside, read, 0.0)
    return read.sum(axis=0, keepdims=True) if rows == 1 else read


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_band_bias_on_the_template_is_on_par_with_its_dense_bias_on_the_reference_path(case):
    """A bf16 call biased by a band (`BandBias`) on the template, its band read a tile at a time
    (`templates.Kind.BAND`), on par (check 1) with the reference path's dense bias from float64: its output
    and the query's, key's, value's and band's gradients, the band's in its own shape. Its every
    configuration of Inkling 2's: a causal layer's band, a sliding one's, one at extent 1024, a full one."""
    options = CASES[case]
    extent, causal, window = options["extent"], options["causal"], options.get("window")
    width = band.band_width(extent)
    offset = jnp.asarray(options.get("offset", 0), jnp.int32)
    rows = 1 if options.get("shared") else 2

    def drawn(seed):
        rng = np.random.default_rng([0x13138, seed])
        query, key, value, cotangent = (
            jnp.asarray(rng.standard_normal(shape), jnp.bfloat16) for shape in (
                (2, QUERIES, HEADS, WIDTH), (2, QUERIES, KV_HEADS, WIDTH), (2, QUERIES, KV_HEADS, WIDTH),
                (2, QUERIES, HEADS, WIDTH)))
        values = jnp.asarray(0.5 * rng.standard_normal((rows, HEADS, QUERIES, width)), jnp.float32)
        return query, key, value, values, cotangent

    def attended(implementation):
        def call(query, key, value, values):
            bias = BandBias(values, offset, extent)
            if implementation == "reference":
                bias = bias.dense(QUERIES)
            return scaled_dot_product_attention(query, key, value, bias=bias, causal=causal,
                                                sliding_window=window, implementation=implementation)

        return with_gradients(call)

    keep = kept(QUERIES, QUERIES, causal=causal, sliding_window=window)

    def terms(query, key, value, values, dout):
        dense = np.asarray(BandBias(values, offset, extent).dense(QUERIES))
        output, dq, dk, dv, bias = attention_terms(query, key, value, dout, keep, bias=dense)
        return output, dq, dk, dv, _banded(bias, extent, np.broadcast_to(np.asarray(offset), (2,)), rows)

    assert_as_exact(jax.jit(attended("template")), jax.jit(attended("reference")), drawn,
                    f"a {case} band of extent {extent} on the template", seeds=SEEDS, terms=terms)


def test_a_band_bias_on_the_template_writes_no_dense_bias():
    """A causal call of 32768 tokens biased by a band of extent 1024 on an H100's template (Inkling 2's global
    layer) forms no array of its every query's and key's bias in the program XLA runs, its training step's
    among them: the band its `[B, H, Q, 2048]`, where a dense bias held 96 GiB at Inkling 2's 32K context."""
    from reference_error import equations

    from dew.nn import programs

    queries, extent = 32768, 1024
    query = jax.ShapeDtypeStruct((1, queries, HEADS, WIDTH), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, queries, KV_HEADS, WIDTH), jnp.bfloat16)
    values = jax.ShapeDtypeStruct((1, HEADS, queries, band.band_width(extent)), jnp.float32)

    def loss(query, key, value, values):
        bias = BandBias(values, jnp.int32(0), extent)
        out = scaled_dot_product_attention(query, key, value, bias=bias, causal=True,
                                           implementation="template")
        return jnp.sum(out.astype(jnp.float32))

    with generation.assumed("sm90"):
        traced = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 2, 3)))(query, key, key, values).jaxpr
    shapes = [getattr(var.aval, "shape", ()) for equation in equations(traced, prune=programs.opaque)
              for var in equation.outvars]
    assert not any(tuple(shape[-2:]) == (queries, queries) for shape in shapes)


@pytest.mark.parametrize("mainloop", ["portable", "triton", "mosaic"])
def test_a_band_of_masked_keys_leaves_the_keys_past_it_unbiased(mainloop):
    """A full call over 512 keys whose band (extent 1, origins at key 256) holds -inf at every column, on each
    template mainloop (the portable one through the call, Triton's and Mosaic GPU's interpreted as an
    H100): the keys past the band are biased by nothing, so the output is their values' mean, 1, finite. A
    tile past the band read its clamped column's -inf and scaled it by zero, NaN, on Mosaic (head
    review)."""
    from dew.nn.kernels.templates import attention as template, attention_mosaic, attention_triton
    from dew.nn.kernels.templates.hooks import Hook, Side

    queries = keys = 512
    width = band.band_width(1)
    offset = jnp.int32(256 + 255)
    origins = band.band_origins(offset, queries, 1)
    assert int(origins[0, 0]) == 256
    query = jnp.zeros((1, queries, HEADS, WIDTH), jnp.float32)
    key = jnp.zeros((1, keys, KV_HEADS, WIDTH), jnp.float32)
    value = jnp.ones((1, keys, KV_HEADS, WIDTH), jnp.float32)
    values = jnp.full((1, HEADS, queries, width), -jnp.inf, jnp.float32)
    if mainloop == "portable":
        with generation.chosen(attention_mainloop="portable"):
            out = scaled_dot_product_attention(query, key, value, bias=BandBias(values, offset, 1),
                                               implementation="template")
    else:
        hooks = template.Hooks(score=Hook(template.Banded(template.Biased()), (
            Side(values, template.Kind.BAND), Side(origins[:, None, :], template.Kind.QUERY))))
        spans = template.spans(queries, keys, 1)
        run = attention_triton.forward if mainloop == "triton" else attention_mosaic.forward
        with generation.assumed("sm90", interpreted=True):
            out = run(query, key, value, spans, hooks, interpret=True)
    np.testing.assert_array_equal(np.asarray(out), 1.0)


def test_a_portable_band_gradient_holds_no_array_of_every_query_and_key():
    """The portable mainloop's backward of a band (`attention_portable._dkv_launch`) adds each program's
    piece into the band's compact gradient: no array of its program holds every query's and key's entry,
    where it stacked every key block's `[Qp, bk]` and gathered the band after (head review)."""
    from reference_error import equations


    queries, extent = 2048, 64
    query = jax.ShapeDtypeStruct((1, queries, HEADS, WIDTH), jnp.float32)
    key = jax.ShapeDtypeStruct((1, queries, KV_HEADS, WIDTH), jnp.float32)
    values = jax.ShapeDtypeStruct((1, HEADS, queries, band.band_width(extent)), jnp.float32)

    def loss(query, key, value, values):
        bias = BandBias(values, jnp.int32(0), extent)
        return jnp.sum(scaled_dot_product_attention(query, key, value, bias=bias, causal=True,
                                                    implementation="template"))

    with generation.chosen(attention_mainloop="portable"):
        traced = jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 2, 3)))(query, key, key, values).jaxpr
    sizes = [int(np.prod(getattr(var.aval, "shape", ()))) for equation in equations(traced)
             for var in equation.outvars]
    assert max(sizes) < HEADS * queries * queries, max(sizes)

