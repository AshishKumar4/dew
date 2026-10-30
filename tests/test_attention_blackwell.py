"""The attention template's Blackwell mainloop (`dew.nn.kernels.templates.attention_blackwell`), first rung:
the full and causal forward of 16-bit operands, interpreted on the CPU by Mosaic GPU's own interpreter,
which models tensor memory and tcgen05's products, held to check 1 against the template's reference from
float64 truth; and compiled for a B200 on this CPU within its shared and tensor memory, no register
spilled. The interpreter is coverage, not proof of the compiled kernel's synchronization; a B200's or a
B300's run is."""

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
from reference_error import assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_blackwell

pytestmark = pytest.mark.kernel

SEEDS = 8
"""Each site's declared draws, check 1's fewest (`reference_error.seeds_for`)."""
CASES = {"full": False, "causal": True}
"""The call forms this rung takes, by whether they are causal."""
SHAPES = {64: (1, 200, 4, 2), 128: (1, 136, 2, 1)}
"""Each head width's rows, tokens (not whole tiles, so the call pads, past one key tile), and query and key
heads (grouped)."""


def _draw(shape, width: int, dtype):
    rows, tokens, heads, kv_heads = shape

    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.normal(size=(rows, tokens, count, width)), dtype)
                     for count in (heads, kv_heads, kv_heads))
    return draw


def _attended(causal: bool, tokens: int, rows: int, *, reference: bool = False, **options):
    """The forward of a call of `rows` rows of `tokens` tokens, causal or full, on this mainloop (`options`
    its own), or the template's reference where `reference` (check 1 widens its inputs for truth)."""
    spans = template.spans(tokens, tokens, rows, causal=causal)

    def run(query, key, value):
        if reference:
            return template.reference(query, key, value, spans)
        return attention_blackwell.forward(query, key, value, spans, **options)
    return run


INTERPRETED_CASES = ("full",)
"""The call forms the interpreter runs: a causal call's per-block walk fails Mosaic GPU's interpreter at the
pinned JAX (an XLA RET_CHECK compiling a role's thread, a tuple of 8 leaves given 9 shardings), which its
compile for a B200 does not (row 2422)."""


@INTERPRETS
@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
@pytest.mark.parametrize("width", list(SHAPES))
@pytest.mark.parametrize("case", INTERPRETED_CASES)
def test_the_blackwell_forward_holds_check_1_interpreted(case, width, natural):
    """The forward of a `case` call of heads `width` wide, interpreted: as exact as the template's reference
    from float64 truth, its padded tokens (masked past the keys) and its grouped heads included."""
    shape = SHAPES[width]
    rows, tokens = shape[:2]
    ours = _attended(CASES[case], tokens, rows, natural=natural, interpret=True)
    reference = _attended(CASES[case], tokens, rows, reference=True)
    assert_as_exact(ours, reference, _draw(shape, width, jnp.bfloat16), f"blackwell {case} {width}",
                    seeds=SEEDS)


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", list(SHAPES))
def test_the_blackwell_forward_compiles_for_a_b200_within_its_resources(width):
    """The causal forward of heads `width` wide at the default tiles, compiled by Mosaic for a compile-only
    B200 on this CPU through ptxas (tests/kernel_compile.py): within the multiprocessor's shared memory
    and its 512 columns of tensor memory, no register spilled."""
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    spans = template.spans(1024, 1024, 2, causal=True)

    def call():
        def forward(query, key, value):
            return attention_blackwell.forward(query, key, value, spans)

        return forward, (jax.ShapeDtypeStruct((2, 1024, 8, width), jnp.bfloat16),
                         jax.ShapeDtypeStruct((2, 1024, 2, width), jnp.bfloat16),
                         jax.ShapeDtypeStruct((2, 1024, 2, width), jnp.bfloat16))

    with generation.assumed("sm100"):
        found = compile_mosaic_kernels([support.KernelCase(f"blackwell attention {width}", call)], "sm100")
    assert len(found) == 1 and found[0].get("kernels"), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


def _forward_of(query, key, value, *, causal: bool = False, **options):
    """The forward interpreted over one row's call, full or causal, in float32 for comparing."""
    spans = template.spans(query.shape[1], key.shape[1], 1, causal=causal)
    out = attention_blackwell.forward(query, key, value, spans, interpret=True, **options)
    return np.asarray(out, np.float32)


@INTERPRETS
@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
def test_the_blackwell_forward_keeps_finite_the_sums_that_pass_fp32s_largest_interpreted(natural):
    """Two keys of zero queries and keys weigh their values of 2^127 equally: their sum passes fp32's
    largest though their mean does not. The rows whose sums overflowed walk again shifting
    (`algebra.walked_launches`), so the output is the mean, 2^127, not infinity."""
    width = 64
    query = jnp.zeros((1, 128, 1, width), jnp.bfloat16)
    key = jnp.zeros((1, 2, 1, width), jnp.bfloat16)
    value = jnp.full((1, 2, 1, width), 2.0 ** 127, jnp.bfloat16)
    np.testing.assert_array_equal(_forward_of(query, key, value, natural=natural), 2.0 ** 127)


@INTERPRETS
def test_the_blackwell_forward_keeps_finite_a_logit_its_scale_keeps_finite_interpreted():
    """A query of 2^127 over a key of 8, 64 wide: the unscaled product, 2^130, passes fp32's largest, the
    default-scaled logit, 2^127, does not. The query is taken by the scale's power before the product
    (`attention.prescaled`), so the call's one key weighs 1 and the output is its value."""
    width = 64
    query = jnp.zeros((1, 128, 1, width), jnp.bfloat16).at[..., 0].set(2.0 ** 127)
    key = jnp.zeros((1, 1, 1, width), jnp.bfloat16).at[..., 0].set(8.0)
    value = jnp.ones((1, 1, 1, width), jnp.bfloat16) * 3
    np.testing.assert_array_equal(_forward_of(query, key, value), 3.0)


@INTERPRETS
@pytest.mark.parametrize("tiles", [attention_blackwell.Tiles(64, 2), attention_blackwell.Tiles(192, 2)],
                         ids=["64-keys", "192-keys"])
def test_the_blackwell_forward_reads_the_calls_keys_at_every_key_tile_interpreted(tiles):
    """Zero queries and keys over unit values at key tiles of 64 and 192, 64 and 192 queries and keys: every
    query weighs the call's keys equally and none of the padding, so every output is 1, and every query's
    row is written (its query blocks the call's padded to whole query and key tiles)."""
    for tokens in (64, 192):
        query = jnp.zeros((1, tokens, 1, 64), jnp.bfloat16)
        value = jnp.ones((1, tokens, 1, 64), jnp.bfloat16)
        np.testing.assert_array_equal(_forward_of(query, query, value, tiles=tiles), 1.0)


@INTERPRETS
@pytest.mark.parametrize(("width", "value_width"), [(32, 32), (64, 128)], ids=["32-wide", "values-wider"])
def test_the_blackwell_forward_keeps_its_widths_through_its_guards_interpreted(width, value_width):
    """Heads 32 wide (the swizzle pads them to 64) and values wider than their queries and keys: the
    default's guards trace the plain-JAX rewalk beside the launch, over the same padded call, so the
    output is the call's `[B, Q, H, Dv]` and on par with the template's reference (check 1)."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.normal(size=(1, 136, 2, size)), jnp.bfloat16)
                     for size in (width, width, value_width))

    ours = _attended(CASES["full"], 136, 1, interpret=True)
    assert jax.eval_shape(ours, *draw(0)).shape == (1, 136, 2, value_width)
    assert_as_exact(ours, _attended(CASES["full"], 136, 1, reference=True), draw,
                    f"blackwell {width} over {value_width}", seeds=SEEDS)


@INTERPRETS
def test_the_blackwell_forward_of_no_keys_is_zero_interpreted():
    """A call of queries over no keys walks one tile of padding, whose zero values weigh to a zero output,
    every query's row written."""
    query = jnp.ones((1, 128, 1, 64), jnp.bfloat16)
    empty = jnp.zeros((1, 0, 1, 64), jnp.bfloat16)
    spans = template.spans(128, 0, 1)
    out = attention_blackwell.forward(query, empty, empty, spans, interpret=True)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.zeros((1, 128, 1, 64), np.float32))
