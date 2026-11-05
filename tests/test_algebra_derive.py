"""Template forms derived from a declaration (`dew.algebra.derive`, docs/design/algebra.md §6.9): attention's
softmax over keys stated on whole operands, walked in key tiles through its derived step, the walk split in
two and merged, each held to the float64 truth (check 1) against the whole-operand function; and the
declarations no monoid along the walk holds, refused."""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew import algebra
from dew.algebra import SATURATION, derive
from dew.nn import precision

QUERIES, KEYS, WIDTH, TILE = 16, 96, 32, 32


def attention(q, k, v, rows, keys):
    """Softmax over keys of the scaled logits, then the values: the max-shifted form a model writes."""
    del rows, keys
    s = (q @ k.T) * WIDTH ** -0.5
    m = jnp.max(s, axis=-1)
    p = jnp.exp(s - m[:, None])
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def causal(q, k, v, rows, keys):
    """The same over each query's keys up to its own place (`rows`, `keys`: the queries' and keys'
    places, operands along their axes, so a tile's keys carry their own), the rest masked out by a
    selection."""
    s = (q @ k.T) * WIDTH ** -0.5
    seen = keys[None, :] <= rows[:, None]
    s = jnp.where(seen, s, -jnp.inf)
    m = jnp.max(s, axis=-1)
    p = jnp.where(seen, jnp.exp(s - m[:, None]), 0.0)
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def base2(q, k, v, rows, keys):
    """The base-2 form, `log2(e)` folded into the logits' scale."""
    del rows, keys
    s = (q @ k.T) * (WIDTH ** -0.5 * float(np.log2(np.e)))
    m = jnp.max(s, axis=-1)
    p = jnp.exp2(s - m[:, None])
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def shifted(q, k, v, rows, keys):
    """Causal attention whose first query sees no key, its maximum shifted safely (`where(m == -inf, 0, m)`)
    and its total guarded: a row of no key gives zeros."""
    s = (q @ k.T) * WIDTH ** -0.5
    seen = (keys[None, :] <= rows[:, None]) & (rows[:, None] > 0)
    s = jnp.where(seen, s, -jnp.inf)
    m = jnp.max(s, axis=-1)
    p = jnp.exp(s - jnp.where(jnp.isneginf(m), 0.0, m)[:, None])
    total = jnp.sum(p, axis=-1)
    return (p @ v) / jnp.where(total == 0, 1.0, total)[:, None]


def products(q, k, v, rows, keys):
    """Causal attention whose products are Dew's (`float32_einsum`, `float32_matmul`): under the suite's wide
    CPU arithmetic each a choice by the platform it lowers for, which the derivation reads as one product."""
    s = precision.float32_einsum("rw,kw->rk", q, k) * WIDTH ** -0.5
    seen = keys[None, :] <= rows[:, None]
    s = jnp.where(seen, s, -jnp.inf)
    m = jnp.max(s, axis=-1)
    p = jnp.where(seen, jnp.exp(s - m[:, None]), 0.0)
    return precision.float32_matmul(p, v) / jnp.sum(p, axis=-1)[:, None]


def softmax_beside_a_half_maximum(q, k, v, rows, keys):
    """A softmax walk beside an unwalked maximum of the queries in fp16, independent of it: each maximum
    takes its own dtype's least value, where one for the whole program refused it (row 3637)."""
    del rows, keys
    s = (q @ k.T) * WIDTH ** -0.5
    p = jnp.exp(s - jnp.max(s, axis=-1, keepdims=True))
    return (p @ v) / jnp.sum(p, axis=-1)[:, None] + jnp.max(q.astype(jnp.float16), axis=-1).astype(
        jnp.float32)[:, None]


DECLARED: dict[str, tuple[Callable, Literal[2, "e"]]] = {
    "attention": (attention, "e"), "causal": (causal, "e"), "base2": (base2, 2), "shifted": (shifted, "e"),
    "products": (products, "e"), "a softmax beside an fp16 maximum": (softmax_beside_a_half_maximum, "e")}
AXES = {"q": ("rows", "width"), "k": ("keys", "width"), "v": ("keys", "values"), "rows": ("rows",),
        "keys": ("keys",)}
ROWS, PLACES = jnp.arange(QUERIES) * (KEYS // QUERIES), jnp.arange(KEYS)
"""The queries' places (each query a few keys on from the last) and the keys'."""


def whole(function):
    """`function` on (q, k, v), its places the whole sequence's."""
    return lambda q, k, v: function(q, k, v, ROWS, PLACES)


def draw(seed):
    """Queries, keys and values whose logits span six decades of magnitude across rows, so a row's maximum
    moves from tile to tile and the rescale matters."""
    rng = np.random.default_rng(seed)
    scale = 10.0 ** rng.uniform(-1, 2, (QUERIES, 1))
    return (jnp.asarray(rng.standard_normal((QUERIES, WIDTH)) * scale, jnp.float32),
            jnp.asarray(rng.standard_normal((KEYS, WIDTH)), jnp.float32),
            jnp.asarray(rng.standard_normal((KEYS, WIDTH)), jnp.float32))


def declared(function, exponent: Literal[2, "e"] = "e") -> derive.Reduction:
    """`function`'s walk over the keys."""
    return derive.Reduction(function, AXES, walked="keys", exponent=exponent)


def operands(q, k, v) -> dict:
    return {"q": q, "k": k, "v": v, "rows": ROWS, "keys": PLACES}


def tile_of(values: dict, tile: int) -> dict:
    """Each operand's keys of key tile `tile` (the whole of an operand with no keys axis)."""
    keys = slice(tile * TILE, (tile + 1) * TILE)
    return {name: value[keys] if AXES[name][0] == "keys" else value for name, value in values.items()}


def shaped(tile: dict) -> dict:
    """A tile's operands' shapes and dtypes, by name: a walk's (`Derived.init`, `merge`, `finish`)."""
    return {name: (value.shape, value.dtype) for name, value in tile.items()}


def walked(forms: derive.Derived, tiles: list[dict], product=None):
    """The state after stepping each of `tiles` (its operands' tiles) in order, from the empty state, each
    product `product`'s where given (a mainloop's issuer)."""
    state = forms.init(shaped(tiles[0]))
    for tile in tiles:
        state = forms.step(state, tile, product=product)
    return state


def walked_whole(forms: derive.Derived):
    """The declaration's derived step over every key tile in order, finished: a function of (q, k, v)."""
    def run(q, k, v):
        values = operands(q, k, v)
        return forms.finish(walked(forms, [tile_of(values, tile) for tile in range(KEYS // TILE)]), values,
                            shapes=shaped(tile_of(values, 0)))

    return run


def merged_whole(forms: derive.Derived):
    """The keys walked in two disjoint halves, each from the empty state, their states merged and finished: a
    function of (q, k, v)."""
    tiles = KEYS // TILE

    def run(q, k, v):
        values = operands(q, k, v)
        first = walked(forms, [tile_of(values, tile) for tile in range(tiles // 2)])
        second = walked(forms, [tile_of(values, tile) for tile in range(tiles // 2, tiles)])
        shapes = shaped(tile_of(values, 0))
        return forms.finish(forms.merge(first, second, values, shapes=shapes), values, shapes=shapes)

    return run


@pytest.mark.parametrize("walk", ["walked", "merged"])
@pytest.mark.parametrize("name", list(DECLARED))
def test_the_derived_walk_computes_the_declared_function(name, walk):
    """The declaration's derived step over every key tile in order, finished, or over the keys in two disjoint
    halves, each from the empty state, their states merged (a split walk's combine, the work scheduler's
    fix-up), finished: on par with the whole-operand function against the float64 truth (check 1)."""
    function, exponent = DECLARED[name]
    forms = derive.derive(declared(function, exponent), accurate=False, saturation=SATURATION)
    run = walked_whole(forms) if walk == "walked" else merged_whole(forms)
    assert_as_exact(jax.jit(run), jax.jit(whole(function)), draw, f"{name} {walk}", seeds=16)


def test_a_walks_sums_added_by_its_mainloop_take_each_tiles_factor_and_sum():
    """A mainloop that makes a walked sum's additions itself (`Derived.step`'s `add`, a compensated walk's
    running sums) takes each tile's rescale factor and its tile's sum: added twice over, the softmax's
    denominator doubles on every tile, and the walk's attention is the declared function's halved, on par
    against the float64 truth (check 1)."""
    forms = derive.derive(declared(attention), accurate=False, saturation=SATURATION)

    def doubled(state, factor, summed):
        return state * factor + 2 * summed

    def run(q, k, v):
        values = operands(q, k, v)
        tiles = [tile_of(values, tile) for tile in range(KEYS // TILE)]
        state = forms.init(shaped(tiles[0]))
        for tile in tiles:
            state = forms.step(state, tile, add=doubled)
        return forms.finish(state, values, shapes=shaped(tiles[0]))

    assert_as_exact(jax.jit(run), jax.jit(lambda q, k, v: whole(attention)(q, k, v) / 2), draw,
                    "attention, its sums added twice", seeds=16)


def held(q, k, v, rows, keys):
    """Attention whose largest logit weighs exactly 1, its exponent held to 0 by a select (the accurate
    form's weights over scaled logits, against a compiler's contraction of the scale into the subtraction)."""
    del rows, keys
    s = (q @ k.T) * WIDTH ** -0.5
    m = jnp.max(s, axis=-1)
    p = jnp.exp(jnp.where(s < m[:, None], s - m[:, None], 0.0))
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def held_floored(q, k, v, rows, keys):
    """The held form over each query's keys up to its own place, its maximum floored at fp32's least finite
    value (`attention_algebra.softmax_of`'s), so a query that sees no key weighs every key 0."""
    s = jnp.where(keys[None, :] <= rows[:, None], (q @ k.T) * WIDTH ** -0.5, -jnp.inf)
    safe = jnp.maximum(jnp.max(s, axis=-1), jnp.finfo(s.dtype).min)[:, None]
    p = jnp.exp(jnp.where(s < safe, s - safe, 0.0))
    return (p @ v) / jnp.where(jnp.sum(p, axis=-1) == 0, 1.0, jnp.sum(p, axis=-1))[:, None]


def natural_exp(x):
    """An exponential of the declaration's own: `exp` by way of `exp2` (a launch's lowering of it)."""
    return jnp.exp2(x * np.float32(np.log2(np.e)))


def compensated_rescale(carried, now):
    """`exp(carried - now)` with the subtraction's rounding carried to first order (TwoSum), as attention's
    accurate form rescales a state."""
    difference, error = precision.two_sum(carried, -now)
    factor = natural_exp(difference)
    return jnp.where(jnp.isfinite(error), factor * (1.0 + error), factor)


@pytest.mark.parametrize("stated", ["held", "held floored", "declared numerics"])
def test_an_accurate_declarations_own_forms_walk_as_its_function(stated):
    """A declaration under the accurate numerics whose weights hold the largest to 1 by a select (against its
    maximum, or that maximum floored where a query may see no key), or which
    states its own exponential and rescale (`Reduction.exponential`, `rescale`: attention's accurate
    forward's), walked over every key tile by its derived step and finished: on par with the whole-operand
    function against the float64 truth (check 1)."""
    function = {"held": held, "held floored": held_floored, "declared numerics": attention}[stated]
    declaration = derive.Reduction(function, AXES, walked="keys", exponent="e",
                                   **({} if stated != "declared numerics" else
                                      {"exponential": natural_exp, "rescale": compensated_rescale}))
    forms = derive.derive(declaration, accurate=True, saturation=SATURATION)
    assert_as_exact(jax.jit(walked_whole(forms)), jax.jit(whole(function)), draw, f"{stated} walked",
                    seeds=16)


def unshifted(q, k, v, rows, keys):
    """Softmax with no maximum: no walked reduction scales the sums, each an independent sum."""
    del rows, keys
    p = jnp.exp((q @ k.T) * WIDTH ** -0.5)
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def test_a_declaration_derives_once_whatever_routes_block_its_traces_record_into():
    """A declaration's forms traced in two route-collecting blocks (`generation.routes`, a fresh collection
    JAX keys its traces on by identity): its forms derive once, its cache keyed by what changes a
    derivation (its mesh, its pricing target, the run's numerics and precision, its shapes), where keyed on
    the whole trace context every lowering derived again and the cache grew without end (row 3647)."""
    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import report

    forms = derive.derive(declared(softmax_beside_a_half_maximum), accurate=False,
                          saturation=SATURATION)
    tiles = shaped(tile_of(operands(*draw(0)), 0))
    with generation.routes():
        forms.init(tiles)
    before = len(report.rewrites())
    with generation.routes():
        forms.init(tiles)
    assert len(report.rewrites()) == before


def test_sums_no_maximum_scales_walk_as_plain_sums():
    """A declaration whose walked sums read no walked maximum derives plain sums: on par with the function."""
    forms = derive.derive(declared(unshifted), accurate=False, saturation=SATURATION)

    def small(seed):
        q, k, v = draw(seed)
        return q / jnp.max(jnp.abs(q)), k, v

    assert_as_exact(jax.jit(walked_whole(forms)), jax.jit(whole(unshifted)), small, "unshifted walked",
                    seeds=16)


def test_a_narrow_declaration_walks_as_its_whole_function():
    """Causal attention in bfloat16, its maxima narrow and its sums widened to float32 as JAX sums a narrow
    value: the derived walk on par with the whole-operand function in bfloat16 (check 1)."""
    forms = derive.derive(declared(causal), accurate=False, saturation=SATURATION)

    def narrow(seed):
        return tuple(operand.astype(jnp.bfloat16) for operand in draw(seed))

    assert_as_exact(jax.jit(walked_whole(forms)), jax.jit(whole(causal)), narrow, "bfloat16 causal walked",
                    seeds=16)


def peaks(q, k, v, rows, keys):
    """Each query's largest logit over the keys it sees, -inf for a query that sees none."""
    del v
    seen = (keys[None, :] <= rows[:, None]) & (rows[:, None] > 0)
    return jnp.max(jnp.where(seen, precision.float32_einsum("rw,kw->rk", q, k), -jnp.inf), axis=-1)


def latest(q, k, v, rows, keys):
    """Each query's latest key it sees, by place (int32), -1 for a query that sees none."""
    del q, k, v
    seen = (keys[None, :] <= rows[:, None]) & (rows[:, None] > 0)
    return jnp.max(jnp.where(seen, keys[None, :], -1), axis=-1)


@pytest.mark.parametrize("function", [peaks, latest])
def test_a_walked_maximum_is_its_monoids(function):
    """A walked maximum's value over a query that sees no key (-inf for a float, its fill for an integer)
    and over the rest, walked in key tiles: the whole-operand function's, exactly (a maximum rounds
    nothing)."""
    forms = derive.derive(declared(function), accurate=False, saturation=SATURATION)
    operands_ = draw(0)
    np.testing.assert_array_equal(walked_whole(forms)(*operands_), whole(function)(*operands_))


def floored(q, k, v, rows, keys):
    """Softmax totals shifted by each query's largest logit or its floor, whichever is larger (`floor`, an
    operand of the queries)."""
    del v, keys
    s = (q @ k.T) * WIDTH ** -0.5
    floor = -jnp.abs(rows.astype(jnp.float32))
    return jnp.sum(jnp.exp(s - jnp.maximum(jnp.max(s, axis=-1), floor)[:, None]), axis=-1)


def test_a_merge_of_a_shift_that_reads_an_operand_computes_the_declared_function():
    """A walk whose shift reads an operand of the queries beside the walked maximum, walked in two halves
    and merged: on par with the whole-operand function (check 1)."""
    forms = derive.derive(declared(floored), accurate=False, saturation=SATURATION)
    assert_as_exact(jax.jit(merged_whole(forms)), jax.jit(whole(floored)), draw, "floored merged", seeds=16)


def with_a_count(q, k, v, rows, keys):
    """Each query's logits' sum and a constant beside it, a literal output."""
    del v, rows, keys
    return jnp.sum(q @ k.T, axis=-1), np.float32(7.0)


def test_a_literal_output_is_finished_as_the_declaration_returns_it():
    """A declaration that returns a constant beside a walked sum: the walk finishes both, the constant as the
    function returns it."""
    forms = derive.derive(declared(with_a_count), accurate=False, saturation=SATURATION)
    operands_ = draw(0)
    _, count = walked_whole(forms)(*operands_)
    assert float(count) == 7.0
    assert_as_exact(jax.jit(lambda *xs: walked_whole(forms)(*xs)[0]),
                    jax.jit(lambda *xs: whole(with_a_count)(*xs)[0]), draw, "literal beside a sum", seeds=16)


def squared_shift(q, k, v, rows, keys):
    del rows, keys
    s = q @ k.T
    m = jnp.max(s, axis=-1)
    return jnp.sum((s - m[:, None]) ** 2, axis=-1)


def weights(q, k, v, rows, keys):
    del rows, keys
    s = q @ k.T
    return jnp.exp(s - jnp.max(s, axis=-1)[:, None])


def one_tile(declaration: derive.Reduction, tile: dict):
    """`declaration`'s derived walk over the one `tile`, finished."""
    forms = derive.derive(declaration, accurate=False, saturation=SATURATION)
    shapes = shaped(tile)
    return forms.finish(forms.step(forms.init(shapes), tile), tile, shapes=shapes)


ROW_ONES = np.ones((2,), np.float32)
"""A declaration's constant, an array the derivation reads as a constant of its program."""


def test_a_declarations_ranges_bound_the_operands_they_name_beside_its_constants():
    """A declaration reading an array constant, its row scale's range declared (2^-10) and its walked
    values unbounded, two of them near float32's largest: the scaled values' sum along the walk is finite
    and the walk keeps the scale inside it, as nothing bounds the values (bible §6). Read past the constant's
    place, the scale's range bounded the values, and the scale taken out of their sum overflowed it."""
    tile = {"s": jnp.full((2,), 2.0 ** -10, jnp.float32),
            "x": jnp.asarray([[3e38, 3e38, 0.0, 0.0]] * 2, jnp.float32)}

    def scaled(s, x):
        return jnp.sum((s * ROW_ONES)[:, None] * x, axis=1)

    declaration = derive.Reduction(scaled, {"s": ("rows",), "x": ("rows", "keys")}, walked="keys",
                                   exponent="e", ranges={"s": 2.0 ** -10})
    assert np.all(np.isfinite(scaled(**tile))) and np.all(np.isfinite(one_tile(declaration, tile)))


def test_a_complex_functions_bound_is_none_of_a_real_ones():
    """`sin` of a complex value is unbounded (`sin(pi/2 + 88j)` is 8.3e37, finite in complex64): a walk's sum
    of eight such sines, each scaled by its row's small float32 scale, is finite, and the walk keeps the scale
    inside its sum (bible §6), where a real sine's bound of 1 would take it out, the eight sines' sum past
    float32's largest."""
    tile = {"s": jnp.full((2,), 2.0 ** -10, jnp.float32),
            "z": jnp.full((2, 8), np.pi / 2 + 88j, jnp.complex64)}

    def scaled(s, z):
        return jnp.sum(s.astype(jnp.complex64)[:, None] * jnp.sin(z), axis=1)

    declaration = derive.Reduction(scaled, {"s": ("rows",), "z": ("rows", "keys")}, walked="keys",
                                   exponent="e")
    assert np.all(np.isfinite(scaled(**tile))) and np.all(np.isfinite(one_tile(declaration, tile)))


def test_a_rounded_values_bound_is_at_least_one():
    """`ceil` of a value declared under 2^-126 is 1: a walk's sum of such ceilings times 3e38, each scaled by
    its row's half, is finite, and the walk keeps the scale inside its sum (bible §6), where a bound of the
    operand's would take it out, the two ceilings' sum past float32's largest."""
    tile = {"s": jnp.full((2,), 0.5, jnp.float32), "x": jnp.full((2, 2), 2.0 ** -126, jnp.float32)}

    def scaled(s, x):
        return jnp.sum(s[:, None] * (jnp.ceil(x) * jnp.float32(3e38)), axis=1)

    declaration = derive.Reduction(scaled, {"s": ("rows",), "x": ("rows", "keys")}, walked="keys",
                                   exponent="e", ranges={"x": 2.0 ** -126})
    assert np.all(np.isfinite(scaled(**tile))) and np.all(np.isfinite(one_tile(declaration, tile)))


def test_an_integer_bitwise_values_bound_is_its_dtypes():
    """int32's `not(-2^31)` is `2^31 - 1`, no boolean's 1: a walk's sum of four such, converted and times
    2^95, each scaled by its row's quarter, is finite, and the walk keeps the scale inside its sum (bible
    §6), where a bound of 1 would take it out, the four terms' sum past float32's largest."""
    tile = {"s": jnp.full((2,), 0.25, jnp.float32), "x": jnp.full((2, 4), -(2 ** 31), jnp.int32)}

    def scaled(s, x):
        converted = jnp.bitwise_not(x).astype(jnp.float32)
        return jnp.sum(s[:, None] * (converted * jnp.float32(2.0 ** 95)), axis=1)

    declaration = derive.Reduction(scaled, {"s": ("rows",), "x": ("rows", "keys")}, walked="keys",
                                   exponent="e")
    assert np.all(np.isfinite(scaled(**tile))) and np.all(np.isfinite(one_tile(declaration, tile)))


@pytest.mark.parametrize("wrapped", ["converted", "negated", "subtracted"])
def test_an_unsigned_values_bound_is_its_dtypes_where_it_wraps(wrapped):
    """int32's -1 converted to uint32, uint32's 1 negated, and uint32's 0 less 1 are each `2**32 - 1`, though
    their operands are declared at 1: a walk's sum of four such, converted to float32 and times 2^94, each
    scaled by its row's quarter, is finite, and the walk keeps the scale inside its sum (bible §6), where the
    real arithmetic's bound would take it out, the four terms' sum past float32's largest."""
    signed = wrapped == "converted"
    tile = {"s": jnp.full((2,), 0.25, jnp.float32),
            "x": jnp.full((2, 4), -1 if signed else 1, jnp.int32 if signed else jnp.uint32)}

    def scaled(s, x):
        wrap = {"converted": lambda: x.astype(jnp.uint32), "negated": lambda: -x,
                "subtracted": lambda: jnp.uint32(0) - x}[wrapped]()
        return jnp.sum(s[:, None] * (wrap.astype(jnp.float32) * jnp.float32(2.0 ** 94)), axis=1)

    declaration = derive.Reduction(scaled, {"s": ("rows",), "x": ("rows", "keys")}, walked="keys",
                                   exponent="e", ranges={"x": 1.0})
    assert np.all(np.isfinite(scaled(**tile))) and np.all(np.isfinite(one_tile(declaration, tile)))


def test_a_float64_programs_reach_is_its_largest_power_of_two():
    """A float64 walk's sum of two values declared at 2^1023, each scaled by its row's half, is float64's
    2^1023, finite, and the walk keeps the scale inside its sum (bible §6): float64's reach is 2^1023, not
    the 2^1024 its largest's `log2` rounds to."""
    with jax.enable_x64(new_val=True):
        tile = {"s": jnp.full((2,), 0.5, jnp.float64), "x": jnp.full((2, 2), 2.0 ** 1023, jnp.float64)}

        def scaled(s, x):
            return jnp.sum(s[:, None] * x, axis=1)

        declaration = derive.Reduction(scaled, {"s": ("rows",), "x": ("rows", "keys")}, walked="keys",
                                       exponent="e", ranges={"x": 2.0 ** 1023})
        assert np.all(np.isfinite(scaled(**tile))) and np.all(np.isfinite(one_tile(declaration, tile)))


def test_a_walks_finish_is_its_own_whatever_walk_came_before():
    """A sum returned in its operand's dtype, walked over bfloat16 tiles and then over float32 ones, whose
    states are alike (a float32 sum each, as JAX sums a narrow value): each walk's finish its own forms', the
    bfloat16 walk's output bfloat16 as the declaration's is."""
    declaration = derive.Reduction(lambda x: jnp.sum(x, axis=0).astype(x.dtype), {"x": ("keys",)},
                                   walked="keys", exponent="e")
    forms = derive.derive(declaration, accurate=False, saturation=SATURATION)
    narrow, wide = ({"x": jnp.linspace(0.0, 1.0, 8, dtype=dtype)} for dtype in (jnp.bfloat16, jnp.float32))
    states = [forms.step(forms.init(shaped(tile)), tile) for tile in (narrow, wide)]
    for tile, state in zip((narrow, wide), states, strict=True):
        finished = forms.finish(state, tile, shapes=shaped(tile))
        assert finished.dtype == tile["x"].dtype
        np.testing.assert_array_equal(np.asarray(finished),
                                      np.asarray(jnp.sum(tile["x"]).astype(tile["x"].dtype)))


@pytest.mark.parametrize("target", [None, "v6e"], ids=["here", "a TPU"])
@pytest.mark.parametrize("function", [squared_shift, weights])
def test_a_declaration_no_monoid_holds_is_refused(function, target):
    """A walked sum that reads the walked maximum other than through its exponent, and an output that varies
    along the walked axis, are refused before any form is built, priced here or for a TPU, whose vector
    units' rate no fact states: a reading no rule rewrote costs past every form whatever the rates."""
    forms = derive.derive(declared(function), accurate=False, saturation=SATURATION)
    first = tile_of(operands(*draw(0)), 0)
    with algebra.passes(target=target), pytest.raises(derive.Refused):
        forms.init({name: (value.shape, value.dtype) for name, value in first.items()})


def backward(q, k, v, do, lse, delta):
    """Attention's gradients from its residuals (each query's log-sum-exp and its `delta`, rowsum(dO * O)):
    the weights recomputed, dV and dK summed over the queries, dQ each query's own."""
    s = (q @ k.T) * WIDTH ** -0.5
    p = jnp.exp(s - lse[:, None])
    dv = p.T @ do
    ds = p * (do @ v.T - delta[:, None])
    return ds @ k * WIDTH ** -0.5, ds.T @ q * WIDTH ** -0.5, dv


def gradients(q, k, v):
    """The arguments `backward` takes for a cotangent of ones' scale: (q, k, v, dO, lse, delta), the
    residuals from the forward on whole operands."""
    s = (q @ k.T) * WIDTH ** -0.5
    lse = jax.nn.logsumexp(s, axis=-1)
    out = jnp.exp(s - lse[:, None]) @ v
    do = jnp.cos(out)
    return q, k, v, do, lse, jnp.sum(do * out, axis=-1)


QUERY_TILE = 4


def test_a_backward_walked_over_queries_sums_dk_dv_and_maps_dq():
    """Attention's backward walked over query tiles (keys first): dK and dV the walk's sums, finished; dQ
    each tile's own (`mapped`), the tiles' joined: on par with the whole-operand gradients."""
    axes = {"q": ("rows", "width"), "k": ("keys", "width"), "v": ("keys", "values"), "do": ("rows", "values"),
            "lse": ("rows",), "delta": ("rows",)}
    forms = derive.derive(derive.Reduction(backward, axes, walked="rows", exponent="e"),
                          accurate=False, saturation=SATURATION)

    def tile(values: dict, index: int) -> dict:
        rows = slice(index * QUERY_TILE, (index + 1) * QUERY_TILE)
        return {name: value if name in ("k", "v") else value[rows] for name, value in values.items()}

    def derived(q, k, v):
        values = dict(zip(("q", "k", "v", "do", "lse", "delta"), gradients(q, k, v), strict=True))
        first = tile(values, 0)
        state = forms.init(shaped(first))
        dqs = []
        for index in range(QUERIES // QUERY_TILE):
            state = forms.step(state, tile(values, index))
            dqs.append(forms.mapped(tile(values, index))[0])
        _, dk, dv = forms.finish(state, values, shapes=shaped(first))
        return jnp.concatenate(dqs), dk, dv

    def small(seed):
        q, k, v = draw(seed)
        return q / jnp.max(jnp.abs(q), axis=-1, keepdims=True), k, v

    assert_as_exact(jax.jit(derived), jax.jit(lambda q, k, v: backward(*gradients(q, k, v))), small,
                    "backward over queries", seeds=16)


@pytest.mark.parametrize("base2", [False, True], ids=["natural", "base2"])
def test_a_keys_first_walk_derives_whatever_its_queries_and_width(base2):
    """The forward walked with its tile's keys first (`attention_algebra.declaration(keys_first=True)`)
    derives at a tile whose queries and head width differ: its weighted sums' product lays the width first
    and the queries second, and its correction broadcasts along the queries, where its anchor lies."""
    from dew.nn.kernels.templates import attention_algebra

    forms = derive.derive(attention_algebra.declaration(base2=base2, keys_first=True), accurate=False,
                          saturation=SATURATION)
    slots = forms.slots({"logits": ((32, 64), jnp.float32), "values": ((32, 128), jnp.bfloat16)})
    assert sorted(role for role, _ in slots) == ["contraction", "max", "sum"]


def sinh_total(q, k, v, rows, keys):
    """Attention's total under an equation outside the language (`sinh`)."""
    del v, rows, keys
    s = (q @ k.T) * WIDTH ** -0.5
    return jnp.sinh(jnp.sum(jnp.exp(s - jnp.max(s, axis=-1)[:, None]), axis=-1) * 1e-3)


def test_an_equation_outside_the_language_over_a_walks_total_is_refused():
    """A declaration that applies an equation outside the language to a value a walked reduction gave is
    refused: no form computes it from the walk's state, where evaluating it at the finish read the operands
    whole."""
    forms = derive.derive(declared(sinh_total), accurate=False, saturation=SATURATION)
    with pytest.raises(derive.Refused):
        walked_whole(forms)(*draw(0))
