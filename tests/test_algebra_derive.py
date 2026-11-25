"""Template forms derived from a declaration (`dew.algebra.derive`, docs/design/algebra.md §6.9): attention's
softmax over keys stated on whole operands, walked in key tiles through its derived step, the walk split in
two and merged, each held to the float64 truth (check 1) against the whole-operand function; and the
declarations no monoid along the walk holds, refused."""

import functools
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew import algebra
from dew.algebra import SATURATION, derive, forms as algebra_forms
from dew.nn import precision
from dew.parity import GateFailure

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


def walked(forms: derive.Derived, tiles: list[dict], product=None, stepped=None):
    """The state after stepping each of `tiles` (its operands' tiles) in order, from the empty state, each
    product `product`'s where given (a mainloop's issuer), each step's sums `stepped`'s where given (a
    mainloop's own step, `Derived.step`'s)."""
    state = forms.init(shaped(tiles[0]))
    for tile in tiles:
        state = forms.step(state, tile, product=product, stepped=stepped)
    return state


def stepping(kind: type, own):
    """A mainloop's own step (`Derived.step`'s `stepped`) taking each state whose parts are of `kind` by
    `own(state, part)`, every other the forms'."""
    def stepped(states, parts):
        return tuple(own(state, part) if isinstance(part, kind) else None
                     for state, part in zip(states, parts, strict=True))

    return stepped


def walked_whole(forms: derive.Derived, stepped=None):
    """The declaration's derived step over every key tile in order (each step's sums `stepped`'s where
    given), finished: a function of (q, k, v)."""
    def run(q, k, v):
        values = operands(q, k, v)
        tiles = [tile_of(values, tile) for tile in range(KEYS // TILE)]
        return forms.finish(walked(forms, tiles, stepped=stepped), values, shapes=shaped(tiles[0]))

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


KEYED_SEEDS = 16
"""A declaration closing over a typed key's declared draws against its whole function (check 1)."""


def test_a_declaration_closing_over_a_typed_key_derives():
    """A declared attention whose logits are scaled by a temperature drawn from a typed PRNG key it closes
    over (`jax.random.key`): its derivation reads each constant's magnitude, sign and dtype (`terms`), a
    key's none and JAX's own, so it derives, and its walk over the key tiles is on par with the whole
    function against the float64 truth (check 1). NumPy reads no typed key's bits, which refused every such
    declaration (row 5860)."""
    key = jax.random.key(5860)

    def biased(q, k, v, rows, keys):
        del rows, keys
        # A temperature the key draws, which moves every weight: a draw over a walked value (each key's own)
        # is outside the declaration language, which refuses it.
        s = (q @ k.T) * WIDTH ** -0.5 * jnp.exp(0.25 * jax.random.normal(key, (), q.dtype))
        p = jnp.exp(s - jnp.max(s, axis=-1)[:, None])
        return (p @ v) / jnp.sum(p, axis=-1)[:, None]

    forms = derive.derive(declared(biased), accurate=False, saturation=SATURATION)
    assert_as_exact(jax.jit(walked_whole(forms)), jax.jit(whole(biased)), draw, "a typed key's bias",
                    seeds=KEYED_SEEDS)


def test_a_walks_sums_added_by_its_mainloop_take_each_tiles_factor_and_sum():
    """A mainloop that makes a walked sum's additions itself (`Derived.step`'s `stepped`, a compensated walk's
    running sums) takes each tile's rescale factor, its tile's sum and the terms it sums (`Added`), the other
    reductions left to the forms: the terms summed and added twice over, the softmax's
    denominator doubles on every tile, and the walk's attention is the declared function's halved, on par
    against the float64 truth (check 1)."""
    forms = derive.derive(declared(attention), accurate=False, saturation=SATURATION)

    doubled = stepping(algebra_forms.Added,
                       lambda state, part: state * part.factor + 2 * jnp.sum(part.terms, -1))

    halved = jax.jit(lambda q, k, v: whole(attention)(q, k, v) / 2)
    assert_as_exact(jax.jit(walked_whole(forms, doubled)), halved, draw, "attention, its sums added twice",
                    seeds=16)


def unscaled(q, k, v, rows, keys):
    """A walked sum of local products that takes no rescale: each key's score times its value, summed."""
    del rows, keys
    return (q @ k.T) @ v


def test_a_walks_plain_sums_of_products_are_the_mainloops_to_take():
    """A walked sum of local products that takes no rescale (a backward's dQ over its key tiles) states its
    tile's weights and values to a mainloop that takes its step (`Derived.step`'s `stepped`, `Accumulated`,
    no factor): accumulated twice over, the walk's sum is the declared function's doubled, on par against the
    float64 truth (check 1)."""
    forms = derive.derive(declared(unscaled), accurate=False, saturation=SATURATION)

    doubled = stepping(algebra_forms.Accumulated,
                       lambda state, part: state + 2 * (part.weights @ part.values))

    twice = jax.jit(lambda q, k, v: 2 * whole(unscaled)(q, k, v))
    assert_as_exact(jax.jit(walked_whole(forms, doubled)), twice, draw,
                    "a plain sum of products, accumulated twice", seeds=16)


def test_a_walks_merge_by_its_mainloop_takes_each_sums_two_factors():
    """A mainloop that merges two walks' sums itself (`Derived.merge`'s `merged`, a guarded walk's join of
    shifts) takes each rescaled sum's two factors (`Merged`), the maximum left to the forms: the denominator's
    merged twice over, the merged walk's attention is the declared function's halved, on par against the
    float64 truth (check 1)."""
    forms = derive.derive(declared(attention), accurate=False, saturation=SATURATION)
    tiles = KEYS // TILE

    def run(q, k, v):
        values = operands(q, k, v)
        shapes = shaped(tile_of(values, 0))
        roles = [role for role, _ in forms.slots(shapes)]

        def doubled(firsts, seconds, parts):
            return tuple(2 * (first * part.first + second * part.second) if role == "sum" else None
                         for role, first, second, part in zip(roles, firsts, seconds, parts, strict=True))

        first = walked(forms, [tile_of(values, tile) for tile in range(tiles // 2)])
        second = walked(forms, [tile_of(values, tile) for tile in range(tiles // 2, tiles)])
        return forms.finish(forms.merge(first, second, values, shapes=shapes, merged=doubled), values,
                            shapes=shapes)

    assert_as_exact(jax.jit(run), jax.jit(lambda q, k, v: whole(attention)(q, k, v) / 2), draw,
                    "attention, its sums merged twice", seeds=16)


def held(q, k, v, rows, keys):
    """Attention whose largest logit weighs exactly 1, its exponent held to 0 by a select (the accurate
    form's weights over scaled logits, against a compiler's contraction of the scale into the subtraction)."""
    del rows, keys
    s = (q @ k.T) * WIDTH ** -0.5
    m = jnp.max(s, axis=-1)
    p = jnp.exp(jnp.where(s < m[:, None], s - m[:, None], 0.0))
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def held_floored(q, k, v, rows, keys):
    """The held form over each query's keys up to its own place, the first query seeing none, its maximum
    floored at fp32's least finite value (`attention_algebra.softmax_of`'s), so that query weighs every key
    0."""
    seen = (keys[None, :] <= rows[:, None]) & (rows[:, None] > 0)
    s = jnp.where(seen, (q @ k.T) * WIDTH ** -0.5, -jnp.inf)
    safe = jnp.maximum(jnp.max(s, axis=-1), jnp.finfo(s.dtype).min)[:, None]
    p = jnp.exp(jnp.where(s < safe, s - safe, 0.0))
    return (p @ v) / jnp.where(jnp.sum(p, axis=-1) == 0, 1.0, jnp.sum(p, axis=-1))[:, None]


def raw(q, k, v, rows, keys):
    """Attention over its raw products, the scale taken after the subtraction (the accurate form's raw
    logits: the largest's exponent exactly 0, whatever the scale)."""
    del rows, keys
    s = q @ k.T
    m = jnp.max(s, axis=-1)
    p = jnp.exp((s - m[:, None]) * SCALE)
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


SCALE = WIDTH ** -0.5


def test_a_maximum_read_only_floored_is_carried_from_its_floor():
    """A walked maximum the declaration reads only floored at the least finite value (`held_floored`'s,
    attention's `max(m, least)`) starts its walk at that floor, as the hand walk carries it, not at -inf,
    and the walk is on par with the declared function against the float64 truth (check 1), the query that
    sees no key among its rows."""
    forms = derive.derive(declared(held_floored), accurate=False, saturation=SATURATION)
    q, k, v = draw(0)
    shapes = shaped(tile_of(operands(q, k, v), 0))
    roles = [role for role, _ in forms.slots(shapes)]
    peak = forms.init(shapes)[roles.index("max")]
    np.testing.assert_array_equal(np.asarray(peak), np.full(peak.shape, np.finfo(peak.dtype).min))
    assert_as_exact(jax.jit(walked_whole(forms)), jax.jit(whole(held_floored)), draw, "floored walk",
                    seeds=16)


def placed(q, k, v, rows, keys):
    """Attention whose logits are placed on a one-device mesh mid-walk (`with_sharding_constraint`), as a
    declaration traced under a mesh places them."""
    del rows, keys
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    s = jax.lax.with_sharding_constraint((q @ k.T) * WIDTH ** -0.5,
                                         jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
    m = jnp.max(s, axis=-1)
    p = jnp.exp(s - m[:, None])
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def natural_exp(x, scale=1.0):
    """An exponential of the declaration's own, of `x` times `scale`: `exp` by way of `exp2` (a launch's
    lowering of it)."""
    return jnp.exp2(x * np.float32(scale * np.log2(np.e)))


def compensated_rescale(carried, now, scale=1.0):
    """`exp((carried - now) scale)` with the subtraction's rounding carried to first order (TwoSum), as
    attention's accurate form rescales a state."""
    difference, error = precision.two_sum(carried, -now)
    factor = natural_exp(difference, scale)
    return jnp.where(jnp.isfinite(error), factor * (1.0 + error * scale), factor)


@pytest.mark.parametrize("stated", ["held", "held floored", "declared numerics", "raw"])
def test_an_accurate_declarations_own_forms_walk_as_its_function(stated):
    """A declaration under the accurate numerics whose weights hold the largest to 1 by a select (against its
    maximum, or that maximum floored where a query may see no key), or over raw products scaled after their
    subtraction, its exponential carrying the scale, or which
    states its own exponential and rescale (`Reduction.exponential`, `rescale`: attention's accurate
    forward's), walked over every key tile by its derived step and finished: on par with the whole-operand
    function against the float64 truth (check 1)."""
    function = {"held": held, "held floored": held_floored, "declared numerics": attention,
                "raw": raw}[stated]
    numerics = {"declared numerics": {"exponential": natural_exp, "rescale": compensated_rescale},
                "raw": {"exponential": natural_exp,
                        "rescale": functools.partial(compensated_rescale, scale=SCALE)}}
    declaration = derive.Reduction(function, AXES, walked="keys", exponent="e", **numerics.get(stated, {}))
    forms = derive.derive(declaration, accurate=True, saturation=SATURATION)
    assert_as_exact(jax.jit(walked_whole(forms)), jax.jit(whole(function)), draw, f"{stated} walked",
                    seeds=16)


def test_a_declared_exponential_is_the_one_its_walk_computes_by():
    """A declaration's exponential (`Reduction.exponential`) is what its derived walk computes its weights by:
    one stated in bfloat16, its rescale in fp32 and its declaration's function computing in fp32, walks off
    par with it (check 1 refuses the walk), where a walk that took JAX's own exponential would be on par."""
    def rounded(x):
        return jnp.exp(x.astype(jnp.bfloat16)).astype(jnp.float32)

    declaration = derive.Reduction(attention, AXES, walked="keys", exponent="e", exponential=rounded,
                                   rescale=lambda carried, now: jnp.exp(carried - now))
    forms = derive.derive(declaration, accurate=True, saturation=SATURATION)
    with pytest.raises(GateFailure):
        assert_as_exact(jax.jit(walked_whole(forms)), jax.jit(whole(attention)), draw, "bfloat16 exponential",
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


REGISTERED_BETWEEN = """
import json
import sys

sys.path[:0] = ["tests"]
import jax.numpy as jnp
import test_algebra_derive as held
from egglog import eq, i64, rule, union, var

from dew.algebra import SATURATION, derive, rules
from dew.algebra.terms import Term
from dew.nn.kernels.templates import report

forms = derive.derive(held.declared(held.attention), accurate=False, saturation=SATURATION)
tiles = held.shaped(held.tile_of(held.operands(*held.draw(0)), 0))
forms.init(tiles)
first = len(report.rewrites())
forms.init(tiles)
again = len(report.rewrites())
value, root, inner, outer = var("value", Term), var("root", Term), var("inner", i64), var("outer", i64)
once = rule(eq(value.unary("abs", inner).unary("abs", outer)).to(root)).then(
    union(root).with_(value.unary("abs", inner)))
family = rules.RuleFamily("a magnitude's magnitude", (once,), (lambda x: jnp.abs(jnp.abs(x)), jnp.abs, None))
rules.register(rules.Rules.declared("registered between", "exact", (family,), "a test's"))
forms.init(tiles)
found = report.rewrites()
print(json.dumps({"first": first, "again": again, "after": len(found), "admitted": list(found[-1].admitted)}))
"""
"""A process that derives a declaration's forms at a shape twice, registers a rule set, and asks for them
again."""


def test_a_rule_set_registered_after_a_derivation_derives_the_forms_afresh():
    """A reduction's forms are kept by what their derivation reads, the rule sets its numerics admit among
    them (they saturate in its e-graph): asked again at the same shapes they are the ones kept, and after a
    set is registered they are derived afresh under it, their report naming it. They were kept by their
    mesh, target, numerics and shapes alone, so a set registered later never reached a shape derived
    before it."""
    import json
    import os
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-c", REGISTERED_BETWEEN], capture_output=True, text=True,
                          check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found["again"] == found["first"] and found["after"] == found["first"] + 1, found
    assert "registered between" in found["admitted"]


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


@pytest.mark.parametrize(("function", "walk"), [(floored, "merged"), (placed, "walked")],
                         ids=["a shift reading an operand, merged", "a walked value placed on a mesh"])
def test_a_declaration_walks_as_its_function(function, walk):
    """A walk whose shift reads an operand of the queries beside the walked maximum, walked in two halves
    and merged, and one whose declaration places a walked value on a mesh (a sharding constraint, which
    leaves its values as they are), walked: each on par with the whole-operand function (check 1)."""
    forms = derive.derive(declared(function), accurate=False, saturation=SATURATION)
    walked = {"merged": merged_whole, "walked": walked_whole}[walk](forms)
    assert_as_exact(jax.jit(walked), jax.jit(whole(function)), draw, f"{function.__name__} {walk}", seeds=16)


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
    assert forms.slots({"logits": ((32, 64), jnp.float32), "values": ((32, 128), jnp.bfloat16)})


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


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_weighted_central_moments_derive_and_match_float64_at_high_mean(mode):
    """Chan-Golub-LeVeque central-moment derivation (`derive._moment_decoupled`, `moments.chunked_moments`,
    row 9036) handles shifted high-mean data (`1e4 + N(0, 1)` where naive `E[x^2] - E[x]^2` cancels all
    float32 variance bits) on par with the declared function in float64, both sequentially and via parallel
    tree merge."""
    from dew.algebra import moments

    def draw_shifted(seed):
        rng = np.random.default_rng([9036, seed])
        x = (1e4 + rng.standard_normal((8, 96))).astype(np.float32)
        w = rng.uniform(0.25, 2.0, (8, 96)).astype(np.float32)
        return x, w

    assert_as_exact(
        jax.jit(lambda x, w: moments.chunked_moments(x, w, tile=32, mode=mode)),
        jax.jit(moments.weighted_moments),
        draw_shifted, f"weighted central moments ({mode})", seeds=16)


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_masked_central_moments_handle_empty_tiles_and_empty_rows(mode):
    """Masked unweighted variance (`moments.chunked_moments` with boolean mask) with wholly-masked tiles
    (first tile empty, middle tile empty) and an all-masked row stays finite and matches the declared
    function (check 1), with zero `0 / 0` NaNs."""
    from dew.algebra import moments

    def draw_masked(seed):
        rng = np.random.default_rng([9037, seed])
        x = (500.0 + rng.standard_normal((6, 96))).astype(np.float32)
        mask = rng.random((6, 96)) > 0.35
        mask[0, :] = False
        mask[1, :32] = False
        mask[2, 32:64] = False
        mask[3, 64:] = False
        return x, mask

    assert_as_exact(
        jax.jit(lambda x, m: moments.chunked_moments(x, m, tile=32, mode=mode)),
        jax.jit(moments.masked_moments),
        draw_masked, f"masked central moments ({mode})", seeds=16)
    x0, m0 = draw_masked(0)
    mean, var = moments.chunked_moments(x0, m0, tile=32, mode=mode)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(var))
    assert float(mean[0]) == 0.0 and float(var[0]) == 0.0


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_weighted_covariance_derives_and_merges_over_tiles(mode):
    """Bilinear central covariance (`moments.chunked_covariance`, `sum(w * (x - mu_x) * (y - mu_y))`)
    derives Chan-Golub-LeVeque step and merge forms and matches float64 truth (check 1)."""
    from dew.algebra import moments

    def draw_pair(seed):
        rng = np.random.default_rng([9038, seed])
        base = rng.standard_normal((8, 96))
        x = (1e3 + base + 0.5 * rng.standard_normal((8, 96))).astype(np.float32)
        y = (-2e3 + 0.7 * base + rng.standard_normal((8, 96))).astype(np.float32)
        w = rng.uniform(0.1, 3.0, (8, 96)).astype(np.float32)
        return x, y, w

    assert_as_exact(
        jax.jit(lambda x, y, w: moments.chunked_covariance(x, y, w, tile=32, mode=mode)),
        jax.jit(moments.weighted_covariance),
        draw_pair, f"weighted covariance ({mode})", seeds=16)


def test_central_moment_derivation_handles_keepdims_scalar_1d_and_square_op():
    """Central-moment derivation handles `keepdims=True`, 1D scalar reductions, `jnp.square`, `** 2`, and
    unguarded `s / n` declarations even when an individual tile has zero total weight."""
    def keepdims_var(x, w):
        n = jnp.sum(w, axis=-1, keepdims=True)
        mean = jnp.sum(w * x, axis=-1, keepdims=True) / n
        return jnp.sum(w * jnp.square(x - mean), axis=-1) / jnp.squeeze(n, axis=-1)

    def scalar_1d_var(x, w):
        n = jnp.sum(w, axis=0)
        mean = jnp.sum(w * x, axis=0) / n
        return jnp.sum(w * ((x - mean) ** 2), axis=0) / n

    forms_kd = derive.derive(
        derive.Reduction(keepdims_var, {"x": ("r", "k"), "w": ("r", "k")}, walked="k",
                         ranges={"w": (0.0, float("inf"))}),
        accurate=False, saturation=SATURATION)
    rng = np.random.default_rng(9039)
    x = (1e3 + rng.standard_normal((4, 64))).astype(np.float32)
    w = rng.uniform(0.5, 2.0, (4, 64)).astype(np.float32)
    w[0, :32] = 0.0
    shapes = {"x": ((4, 32), jnp.float32), "w": ((4, 32), jnp.float32)}
    t0, t1 = {"x": x[:, :32], "w": w[:, :32]}, {"x": x[:, 32:], "w": w[:, 32:]}
    st = forms_kd.step(forms_kd.step(forms_kd.init(shapes), t0), t1)
    got = np.asarray(forms_kd.finish(st, t0, shapes=shapes))
    assert np.all(np.isfinite(got))
    assert_as_exact(
        jax.jit(lambda x_, w_: forms_kd.finish(
            forms_kd.step(forms_kd.step(forms_kd.init(shapes), {"x": x_[:, :32], "w": w_[:, :32]}),
                          {"x": x_[:, 32:], "w": w_[:, 32:]}),
            {"x": x_[:, :32], "w": w_[:, :32]}, shapes=shapes)),
        jax.jit(keepdims_var),
        lambda seed: ((1e3 + np.random.default_rng(seed).standard_normal((4, 64))).astype(np.float32),
                      np.random.default_rng(seed + 100).uniform(0.5, 2.0, (4, 64)).astype(np.float32)),
        "keepdims variance", seeds=16)

    forms_1d = derive.derive(
        derive.Reduction(scalar_1d_var, {"x": ("k",), "w": ("k",)}, walked="k",
                         ranges={"w": (0.0, float("inf"))}),
        accurate=False, saturation=SATURATION)
    shapes_1d = {"x": ((32,), jnp.float32), "w": ((32,), jnp.float32)}
    assert_as_exact(
        jax.jit(lambda x_, w_: forms_1d.finish(
            forms_1d.merge(forms_1d.step(forms_1d.init(shapes_1d), {"x": x_[:32], "w": w_[:32]}),
                           forms_1d.step(forms_1d.init(shapes_1d), {"x": x_[32:], "w": w_[32:]}),
                           shapes=shapes_1d),
            {"x": x_[:32], "w": w_[:32]}, shapes=shapes_1d)),
        jax.jit(scalar_1d_var),
        lambda seed: ((1e3 + np.random.default_rng(seed).standard_normal((64,))).astype(np.float32),
                      np.random.default_rng(seed + 200).uniform(0.5, 2.0, (64,)).astype(np.float32)),
        "1d scalar variance", seeds=16)


def test_uncovered_moments_and_misaligned_mean_broadcasts_are_refused():
    """Higher central moments (third moment) and misaligned mean broadcasts on a square tile (`mean[None, :]`
    along the walked axis instead of `mean[:, None]`) are refused by `derive.derive`."""
    def third_moment(x, w):
        n = jnp.sum(w, axis=-1)
        mean = jnp.sum(w * x, axis=-1) / n
        diff = x - mean[:, None]
        return jnp.sum(w * diff * diff * diff, axis=-1) / n

    def misaligned_square(x, w):
        n = jnp.sum(w, axis=-1)
        mean = jnp.sum(w * x, axis=-1) / n
        diff = x - mean[None, :]
        return jnp.sum(w * diff * diff, axis=-1) / n

    shapes = {"x": ((8, 8), jnp.float32), "w": ((8, 8), jnp.float32)}
    for fn in (third_moment, misaligned_square):
        forms = derive.derive(
            derive.Reduction(fn, {"x": ("r", "k"), "w": ("r", "k")}, walked="k",
                             ranges={"w": (0.0, float("inf"))}),
            accurate=False, saturation=SATURATION)
        with pytest.raises(derive.Refused):
            forms.init(shapes)


def test_central_moment_derivation_requires_nonnegative_weights_and_handles_extreme_ranges():
    """Central-moment derivation refuses unproven/signed weights, accepts weights proven nonnegative in the
    declaration (`maximum(w, 0)`), and keeps finite single-tile and Chan-combine weighted moments finite and
    accurate without overflow or count-ratio underflow in either tile order."""
    from dew.algebra import moments

    unproven = derive.derive(
        derive.Reduction(moments.weighted_moments, {"x": ("r", "k"), "w": ("r", "k")}, walked="k"),
        accurate=False, saturation=SATURATION)
    with pytest.raises(derive.Refused):
        unproven.init({"x": ((2, 4), jnp.float32), "w": ((2, 4), jnp.float32)})

    def clamped_var(x, raw_w):
        w = jnp.maximum(raw_w, 0.0)
        return moments.weighted_moments(x, w)

    proven = derive.derive(
        derive.Reduction(clamped_var, {"x": ("r", "k"), "raw_w": ("r", "k")}, walked="k"),
        accurate=False, saturation=SATURATION)
    shapes = {"x": ((2, 4), jnp.float32), "raw_w": ((2, 4), jnp.float32)}
    assert proven.slots(shapes)

    # Single-tile finite moment where naive `dx * dx` overflows float32 before small weights are applied.
    x_one = np.asarray([[2.0 ** 66, -(2.0 ** 66), 0.0]], np.float32)
    w_one = np.asarray([[2.0 ** -66, 2.0 ** -66, 1.0]], np.float32)
    for mode in ("walk", "merge"):
        mean_one, var_one = moments.chunked_moments(x_one, w_one, tile=3, mode=mode)
        assert np.all(np.isfinite(mean_one)) and np.all(np.isfinite(var_one))
        assert float(var_one[0]) == 2.0 ** 67

    # Two-tile Chan combine extremes in both tile orders (large `dx` with small weight; huge weight ratio).
    cases = (
        (np.asarray([[0.0, 2.0 ** 70]], np.float32), np.asarray([[1.0, 2.0 ** -70]], np.float32), 2.0 ** 70),
        (np.asarray([[2.0 ** 70, 0.0]], np.float32), np.asarray([[2.0 ** -70, 1.0]], np.float32), 2.0 ** 70),
        (np.asarray([[2.0 ** 60, 0.0]], np.float32), np.asarray([[2.0 ** -40, 2.0 ** 120]], np.float32),
         2.0 ** -40),
        (np.asarray([[0.0, 2.0 ** 60]], np.float32), np.asarray([[2.0 ** 120, 2.0 ** -40]], np.float32),
         2.0 ** -40),
    )
    for x_c, w_c, expected_var in cases:
        for mode in ("walk", "merge"):
            mean_c, var_c = moments.chunked_moments(x_c, w_c, tile=1, mode=mode)
            assert np.all(np.isfinite(mean_c)) and np.all(np.isfinite(var_c))
            assert float(var_c[0]) == expected_var


def test_chunked_moments_contract_broadcasts_promotes_scans_and_rejects_empty():
    """`chunked_moments` rejects empty walked extents with `ValueError`, promotes mixed floating dtypes and
    broadcasts compatible weight shapes, stages `mode="walk"` through one `lax.scan` body independent of tile
    count, and handles odd tile counts in `mode="merge"`."""
    from dew.algebra import moments

    with pytest.raises(ValueError):
        moments.chunked_moments(np.zeros((2, 0), np.float32), np.ones((2, 0), np.float32), tile=4)

    def draw_mixed(seed):
        rng = np.random.default_rng([9040, seed])
        return ((1e3 + rng.standard_normal((4, 48))).astype(np.float32),
                rng.uniform(0.25, 2.0, (1, 48)).astype(np.float16))

    def ref_mixed(values, weights_f16):
        return moments.weighted_moments(values, jnp.broadcast_to(weights_f16.astype(values.dtype),
                                                                 values.shape))

    for mode in ("walk", "merge"):
        assert_as_exact(
            jax.jit(lambda values, weights_f16, kind=mode: moments.chunked_moments(
                values, weights_f16, tile=16, mode=kind)),
            jax.jit(ref_mixed), draw_mixed, f"mixed broadcast moments ({mode})", seeds=16)

    # Walk stages one scan body regardless of tile count.
    from dew.nn.programs import equations

    j2 = jax.make_jaxpr(lambda a, b: moments.chunked_moments(a, b, tile=16, mode="walk"))(
        np.ones((2, 32), np.float32), np.ones((2, 32), np.float32)).jaxpr
    j8 = jax.make_jaxpr(lambda a, b: moments.chunked_moments(a, b, tile=16, mode="walk"))(
        np.ones((2, 128), np.float32), np.ones((2, 128), np.float32)).jaxpr
    eq2, eq8 = list(equations(j2)), list(equations(j8))
    assert any(eq.primitive is jax.lax.scan_p for eq in eq2)
    assert len(eq2) == len(eq8)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_chunked_moments_handles_narrow_float16_and_bfloat16_inputs(dtype, mode):
    """Same-dtype float16 and bfloat16 inputs work in weighted, masked, and covariance chunked moments in
    both walk and merge modes, returning the input floating dtype and matching float64 truth (Check 1)."""
    from dew.algebra import moments

    def draw_weighted(seed):
        rng = np.random.default_rng([9041, seed])
        return ((10.0 + rng.standard_normal((4, 32))).astype(dtype),
                rng.uniform(0.25, 2.0, (4, 32)).astype(dtype))

    def draw_masked(seed):
        rng = np.random.default_rng([9042, seed])
        mask = rng.random((4, 32)) > 0.25
        mask[:, 0] = True
        return (10.0 + rng.standard_normal((4, 32))).astype(dtype), mask

    def draw_cov(seed):
        rng = np.random.default_rng([9043, seed])
        return ((10.0 + rng.standard_normal((4, 32))).astype(dtype),
                (-5.0 + rng.standard_normal((4, 32))).astype(dtype),
                rng.uniform(0.25, 2.0, (4, 32)).astype(dtype))

    x0, w0 = draw_weighted(0)
    mean_w, var_w = moments.chunked_moments(x0, w0, tile=8, mode=mode)
    assert mean_w.dtype == jnp.dtype(dtype) and var_w.dtype == jnp.dtype(dtype)
    assert_as_exact(
        jax.jit(lambda x_, w_, kind=mode: moments.chunked_moments(x_, w_, tile=8, mode=kind)),
        jax.jit(moments.weighted_moments), draw_weighted, f"narrow weighted {dtype} ({mode})", seeds=8)

    xm0, m0 = draw_masked(0)
    mean_m, var_m = moments.chunked_moments(xm0, m0, tile=8, mode=mode)
    assert mean_m.dtype == jnp.dtype(dtype) and var_m.dtype == jnp.dtype(dtype)
    assert_as_exact(
        jax.jit(lambda x_, m_, kind=mode: moments.chunked_moments(x_, m_, tile=8, mode=kind)),
        jax.jit(moments.masked_moments), draw_masked, f"narrow masked {dtype} ({mode})", seeds=8)

    xc0, yc0, wc0 = draw_cov(0)
    mx_c, my_c, cov_c = moments.chunked_covariance(xc0, yc0, wc0, tile=8, mode=mode)
    assert mx_c.dtype == jnp.dtype(dtype) and my_c.dtype == jnp.dtype(dtype)
    assert cov_c.dtype == jnp.dtype(dtype)
    assert_as_exact(
        jax.jit(lambda x_, y_, w_, kind=mode: moments.chunked_covariance(x_, y_, w_, tile=8, mode=kind)),
        jax.jit(moments.weighted_covariance), draw_cov, f"narrow cov {dtype} ({mode})", seeds=8)


def test_chunked_moments_preserves_zero_weight_derivatives_and_extreme_anchors():
    """Derived central moments preserve exact JVP and VJP derivatives with respect to zero weights within and
    across tiles (including multi-tile zero-mass merges and per-axis extreme cross scales) and keep
    opposite-sign extreme anchors finite and accurate."""
    from dew.algebra import moments

    for x_vals, w_vals in (([0.0, 2.0], [1.0, 0.0]), ([10.0, 12.0], [1.0, 0.0]),
                           ([12.0, 10.0], [0.0, 1.0]), ([2.0 ** 70, 2.0 ** 70], [1.0, 0.0]),
                           ([2.0 ** 70, 2.0 ** 70], [0.0, 1.0])):
        x, w, dw = (jnp.asarray([vals], jnp.float32) for vals in (x_vals, w_vals, [1.0, 1.0]))
        ref_val, ref_jvp = jax.jvp(lambda w_, x_=x: moments.weighted_moments(x_, w_), (w,), (dw,))
        ref_gm = jax.grad(lambda w_, x_=x: moments.weighted_moments(x_, w_)[0].sum())(w)
        ref_gv = jax.grad(lambda w_, x_=x: moments.weighted_moments(x_, w_)[1].sum())(w)
        for tile in (1, 2):
            for mode in ("walk", "merge"):
                def run(w_, x_=x, t=tile, m=mode):
                    return moments.chunked_moments(x_, w_, tile=t, mode=m)
                got_val, got_jvp = jax.jvp(run, (w,), (dw,))
                got_gm = jax.grad(lambda w_, f=run: f(w_)[0].sum())(w)
                got_gv = jax.grad(lambda w_, f=run: f(w_)[1].sum())(w)
                for g, r in (*zip(got_val, ref_val, strict=True), *zip(got_jvp, ref_jvp, strict=True),
                             (got_gm, ref_gm), (got_gv, ref_gv)):
                    assert np.all(np.isfinite(np.asarray(g)))
                    np.testing.assert_array_equal(np.asarray(g), np.asarray(r))

    # All-zero weights and chained zero-mass tiles: exact primal and weight gradients.
    x_zero, w_zero = jnp.asarray([[5.0, 7.0]], jnp.float32), jnp.asarray([[0.0, 0.0]], jnp.float32)
    x_three = jnp.asarray([[10.0, 20.0, 30.0]], jnp.float32)
    w_three = jnp.asarray([[0.0, 0.0, 1.0]], jnp.float32)
    ref_v3 = jax.grad(lambda w_: moments.weighted_moments(x_three, w_)[1].sum())(w_three)
    for mode in ("walk", "merge"):
        for tile in (1, 2):
            m_z, v_z = moments.chunked_moments(x_zero, w_zero, tile=tile, mode=mode)
            assert float(m_z[0]) == 0.0 and float(v_z[0]) == 0.0
            np.testing.assert_array_equal(
                np.asarray(jax.grad(lambda w_, t=tile, m=mode: moments.chunked_moments(
                    x_zero, w_, tile=t, mode=m)[0].sum())(w_zero)), np.asarray(x_zero))
            np.testing.assert_array_equal(
                np.asarray(jax.grad(lambda w_, t=tile, m=mode: moments.chunked_moments(
                    x_zero, w_, tile=t, mode=m)[1].sum())(w_zero)), np.asarray(x_zero ** 2))
        np.testing.assert_array_equal(
            np.asarray(jax.grad(lambda w_, m=mode: moments.chunked_moments(
                x_three, w_, tile=1, mode=m)[1].sum())(w_three)), np.asarray(ref_v3))

    # Large-offset, cross-scale, and >277-bit zero-mass covariance merges (final-center JVP/VJP).
    large, hi_x, lo_y = np.float32(2.0 ** 70), np.float32(2.0 ** 120), np.float32(2.0 ** -100)
    small, big, tiny = np.float32(2.0 ** -20), np.float32(2.0 ** 127), np.finfo(np.float32).tiny
    for x_row, y_row, w_row, dw_row, tile, expect_jvp in (
        ([large, 0.0, large], [large, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 1, 0.0),
        ([large, large, 0.0], [0.0, large, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1, 0.0),
        ([large, 0.0, large, large], [large, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0],
         [1.0, 0.0, 0.0, 0.0], 2, 0.0),
        ([hi_x, 0.0, small], [0.0, hi_x, small], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1, float(2.0 ** -40)),
        ([hi_x, 0.0, small], [0.0, hi_x, small], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1, 0.0),
        ([hi_x, 0.0], [lo_y, 0.0], [0.0, 0.0], [1.0, 0.0], 1, float(2.0 ** 20)),
        ([hi_x, 0.0], [lo_y, 0.0], [0.0, 0.0], [1.0, 0.0], 2, float(2.0 ** 20)),
        ([hi_x, 0.0, 0.0], [lo_y, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 1, float(2.0 ** 20)),
        ([hi_x, 0.0, hi_x], [lo_y, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 1, 0.0),
        ([big, 0.0], [tiny, 0.0], [0.0, 0.0], [1.0, 0.0], 1, 2.0),
    ):
        xl, yl, wl, dwl = (jnp.asarray([row], jnp.float32) for row in (x_row, y_row, w_row, dw_row))
        for mode in ("walk", "merge"):
            def cov_fn(w_, x_=xl, y_=yl, t=tile, m=mode):
                return moments.chunked_covariance(x_, y_, w_, tile=t, mode=m)[2]
            val_l, jvp_l = jax.jvp(cov_fn, (wl,), (dwl,))
            grad_l = jax.grad(lambda w_, f=cov_fn: f(w_).sum())(wl)
            assert float(val_l[0]) == 0.0 and float(jvp_l[0]) == expect_jvp
            assert np.all(np.isfinite(np.asarray(grad_l)))
            assert float(jnp.sum(grad_l * dwl)) == expect_jvp

    # Opposite-sign extreme anchors (`2^127 - (-2^127)`) stay finite and exact in both orders and modes.
    for x_pair, y_pair in (([-big, big], [-1.0, 1.0]), ([big, -big], [1.0, -1.0])):
        x_ext, y_ext = jnp.asarray([x_pair], jnp.float32), jnp.asarray([y_pair], jnp.float32)
        for w_pair in ([0.5, 0.5], [0.25, 0.25]):
            w_ext = jnp.asarray([w_pair], jnp.float32)
            for mode in ("walk", "merge"):
                mx, my, cov = moments.chunked_covariance(x_ext, y_ext, w_ext, tile=1, mode=mode)
                assert float(mx[0]) == 0.0 and float(my[0]) == 0.0 and float(cov[0]) == float(big)
    x_ext0, w_ext0 = jnp.asarray([[big, -big]], jnp.float32), jnp.asarray([[0.0, 1.0]], jnp.float32)
    for mode in ("walk", "merge"):
        m_e0, v_e0 = moments.chunked_moments(x_ext0, w_ext0, tile=1, mode=mode)
        assert float(m_e0[0]) == -float(big) and float(v_e0[0]) == 0.0
        dv_e0 = jax.grad(
            lambda w_, m=mode: moments.chunked_moments(x_ext0, w_, tile=1, mode=m)[1].sum())(w_ext0)
        assert float(dv_e0[0, 1]) == 0.0


def test_chunked_moments_preserves_scaled_tangent_under_large_total_weight():
    """Moment and covariance tangents stay finite and exact via `dew.algebra.ratios.ratio` across huge total
    weights (`x=[0, 2^-60]`, `w=[0, 2^100]`), underflowing centered products, zero coefficients at min-normal
    weight (`x=[0]`, `w=[2^-126]`, `dw=[4]`), mixed forward-over-reverse derivatives across weight binades
    (`w=[2^-60, 2^60]`, `dw=[2^70, -2^70]`), uniform and non-uniform shifts under underflowing recentering
    weights, and cancelling `cx*cy - cov` (row 12992)."""
    from dew.algebra import moments

    x_u = jnp.asarray([[0.0, np.float32(2.0 ** -60)]], jnp.float32)
    w_u = jnp.asarray([[0.0, np.float32(2.0 ** 100)]], jnp.float32)
    dw_u = jnp.asarray([[np.float32(2.0 ** 100), 0.0]], jnp.float32)
    x_q = jnp.asarray([[0.0, np.float32(2.0 ** -70)]], jnp.float32)
    w_q = jnp.asarray([[0.0, np.float32(2.0 ** -70)]], jnp.float32)
    dw_q = jnp.asarray([[np.float32(2.0 ** 70), 0.0]], jnp.float32)
    x_z = jnp.asarray([[0.0]], jnp.float32)
    w_z = jnp.asarray([[np.finfo(np.float32).tiny]], jnp.float32)
    dw_z = jnp.asarray([[4.0]], jnp.float32)
    x_mix = jnp.asarray([[0.0, 0.0]], jnp.float32)
    w_mix = jnp.asarray([[np.float32(2.0 ** -60), np.float32(2.0 ** 60)]], jnp.float32)
    dw_mix = jnp.asarray([[np.float32(2.0 ** 70), np.float32(-(2.0 ** 70))]], jnp.float32)
    x_vx = jnp.asarray([[np.float32(2.0 ** -30), 0.0]], jnp.float32)
    w_vx = jnp.asarray([[np.float32(2.0 ** -30), np.float32(2.0 ** 100)]], jnp.float32)
    dx_vx = jnp.asarray([[np.float32(2.0 ** 100), 0.0]], jnp.float32)
    dx_unif = jnp.asarray([[np.float32(2.0 ** 100), np.float32(2.0 ** 100)]], jnp.float32)
    dx_nonunif = jnp.asarray([[0.0, np.float32(2.0 ** 100)]], jnp.float32)
    x_cn = jnp.asarray([[np.float32(2.0 ** 50), np.float32(-(2.0 ** 50))]], jnp.float32)
    w_cn = jnp.asarray([[np.float32(2.0 ** 50), np.float32(2.0 ** 50)]], jnp.float32)
    for mode in ("walk", "merge"):
        (m_z, v_z), (dm_z, dv_z) = jax.jvp(
            lambda w_, m=mode: moments.chunked_moments(x_z, w_, tile=1, mode=m), (w_z,), (dw_z,))
        assert float(m_z[0]) == 0.0 and float(v_z[0]) == 0.0
        assert float(dm_z[0]) == 0.0 and float(dv_z[0]) == 0.0
        for tile in (1, 2):
            (m_u, v_u), (dm_u, dv_u) = jax.jvp(
                lambda w_, t=tile, m=mode: moments.chunked_moments(x_u, w_, tile=t, mode=m),
                (w_u,), (dw_u,))
            assert float(m_u[0]) == float(np.float32(2.0 ** -60)) and float(v_u[0]) == 0.0
            assert float(dm_u[0]) == float(np.float32(-(2.0 ** -60)))
            assert float(dv_u[0]) == float(np.float32(2.0 ** -120))
            _, (dmx_u, dmy_u, dc_u) = jax.jvp(
                lambda w_, t=tile, m=mode: moments.chunked_covariance(x_u, x_u, w_, tile=t, mode=m),
                (w_u,), (dw_u,))
            assert float(dmx_u[0]) == float(np.float32(-(2.0 ** -60)))
            assert float(dmy_u[0]) == float(np.float32(-(2.0 ** -60)))
            assert float(dc_u[0]) == float(np.float32(2.0 ** -120))
            _, pb_u = jax.vjp(
                lambda w_, t=tile, m=mode: moments.chunked_moments(x_u, w_, tile=t, mode=m), w_u)
            (gw_u,) = pb_u((jnp.asarray([np.float32(2.0 ** 100)], jnp.float32), jnp.zeros((1,), jnp.float32)))
            assert float(gw_u[0, 0]) == float(np.float32(-(2.0 ** -60))) and float(gw_u[0, 1]) == 0.0
            _, (_, dv_q) = jax.jvp(
                lambda w_, t=tile, m=mode: moments.chunked_moments(x_q, w_, tile=t, mode=m),
                (w_q,), (dw_q,))
            assert float(dv_q[0]) == 1.0
            _, d_gx = jax.jvp(
                lambda w_, t=tile, m=mode: jax.grad(
                    lambda x_: moments.chunked_moments(x_, w_, tile=t, mode=m)[0].sum())(x_mix),
                (w_mix,), (dw_mix,))
            np.testing.assert_array_equal(
                np.asarray(d_gx),
                np.asarray([[np.float32(2.0 ** 10), np.float32(-(2.0 ** 10))]], np.float32))
            _, (_, dv_vx) = jax.jvp(
                lambda x_, t=tile, m=mode: moments.chunked_moments(x_, w_vx, tile=t, mode=m),
                (x_vx,), (dx_vx,))
            assert float(dv_vx[0]) == float(np.float32(2.0 ** -59))
            _, (_, _, dc_unif) = jax.jvp(
                lambda x_, t=tile, m=mode: moments.chunked_covariance(x_, x_vx, w_vx, tile=t, mode=m),
                (x_mix,), (dx_unif,))
            assert float(dc_unif[0]) == 0.0
            _, (_, _, dc_nu) = jax.jvp(
                lambda x_, t=tile, m=mode: moments.chunked_covariance(x_, x_vx, w_vx, tile=t, mode=m),
                (x_mix,), (dx_nonunif,))
            assert float(dc_nu[0]) == float(np.float32(-(2.0 ** -60)))
            _, pb_cx = jax.vjp(
                lambda x_, t=tile, m=mode: moments.chunked_covariance(x_, x_vx, w_vx, tile=t, mode=m)[2],
                x_mix)
            (gx_cx,) = pb_cx(jnp.asarray([np.float32(2.0 ** 100)], jnp.float32))
            np.testing.assert_array_equal(
                np.asarray(gx_cx),
                np.asarray([[np.float32(2.0 ** -60), np.float32(-(2.0 ** -60))]], np.float32))
            _, (_, dv_cn) = jax.jvp(
                lambda w_, t=tile, m=mode: moments.chunked_moments(x_cn, w_, tile=t, mode=m),
                (w_cn,), (dw_u,))
            assert float(dv_cn[0]) == 0.0
            gw_cov = jax.grad(
                lambda w_, t=tile, m=mode: moments.chunked_covariance(x_u, x_u, w_, tile=t, mode=m)[2].sum()
            )(w_u)
            assert np.all(np.isfinite(np.asarray(gw_cov))) and float(gw_cov[0, 1]) == 0.0
            _, pb_cw = jax.vjp(
                lambda w_, t=tile, m=mode: moments.chunked_covariance(x_u, x_u, w_, tile=t, mode=m)[2], w_u)
            (gw_cw,) = pb_cw(jnp.asarray([np.float32(2.0 ** 100)], jnp.float32))
            assert float(gw_cw[0, 0]) == float(np.float32(2.0 ** -120)) and float(gw_cw[0, 1]) == 0.0


def test_chunked_moments_directed_overflow_mixed_zero_weight_and_direct_jvp():
    """Weight tangents stay finite when `cx * cy` overflows, mixed 2nd derivatives at zero individual and
    all-zero weights match the plain declaration, and direct `derive` forms preserve zero-weight JVP."""
    from dew.algebra import moments

    # Overflowing centered product (`cx * cy = 2^140`) with finite directed weight tangent (`dw = 2^-70`):
    # multiplying `dtw` before `cx * cy` stays finite and equals `2^70` (matches plain covariance JVP).
    scale_ov = np.float32(2.0 ** 70)
    x_ov = jnp.asarray([[scale_ov, 0.0]], jnp.float32)
    w_ov = jnp.asarray([[0.0, 1.0]], jnp.float32)
    dw_ov = jnp.asarray([[np.float32(2.0 ** -70), 0.0]], jnp.float32)
    for mode in ("walk", "merge"):
        _, (_, _, dc_ov) = jax.jvp(
            lambda w_, m=mode: moments.chunked_covariance(x_ov, x_ov, w_, tile=1, mode=m),
            (w_ov,), (dw_ov,))
        assert float(dc_ov[0]) == float(scale_ov)
    # Mixed 2nd-order derivatives at zero individual weights and all-zero weights match plain declaration.
    x_zm = jnp.asarray([[0.0, 2.0]], jnp.float32)
    dw_unit = jnp.asarray([[0.0, 1.0]], jnp.float32)
    for w_list in ([[1.0, 0.0]], [[0.0, 0.0]]):
        w_zm = jnp.asarray(w_list, jnp.float32)
        plain_dm_dx = jax.grad(lambda x_, w_: moments.weighted_moments(x_, w_)[0].sum(), argnums=0)
        plain_dv_dx = jax.grad(lambda x_, w_: moments.weighted_moments(x_, w_)[1].sum(), argnums=0)
        ref_hm = jax.jvp(lambda w_, fn=plain_dm_dx: fn(x_zm, w_), (w_zm,), (dw_unit,))[1]
        ref_hv = jax.jvp(lambda w_, fn=plain_dv_dx: fn(x_zm, w_), (w_zm,), (dw_unit,))[1]
        for mode in ("walk", "merge"):
            ch_dm_dx = jax.grad(
                lambda x_, w_, m=mode: moments.chunked_moments(x_, w_, tile=1, mode=m)[0].sum(), argnums=0)
            ch_dv_dx = jax.grad(
                lambda x_, w_, m=mode: moments.chunked_moments(x_, w_, tile=1, mode=m)[1].sum(), argnums=0)
            got_hm = jax.jvp(lambda w_, fn=ch_dm_dx: fn(x_zm, w_), (w_zm,), (dw_unit,))[1]
            got_hv = jax.jvp(lambda w_, fn=ch_dv_dx: fn(x_zm, w_), (w_zm,), (dw_unit,))[1]
            np.testing.assert_array_equal(np.asarray(got_hm), np.asarray(ref_hm))
            np.testing.assert_array_equal(np.asarray(got_hv), np.asarray(ref_hv))
            # Jitted value VJP with zero individual weights must stay linear in value tangents.
            got_gx_m = jax.jit(ch_dm_dx)(x_zm, w_zm)
            got_gx_v = jax.jit(ch_dv_dx)(x_zm, w_zm)
            np.testing.assert_array_equal(np.asarray(got_gx_m), np.asarray(plain_dm_dx(x_zm, w_zm)))
            np.testing.assert_array_equal(np.asarray(got_gx_v), np.asarray(plain_dv_dx(x_zm, w_zm)))

    # Direct `derive.derive` forms (without `chunked_moments` custom tangent) preserve zero-weight JVP
    # without `L^2` overflow on `x=[2^70, 2^70]`, `w=[1, 0]` (or `[0, 1]`), both via `step` and `merge`.
    forms_w = derive.derive(moments.declaration("weighted"), accurate=False, saturation=SATURATION)
    shapes_w = dict.fromkeys(("x", "w"), ((1, 1), jnp.float32))
    for x_dir, w_dir, dw_dir, exp_dm, exp_dv in (
        ([[0.0, 2.0]], [[1.0, 0.0]], [[0.0, 1.0]], 2.0, 4.0),
        ([[10.0, 12.0]], [[1.0, 0.0]], [[0.0, 1.0]], 2.0, 4.0),
        ([[float(scale_ov), float(scale_ov)]], [[1.0, 0.0]], [[0.0, 1.0]], 0.0, 0.0),
        ([[float(scale_ov), float(scale_ov)]], [[0.0, 1.0]], [[1.0, 0.0]], 0.0, 0.0),
        ([[3.0, 5.0]], [[0.0, 0.0]], [[1.0, 2.0]], 13.0, 59.0),
    ):
        xd, wd, dwd = (jnp.asarray(v, jnp.float32) for v in (x_dir, w_dir, dw_dir))
        for use_merge in (False, True):
            def direct_walk(w_, merge=use_merge, xv=xd):
                empty = forms_w.init(shapes_w)
                s0 = forms_w.step(empty, {"x": xv[:, :1], "w": w_[:, :1]})
                if merge:
                    s1 = forms_w.step(empty, {"x": xv[:, 1:], "w": w_[:, 1:]})
                    fin = forms_w.merge(s0, s1, shapes=shapes_w)
                else:
                    fin = forms_w.step(s0, {"x": xv[:, 1:], "w": w_[:, 1:]})
                return forms_w.finish(fin, {"x": xv[:, :1], "w": w_[:, :1]}, shapes=shapes_w)

            _, (dm_dir, dv_dir) = jax.jvp(direct_walk, (wd,), (dwd,))
            assert float(dm_dir[0]) == exp_dm and float(dv_dir[0]) == exp_dv
            if exp_dv == 0.0:
                gw_v = jax.grad(lambda w_, fn=direct_walk: fn(w_)[1].sum())(wd)
                np.testing.assert_array_equal(np.asarray(gw_v), np.zeros((1, 2), np.float32))


def test_chunked_moments_balances_unequal_scales_value_gradients_and_explicit_sharding():
    """Positive-mass combines and tile products balance unequal covariance scales and minimum-normal weights
    without underflow; value gradients hold Check 1 (`assert_as_exact`) and explicit sharding lowers."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    from dew.algebra import moments

    # Unequal covariance scales (`x ~ 2^100, y ~ 2^-100, w ~ 2^-100`): balanced associativity stays normal
    # and matches float64 truth even where naive float32 `(w * dx) * dy` underflows to zero.
    big_d, sml_d = np.float32(2.0 ** 100), np.float32(2.0 ** -100)
    for x_pair, y_pair, w_pair in (([0.0, big_d], [0.0, sml_d], [1.0, sml_d]),
                                   ([0.0, sml_d], [0.0, big_d], [1.0, sml_d]),
                                   ([0.0, big_d], [0.0, sml_d], [big_d, 1.0])):
        xu, yu, wu = (jnp.asarray([vals], jnp.float32) for vals in (x_pair, y_pair, w_pair))
        w64 = np.asarray(w_pair, np.float64)
        exact_cov = np.asarray(
            [(w64[0] / w64.sum()) * (w64[1] / w64.sum())
             * (float(x_pair[1]) - float(x_pair[0])) * (float(y_pair[1]) - float(y_pair[0]))],
            np.float32)
        for tile in (1, 2):
            for mode in ("walk", "merge"):
                _, _, got_cov = moments.chunked_covariance(xu, yu, wu, tile=tile, mode=mode)
                assert np.all(np.isfinite(np.asarray(got_cov))) and float(got_cov[0]) != 0.0
                np.testing.assert_array_equal(np.asarray(got_cov), exact_cov)
    # Minimum-normal float32 weights (`2^-126`): normalized monoid keeps primal and value VJPs finite/exact
    # both through `chunked_covariance` and through direct `derive.derive` without `custom_jvp`.
    tiny = float(np.finfo(np.float32).tiny)
    for x_pair, w_scale, expect_var in (([-0.5, 0.5], 2.0 ** -124, 0.25), ([-1.0, 1.0], tiny, 1.0)):
        x_tiny = jnp.asarray([x_pair], jnp.float32)
        w_tiny = jnp.asarray([[w_scale, w_scale]], jnp.float32)
        for mode in ("walk", "merge"):
            m_t, v_t = moments.chunked_moments(x_tiny, w_tiny, tile=1, mode=mode)
            assert float(m_t[0]) == 0.0 and float(v_t[0]) == expect_var

    x_mn = jnp.asarray([[0.0, 16.0]], jnp.float32)
    y_mn = jnp.asarray([[0.0, 32.0]], jnp.float32)
    w_mn = jnp.asarray([[tiny, tiny]], jnp.float32)
    forms_cov = derive.derive(moments.declaration("covariance"), accurate=False, saturation=SATURATION)
    shapes_cov = dict.fromkeys(("x", "y", "w"), ((1, 1), jnp.float32))

    def direct_cov(x_, y_, w_):
        s0 = forms_cov.step(forms_cov.init(shapes_cov), {"x": x_[:, :1], "y": y_[:, :1], "w": w_[:, :1]})
        s1 = forms_cov.step(s0, {"x": x_[:, 1:], "y": y_[:, 1:], "w": w_[:, 1:]})
        return forms_cov.finish(s1, {"x": x_[:, :1], "y": y_[:, :1], "w": w_[:, :1]},
                                shapes=shapes_cov)[2].sum()

    assert float(direct_cov(x_mn, y_mn, w_mn)) == 128.0
    dgx, dgy = jax.grad(direct_cov, argnums=(0, 1))(x_mn, y_mn, w_mn)
    np.testing.assert_array_equal(np.asarray(dgx), np.asarray([[-8.0, 8.0]], np.float32))
    np.testing.assert_array_equal(np.asarray(dgy), np.asarray([[-4.0, 4.0]], np.float32))
    for tile in (1, 2):
        for mode in ("walk", "merge"):
            gx, gy, gw = jax.grad(
                lambda x_, y_, w_, t=tile, m=mode: moments.chunked_covariance(
                    x_, y_, w_, tile=t, mode=m)[2].sum(), argnums=(0, 1, 2))(x_mn, y_mn, w_mn)
            np.testing.assert_array_equal(np.asarray(gx), np.asarray([[-8.0, 8.0]], np.float32))
            np.testing.assert_array_equal(np.asarray(gy), np.asarray([[-4.0, 4.0]], np.float32))
            np.testing.assert_array_equal(np.asarray(gw), np.asarray([[0.0, 0.0]], np.float32))

    # Ordinary value gradients (`dx`, `dy`): Check 1 (`assert_as_exact`) against the declared reductions.
    def draw_xyw(seed):
        rng = np.random.default_rng([9044, seed])
        return ((10.0 + rng.standard_normal((4, 16))).astype(np.float32),
                (-5.0 + rng.standard_normal((4, 16))).astype(np.float32),
                rng.uniform(0.25, 2.0, (4, 16)).astype(np.float32))

    for mode in ("walk", "merge"):
        assert_as_exact(
            jax.jit(jax.grad(lambda x_, y_, w_, m=mode: moments.chunked_covariance(
                x_, y_, w_, tile=4, mode=m)[2].sum(), argnums=(0, 1))),
            jax.jit(jax.grad(lambda x_, y_, w_: moments.weighted_covariance(
                x_, y_, w_)[2].sum(), argnums=(0, 1))),
            draw_xyw, f"chunked covariance value gradients ({mode})", seeds=8)

    # Explicit sharding (`AxisType.Explicit`): derived walk and merge lower cleanly under explicit meshes.
    xv = jnp.asarray([[1.0, 3.0, 5.0, 7.0]], jnp.float32)
    wv = jnp.asarray([[0.5, 1.5, 1.0, 1.0]], jnp.float32)
    mesh = jax.make_mesh((1,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:1])
    sharding = NamedSharding(mesh, P("data", None))
    with jax.set_mesh(mesh):
        xs, ws = jax.device_put(xv, sharding), jax.device_put(wv, sharding)
        for mode in ("walk", "merge"):
            ms, vs = jax.jit(lambda a, b, m=mode: moments.chunked_moments(a, b, tile=2, mode=m))(xs, ws)
            ref_ms, ref_vs = moments.weighted_moments(xv, wv)
            np.testing.assert_array_equal(np.asarray(ms), np.asarray(ref_ms))
            np.testing.assert_array_equal(np.asarray(vs), np.asarray(ref_vs))


def test_zero_mass_scaled_covariance_ad():
    """Two-band zero-mass covariance (`moment_hi`, `moment_scale`) preserves exact direct `derive` JVP/VJP
    across orthogonal, mixed-scale, multi-scale high-band, and one-sided shifted tiles."""
    from dew.algebra import moments
    forms_cov = derive.derive(moments.declaration("covariance"), accurate=False, saturation=SATURATION)

    def direct_eval(x_, y_, w_, *, tile: int, mode: str):
        shapes = dict.fromkeys(("x", "y", "w"), ((1, tile), jnp.float32))
        ntiles = x_.shape[1] // tile
        tiles = [{"x": x_[:, i * tile:(i + 1) * tile], "y": y_[:, i * tile:(i + 1) * tile],
                  "w": w_[:, i * tile:(i + 1) * tile]} for i in range(ntiles)]
        if mode == "walk":
            state = forms_cov.init(shapes)
            for part in tiles:
                state = forms_cov.step(state, part)
        else:
            empty = forms_cov.init(shapes)
            partials = [forms_cov.step(empty, part) for part in tiles]
            state = partials[0]
            for part in partials[1:]:
                state = forms_cov.merge(state, part, shapes=shapes)
        return forms_cov.finish(state, tiles[0], shapes=shapes)[2].sum()

    # Orthogonal zero-mass witness: `x = [2^70, 0], y = [0, 2^70], w = [0, 0]` has zero exact derivative in
    # both weight directions and both direct JVP and direct VJP at `tile=1` (step & merge) and `tile=2`.
    big = np.float32(2.0 ** 70)
    xo = jnp.asarray([[big, 0.0]], jnp.float32)
    yo = jnp.asarray([[0.0, big]], jnp.float32)
    wo = jnp.asarray([[0.0, 0.0]], jnp.float32)
    for tile in (1, 2):
        for mode in ("walk", "merge"):
            def fn(w_, t=tile, m=mode):
                return direct_eval(xo, yo, w_, tile=t, mode=m)

            for dw_vals in ([1.0, 0.0], [0.0, 1.0]):
                val, dval = jax.jvp(fn, (wo,), (jnp.asarray([dw_vals], jnp.float32),))
                assert float(val) == 0.0 and float(dval) == 0.0
            gw = jax.grad(fn)(wo)
            np.testing.assert_array_equal(np.asarray(gw), np.zeros((1, 2), np.float32))
            gx, gy = jax.grad(
                lambda x_, y_, t=tile, m=mode: direct_eval(x_, y_, wo, tile=t, mode=m),
                argnums=(0, 1))(xo, yo)
            np.testing.assert_array_equal(np.asarray(gx), np.zeros((1, 2), np.float32))
            np.testing.assert_array_equal(np.asarray(gy), np.zeros((1, 2), np.float32))

    # Multi-tile and same-tile witnesses: finite scaled derivatives, `2^127` sign cancel, small entries beside
    # large entries, boundary finite (`1.25*2^63`) and overflowing (`2^64`) entries beside `2^127` and `fmax`
    # with minimum-normal `dw = 2^-126`, multi-scale high-band merges, and one-sided shifts beside `±2^127`.
    huge = np.float32(2.0 ** 127)
    fmax = np.finfo(np.float32).max
    mid_big = np.float32(2.0 ** 63)
    near_ov = np.float32(1.25 * 2.0 ** 63)
    hi_ov = np.float32(2.0 ** 64)
    small = np.float32(2.0 ** -45)
    tiny = float(np.finfo(np.float32).tiny)
    for x_vals, y_vals, w_vals, dw_vals, expect in (
        ([big, 0.0, big], [big, 0.0, big], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 0.0),
        ([big, 0.0, 0.0], [big, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], 0.0),
        ([big, 0.0], [big, 0.0], [0.0, 0.0], [2.0 ** -70, 0.0], float(big)),
        ([big, 0.0], [big, 0.0], [0.0, 1.0], [2.0 ** -70, 0.0], float(big)),
        ([huge, huge], [huge, -huge], [0.0, 1.0], [1.0, 0.0], 0.0),
        ([-mid_big, mid_big, small], [-mid_big, mid_big, small], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
         float(small * small)),
        ([-huge, huge, small], [-huge, huge, small], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], float(small * small)),
        ([-mid_big, mid_big, small, 0.0], [-mid_big, mid_big, small, 0.0], [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 0.0], float(small * small)),
        ([-huge, huge, mid_big], [-huge, huge, mid_big], [0.0, 0.0, 0.0], [0.0, 0.0, tiny], 1.0),
        ([-huge, huge, near_ov], [-huge, huge, near_ov], [0.0, 0.0, 0.0], [0.0, 0.0, tiny], 1.5625),
        ([-huge, huge, hi_ov], [-huge, huge, hi_ov], [0.0, 0.0, 0.0], [0.0, 0.0, tiny], 4.0),
        ([-fmax, fmax, hi_ov], [-fmax, fmax, hi_ov], [0.0, 0.0, 0.0], [0.0, 0.0, tiny], 4.0),
        ([-huge, huge, 2.0, 1.0], [-huge, huge, 2.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, tiny, 0.0], tiny),
        ([-huge, huge, hi_ov, 0.0], [-huge, huge, hi_ov, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, tiny, 0.0],
         4.0),
    ):
        xc = jnp.asarray([x_vals], jnp.float32)
        yc = jnp.asarray([y_vals], jnp.float32)
        wc = jnp.asarray([w_vals], jnp.float32)
        dwc = jnp.asarray([dw_vals], jnp.float32)
        for mode in ("walk", "merge"):
            val, dval = jax.jvp(
                lambda w_, x_=xc, y_=yc, m=mode: direct_eval(x_, y_, w_, tile=1, mode=m), (wc,), (dwc,))
            assert float(val) == 0.0 and float(dval) == expect
    for x_row, dw_row, expect, tile in (
        ([-mid_big, mid_big, small], [0.0, 0.0, 1.0], float(small * small), 3),
        ([-huge, huge, small], [0.0, 0.0, 1.0], float(small * small), 3),
        ([-huge, huge, mid_big], [0.0, 0.0, tiny], 1.0, 3),
        ([-huge, huge, near_ov], [0.0, 0.0, tiny], 1.5625, 3),
        ([-fmax, fmax, near_ov], [0.0, 0.0, tiny], 1.5625, 3),
        ([-huge, huge, hi_ov], [0.0, 0.0, tiny], 4.0, 3),
        ([-fmax, fmax, hi_ov], [0.0, 0.0, tiny], 4.0, 3),
        ([hi_ov, -hi_ov, -huge, huge], [tiny, 0.0, 0.0, 0.0], 4.0, 2),
    ):
        xs_mix = jnp.asarray([x_row], jnp.float32)
        ws_mix = jnp.zeros((1, len(x_row)), jnp.float32)
        dws_mix = jnp.asarray([dw_row], jnp.float32)
        for mode in ("walk", "merge"):
            val, dval = jax.jvp(lambda w_, x_=xs_mix, t=tile, m=mode: direct_eval(x_, x_, w_, tile=t, mode=m),
                                (ws_mix,), (dws_mix,))
            assert float(val) == 0.0 and float(dval) == expect
    gw_mix = jax.grad(lambda w_: direct_eval(
        jnp.asarray([[-mid_big, mid_big, small]], jnp.float32),
        jnp.asarray([[-mid_big, mid_big, small]], jnp.float32), w_, tile=3, mode="walk"))(
        jnp.asarray([[0.0, 0.0, 0.0]], jnp.float32))
    np.testing.assert_array_equal(
        np.asarray(gw_mix), np.asarray([[float(mid_big * mid_big), float(mid_big * mid_big),
                                         float(small * small)]], np.float32))


def test_term_clamp_preserves_explicit_sharding():
    """`Term.clamp` under `AxisType.Explicit` matches JAX's native `clamp` placement rather than copying a
    split bound's sharding onto a replicated middle operand."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    from dew.algebra import forms, terms
    from dew.nn.sharding import explicit_spec

    lo_v = jnp.asarray([[0.0, 0.0], [0.0, 0.0]], jnp.float32)
    mid_v = jnp.asarray([[-1.0, 2.0], [0.5, 4.0]], jnp.float32)
    hi_v = jnp.asarray([[1.0, 1.0], [1.0, 1.0]], jnp.float32)
    region = terms.region(jax.make_jaxpr(jax.lax.clamp)(lo_v, mid_v, hi_v))
    out_term = region.outputs[0]
    assert out_term is not None
    mesh = jax.make_mesh((1,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:1])
    split = NamedSharding(mesh, P("data", None))
    replicated = NamedSharding(mesh, P(None, None))
    with jax.set_mesh(mesh):
        lo_s = jax.device_put(lo_v, split)
        mid_r = jax.device_put(mid_v, replicated)
        hi_r = jax.device_put(hi_v, replicated)
        ref_out = jax.jit(jax.lax.clamp)(lo_s, mid_r, hi_r)
        got_out = jax.jit(lambda a, b, c: forms.evaluated(region, out_term, [a, b, c], {}))(
            lo_s, mid_r, hi_r)
        np.testing.assert_array_equal(np.asarray(got_out), np.asarray(ref_out))
        assert explicit_spec(got_out) == explicit_spec(ref_out)
