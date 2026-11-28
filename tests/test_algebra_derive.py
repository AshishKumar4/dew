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


def drawn(rows: int, keys: int):
    """Queries, keys and values at `rows` queries and `keys` keys whose logits span six decades of magnitude
    across rows, so a row's maximum moves from tile to tile and the rescale matters: a function of a
    seed."""
    def made(seed):
        rng = np.random.default_rng(seed)
        scale = 10.0 ** rng.uniform(-1, 2, (rows, 1))
        return (jnp.asarray(rng.standard_normal((rows, WIDTH)) * scale, jnp.float32),
                jnp.asarray(rng.standard_normal((keys, WIDTH)), jnp.float32),
                jnp.asarray(rng.standard_normal((keys, WIDTH)), jnp.float32))

    return made


def draw(seed):
    """`drawn`'s at the tests' queries and keys."""
    return drawn(QUERIES, KEYS)(seed)


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
    def stepped(states, parts, _advanced):
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


def test_a_walk_reuses_its_price_across_capacities_but_not_bandwidths():
    import dataclasses

    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import report

    target = generation.gpu_target("sm90")
    constrained = dataclasses.replace(target, memory=65_283_502_899)
    slower = dataclasses.replace(constrained, memory_bandwidth=1e12)
    forms = derive.derive(declared(softmax_beside_a_half_maximum), accurate=False, saturation=SATURATION)
    tiles = shaped(tile_of(operands(*draw(0)), 0))

    with report.refusals():
        with generation.assumed(target):
            first = forms.init(tiles)
        with generation.assumed(constrained):
            second = forms.init(tiles)
        assert len(report.rewrites()) == 1
        with generation.assumed(slower):
            forms.init(tiles)
        assert len(report.rewrites()) == 2

    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(left, right)


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

import os

from dew.algebra import stored

forms = derive.derive(held.declared(held.attention), accurate=False, saturation=SATURATION)
tiles = held.shaped(held.tile_of(held.operands(*held.draw(0)), 0))
forms.init(tiles)
first = len(report.rewrites())
entries = len(os.listdir(stored.directory()))
looked = dict(report.stored())
forms.init(tiles)
again = len(report.rewrites())
value, root, inner, outer = var("value", Term), var("root", Term), var("inner", i64), var("outer", i64)
once = rule(eq(value.unary("abs", inner).unary("abs", outer)).to(root)).then(
    union(root).with_(value.unary("abs", inner)))
family = rules.RuleFamily("a magnitude's magnitude", (once,), (lambda x: jnp.abs(jnp.abs(x)), jnp.abs, None))
rules.register(rules.Rules.declared("registered between", "exact", (family,), "a test's"))
forms.init(tiles)
found = report.rewrites()
print(json.dumps({"first": first, "again": again, "after": len(found), "admitted": list(found[-1].admitted),
                  "reused": found[-1].reused, "entries": [entries, len(os.listdir(stored.directory()))],
                  "stored": [looked, dict(report.stored())]}))
"""
"""A process that derives a declaration's forms at a shape twice, registers a rule set, and asks for them
again."""


def test_a_rule_set_registered_after_a_derivation_derives_the_forms_afresh(tmp_path):
    """A reduction's forms are kept by what their derivation reads, the rule sets its numerics admit among
    them (they saturate in its e-graph): asked again at the same shapes they are the ones kept, and after a
    set is registered they are derived afresh under it, their report naming it. They were kept by their
    mesh, target, numerics and shapes alone, so a set registered later never reached a shape derived
    before it. A process outside the test session, as a run, stores its derivations (`dew.algebra.stored`):
    one under a set registered at run time is derived afresh with it and stored nowhere (`unstorable`)."""
    import json
    import os
    import subprocess
    import sys

    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XDG_CACHE_HOME": str(tmp_path)}
    done = subprocess.run([sys.executable, "-c", REGISTERED_BETWEEN], capture_output=True, text=True,
                          check=True, env=env)
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found["again"] == found["first"] and found["after"] == found["first"] + 1, found
    assert "registered between" in found["admitted"] and not found["reused"]
    before, after = found["stored"]
    assert after.get("unstorable", 0) == before.get("unstorable", 0) + 1, found
    assert after.get("miss", 0) == before.get("miss", 0) and found["entries"][0] == found["entries"][1] > 0


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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_masked_variance_ignores_nonfinite_samples_before_the_square(dtype):
    from dew.algebra import moments

    x = np.asarray([[1, 3, np.inf, -np.inf], [np.nan, np.inf, -np.inf, 1]], dtype)
    mask = np.asarray([[True, True, False, False], [False, False, False, False]])
    direction = np.asarray([[1, 0, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf]], dtype)

    def orders(x, mask, direction):
        def sum_var(value):
            return moments.masked_moments(value, mask)[1].sum()

        gradient, mixed = jax.jvp(jax.grad(sum_var), (x,), (direction,))
        return moments.masked_moments(x, mask), gradient, mixed

    (mean, variance), gradient, mixed = jax.jit(orders)(x, mask, direction)
    np.testing.assert_array_equal(mean, [2, 0])
    np.testing.assert_array_equal(variance, [1, 0])
    np.testing.assert_array_equal(gradient, [[-1, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(mixed, [[0.5, -0.5, 0, 0], [0, 0, 0, 0]])


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("mode", ["direct", "walk", "merge"])
def test_subnormal_weight_dispatch_keeps_explicit_row_sharding(mode):
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    from dew.algebra import moments

    weighted = (moments.weighted_moments if mode == "direct"
                else functools.partial(moments.chunked_moments, tile=2, mode=mode))
    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    rows, values = NamedSharding(mesh, P("data")), NamedSharding(mesh, P("data", None))
    x = np.asarray([[2.0 ** 127, 0], [1, 3], [1, 3], [1, 3]], np.float32)
    w = np.ones_like(x)
    w[0, 0] = 2.0 ** -128

    @jax.jit
    def orders(x, w):
        mean, variance = weighted(x, w)
        gradient = jax.grad(lambda x: weighted(x, w)[0].sum())(x)
        return mean, variance, gradient

    with jax.set_mesh(mesh):
        mean, variance, gradient = orders(jax.device_put(x, values), jax.device_put(w, values))
        normal_x = jax.device_put(np.broadcast_to(x[1], x.shape), values)
        normal = orders(normal_x, jax.device_put(np.ones_like(w), values))
    assert mean.sharding == variance.sharding == rows
    assert gradient.sharding == values
    np.testing.assert_array_equal(mean, [0.5, 2, 2, 2])
    np.testing.assert_array_equal(variance, [2.0 ** 126, 1, 1, 1])
    expected = np.asarray([[2.0 ** -128, 1], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], np.float32)
    np.testing.assert_array_equal(np.asarray(gradient).view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(normal[0], np.full(4, 2, np.float32))
    np.testing.assert_array_equal(normal[1], np.ones(4, np.float32))
    np.testing.assert_array_equal(normal[2], np.full(x.shape, 0.5, np.float32))


def test_weighted_subnormal_mass_broadcasts_a_scalar_observation():
    from dew.algebra import moments

    tiny = np.full(2, 2.0 ** -128, np.float32)
    run = jax.jit(moments.weighted_moments)
    mean, variance = run(np.float32(1), tiny)
    np.testing.assert_array_equal(mean, np.float32(1))
    np.testing.assert_array_equal(variance, np.float32(0))


def test_weighted_moments_keep_the_written_complex_square():
    from dew.algebra import moments

    x = np.asarray([1 + 2j, 3 + 4j], np.complex64)
    w = np.ones(2, np.float32)
    direction = np.asarray([1, 0], np.complex64)

    @jax.jit
    def evaluate(x, w, direction):
        return jax.jvp(lambda x: moments.weighted_moments(x, w), (x,), (direction,))

    primal, tangent = evaluate(x, w, direction)
    np.testing.assert_array_equal(primal, np.asarray([2 + 3j, 2j], x.dtype))
    np.testing.assert_array_equal(tangent, np.asarray([0.5, -1 - 1j], x.dtype))


@pytest.mark.parametrize("mapped", [False, True], ids=["batched", "vmapped"])
@pytest.mark.parametrize("mode", ["direct", "walk", "merge"])
def test_weighted_moments_keep_subnormal_mass_in_their_mean_variance_and_pullback(mode, mapped):
    from dew.algebra import moments

    x = np.asarray([[2.0 ** 127, 0], [1, 0], [1, 3]], np.float32)
    w = np.asarray([[2.0 ** -128, 1], [2.0 ** -128, 2.0 ** -128], [1, 1]], np.float32)

    weighted = (moments.weighted_moments if mode == "direct"
                else functools.partial(moments.chunked_moments, tile=1, mode=mode))
    if mapped:
        weighted = jax.vmap(weighted)

    @jax.jit
    def orders(x, w):
        derivatives = jax.grad(lambda x, w: weighted(x, w)[0].sum(), argnums=(0, 1))(x, w)
        return weighted(x, w), derivatives

    (mean, variance), (dx, dw) = orders(x, w)
    np.testing.assert_array_equal(mean, [0.5, 0.5, 2])
    np.testing.assert_array_equal(variance, [2.0 ** 126, 0.25, 1])
    expected_dx = np.asarray([[2.0 ** -128, 1], [0.5, 0.5], [0.5, 0.5]], np.float32)
    np.testing.assert_array_equal(np.asarray(dx).view(np.uint32), expected_dx.view(np.uint32))
    np.testing.assert_array_equal(dw, [[2.0 ** 127, -0.5], [2.0 ** 126, -(2.0 ** 126)], [-0.5, 0.5]])


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_scalar_weights_broadcast_before_subnormal_moment_dispatch(mode):
    from dew.algebra import moments

    x = np.broadcast_to(np.asarray([1, 3, 1, 3], np.float32), (2, 4))
    expected = np.full((2,), 2, np.float32), np.ones(2, np.float32)

    @jax.jit
    def evaluate(x, weight):
        def mean(value):
            return moments.chunked_moments(value, weight, tile=2, mode=mode)[0].sum()

        return moments.chunked_moments(x, weight, tile=2, mode=mode), jax.grad(mean)(x)

    for weight in (np.float32(1), np.float32(2.0 ** -128)):
        found, gradient = evaluate(x, weight)
        np.testing.assert_array_equal(found, expected)
        np.testing.assert_array_equal(gradient, np.full(x.shape, 0.25, np.float32))


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_a_zero_leading_extent_masked_moment_retains_its_shapes_and_pullback(mode):
    from dew.algebra import moments

    x = np.zeros((0, 4), np.float32)
    mask = np.ones((1, 4), np.bool_)

    @jax.jit
    def evaluate(x, mask):
        def tiled(value):
            return moments.chunked_moments(value, mask, tile=2, mode=mode)

        (mean, variance), pull = jax.vjp(tiled, x)
        return mean, variance, pull((jnp.ones_like(mean), jnp.ones_like(variance)))[0]

    mean, variance, gradient = evaluate(x, mask)
    np.testing.assert_array_equal(mean, np.zeros((0,), x.dtype))
    np.testing.assert_array_equal(variance, np.zeros((0,), x.dtype))
    np.testing.assert_array_equal(gradient, x)


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


def test_a_binary_mask_proves_weight_domain_without_a_caller_range():
    from dew.algebra import moments

    def masked_weighted(x, mask):
        return moments.weighted_moments(x, jnp.where(mask, 1.0, 0.0))

    axes = {"x": ("r", "k"), "mask": ("r", "k")}
    shapes = {"x": ((1, 4), jnp.float32), "mask": ((1, 4), jnp.bool_)}
    forms = derive.derive(derive.Reduction(masked_weighted, axes, walked="k"),
                          accurate=False, saturation=SATURATION)
    state = forms.init(shapes)
    first = {"x": np.asarray([[1, 100, 3, -100]], np.float32),
             "mask": np.asarray([[True, False, True, False]])}
    second = {"x": np.asarray([[100, 5, -100, 7]], np.float32),
              "mask": np.asarray([[False, True, False, True]])}
    joined = forms.merge(forms.step(state, first), forms.step(state, second), shapes=shapes)
    np.testing.assert_array_equal(forms.finish(joined, shapes=shapes), ([4], [5]))

    def maybe_tiny_weighted(x, mask):
        w = jnp.where(mask, np.float32(2.0 ** -128), np.float32(0))
        return moments.weighted_moments(x, w)

    tiny = derive.derive(derive.Reduction(maybe_tiny_weighted, axes, walked="k"),
                         accurate=False, saturation=SATURATION)
    with pytest.raises(derive.Refused):
        tiny.init(shapes)


def test_central_moment_derivation_checks_the_weight_domain_and_handles_extreme_ranges():
    """The public runtime handles subnormal weights; the float-mass walk requires proof they are absent."""
    from dew.algebra import moments

    axes = {"x": ("r", "k"), "w": ("r", "k")}
    shapes = {"x": ((1, 2), jnp.float32), "w": ((1, 2), jnp.float32)}
    tiny = np.full((1, 2), 2.0 ** -128, np.float32)
    values = np.asarray([[1, 0]], np.float32)
    np.testing.assert_array_equal(jax.jit(moments.weighted_moments)(values, tiny), ([0.5], [0.25]))
    for ranges in ({}, {"w": (0.0, 4.0)}):
        unproven = derive.derive(derive.Reduction(moments.weighted_moments, axes, walked="k", ranges=ranges),
                                 accurate=False, saturation=SATURATION)
        with pytest.raises(derive.Refused):
            unproven.init(shapes)

    def clamped_var(x, raw_w):
        w = jnp.maximum(raw_w, 0.0)
        return moments.weighted_moments(x, w)

    clamped = {"x": ("r", "k"), "raw_w": ("r", "k")}
    declared = derive.Reduction(clamped_var, clamped, walked="k", ranges={"raw_w": (0.25, 4.0)})
    proven = derive.derive(declared, accurate=False, saturation=SATURATION)
    shapes = {"x": ((1, 2), jnp.float32), "raw_w": ((1, 2), jnp.float32)}
    weights = np.ones((1, 2), np.float32)
    first = proven.step(proven.init(shapes), {"x": np.asarray([[1, 3]], np.float32), "raw_w": weights})
    second = proven.step(proven.init(shapes), {"x": np.asarray([[5, 7]], np.float32), "raw_w": weights})
    joined = proven.merge(first, second, shapes=shapes)
    np.testing.assert_array_equal(proven.finish(joined, shapes=shapes), ([4], [5]))

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


def test_direct_weighted_broadcast_keeps_marginal_shapes_and_counts_each_weight_once():
    from dew.algebra import moments

    x = np.asarray([1, 3], np.float32)
    y = np.asarray([[2, 6], [3, 9]], np.float32)
    run = jax.jit(moments.weighted_covariance)
    mean_x, mean_y, covariance = run(x, y, np.ones(2, np.float32))
    np.testing.assert_array_equal(mean_x, np.asarray(2, np.float32))
    np.testing.assert_array_equal(mean_y, [4, 6])
    np.testing.assert_array_equal(covariance, [2, 3])

    mean_x, mean_y, covariance = run(x, y, np.ones(1, np.float32))
    np.testing.assert_array_equal(mean_x, np.asarray(4, np.float32))
    np.testing.assert_array_equal(mean_y, [8, 12])
    np.testing.assert_array_equal(covariance, [20, 30])
    mean, variance = jax.jit(moments.weighted_moments)(x, np.ones(1, np.float32))
    np.testing.assert_array_equal((mean, variance), (4, 10))

    y[1, 1] = np.inf
    w = np.asarray([1, 0], np.float32)
    (mean_x, mean_y, covariance), (_, _, moved) = jax.jit(
        lambda w, dw: jax.jvp(lambda weights: moments.weighted_covariance(x, y, weights), (w,), (dw,)))(
            w, np.asarray([0, 1], np.float32))
    np.testing.assert_array_equal(mean_x, np.asarray(1, np.float32))
    np.testing.assert_array_equal(mean_y, [2, 3])
    np.testing.assert_array_equal(covariance, [0, 0])
    np.testing.assert_array_equal(moved, [8, 0])


@pytest.mark.parametrize("singleton", [False, True], ids=["flat", "broadcast"])
def test_direct_covariance_keeps_each_marginals_own_zero_weight_direction(singleton):
    from dew.algebra import moments

    x = np.asarray([1, 3], np.float32)
    y = np.asarray([2, np.inf], np.float32)
    if singleton:
        y = y[None]
    weights = np.asarray([1, 0], np.float32)
    direction = np.asarray([0, 1], np.float32)

    @jax.jit
    def evaluate(x, y, w, dw):
        def covariance(w):
            return moments.weighted_covariance(x, y, w)

        primal, tangent = jax.jvp(covariance, (w,), (dw,))
        gradient = jax.grad(lambda w: covariance(w)[2].sum())
        _, second = jax.jvp(gradient, (w,), (dw,))
        return primal, tangent, second

    (mean_x, mean_y, cov), (dx, dy, dc), second = evaluate(x, y, weights, direction)
    shape = y.shape[:-1]
    np.testing.assert_array_equal(mean_x, np.asarray(1, np.float32))
    np.testing.assert_array_equal(dx, np.asarray(2, np.float32))
    np.testing.assert_array_equal(mean_y, np.full(shape, 2, np.float32))
    for part in (cov, dy, dc):
        np.testing.assert_array_equal(part, np.zeros(shape, np.float32))
    np.testing.assert_array_equal(second, np.zeros_like(weights))


@pytest.mark.parametrize("mode", ["direct", "walk", "merge"])
def test_weighted_moments_exclude_nonfinite_zero_weight_samples_without_losing_finite_directions(mode):
    from dew.algebra import moments

    x = np.asarray([1, 3, 5, np.inf], np.float32)
    y = np.asarray([2, 6, 10, np.nan], np.float32)
    weights = np.asarray([1, 1, 0, 0], np.float32)
    direction = np.asarray([0, 0, 1, 2.0 ** 127], np.float32)

    def variance(x, w):
        if mode == "direct":
            return moments.weighted_moments(x, w)
        return moments.chunked_moments(x, w, tile=2, mode=mode)

    def covariance(x, y, w):
        if mode == "direct":
            return moments.weighted_covariance(x, y, w)
        return moments.chunked_covariance(x, y, w, tile=2, mode=mode)

    @jax.jit
    def variance_orders(x, w, dw):
        def value(weights):
            return variance(x, weights)[1]

        first, second = jax.jvp(jax.grad(value), (w,), (dw,))
        return variance(x, w), first, second

    @jax.jit
    def covariance_orders(x, y, w, dw):
        return jax.jvp(lambda weights: covariance(x, y, weights), (w,), (dw,))

    value, first, second = variance_orders(x, weights, direction)
    np.testing.assert_array_equal(value, [2, 1])
    np.testing.assert_array_equal(first, [0, 0, 4, 0])
    np.testing.assert_array_equal(second, [-0.5, -3.5, -8.5, 0])
    cov, moved = covariance_orders(x, y, weights, direction)
    np.testing.assert_array_equal(cov, [2, 4, 2])
    np.testing.assert_array_equal(moved, [1.5, 3, 8])


def _moment_weight_order(function, projection):
    if projection is None:
        def forward(x, w, dw):
            return jax.jvp(lambda weights: function(x, weights), (w,), (dw,))

        return jax.jit(forward)

    # Discard the other result before reverse AD; a materialized zero would exercise another path.
    def projected(x, w, dw):
        return function(x, w)[projection].sum()

    return jax.jit(jax.grad(projected, argnums=1))


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
@pytest.mark.parametrize("projection", [None, 0, 1], ids=["weight-jvp", "mean-vjp", "variance-vjp"])
def test_chunked_moments_preserves_zero_weight_derivatives(mode, tile, projection):
    from dew.algebra import moments

    run = _moment_weight_order(functools.partial(moments.chunked_moments, tile=tile, mode=mode), projection)
    plain = _moment_weight_order(moments.weighted_moments, projection)
    dw = jnp.ones((1, 2), jnp.float32)
    for x_vals, w_vals in (([0.0, 2.0], [1.0, 0.0]), ([10.0, 12.0], [1.0, 0.0]),
                           ([12.0, 10.0], [0.0, 1.0]), ([2.0 ** 70, 2.0 ** 70], [1.0, 0.0]),
                           ([2.0 ** 70, 2.0 ** 70], [0.0, 1.0])):
        x, w = (jnp.asarray([values], jnp.float32) for values in (x_vals, w_vals))
        actual, expected = jax.tree.leaves(run(x, w, dw)), jax.tree.leaves(plain(x, w, dw))
        for found, want in zip(actual, expected, strict=True):
            assert np.all(np.isfinite(np.asarray(found)))
            np.testing.assert_array_equal(found, want)

    x_zero = jnp.asarray([[5.0, 7.0]], jnp.float32)
    zero = run(x_zero, jnp.zeros_like(x_zero), dw)
    if projection is None:
        (mean, variance), _ = zero
        assert float(mean[0]) == 0.0 and float(variance[0]) == 0.0
    else:
        np.testing.assert_array_equal(zero, x_zero if projection == 0 else x_zero ** 2)


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("anchor", ["zero-prefix", "opposite-sign-jvp", "opposite-sign-vjp"])
def test_chunked_moments_preserves_extreme_anchors(mode, anchor):
    from dew.algebra import moments

    run = _moment_weight_order(functools.partial(moments.chunked_moments, tile=1, mode=mode),
                               None if anchor == "opposite-sign-jvp" else 1)
    if anchor == "zero-prefix":
        x = jnp.asarray([[10.0, 20.0, 30.0]], jnp.float32)
        w = jnp.asarray([[0.0, 0.0, 1.0]], jnp.float32)
        dw = jnp.ones_like(w)
        plain = _moment_weight_order(moments.weighted_moments, 1)
        np.testing.assert_array_equal(run(x, w, dw), plain(x, w, dw))
        return

    big = np.float32(2.0 ** 127)
    x, w = jnp.asarray([[big, -big]], jnp.float32), jnp.asarray([[0.0, 1.0]], jnp.float32)
    found = run(x, w, jnp.ones_like(w))
    if anchor == "opposite-sign-jvp":
        (mean, variance), _ = found
        assert float(mean[0]) == -float(big) and float(variance[0]) == 0.0
    else:
        assert float(found[0, 1]) == 0.0


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_chunked_covariance_preserves_zero_weight_derivatives_and_extreme_anchors(mode):
    """Keep the directional covariance and its transpose through zero-mass tiles and exponent gaps."""
    from dew.algebra import moments

    def covariance(x, y, w, *, tile):
        return moments.chunked_covariance(x, y, w, tile=tile, mode=mode)[2]

    def forward(x, y, w, dw, *, tile):
        return jax.jvp(lambda weights: covariance(x, y, weights, tile=tile), (w,), (dw,))

    def total(x, y, w, *, tile):
        return covariance(x, y, w, tile=tile).sum()

    evaluate = jax.jit(forward, static_argnames=("tile",))
    pullback = jax.jit(jax.grad(total, argnums=2), static_argnames=("tile",))
    states = jax.jit(functools.partial(moments.chunked_covariance, tile=1, mode=mode))

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
        val_l, jvp_l = evaluate(xl, yl, wl, dwl, tile=tile)
        grad_l = pullback(xl, yl, wl, tile=tile)
        assert float(val_l[0]) == 0.0 and float(jvp_l[0]) == expect_jvp
        assert np.all(np.isfinite(np.asarray(grad_l)))
        assert float(jnp.sum(grad_l * dwl)) == expect_jvp

    # Opposite-sign extreme anchors (`2^127 - (-2^127)`) stay finite and exact in both orders and modes.
    for x_pair, y_pair in (([-big, big], [-1.0, 1.0]), ([big, -big], [1.0, -1.0])):
        x_ext, y_ext = jnp.asarray([x_pair], jnp.float32), jnp.asarray([y_pair], jnp.float32)
        for w_pair in ([0.5, 0.5], [0.25, 0.25]):
            w_ext = jnp.asarray([w_pair], jnp.float32)
            mx, my, cov = states(x_ext, y_ext, w_ext)
            assert float(mx[0]) == 0.0 and float(my[0]) == 0.0 and float(cov[0]) == float(big)


def _moment_weight_jvp(mode, tile):
    from dew.algebra import moments

    @jax.jit
    def directed(x, w, dw):
        return jax.jvp(lambda weights: moments.chunked_moments(x, weights, tile=tile, mode=mode), (w,), (dw,))

    return directed


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_moments_preserves_scaled_tangent_under_large_total_weight(mode, tile):
    directed = _moment_weight_jvp(mode, tile)

    x_u = jnp.asarray([[0.0, np.float32(2.0 ** -60)]], jnp.float32)
    w_u = jnp.asarray([[0.0, np.float32(2.0 ** 100)]], jnp.float32)
    dw_u = jnp.asarray([[np.float32(2.0 ** 100), 0.0]], jnp.float32)
    x_q = jnp.asarray([[0.0, np.float32(2.0 ** -70)]], jnp.float32)
    w_q = jnp.asarray([[0.0, np.float32(2.0 ** -70)]], jnp.float32)
    dw_q = jnp.asarray([[np.float32(2.0 ** 70), 0.0]], jnp.float32)
    x_ovq = jnp.asarray([[np.float32(1.5 * 2.0 ** 127), 0.0]], jnp.float32)
    dw_ovq = jnp.asarray([[np.float32(2.0 ** -100), 0.0]], jnp.float32)
    x_sub = jnp.asarray([[np.finfo(np.float32).tiny, 0.0]], jnp.float32)
    w_sub = jnp.asarray([[0.0, np.float32(1.5 * np.finfo(np.float32).tiny)]], jnp.float32)
    dw_sub = jnp.asarray([[1.0, 0.0]], jnp.float32)
    x_cn = jnp.asarray([[np.float32(2.0 ** 50), np.float32(-(2.0 ** 50))]], jnp.float32)
    w_cn = jnp.asarray([[np.float32(2.0 ** 50), np.float32(2.0 ** 50)]], jnp.float32)

    (m_u, v_u), (dm_u, dv_u) = directed(x_u, w_u, dw_u)
    assert float(m_u[0]) == float(np.float32(2.0 ** -60)) and float(v_u[0]) == 0.0
    assert float(dm_u[0]) == float(np.float32(-(2.0 ** -60)))
    assert float(dv_u[0]) == float(np.float32(2.0 ** -120))

    _, (_, dv_q) = directed(x_q, w_q, dw_q)
    assert float(dv_q[0]) == 1.0

    _, (_, dv_ovq) = directed(x_ovq, w_u, dw_ovq)
    assert float(dv_ovq[0]) == float(np.float32(2.25 * 2.0 ** 54))

    _, (dm_sub, _) = directed(x_sub, w_sub, dw_sub)
    assert float(dm_sub[0]) == float(np.float32(2.0 / 3.0))
    _, (_, dv_cn) = directed(x_cn, w_cn, dw_u)
    assert float(dv_cn[0]) == 0.0


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_a_minimum_normal_weight_keeps_zero_moment_tangents(mode):
    directed = _moment_weight_jvp(mode, 1)

    x_z = jnp.asarray([[0.0]], jnp.float32)
    w_z = jnp.asarray([[np.finfo(np.float32).tiny]], jnp.float32)
    dw_z = jnp.asarray([[4.0]], jnp.float32)

    (m_z, v_z), (dm_z, dv_z) = directed(x_z, w_z, dw_z)
    assert float(m_z[0]) == 0.0 and float(v_z[0]) == 0.0
    assert float(dm_z[0]) == 0.0 and float(dv_z[0]) == 0.0


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_covariance_keeps_its_large_weight_tangents(mode, tile):
    from dew.algebra import moments

    @jax.jit
    def directed_covariance(x, y, w, dw):
        return jax.jvp(lambda weights: moments.chunked_covariance(x, y, weights, tile=tile, mode=mode),
                       (w,), (dw,))

    x_u = jnp.asarray([[0.0, np.float32(2.0 ** -60)]], jnp.float32)
    w_u = jnp.asarray([[0.0, np.float32(2.0 ** 100)]], jnp.float32)
    dw_u = jnp.asarray([[np.float32(2.0 ** 100), 0.0]], jnp.float32)
    x_ovq = jnp.asarray([[np.float32(1.5 * 2.0 ** 127), 0.0]], jnp.float32)
    x_sub = jnp.asarray([[np.finfo(np.float32).tiny, 0.0]], jnp.float32)

    _, (dmx_u, dmy_u, dc_u) = directed_covariance(x_u, x_u, w_u, dw_u)
    assert float(dmx_u[0]) == float(np.float32(-(2.0 ** -60)))
    assert float(dmy_u[0]) == float(np.float32(-(2.0 ** -60)))
    assert float(dc_u[0]) == float(np.float32(2.0 ** -120))

    _, (_, _, dc_ovsub) = directed_covariance(x_ovq, x_sub, w_u, dw_u)
    assert float(dc_ovsub[0]) == float(np.float32(3.0))


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_moments_preserve_the_large_mean_cotangent(mode, tile):
    from dew.algebra import moments

    x_u = jnp.asarray([[0.0, np.float32(2.0 ** -60)]], jnp.float32)
    w_u = jnp.asarray([[0.0, np.float32(2.0 ** 100)]], jnp.float32)

    _, pb_u = jax.vjp(
        lambda w_, t=tile, m=mode: moments.chunked_moments(x_u, w_, tile=t, mode=m), w_u)
    (gw_u,) = pb_u((jnp.asarray([np.float32(2.0 ** 100)], jnp.float32), jnp.zeros((1,), jnp.float32)))
    assert float(gw_u[0, 0]) == float(np.float32(-(2.0 ** -60))) and float(gw_u[0, 1]) == 0.0


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_moments_move_value_cotangents_along_weights(mode, tile):
    from dew.algebra import moments

    x_mix = jnp.asarray([[0.0, 0.0]], jnp.float32)
    w_mix = jnp.asarray([[np.float32(2.0 ** -60), np.float32(2.0 ** 60)]], jnp.float32)
    dw_mix = jnp.asarray([[np.float32(2.0 ** 70), np.float32(-(2.0 ** 70))]], jnp.float32)

    _, d_gx = jax.jvp(
        lambda w_, t=tile, m=mode: jax.grad(
            lambda x_: moments.chunked_moments(x_, w_, tile=t, mode=m)[0].sum())(x_mix),
        (w_mix,), (dw_mix,))
    np.testing.assert_array_equal(
        np.asarray(d_gx),
        np.asarray([[np.float32(2.0 ** 10), np.float32(-(2.0 ** 10))]], np.float32))


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_covariance_cancels_a_uniform_value_direction(mode, tile):
    from dew.algebra import moments

    x_mix = jnp.asarray([[0.0, 0.0]], jnp.float32)
    x_vx = jnp.asarray([[np.float32(2.0 ** -30), 0.0]], jnp.float32)
    w_vx = jnp.asarray([[np.float32(2.0 ** -30), np.float32(2.0 ** 100)]], jnp.float32)
    dx_unif = jnp.asarray([[np.float32(2.0 ** 100), np.float32(2.0 ** 100)]], jnp.float32)

    _, (_, _, dc_unif) = jax.jvp(
        lambda x_, t=tile, m=mode: moments.chunked_covariance(x_, x_vx, w_vx, tile=t, mode=m),
        (x_mix,), (dx_unif,))
    assert float(dc_unif[0]) == 0.0


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_moments_move_weight_cotangents_along_values(mode, tile):
    from dew.algebra import moments

    x_u = jnp.asarray([[0.0, np.float32(2.0 ** -60)]], jnp.float32)
    w_gw = jnp.asarray([[0.0, np.float32(2.0 ** 60)]], jnp.float32)
    dx_gw = jnp.asarray([[np.float32(2.0 ** 70), 0.0]], jnp.float32)

    _, d_gw = jax.jvp(
        lambda x_, t=tile, m=mode: jax.grad(
            lambda w_: moments.chunked_moments(x_, w_, tile=t, mode=m)[0].sum())(w_gw),
        (x_u,), (dx_gw,))
    np.testing.assert_array_equal(
        np.asarray(d_gw), np.asarray([[np.float32(2.0 ** 10), 0.0]], np.float32))


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_covariance_keeps_weight_cotangents_finite(mode, tile):
    from dew.algebra import moments

    x_u = jnp.asarray([[0.0, np.float32(2.0 ** -60)]], jnp.float32)
    w_u = jnp.asarray([[0.0, np.float32(2.0 ** 100)]], jnp.float32)

    gw_cov = jax.grad(
        lambda w_, t=tile, m=mode: moments.chunked_covariance(x_u, x_u, w_, tile=t, mode=m)[2].sum()
    )(w_u)
    assert np.all(np.isfinite(np.asarray(gw_cov))) and float(gw_cov[0, 1]) == 0.0
    _, pb_cw = jax.vjp(
        lambda w_, t=tile, m=mode: moments.chunked_covariance(x_u, x_u, w_, tile=t, mode=m)[2], w_u)
    (gw_cw,) = pb_cw(jnp.asarray([np.float32(2.0 ** 100)], jnp.float32))
    assert float(gw_cw[0, 0]) == float(np.float32(2.0 ** -120)) and float(gw_cw[0, 1]) == 0.0


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_moments_balanced_quotient_direction_jvp(mode, tile):
    directed = _moment_weight_jvp(mode, tile)

    x_jvp = jnp.asarray([[1.0, 0.0]], jnp.float32)
    w_lo = jnp.asarray([[0.0, np.float32(1.5 * 2.0 ** -100)]], jnp.float32)
    dw_hi = jnp.asarray([[np.float32(2.0 ** 28), 0.0]], jnp.float32)
    x_tiny = jnp.asarray([[np.finfo(np.float32).tiny, 0.0]], jnp.float32)
    w_mid = jnp.asarray([[0.0, np.float32(1.5 * 2.0 ** 10)]], jnp.float32)
    dw_mid = jnp.asarray([[np.float32(2.0 ** 20), 0.0]], jnp.float32)
    a_st = np.float32(1.0 + 2.0 ** -23)
    x_st = jnp.asarray([[a_st, -a_st]], jnp.float32)
    w_st = jnp.asarray([[0.5, 0.5]], jnp.float32)
    dw_st = jnp.asarray([[np.float32(2.0 ** 100), 0.0]], jnp.float32)
    a_hi = np.float32((1.5 + 2.0 ** -22) * 2.0 ** 63)
    x_hi_st = jnp.asarray([[a_hi, -a_hi]], jnp.float32)
    dw_one = jnp.asarray([[1.0, 0.0]], jnp.float32)
    a_lo = np.float32(2.0 ** -80)
    x_lo_st = jnp.asarray([[a_lo, -a_lo]], jnp.float32)
    x_unf = jnp.asarray([[0.0, a_lo]], jnp.float32)
    w_unf = jnp.asarray([[0.0, a_lo]], jnp.float32)
    dw_unf = jnp.asarray([[np.float32(2.0 ** 80), 0.0]], jnp.float32)

    _, (dm, _) = directed(x_jvp, w_lo, dw_hi)
    assert float(dm[0]) == float(np.float32((2.0 / 3.0) * 2.0 ** 128))
    _, (dm_t, _) = directed(x_tiny, w_mid, dw_mid)
    assert float(dm_t[0]) == float(np.float32((2.0 / 3.0) * 2.0 ** -116))

    _, (_, dv_st) = directed(x_st, w_st, dw_st)
    assert float(dv_st[0]) == 0.0
    _, (_, dv_hi_st) = directed(x_hi_st, w_st, dw_one)
    assert float(dv_hi_st[0]) == 0.0
    _, (_, dv_lo_st) = directed(x_lo_st, w_st, dw_st)
    assert float(dv_lo_st[0]) == 0.0
    _, (_, dv_unf) = directed(x_unf, w_unf, dw_unf)
    assert float(dv_unf[0]) == 1.0


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_moments_balanced_quotient_cotangent(mode, tile):
    from dew.algebra import moments

    x_vjp = jnp.asarray([[np.float32(1.75), 0.0]], jnp.float32)
    w_hi = jnp.asarray([[0.0, np.float32(2.0 ** 100)]], jnp.float32)
    cot_hi = jnp.asarray([np.float32(1.5 * 2.0 ** 127)], jnp.float32)

    _, pb = jax.vjp(
        lambda w_, t=tile, m=mode: moments.chunked_moments(x_vjp, w_, tile=t, mode=m)[0], w_hi)
    (gw,) = pb(cot_hi)
    assert float(gw[0, 0]) == float(np.float32(2.625 * 2.0 ** 27)) and float(gw[0, 1]) == 0.0


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_mean_balances_mixed_value_weight_derivatives(mode, tile):
    from dew.algebra import moments

    @jax.jit
    def moving_weight_pullback(x, w, dx):
        def weights_gradient(x):
            def mean(weights):
                return moments.chunked_moments(x, weights, tile=tile, mode=mode)[0].sum()

            return jax.grad(mean)(w)

        return jax.jvp(weights_gradient, (x,), (dx,))[1]

    x_tiny = jnp.asarray([[np.finfo(np.float32).tiny, 0.0]], jnp.float32)
    w_mid = jnp.asarray([[0.0, np.float32(1.5 * 2.0 ** 10)]], jnp.float32)
    w_p2 = jnp.asarray([[0.0, np.float32(2.0 ** 10)]], jnp.float32)
    dx_ov = jnp.asarray([[np.float32(1.5 * 2.0 ** 127), 0.0]], jnp.float32)
    x_qu = jnp.asarray([[np.float32(2.0 ** -70), 0.0]], jnp.float32)
    w_up = jnp.asarray([[0.0, np.float32(1.5 * 2.0 ** -70)]], jnp.float32)
    dx_up = jnp.asarray([[np.float32(2.25 * 2.0 ** 57), 0.0]], jnp.float32)

    for w_case, expected in ((w_p2, 1.5 * 2.0 ** 117), (w_mid, 2.0 ** 117)):
        d_gw = moving_weight_pullback(x_tiny, w_case, dx_ov)
        np.testing.assert_array_equal(
            np.asarray(d_gw), np.asarray([[np.float32(expected), 0.0]], np.float32))

    d_gup = moving_weight_pullback(x_qu, w_up, dx_up)
    np.testing.assert_array_equal(
        np.asarray(d_gup), np.asarray([[np.float32(1.5 * 2.0 ** 127), 0.0]], np.float32))


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_covariance_balances_mixed_value_weight_derivatives(mode, tile):
    from dew.algebra import moments

    x_qu = jnp.asarray([[np.float32(2.0 ** -70), 0.0]], jnp.float32)
    y_qu = jnp.asarray([[np.float32(2.0 ** -71), 0.0]], jnp.float32)
    w_qu = jnp.asarray([[0.0, 1.0]], jnp.float32)
    dx_qu = jnp.asarray([[np.float32(2.0 ** 70), 0.0]], jnp.float32)

    _, d_gcov = jax.jvp(
        lambda x_, t=tile, m=mode: jax.grad(
            lambda w__: moments.chunked_covariance(x_, y_qu, w__, tile=t, mode=m)[2].sum())(w_qu),
        (x_qu,), (dx_qu,))
    np.testing.assert_array_equal(
        np.asarray(d_gcov), np.asarray([[np.float32(0.5), 0.0]], np.float32))


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_mean_balances_higher_weight_derivatives(mode, tile):
    from dew.algebra import moments

    x_ww = jnp.asarray([[np.float32(2.0 ** -100), 0.0]], jnp.float32)
    w_ww = jnp.asarray([[0.0, np.float32(2.0 ** -40)]], jnp.float32)
    dw_ww = jnp.asarray([[np.float32(2.0 ** 99), np.float32(2.0 ** 99)]], jnp.float32)

    _, d_gww = jax.jvp(
        lambda w_, t=tile, m=mode: jax.grad(
            lambda w__: moments.chunked_moments(x_ww, w__, tile=t, mode=m)[0].sum())(w_),
        (w_ww,), (dw_ww,))
    np.testing.assert_array_equal(
        np.asarray(d_gww),
        np.asarray([[np.float32(-1.5 * 2.0 ** 80), np.float32(-(2.0 ** 79))]], np.float32))


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 2])
def test_chunked_variance_cancels_joint_value_weight_directions(mode, tile):
    from dew.algebra import moments

    x_jt = jnp.asarray([[-8.0, 8.0]], jnp.float32)
    w_jt = jnp.asarray([[0.75, 0.25]], jnp.float32)
    dx_jt = jnp.asarray([[np.float32(-(2.0 ** 125)), np.float32(2.0 ** 125)]], jnp.float32)
    dw_jt = jnp.asarray([[np.float32(3.0 * 2.0 ** 121), np.float32(-(2.0 ** 121))]], jnp.float32)

    _, (_, dv_jt) = jax.jvp(
        lambda x_, w_, t=tile, m=mode: moments.chunked_moments(x_, w_, tile=t, mode=m),
        (x_jt, w_jt), (dx_jt, dw_jt))
    assert float(dv_jt[0]) == 0.0


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("tile", [1, 3])
def test_chunked_covariance_balances_odd_width_mixed_derivatives(mode, tile):
    from dew.algebra import moments

    a_c, b_c, u_c = np.float32(2.0 ** -100), np.float32(2.0 ** -40), np.float32(2.0 ** -23)
    x_c = jnp.asarray([[a_c, a_c, -a_c]], jnp.float32)
    y_c = jnp.asarray([[b_c, b_c, -b_c]], jnp.float32)
    w_c = jnp.asarray([[0.0, 0.75, 0.75]], jnp.float32)
    dx_c = jnp.asarray([[(np.float32(1.5) + u_c) / b_c, (np.float32(1.5) + np.float32(2.0) * u_c) / b_c,
                         -(np.float32(1.5) + np.float32(2.0) * u_c) / b_c]], jnp.float32)

    _, d_gc = jax.jvp(
        lambda x_, t=tile, m=mode: jax.grad(
            lambda w__: moments.chunked_covariance(x_, y_c, w__, tile=t, mode=m)[2].sum())(w_c),
        (x_c,), (dx_c,))
    assert float(d_gc[0, 0]) == float(np.float32(-u_c / np.float32(1.5)))


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_chunked_covariance_retains_small_value_tangent(mode):
    """A direction spanning 100 binades must not lose its finite covariance derivative."""
    from dew.algebra import moments

    x = jnp.asarray([[-1.0, -1.0, 1.0]], jnp.float32)
    y = jnp.asarray([[-1.0, 1.0, 0.0]], jnp.float32)
    w = jnp.asarray([[0.25, 0.25, 0.5]], jnp.float32)
    dx = jnp.asarray([[0.0, 1.0, 2.0 ** 100]], jnp.float32)
    _, tangent = jax.jvp(lambda v: moments.chunked_covariance(v, y, w, tile=1, mode=mode)[2], (x,), (dx,))
    np.testing.assert_array_equal(tangent, [0.25])


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("inner", ["weight", "value"])
def test_a_stationary_weight_direction_cannot_erase_another_samples_tangent(mode, inner):
    from dew.algebra import moments

    x = jnp.asarray([[-1.0, 1.0, 0.0]], jnp.float32)
    w = jnp.asarray([[0.5, 0.5, 0.0]], jnp.float32)
    dw = jnp.asarray([[2.0 ** 125, 0.0, 2.0 ** -40]], jnp.float32)

    def variance(x, w):
        return moments.chunked_moments(x, w, tile=1, mode=mode)[1]

    if inner == "weight":
        def directed(x):
            return jax.jvp(lambda q: variance(x, q), (w,), (dw,))[1]

        first, second = jax.jvp(directed, (x,), (x,))
        np.testing.assert_array_equal(first, [-2.0 ** -40])
    else:
        def directed(w):
            return jax.jvp(lambda v: variance(v, w), (x,), (x,))[1]

        _, second = jax.jvp(directed, (w,), (dw,))
    np.testing.assert_array_equal(second, [-2.0 ** -39])


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_chunked_covariance_keeps_both_independent_value_directions(mode):
    from dew.algebra import moments

    big, small = 2.0 ** 125, 2.0 ** -40
    w = jnp.asarray([[0.5, 0.25, 0.25, 0.0], [0.25, 0.25, 0.25, 0.25]], jnp.float32)
    dx = jnp.asarray([[big, small, 0.0, 0.0], [big, big, 0.0, small]], jnp.float32)
    dy = jnp.asarray([[2.0 ** 50, 0.0, 2.0 ** 51, 2.0 ** -126], [big, 0.0, big, small]], jnp.float32)

    def mixed(weights, inner, outer):
        x = jnp.zeros_like(weights)

        def covariance(a, b):
            return moments.chunked_covariance(a, b, weights, tile=1, mode=mode)[2]

        return jax.jvp(lambda y: jax.jvp(lambda x_: covariance(x_, y), (x,), (inner,))[1],
                       (x,), (outer,))[1]

    expected = [-2.0 ** 8, -2.0 ** 83]
    checked = jax.jit(mixed)
    np.testing.assert_array_equal(checked(w, dx, dy), expected)
    np.testing.assert_array_equal(checked(w, dy, dx), expected)
    np.testing.assert_array_equal(jax.jit(jax.vmap(mixed))(w, dx, dy), expected)


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_mixed_covariance_pullback_keeps_faint_stationary_coefficients(mode):
    from dew.algebra import moments

    big, small = 2.0 ** 125, 2.0 ** -40
    w = jnp.full((1, 4), 0.25, jnp.float32)
    x = jnp.zeros_like(w)
    direction = jnp.asarray([[big, big / 2, big / 2, small]], jnp.float32)

    def covariance(a, b):
        return moments.chunked_covariance(a, b, w, tile=1, mode=mode)[2].sum()

    def out(direction):
        reverse_after = jax.grad(lambda y: jax.jvp(lambda a: covariance(a, y), (x,), (direction,))[1])(x)
        forward_after = jax.jvp(lambda a: jax.grad(lambda b: covariance(a, b))(x), (x,), (direction,))[1]
        return reverse_after, forward_after

    for pullback in jax.jit(out)(direction):
        np.testing.assert_array_equal(pullback, [[big / 8, -small / 16, -small / 16, -big / 8]])


def test_a_dyadic_bin_keeps_integer_totals_past_float32_precision():
    from dew.algebra import bands, windows

    exact = [(1 << 24) + 1, -(1 << 24) - 1, (1 << 31) - 1, -(1 << 31), 1, 0, -1]
    high, low, exponent = bands.placed_totals(jnp.asarray(exact, jnp.int32), 0, jnp.float32, may_round=True)
    found = np.ldexp(np.asarray(high, np.float64) + np.asarray(low, np.float64), np.asarray(exponent))
    np.testing.assert_array_equal(found, exact)

    opposite = bands.placed_totals(jnp.int32(-1), 0, jnp.float32, may_round=True)
    prod, rest, scale = windows._times((high, low, exponent), opposite)
    negative = np.ldexp(np.asarray(prod, np.float64) + np.asarray(rest, np.float64), np.asarray(scale))
    np.testing.assert_array_equal(negative, -np.asarray(exact))


@pytest.mark.parametrize(("rows", "states"), [(0, 4), (1, 65536)])
def test_mixed_covariance_in_a_constant_direction_is_zero(rows, states):
    from dew.algebra import moments

    def mixed(x, dx, dy):
        def first(y):
            def covariance(v):
                return moments.chunked_covariance(v, y, w, tile=min(states, 128), mode="merge")[2]

            return jax.jvp(covariance, (x,), (dx,))[1]

        return jax.jvp(first, (x,), (dy,))[1]

    with jax.enable_x64(new_val=True):
        w = jnp.ones((rows, states), jnp.float64)
        x = jnp.zeros_like(w)
        dy = x.at[:, 0].set(1.0)
        np.testing.assert_array_equal(jax.jit(mixed)(x, w, dy), np.zeros((rows,), np.float64))


def test_covariance_means_keep_distinct_value_directions_with_shared_weights():
    from dew.algebra import moments

    rows = jnp.asarray([[0.0, 4.0], [2.0, -2.0]], jnp.float32)
    weights = jnp.asarray([1.0, 3.0], jnp.float32)
    direction = jnp.asarray([[4.0, 0.0], [0.0, 4.0]], jnp.float32)
    dw = jnp.asarray([1.0, -1.0], jnp.float32)

    def means(x, y, w):
        a, b, _ = moments.chunked_covariance(x, y, w, tile=1, mode="walk")
        return jnp.stack((a, b))

    @jax.jit
    def differentiated(x, y, w, dx, dy, dw):
        both = jax.jvp(means, (x, y, w), (dx, dy, dw))[1]
        first = jax.jvp(lambda a: means(a, y, w), (x,), (dx,))[1]
        _, pullback = jax.vjp(lambda q: means(x, y, q), w)
        (weighted,) = pullback(jnp.asarray([2.0, -3.0], w.dtype))
        return both, first, weighted

    both, first, weighted = differentiated(*rows, weights, *direction, dw)
    np.testing.assert_array_equal(both, [0.0, 4.0])
    np.testing.assert_array_equal(first, [1.0, 0.0])
    np.testing.assert_array_equal(weighted, [-3.75, 1.25])


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_weighted_moments_keep_the_direction_when_dividing_first_would_flush(mode):
    from dew.algebra import moments

    x = jnp.asarray([[0.0, 2.0 ** -60]], jnp.float32)
    w = jnp.asarray([[0.0, 2.0 ** 100]], jnp.float32)
    dw = jnp.asarray([[2.0 ** 100, 0.0]], jnp.float32)

    def weighted(weights):
        return moments.chunked_moments(x, weights, tile=1, mode=mode)

    _, (mean_direction, var_direction) = jax.jvp(weighted, (w,), (dw,))
    np.testing.assert_array_equal(mean_direction, [-2.0 ** -60])
    np.testing.assert_array_equal(var_direction, [2.0 ** -120])

    def covariance(weights):
        return moments.chunked_covariance(x, x, weights, tile=1, mode=mode)[2]

    _, pull = jax.vjp(covariance, w)
    (weight_gradient,) = pull(jnp.asarray([2.0 ** 100], jnp.float32))
    np.testing.assert_array_equal(weight_gradient, [[2.0 ** -120, 0.0]])


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_zero_weighted_values_keep_zero_tangents_when_finite_weights_overflow_the_mass(mode):
    from dew.algebra import moments

    x = jnp.zeros((1, 2), jnp.float32)
    w = jnp.full_like(x, 2.0 ** 127)
    dw = jnp.asarray([[1.0, 0.0]], jnp.float32)

    def run(weights):
        return moments.chunked_moments(x, weights, tile=1, mode=mode)

    value, tangent = jax.jvp(run, (w,), (dw,))
    np.testing.assert_array_equal(jnp.stack((*value, *tangent)), np.zeros((4, 1), np.float32))


def test_a_normal_weight_quotient_preserves_stationary_covariance_rounding():
    from dew.algebra import moments

    x = jnp.asarray([[2.0 ** 64, -(2.0 ** 64)]], jnp.float32)
    y = jnp.asarray([[2.0 ** 63, -(2.0 ** 63)]], jnp.float32)
    w = jnp.full_like(x, 0.75)
    dw = jnp.asarray([[1.5 + 2.0 ** -22, 0.0]], jnp.float32)

    def covariance(weights):
        return moments.chunked_covariance(x, y, weights, tile=1, mode="merge")[2]

    value, tangent = jax.jvp(covariance, (w,), (dw,))
    np.testing.assert_array_equal(value, [2.0 ** 127])
    np.testing.assert_array_equal(tangent, [0.0])


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_covariance_weight_jvp_stays_finite_when_the_centered_product_overflows(mode):
    from dew.algebra import moments

    # The centered product is 2^140, but its weight direction is only 2^-70.
    x = jnp.asarray([[2.0 ** 70, 0.0]], jnp.float32)
    w = jnp.asarray([[0.0, 1.0]], jnp.float32)
    dw = jnp.asarray([[2.0 ** -70, 0.0]], jnp.float32)
    _, (_, _, dc) = jax.jvp(lambda q: moments.chunked_covariance(x, x, q, tile=1, mode=mode), (w,), (dw,))
    np.testing.assert_array_equal(dc, [2.0 ** 70])


def _value_gradients(mode, moment):
    from dew.algebra import moments

    def written(x, w):
        return moments.weighted_moments(x, w)[moment].sum()

    def chunked(x, w):
        return moments.chunked_moments(x, w, tile=1, mode=mode)[moment].sum()

    return jax.grad(written), jax.grad(chunked)


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("moment", [0, 1], ids=["mean", "variance"])
@pytest.mark.parametrize("mass", [1.0, 0.0], ids=["one-zero", "all-zero"])
def test_chunked_moment_value_gradient_keeps_its_mixed_zero_weight_direction(mode, moment, mass):
    x = jnp.asarray([[0.0, 2.0]], jnp.float32)
    w = jnp.asarray([[mass, 0.0]], jnp.float32)
    dw = jnp.asarray([[0.0, 1.0]], jnp.float32)
    plain, chunked = _value_gradients(mode, moment)
    _, expected = jax.jvp(lambda q: plain(x, q), (w,), (dw,))
    _, actual = jax.jvp(lambda q: chunked(x, q), (w,), (dw,))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("moment", [0, 1], ids=["mean", "variance"])
@pytest.mark.parametrize("mass", [1.0, 0.0], ids=["one-zero", "all-zero"])
def test_jitted_chunked_moment_value_gradient_matches_the_direct_zero_weight_gradient(mode, moment, mass):
    x = jnp.asarray([[0.0, 2.0]], jnp.float32)
    w = jnp.asarray([[mass, 0.0]], jnp.float32)
    plain, chunked = _value_gradients(mode, moment)
    np.testing.assert_array_equal(jax.jit(chunked)(x, w), plain(x, w))


def _direct_moment_forms(x, mode):
    """Differentiate the derived forms themselves, not chunked moments' custom tangent."""
    from dew.algebra import moments

    forms = derive.derive(moments.declaration("weighted"), accurate=False, saturation=SATURATION)
    shapes = dict.fromkeys(("x", "w"), ((1, 1), jnp.float32))

    def evaluate(w):
        empty = forms.init(shapes)
        left = {"x": x[:, :1], "w": w[:, :1]}
        right = {"x": x[:, 1:], "w": w[:, 1:]}
        first = forms.step(empty, left)
        final = (forms.step(first, right) if mode == "walk"
                 else forms.merge(first, forms.step(empty, right), shapes=shapes))
        return forms.finish(final, left, shapes=shapes)

    return evaluate


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize(("values", "weights", "direction", "expected"), [
    ([0.0, 2.0], [1.0, 0.0], [0.0, 1.0], (2.0, 4.0)),
    ([10.0, 12.0], [1.0, 0.0], [0.0, 1.0], (2.0, 4.0)),
    ([2.0 ** 70, 2.0 ** 70], [1.0, 0.0], [0.0, 1.0], (0.0, 0.0)),
    ([2.0 ** 70, 2.0 ** 70], [0.0, 1.0], [1.0, 0.0], (0.0, 0.0)),
    ([3.0, 5.0], [0.0, 0.0], [1.0, 2.0], (13.0, 59.0)),
], ids=["one-zero", "shifted", "large-right-zero", "large-left-zero", "all-zero"])
def test_direct_weighted_moment_forms_keep_zero_weight_jvp(mode, values, weights, direction, expected):
    x, w, dw = (jnp.asarray([row], jnp.float32) for row in (values, weights, direction))
    _, (dm, dv) = jax.jvp(_direct_moment_forms(x, mode), (w,), (dw,))
    np.testing.assert_array_equal(dm, [expected[0]])
    np.testing.assert_array_equal(dv, [expected[1]])


@pytest.mark.parametrize("mode", ["walk", "merge"])
@pytest.mark.parametrize("weights", [[1.0, 0.0], [0.0, 1.0]], ids=["right-zero", "left-zero"])
def test_direct_weighted_variance_pullback_is_zero_at_equal_large_values(mode, weights):
    x = jnp.full((1, 2), 2.0 ** 70, jnp.float32)
    w = jnp.asarray([weights], jnp.float32)
    evaluate = _direct_moment_forms(x, mode)
    dw = jax.grad(lambda q: evaluate(q)[1].sum())(w)
    np.testing.assert_array_equal(dw, np.zeros((1, 2), np.float32))


def test_chunked_covariance_balances_unequal_scales():
    """Positive-mass combines and tile products balance unequal covariance scales (`x ~ 2^100, y ~ 2^-100,
    w ~ 2^-100`): the covariance stays normal and matches float64 truth at each tile and in each mode, where
    naive float32 `(w * dx) * dy` underflows to zero."""
    from dew.algebra import moments

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


def test_chunked_moments_keep_minimum_normal_weights_exact():
    """Minimum-normal float32 weights (`2^-126`): the normalized monoid keeps the primal and the value VJPs
    finite and exact through `chunked_moments`, `chunked_covariance` and direct `derive.derive` forms (no
    `custom_jvp`)."""
    from dew.algebra import moments

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


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_chunked_covariance_value_gradients_hold_check_1(mode):
    """Ordinary value gradients (`dx`, `dy`) of the chunked covariance on par with the declared reduction's
    (check 1, `assert_as_exact`)."""
    from dew.algebra import moments

    def draw_xyw(seed):
        rng = np.random.default_rng([9044, seed])
        return ((10.0 + rng.standard_normal((4, 16))).astype(np.float32),
                (-5.0 + rng.standard_normal((4, 16))).astype(np.float32),
                rng.uniform(0.25, 2.0, (4, 16)).astype(np.float32))

    assert_as_exact(
        jax.jit(jax.grad(lambda x_, y_, w_: moments.chunked_covariance(
            x_, y_, w_, tile=4, mode=mode)[2].sum(), argnums=(0, 1))),
        jax.jit(jax.grad(lambda x_, y_, w_: moments.weighted_covariance(
            x_, y_, w_)[2].sum(), argnums=(0, 1))),
        draw_xyw, f"chunked covariance value gradients ({mode})", seeds=8)


@pytest.mark.parametrize("mode", ["walk", "merge"])
def test_chunked_moments_lower_under_an_explicit_mesh(mode):
    """The derived walk and merge lower under an explicit mesh (`AxisType.Explicit`) and give the weighted
    moments bit for bit."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    from dew.algebra import moments

    xv = jnp.asarray([[1.0, 3.0, 5.0, 7.0]], jnp.float32)
    wv = jnp.asarray([[0.5, 1.5, 1.0, 1.0]], jnp.float32)
    ref_ms, ref_vs = moments.weighted_moments(xv, wv)
    mesh = jax.make_mesh((1,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:1])
    sharding = NamedSharding(mesh, P("data", None))
    with jax.set_mesh(mesh):
        xs, ws = jax.device_put(xv, sharding), jax.device_put(wv, sharding)
        ms, vs = jax.jit(lambda a, b: moments.chunked_moments(a, b, tile=2, mode=mode))(xs, ws)
    np.testing.assert_array_equal(np.asarray(ms), np.asarray(ref_ms))
    np.testing.assert_array_equal(np.asarray(vs), np.asarray(ref_vs))


def _zero_mass_covariance(x_, y_, w_, *, tile: int, mode: str):
    """The covariance's direct `derive` forms (`moments.derived("covariance")`, derived once a process) over
    `tile`-wide tiles of one row, walked (`walk`) or each tile stepped alone and merged (`merge`), its
    covariance summed."""
    from dew.algebra import moments

    forms_cov = moments.derived("covariance")
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


_ZERO_MASS_BIG = np.float32(2.0 ** 70)
_ZERO_MASS_HUGE = np.float32(2.0 ** 127)
_ZERO_MASS_FMAX = np.finfo(np.float32).max
_ZERO_MASS_MID = np.float32(2.0 ** 63)
_ZERO_MASS_NEAR = np.float32(1.25 * 2.0 ** 63)
_ZERO_MASS_OVER = np.float32(2.0 ** 64)
_ZERO_MASS_SMALL = np.float32(2.0 ** -45)
_ZERO_MASS_TINY = float(np.finfo(np.float32).tiny)
"""The zero-mass covariance witnesses' values: `2^70` beside zero, `2^127` and float32's largest whose
signs cancel, `2^63`, `1.25 * 2^63` (its square's boundary) and `2^64` (whose square overflows) beside them,
`2^-45` (a small entry beside large ones) and the minimum normal `2^-126` (a weight direction)."""


def test_zero_mass_scaled_covariance_ad_is_zero_in_orthogonal_directions():
    """Two-band zero-mass covariance (`moment_hi`, `moment_scale`): `x = [2^70, 0], y = [0, 2^70], w = [0, 0]`
    has zero exact derivative in both weight directions and both value directions, direct JVP and VJP, at
    `tile=1` (walked and merged) and `tile=2`."""
    big = _ZERO_MASS_BIG
    xo = jnp.asarray([[big, 0.0]], jnp.float32)
    yo = jnp.asarray([[0.0, big]], jnp.float32)
    wo = jnp.asarray([[0.0, 0.0]], jnp.float32)
    for tile in (1, 2):
        for mode in ("walk", "merge"):
            def fn(w_, t=tile, m=mode):
                return _zero_mass_covariance(xo, yo, w_, tile=t, mode=m)

            for dw_vals in ([1.0, 0.0], [0.0, 1.0]):
                val, dval = jax.jvp(fn, (wo,), (jnp.asarray([dw_vals], jnp.float32),))
                assert float(val) == 0.0 and float(dval) == 0.0
            gw = jax.grad(fn)(wo)
            np.testing.assert_array_equal(np.asarray(gw), np.zeros((1, 2), np.float32))
            gx, gy = jax.grad(
                lambda x_, y_, t=tile, m=mode: _zero_mass_covariance(x_, y_, wo, tile=t, mode=m),
                argnums=(0, 1))(xo, yo)
            np.testing.assert_array_equal(np.asarray(gx), np.zeros((1, 2), np.float32))
            np.testing.assert_array_equal(np.asarray(gy), np.zeros((1, 2), np.float32))


def test_zero_mass_scaled_covariance_ad_keeps_each_tiles_scaled_tangent():
    """Two-band zero-mass covariance's direct JVP at `tile=1`, walked and merged: finite scaled derivatives,
    `2^127` sign cancellation, small entries beside large ones, boundary finite (`1.25*2^63`) and
    overflowing (`2^64`) entries beside `2^127` and float32's largest under a minimum-normal `dw = 2^-126`,
    and one-sided shifts beside `+-2^127`, each exact."""
    big, huge, fmax = _ZERO_MASS_BIG, _ZERO_MASS_HUGE, _ZERO_MASS_FMAX
    mid_big, near_ov, hi_ov = _ZERO_MASS_MID, _ZERO_MASS_NEAR, _ZERO_MASS_OVER
    small, tiny = _ZERO_MASS_SMALL, _ZERO_MASS_TINY
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
                lambda w_, x_=xc, y_=yc, m=mode: _zero_mass_covariance(x_, y_, w_, tile=1, mode=m),
                (wc,), (dwc,))
            assert float(val) == 0.0 and float(dval) == expect


def test_zero_mass_scaled_covariance_ad_holds_across_multi_scale_tiles():
    """Two-band zero-mass covariance's direct JVP over tiles of three (and two) that hold high-band entries
    of several scales beside a small one, walked and merged, and its VJP in the weights, each exact."""
    huge, fmax = _ZERO_MASS_HUGE, _ZERO_MASS_FMAX
    mid_big, near_ov, hi_ov = _ZERO_MASS_MID, _ZERO_MASS_NEAR, _ZERO_MASS_OVER
    small, tiny = _ZERO_MASS_SMALL, _ZERO_MASS_TINY
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
            val, dval = jax.jvp(
                lambda w_, x_=xs_mix, t=tile, m=mode: _zero_mass_covariance(x_, x_, w_, tile=t, mode=m),
                (ws_mix,), (dws_mix,))
            assert float(val) == 0.0 and float(dval) == expect
    gw_mix = jax.grad(lambda w_: _zero_mass_covariance(
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
