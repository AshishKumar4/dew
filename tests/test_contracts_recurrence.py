"""Float-range and nested-derivative contracts of the recurrences' seams: the chunked delta rule, its ports
and template, the decode step, the SSD port and the TPU scan, the short convolution, and the chunk forms
`dew.algebra.scan` derives (ledger 2641; the harness is tests/contract_support.py)."""

import functools
import itertools

import delta_rule_support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import (
    EDGES,
    FORWARD_ORDERS,
    ORDER_SETS,
    REVERSE_ONLY,
    assert_contract,
    assert_drawn_contract,
    normal_draw,
    normal_float32,
    order_set,
)
from generation_support import interprets
from map_judge import marked_seeds
from ssd_support import stepwise_scan

from dew.algebra import SATURATION, scan
from dew.nn.kernels import diagonal, generation, recurrence, recurrence_chunk, short_conv, ssd
from dew.nn.kernels.templates.recurrence import turned
from dew.nn.linear import causal_conv1d
from dew.nn.mixers.mamba2 import chunk_ssd
from dew.nn.sharding import manual_map

DELTA_RULE_EDGES = {"normal": 1.0, "the largest its chunk holds": 2.0**62, "near the largest": 2.0**63,
                    "near the least normal": 2.0**-63}
"""The chunked delta rule's edges. Near float32's largest its chunk's own products overflow at entries the
token recurrence keeps finite, as fla's chunked form's do at the same entries (`precedent_chunked`, on an
H100: ledger 2470), which that edge compares it with in place of the recurrence."""


@pytest.mark.parametrize("edge", EDGES)
def test_a_run_updating_in_its_backward_moves_as_its_step_would(edge):
    """A scanned run that updates its weights in its backward (`dew.nn.in_backward.scanned_run`, a rule whose
    weights' cotangent is the step's new weights, row 4796), over three layers of a tanh projection and an
    SGD step: its primal and its one reverse pass finite wherever its float64 run is a normal float32. It
    runs inside a step alone, which takes it once in reverse: no other order (`REVERSE_ONLY`)."""
    from dew.nn import in_backward

    def update(path, index, gradient, weights, moments):
        return jax.tree.map(lambda w, g: w - 0.5 * g, weights, gradient), {}

    def layer(weights, x, key, metadata):
        return jnp.tanh(x @ weights["kernel"])

    def run(kernel, x):
        norms = {"sumsq": jnp.zeros((3,), kernel.dtype), "nonfinite": jnp.zeros((3,), kernel.dtype)}
        with in_backward.planned(in_backward.InBackward(update=update, moments=())):
            return in_backward.scanned_run(("layers_0_2",), layer, {"kernel": kernel}, None, {}, norms, x,
                                           None, (None,))

    def draw(seed, scale):
        primals = normal_draw(4796 + 4096 * seed, (3, 4, 4), (2, 4), scale=scale)
        direction = normal_draw(4797 + 4096 * seed, (3, 4, 4), (2, 4), scale=scale)
        return primals, direction, normal_draw(4798 + 4096 * seed, (2, 4))[0]

    assert_drawn_contract(run, draw, EDGES[edge], f"a run updating in its backward {edge}",
                          without=REVERSE_ONLY)


def draw_stream(salt: int, seed: int, at: int, *, fixture: int | None = None) -> int | list[int]:
    """The seed of a contract cell's draw `at` at seed `seed` (`normal_draw`'s, NumPy's): seed zero's its
    fixture's (`fixture`, by default `salt + at`: the cell's own inputs, their streams as its contract was
    measured), a later seed's `[salt, seed, at]` (`SeedSequence` entropy), each of the cell's draws a stream
    of its own, where `salt + at + 4096 seed` gave two of a cell's draws one: the KDA port's query direction
    and cotangent, the template's, the turned call's turns and primals (row 8449)."""
    if seed == 0:
        return salt + at if fixture is None else fixture
    return [salt, seed, at]


def tangent_keys(key: jax.Array, count: int, seed: int) -> tuple:
    """A cell's `count` operands' tangent keys from its one tangent key `key`: seed zero's that key for each
    (its fixture's, as its contract was measured), a later seed's each operand its own (`jax.random.split`),
    where one key drew every operand's tangent, a call's queries', keys' and values' the same (row 8449)."""
    return (key,) * count if seed == 0 else tuple(jax.random.split(key, count))


def seeded_key(seed: int) -> jax.Array:
    """`jax.random.key(seed)` with every bit of `seed`: formed with 64-bit integers on (its draws taken after,
    in float32), where a 32-bit one wrapped a draw's `salt + 4096 seed` past 2^32 onto a smaller seed's, the
    sizing's pilot seeds (from 2^20, `reference_error.SIZING_FROM`) onto the verdict's (seed zero's key
    unchanged)."""
    with jax.enable_x64(new_val=True):
        return jax.random.key(seed)


INTERPRETED = interprets()
"""The ported kernels interpret off a GPU."""


def flattened(fn):
    """`fn`'s outputs as one vector, which the orders through its gradient sum against one cotangent."""
    return lambda *operands: jnp.concatenate([out.ravel() for out in jax.tree.leaves(fn(*operands))])


def precedent_chunked(query, key, value, g, beta, *, chunk_size: int, starts=None):
    """fla's upstream chunked gated delta rule as it is written (`naive_chunk_gated_delta_rule`,
    fla/ops/gated_delta_rule/naive.py), in JAX for its derivatives, over each packed document alone as fla
    runs a varlen batch's sequences: the precedent the chunked rule's largest edge compares with. Its row
    correction one row at a time and its masks filled after their products, as fla's are; `g` `[B, S, H, 1]`,
    from a zero memory."""
    if starts is not None:
        begun = np.flatnonzero(np.asarray(starts)[0])
        cuts = [0, *begun[begun > 0].tolist(), query.shape[1]]
        pieces = [precedent_chunked(*(x[:, a:b] for x in (query, key, value, g, beta)), chunk_size=chunk_size)
                  for a, b in itertools.pairwise(cuts) if b > a]
        return jnp.concatenate(pieces, axis=1)
    batch, length, heads, width = query.shape
    pad = (-length) % chunk_size
    q, k, v = (jnp.pad(jnp.moveaxis(x, 1, 2), [(0, 0), (0, 0), (0, pad), (0, 0)])
               for x in (query, key, value))
    beta, g = (jnp.pad(jnp.moveaxis(x, 1, 2), [(0, 0), (0, 0), (0, pad)]) for x in (beta, g[..., 0]))
    q, v, keyed = q * width ** -0.5, v * beta[..., None], k * beta[..., None]
    chunks = (length + pad) // chunk_size
    q, k, v, keyed = (x.reshape(batch, heads, chunks, chunk_size, -1) for x in (q, k, v, keyed))
    decay = jnp.cumsum(g.reshape(batch, heads, chunks, chunk_size), axis=-1)
    upper = jnp.triu(jnp.ones((chunk_size, chunk_size), bool))
    weights = jnp.tril(jnp.exp(jnp.tril(decay[..., :, None] - decay[..., None, :])))
    system = -jnp.where(upper, 0.0, (keyed @ jnp.swapaxes(k, -1, -2)) * weights)
    for row in range(1, chunk_size):
        system = system.at[..., row, :row].set(
            system[..., row, :row] + jnp.sum(system[..., row, :row, None] * system[..., :row, :row], axis=-2))
    system = system + jnp.eye(chunk_size)
    v, written = system @ v, system @ (keyed * jnp.exp(decay)[..., None])
    memory = jnp.zeros((batch, heads, width, v.shape[-1]))
    strict = jnp.triu(jnp.ones((chunk_size, chunk_size), bool), 1)
    outputs = []
    for index in range(chunks):
        scores = jnp.where(strict, 0.0, (q[:, :, index] @ jnp.swapaxes(k[:, :, index], -1, -2))
                           * weights[:, :, index])
        corrected = v[:, :, index] - written[:, :, index] @ memory
        read = (q[:, :, index] * jnp.exp(decay[:, :, index, :, None])) @ memory
        outputs.append(read + scores @ corrected)
        last = decay[:, :, index, -1]
        memory = memory * jnp.exp(last)[..., None, None] + jnp.swapaxes(
            k[:, :, index] * jnp.exp(last[..., None] - decay[:, :, index])[..., None], -1, -2) @ corrected
    out = jnp.stack(outputs, axis=2).reshape(batch, heads, chunks * chunk_size, -1)[:, :, :length]
    return jnp.moveaxis(out, 1, 2)


INTERPRETED_REMAT = ("jvp of grad, remat", "second jvp, remat") if INTERPRETED else ()
"""The orders a kernel Mosaic GPU's interpreter runs takes on no CPU: JAX's checkpoint takes no ordered
effect, which the interpreter's callbacks carry (`NotImplementedError: Effects not supported in AD of
checkpoint`); a GPU's kernel carries none."""


@pytest.mark.parametrize("taken", list(ORDER_SETS))
@pytest.mark.parametrize(("packed", "edge"), list(itertools.product((False, True), DELTA_RULE_EDGES)),
                         ids=lambda value: {False: "one sequence", True: "packed"}.get(value, value))
def test_the_chunked_delta_rule_holds_every_order(packed, edge, taken):
    """`recurrence_chunk.xla_mainloop` (the recurrence template's plain-JAX mainloop, the portable route, its
    derivatives the derived forms' rules, `dew.algebra.scan`): every order runs, its boolean document starts a
    residual with no tangent space, and each is finite wherever float64 is a normal float32 and JAX's own
    derivatives of the token recurrence are (`recurrent_delta_rule`, an independent formula of the same
    rule, in float64 the truth too: the chunk's own float64 run excused its float32 NaNs wherever it was NaN
    itself), near float32's largest of fla's own chunked form (`precedent_chunked`), at queries, keys and
    values whose products near float32's ends, the decays and gates as drawn (`DELTA_RULE_EDGES`). A pair
    the chunk weighs zero once multiplied its overflowed tangent by zero, the queries' second derivative NaN
    from 2^48 (ledger 2470).

    Packed documents sharing chunks hold the same orders, including forward over reverse through the replay's
    gradient. Reverse over reverse at 2^63 retains the owner's accepted overflow-only difference (ledger
    2470)."""
    rows, length, heads, width = 1, 37, 2, 8
    # The documents a constant of the site, its draws' operands about them.
    marked = jax.random.bernoulli(jax.random.split(jax.random.key(1861), 8)[5], 0.1, (rows, length))
    starts = marked if packed else None

    def draw(seed, scale):
        keys = jax.random.split(seeded_key(1861 + 4096 * seed), 8)
        query, key, value = (jax.random.normal(k, (rows, length, heads, width)) * scale for k in keys[:3])
        g = -jax.nn.softplus(jax.random.normal(keys[3], (rows, length, heads, 1)))
        beta = jax.nn.sigmoid(jax.random.normal(keys[4], (rows, length, heads)))
        primals = (query, key, value, g, beta)
        direction = tuple(jax.random.normal(along, x.shape) * (scale if at < 3 else 1.0) for at, (along, x)
                          in enumerate(zip(tangent_keys(keys[6], len(primals), seed), primals, strict=True)))
        return primals, direction, jax.random.normal(keys[7], query.shape)

    def rule(query, key, value, g, beta):
        return recurrence_chunk.xla_mainloop(recurrence_chunk.Tiles(16), query, key, value, g[..., 0], beta,
                                             None, starts, scale=width ** -0.5, delta=True,
                                             dtype=jnp.float32)[0]

    def recurrence(query, key, value, g, beta):
        return recurrence_chunk.recurrent_delta_rule(query, key, value, g, beta, starts=starts)[0]

    plain = (functools.partial(precedent_chunked, chunk_size=16, starts=starts) if edge == "near the largest"
             else recurrence)
    # Each order one program (`jax.jit`): run op by op, an order's eager dispatch compiled 2,570 programs of
    # one or two ops, 109 of the case's 156 s near float32's largest (cProfile on armada, row 4433).
    assert_drawn_contract(jax.jit(rule), draw, DELTA_RULE_EDGES[edge], f"the chunked delta rule {edge}",
                          truth_fn=jax.jit(recurrence), plain_fn=jax.jit(plain),
                          taking=ORDER_SETS[taken])


@pytest.mark.parametrize("order", ("jvp", "vjp", "jvp of grad", "grad of grad"))
def test_the_chunked_delta_rules_reader_scale_holds_every_order(order):
    """The packed chunked delta rule at the input that exposed row 8428 (queries, keys and values of 2^63,
    drawn from `fold_in(key(1861), 0)`): each order finite wherever float64's is a normal float32, where the
    queries' second orders formed `dB r` at a document's first read before the reader's scale `8^-1/2` brought
    it back, past float32's largest (`scan._read`'s scaled contraction applies it first)."""
    rows, length, heads, width = 1, 37, 2, 8
    scale = 2.0 ** 63
    starts_key = jax.random.split(jax.random.key(1861), 8)[5]
    starts = jax.random.bernoulli(starts_key, 0.1, (rows, length))
    keys = jax.random.split(jax.random.fold_in(jax.random.key(1861), 0), 8)
    query, key, value = (jax.random.normal(k, (rows, length, heads, width)) * scale for k in keys[:3])
    g = -jax.nn.softplus(jax.random.normal(keys[3], (rows, length, heads, 1)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (rows, length, heads)))
    primals = (query, key, value, g, beta)
    direction = tuple(jax.random.normal(keys[6], x.shape) * (scale if at < 3 else 1.0)
                      for at, x in enumerate(primals))
    cotangent = jax.random.normal(keys[7], query.shape)

    def rule(query, key, value, g, beta):
        return recurrence_chunk.xla_mainloop(
            recurrence_chunk.Tiles(16), query, key, value, g[..., 0], beta, None, starts,
            scale=width ** -0.5, delta=True, dtype=query.dtype)[0]

    assert_contract(jax.jit(rule), primals, direction, cotangent,
                    f"the chunked delta rule's reader scale ({order})", taking=(order,))


def test_the_chunked_delta_rules_reader_scale_precedes_a_small_write():
    """A one-token delta rule whose queries and keys are 2^63 and values 2^-63 (a write whose reader's
    product passes float32's largest before its small value brings it back): its forward derivative finite
    where float64's is, the reader's scale applied before the contraction's products (row 8428)."""
    shape = (1, 1, 1, 8)
    primals = (jnp.full(shape, 2.0 ** 63), jnp.full(shape, 2.0 ** 63),
               jnp.full(shape, 2.0 ** -63), jnp.full((1, 1, 1), -1.0), jnp.ones((1, 1, 1)))
    direction = (*primals[:3], jnp.zeros_like(primals[3]), jnp.zeros_like(primals[4]))

    def rule(*operands):
        return recurrence_chunk.xla_mainloop(
            recurrence_chunk.Tiles(16), *operands, scale=8 ** -0.5, delta=True,
            dtype=operands[0].dtype)[0]

    with jax.enable_x64(new_val=True):
        wide, along = jax.tree.map(lambda x: x.astype(jnp.float64), (primals, direction))
        truth = np.asarray(jax.jit(lambda *x: jax.jvp(rule, x, along)[1])(*wide))
    assert np.all(normal_float32(truth) & np.isfinite(truth))
    assert_contract(jax.jit(rule), primals, direction, jnp.ones_like(primals[0]),
                    "the chunked delta rule's reader scale before a small write", taking=("jvp",))


def plain_delta_rule(query, key, value, g, beta, state, scale):
    """The gated delta rule a token at a time (`delta_rule_support.recurrence` over `rule_math`, the
    oracle its kernels' tests share), the query scaled by `scale`: the output `[B, S, Hv, Dv]` and the final
    memory `[B, Hv, Dk, Dv]`, key heads serving `Hv // Hk` value heads each."""
    return delta_rule_support.recurrence(query, key, value, g, beta, state,
                                         rule=functools.partial(delta_rule_support.rule_math, scale=scale))


def plain_ssd(x, dt, a, b, c, skip, state):
    """Mamba2's state space recurrence a token at a time: `h_t = exp(dt_t a) h_{t-1} + dt_t x_t b_t`,
    `y_t = h_t c_t + skip x_t` (`ssd_support.stepwise_scan` of the inputs scaled by their steps), `x`
    `[B, S, H, P]`, `dt` `[B, S, H]`, `a` and `skip` `[H]`, `b` and `c` `[B, S, G, N]` each serving
    `H // G` heads, `state` `[B, H, P, N]`. Returns `y` and the final state."""
    group = x.shape[2] // b.shape[2]
    b, c = (jnp.repeat(t, group, axis=2) for t in (b, c))
    y, final = stepwise_scan(dt[..., None] * x, b, c, dt * a, state)
    return y + skip[:, None] * x, final


def delta_rule_draw(salt: int, seed: int, scale: float, *, key_heads: int, value_heads: int, width: int,
                    length: int, per_key: bool, unit_keys: bool = False, fixture: int | None = None):
    """A delta rule's operands `(query, key, value, g, beta, state)`, their direction and the outputs'
    cotangents, a cell's draws 0 to 2 at `seed` (`draw_stream`, its `salt`; seed zero's from `fixture` where
    another cell's fixture is its own): queries, values and memory at `scale`, keys at it or of unit norm, a
    log decay in (-inf, 0) a head or a key dimension, write gates in (0, 1)."""
    def stream(at):
        return draw_stream(salt, seed, at, fixture=None if fixture is None else fixture + at)

    query, key, value, decay, gate, state = normal_draw(
        stream(0), (1, length, key_heads, width), (1, length, key_heads, width),
        (1, length, value_heads, width), (1, length, value_heads, *((width,) if per_key else ())),
        (1, length, value_heads),
        (1, value_heads, width, width))
    key = key / jnp.linalg.norm(key, axis=-1, keepdims=True) if unit_keys else key * scale
    primals = (query * scale, key, value * scale, -jax.nn.softplus(decay), jax.nn.sigmoid(gate),
               state * scale)
    drawn = normal_draw(stream(1), *(x.shape for x in primals))
    direction = tuple(along * (scale if at in (0, 2, 5) or (at == 1 and not unit_keys) else 1.0)
                      for at, along in enumerate(drawn))
    cotangent = normal_draw(stream(2), value.shape, state.shape)
    return primals, direction, cotangent


@pytest.mark.mesh
@pytest.mark.parametrize("edge", sorted(EDGES))
def test_a_split_convs_bias_holds_its_orders(edge):
    """A conv over a sequence split four ways (`linear.sequence_conv` under `over_sequence`), its replicated
    bias read through `_shards_summed` (a custom VJP whose backward adds the shards' partial sums in one
    rounding), on packed rows: the primal and every reverse order finite wherever float64's are normal
    float32s and the unsplit conv's are, at activations and biases scaled to the edge. Forward mode does
    not run (`FORWARD_ORDERS`, a custom VJP's)."""
    from sequence_split_support import SPLIT_ROWS

    from dew.nn.linear import Heads, held_conv1d, over_sequence, sequence_conv
    from dew.training import MeshSpec

    valid, segments = (jnp.asarray(np.broadcast_to(field, (2, 32))) for field in SPLIT_ROWS["packed"])
    mixed, direction, cotangent = normal_draw(2169, (2, 32, 8), (2, 32, 8), (2, 32, 8), scale=EDGES[edge])
    taps, bias = normal_draw(2170, (8, 4), (8,))
    mesh = MeshSpec(sequence=4).build(jax.devices()[:4])

    def split(mixed, bias):
        return over_sequence(lambda mixed, valid, segments, taps, bias: sequence_conv(
            mixed, taps, bias, segments, valid), 4, (mixed, valid, segments),
            (taps.astype(mixed.dtype), bias), heads=Heads(8, 8, {0: 2, 3: 0, 4: 0}))

    def whole(mixed, bias):
        return held_conv1d(mixed, taps.astype(mixed.dtype), valid, bias=bias, segments=segments)[0]

    # The mesh set around the contract, whose orders trace the split inside their own programs.
    with jax.set_mesh(mesh):
        assert_contract(split, (mixed, bias), (direction, jnp.ones_like(bias)), cotangent,
                        f"a split conv's bias {edge}", truth_fn=whole, plain_fn=whole, without=FORWARD_ORDERS)


@pytest.mark.parametrize("edge", sorted(EDGES))
def test_a_convs_bias_holds_every_order(edge):
    """A conv's bias spread over its rows and tokens (`short_conv.conv_bias`: reverse mode one compensated
    sum, row 1710): every order runs, finite wherever float64 is a normal float32 and the plain broadcast
    is."""
    conv_bias = short_conv.conv_bias
    def draw(seed, scale):
        return (normal_draw(1710 + 4096 * seed, (8,), scale=scale),
                normal_draw(1711 + 4096 * seed, (8,), scale=scale),
                normal_draw(1712 + 4096 * seed, (2, 5, 8))[0])

    assert_drawn_contract(lambda bias: conv_bias(bias, (2, 5)), draw, EDGES[edge], f"a conv's bias {edge}",
                          plain_fn=lambda bias: jnp.broadcast_to(bias, (2, 5, 8)))


# The ports near float32's largest: their reverse derivatives are NaN at entries the recurrence keeps finite,
# the gated delta rule's where fla's own chunked form keeps them too (ledger 2656).
@pytest.mark.parametrize("taken", list(ORDER_SETS))
@pytest.mark.parametrize("per_key", [False, True], ids=["fla_gdn", "fla_kda"])
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_delta_rule_ports_hold_their_orders(per_key, edge, taken):
    """fla's chunked gated delta rule and Kimi delta attention on the ported kernels
    (`fla_gdn.chunk_gated_delta_rule`, `fla_kda.chunk_kda`, custom VJPs over Triton kernels, interpreted off a
    GPU) as every caller reaches them: the delta rule's op with the port named
    (`recurrence_chunk.chunk_gated_delta_rule`), whose products' wrapper takes every order through the
    portable math (`_per_product`, row 2655), from a zero memory: each order of `taken`'s set finite
    wherever float64 is a normal float32 and the delta rule a token at a time is (`plain_delta_rule`, in
    float64 the truth), at queries, keys and values whose products near float32's least normal."""
    def draw(seed, scale):
        (query, key, value, g, beta, _), direction, _ = delta_rule_draw(
            2661 + per_key, seed, scale, key_heads=2, value_heads=2, width=64, length=64, per_key=per_key)
        cotangent = normal_draw(draw_stream(2661 + per_key, seed, 3, fixture=2663), value.shape)[0]
        return (query, key, value, g, beta), direction[:5], cotangent

    memory = jnp.zeros((1, 2, 64, 64), jnp.float32)

    def ported(*operands):
        with generation.chosen(gated_delta_rule="fla"):
            return recurrence_chunk.chunk_gated_delta_rule(*operands, memory)[0]

    def recurrent(*operands):
        return plain_delta_rule(*operands, memory.astype(operands[0].dtype), 64 ** -0.5)[0]

    assert_drawn_contract(jax.jit(ported), draw, EDGES[edge],
                          f"the {'kda' if per_key else 'gdn'} port {edge}", truth_fn=jax.jit(recurrent),
                          plain_fn=jax.jit(recurrent), taking=ORDER_SETS[taken])


TEMPLATE_ORDER_SETS = {"primal and jvp": ("primal", "jvp"), "vjp": ("vjp",),
                       **{name: held for name, held in ORDER_SETS.items() if name != "first orders"}}
"""The template rule's order sets: `ORDER_SETS`' with its first orders two cases, the primal and its forward
derivative, and its reverse one, whose interpreted launches compile apart: together at normal draws 121 to
129 s (tests/test_durations.json at dc7ebf822, row 9776; 169 compiles, 171 lowerings and 133 interpreted
thread runs, armada's p-9776)."""


# The template rule near float32's largest: its reverse derivative is NaN at entries fla's chunked form keeps
# finite (ledger 2656).
@pytest.mark.parametrize("taken", list(TEMPLATE_ORDER_SETS))
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_template_rule_holds_its_orders(edge, taken):
    """The recurrence template's chunked rule (`template.mosaic_rule`, a custom VJP over the
    mainloop's launches, interpreted off a GPU) as every caller reaches it: the delta rule's op with the
    template named, whose products' wrapper takes every order through the portable math (`_per_product`, row
    2655), a 16-bit call with its chunk inverse's declared bound (keys of unit norm, write gates at most one:
    its domain), from a zero memory: each order of `taken`'s set finite wherever float64 is a normal float32
    and the delta rule a token at a time is (`plain_delta_rule`), at queries and values whose products near
    float32's least normal."""
    def draw(seed, scale):
        (query, key, value, g, beta, _), direction, _ = delta_rule_draw(
            2664, seed, scale, key_heads=1, value_heads=1, width=64, length=64, per_key=False,
            unit_keys=True)
        primals = (query.astype(jnp.bfloat16), key.astype(jnp.bfloat16), value.astype(jnp.bfloat16), g,
                   beta.astype(jnp.bfloat16))
        direction = tuple(along.astype(operand.dtype)
                          for along, operand in zip(direction, primals, strict=False))
        return primals, direction, normal_draw(draw_stream(2664, seed, 3, fixture=2665), value.shape)[0]

    def rule(*operands):
        with generation.chosen(gated_delta_rule="mosaic_gpu"):
            return recurrence_chunk.chunk_gated_delta_rule(*operands, write_bound=1.0)[0]

    def recurrent(*operands):
        operands = tuple(x.astype(jnp.promote_types(x.dtype, jnp.float32)) for x in operands)
        return plain_delta_rule(*operands, jnp.zeros((1, 1, 64, 64), operands[0].dtype), 64 ** -0.5)[0]

    assert_drawn_contract(jax.jit(rule), draw, EDGES[edge], f"the template rule {edge}",
                          truth_fn=jax.jit(recurrent), plain_fn=jax.jit(recurrent),
                          taking=order_set(taken, INTERPRETED_REMAT, TEMPLATE_ORDER_SETS),
                          without=INTERPRETED_REMAT)


EDGES_BEYOND = [edge for edge in EDGES if edge != "normal"]
"""The edges whose cells measure no draws (`contract_support.assert_drawn_contract`): the normal domain's,
which check 1 measures, are cells of their own, their declared seeds sharded across CI's map tasks
(`check1_shards`): sized each run, a normal case took 382 to 720 s (rows 8414, 8488, 8525, 8552, 8553,
8558); each of their sites sizes 8 (a qualification run at f05f038f8, jsf-qual6)."""


@pytest.mark.parametrize("taken", list(ORDER_SETS))
@pytest.mark.parametrize("edge", EDGES_BEYOND)
def test_the_selective_scan_holds_its_orders(edge, taken):
    """`assert_selective_contract` at an edge of float32's range."""
    assert_selective_contract(edge, order_set(taken, INTERPRETED_REMAT))


CHECK1_SHARDS = {
    "test_the_selective_scan_holds_its_orders_over_normal_draws[first-orders]": {"seeds": 8, "per_task": 4},
    "test_the_selective_scan_holds_its_orders_over_normal_draws[jvp-of-grad]": {"seeds": 8, "per_task": 4},
    "test_the_selective_scan_holds_its_orders_over_normal_draws[second-jvp]": {"seeds": 8, "per_task": 8},
    "test_the_selective_scan_holds_its_orders_over_normal_draws[grad-of-grad]": {"seeds": 8, "per_task": 4},
    "test_the_selective_scan_holds_its_orders_over_normal_draws[jvp-of-grad-remat]":
        {"seeds": 8, "per_task": 4},
    "test_the_selective_scan_holds_its_orders_over_normal_draws[second-jvp-remat]":
        {"seeds": 8, "per_task": 4},
    "test_the_selective_scan_holds_its_orders_over_normal_draws[jvp-of-grad-nested-jit]":
        {"seeds": 8, "per_task": 4},
    "test_the_selective_scan_holds_its_orders_over_normal_draws[jvp-of-grad-scanned]":
        {"seeds": 8, "per_task": 4},
}
"""The selective scan's normal-draw sites, an order set each (`ORDER_SETS`), by node id, a literal the CI plan
reads as it reads a mark's (tools/armada/ci.py): a map task compiles a set's programs once for its draws (an
interpreted second order's 60 to 170 s, then 30 to 40 s a draw: the five sets in two tasks of four draws each
ran past 20 minutes a task, a set's eight draws in one 141 to 421 s), four draws a task but the second jvp's
eight."""


def normal_order_cases() -> list:
    """`ORDER_SETS` as the selective scan's normal-draw parameters, each marked from `CHECK1_SHARDS`."""
    test = "test_the_selective_scan_holds_its_orders_over_normal_draws"
    cases = [(taken, taken.replace(",", "").replace(" ", "-")) for taken in ORDER_SETS]
    return [pytest.param(taken, id=ident,
                         marks=pytest.mark.check1_shards(**CHECK1_SHARDS[f"{test}[{ident}]"]))
            for taken, ident in cases]


@pytest.mark.parametrize("taken", normal_order_cases())
def test_the_selective_scan_holds_its_orders_over_normal_draws(taken, request):
    """`assert_selective_contract` in the normal domain over an order set (`ORDER_SETS`), check 1 over its
    declared draws (`CHECK1_SHARDS`); the remat orders, which an interpreted kernel takes none of
    (`INTERPRETED_REMAT`, `order_set`: their sets skip there), beside the nested one."""
    assert_selective_contract("normal", order_set(taken, INTERPRETED_REMAT), seeds=marked_seeds(request))


def assert_selective_contract(edge: str, taken: tuple[str, ...], seeds: int | None = None) -> None:
    """Mamba's selective scan on the recurrence template (`selective.selective_scan`, a custom VJP over its
    forward and backward launches, interpreted off a GPU) as every caller reaches it: the scan's route with
    the template named (`recurrence_chunk.selective_scan`), whose products' wrapper takes every order through
    the portable scan (`_per_product`, row 2655), fp32 over one block of 128 channels and 16 states from no
    state: each order of `taken` finite wherever float64 is a normal float32 and the portable scan is
    (`dew.nn.mixers.mamba.selective_scan`, in float64 the truth), at inputs and skips scaled to the edge."""
    from dew.nn.mixers.mamba import selective_scan as portable

    def draw(seed, scale):
        x, dt, a, b, c, skip = normal_draw(2977 + 4096 * seed, (1, 64, 128), (1, 64, 128), (128, 16),
                                           (1, 64, 16), (1, 64, 16), (128,))
        primals = (x * scale, jax.nn.softplus(dt - 3.0), -jnp.exp(0.5 * a), b, c, skip)
        drawn = normal_draw(2978 + 4096 * seed, *(operand.shape for operand in primals))
        return primals, (drawn[0] * scale, *drawn[1:]), normal_draw(2979 + 4096 * seed, x.shape)[0]

    def scanned(*operands):
        with generation.chosen(selective_scan="mosaic_gpu"):
            return recurrence_chunk.selective_scan(*operands, None, None, portable=portable)[0]

    def plain(*operands):
        return portable(*operands, None, None, chunk_size=32)[0]

    assert_drawn_contract(jax.jit(scanned), draw, EDGES[edge], f"the selective scan {edge}",
                          truth_fn=jax.jit(plain), plain_fn=jax.jit(plain),
                          taking=taken, without=INTERPRETED_REMAT, seeds=seeds)


@pytest.mark.parametrize("edge", EDGES)
def test_the_chunked_op_holds_every_order(edge):
    """The chunked op (`recurrence_chunk.chunk`: its kernel a product, `_per_product`, differentiated
    through the portable math, and the skip `D v`, `recurrence.skipped`, each through
    `generation.linear_tangent`), the gated delta rule with a decay a head, its queries scaled, and the skip,
    from a zero memory: every order of its output finite wherever float64 is a normal float32 and the rule a
    token at a time with its skip is (`plain_delta_rule`, in float64 the truth), at queries, keys and values
    whose products near float32's ends; near float32's largest wherever fla's own chunked form with the skip
    is (`precedent_chunked`, as the portable rule's edge: ledger 2470)."""
    def draw(seed, scale):
        # Its own streams past seed zero (2680), its fixture the template cell's (2664).
        (query, key, value, g, beta, _), direction, _ = delta_rule_draw(
            2680, seed, scale, key_heads=2, value_heads=2, width=8, length=12, per_key=False, fixture=2664)
        (skip,) = normal_draw(draw_stream(2680, seed, 3, fixture=2667), (2,))
        (skip_direction,) = normal_draw(draw_stream(2680, seed, 4, fixture=2668), (2,))
        return ((query, key, value, g, beta, skip), (*direction[:5], skip_direction),
                normal_draw(draw_stream(2680, seed, 5, fixture=2669), value.shape)[0])
    hooks = recurrence.Hooks(decay="scalar", skip=True, scale=8 ** -0.5)

    def chunked(query, key, value, g, beta, skip):
        operands = recurrence_chunk.Chunk(query, key, value, log_decay=g, beta=beta, skip=skip)
        return recurrence_chunk.chunk(hooks, operands, chunk_size=4)[0]

    def recurrent(query, key, value, g, beta, skip):
        out, _ = plain_delta_rule(query, key, value, g, beta, jnp.zeros((1, 2, 8, 8), query.dtype),
                                  8 ** -0.5)
        return out + skip[:, None] * value

    def precedent(query, key, value, g, beta, skip):
        return precedent_chunked(query, key, value, g[..., None], beta, chunk_size=4) + skip[:, None] * value

    assert_drawn_contract(chunked, draw, EDGES[edge], f"the chunked op {edge}", truth_fn=recurrent,
                          plain_fn=precedent if edge == "near the largest" else recurrent)

@pytest.mark.parametrize("edge", EDGES_BEYOND)
def test_the_decode_step_holds_every_order(edge):
    """`assert_decode_contract` at an edge of float32's range."""
    assert_decode_contract(edge)


@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_the_decode_step_holds_every_order_over_normal_draws(request):
    """`assert_decode_contract` in the normal domain, check 1 over its declared draws."""
    assert_decode_contract("normal", seeds=marked_seeds(request))


def assert_decode_contract(edge: str, seeds: int | None = None) -> None:
    """A decode token in one launch (`recurrence.step`, the launch differentiated through its reference by
    `generation.linear_tangent`; interpreted off a GPU), the gated delta rule with a decay a head and the
    skip: every order of its output and updated memory finite wherever float64 is a normal float32 and the
    token's update as the rule states it is (`plain_delta_rule` over one token), at queries, keys, values
    and memory whose products near float32's ends, at a model's 128 keys and values a head (the launch's
    warpgroup takes a value column a thread)."""
    def draw(seed, scale):
        (query, key, value, g, beta, state), direction, cotangent = delta_rule_draw(
            2669, seed, scale, key_heads=1, value_heads=2, width=128, length=1, per_key=False)
        (skip,) = normal_draw(draw_stream(2669, seed, 3, fixture=2672), (2,))
        (skip_direction,) = normal_draw(draw_stream(2669, seed, 4, fixture=2673), (2,))
        flat = jnp.concatenate([cotangent[0][:, 0].ravel(), cotangent[1].ravel()])
        return (query, key, value, g, beta, state, skip), (*direction, skip_direction), flat
    hooks = recurrence.Hooks(decay="scalar", skip=True)

    def stepped(query, key, value, g, beta, state, skip):
        operands = recurrence.Step(state, query[:, 0], key[:, 0], value[:, 0], log_decay=g[:, 0],
                                   beta=beta[:, 0], skip=skip)
        out, _, updated = recurrence.step(hooks, operands)
        return out, updated

    def plain(query, key, value, g, beta, state, skip):
        out, final = plain_delta_rule(query, key, value, g, beta, state, 1.0)
        return out[:, 0] + skip[:, None] * value[:, 0], final

    assert_drawn_contract(flattened(stepped), draw, EDGES[edge], f"the decode step {edge}",
                          truth_fn=flattened(plain), plain_fn=flattened(plain), without=INTERPRETED_REMAT,
                          seeds=seeds)


def ssd_draw(seed: int, scale: float, rows: int = 1):
    """Mamba2's operands `(x, dt, a, b, c, skip, state)` over `rows` rows, their direction and the
    outputs' cotangents: inputs, the groups' input gates and the state at `scale`, positive steps, decays
    under zero."""
    x, steps, decay, b, c, skip, state = normal_draw(
        seed, (rows, 64, 2, 16), (rows, 64, 2), (2,), (rows, 64, 1, 16), (rows, 64, 1, 16), (2,),
        (rows, 2, 16, 16))
    primals = (x * scale, jax.nn.softplus(steps), -jnp.exp(decay), b * scale, c, skip, state * scale)
    drawn = normal_draw(seed + 1, *(t.shape for t in primals))
    direction = tuple(along * (scale if at in (0, 3, 6) else 1.0) for at, along in enumerate(drawn))
    return primals, direction, normal_draw(seed + 2, x.shape, state.shape)


# mamba_ssm's port near float32's largest: no transcription of mamba_ssm's chunked form to hold it to yet
# (ledger 2656).
@pytest.mark.parametrize("taken", list(ORDER_SETS))
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_ssd_port_holds_its_orders(edge, taken):
    """Mamba2's chunked scan on the port of mamba_ssm's kernels (`mamba_ssd.ssd_scan`, a custom VJP over
    Triton kernels, interpreted off a GPU) as every caller reaches it: Mamba-2's scan with the port named
    (`mamba2.chunk_ssd`), whose products' wrapper takes every order through the portable math
    (`_per_product`, row 2655): each order of `taken`'s set finite wherever float64 is a normal float32 and
    the recurrence a token at a time is (`plain_ssd`), at inputs, gates and states whose products near
    float32's least normal."""
    def draw(seed, scale):
        return ssd_draw(2674 + 4096 * seed, scale)

    def ported(*operands):
        with generation.chosen(ssd="pallas"):
            return chunk_ssd(*operands, chunk_size=64)

    assert_drawn_contract(jax.jit(ported), draw, EDGES[edge], f"the ssd port {edge}",
                          truth_fn=jax.jit(plain_ssd), plain_fn=jax.jit(plain_ssd),
                          taking=ORDER_SETS[taken])


# A TPU's scan near float32's largest is out of its contract: at the 2^63 draw 23 of the 731 outputs float64
# puts past 2^127 are infinite, the least 2^126.9 (a chunk's two terms summed past float32's largest), where
# mamba_ssm's own chunked scan leaves 55 of the same draw's outputs non-finite (row 2656).
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_tpu_ssd_scan_holds_its_orders(edge):
    """Mamba2's chunked scan on a TPU's Pallas kernel (`ssd._chunk_scan`, a custom VJP, interpreted off a
    TPU), its operands the inputs scaled by their steps, each head's gates and each step's decay `A dt`: its
    primal and reverse derivative finite wherever float64 is a normal float32 and the recurrence a token at a
    time is (`ssd_support.stepwise_scan`), at inputs, gates and states whose products near float32's least
    normal. Its other orders are its route's, the products' wrapper the other kernels' contracts hold
    (`_per_product`, row 2655), which a CPU cannot reach: its selector takes the kernel on a TPU alone, and
    under an assumed one it lowers the kernel itself (`ssd._interpreting`). Two rows, which a mesh's data axis
    splits as the rated scan launches the kernel (`recurrence_chunk.ssd_specs`)."""
    def draw(seed, scale):
        # The kernel's operands: the inputs scaled by their steps, each group's gates a head's, `A dt`.
        (x, steps, decay, b, c, _, state), direction, cotangent = ssd_draw(2677 + 4096 * seed, scale, rows=2)
        heads = tuple(jnp.repeat(t, 2, axis=2) for t in (b, c))
        primals = (x * steps[..., None], *heads, steps * decay, state)
        along = (direction[0], *(jnp.repeat(t, 2, axis=2) for t in direction[3:5]), direction[1],
                 direction[6])
        return primals, along, cotangent

    def tpu(x, b, c, a, state):
        # Launched per shard as the rated scan launches it: a kernel takes no axis the partitioner holds.
        def local(x, b, c, a, state, documents):
            return ssd._chunk_scan(x, b, c, a, state, documents, 64, "tpu")

        return manual_map(local, *recurrence_chunk.ssd_specs(x.shape))(x, b, c, a, state, None)

    assert_drawn_contract(tpu, draw, EDGES[edge], f"the TPU ssd scan {edge}", truth_fn=stepwise_scan,
                          plain_fn=stepwise_scan, without=REVERSE_ONLY)


@pytest.mark.parametrize("edge", EDGES)
def test_the_short_conv_holds_every_order(edge):
    """The short conv on its kernels (`short_conv.conv`, differentiated through XLA's conv by
    `generation.linear_tangent`; interpreted off a GPU) as a layer reaches them (`linear.causal_conv1d`, the
    kernels named, on an interpreted H100), with its bias and silu: every order finite wherever float64 is a
    normal float32 and the causal depthwise conv as its definition sums it is, at inputs and taps whose
    products near float32's ends. Two rows, which a mesh's data axis splits as the layer launches the kernels
    per shard of its rows (`linear._per_row_shard`)."""
    def draw(seed, scale):
        # A warpgroup's channels, the kernels' block (`templates.recurrence.conv.fits`).
        x, taps, bias = normal_draw(2677 + 4096 * seed, (2, 12, 128), (128, 4), (128,))
        primals = (x * scale, taps, bias * scale)
        drawn = normal_draw(2678 + 4096 * seed, x.shape, taps.shape, bias.shape)
        direction = tuple(along * (scale if at != 1 else 1.0) for at, along in enumerate(drawn))
        return primals, direction, normal_draw(2679 + 4096 * seed, x.shape)[0]

    def kernel(x, taps, bias):
        # Off a GPU the kernels interpreted at an H100's facts (`generation.assumed`'s `interpreted`).
        with (generation.assumed("sm90" if INTERPRETED else None, interpreted=INTERPRETED),
              generation.chosen(short_conv="mosaic_gpu")):
            return causal_conv1d(x, taps, activation=True, bias=bias)

    def plain(x, taps, bias):
        width = taps.shape[1]
        padded = jnp.pad(x, ((0, 0), (width - 1, 0), (0, 0)))
        summed = sum(padded[:, tap:tap + x.shape[1]] * taps[:, tap] for tap in range(width)) + bias
        return jax.nn.silu(summed)

    assert_drawn_contract(kernel, draw, EDGES[edge], f"the short conv {edge}", truth_fn=plain,
                          plain_fn=plain, without=INTERPRETED_REMAT)


@pytest.mark.parametrize("edge", EDGES)
def test_the_diagonal_scan_holds_its_orders(edge):
    """The diagonal scan on the template's kernels (`templates.recurrence.diagonal`, a custom VJP over its
    launches, interpreted off a GPU) as every caller reaches it (`kernels.diagonal.diagonal_scan` with the
    kernels named, whose products' wrapper takes every order through the portable math, `_per_product`),
    over two spans of 64 tokens at an interpreted H100's facts from a state: each token's output and the
    final state finite at every order wherever float64 is a normal float32 and the recurrence a token at a
    time is, at writes and states scaled to the edge, the log forget gates HGRN's."""
    def draw(seed, scale):
        x, g, state = normal_draw(1453 + 4096 * seed, (2, 70, 128), (2, 70, 128), (2, 128))
        primals = (x * scale, -jax.nn.softplus(-(g + 2.0)), state * scale)
        drawn = normal_draw(1454 + 4096 * seed, x.shape, g.shape, state.shape)
        direction = (drawn[0] * scale, drawn[1], drawn[2] * scale)
        return primals, direction, normal_draw(1455 + 4096 * seed, (2, 71, 128))[0]

    def joined(out, final):
        return jnp.concatenate([out, final[:, None].astype(out.dtype)], axis=1)

    def kernel(x, g, state):
        with (generation.assumed("sm90" if INTERPRETED else None, interpreted=INTERPRETED),
              generation.chosen(diagonal_scan="mosaic_gpu")):
            return joined(*diagonal.diagonal_scan(x, g, state))

    def plain(x, g, state):
        def step(h, token):
            h = jnp.exp(token[1]) * h + token[0]
            return h, h

        final, out = jax.lax.scan(step, state, (jnp.moveaxis(x, 1, 0), jnp.moveaxis(g, 1, 0)))
        return joined(jnp.moveaxis(out, 0, 1), final)

    assert_drawn_contract(kernel, draw, EDGES[edge], f"the diagonal scan {edge}", truth_fn=plain,
                          plain_fn=plain, without=INTERPRETED_REMAT)

def _gated(h, q, k, v, g, beta):
    """The delta rule decayed per key, each key's old value corrected toward the new one: `g` a scalar (the
    gated delta rule) or a vector of the keys' own (Kimi's delta attention)."""
    del q
    decayed = jnp.exp(g)[..., None] * h if g.ndim else jnp.exp(g) * h
    return decayed + beta * jnp.outer(k, v - k @ decayed)


def _read(h, q, k, v, g, beta):
    del k, v, g, beta
    return q @ h


# Near float32's largest a chunk's corrections' overlaps (keys' products near 2^126 against states as large)
# overflow at entries the token recurrence's float64 keeps normal, as fla's chunked form's do at the same
# entries (ledger 2470, `precedent_chunked`).
@pytest.mark.parametrize("taken", list(ORDER_SETS))
@pytest.mark.parametrize("per_key", [False, True], ids=["gated_delta", "kimi_delta"])
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_derived_delta_rule_chunks_hold_their_orders(per_key, edge, taken):
    """`dew.algebra.scan`'s chunk forms of the delta rule, its decay a scalar (the gated delta rule) or each
    key's own (Kimi's delta attention), walked over three chunks from a zero state through their custom
    derivatives (the decays' range sums, `_decays_of`; the solve's inverse, `_lowered`): every order,
    forward mode among them, finite wherever float64 is a normal float32 and the token-by-token scan's are,
    at unit queries, keys and values and at ones whose products near float32's least normal, an order set a
    case (`ORDER_SETS`)."""
    tokens, chunk, keys_, values = 24, 8, 4, 6
    def draw(seed, scale):
        draws = jax.random.split(seeded_key(3311 + per_key + 4096 * seed), 7)
        q, k, v = (jax.random.normal(d, (tokens, n)) * scale
                   for d, n in zip(draws[:3], (keys_, keys_, values), strict=True))
        g = -jax.nn.softplus(jax.random.normal(draws[3], (tokens, keys_) if per_key else (tokens,)))
        beta = jax.nn.sigmoid(jax.random.normal(draws[4], (tokens,)))
        primals = (q, k, v, g, beta)
        tangents = tangent_keys(draws[5], len(primals), seed)
        direction = tuple(jax.random.normal(along, x.shape)
                          for along, x in zip(tangents, primals, strict=True))
        return primals, direction, jax.random.normal(draws[6], (tokens, values))
    axes = {"q": ("time", "key"), "k": ("time", "key"), "v": ("time", "value"),
            "g": ("time", "key") if per_key else ("time",), "beta": ("time",)}
    forms = scan.derive_scan(scan.Scan(_gated, _read, axes, walked="time",
                                       state=((keys_, values), jnp.float32)), saturation=SATURATION)

    def derived(q, k, v, g, beta):
        values_ = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        state, outputs = forms.init({}), []
        for index in range(tokens // chunk):
            part = {name: value[index * chunk:(index + 1) * chunk] for name, value in values_.items()}
            prepared = forms.prepare(part)
            outputs.append(forms.mapped(prepared, state))
            state = forms.step(state, prepared)
        return jnp.concatenate(outputs)

    def tokenwise(q, k, v, g, beta):
        def one(h, token):
            h = _gated(h, *token)
            return h, _read(h, *token)
        return jax.lax.scan(one, jnp.zeros((keys_, values), q.dtype), (q, k, v, g, beta))[1]

    scanned = jax.jit(tokenwise)
    assert_drawn_contract(jax.jit(derived), draw, EDGES[edge], f"the derived delta rule {edge}",
                          truth_fn=scanned, plain_fn=scanned, taking=ORDER_SETS[taken])

@pytest.mark.parametrize("edge", EDGES)
def test_the_derived_delta_rules_chunk_adjoint_holds_its_orders(edge):
    """`dew.algebra.scan`'s chunk adjoint (`backward`, its token update replayed per chunk) walked back over
    three chunks of the gated delta rule: the queries' gradient, and every order of it, forward mode among
    them, finite wherever float64 is a normal float32 and the token-by-token scan's gradient is."""
    tokens, chunk, keys_, values = 24, 8, 4, 6
    draws = jax.random.split(jax.random.key(3312), 6)
    scale = EDGES[edge]
    q, k, v = (jax.random.normal(d, (tokens, n)) * scale for d, n in zip(draws[:3], (keys_, keys_, values),
                                                                         strict=True))
    g = -jax.nn.softplus(jax.random.normal(draws[3], (tokens,)))
    beta = jax.nn.sigmoid(jax.random.normal(draws[4], (tokens,)))
    d_out = jax.random.normal(draws[5], (tokens, values))
    axes = {"q": ("time", "key"), "k": ("time", "key"), "v": ("time", "value"), "g": ("time",),
            "beta": ("time",)}
    forms = scan.derive_scan(scan.Scan(_gated, _read, axes, walked="time",
                                       state=((keys_, values), jnp.float32)), saturation=SATURATION)

    def gradient(q, k, v, g, beta):
        values_ = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        parts = [{name: value[index * chunk:(index + 1) * chunk] for name, value in values_.items()}
                 for index in range(tokens // chunk)]
        states = [forms.init({})]
        for part in parts:
            states.append(forms.step(states[-1], forms.prepare(part)))
        d_state, d_queries = jnp.zeros((keys_, values)), []
        for index in reversed(range(len(parts))):
            d_state, d_chunk = forms.backward(parts[index], states[index], d_state,
                                              d_out[index * chunk:(index + 1) * chunk])
            d_queries.append(d_chunk["q"])
        return jnp.concatenate(d_queries[::-1])

    def tokenwise(q, k, v, g, beta):
        def one(h, token):
            h = _gated(h, *token)
            return h, _read(h, *token)

        def loss(q):
            outputs = jax.lax.scan(one, jnp.zeros((keys_, values), q.dtype), (q, k, v, g, beta))[1]
            return jnp.sum(outputs * d_out)
        return jax.grad(loss)(q)

    primals = (q, k, v, g, beta)
    direction = tuple(jax.random.normal(draws[5], x.shape) for x in primals)
    cotangent = jax.random.normal(draws[0], (tokens, keys_))
    assert_contract(gradient, primals, direction, cotangent, f"the derived delta-rule adjoint {edge}",
                    truth_fn=tokenwise, plain_fn=tokenwise)


TURNED_STARTS = np.zeros((1, 40), bool)
TURNED_STARTS[0, [0, 17]] = True
"""The turned call's contract's two packed documents."""


def _framed_scores(query, key, value, starts):
    """A turned call's `(output, final memory in its last frame)` in plain arithmetic from its frames' keys
    and queries (`query`, `key` `[B, T, H, K]`), its decay `exp(-0.05)` a token, write gate one, scale one, a
    readout after its update: every pair of one document summed whole, no chunk."""
    length = query.shape[1]
    ids = jnp.cumsum(jnp.asarray(starts, jnp.int32), axis=1)
    same = (ids[:, :, None] == ids[:, None, :]) & jnp.tril(jnp.ones((length, length), bool))[None]
    lag = jnp.arange(length)[:, None] - jnp.arange(length)[None, :]
    weight = jnp.where(same, jnp.exp(-0.05 * lag)[None], 0.0)
    out = jnp.einsum('bsu,bshk,buhk,buhv->bshv', weight, query, key, value)
    last = (ids == ids[:, -1:]) * jnp.exp(-0.05 * (length - 1 - jnp.arange(length)))[None]
    return out, jnp.einsum('bu,buhk,buhv->bhkv', last, key, value)


def _turned_plainly(query, key, value, turn):
    """`_framed_scores` of keys and queries turned by their documents' summed turns, each plainly (the
    contract's truth and plain formula)."""
    ids = jnp.cumsum(jnp.asarray(TURNED_STARTS, jnp.int32), axis=1)
    within = (ids[:, :, None] == ids[:, None, :]) & jnp.tril(jnp.ones((ids.shape[1],) * 2, bool))[None]
    angle = jnp.einsum('bsu,buha->bsha', within.astype(turn.dtype), turn)
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    out, held = _framed_scores(*(scan.rotated(x, cos, sin) for x in (query, key)), value, TURNED_STARTS)
    return out, scan.rotated(held, cos[:, -1, :, None], -sin[:, -1, :, None], axis=-2)


@pytest.mark.parametrize("taken", list(ORDER_SETS))
@pytest.mark.parametrize("edge", EDGES)
def test_a_turned_template_call_holds_every_order(edge, taken):
    """A turned call on the template (`templates.recurrence.turned.call`: its keys and queries in their
    documents' frames, its turns' gradient in the crossing form) over a plain launch of the template's
    arithmetic (`_framed_scores`), its keys, queries and values at `edge` and its turns of up to 3 radians:
    every order finite where the plain formula's is (`_turned_plainly`, its float64 the truth; row 7419),
    forward mode through its launch's own tangent (row 8573), an order set a case (`ORDER_SETS`)."""
    shapes = ((1, 40, 1, 6),) * 2 + ((1, 40, 1, 2),)
    def draw(seed, scale):
        operands = normal_draw(draw_stream(7419, seed, 0), *shapes, scale=scale)
        turns = np.random.default_rng(draw_stream(7419, seed, 1, fixture=7419))
        turn = jnp.asarray(turns.uniform(-3.0, 3.0, (1, 40, 1, 2)), jnp.float32)
        direction = normal_draw(draw_stream(7419, seed, 2, fixture=7420), *shapes, (1, 40, 1, 2), scale=scale)
        return ((*operands, turn), direction,
                normal_draw(draw_stream(7419, seed, 3, fixture=7421), (1, 40, 1, 2), (1, 1, 6, 2)))
    ones, decays = jnp.ones((1, 40, 1), jnp.float32), jnp.full((1, 40, 1), -0.05, jnp.float32)

    def launch(context, query, key, value, g, beta, state, skip, entry, turning):
        del context, g, beta, state, skip, entry, turning
        return _framed_scores(query, key, value, TURNED_STARTS)

    def rule(query, key, value, turn):
        return turned.call(launch, query, key, value, decays, ones, None, None, turn, TURNED_STARTS,
                           scale=1.0, group=1)

    plainly = jax.jit(_turned_plainly)
    assert_drawn_contract(jax.jit(rule), draw, EDGES[edge], f"a turned template call {edge}",
                          truth_fn=plainly, plain_fn=plainly, taking=ORDER_SETS[taken])
