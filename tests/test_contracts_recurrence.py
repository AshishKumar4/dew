"""Float-range and nested-derivative contracts of the recurrences' seams: the chunked delta rule, its ports
and template, the decode step, the SSD port and the TPU scan, the short convolution (ledger 2641; the
harness is tests/contract_support.py)."""

import functools
import itertools

import delta_rule_support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from contract_support import EDGES, REVERSE_ONLY, assert_contract, normal_draw
from ssd_support import stepwise_scan

from dew.nn.kernels import fla_gdn, fla_kda, mamba_ssd, recurrence, recurrence_chunk, short_conv, ssd
from dew.nn.kernels.templates.recurrence import mainloop

DELTA_RULE_EDGES = {"normal": 1.0, "the largest its chunk holds": 2.0**62, "near the largest": 2.0**63,
                    "near the least normal": 2.0**-63}
"""The chunked delta rule's edges. Near float32's largest its chunk's own products overflow at entries the
token recurrence keeps finite, as fla's chunked form's do at the same entries (`precedent_chunked`, on an
H100: ledger 2470), which that edge compares it with in place of the recurrence."""


INTERPRETED = jax.default_backend() != "gpu"
"""The ported kernels interpret off a GPU."""


def flattened(fn):
    """`fn`'s outputs as one vector, which the orders through its gradient sum against one cotangent."""
    return lambda *operands: jnp.concatenate([out.ravel() for out in jax.tree.leaves(fn(*operands))])


PACKED_OVERFLOW_ORDERS = ("primal", "jvp", "vjp", "second jvp", "second jvp, remat")
"""The orders a packed row of the chunked delta rule holds from 2^62: all but the second orders through its
gradient, forward over reverse and reverse over reverse, which the owner's decision accepts there (one
cause, documents sharing chunks, an overflow-only difference: ledger 2470;
`test_the_chunked_delta_rule_holds_every_order`)."""


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


@pytest.mark.parametrize(("packed", "edge"), list(itertools.product((False, True), DELTA_RULE_EDGES)),
                         ids=lambda value: {False: "one sequence", True: "packed"}.get(value, value))
def test_the_chunked_delta_rule_holds_every_order(packed, edge):
    """`recurrence_chunk.chunk_delta_rule` (the portable rule's custom derivative through
    `generation.linear_tangent`, rows 1861 and 1862): every order runs, its boolean document starts a
    residual with no tangent space, and each is finite wherever float64 is a normal float32 and JAX's own
    derivatives of the token recurrence are (`recurrent_delta_rule`, an independent formula of the same
    rule, in float64 the truth too: the chunk's own float64 run excused its float32 NaNs wherever it was NaN
    itself), near float32's largest of fla's own chunked form (`precedent_chunked`), at queries, keys and
    values whose products near float32's ends, the decays and gates as drawn (`DELTA_RULE_EDGES`). A pair
    the chunk weighs zero once multiplied its overflowed tangent by zero, the queries' second derivative NaN
    from 2^48 (ledger 2470).

    A packed row's documents share chunks, so from 2^62 one document's overflow reaches another's forward
    over reverse (ledger 2608); its other orders hold. The owner accepted that under the accuracy-for-speed
    rule (ledger 2470, 10-30): documents in chunks of their own held every order and cost the portable
    packed gradient 25% (JS-cost2470d-h100-80113cc1c) for a difference at overflow alone, where the
    parameters' gradient is NaN either way."""
    keys = jax.random.split(jax.random.key(1861), 8)
    rows, length, heads, width = 1, 37, 2, 8
    scale = DELTA_RULE_EDGES[edge]
    query, key, value = (jax.random.normal(k, (rows, length, heads, width)) * scale for k in keys[:3])
    g = -jax.nn.softplus(jax.random.normal(keys[3], (rows, length, heads, 1)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (rows, length, heads)))
    starts = jax.random.bernoulli(keys[5], 0.1, (rows, length)) if packed else None

    def rule(query, key, value, g, beta):
        return recurrence_chunk.chunk_delta_rule(query, key, value, g, beta, chunk_size=16, starts=starts)[0]

    def recurrence(query, key, value, g, beta):
        return recurrence_chunk.recurrent_delta_rule(query, key, value, g, beta, starts=starts)[0]

    plain = (functools.partial(precedent_chunked, chunk_size=16, starts=starts) if edge == "near the largest"
             else recurrence)
    primals = (query, key, value, g, beta)
    direction = tuple(jax.random.normal(keys[6], x.shape) * (scale if at < 3 else 1.0)
                      for at, x in enumerate(primals))
    cotangent = jax.random.normal(keys[7], query.shape)
    # A packed row's second orders through its gradient from 2^62: the owner's accuracy-for-speed acceptance
    # (ledger 2470, 10-30).
    accepted = packed and scale >= 2.0**62
    assert_contract(rule, primals, direction, cotangent, f"the chunked delta rule {edge}",
                    truth_fn=recurrence, plain_fn=plain, held=PACKED_OVERFLOW_ORDERS if accepted else None)


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


def delta_rule_draw(seed: int, scale: float, *, key_heads: int, value_heads: int, width: int, length: int,
                    per_key: bool, unit_keys: bool = False):
    """A delta rule's operands `(query, key, value, g, beta, state)`, their direction and the outputs'
    cotangents: queries, values and memory at `scale`, keys at it or of unit norm, a log decay in (-inf, 0)
    a head or a key dimension, write gates in (0, 1)."""
    query, key, value, decay, gate, state = normal_draw(
        seed, (1, length, key_heads, width), (1, length, key_heads, width), (1, length, value_heads, width),
        (1, length, value_heads, *((width,) if per_key else ())), (1, length, value_heads),
        (1, value_heads, width, width))
    key = key / jnp.linalg.norm(key, axis=-1, keepdims=True) if unit_keys else key * scale
    primals = (query * scale, key, value * scale, -jax.nn.softplus(decay), jax.nn.sigmoid(gate),
               state * scale)
    drawn = normal_draw(seed + 1, *(x.shape for x in primals))
    direction = tuple(along * (scale if at in (0, 2, 5) or (at == 1 and not unit_keys) else 1.0)
                      for at, along in enumerate(drawn))
    cotangent = normal_draw(seed + 2, value.shape, state.shape)
    return primals, direction, cotangent


# The ports near float32's largest: their reverse derivatives are NaN at entries the recurrence keeps finite,
# the gated delta rule's where fla's own chunked form keeps them too (ledger 2656).
@pytest.mark.parametrize("per_key", [False, True], ids=["fla_gdn", "fla_kda"])
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_delta_rule_ports_hold_their_orders(per_key, edge):
    """fla's chunked gated delta rule and Kimi delta attention on the ported kernels
    (`fla_gdn.chunk_gated_delta_rule`, `fla_kda.chunk_kda`, custom VJPs over Triton kernels, interpreted off a
    GPU), from a zero memory: the primal and reverse derivative of the output finite wherever float64 is a
    normal float32 and the delta rule a token at a time is (`plain_delta_rule`, in float64 the truth), at
    queries, keys and values whose products near float32's least normal. Forward mode and reverse over
    reverse do not run (`REVERSE_ONLY`)."""
    (query, key, value, g, beta, _), direction, _ = delta_rule_draw(
        2661 + per_key, EDGES[edge], key_heads=2, value_heads=2, width=64, length=64, per_key=per_key)
    (cotangent,) = normal_draw(2663, value.shape)
    memory = jnp.zeros((1, 2, 64, 64), jnp.float32)
    scale = 64 ** -0.5
    port = fla_kda.chunk_kda if per_key else fla_gdn.chunk_gated_delta_rule

    def ported(*operands):
        return port(*operands, memory, None, scale, 64, INTERPRETED)[0]

    def recurrent(*operands):
        return plain_delta_rule(*operands, memory.astype(operands[0].dtype), scale)[0]

    assert_contract(ported, (query, key, value, g, beta), direction[:5], cotangent,
                    f"the {'kda' if per_key else 'gdn'} port {edge}", truth_fn=recurrent, plain_fn=recurrent,
                    without=REVERSE_ONLY)


# The template rule near float32's largest: its reverse derivative is NaN at entries fla's chunked form keeps
# finite (ledger 2656).
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_template_rule_holds_its_orders(edge):
    """The recurrence template's chunked rule (`recurrence_chunk.mosaic_rule`, a custom VJP over the
    mainloop's launches, interpreted off a GPU), a 16-bit call with its chunk inverse's declared bound (keys
    of unit norm, sigmoid write gates: its domain), from a zero memory: the primal and reverse derivative of
    the output finite wherever float64 is a normal float32 and the delta rule a token at a time is
    (`plain_delta_rule`), at queries and values whose products near float32's least normal. Forward mode and
    reverse over reverse do not run (`REVERSE_ONLY`)."""
    (query, key, value, g, beta, _), direction, _ = delta_rule_draw(
        2664, EDGES[edge], key_heads=1, value_heads=1, width=64, length=64, per_key=False, unit_keys=True)
    primals = (query.astype(jnp.bfloat16), key.astype(jnp.bfloat16), value.astype(jnp.bfloat16), g,
               beta.astype(jnp.bfloat16))
    direction = tuple(along.astype(operand.dtype) for along, operand in zip(direction, primals, strict=False))
    (cotangent,) = normal_draw(2665, value.shape)

    def rule(*operands):
        return recurrence_chunk.mosaic_rule(*operands, tiles=mainloop.Tiles(32, 1), bound=1.0)[0]

    def recurrent(*operands):
        operands = tuple(x.astype(jnp.promote_types(x.dtype, jnp.float32)) for x in operands)
        return plain_delta_rule(*operands, jnp.zeros((1, 1, 64, 64), operands[0].dtype), 64 ** -0.5)[0]

    assert_contract(rule, primals, direction, cotangent, f"the template rule {edge}", truth_fn=recurrent,
                    plain_fn=recurrent, without=REVERSE_ONLY)


@pytest.mark.parametrize("edge", EDGES)
def test_the_chunked_op_holds_every_order(edge):
    """The chunked op (`recurrence_chunk.chunk`: its kernel a product, `_per_product`, differentiated
    through the portable math, and the skip `D v`, `recurrence.skipped`, each through
    `generation.linear_tangent`), the gated delta rule with a decay a head, its queries scaled, and the skip,
    from a zero memory: every order of its output finite wherever float64 is a normal float32 and the rule a
    token at a time with its skip is (`plain_delta_rule`, in float64 the truth), at queries, keys and values
    whose products near float32's ends; near float32's largest wherever fla's own chunked form with the skip
    is (`precedent_chunked`, as the portable rule's edge: ledger 2470)."""
    (query, key, value, g, beta, _), direction, _ = delta_rule_draw(
        2664, EDGES[edge], key_heads=2, value_heads=2, width=8, length=12, per_key=False)
    (skip,) = normal_draw(2667, (2,))
    (skip_direction,) = normal_draw(2668, (2,))
    (cotangent,) = normal_draw(2669, value.shape)
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

    assert_contract(chunked, (query, key, value, g, beta, skip), (*direction[:5], skip_direction), cotangent,
                    f"the chunked op {edge}", truth_fn=recurrent,
                    plain_fn=precedent if edge == "near the largest" else recurrent)


@pytest.mark.parametrize("edge", EDGES)
def test_the_decode_step_holds_every_order(edge):
    """A decode token in one launch (`recurrence.step`, the launch differentiated through its reference by
    `generation.linear_tangent`; interpreted off a GPU), the gated delta rule with a decay a head and the
    skip: every order of its output and updated memory finite wherever float64 is a normal float32 and the
    token's update as the rule states it is (`plain_delta_rule` over one token), at queries, keys, values
    and memory whose products near float32's ends."""
    (query, key, value, g, beta, state), direction, cotangent = delta_rule_draw(
        2669, EDGES[edge], key_heads=1, value_heads=2, width=8, length=1, per_key=False)
    (skip,) = normal_draw(2672, (2,))
    (skip_direction,) = normal_draw(2673, (2,))
    hooks = recurrence.Hooks(decay="scalar", skip=True)

    def stepped(query, key, value, g, beta, state, skip):
        operands = recurrence.Step(state, query[:, 0], key[:, 0], value[:, 0], log_decay=g[:, 0],
                                   beta=beta[:, 0], skip=skip)
        out, _, updated = recurrence.step(hooks, operands)
        return out, updated

    def plain(query, key, value, g, beta, state, skip):
        out, final = plain_delta_rule(query, key, value, g, beta, state, 1.0)
        return out[:, 0] + skip[:, None] * value[:, 0], final

    flat_cotangent = jnp.concatenate([cotangent[0][:, 0].ravel(), cotangent[1].ravel()])
    assert_contract(flattened(stepped), (query, key, value, g, beta, state, skip),
                    (*direction, skip_direction), flat_cotangent, f"the decode step {edge}",
                    truth_fn=flattened(plain), plain_fn=flattened(plain))


def ssd_draw(seed: int, scale: float):
    """Mamba2's operands `(x, dt, a, b, c, skip, state)`, their direction and the outputs' cotangents: inputs,
    the groups' input gates and the state at `scale`, positive steps, decays under zero."""
    x, steps, decay, b, c, skip, state = normal_draw(
        seed, (1, 64, 2, 16), (1, 64, 2), (2,), (1, 64, 1, 16), (1, 64, 1, 16), (2,), (1, 2, 16, 16))
    primals = (x * scale, jax.nn.softplus(steps), -jnp.exp(decay), b * scale, c, skip, state * scale)
    drawn = normal_draw(seed + 1, *(t.shape for t in primals))
    direction = tuple(along * (scale if at in (0, 3, 6) else 1.0) for at, along in enumerate(drawn))
    return primals, direction, normal_draw(seed + 2, x.shape, state.shape)


# mamba_ssm's port near float32's largest: no transcription of mamba_ssm's chunked form to hold it to yet
# (ledger 2656).
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_ssd_port_holds_its_orders(edge):
    """Mamba2's chunked scan on the port of mamba_ssm's kernels (`mamba_ssd.ssd_scan`, a custom VJP over
    Triton kernels, interpreted off a GPU): its primal and reverse derivative finite wherever float64 is a
    normal float32 and the recurrence a token at a time is (`plain_ssd`), at inputs, gates and states whose
    products near float32's least normal. Forward mode and reverse over reverse do not run
    (`REVERSE_ONLY`)."""
    primals, direction, cotangent = ssd_draw(2674, EDGES[edge])

    def ported(*operands):
        return mamba_ssd.ssd_scan(*operands, None, 64, INTERPRETED)

    assert_contract(ported, primals, direction, cotangent, f"the ssd port {edge}", truth_fn=plain_ssd,
                    plain_fn=plain_ssd, without=REVERSE_ONLY)


# A TPU's scan near float32's largest: its output is infinite at an entry the recurrence keeps finite
# (ledger 2656).
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_tpu_ssd_scan_holds_its_orders(edge):
    """Mamba2's chunked scan on a TPU's Pallas kernel (`ssd.ssd_chunk_scan`, a custom VJP, interpreted off a
    TPU), its operands the inputs scaled by their steps, each head's gates and each step's decay `A dt`: its
    primal and reverse derivative finite wherever float64 is a normal float32 and the recurrence a token at a
    time is (`ssd_support.stepwise_scan`), at inputs, gates and states whose products near float32's least
    normal. Forward mode and reverse over reverse do not run (`REVERSE_ONLY`)."""
    (x, steps, decay, b, c, _, state), direction, cotangent = ssd_draw(2677, EDGES[edge])

    def tpu(x, b, c, a, state):
        return ssd.ssd_chunk_scan(x, b, c, a, state, 64, "tpu")

    heads = tuple(jnp.repeat(t, 2, axis=2) for t in (b, c))
    primals = (x * steps[..., None], *heads, steps * decay, state)
    along = (direction[0], *(jnp.repeat(t, 2, axis=2) for t in direction[3:5]), direction[1], direction[6])
    assert_contract(tpu, primals, along, cotangent, f"the TPU ssd scan {edge}", truth_fn=stepwise_scan,
                    plain_fn=stepwise_scan, without=REVERSE_ONLY)


@pytest.mark.parametrize("edge", EDGES)
def test_the_short_conv_holds_every_order(edge):
    """The short conv on its kernels (`short_conv.conv`, differentiated through XLA's conv by
    `generation.linear_tangent`; interpreted off a GPU), with its bias and silu: every order finite wherever
    float64 is a normal float32 and the causal depthwise conv as its definition sums it is, at inputs and
    taps whose products near float32's ends."""
    scale = EDGES[edge]
    x, taps, bias = normal_draw(2677, (1, 12, 16), (16, 4), (16,))
    primals = (x * scale, taps, bias * scale)
    direction = tuple(along * (scale if at != 1 else 1.0)
                      for at, along in enumerate(normal_draw(2678, x.shape, taps.shape, bias.shape)))
    (cotangent,) = normal_draw(2679, x.shape)

    def kernel(x, taps, bias):
        return short_conv.conv(x, taps, bias, None, None, ((16, None),), activation=True, eps=1e-6,
                               dtype=x.dtype, interpret=INTERPRETED)[0]

    def plain(x, taps, bias):
        width = taps.shape[1]
        padded = jnp.pad(x, ((0, 0), (width - 1, 0), (0, 0)))
        summed = sum(padded[:, tap:tap + x.shape[1]] * taps[:, tap] for tap in range(width)) + bias
        return jax.nn.silu(summed)

    assert_contract(kernel, primals, direction, cotangent, f"the short conv {edge}", truth_fn=plain,
                    plain_fn=plain)
