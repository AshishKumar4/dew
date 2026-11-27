"""Printable recurrence components, held off the runtime path (`dew.algebra.generate`, docs/design/algebra.md
§6.11): the chunked delta rule's scores, carries and solver, KDA's and GDN's gated pieces, the Mamba-SSD
chunk and the selective scan's stages, forward and backward."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra.generate import Cases, ComponentForm, declare, float32_cases, normal_operands
from dew.nn.kernels import recurrence_algebra, ssd
from dew.nn.kernels.recurrence_algebra import apart
from dew.nn.kernels.templates.operands import Parts
from dew.nn.kernels.templates.recurrence import selective


def _scores_truth(products, decays, scales):
    return jnp.tril(products * (decays * scales))


def _apart_truth(rows, rows_lo, columns, columns_lo):
    left = rows + rows_lo
    right = columns + columns_lo
    return left[:, None] - right[None, :]


def _apart_draw(seed, operands):
    rows, rows_lo, columns, columns_lo = normal_operands(seed, operands)
    if seed == 1:
        # Rounding either compensated prefix before subtraction would lose every difference.
        rows.fill(2 ** 24)
        columns.fill(2 ** 24)
        rows_lo.fill(0.25)
        columns_lo.fill(0)
    return rows, rows_lo, columns, columns_lo


def _carry_product(name, left, right):
    del name
    left, right = (x.value() if isinstance(x, Parts) else x for x in (left, right))
    return jnp.matmul(left, right, precision=recurrence_algebra.STATE_PRECISION)


def _chunk_carry(state, keys, values, weights, kept):
    corrected = recurrence_algebra.corrected(values, lambda _: weights, [state], keys.dtype, _carry_product)
    (leaving,) = recurrence_algebra.carried([state], lambda _: keys, corrected, [kept], keys.dtype,
                                           _carry_product)
    return leaving, corrected


def _chunk_carry_truth(state, keys, values, weights, kept):
    corrected = values - weights @ state
    return kept * state + keys.T @ corrected, corrected


def _chunk_pullback(grad, keys, weights, queries, out_grad, local, kept):
    key_tile, weight_tile, query_tile = (Parts((a,), a.dtype) for a in (keys, weights, queries))
    (entered,), corrected = recurrence_algebra.key_reverse_carried(
        [grad], lambda _: key_tile, lambda _: weight_tile, lambda _: query_tile,
        Parts((out_grad,), out_grad.dtype), local, [kept], keys.dtype, _carry_product)
    return entered, corrected


def _chunk_pullback_truth(grad, keys, weights, queries, out_grad, local, kept):
    corrected = keys @ grad + local
    return kept * grad + queries.T @ out_grad - weights.T @ corrected, corrected


def _chunk_inputs(seed, operands):
    state, keys, values, weights, kept = normal_operands(seed, operands)
    kept[:] = 2.0 ** -np.abs(np.round(kept))
    # Padding supplies no write or correction. A huge surviving memory needs its own decay.
    keys[0], values[0], weights[0] = 0, 0, 0
    if seed % 2:
        state[0], weights[:, 0], kept[0] = 2.0 ** 127, 0, 2.0 ** -126
    return state, keys, values, weights, kept


def _chunk_pullback_inputs(seed, operands):
    grad, keys, weights, queries, out_grad, local, kept = normal_operands(seed, operands)
    kept[:] = 2.0 ** -np.abs(np.round(kept))
    keys[0], weights[0], queries[0], out_grad[0], local[0] = 0, 0, 0, 0, 0
    if seed % 2:
        grad[0], keys[:, 0], kept[0] = 2.0 ** 127, 0, 2.0 ** -126
    return grad, keys, weights, queries, out_grad, local, kept


def _chunk_level(system, inverse, level):
    rows, cols = recurrence_algebra.tile_indices(system.shape)
    return recurrence_algebra.doubled_level(-system, inverse, rows, cols, level, _carry_product)


def _chunk_level_truth(system, inverse, level):
    width = 1 << level
    groups = jnp.arange(system.shape[0]) // width
    join = (groups[:, None] % 2 == 1) & (groups[:, None] == groups[None, :] + 1)
    return inverse + jnp.where(join, inverse @ -system, 0) @ inverse


def _chunk_level_draw(seed, operands):
    size = operands[0].shape[0]
    system = _chunk_system(seed, size)
    level = 1 + seed % (size.bit_length() - 2)
    group = np.arange(size) // (1 << level)
    identity = np.eye(size, dtype=np.float64)
    diagonal = np.where(group[:, None] == group[None, :], system, 0)
    inverse = np.linalg.solve(diagonal + identity, identity).astype(system.dtype)
    return system, inverse, np.asarray(level, np.int32)


def _inverse_pullback(lower, cotangent):
    def issue(_, left, right):
        return jnp.matmul(left, right, precision=recurrence_algebra.STATE_PRECISION,
                          preferred_element_type=jnp.float32)

    return recurrence_algebra.key_system_back(cotangent, lower, lower.dtype, issue)


def _inverse_pullback_truth(lower, cotangent):
    inverse = jnp.eye(lower.shape[0], dtype=lower.dtype) + lower
    return jnp.tril(-(inverse.T @ jnp.tril(cotangent, -1)) @ inverse.T, -1)


def _inverse_pullback_draw(seed, operands):
    values, cotangent = normal_operands(seed, operands)
    lower = np.tril(values / (2 * values.shape[0]), -1)
    if seed % 2:
        cotangent[np.triu_indices(values.shape[0], 1)] = np.nan
        np.fill_diagonal(cotangent, np.inf)
    else:
        lower[:2], lower[2:, :2] = 0, 0
        cotangent[:2], cotangent[2:, :2] = 0, 0
        cotangent[1, 0] = 2.0 ** 127
    return lower, cotangent


def _chunk_weighted(lower, values, strengths):
    return recurrence_algebra.weighted("apply_u", lower, values, strengths, _carry_product)


def _chunk_weighted_truth(lower, values, strengths):
    return (jnp.eye(lower.shape[0], dtype=lower.dtype) + lower) @ (values * strengths[:, None])


def _chunk_system(seed, size):
    rng = np.random.default_rng(seed)
    system = np.tril(rng.standard_normal((size, size)).astype(np.float32) / (2 * size), -1)
    # Split at a padded token: neither side may read across the block boundary.
    middle = system.shape[0] // 2
    system[middle:, :middle] = 0
    if seed % 2:
        # Isolate a top-binade edge: the inverse holds it without a product of two huge edges.
        system[:2], system[2:, :2] = 0, 0
        system[1, 0] = 2.0 ** 127
    return system


def _chunk_weighted_draw(seed, operands):
    lower, values, weights = normal_operands(seed, operands)
    lower = np.tril(lower * 0.25, -1)
    weights = (np.abs(weights) * 0.25).astype(weights.dtype)
    weights[1] = 0
    if seed % 2:
        lower[1, 0], values[0], weights[0] = 2.0 ** 127, 0.125, 0.5
    return lower, values, weights


def _selective_decay_truth(step, rate, kept=None):
    decay = jnp.exp(step * rate)
    return decay if kept is None else jnp.where(kept != 0, decay, 0)


def _selective_decay_draw(seed, operands):
    rng = np.random.default_rng(seed)
    step = rng.uniform(0, 2, operands[0].shape).astype(np.float32)
    rate = np.asarray(-rng.uniform(0, 2, operands[1].shape), dtype=np.float32)
    if seed == 1:
        step[:] = 0
        rate[...] = -np.float32(3 * 2.0 ** 126)
    elif seed == 2:
        step[:] = np.float32(3 * 2.0 ** 126)
        rate[...] = -np.float32(2.0 ** -126)
    elif seed == 3:
        rate[...] = 0
    kept = () if len(operands) == 2 else ((rng.random(operands[2].shape) > 0.25).astype(np.float32),)
    return step, rate, *kept


def _key_memory_decay_truth(entered, leaving):
    return jnp.einsum("kv,kv->k", entered, leaving)


def _key_memory_decay_draw(seed, operands):
    entered, leaving = normal_operands(seed, operands)
    if seed == 1:
        # A large carry and its reciprocal-scale cotangent produce ordinary finite terms.
        entered[0, :4], leaving[0, :4] = [2.0 ** 60, -2.0 ** 60, 2.0 ** 60, 0], 2.0 ** -60
    elif seed == 2:
        entering_mask = np.arange(entered.shape[0]) % 2 == 0
        entered[entering_mask] = 0
    return entered, leaving


SCORES = declare(__name__, "scores", ComponentForm(
    recurrence_algebra.scores_tile, _scores_truth,
    float32_cases(*[((n, n), (n, n), (n,)) for n in (16, 32, 64)])))


PREFIX_DIFFERENCE = declare(__name__, "prefix_difference", ComponentForm(
    apart, _apart_truth,
    dataclasses.replace(float32_cases(((32,), (32,), (64,), (64,))), draw=_apart_draw)))


CHUNK_CARRY = declare(__name__, "chunk_carry", ComponentForm(
    _chunk_carry, _chunk_carry_truth, dataclasses.replace(
        float32_cases(*[((k, v), (c, k), (c, v), (c, k), (k, 1))
                        for c, k, v in ((16, 32, 32), (64, 64, 32))]),
        draw=_chunk_inputs)))


CHUNK_PULLBACK = declare(__name__, "chunk_pullback", ComponentForm(
    _chunk_pullback, _chunk_pullback_truth, dataclasses.replace(
        float32_cases(*[((k, v), (c, k), (c, k), (c, k), (c, v), (c, v), (k, 1))
                 for c, k, v in ((16, 32, 32), (64, 64, 32))]), draw=_chunk_pullback_inputs)))


CHUNK_SOLVE_LEVEL = declare(__name__, "chunk_solve_level", ComponentForm(
    _chunk_level, _chunk_level_truth, Cases(
        tuple((jax.ShapeDtypeStruct((c, c), jnp.float32), jax.ShapeDtypeStruct((c, c), jnp.float32),
               jax.ShapeDtypeStruct((), jnp.int32)) for c in (16, 32, 64)),
        _chunk_level_draw, seeds=8, policies=("default",))))


CHUNK_WEIGHTED = declare(__name__, "chunk_weighted", ComponentForm(
    _chunk_weighted, _chunk_weighted_truth,
    dataclasses.replace(float32_cases(*[((c, c), (c, v), (c,)) for c, v in ((16, 32), (32, 32), (64, 64))]),
                        draw=_chunk_weighted_draw)))


CHUNK_INVERSE_PULLBACK = declare(__name__, "chunk_inverse_pullback", ComponentForm(
    _inverse_pullback, _inverse_pullback_truth,
    dataclasses.replace(float32_cases(*[((c, c), (c, c)) for c in (16, 32, 64)]),
                        draw=_inverse_pullback_draw)))


SELECTIVE_DECAY = declare(__name__, "selective_decay", ComponentForm(
    recurrence_algebra.selective_decay, _selective_decay_truth, dataclasses.replace(
        float32_cases(((128,), (128,)), ((128,), (128,), ()), ((256,), (), (256,))),
        draw=_selective_decay_draw)))


KEY_MEMORY_DECAY = declare(__name__, "key_memory_decay", ComponentForm(
    recurrence_algebra.key_memory_decay, _key_memory_decay_truth, Cases(
        tuple((jax.ShapeDtypeStruct(shape, dtype), jax.ShapeDtypeStruct(shape, jnp.float32))
              for shape, dtype in (((64, 64), jnp.float32), ((128, 64), jnp.bfloat16))),
        _key_memory_decay_draw, seeds=8, policies=("default",))))


def _selective_adjoint_truth(cotangent, before, decay, written, step, value, b, dy, rate):
    carried = cotangent * decay * before
    write = cotangent * b
    return (step * write, rate * carried + value * write, (step * value) * cotangent,
            (before * decay + written) * dy, step * carried)


def _selective_adjoint_draw(seed, operands):
    rng = np.random.default_rng(seed)
    cotangent, before, decay, written, step, value, b, dy, rate = normal_operands(seed, operands)
    decay = rng.uniform(0, 1, operands[2].shape).astype(np.float32)
    step = np.abs(step)
    rate = -np.abs(rate)
    # A reset contributes no entering-state derivative; a padded token has no write.
    decay[0], written[1], step[1] = 0, 0, 0
    return cotangent, before, decay, written, step, value, b, dy, rate


def _selective_lifted_adjoint(*operands):
    """`recurrence_algebra.selective_adjoint` with its `dx_step` given, as the backward block calls it."""
    *ends, dx_step = operands
    return recurrence_algebra.selective_adjoint(*ends, dx_step=dx_step)


def _selective_lifted_adjoint_truth(cotangent, before, decay, written, step, value, b, dy, rate, dx_step):
    _, *rest = _selective_adjoint_truth(cotangent, before, decay, written, step, value, b, dy, rate)
    return (dx_step * (cotangent * b), *rest)


def _selective_lifted_adjoint_draw(seed, operands):
    drawn = _selective_adjoint_draw(seed, operands[:-1])
    step = drawn[4]
    dx_step = step.copy()
    if seed % 2:
        # Subnormal steps lifted by 2^64 (`selective_dx_scale`): dx's factor normal, the others' subnormal.
        step[2:10] = np.float32(2.0 ** -140) * np.arange(1, 9, dtype=np.float32)
        dx_step[2:10] = step[2:10] * np.float32(2.0 ** 64)
    return (*drawn, dx_step)


def _kda_strict_band(bands):
    return recurrence_algebra.block_masked(bands, strict=True, scale=1.0)


def _kda_band_truth(bands):
    sub = bands.shape[1]
    return jnp.tril(bands.reshape(-1, sub, sub), k=-1).reshape(bands.shape)


def _kda_band_draw(seed, operands):
    (bands,) = normal_operands(seed, operands)
    # The inverse must never read the diagonal or an upper entry of a sub-chunk.
    token = np.arange(bands.shape[0]) % bands.shape[1]
    columns = np.arange(bands.shape[1])
    bands[columns[None, :] >= token[:, None]] = np.nan if seed % 2 else np.inf
    return (bands,)


def _gdn_values_truth(cotangent, values, beta):
    return jnp.einsum("cv,c->cv", cotangent, beta), jnp.einsum("cv,cv->c", cotangent, values)


def _gdn_values_draw(seed, operands):
    cotangent, values, beta = normal_operands(seed, operands)
    beta = np.abs(beta) * np.float32(0.25)
    beta[0], values[1] = 0, 0
    return cotangent, values, beta


def _ssd_truth(x, b, c, a, state):
    def token(previous, inputs):
        value, write, read, decay = inputs
        next_state = jnp.exp(decay) * previous + value[:, None] * write[None, :]
        return next_state, next_state @ read

    final, out = jax.lax.scan(token, state, (x, b, c, a))
    return out, final


def _ssd_pullback_truth(x, b, c, a, state, dy, dnext):
    _, pull = jax.vjp(_ssd_truth, x, b, c, a, state)
    return pull((dy, dnext))


def _ssd_component_draw(seed, operands):
    vals = normal_operands(seed, operands)
    x, b, c, a, state, *adjoints = vals
    a = -np.abs(a) * np.float32(0.125)
    a[0] = 0
    x[1], b[1] = 0, 0
    if seed == 1:
        a[len(a) // 2] = -32
    return x, b, c, a, state, *adjoints


def _ssd_cases(*, reverse):
    cases = []
    for chunk, channels, states in ((64, 32, 32), (64, 64, 32)):
        args = ((chunk, channels), (chunk, states), (chunk, states), (chunk,), (channels, states))
        cases.append((*args, args[0], args[-1]) if reverse else args)
    return dataclasses.replace(float32_cases(*cases), draw=_ssd_component_draw)


SELECTIVE_ADJOINT = declare(__name__, "selective_adjoint", ComponentForm(
    recurrence_algebra.selective_adjoint, _selective_adjoint_truth, dataclasses.replace(
        float32_cases(*[((n,),) * 8 + ((),) for n in (128, 256)]), draw=_selective_adjoint_draw)))


SELECTIVE_LIFTED_ADJOINT = declare(__name__, "selective_lifted_adjoint", ComponentForm(
    _selective_lifted_adjoint, _selective_lifted_adjoint_truth, dataclasses.replace(
        float32_cases(*[((n,),) * 8 + ((), (n,)) for n in (128, 256)]), draw=_selective_lifted_adjoint_draw)))


KDA_STRICT_BAND = declare(__name__, "kda_strict_band", ComponentForm(
    _kda_strict_band, _kda_band_truth, dataclasses.replace(
        float32_cases(((32, recurrence_algebra.SUB),), ((64, recurrence_algebra.SUB),)),
        draw=_kda_band_draw)))


GDN_VALUE_PULLBACK = declare(__name__, "gdn_value_pullback", ComponentForm(
    recurrence_algebra.gated_values_back, _gdn_values_truth, dataclasses.replace(
        float32_cases(*[((c, v), (c, v), (c,)) for c, v in ((32, 64), (64, 128))]), draw=_gdn_values_draw)))


SSD_CHUNK_FORWARD = declare(__name__, "ssd_chunk_forward", ComponentForm(
    ssd._chunk_forward, _ssd_truth, _ssd_cases(reverse=False)))


SSD_CHUNK_PULLBACK = declare(__name__, "ssd_chunk_pullback", ComponentForm(
    ssd._chunk_backward, _ssd_pullback_truth, _ssd_cases(reverse=True)))


def _selective_read(*operands):
    *tiles, skip, value = operands
    count = len(tiles) // 2
    return recurrence_algebra.selective_read(tiles[:count], tiles[count:], skip, value)


def _selective_read_truth(*operands):
    *tiles, skip, value = operands
    count = len(tiles) // 2
    return jnp.sum(jnp.stack(tiles[:count]) * jnp.stack(tiles[count:])[:, None], axis=0) + skip * value


def _selective_read_draw(seed, operands):
    *tiles, skip, value = normal_operands(seed, operands)
    count = len(tiles) // 2
    # The scan's: a block's lane of each state, and C a scalar a state (the token's row of C).
    states, reads = list(tiles[:count]), list(tiles[count:])
    if seed % 2:
        for n, edge in enumerate([2.0 ** 35, -(2.0 ** 35), 1, 0]):
            states[n][0], reads[n] = edge, np.asarray(1, np.float32)
    states[-1][1], reads[-1] = 2.0 ** 60, np.asarray(2.0 ** -60, np.float32)
    return *states, *reads, skip, value


SELECTIVE_READ = declare(__name__, "selective_read", ComponentForm(
    _selective_read, _selective_read_truth, dataclasses.replace(
        float32_cases(*[((block,),) * 16 + ((),) * 16 + ((block,),) * 2 for block in (64, 128)]),
        draw=_selective_read_draw)))


def _selective_token_parts(operands):
    count, kept = divmod(len(operands) - 2, 3)
    states, rates, b = (operands[part * count:(part + 1) * count] for part in range(3))
    step, value, *rest = operands[3 * count:]
    return list(states), rates, list(b), step, value, rest[0] if kept else None


def _selective_token(*operands):
    """The forward scan's token stage (`selective.token_stage`) over its flat operands: states and rates a
    channel tile each, B's a scalar each."""
    states, rates, b, step, value, kept = _selective_token_parts(operands)
    return tuple(selective.token_stage(states, list(rates), b.__getitem__, step, value, kept))


def _selective_token_truth(*operands):
    states, rates, b, step, value, kept = _selective_token_parts(operands)
    held = 1 if kept is None else kept
    return tuple(jnp.exp(step * rate) * held * state + step * value * b_n
                 for state, rate, b_n in zip(states, rates, b, strict=True))


def _selective_token_draw(seed, operands):
    count = (len(operands) - 2) // 3
    rng = np.random.default_rng(seed)
    states = [rng.standard_normal(each.shape).astype(np.float32) for each in operands[:count]]
    rates = [-rng.uniform(0, 2, each.shape).astype(np.float32) for each in operands[count:2 * count]]
    b = [np.float32(rng.standard_normal()) for _ in range(count)]
    step = rng.uniform(0, 2, operands[3 * count].shape).astype(np.float32)
    value = rng.standard_normal(step.shape).astype(np.float32)
    if seed == 1:
        # A padded token past a rate float32's base-2 logarithm overflows (row 8491).
        step[:4] = 0
        rates[0][:4] = -np.float32(3 * 2.0 ** 126)
    elif seed == 2:
        # A step past float32's range in base 2 times a least-normal rate: exp(-3), its write kept finite.
        step[:4], value[:4] = np.float32(3 * 2.0 ** 126), np.float32(2.0 ** -8)
        rates[-1][:4] = -np.float32(2.0 ** -126)
    elif seed == 3:
        rates[1][:] = 0
    kept = () if len(operands) == 3 * count + 2 else (np.float32(seed % 2),)
    return *states, *rates, *b, step, value, *kept


def _selective_dx_scale_truth(skip, dy, step):
    base = skip * dy
    lifted = (jnp.abs(step) < 2.0 ** -100) & (jnp.abs(base) < 2.0 ** 60)
    scale = jnp.where(lifted, 2.0 ** 64, 1.0)
    return base * scale, step * scale, 1 / scale


def _selective_dx_scale_draw(seed, operands):
    skip, dy, step = normal_operands(seed, operands)
    step = np.abs(step)
    step[1] = 0
    if seed % 2:
        # Subnormal and small normal steps the lift raises into the normal range, with `skip dy` small.
        step[2:10] = np.float32(2.0 ** -140) * np.arange(1, 9, dtype=np.float32)
        step[10:14] = np.float32(2.0 ** -110)
        dy[2:6] = 0
    if seed == 2:
        # Past the lift's reach: `skip dy` at 2^61, its small steps kept as they are.
        skip = np.float32(1.5)
        step[2:6], dy[2:6] = np.float32(2.0 ** -120), np.float32(2.0 ** 61)
    return skip, dy, step


def _selective_state_tangent(dx, ddt, cotangent, before, decay, written, step, value, b, dy, rate, dx_step):
    """One state's turn of a backward block's state loop (`selective.backward`'s `state`): its `dx` term,
    taken at the lifted step (`selective_dx_scale`'s `dx_step`), and its step term, added to the block's
    running sums."""
    d_value, d_step, *_ = recurrence_algebra.selective_adjoint(
        cotangent, before, decay, written, step, value, b, dy, rate, dx_step=dx_step)
    return dx + d_value, ddt + d_step


def _selective_state_tangent_truth(dx, ddt, cotangent, before, decay, written, step, value, b, dy, rate,
                                   dx_step):
    return (dx + cotangent * b * dx_step,
            ddt + cotangent * b * value + cotangent * before * decay * rate)


def _selective_state_tangent_draw(seed, operands):
    drawn = normal_operands(seed, operands)
    dx, ddt, cotangent, before, decay, written, step, value, b, dy, rate, dx_step = drawn
    rng = np.random.default_rng(seed)
    decay = rng.uniform(0, 1, decay.shape).astype(np.float32)
    step, rate = np.abs(step), -np.abs(rate)
    decay[0], step[1] = 0, 0
    dx_step = step.copy()
    if seed % 2:
        # A subnormal step's `dx` term taken at its lift, 2^64 times the step, normal.
        step[2:10] = np.float32(2.0 ** -140) * np.arange(1, 9, dtype=np.float32)
        dx_step[2:10] = step[2:10] * np.float32(2.0 ** 64)
    return dx, ddt, cotangent, before, decay, written, step, value, b, dy, rate, dx_step


SELECTIVE_TOKEN = declare(__name__, "selective_token", ComponentForm(
    _selective_token, _selective_token_truth, dataclasses.replace(
        float32_cases(((64,),) * 8 + ((),) * 4 + ((64,),) * 2,
                      ((128,),) * 32 + ((),) * 16 + ((128,),) * 2 + ((),)),
        draw=_selective_token_draw)))


SELECTIVE_DX_SCALE = declare(__name__, "selective_dx_scale", ComponentForm(
    recurrence_algebra.selective_dx_scale, _selective_dx_scale_truth, dataclasses.replace(
        float32_cases(((), (64,), (64,)), ((), (128,), (128,))), draw=_selective_dx_scale_draw)))


SELECTIVE_STATE_TANGENT = declare(__name__, "selective_state_tangent", ComponentForm(
    _selective_state_tangent, _selective_state_tangent_truth, dataclasses.replace(
        float32_cases(*[((positions,),) * 10 + ((),) + ((positions,),) for positions in (64, 128)]),
        draw=_selective_state_tangent_draw)))


def _skipped_back_truth(out_grad, values, skip):
    return skip * out_grad, jnp.sum(out_grad * values, axis=-1)


def _skipped_back_draw(seed, operands):
    out_grad, values, skip = normal_operands(seed, operands)
    if seed % 2:
        # A large read and its reciprocal-scale value cancel in the skip's own sum.
        first = (0,) * (out_grad.ndim - 1)
        out_grad[(*first, slice(2))], values[(*first, slice(2))] = [2.0 ** 60, -(2.0 ** 60)], [2.0 ** -60] * 2
    return out_grad, values, skip


SKIPPED_BACK = declare(__name__, "skipped_back", ComponentForm(
    recurrence_algebra.skipped_back, _skipped_back_truth, dataclasses.replace(
        # The tiled backward's (a block's rows of the value columns, the head's scalar skip) and the laid
        # backward's (the call's rows of each head's value columns, a skip a head).
        float32_cases(((64, 64), (64, 64), ()), ((64, 128), (64, 128), ()),
                      ((128, 4, 64), (128, 4, 64), (4, 1)), ((256, 8, 64), (256, 8, 64), (8, 1))),
        draw=_skipped_back_draw)))
