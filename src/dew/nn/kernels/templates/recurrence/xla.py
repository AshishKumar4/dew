"""The recurrence template's portable mainloop (the template design's §9): the chunked rule's launches as
the GPU mainloop runs them (fla's four, `dew.nn.kernels.fla_gdn`'s), each program's tiles gathered from
its operands as the kernel loads them, and the algebra's tile functions (`recurrence_algebra`) mapped
over every program at once. At the GPU mainloop's tiles it computes what that mainloop interpreted
computes, bit for bit, which tests/test_recurrence_template.py holds it to.

A program reads a chunk's rows from its first token, rows past the chunk's count zero (the kernels'
masked loads) and its prefix's rows past the count the last real one (`chunked_launch._gates`), and
writes only the rows the chunk holds. Its products are the GPU mainloop's interpreted ones
(`fla_gdn._issuer`): its operands in fp32, the CPU's own dot.
"""

import functools

import jax
import jax.numpy as jnp

from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.chunked_launch import Sequences, Tiles
from dew.nn.kernels.fla_gdn import _issuer

SLICE = 64
"""The memory's key slice: the state pass keeps the memory in 64-key slices (fla's blockdim64)."""

ISSUE = _issuer(exact=True)
"""The products as the GPU mainloop interprets them (`fla_gdn._issuer`): fp32 operands, the CPU's dot."""


def _rows(x, start, size: int, count, offset: int = 0):
    """`size` rows of `x` (padded by a chunk at its end) from `start + offset`, zeros past `count`."""
    rows = jax.lax.dynamic_slice_in_dim(x, start + offset, size, 0)
    kept = (jnp.arange(size) < count - offset).reshape((size,) + (1,) * (rows.ndim - 1))
    return jnp.where(kept, rows, jnp.zeros((), rows.dtype))


def _gates(g, start, size: int, count, offset: int = 0):
    """A chunk's prefix rows, fp32, those past `count` holding the last real row's."""
    rows = jax.lax.dynamic_slice_in_dim(g, start + offset, size, 0).astype(jnp.float32)
    last = jax.lax.dynamic_index_in_dim(g, start + count - 1, 0, keepdims=False).astype(jnp.float32)
    return jnp.where(jnp.arange(size) < count - offset, rows, last)


def _padded(x, chunk: int):
    """`x` `[N, ...]` with a chunk of zero rows after its last, so every chunk's rows slice in bounds."""
    return jnp.pad(x, [(0, chunk)] + [(0, 0)] * (x.ndim - 1))


def _placed(results, starts, counts, tokens: int):
    """Each program's rows `[P, C, ...]` at their tokens from `starts`, those past `counts` dropped, into
    `[tokens, ...]` zeros: a kernel's masked stores."""
    size = results.shape[1]
    rows = starts[:, None] + jnp.arange(size)[None, :]
    rows = jnp.where(jnp.arange(size)[None, :] < counts[:, None], rows, tokens)
    out = jnp.zeros((tokens, *results.shape[2:]), results.dtype)
    return out.at[rows.reshape(-1)].set(results.reshape(-1, *results.shape[2:]), mode="drop")


def _headed(x, heads: int):
    """`[N, H, ...]` as `[heads, N, ...]`, each of `heads` value heads reading its key head's rows."""
    return jnp.moveaxis(jnp.repeat(x, heads // x.shape[1], axis=1), 1, 0)


def _per_chunk(program, spans: Sequences, tokens: int, chunk: int, *operands):
    """`program(start, count, *operands)` for every chunk of `spans` and value head, the operands
    `[heads, N + chunk, ...]`: its rows `[heads, NT, C, ...]` placed at their tokens `[N, heads, ...]`."""
    starts, counts = spans.chunks[:, 0], spans.chunks[:, 1]
    over_chunks = jax.vmap(program, in_axes=(0, 0) + (None,) * len(operands))
    results = jax.vmap(over_chunks, in_axes=(None, None) + (0,) * len(operands))(starts, counts, *operands)
    placed = jax.vmap(lambda r: _placed(r, starts, counts, tokens))(results)
    return jnp.moveaxis(placed, 0, 1)


def _solve_program(start, count, k, g, gl, beta, *, chunk: int, bk: int):
    """`fla_gdn._kkt_solve_chunk`'s program: the chunk's inverse `[C, C]`, zeros above its blocks."""
    sub, count_subs = algebra.SUB, chunk // algebra.SUB
    gates = [_gates(g, start, sub, count, i * sub) for i in range(count_subs)]
    lows = [_gates(gl, start, sub, count, i * sub) for i in range(count_subs)]
    betas = [_rows(beta, start, sub, count, i * sub).astype(jnp.float32) for i in range(count_subs)]
    inverse = algebra.solved(lambda i, c: _rows(k[:, c * bk:(c + 1) * bk], start, sub, count, i * sub),
                             gates, lows, betas, k.shape[-1] // bk, _issuer(exact=True, solve=k.dtype))
    zero = jnp.zeros((sub, sub), jnp.float32)
    return jnp.concatenate([jnp.concatenate([inverse[i, j] if j <= i else zero for j in range(count_subs)], 1)
                            for i in range(count_subs)], 0)


def _weighted_program(start, count, k, v, beta, a, g, *, chunk: int, bk: int, bv: int):
    """`fla_gdn._w_u_kernel`'s program: the chunk's `w` `[C, K]` and `u` `[C, V]`, fp32."""
    beta, inverse = _rows(beta, start, chunk, count), _rows(a, start, chunk, count)
    u = jnp.concatenate([algebra.weighted("apply_u", inverse, _rows(v[:, c:c + bv], start, chunk, count),
                                          beta, ISSUE) for c in range(0, v.shape[-1], bv)], 1)
    decay = jnp.exp2(_gates(g, start, chunk, count))
    w = jnp.concatenate([algebra.weighted("apply_w", inverse, _rows(k[:, c:c + bk], start, chunk, count),
                                          beta, ISSUE, decay[:, None]) for c in range(0, k.shape[-1], bk)], 1)
    return w, u


def _output_program(start, count, entered, q, k, v, h, g, gl, *, chunk: int, bk: int, scale: float):
    """`fla_gdn._o_kernel`'s program over one value block: the chunk's output `[C, BV]`, fp32."""
    def columns(x, c):
        return x[..., c * bk:(c + 1) * bk]

    return algebra.chunk_output(lambda c: _rows(columns(q, c), start, chunk, count),
                                lambda c: h[entered, c * bk:(c + 1) * bk],
                                lambda c: _rows(columns(k, c), start, chunk, count), q.shape[-1] // bk,
                                _gates(g, start, chunk, count), _gates(gl, start, chunk, count),
                                _rows(v, start, chunk, count), scale, ISSUE)


def _state_program(span, k, u, w, g, gl, h0, *, chunk: int, length: int, steps: int, per_key: bool):
    """`fla_gdn._h_kernel`'s program, one sequence of one head over one value block: the memory each of
    its chunks enters `[steps, K, BV]` and their corrected values `[steps, C, BV]` (fp32), and the
    memory it leaves. `steps` bounds its chunks; past its own, nothing moves."""
    start, count = span[0], span[1]
    row, first = start // length, start % length == 0
    slices = k.shape[-1] // SLICE
    state = [jnp.where(first, h0[row, i * SLICE:(i + 1) * SLICE].astype(jnp.float32), 0.0)
             for i in range(slices)]
    issue = ISSUE

    def step(state, n):
        begin = start + n * chunk
        taken = jnp.minimum(chunk, count - n * chunk)

        def read(x, c=None):
            return _rows(x if c is None else x[:, c * SLICE:(c + 1) * SLICE], begin, chunk, taken)

        corrected = algebra.corrected(read(u), lambda i: read(w, i), state, w.dtype, issue)
        final = begin + taken - 1
        if per_key:
            last = jax.lax.dynamic_index_in_dim(g, final, 0, keepdims=False)
            decay = [jnp.exp2(last[i * SLICE:(i + 1) * SLICE].astype(jnp.float32))[:, None]
                     for i in range(slices)]
            leaving = algebra.carried(state, lambda i: read(k, i), corrected, decay, k.dtype, issue)
        else:
            gated, decay = algebra.scalar_write(
                corrected, jax.lax.dynamic_index_in_dim(g, final, 0, keepdims=False).astype(jnp.float32),
                _gates(g, begin, chunk, taken), jax.lax.dynamic_index_in_dim(gl, final, 0, keepdims=False)
                .astype(jnp.float32), _gates(gl, begin, chunk, taken))
            leaving = algebra.carried(state, lambda i: read(k, i), gated, [decay] * slices, k.dtype, issue)
        moving = n * chunk < count
        kept = [jnp.where(moving, new, old) for new, old in zip(leaving, state, strict=True)]
        return kept, (jnp.concatenate(state, 0), corrected)

    state, (entering, corrected) = jax.lax.scan(step, state, jnp.arange(steps))
    return entering, corrected, jnp.concatenate(state, 0)


def _states(k, u, w, g, gl, h0, spans: Sequences, *, batch: int, length: int, chunk: int, bv: int,
            held: int, per_key: bool):
    """`fla_gdn.chunk_fwd_h`: `(h [held, HV, K, V], corrected [N, HV, V], final [B, HV, K, V])`, h in
    `k`'s dtype, the corrected values in `u`'s."""
    tokens, heads, columns = u.shape[0], u.shape[1], u.shape[2]
    steps = -(-length // chunk)
    first, count, base, entered = (spans.spans[:, i] for i in range(4))
    program = functools.partial(_state_program, chunk=chunk, length=length, steps=steps, per_key=per_key)
    keyed = _padded(_headed(k, heads).swapaxes(0, 1), chunk).swapaxes(0, 1)
    blocks = [jax.vmap(jax.vmap(program, in_axes=(0, None, None, None, None, None, None)),
                       in_axes=(None, 0, 0, 0, 0, 0, 1))(
        spans.spans, keyed, jnp.moveaxis(_padded(u[..., c:c + bv], chunk), 1, 0),
        jnp.moveaxis(_padded(w, chunk), 1, 0), jnp.moveaxis(_padded(g, chunk), 1, 0),
        jnp.moveaxis(_padded(gl, chunk), 1, 0), h0[..., c:c + bv]) for c in range(0, columns, bv)]
    entering = jnp.concatenate([b[0] for b in blocks], -1)  # [HV, S, steps, K, V]
    corrected = jnp.concatenate([b[1] for b in blocks], -1)  # [HV, S, steps, C, V]
    final = jnp.concatenate([b[2] for b in blocks], -1)  # [HV, S, K, V]
    n = jnp.arange(steps)
    moving = (n[None, :] * chunk < count[:, None]) & (count[:, None] > 0)
    slot = jnp.where(moving, jnp.where(n[None, :] > 0, base[:, None] + n[None, :], entered[:, None]), held)
    h = jnp.zeros((held, heads, k.shape[-1], columns), k.dtype)
    h = h.at[slot.reshape(-1)].set(jnp.moveaxis(entering, 0, 2).reshape(-1, heads, k.shape[-1], columns)
                                   .astype(k.dtype), mode="drop")
    begins = (first[:, None] + n[None, :] * chunk).reshape(-1)
    takens = jnp.where(moving, jnp.minimum(chunk, count[:, None] - n[None, :] * chunk), 0).reshape(-1)
    rows = jnp.moveaxis(corrected, 0, 3).reshape(-1, chunk, heads, columns).astype(u.dtype)
    corrected = _placed(rows, begins, takens, tokens)
    last = (count > 0) & ((first + count) % length == 0)
    finals = jnp.zeros((batch, heads, k.shape[-1], columns), jnp.float32)
    finals = finals.at[jnp.where(last, first // length, batch)].set(jnp.moveaxis(final, 0, 1), mode="drop")
    return h, corrected, finals


def forward(q, k, v, g, beta, h0, spans: Sequences, *, scale: float, chunk: int, held: int,
            tiles: dict[str, Tiles], wide: bool = False):
    """`fla_gdn.cumulated_fwd`'s values on this mainloop, a decay a head: `(o, final)` and the inverse,
    `w`, `u`, the memories the chunks enter and the corrected values. `g` is the compensated prefix
    `(hi, lo)` `[B, T, HV]` (`recurrence.cumulated`), `spans` the call's `Sequences`, `held` the
    memories' slots (`chunked_launch.slots`), `tiles` each launch's ('solve', 'weighted', 'states',
    'output') as the GPU mainloop ran them."""
    batch, length, _, width = k.shape
    heads, columns = v.shape[2:]
    tokens = batch * length
    hi, lo = (x.reshape(tokens, heads) for x in g)
    flat = {name: x.reshape(tokens, *x.shape[2:])
            for name, x in (("q", q), ("k", k), ("v", v), ("beta", beta))}

    def by_head(x):
        return _headed(_padded(x, chunk), heads)

    solve = functools.partial(_solve_program, chunk=chunk, bk=min(tiles["solve"].bk, width))
    a = _per_chunk(solve, spans, tokens, chunk, by_head(flat["k"]), by_head(hi), by_head(lo),
                   by_head(flat["beta"])).astype(k.dtype)
    tile = tiles["weighted"]
    weighted = functools.partial(_weighted_program, chunk=chunk, bk=min(tile.bk, width),
                                 bv=min(tile.bv, columns))
    starts, counts = spans.chunks[:, 0], spans.chunks[:, 1]
    over = jax.vmap(jax.vmap(weighted, in_axes=(0, 0) + (None,) * 5), in_axes=(None, None) + (0,) * 5)
    w, u = over(starts, counts, by_head(flat["k"]), by_head(flat["v"]), by_head(flat["beta"]), by_head(a),
                by_head(hi))
    w = jnp.moveaxis(jax.vmap(lambda r: _placed(r, starts, counts, tokens))(w), 0, 1).astype(k.dtype)
    u = jnp.moveaxis(jax.vmap(lambda r: _placed(r, starts, counts, tokens))(u), 0, 1).astype(v.dtype)
    h, corrected, final = _states(flat["k"], u, w, hi, lo, h0, spans, batch=batch, length=length, chunk=chunk,
                                  bv=min(tiles["states"].bv, columns), held=held, per_key=False)
    tile = tiles["output"]
    bv = min(tile.bv, columns)
    output = functools.partial(_output_program, chunk=chunk, bk=min(tile.bk, width), scale=scale)
    entered = spans.chunks[:, 2]
    over = jax.vmap(jax.vmap(output, in_axes=(0, 0, 0) + (None,) * 6), in_axes=(None, None, None) + (0,) * 6)
    blocks = [over(starts, counts, entered, by_head(flat["q"]), by_head(flat["k"]),
                   by_head(corrected[..., c:c + bv]), jnp.moveaxis(h[..., c:c + bv], 1, 0), by_head(hi),
                   by_head(lo)) for c in range(0, columns, bv)]
    out = jnp.concatenate([jnp.moveaxis(jax.vmap(lambda r: _placed(r, starts, counts, tokens))(b), 0, 1)
                           for b in blocks], -1)
    out = out.astype(jnp.float32 if wide else v.dtype).reshape(v.shape)
    return (out, final), (a.reshape(batch, length, heads, chunk), w.reshape(batch, length, heads, width),
                          u.reshape(v.shape), h, corrected.reshape(v.shape))
