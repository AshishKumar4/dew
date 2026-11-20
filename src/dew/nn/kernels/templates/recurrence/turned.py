"""A turned call on the recurrence template (`recurrence.Hooks.turns`, Mamba-3's rotating basis).

The template's kernels take no turn. A turned call's queries and keys are turned into each document's own
frame (`frames`: its turns through each token, from the document's first), so the template's pairs and
memories read their relative turn, `q_s . R(Phi_u - Phi_s) k_u`, as the turned recurrence does, and its final
memory is turned back by its document's last angle (`call`). The turns' gradient is formed in the crossing
form (`turn_gradient`): a token's turn moves each pair it lies between, `u < t <= s`, and no other, so its
cotangent sums those pairs' terms alone. Formed as a suffix sum of each token's frame cotangent (the turned
keys' and queries' gradients against their quarter turns, mamba_ssm's `angle_dt` backward), its terms are
each a turned key's or query's size and cancel to the template's rounding of those gradients: Mamba-3's
chunked angle gradient's worst row was 120 times the recurrence's from float64 (row 7419)."""
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra import scan
from dew.nn.derivatives import KernelGlue, contracted
from dew.nn.kernels.chunked_launch import AlignedDocuments, Forked, entry_write
from dew.nn.precision import at_least_fp32, compensated_add

TWO_PI = (np.float32(2 * np.pi), np.float32(2 * np.pi - float(np.float32(2 * np.pi))))
"""2 pi as fp32's nearest and what it drops, a turn's whole in double float."""

CHUNK = 64
"""The crossing form's chunk: its own pairs taken in halves (`_halves`), the rest through its memories."""


def _wrapped(angle):
    """A double-float angle `(high, low)` within 5 pi of zero (a sum of two within pi or two pi) less its
    whole turns, within pi of zero: at most two, each turn's parts' product exact (`TWO_PI` times one or
    two), so no residual rests on a product's rounding, which XLA's contraction of a multiply into its
    neighbouring add moves under jit: Dekker's product of a document's whole count of turns lost its
    residual so, every frame 9.5e-7 radians off on XLA:CPU (1.2e-13 eager), a forked call's final memory
    2.2 times the recurrence's error from float64 (row 7419)."""
    turns = jnp.round(angle[0] * np.float32(1 / (2 * np.pi)))
    return compensated_add(angle, (-turns * TWO_PI[0], -turns * TWO_PI[1]))


def _wide(x) -> jax.Array:
    """`x` in at least fp32, its own dtype where wider (a float64 truth's)."""
    x = jnp.asarray(x)
    return x.astype(at_least_fp32(x.dtype))


def frames(turn, starts=None) -> tuple[jax.Array, jax.Array]:
    """Each token's frame `[B, T, ..., A]`: its document's turns through it (`turn` `[B, T, ..., A]`; `starts`
    `[B, T]` each document's first token, where given), each within pi of zero first (`scan.within_pi`), their
    compensated prefix (`_prefix`, its whole turns taken off as it runs), a double float `(high, low)` within
    pi of zero, which holds an angle to 2^-48 of pi, its whole turns' 2 pi each to 2^-48 of itself: past
    2^20 tokens' turns, their drift below fp32's rounding of a turned key."""
    turn = jnp.asarray(turn)
    within = scan.within_pi(turn.astype(at_least_fp32(turn.dtype)))
    restarts = jnp.zeros(within.shape[:2], bool) if starts is None else jnp.asarray(starts, bool)
    restarts = jnp.reshape(restarts, (*restarts.shape, *(1,) * (within.ndim - 2)))
    return _prefix(within, restarts)


def _prefix(values, restarts) -> tuple[jax.Array, jax.Array]:
    """`values` `[B, T, ...]`'s (angles within pi of zero) inclusive prefix sums along their tokens as double
    floats less their whole turns, each within pi of zero (`_wrapped` at every sum of two), restarting at
    each of `restarts` (`[B, T, ...]` broadcast): a chunk's own a walk of its `CHUNK` tokens (each
    renormalized, `compensated_add`), each chunk's entering the walk of the chunks' ends, so every token is
    read once (an associative scan's every level reads all of them: 7402's angles took a Mamba-3 step's
    wall 13% past its kernels on a GH200)."""
    batch, length = values.shape[:2]
    pad = (-length) % CHUNK
    chunks = (length + pad) // CHUNK
    restarts = jnp.broadcast_to(restarts, values.shape)

    def laid(x):
        x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        return jnp.moveaxis(jnp.reshape(x, (batch, chunks, CHUNK, *x.shape[2:])), 2, 0)

    def within(total, token):
        value, restart = token
        total = tuple(jnp.where(restart, 0.0, part) for part in total)
        total = _wrapped(compensated_add(total, (value, jnp.zeros_like(value))))
        return total, total

    zero = jnp.zeros((batch, chunks, *values.shape[2:]), values.dtype)
    flags = laid(restarts)
    _, local = jax.lax.scan(within, (zero, zero), (laid(values), flags))
    restarted = jnp.any(flags, axis=0)

    def across(entering, chunk):
        end, broke = chunk
        leaving = _wrapped(compensated_add(entering, end))
        return tuple(jnp.where(broke, e, part) for e, part in zip(end, leaving, strict=True)), entering

    ends = tuple(jnp.moveaxis(part[-1], 1, 0) for part in local)
    _, offsets = jax.lax.scan(across, (zero[:, 0], zero[:, 0]), (ends, jnp.moveaxis(restarted, 1, 0)))
    seen = jnp.cumsum(flags.astype(jnp.int32), axis=0) > 0
    offsets = tuple(jnp.moveaxis(part, 0, 1)[None] for part in offsets)
    summed = _wrapped(compensated_add(local, offsets))
    summed = tuple(jnp.where(seen, part, whole) for part, whole in zip(local, summed, strict=True))
    high, low = (jnp.reshape(jnp.moveaxis(part, 0, 2), (batch, chunks * CHUNK, *values.shape[2:]))[:, :length]
                 for part in summed)
    return high, low


def call(launch: Callable, query, key, value, g, beta, state, skip, turn, starts, *, scale: float,
         group: int, documents=None, held: int = 0, entry=None,
         context=None) -> tuple[jax.Array, jax.Array]:
    """`launch(context, query, key, value, g, beta, state, skip, entry, turning)`'s `(output, final memory)`
    (the template's, `context` the arrays it reads beside them, an operand of the rule, closed over by none of
    its maps; its memories `[B, H, K, V]`) of a call whose state turns its first `A` key pairs each token by
    `turn` `[B, T, H, A]` (`recurrence.Hooks.turns`), `starts` `[B, T]` each document's first token or None, a
    readout `apart` from its token's `group` of writes where `group` is past one (`mainloop.Call.apart`): each
    document read in its own last token's frame, its queries and keys turned by their own turns less its
    last's (`frames`, `_in_last_frame`), in at least fp32 and rounded once to their dtype, and the memory it
    enters turned in by minus its own turns, so a document's arithmetic reads no other's turns and the final
    memory each leaves is in its own last frame, the recurrence's. A forked call (`documents`, its `(firsts,
    loads, stores)` `[B, D]` and `held` stored slots, `chunked_launch.Forked`'s; `state` its table's given
    memories `[B, S', H, K, V]`; `entry` its documents' entry writes `(key [B, D, R, H, K], value [B, D, R, H,
    V], weight [B, D, H])` or None) has its walk turn each memory a document takes up as it loads it
    (`launch`'s `turning`, `chunked_launch.Forked.turning`) and its entry keys with it. Its gradient takes the
    launch's through the frames, its cotangents halved into the launch and its gradients doubled out (exact),
    so a gradient's components in a frame, up to its norm, stay finite where the true frame's are; its turns'
    in the crossing form (`turn_gradient`) from frames each root's first token composes (`_forked_frames`),
    its final memory's cotangent turned into them."""
    if documents is not None:
        # A forked call's documents begin at their first slots (`documents`' `firsts`), whatever `starts` it
        # gave (base review fbb1c2f71 F4: a call giving its firsts alone read as one document).
        starts = _first_tokens(documents[0], value.shape[1])
    rule = _turned(launch, scale, group, held)
    return rule(query, key, value, g, beta, state, skip, entry, turn, starts, documents, context)


def _first_tokens(firsts, length: int) -> jax.Array:
    """Each document's first token `[B, T]` of its first slots `firsts` `[B, D]` (one past a row's last,
    none)."""
    firsts = jnp.asarray(firsts, jnp.int32)
    rows = jnp.arange(firsts.shape[0])[:, None]
    return jnp.zeros((firsts.shape[0], length), bool).at[rows, firsts].set(True, mode='drop')


def _turned_primal(launch, scale, group, held, query, key, value, g, beta, state, skip, entry, turn, starts,
                   documents, context) -> tuple[jax.Array, jax.Array]:
    del scale, group, held
    framed, turning, _ = _framed(query, key, state, entry, turn, starts, documents)
    return launch(context, *framed[:2], value, g, beta, framed[2], skip, framed[3], turning)


def _in_last_frame(local, starts, documents):
    """Each token's frame in its document's last token's (`local`, `frames`' each document's own turns from
    its first token, less its last's) as rotations `(cos, sin)` `[B, T, H, A]`, and the rotation each
    document turns the memory it enters by, minus its own turns: a forked call's documents' `[B, D, H, A]`
    (`documents`' firsts), else the first document's `[B, 1, H, A]`. Read so, a recent write's key turns by
    nearly nothing and a final memory is its document's last frame's, the recurrence's own: read in frames
    from a document's first token and turned back, the final memory under a decay keeping a few tokens was
    9.6 times the recurrence's worst row from float64 (test_recurrence_chunk's long documents); read in a
    row's last token's, a document read later documents' turns (row 7419)."""
    batch, length = local[0].shape[:2]
    place = jnp.arange(length)
    ends = (jnp.ones((batch, length), bool).at[:, :-1].set(jnp.asarray(starts, bool)[:, 1:])
            if starts is not None else jnp.broadcast_to(place == length - 1, (batch, length)))
    lasts = jax.lax.cummin(jnp.where(ends, place[None], length), axis=1, reverse=True)
    last = tuple(_taken(part, lasts) for part in local)
    relative = compensated_add(local, tuple(-part for part in last))
    if documents is None:
        # The first document's last token, whose frame the memory entering is turned into.
        ending = lasts[:, :1]
    else:
        firsts = jnp.asarray(documents[0], jnp.int32)
        following = jnp.concatenate([firsts[:, 1:], jnp.full((batch, 1), length, jnp.int32)], axis=1)
        ending = jnp.clip(following - 1, 0, length - 1)
    owned = tuple(-_taken(part, ending) for part in local)
    return scan.rotation(*relative), scan.rotation(*owned)


def _memory_axis(rotation, memory):
    """A row's rotation `[B, 1, H, A]` laid against a memory's key axis: `[B, H, 1, A]` for a call's memory
    `[B, H, K, V]`, `[B, 1, H, 1, A]` for a table's `[B, S', H, K, V]`."""
    if memory.ndim == 4:
        return tuple(part[:, 0, :, None, :] for part in rotation)
    return tuple(part[:, :, :, None, :] for part in rotation)


def _framed(query, key, state, entry, turn, starts, documents):
    """A turned call's launch operands in its documents' frames (`_in_last_frame`): its queries and keys,
    its memory entering (a forked call's table as given) and its entry write's keys, as `(query, key, state,
    entry)`; the load rotation a forked call's walk turns each memory it takes up by (`Forked.turning`, None
    for another call); and the frames' `((cos, sin), owned)` its derivative reads."""
    local = frames(turn, starts)
    (cos, sin), owned = _in_last_frame(local, starts, documents)
    turned = [scan.rotated(_wide(x), cos, sin).astype(x.dtype) for x in (query, key)]
    turning = None
    if documents is None:
        if state is not None:
            # The memory entering, its first document's, in that document's last frame.
            state = scan.rotated(_wide(state), *_memory_axis(owned, state), axis=-2).astype(state.dtype)
    else:
        # Each document's memory turned in as its walk takes it up, its entry keys with it.
        turning = owned
        if entry is not None:
            keys = scan.rotated(_wide(entry[0]), *(part[:, :, None] for part in owned))
            entry = (keys.astype(entry[0].dtype), *entry[1:])
    return (*turned, state, entry), turning, ((cos, sin), owned)


def _turned_forward(launch, scale, group, held, query, key, value, g, beta, state, skip, entry, turn, starts,
                    documents, context) -> tuple[tuple[jax.Array, jax.Array], tuple]:
    """`_turned_primal` with its launch's pullback: the derivative's forward alone traces the launch for one
    (a forward-only call launches as inference does, audit KPM-04)."""
    del scale, group, held
    (q, k, framed_state, framed_entry), turning, ((cos, sin), owned) = _framed(query, key, state, entry, turn,
                                                                              starts, documents)
    (out, final), pullback = jax.vjp(lambda *operands: launch(context, *operands, turning), q, k, value, g,
                                     beta, framed_state, skip, framed_entry)
    kind = jnp.zeros((), turn.dtype)
    return (out, final), (pullback, query, key, value, g, beta, state, entry, turn, starts, documents, cos,
                          sin, owned, kind)


def _turned_backward(launch, scale, group, held, residuals, cotangents):
    del launch
    (pullback, query, key, value, g, beta, state, entry, turn, starts, documents, cos, sin, owned,
     turn_kind) = residuals
    d_out, d_final = cotangents

    def doubled(x):
        return None if x is None else x * 2

    # Halved into the launch, doubled out after their frames: exact, and a gradient's components in a
    # document's frame (each up to its norm) stay finite where the true frame's do (the turned contract near
    # fp32's largest, test_contracts_recurrence).
    d_query, d_key, d_value, d_g, d_beta, d_state, d_skip, d_entry = pullback(
        tuple(d * 0.5 for d in (d_out, d_final)))
    unturned = [doubled(scan.rotated(_wide(d), cos, -sin).astype(d.dtype)) for d in (d_query, d_key)]
    if d_state is not None and documents is None:
        cos_in, sin_in = _memory_axis(owned, d_state)
        d_state = scan.rotated(_wide(d_state), cos_in, -sin_in, axis=-2).astype(d_state.dtype)
    if d_entry is not None:
        keys = scan.rotated(_wide(d_entry[0]), owned[0][:, :, None], -owned[1][:, :, None])
        d_entry = tuple(doubled(part) for part in (keys.astype(d_entry[0].dtype), *d_entry[1:]))
    d_turn = turn_gradient(query, key, value, d_out, g, beta, state, d_final, turn, starts,
                           scale=scale, group=group, pairs=cos.shape[-1], documents=documents, held=held,
                           entry=entry)
    return (*unturned, *(doubled(d) for d in (d_value, d_g, d_beta, d_state, d_skip)), d_entry,
            d_turn.astype(turn_kind.dtype), None, None)


def _none_along(mark):
    """A cotangent of none along `mark`: float0 zeros of an integer or boolean mark, a float one's zeros."""
    if jnp.issubdtype(jnp.result_type(mark), jnp.inexact):
        return jnp.zeros_like(mark)
    return np.zeros(np.shape(mark), jax.dtypes.float0)


def _turned(launch, scale: float, group: int, held: int) -> Callable:
    """`call`'s rule over `launch` (`generation.linear_tangent`): its reverse mode the crossing form's
    (`_turned_forward`, `_turned_backward`), the launches opaque within (`KernelGlue`), and its tangent
    `_turned_primal`'s, the frames' and the launch's own (the portable walk's token replay), so forward mode
    and every higher order run where the launch's derivative does, where a custom VJP alone refused a
    jvp of the portable math (row 8573). A call no derivative reaches runs `_turned_primal`, launching as
    inference does, saving nothing (audit KPM-04); every order past the first differentiates the crossing
    form as written (`traced`), its launches by their own rules."""
    from dew.nn.kernels import generation

    def forward(*operands):
        return _turned_forward(launch, scale, group, held, *operands)

    def backward(saved, operands, cotangent):
        # The marks' (starts, documents' and the launch's context's) cotangents none, as their leaves:
        # `linear_tangent` reads the cotangents leaf by leaf beside the operands'.
        *cotangents, _, _ = _turned_backward(launch, scale, group, held, saved, cotangent)
        return (*cotangents, *jax.tree.map(_none_along, operands[-3:]))

    def portable(*operands):
        return _turned_primal(launch, scale, group, held, *operands)

    return contracted("test_a_turned_template_call_holds_every_order")(
        generation.linear_tangent(forward, backward, portable, kind=KernelGlue(), primal=portable,
                                  traced=True))


def _taken(x, index):
    """`x` `[B, D, ...]` at each row's `index` `[B, ...]` (its documents' axis, gathered row by row)."""
    index = jnp.asarray(index, jnp.int32)
    return jax.vmap(lambda row, at: row[at])(x, index)


def _crossed(x, y):
    """Each pair's `(J x) . y` `[..., A]` of `x` and `y` `[..., 2 A]` (`J` a quarter turn, `i` on a pair
    `x_2a + i x_2a+1`): a pair's score's derivative in its reader's frame, `x` the reader's side."""
    x, y = (jnp.reshape(t, (*t.shape[:-1], -1, 2)) for t in (x, y))
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def _before(x, axis: int):
    """`x`'s exclusive prefix sums along `axis`: each place's sum of those before it, the inclusive sums
    shifted one place, where the inclusive sum less the place's own lost a small sum beside a large term to
    its rounding (a chunk's last writer's whole weight beside its earlier ones' exp(-30) each under decays
    near zero: the turns' gradient's worst row 3.75 from float64 over its size)."""
    summed = jnp.cumsum(x, axis=axis)
    axis %= x.ndim
    padding = [(0, 0)] * x.ndim
    padding[axis] = (1, 0)
    return jnp.pad(jax.lax.slice_in_dim(summed, 0, x.shape[axis] - 1, axis=axis), padding)


def _suffix(x, axis: int):
    """`x`'s inclusive suffix sums along `axis`."""
    return jnp.flip(jnp.cumsum(jnp.flip(x, axis), axis=axis), axis)


def _split(x, trailing: int, *, blocks: int, half: int):
    """`x` `[..., C, *trailing]` as `[..., blocks, 2, half, *trailing]`, each block's two halves."""
    lead = x.shape[:x.ndim - 1 - trailing]
    return jnp.reshape(x, (*lead, blocks, 2, half, *x.shape[x.ndim - trailing:]))


def _halves(query, key, phi, value, d_out, log, beta, ids, teams, *, scale: float, group: int) -> jax.Array:
    """The crossing sums of a chunk's own pairs `[..., C, A]`: each token `t`'s `sum_{u < t <= s} dA_su (J
    q_s) . k_u`, `dA_su = scale beta_u exp(G_s - G_u) (do_s . v_u)` over pairs of one document (`ids`) and of
    later groups (`teams`, each token's group, a readout apart from its group's writes), by halves: a block's
    later half's pairs with its earlier half's cross each token of either half they lie between, rotated into
    the block midpoint's frame (`phi` less the earlier half's last angle, so readers and writers sum disjoint
    turns and share no prefix under second-order differentiation, row 8656), as a suffix over the later half
    of `(J q_s) . sum_u dA_su k_u` and a prefix over the earlier half of `(J sum_s dA_su q_s) . k_u`; each
    half's own pairs are the next level's. Every term summed crosses its token: no term cancels another. A
    chunk is a power of two of whole groups (`turn_gradient`'s), and a level whose blocks are one group each
    holds no pair a readout apart keeps: the levels stop there (a rank-four call's two smallest, each three
    small products a chunk). `query` and `key` `[..., C, 2 A]` unrotated, `phi` the chunk's within-chunk
    double-float turn prefix `(high, low)` `[..., C, A]`, `value` and `d_out` `[..., C, V]`, `log` the log
    decays' prefix `G` as a double float (`_between`) and `beta` `[..., C]`, `ids` and `teams` `[..., C]`
    broadcast against them."""
    length = query.shape[-2]
    total = jnp.zeros((*query.shape[:-1], query.shape[-1] // 2), query.dtype)
    size = length
    while size > group:
        half = size // 2
        split = functools.partial(_split, blocks=length // size, half=half)
        q, k, v, do = (split(x, 1) for x in (query, key, value, d_out))
        p = tuple(split(part, 1) for part in phi)
        G = tuple(split(part, 0) for part in log)
        b, doc, team = (split(x, 0) for x in (beta, *(jnp.broadcast_to(t, beta.shape) for t in (ids, teams))))
        kept = ((team[..., 1, :, None] > team[..., 0, None, :])
                & (doc[..., 1, :, None] == doc[..., 0, None, :]))
        lag = _between(tuple(part[..., 1, :, None] for part in G), tuple(part[..., 0, None, :] for part in G))
        # A pair of another document or group selected out before any product meets it, its score, decay
        # and gate weight each zero there: weighed by exp(-inf) after them, two documents' cotangent of
        # 2^127 and value of two were inf times zero, and a gate weight past the largest likewise (row 8566).
        scores = jnp.where(kept, jnp.einsum('...iv,...jv->...ij', do[..., 1, :, :], v[..., 0, :, :]), 0.0)
        weights = jnp.where(kept, scale * b[..., 0, None, :], 0.0)
        pairs = scores * jnp.exp(jnp.where(kept, lag, -jnp.inf)) * weights
        # Each half rotated into the block midpoint's frame: readers sum (mid, s], writers sum -(u, mid].
        # Scale operands above 2^125 by 0.5 (exact power of two) before rotating so a rotation's <=sqrt(2)
        # growth cannot overflow before the pair weight enters (base review b9f004f74 F1).
        neg_mid = tuple(-part[..., 0, -1:, :] for part in p)
        p_later = compensated_add(tuple(part[..., 1, :, :] for part in p), neg_mid)
        p_earlier = compensated_add(tuple(part[..., 0, :, :] for part in p), neg_mid)
        q_raw, k_raw = q[..., 1, :, :], k[..., 0, :, :]
        sq, inv_q = _rotated_scale(q_raw)
        sk, inv_k = _rotated_scale(k_raw)
        q_later = scan.rotated(q_raw * sq, *scan.rotation(*p_later)).astype(q_raw.dtype)
        k_earlier = scan.rotated(k_raw * sk, *scan.rotation(*p_earlier)).astype(k_raw.dtype)
        limit = np.float32(2.0 ** 125)
        small_p = jax.lax.stop_gradient(jnp.abs(pairs) <= limit)
        p_big = jnp.where(small_p, 0.0, pairs)
        p_small_k = jnp.where(small_p, pairs * jnp.swapaxes(inv_k, -1, -2), 0.0)
        k_inv = jnp.where(jax.lax.stop_gradient(jnp.abs(k_earlier) <= limit), k_earlier * inv_k, k_earlier)
        k_side = (jnp.einsum('...ij,...jk->...ik', p_small_k, k_earlier)
                  + jnp.einsum('...ij,...jk->...ik', p_big, k_inv))
        later = _suffix(_crossed(q_later, k_side) * inv_q, axis=-2)
        p_small_q = jnp.where(small_p, pairs * inv_q, 0.0)
        q_inv = jnp.where(jax.lax.stop_gradient(jnp.abs(q_later) <= limit), q_later * inv_q, q_later)
        q_side = (jnp.einsum('...ij,...ik->...jk', p_small_q, q_later)
                  + jnp.einsum('...ij,...ik->...jk', p_big, q_inv))
        terms = _crossed(q_side, k_earlier) * inv_k
        earlier = _before(terms, axis=-2)
        level = jnp.stack([earlier, later], axis=-3)
        total = total + jnp.reshape(level, (*level.shape[:-4], length, level.shape[-1]))
        size = half
    return total


def turn_gradient(query, key, value, d_out, g, beta, state, d_final, turn, starts, *, scale: float,
                  group: int, pairs: int, documents=None, held: int = 0, entry=None) -> jax.Array:
    """The cotangent of each token's turn `[B, T, H, A]` of a turned call on the template (`call`), in the
    crossing form: `sum_{u < t <= s} dA_su (J q_s) . k_u` over its pairs, `dA_su` a pair's score's cotangent
    (`scale beta_u` times its decays and `do_s . v_u`, of later groups of `group` where a readout is apart
    from its group's writes), `query` and `key` the unrotated queries and keys `[B, T, H, K]`, `value` and
    `d_out` `[B, T, H, V]`, `g` the log decays and `beta` the write gates `[B, T, H]`, `d_final` the final
    memory's cotangent in its last frame, `turn` `[B, T, H, A]`, `starts` `[B, T]` or None. Each chunk works
    in its own local frame (its entering memory in its opening frame, its leaving cotangent in its closing
    frame, and each level of `_halves` in its block midpoint's frame) so every contraction's two sides sum
    disjoint turns and cancel no shared prefix in first or second order (row 8656)."""
    turned = 2 * pairs
    # Chunks of whole groups, a power of two of them, so a group never straddles two chunks' memories.
    chunk = group * 2 ** max((CHUNK // group).bit_length() - 1, 0)
    rows = (query, key, value, d_out, g, beta, turn)
    if documents is None:
        walked = _packed(*rows, state, d_final, starts, scale=scale, turned=turned, group=group, chunk=chunk)
    else:
        walked = _laid_documents(*rows, state, d_final, documents, entry, scale=scale, turned=turned,
                                 held=held, group=group, chunk=chunk)
    ((q, k, phi, v, do, log, b, ids, teams, sides, memories_pair, leaving_pair, passes, chunk_rot),
     unlaid) = walked
    memories, mem_scales = memories_pair
    leaving, cot_scales = leaving_pair
    (readers, rem_q, reads), (writers, rem_k, writes) = sides
    own = _halves(q, k, phi, v, do, log, b, ids, teams, scale=scale, group=group)
    # A memory's contraction over the tokens of its document that read it, a cotangent's over those that
    # write it (`reads`, `writes`, each chunk's document's alone, a structure, not the readers' values, so
    # every order's derivative stays): another document's token taken out before the products, where a
    # discarded memory of 2^127 times its cotangent was inf and its zero reader made it NaN (row 8519).
    read_out = jnp.where(reads[..., None], do, 0.0)
    written_in = jnp.where(writes[..., None], v, 0.0)
    read_mat, read_post = _scaled_einsum(memories, read_out, rem_q * mem_scales)
    entered = _suffix(_crossed(readers, read_mat) * read_post, axis=-2)
    write_mat, write_post = _scaled_einsum(leaving, written_in, rem_k * cot_scales)
    terms = -(_crossed(writers, write_mat) * write_post)
    left = _before(terms, axis=-2)
    c_cos, c_sin = chunk_rot
    decayed = passes * memories
    p_fac, p_inv = _rotated_scale(jnp.swapaxes(decayed, -1, -2))
    p_fac, p_inv = jnp.swapaxes(p_fac, -1, -2), jnp.swapaxes(p_inv, -1, -2)
    passed = scan.rotated(decayed * p_fac, c_cos[..., None, :], -c_sin[..., None, :], axis=-2)
    total_scale = mem_scales * cot_scales * p_inv
    left_small = jax.lax.stop_gradient(
        jnp.max(jnp.abs(leaving), axis=(-2, -1), keepdims=True)
        <= jnp.max(jnp.abs(passed), axis=(-2, -1), keepdims=True))
    one_f = jnp.asarray(1.0, leaving.dtype)
    l_scaled = leaving * jnp.where(left_small, total_scale, one_f)
    p_scaled = passed * jnp.where(left_small, one_f, total_scale)
    through = jnp.sum(_crossed(jnp.swapaxes(l_scaled, -1, -2), jnp.swapaxes(p_scaled, -1, -2)), axis=-2)
    return unlaid(own + entered + left + through[..., None, :])

def _chunked(x, chunks: int, chunk: int):
    """`[B, L, ...]` as `[B, H, chunks, C, ...]`, heads after the batch (`[B, chunks, C]` where it has no
    heads)."""
    x = jnp.reshape(x, (x.shape[0], chunks, chunk, *x.shape[2:]))
    return jnp.moveaxis(x, 3, 1) if x.ndim > 3 else x


def _walk(step, carry, *xs, reverse: bool = False):
    """A scan of `step` over the chunks' axis (2) of each of `xs`, its outputs' laid there again."""
    _, ys = jax.lax.scan(step, carry, tuple(jnp.moveaxis(x, 2, 0) for x in xs), reverse=reverse)
    return jax.tree.map(lambda y: jnp.moveaxis(y, 0, 2), ys)


def _between(later, earlier) -> jax.Array:
    """A pair's log decay `G_later - G_earlier` of two double-float prefixes `(high, low)`, the highs'
    difference first (exact where they are near) then the lows': a prefix rounded once to fp32 put a pair's
    decay `eps |G|` off, where the chunk's `|G|` is many times the pair's own (the crossing form's turn
    gradient 2.5 to 5.3 times the recurrence's error from float64, its arithmetic in float64 0.5 times)."""
    return (later[0] - earlier[0]) + (later[1] - earlier[1])


def _rotated_scale(x):
    """Exact power-of-two factor (0.5 where any component exceeds 2^125, else 1.0) and its inverse so a 2-D
    rotation's <=sqrt(2) growth cannot overflow before a weight enters (base review b9f004f74 F1)."""
    limit = np.float32(2.0 ** 125)
    half, one = jnp.asarray(0.5, x.dtype), jnp.asarray(1.0, x.dtype)
    factor = jax.lax.stop_gradient(jnp.where(jnp.max(jnp.abs(x), axis=-1, keepdims=True) > limit, half, one))
    return factor, one / factor


def _reduced_turn(turn, work):
    """Reduce `turn` within pi in at least fp32, preserving wider angle dtypes (e.g. float64 under x64)
    through prefix sums and trigonometry (base reviews da734bf54 F2, b65e3cab4 F3, d80c85a37 F3)."""
    return scan.within_pi(jnp.asarray(turn, jnp.promote_types(at_least_fp32(turn.dtype), work)))


def _scaled_einsum(mat, vec, rem):
    """Contract `mat` `[..., K, V]` with `vec` `[..., C, V]` scaled by `rem` `[..., C, 1]`, returning
    `(product, post_scale)`: `rem` enters before contracting where `|vec| * rem <= 2^125`, else stays in
    `post_scale` to apply after the cross product so `2^127 * rem` never overflows before a small paired
    component absorbs it (base reviews b65e3cab4 F2, 1cf389a60 F1)."""
    limit = np.float32(2.0 ** 125)
    one = jnp.asarray(1.0, rem.dtype)
    small = jax.lax.stop_gradient(jnp.max(jnp.abs(vec), axis=-1, keepdims=True) <= limit / rem)
    pre, post = jnp.where(small, rem, one), jnp.where(small, one, rem)
    return jnp.einsum('bhnkv,bhncv->bhnck', mat, vec * pre), post


def _chunk_frames(g_laid, turn_laid, cut_laid):
    """Chunk-local log-decay prefix, turn-angle prefix, and closing rotation from laid operands and restarts
    (base review b9f004f74 F8)."""
    log = scan.compensated(g_laid, axis=-1, restarts=cut_laid)
    phi = scan.compensated(turn_laid, axis=-2,
                           restarts=jnp.broadcast_to(cut_laid[..., None], turn_laid.shape))
    return log, phi, scan.rotation(phi[0][..., -1, :], phi[1][..., -1, :])


def _transfer(state, kept, added, cos_c, sin_c):
    """Rotate a chunk's scaled `(matrix, scale)` by `(cos_c, sin_c)` and add `added`, halving before
    rotating whenever `kept * matrix` exceeds 2^125 so `sqrt(2)` never overflows (b65e3cab4 F1)."""
    matrix, m_scale = state
    decayed = kept * matrix
    limit, floor = np.float32(2.0 ** 125), np.float32(2.0 ** 124)
    half, one, two = (jnp.asarray(v, matrix.dtype) for v in (0.5, 1.0, 2.0))
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(decayed), axis=(-2, -1), keepdims=True))
    halve = peak > limit
    reset = (peak == 0.0) & (m_scale > one)
    grow = (m_scale > one) & (peak > 0.0) & (peak <= floor)
    rot_fac = jnp.where(halve, half, jnp.where(reset, m_scale, jnp.where(grow, two, one)))
    new_scale = jnp.where(reset, one, m_scale * (one / rot_fac))
    rotated = scan.rotated(decayed * rot_fac, cos_c[..., None, :], sin_c[..., None, :],
                           axis=-2).astype(matrix.dtype)
    return rotated + added * (one / new_scale), new_scale


def _sides(q, k, phi, log, b, reads, writes, scale: float):
    """A chunk's scaled readers and writers with any unabsorbed inverse rotation scale `(readers, rem_q)`
    and `(writers, rem_k)` kept apart: absorb `inv` into the bounded weight when `|weight| <= 0.5` so
    `|weight * inv| <= 1` and neither side overflows (base reviews da734bf54 F1, b65e3cab4 F2)."""
    sq, inv_q = _rotated_scale(q)
    sk, inv_k = _rotated_scale(k)
    q_open = scan.rotated(q * sq, *scan.rotation(*phi)).astype(q.dtype)
    phi_to_end = compensated_add(phi, tuple(-part[..., -1:, :] for part in phi))
    k_close = scan.rotated(k * sk, *scan.rotation(*phi_to_end)).astype(k.dtype)
    half, one = jnp.asarray(0.5, q.dtype), jnp.asarray(1.0, q.dtype)
    w_q = (scale * jnp.exp(jnp.where(reads, log[0] + log[1], -jnp.inf)))[..., None]
    abs_q = jax.lax.stop_gradient(jnp.abs(w_q) <= half)
    readers = q_open * jnp.where(abs_q, w_q * inv_q, w_q)
    rem_q = jnp.where(abs_q, one, inv_q)
    lag = _between(tuple(part[..., -1:] for part in log), log)
    w_k = (b * jnp.exp(jnp.where(writes, lag, -jnp.inf)))[..., None]
    abs_k = jax.lax.stop_gradient(jnp.abs(w_k) <= half)
    writers = k_close * jnp.where(abs_k, w_k * inv_k, w_k)
    rem_k = jnp.where(abs_k, one, inv_k)
    return (readers, rem_q), (writers, rem_k)

def _packed(query, key, value, d_out, g, beta, turn, state, d_final, starts, *, scale: float, turned: int,
            group: int, chunk: int):
    """`turn_gradient`'s chunks of a call of packed documents, each from a zero memory but a row's first
    (`state`): its tokens in `chunk`-token chunks (padded), each token's document's id (`starts`' count
    through it), and the memories entering each chunk (in its opening frame) and their cotangents leaving it
    (in its closing frame), walked over a row's chunks, a document's memory crossing no other's."""
    work = at_least_fp32(jnp.result_type(query, key, value, d_out))
    batch, length, heads, _ = query.shape
    pad = (-length) % chunk
    chunks = (length + pad) // chunk

    def laid(x, mode: str = 'constant'):
        return _chunked(jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2), mode=mode), chunks, chunk)

    q, k = (laid(jnp.asarray(x, work)[..., :turned]) for x in (query, key))
    v, do = (laid(jnp.asarray(x, work)) for x in (value, d_out))
    # Each chunk's log decays' prefix restarts at each document's first token: summed across a chunk's
    # documents, a pair of a later one took its decays as a difference of the earlier ones' (PaTH's row
    # 7609: a first document's -2^30 rounded the next one's to nothing). A token whose decay vanishes cuts
    # a head's as a document's start does (`scan.vanished`).
    begins = jnp.zeros((batch, length), bool) if starts is None else jnp.asarray(starts, bool)
    g, cut = scan.vanished(jnp.asarray(g, work))
    cut = cut | begins[..., None]
    cut_laid = laid(cut)
    log, phi, chunk_rot = _chunk_frames(laid(g), laid(_reduced_turn(turn, work)), cut_laid)
    b = laid(jnp.asarray(beta, work))
    ids = laid(jnp.cumsum(cut.astype(jnp.int32), axis=1), mode='edge')
    # Each token's group: a row's tokens' ranks in turn from its first, each document beginning a token.
    teams = laid(jnp.broadcast_to(jnp.arange(length) // group, (batch, length)), mode='edge')[:, None]
    before = jnp.concatenate([jnp.zeros_like(ids[..., :1, -1]), ids[..., :-1, -1]], axis=-1)
    reads, writes = ids == before[..., None], ids == ids[..., -1:]
    (readers, rem_q), (writers, rem_k) = _sides(q, k, phi, log, b, reads, writes, scale)
    # A chunk its memory passes through whole: one document's, no decay of it vanished.
    whole = jnp.where(ids[..., -1] == before, log[0][..., -1] + log[1][..., -1], -jnp.inf)
    passes = jnp.exp(whole)[..., None, None]
    written = jnp.einsum('bhnck,bhncv->bhnkv', writers, v * rem_k)
    read = jnp.einsum('bhnck,bhncv->bhnkv', readers, do * rem_q)
    entering = (jnp.zeros((batch, heads, turned, v.shape[-1]), work) if state is None
                else jnp.asarray(state, work)[:, :, :turned])
    unit = jnp.ones((batch, heads, 1, 1), work)

    def forward(carry, chunk):
        kept, wrote, cos_c, sin_c = chunk
        return _transfer(carry, kept, wrote, cos_c, -sin_c), carry

    def backward(carry, chunk):
        kept, reader, cos_c, sin_c = chunk
        return _transfer(carry, kept, reader, cos_c, sin_c), carry

    memories = _walk(forward, (entering, unit), passes, written, *chunk_rot)
    leaving = _walk(backward, (jnp.asarray(d_final, work)[:, :, :turned], unit), passes, read, *chunk_rot,
                    reverse=True)

    def unlaid(total):
        total = jnp.reshape(jnp.moveaxis(total, 1, 3), (batch, chunks * chunk, heads, total.shape[-1]))
        return total[:, :length]

    sides = ((readers, rem_q, reads), (writers, rem_k, writes))
    return (q, k, phi, v, do, log, b, ids, teams, sides, memories, leaving, passes, chunk_rot), unlaid

def _laid_documents(query, key, value, d_out, g, beta, turn, given, d_final, documents, entry, *,
                    scale: float, turned: int, held: int, group: int, chunk: int):
    """`turn_gradient`'s chunks of a forked call: each document laid from a chunk's edge (padded to whole
    chunks, as the template and the portable math lay it), the memories entering each chunk (in its opening
    frame) walked through the call's table (a document's first chunk enters its `loads` slot's memory and its
    entry write, its last leaves its final in its `stores` slot; slot zero none, the next `S'` the given
    memories `given`, then the stored ones), and their cotangents (in each chunk's closing frame) walked back
    (a stored slot's the sum of each document entering it, the row's last document's the final memory's)."""
    work = at_least_fp32(jnp.result_type(query, key, value, d_out))
    firsts, loads, stores = (jnp.asarray(t, jnp.int32) for t in documents)
    batch, length, heads, _ = query.shape
    count = firsts.shape[1]
    aligned = AlignedDocuments.of(Forked(firsts, loads, stores, None, held), length, chunk)
    chunks = aligned.length // chunk
    real = aligned.laid(jnp.ones((batch, length), jnp.bool_))[0]

    def laid(x):
        return _chunked(aligned.laid(x)[0], chunks, chunk)

    q, k = (laid(jnp.asarray(x, work)[..., :turned]) for x in (query, key))
    v, do = (laid(jnp.asarray(x, work)) for x in (value, d_out))
    # A token whose decay vanishes cuts its chunk's prefix (`scan.vanished`): a span through it is none.
    g, cut = scan.vanished(jnp.asarray(g, work))
    cut_laid = laid(cut)
    log, phi, chunk_rot = _chunk_frames(laid(g), laid(_reduced_turn(turn, work)), cut_laid)
    seen = jnp.cumsum(cut_laid.astype(jnp.int32), axis=-1)
    b = laid(jnp.asarray(beta, work))
    on = _chunked(real, chunks, chunk)[:, None]
    reads, writes = on & (seen == 0), on & (seen == seen[..., -1:])
    (readers, rem_q), (writers, rem_k) = _sides(q, k, phi, log, b, reads, writes, scale)
    head, alive, last, final = aligned.document, aligned.alive, aligned.last, aligned.final
    first = aligned.loads >= 0
    whole = jnp.where(seen[..., -1] == 0, log[0][..., -1] + log[1][..., -1], -jnp.inf)
    passes = (alive[:, None] * jnp.exp(whole))[..., None, None]
    written = jnp.einsum('bhnck,bhncv->bhnkv', writers, v * rem_k)
    read = jnp.einsum('bhnck,bhncv->bhnkv', readers, do * rem_q)
    shape = (batch, heads, turned, v.shape[-1])
    opened = jnp.zeros((batch, count, *shape[1:]), work)
    if entry is not None:
        # Each document's entry write, its terms summed aligned (`chunked_launch.entry_write`): formed one at
        # a time, a rank-two write of 2^80 keys and 2^80 values that cancels was inf - inf (base review
        # 2354cbaf7 F1).
        key, value, weight = (jnp.asarray(t, work) for t in entry)
        opened = entry_write(jnp.moveaxis(key[..., :turned], 2, 3), jnp.moveaxis(value, 2, 3),
                             weight[..., None, None])
    entries = jnp.moveaxis(jax.vmap(lambda row, at: row[at])(opened, head), 1, 2)
    slots = 1 + given.shape[1] + held
    table = jnp.concatenate([jnp.zeros((batch, 1, *shape[1:]), work),
                             jnp.asarray(given, work)[..., :turned, :],
                             jnp.zeros((batch, held, *shape[1:]), work)], axis=1)
    into = jnp.maximum(aligned.loads, 0)
    out = jnp.where(aligned.stores > 0, given.shape[1] + aligned.stores, 0)
    flags = tuple(x[:, None] for x in (first, last, final, into, out))
    empty = jnp.zeros(shape, work)
    unit = jnp.ones((batch, heads, 1, 1), work)
    table_unit = jnp.ones((batch, slots, heads, 1, 1), work)
    memories = _walk(functools.partial(_entering, slots=slots), (empty, unit, table, table_unit), *flags,
                     passes, written, entries, *chunk_rot)
    d_held = jnp.asarray(d_final, work)[:, :, :turned]
    leaving = _walk(functools.partial(_leaving, slots=slots, d_final=d_held),
                    (empty, unit, jnp.zeros_like(table), table_unit), *flags, passes, read, *chunk_rot,
                    reverse=True)

    def unlaid(total):
        return aligned.unlaid(jnp.reshape(jnp.moveaxis(total, 1, 3),
                                          (batch, aligned.length, heads, total.shape[-1])))

    # A chunk's tokens apart where a vanished decay cuts them (each chunk one document's).
    ids = head[:, None, :, None] * (chunk + 1) + seen
    # Each token's group, counted from its document's first token (a chunk's edge).
    teams = _chunked(aligned.within // group, chunks, chunk)[:, None]
    sides = ((readers, rem_q, reads), (writers, rem_k, writes))
    return (q, k, phi, v, do, log, b, ids, teams, sides, memories, leaving, passes, chunk_rot), unlaid


def _slot(at, slots: int, dtype):
    """A one-hot `[B, slots]` in `dtype` of each row's table slot `at` `[B, 1]`."""
    return jax.nn.one_hot(at[:, 0], slots, dtype=dtype)


def _entering(carry, chunk, *, slots: int):
    """A forked walk's chunk forward (`_laid_documents`): the scaled memory entering it (its document's first
    chunk's from its slot and its entry write), the memory leaving it (rotated into the chunk's closing
    frame), stored where its document's last chunk stores."""
    memory, m_scale, table, t_scale = carry
    begins, ends, _, into, out, kept, wrote, entry, cos_c, sin_c = chunk
    taken_mask = (_slot(into, slots, table.dtype) > 0)[..., None, None, None]
    taken_scale = jnp.sum(jnp.where(taken_mask, t_scale, 0.0), axis=1)
    taken = jnp.sum(jnp.where(taken_mask, table, 0.0), axis=1) + entry * (1.0 / taken_scale)
    memory = jnp.where(begins[..., None, None], taken, memory)
    m_scale = jnp.where(begins[..., None, None], taken_scale, m_scale)
    leaving, next_scale = _transfer((memory, m_scale), kept, wrote, cos_c, -sin_c)
    into_slot = _slot(out, slots, table.dtype) > 0
    stores = (ends & (out > 0))[..., None, None, None] & into_slot[..., None, None, None]
    return (leaving, next_scale, jnp.where(stores, leaving[:, None], table),
            jnp.where(stores, next_scale[:, None], t_scale)), (memory, m_scale)


def _leaving(carry, chunk, *, slots: int, d_final):
    """A forked walk's chunk backward (`_laid_documents`): the scaled cotangent of the memory leaving it (its
    document's last chunk's its stored slot's and, the row's last, the final memory's), the memory entering
    it's (rotated into the chunk's opening frame) summed into its slot where its document's first chunk
    enters it."""
    cotangent, c_scale, d_table, dt_scale = carry
    begins, ends, final, into, out, kept, reader, cos_c, sin_c = chunk
    slot_mask = ((out > 0)[..., None, None, None]
                 & (_slot(out, slots, d_table.dtype) > 0)[..., None, None, None])
    one = jnp.asarray(1.0, d_table.dtype)
    from_scale = jnp.where((out > 0)[..., None, None],
                           jnp.sum(jnp.where(slot_mask, dt_scale, 0.0), axis=1), one)
    from_slot = jnp.sum(jnp.where(slot_mask, d_table, 0.0), axis=1)
    ending = from_slot + jnp.where(final[..., None, None], d_final * (one / from_scale), 0.0)
    cotangent = jnp.where(ends[..., None, None], ending, cotangent)
    c_scale = jnp.where(ends[..., None, None], from_scale, c_scale)
    entering, next_scale = _transfer((cotangent, c_scale), kept, reader, cos_c, sin_c)
    enter_mask = ((begins & (into > 0))[..., None, None, None]
                  & (_slot(into, slots, d_table.dtype) > 0)[..., None, None, None])
    merged_scale = jnp.where(enter_mask, jnp.maximum(dt_scale, next_scale[:, None]), dt_scale)
    d_table = (d_table * (dt_scale / merged_scale)
               + jnp.where(enter_mask, entering[:, None] * (next_scale[:, None] / merged_scale), 0.0))
    return (entering, next_scale, d_table, merged_scale), (cotangent, c_scale)
