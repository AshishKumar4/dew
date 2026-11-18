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
from dew.nn.derivatives import KernelGlue, contracted, declared_derivative
from dew.nn.precision import at_least_fp32, compensated_add, two_product

TWO_PI = (np.float32(2 * np.pi), np.float32(2 * np.pi - float(np.float32(2 * np.pi))))
"""2 pi as fp32's nearest and what it drops, a turn's whole in double float."""

CHUNK = 64
"""The crossing form's chunk: its own pairs taken in halves (`_halves`), the rest through its memories."""


def _wrapped(angle):
    """A double-float angle `(high, low)` less its whole turns (`TWO_PI`, each turn's product exact,
    Dekker's), about [0, 2 pi)."""
    turns = jnp.floor(angle[0] * np.float32(1 / (2 * np.pi)))
    whole, dropped = two_product(-turns, jnp.full_like(turns, TWO_PI[0]))
    return compensated_add(angle, (whole, dropped - turns * TWO_PI[1]))


def _wide(x) -> jax.Array:
    """`x` in at least fp32, its own dtype where wider (a float64 truth's)."""
    x = jnp.asarray(x)
    return x.astype(at_least_fp32(x.dtype))


def frames(turn, starts=None, entering=None) -> tuple[jax.Array, jax.Array]:
    """Each token's frame `[B, T, ..., A]`: its document's turns through it (`turn` `[B, T, ..., A]`; `starts`
    `[B, T]` each document's first token, where given), each within pi of zero first (`scan.within_pi`), their
    compensated prefix (`_prefix`) on from `entering` `[B, ..., A]` in the row's first document (where given:
    a carried angle), less its whole turns, a double float `(high, low)` in about [0, 2 pi). Within pi, a
    document's prefix is at most its length times pi, which a double float holds to 2^-48 of itself, below
    fp32's rounding of a turned key past 2^20 tokens, and its whole turns' count under 2^19, its product with
    2 pi's high part exact."""
    turn = jnp.asarray(turn)
    within = scan.within_pi(turn.astype(at_least_fp32(turn.dtype)))
    restarts = jnp.zeros(within.shape[:2], bool) if starts is None else jnp.asarray(starts, bool)
    restarts = jnp.reshape(restarts, (*restarts.shape, *(1,) * (within.ndim - 2)))
    summed = _prefix(within, restarts)
    if entering is not None:
        first = jnp.cumsum(restarts.astype(jnp.int32), axis=1) == 0
        on = jnp.where(first, jnp.asarray(entering, within.dtype)[:, None], 0.0)
        summed = compensated_add(summed, (jnp.broadcast_to(on, within.shape), jnp.zeros_like(within)))
    return _wrapped(summed)


def _prefix(values, restarts) -> tuple[jax.Array, jax.Array]:
    """`values` `[B, T, ...]`'s inclusive prefix sums along their tokens as double floats, restarting at
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
        total = compensated_add(total, (value, jnp.zeros_like(value)))
        return total, total

    zero = jnp.zeros((batch, chunks, *values.shape[2:]), values.dtype)
    flags = laid(restarts)
    _, local = jax.lax.scan(within, (zero, zero), (laid(values), flags))
    restarted = jnp.any(flags, axis=0)

    def across(entering, chunk):
        end, broke = chunk
        leaving = compensated_add(entering, end)
        return tuple(jnp.where(broke, e, part) for e, part in zip(end, leaving, strict=True)), entering

    ends = tuple(jnp.moveaxis(part[-1], 1, 0) for part in local)
    _, offsets = jax.lax.scan(across, (zero[:, 0], zero[:, 0]), (ends, jnp.moveaxis(restarted, 1, 0)))
    seen = jnp.cumsum(flags.astype(jnp.int32), axis=0) > 0
    offsets = tuple(jnp.moveaxis(part, 0, 1)[None] for part in offsets)
    summed = compensated_add(local, offsets)
    summed = tuple(jnp.where(seen, part, whole) for part, whole in zip(local, summed, strict=True))
    high, low = (jnp.reshape(jnp.moveaxis(part, 0, 2), (batch, chunks * CHUNK, *values.shape[2:]))[:, :length]
                 for part in summed)
    return high, low


def call(launch: Callable, query, key, value, g, beta, state, skip, turn, starts, *, scale: float,
         group: int, documents=None, held: int = 0, entry=None) -> tuple[jax.Array, jax.Array]:
    """`launch(query, key, value, g, beta, state, skip, entry)`'s `(output, final memory)` (the template's,
    its memories `[B, H, K, V]`) of a call whose state turns its first `A` key pairs each token by `turn` `[B,
    T, H, A]` (`recurrence.Hooks.turns`), `starts` `[B, T]` each document's first token or None, a readout
    `apart` from its token's `group` of writes where `group` is past one (`mainloop.Call.apart`): its queries
    and keys turned into their documents' frames (`frames`) less the row's last token's (`_in_last_frame`),
    in at least fp32 and rounded once to their dtype, its memory entering turned into that frame too, so its
    final memory is in the last token's frame as the launch leaves it. A forked call (`documents`, its `(firsts, loads, stores)` `[B,
    D]` and `held` stored slots, `chunked_launch.Forked`'s; `state` its table's given memories `[B, S', H, K,
    V]`; `entry` its documents' entry writes `(key [B, D, R, H, K], value [B, D, R, H, V], weight [B, D, H])`
    or None) runs each document's frame on from the one its source left (`_forked_frames`), so a memory a
    document stores is in the frame a document entering it starts in, and turns its entry keys into it. Its
    gradient takes the template's through the frames and its turns' in the crossing form (`turn_gradient`);
    a given memory is in the frame of its document's first token, as given, turned into the common frame."""
    return _turned(launch, scale, group, held, query, key, value, g, beta, state, skip, entry, turn, starts,
                   documents)


def _turned_primal(launch, scale, group, held, query, key, value, g, beta, state, skip, entry, turn, starts,
                   documents) -> tuple[jax.Array, jax.Array]:
    return _turned_forward(launch, scale, group, held, query, key, value, g, beta, state, skip, entry, turn,
                           starts, documents)[0]


def _in_last_frame(tokens, opening):
    """Each token's frame and each document's opening frame (`frames`, `_forked_frames`) less its row's last
    token's (`Psi`), as rotations `(cos, sin)`, and the rotation by `-Psi` `[B, 1, H, A]`: the common frame
    the call is read in its row's last token's, so a recent write's key turns by nearly nothing and the
    final memory needs no turn back. Read in the first document's frame and turned back, every recent write
    took two roundings of a full turn the recurrence's takes none of: under a decay keeping a few tokens the
    final memory 1.9 times the recurrence's error from float64, an angle of 1.2e-7 radians its most."""
    last = tuple(part[:, -1:] for part in tokens)
    back = tuple(-part for part in last)
    relative = compensated_add(tokens, tuple(jnp.broadcast_to(part, tokens[0].shape) for part in back))
    opened = None
    if opening is not None:
        opened = scan.rotation(*compensated_add(opening, tuple(jnp.broadcast_to(part, opening[0].shape)
                                                               for part in back)))
    return scan.rotation(*relative), opened, scan.rotation(*back)


def _memory_axis(rotation, memory):
    """A row's rotation `[B, 1, H, A]` laid against a memory's key axis: `[B, H, 1, A]` for a call's memory
    `[B, H, K, V]`, `[B, 1, H, 1, A]` for a table's `[B, S', H, K, V]`."""
    if memory.ndim == 4:
        return tuple(part[:, 0, :, None, :] for part in rotation)
    return tuple(part[:, :, :, None, :] for part in rotation)


def _turned_forward(launch, scale, group, held, query, key, value, g, beta, state, skip, entry, turn, starts,
                    documents) -> tuple[tuple[jax.Array, jax.Array], tuple]:
    del scale, group, held
    if documents is None:
        tokens, opening = frames(turn, starts), None
    else:
        tokens, opening = _forked_frames(turn, starts, documents, given=state.shape[1])
    (cos, sin), opened, back = _in_last_frame(tokens, opening)
    turned = [scan.rotated(_wide(x), cos, sin).astype(x.dtype) for x in (query, key)]
    if state is not None:
        # The memory entering, in the first frame, turned into the common one.
        state = scan.rotated(_wide(state), *_memory_axis(back, state), axis=-2).astype(state.dtype)
    if entry is not None and opened is not None:
        # The entry keys in the frame their document opens in.
        keys = scan.rotated(_wide(entry[0]), *(part[:, :, None] for part in opened))
        entry = (keys.astype(entry[0].dtype), *entry[1:])
    (out, final), pullback = jax.vjp(launch, *turned, value, g, beta, state, skip, entry)
    kind = jnp.zeros((), turn.dtype)
    return (out, final), (pullback, *turned, value, g, beta, state, entry, starts, documents, cos, sin,
                          opened, back, kind)


def _turned_backward(launch, scale, group, held, residuals, cotangents):
    del launch
    (pullback, query, key, value, g, beta, state, entry, starts, documents, cos, sin, opened, back,
     turn_kind) = residuals
    d_out, d_final = cotangents
    d_query, d_key, d_value, d_g, d_beta, d_state, d_skip, d_entry = pullback((d_out, d_final))
    unturned = [scan.rotated(_wide(d), cos, -sin).astype(d.dtype) for d in (d_query, d_key)]
    if d_state is not None:
        cos_back, sin_back = _memory_axis(back, d_state)
        d_state = scan.rotated(_wide(d_state), cos_back, -sin_back, axis=-2).astype(d_state.dtype)
    if d_entry is not None and opened is not None:
        keys = scan.rotated(_wide(d_entry[0]), opened[0][:, :, None], -opened[1][:, :, None])
        d_entry = (keys.astype(d_entry[0].dtype), *d_entry[1:])
    d_turn = turn_gradient(query, key, value, d_out, g, beta, state, d_final, starts, scale=scale,
                           group=group, pairs=cos.shape[-1], documents=documents, held=held, entry=entry)
    return (*unturned, d_value, d_g, d_beta, d_state, d_skip, d_entry, d_turn.astype(turn_kind.dtype),
            None, None)


_turned = contracted("test_a_turned_template_call_holds_every_order")(
    declared_derivative(KernelGlue(), _turned_primal, _turned_forward, _turned_backward,
                        nondiff_argnums=(0, 1, 2, 3)))
"""`call`'s rule: the template's launches opaque within (`KernelGlue`), its turns' gradient the crossing
form's."""


def _taken(x, index):
    """`x` `[B, D, ...]` at each row's `index` `[B, ...]` (its documents' axis, gathered row by row)."""
    index = jnp.asarray(index, jnp.int32)
    return jax.vmap(lambda row, at: row[at])(x, index)


def _forked_frames(turn, starts, documents, *, given: int):
    """A forked call's frames: each token's `(high, low)` `[B, T, H, A]` and each document's opening frame
    `[B, D, H, A]`, a document's its source's last token's where its slot is one an earlier document stored
    (`loads` past `given`), else zero (a given memory's frame), its tokens' on from it (`frames`, each
    wrapped once more)."""
    firsts, loads, stores = (jnp.asarray(t, jnp.int32) for t in documents)
    local = frames(turn, starts)
    batch, length = turn.shape[:2]
    count = firsts.shape[1]
    following = jnp.concatenate([firsts[:, 1:], jnp.full((batch, 1), length)], axis=1)
    lasts = jnp.clip(following - 1, 0, length - 1)
    ends = tuple(_taken(part, lasts) for part in local)
    zero = jnp.zeros((batch, *turn.shape[2:]), local[0].dtype)
    slots = tuple(jnp.zeros((batch, count + 1, *turn.shape[2:]), local[0].dtype) for _ in range(2))
    opening = []
    for document in range(count):
        stored = loads[:, document] > given
        source = jnp.clip(loads[:, document] - given, 0, count)
        start = tuple(jnp.where(stored.reshape(-1, *(1,) * (zero.ndim - 1)), _taken(slot, source), zero)
                      for slot in slots)
        opening.append(start)
        end = _wrapped(compensated_add(start, tuple(part[:, document] for part in ends)))
        into = jax.nn.one_hot(stores[:, document], count + 1, dtype=bool)
        into = into & (stores[:, document] > 0)[:, None]
        into = into.reshape(*into.shape, *(1,) * (zero.ndim - 1))
        slots = tuple(jnp.where(into, part[:, None], slot) for part, slot in zip(end, slots, strict=True))
    opening = tuple(jnp.stack(parts, axis=1) for parts in zip(*opening, strict=True))
    document_of = jnp.sum(jnp.arange(length)[None, :, None] >= firsts[:, None, :], axis=-1) - 1
    on = tuple(_taken(part, jnp.clip(document_of, 0, count - 1)) for part in opening)
    return _wrapped(compensated_add(local, on)), opening


def _crossed(x, y):
    """Each pair's `(J x) . y` `[..., A]` of `x` and `y` `[..., 2 A]` (`J` a quarter turn, `i` on a pair
    `x_2a + i x_2a+1`): a pair's score's derivative in its reader's frame, `x` the reader's side."""
    x, y = (jnp.reshape(t, (*t.shape[:-1], -1, 2)) for t in (x, y))
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def _suffix(x, axis: int):
    """`x`'s inclusive suffix sums along `axis`."""
    return jnp.flip(jnp.cumsum(jnp.flip(x, axis), axis=axis), axis)


def _split(x, trailing: int, *, blocks: int, half: int):
    """`x` `[..., C, *trailing]` as `[..., blocks, 2, half, *trailing]`, each block's two halves."""
    lead = x.shape[:x.ndim - 1 - trailing]
    return jnp.reshape(x, (*lead, blocks, 2, half, *x.shape[x.ndim - trailing:]))


def _halves(query, key, value, d_out, log, beta, ids, teams, *, scale: float, group: int) -> jax.Array:
    """The crossing sums of a chunk's own pairs `[..., C, A]`: each token `t`'s `sum_{u < t <= s} dA_su (J
    q_s) . k_u`, `dA_su = scale beta_u exp(G_s - G_u) (do_s . v_u)` over pairs of one document (`ids`) and of
    later groups (`teams`, each token's group, a readout apart from its group's writes), by halves: a block's
    later half's pairs with its earlier half's cross each token of either half they lie between, as a suffix
    over the later half of `(J q_s) . sum_u dA_su k_u` and a prefix over the earlier half of `(J sum_s dA_su
    q_s) . k_u`; each half's own pairs are the next level's. Every term summed crosses its token: no term
    cancels another. A chunk is a power of two of whole groups (`turn_gradient`'s), and a level whose
    blocks are one group each holds no pair a readout apart keeps: the levels stop there (a rank-four call's
    two smallest, each three small products a chunk). `query` and `key` `[..., C, 2 A]`,
    `value` and `d_out` `[..., C, V]`, `log` the log decays' prefix `G` as a double float (`_between`) and
    `beta` `[..., C]`, `ids` and
    `teams` `[..., C]` broadcast against them."""
    length = query.shape[-2]
    total = jnp.zeros((*query.shape[:-1], query.shape[-1] // 2), query.dtype)
    size = length
    while size > group:
        half = size // 2
        split = functools.partial(_split, blocks=length // size, half=half)
        q, k, v, do = (split(x, 1) for x in (query, key, value, d_out))
        G = tuple(split(part, 0) for part in log)
        b, doc, team = (split(x, 0) for x in (beta, *(jnp.broadcast_to(t, beta.shape) for t in (ids, teams))))
        kept = ((team[..., 1, :, None] > team[..., 0, None, :])
                & (doc[..., 1, :, None] == doc[..., 0, None, :]))
        lag = _between(tuple(part[..., 1, :, None] for part in G), tuple(part[..., 0, None, :] for part in G))
        pairs = (jnp.einsum('...iv,...jv->...ij', do[..., 1, :, :], v[..., 0, :, :])
                 * jnp.exp(jnp.where(kept, lag, -jnp.inf)))
        pairs = pairs * (scale * b[..., 0, None, :])
        later = _suffix(_crossed(q[..., 1, :, :], jnp.einsum('...ij,...jk->...ik', pairs, k[..., 0, :, :])),
                        axis=-2)
        terms = _crossed(jnp.einsum('...ij,...ik->...jk', pairs, q[..., 1, :, :]), k[..., 0, :, :])
        earlier = jnp.cumsum(terms, axis=-2) - terms
        level = jnp.stack([earlier, later], axis=-3)
        total = total + jnp.reshape(level, (*level.shape[:-4], length, level.shape[-1]))
        size = half
    return total


def turn_gradient(query, key, value, d_out, g, beta, state, d_final, starts, *, scale: float, group: int,
                  pairs: int, documents=None, held: int = 0, entry=None) -> jax.Array:
    """The cotangent of each token's turn `[B, T, H, A]` of a turned call on the template (`call`), in the
    crossing form: `sum_{u < t <= s} dA_su (J q_s) . k_u` over its pairs, `dA_su` a pair's score's cotangent
    (`scale beta_u` times its decays and `do_s . v_u`, of later groups of `group` where a readout is apart
    from its group's writes), `q` and `k` the turned queries and keys `[B, T, H, K]` as the template read
    them, `value` and `d_out` `[B, T, H, V]`, `g` the log decays and `beta` the write gates `[B, T, H]`,
    `d_final` the final memory's cotangent in its last frame, `starts` `[B, T]` or None. A chunk's own pairs
    by halves (`_halves`); a pair across chunks through the memory entering a chunk (`S_c`, its readers
    within the chunk: a suffix over them), the cotangent of the memory leaving it (`dS_c`, its writers within:
    a prefix) and both for a chunk it passes through whole (`<S_c, J dS_c>`, its every token). `state` the
    memory entering `[B, H, K, V]` (in its document's first frame) or None; a forked call's (`documents`,
    `held`, `entry`, `call`'s) its table's given memories, each document laid from a chunk's edge and its
    memories walked through the table (`_laid_documents`). Only the first `pairs` pairs of a key turn."""
    turned = 2 * pairs
    # Chunks of whole groups, a power of two of them, so a group never straddles two chunks' memories.
    chunk = group * 2 ** max((CHUNK // group).bit_length() - 1, 0)
    rows = (query, key, value, d_out, g, beta)
    if documents is None:
        walked = _packed(*rows, state, d_final, starts, scale=scale, turned=turned, group=group, chunk=chunk)
    else:
        walked = _laid_documents(*rows, state, d_final, documents, entry, scale=scale, turned=turned,
                                 held=held, group=group, chunk=chunk)
    (q, k, v, do, log, b, ids, teams, readers, writers, memories, leaving, passes), unlaid = walked
    own = _halves(q, k, v, do, log, b, ids, teams, scale=scale, group=group)
    entered = _suffix(_crossed(readers, jnp.einsum('bhnkv,bhncv->bhnck', memories, do)), axis=-2)
    terms = -_crossed(writers, jnp.einsum('bhnkv,bhncv->bhnck', leaving, v))
    left = jnp.cumsum(terms, axis=-2) - terms
    through = jnp.sum(_crossed(jnp.swapaxes(leaving, -1, -2), jnp.swapaxes(passes * memories, -1, -2)),
                      axis=-2)
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


def _sides(q, k, log, b, reads, writes, scale: float):
    """A chunk's readers of the memory entering it, `q_s scale exp(G_s)`, and writers into the memory
    leaving it, `k_u beta_u exp(G_end - G_u)`, of the chunk's document alone (`reads`, `writes`)."""
    readers = q * (scale * jnp.exp(log[0] + log[1]) * reads)[..., None]
    writers = k * (b * jnp.exp(_between(tuple(part[..., -1:] for part in log), log)) * writes)[..., None]
    return readers, writers


def _packed(query, key, value, d_out, g, beta, state, d_final, starts, *, scale: float, turned: int,
            group: int, chunk: int):
    """`turn_gradient`'s chunks of a call of packed documents, each from a zero memory but a row's first
    (`state`): its tokens in `chunk`-token chunks (padded), each token's document's id (`starts`' count
    through it), and the memories entering each chunk and their cotangents leaving it, walked over a row's
    chunks, a document's memory crossing no other's."""
    work = at_least_fp32(jnp.result_type(query, key, value, d_out))
    batch, length, heads, _ = query.shape
    pad = (-length) % chunk
    chunks = (length + pad) // chunk

    def laid(x, mode: str = 'constant'):
        return _chunked(jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2), mode=mode), chunks, chunk)

    q, k = (laid(jnp.asarray(x, work)[..., :turned]) for x in (query, key))
    v, do = (laid(jnp.asarray(x, work)) for x in (value, d_out))
    begins = jnp.zeros((batch, length), bool) if starts is None else jnp.asarray(starts, bool)
    # Each chunk's log decays' prefix restarts at each document's first token: summed across a chunk's
    # documents, a pair of a later one took its decays as a difference of the earlier ones' (PaTH's row
    # 7609: a first document's -2^30 rounded the next one's to nothing).
    g = laid(jnp.asarray(g, work))
    log = scan.compensated(g, axis=-1, restarts=jnp.broadcast_to(laid(begins)[:, None], g.shape))
    b = laid(jnp.asarray(beta, work))
    ids = laid(jnp.cumsum(begins.astype(jnp.int32), axis=1), mode='edge')[:, None]
    # Each token's group: a row's tokens' ranks in turn from its first, each document beginning a token.
    teams = laid(jnp.broadcast_to(jnp.arange(length) // group, (batch, length)), mode='edge')[:, None]
    before = jnp.concatenate([jnp.zeros_like(ids[..., :1, -1]), ids[..., :-1, -1]], axis=-1)
    readers, writers = _sides(q, k, log, b, ids == before[..., None], ids == ids[..., -1:], scale)
    whole = jnp.exp(log[0][..., -1] + log[1][..., -1])
    passes = ((ids[..., -1] == before) * whole)[..., None, None]
    written = jnp.einsum('bhnck,bhncv->bhnkv', writers, v)
    read = jnp.einsum('bhnck,bhncv->bhnkv', readers, do)
    entering = (jnp.zeros((batch, heads, turned, v.shape[-1]), work) if state is None
                else jnp.asarray(state, work)[:, :, :turned])

    def forward(memory, chunk):
        kept, wrote = chunk
        return kept * memory + wrote, memory

    def backward(cotangent, chunk):
        kept, reader = chunk
        return reader + kept * cotangent, cotangent

    memories = _walk(forward, entering, passes, written)
    leaving = _walk(backward, jnp.asarray(d_final, work)[:, :, :turned], passes, read, reverse=True)

    def unlaid(total):
        total = jnp.reshape(jnp.moveaxis(total, 1, 3), (batch, chunks * chunk, heads, total.shape[-1]))
        return total[:, :length]

    return (q, k, v, do, log, b, ids, teams, readers, writers, memories, leaving, passes), unlaid


def _document_of(firsts, length: int):
    """Each token's document `[B, T]`: the count of `firsts` `[B, D]` at or before it, less one."""
    return jnp.sum(jnp.arange(length)[None, :, None] >= firsts[:, None, :], axis=-1) - 1


def _laid_documents(query, key, value, d_out, g, beta, given, d_final, documents, entry, *, scale: float,
                    turned: int, held: int, group: int, chunk: int):
    """`turn_gradient`'s chunks of a forked call: each document laid from a chunk's edge (padded to whole
    chunks, as the template and the portable math lay it), the memories entering each chunk walked through
    the call's table (a document's first chunk enters its `loads` slot's memory and its entry write, its last
    leaves its final in its `stores` slot; slot zero none, the next `S'` the given memories `given`, then the
    stored ones), and their cotangents walked back (a stored slot's the sum of each document entering it,
    the row's last document's the final memory's)."""
    work = at_least_fp32(jnp.result_type(query, key, value, d_out))
    firsts, loads, stores = (jnp.asarray(t, jnp.int32) for t in documents)
    batch, length, heads, _ = query.shape
    count = firsts.shape[1]
    ends = jnp.concatenate([firsts[:, 1:], jnp.full((batch, 1), length)], axis=1)
    sizes = jnp.clip(jnp.minimum(ends, length) - jnp.minimum(firsts, length), 0)
    spans = -(-sizes // chunk) * chunk
    offsets = jnp.cumsum(spans, axis=1) - spans
    chunks = -(-(length + count * (chunk - 1)) // chunk)
    places = jnp.arange(chunks * chunk)
    document = jnp.sum(places[None, :, None] >= (offsets + spans)[:, None, :], axis=-1)
    live = document < count
    document = jnp.minimum(document, count - 1)
    within = places[None] - _taken(offsets, document)
    real = live & (within < _taken(sizes, document))
    token = jnp.where(real, _taken(firsts, document) + within, length)

    def laid(x):
        x = jnp.concatenate([x, jnp.zeros_like(x[:, :1])], axis=1)
        return _chunked(jax.vmap(lambda row, at: row[at])(x, token), chunks, chunk)

    q, k = (laid(jnp.asarray(x, work)[..., :turned]) for x in (query, key))
    v, do = (laid(jnp.asarray(x, work)) for x in (value, d_out))
    log = scan.compensated(laid(jnp.asarray(g, work)), axis=-1)
    b = laid(jnp.asarray(beta, work))
    on = _chunked(real, chunks, chunk)[:, None]
    readers, writers = _sides(q, k, log, b, on, on, scale)
    head = _chunked(document, chunks, chunk)[..., 0]
    alive = _chunked(live, chunks, chunk)[..., 0]
    first = alive & (jnp.arange(chunks)[None] * chunk == _taken(offsets, head))
    last = alive & ((jnp.arange(chunks)[None] + 1) * chunk == _taken(offsets + spans, head))
    final = last & (head == _document_of(firsts, length)[:, -1:])
    passes = (alive[:, None] * jnp.exp(log[0][..., -1] + log[1][..., -1]))[..., None, None]
    written = jnp.einsum('bhnck,bhncv->bhnkv', writers, v)
    read = jnp.einsum('bhnck,bhncv->bhnkv', readers, do)
    shape = (batch, heads, turned, v.shape[-1])
    opened = jnp.zeros((batch, count, *shape[1:]), work)
    if entry is not None:
        opened = jnp.einsum('bdrhk,bdrhv,bdh->bdhkv', jnp.asarray(entry[0], work)[..., :turned],
                            *(jnp.asarray(t, work) for t in entry[1:]))
    entries = jnp.moveaxis(jax.vmap(lambda row, at: row[at])(opened, head), 1, 2)
    slots = 1 + given.shape[1] + held
    table = jnp.concatenate([jnp.zeros((batch, 1, *shape[1:]), work),
                             jnp.asarray(given, work)[..., :turned, :],
                             jnp.zeros((batch, held, *shape[1:]), work)], axis=1)
    into = _taken(loads, head)
    out = _taken(stores, head)
    out = jnp.where(out > 0, given.shape[1] + out, 0)
    flags = tuple(x[:, None] for x in (first, last, final, into, out))
    empty = jnp.zeros(shape, work)
    memories = _walk(functools.partial(_entering, slots=slots), (empty, table), *flags, passes, written,
                     entries)
    d_held = jnp.asarray(d_final, work)[:, :, :turned]
    leaving = _walk(functools.partial(_leaving, slots=slots, d_final=d_held), (empty, jnp.zeros_like(table)),
                    *flags, passes, read, reverse=True)
    of = _document_of(firsts, length)
    position = _taken(offsets, of) + jnp.arange(length)[None] - _taken(firsts, of)

    def unlaid(total):
        total = jnp.reshape(jnp.moveaxis(total, 1, 3), (batch, chunks * chunk, heads, total.shape[-1]))
        return jax.vmap(lambda row, at: row[at])(total, position)

    ids = jnp.broadcast_to(head[:, None, :, None], (batch, 1, chunks, chunk))
    # Each token's group, counted from its document's first token (a chunk's edge).
    teams = _chunked(within // group, chunks, chunk)[:, None]
    return (q, k, v, do, log, b, ids, teams, readers, writers, memories, leaving, passes), unlaid


def _slot(at, slots: int, dtype):
    """A one-hot `[B, slots]` in `dtype` of each row's table slot `at` `[B, 1]`."""
    return jax.nn.one_hot(at[:, 0], slots, dtype=dtype)


def _entering(carry, chunk, *, slots: int):
    """A forked walk's chunk forward (`_laid_documents`): the memory entering it (its document's first
    chunk's from its slot and its entry write), the memory leaving it, stored where its document's last
    chunk stores."""
    memory, table = carry
    begins, ends, _, into, out, kept, wrote, entry = chunk
    taken = jnp.einsum('bs,bshkv->bhkv', _slot(into, slots, table.dtype), table) + entry
    memory = jnp.where(begins[..., None, None], taken, memory)
    leaving = kept * memory + wrote
    into_slot = _slot(out, slots, table.dtype) > 0
    stores = (ends & (out > 0))[..., None, None, None] & into_slot[..., None, None, None]
    return (leaving, jnp.where(stores, leaving[:, None], table)), memory


def _leaving(carry, chunk, *, slots: int, d_final):
    """A forked walk's chunk backward (`_laid_documents`): the cotangent of the memory leaving it (its
    document's last chunk's its stored slot's and, the row's last, the final memory's), the memory entering
    it's summed into its slot where its document's first chunk enters it."""
    cotangent, d_table = carry
    begins, ends, final, into, out, kept, reader = chunk
    from_slot = jnp.einsum('bs,bshkv->bhkv', _slot(out, slots, d_table.dtype), d_table)
    from_slot = from_slot * (out > 0)[..., None, None]
    ending = from_slot + jnp.where(final[..., None, None], d_final, 0.0)
    cotangent = jnp.where(ends[..., None, None], ending, cotangent)
    entering = reader + kept * cotangent
    entered = _slot(into, slots, d_table.dtype)[..., None, None, None] * entering[:, None]
    d_table = d_table + jnp.where(begins[..., None, None, None], entered, 0.0)
    return (entering, d_table), cotangent
