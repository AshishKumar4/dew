"""A program of the template's forward and its backward (its first pass, and its dQ and dK/dV walks by the
derived forms, `derived_backward`), as every mainloop that walks a block's tiles in a loop composes them (the
Pallas-Triton one, `attention_triton`, and the plain-JAX one,
`attention_portable`): each mainloop gives its own reads of a key tile (`keys_at`, `formed`) and of a key's
value row (`rows`), and these compose the algebra's steps over them (`attention_algebra`,
`attention_gradient`), once.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from dew.algebra.forms import Accumulated
from dew.nn.kernels.templates import attention_algebra as algebra, attention_gradient as gradient
from dew.nn.kernels.templates.attention import Kind, by_power
from dew.nn.kernels.templates.attention_backward_algebra import derived_backward


def walk(spec: algebra.Forward, lo, hi, keys_at: Callable[[jax.Array], algebra.Keys],
         rows: Callable[[jax.Array], jax.Array], resumed: tuple[jax.Array, jax.Array, jax.Array] | None,
         block_q: int, width: int, *, live: jax.Array | None, training: bool, dropping: bool = False):
    """A program's walk of its key tiles `lo` to `hi` (`keys_at(index)`, in `spec`'s form) from the initial
    state `resumed` `(m0, total0, acc0)` or none: the plain walk (`algebra.plain_walk`), or with `training`
    the training walk carrying the anchor key and split sums (`algebra.training_walk`, `rows(keys)` each key's
    value row); on the guarded path its `live` rows shifting (`algebra.started`)."""
    start = algebra.started(spec, resumed, block_q, width, live)
    if training:
        return algebra.training_walk(spec, lo, hi, keys_at, rows, start, dropping=dropping)
    return algebra.plain_walk(spec, lo, hi, keys_at, start)


def finished(spec: algebra.Forward, walked, keys_at: Callable[[jax.Array], algebra.Keys],
             rows: Callable[[jax.Array], jax.Array], bounds,
             resumed: tuple[jax.Array, jax.Array, jax.Array] | None, *, training: bool, output_mod=None,
             output_params: tuple = ()) -> tuple:
    """A program's outputs from its walk (`walk`): the normalized output, its rows' maximum and denominator
    (`algebra.finish`), and with `training` its anchor keys and split sums (`algebra.split_written`), a
    guarded walk's live rows' from a normalized pass over its key tiles `bounds` at the walk's maximum and
    denominator (`algebra.normalized_pass`, from the initial state's share)."""
    kept = ()
    state = walked
    if training:
        def normalized():
            return algebra.normalized_pass(spec, walked, keys_at, rows, bounds,
                                           None if resumed is None else (resumed[0], resumed[2]))

        state, anchors, sums = algebra.split_written(walked, rows, normalized)
        kept = (anchors, *sums)
    out, m, denominator = algebra.finish(spec, state, initial=resumed is not None, output_mod=output_mod,
                                         output_params=output_params)
    return (out, m, denominator, *kept)


def off_anchor_pass(lo, hi, formed: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
                    before: gradient.OffAnchor, maximum: jax.Array, chosen: jax.Array, shift: jax.Array,
                    how: algebra.Recompute, dot: Callable[[jax.Array, jax.Array], jax.Array]
                    ) -> gradient.OffAnchor:
    """A dQ program's first pass over its key tiles `lo` to `hi` without the forward's split sums
    (`gradient.off_anchor`): each tile's weights against the forward's `maximum`, the anchor keys `chosen`'s
    own apart, its products with the tile's values summed from `before` (`gradient.off_anchor_start`);
    `formed(index)` a tile's key positions `[1, bk]`, logits as the forward formed them and values."""
    def step(index, state):
        k_pos, logits, values = formed(index)
        at_anchor = k_pos == chosen[:, None]
        exponentials, rest, own, small = gradient.off_anchor_weights(logits, maximum, at_anchor, values.dtype,
                                                                    shift, how)
        return gradient.off_anchor(state, exponentials, rest, own, dot(rest, values), shift, small,
                                   dot(small, values))

    return jax.lax.fori_loop(lo, hi, step, before)


def accurate_sums(dot: Callable) -> Callable:
    """The accurate form's walked sums of products (`Derived.step`'s `stepped`): each the score cotangents'
    product with the keys (dQ's, `Accumulated`) by their parts (`algebra.cotangent_product`, the accurate
    form's three), added into the sum the step carries; every other sum the derived forms' own."""
    def stepped(states, parts):
        return tuple(None if not isinstance(part, Accumulated) else
                     (state if part.factor is None else algebra.expand(part.factor, state) * state)
                     + algebra.cotangent_product(part.weights, part.values, dot, accurate=True)
                     for state, part in zip(states, parts, strict=True))

    return stepped


def dq_sums(formed: Callable, q: jax.Array, do: jax.Array, m: jax.Array, anchor: jax.Array, delta: jax.Array,
            total: jax.Array, lo, hi, chosen: jax.Array, keep: Callable | None, *, hooked: bool,
            score_kinds: tuple[Kind, ...], summed: tuple[bool, ...], prescale: float, grad_scale: float,
            how: algebra.Recompute, accurate: bool, dot: Callable[[jax.Array, jax.Array], jax.Array]
            ) -> gradient.DqSums:
    """A dQ program's sums by the backward's derived forms (`derived_backward`, in the forward's units `how`
    and the form `accurate` names): dQ the walk's over its key tiles `lo` to `hi` (`formed(index)` a tile's
    key positions `[1, bk]`, mask, keys, values, logits as the forward formed them and their hook's slopes),
    at the pass's D (`anchor`, `delta`, its anchor keys `chosen`) and denominator `total`, dQ at the call's
    whole scale, its query the call's own (`q` the logits' operand, taken by the power of two `prescale`,
    `attention.prescale`: exactly undone), under dropout's multipliers `keep(k_pos)` where it drops; the row
    sums D's refinement and the hook's parameters read (`gradient.dq_rows`) from each tile's weights and
    cotangents. The accurate form's dQ products issued through `dot` (`accurate_sums`), its keyed sums its own
    (`gradient.dq_issued`)."""
    # dV's operands bf16's order where dO is bf16 (`derived_backward`'s `normalized_values`).
    forms = derived_backward(grad_scale, "keys", hooked, keep is not None, how, accurate, anchored=True,
                             normalized_values=do.dtype == jnp.bfloat16)
    call_q = by_power(q, 1.0 / prescale)

    def tiles_at(index):
        k_pos, seen, k_tile, v_tile, logits, slopes = formed(index)
        tiles = {"logits": logits, "q": call_q, "k": k_tile, "values": v_tile, "dout": do, "peak": m,
                 "total": total, "anchor": anchor, "delta": delta, "seen": seen}
        if slopes is not None:
            tiles["slope"] = slopes[0]
        if keep is not None:
            tiles["keep"] = keep(k_pos)
        # Each row's anchor key's own term exactly 0 in either form (`derived_backward`'s `anchored`).
        tiles["at_anchor"] = k_pos == chosen[:, None]
        return tiles, slopes, seen

    def step(index, carry):
        state, sums = carry
        tiles, slopes, seen = tiles_at(index)
        state, (weights, cotangents) = forms.step(state, tiles, product=dot, outputs=(3, 4),
                                                  stepped=accurate_sums(dot) if accurate else None)
        if not accurate:
            sums, _ = gradient.dq_rows(sums, weights, cotangents, slopes, seen, kinds=score_kinds,
                                       summed=summed)
            return state, sums
        k_tile = tiles["k"]
        sums, _, issued = gradient.dq_issued(sums, weights, cotangents, slopes, seen, k_tile.dtype, total,
                                             scale=grad_scale, kinds=score_kinds, summed=summed,
                                             accurate=True)
        assert issued is not None  # the accurate form's keyed sums
        factor, keyed = issued
        held = sums.keyed if factor is None else algebra.expand(factor, sums.keyed) * sums.keyed
        return state, sums._replace(keyed=held + algebra.summed_products(keyed, k_tile, dot))

    shapes = {name: (value.shape, value.dtype) for name, value in tiles_at(lo)[0].items()}
    empty = gradient.no_dq_sums(q.shape[0], q.shape[1], len(score_kinds))
    state, sums = jax.lax.fori_loop(lo, hi, step, (forms.init(shapes), empty))
    return sums._replace(dq=forms.finish(state, shapes=shapes)[0])


def dkv_sums(formed: Callable, k_tile: jax.Array, v_tile: jax.Array, lo, hi, pulled: Callable, hooks, *,
             hooked: bool, dropped: bool, grad_scale: float, how: algebra.Recompute, accurate: bool,
             score_kinds: tuple[Kind, ...], dot: Callable[[jax.Array, jax.Array], jax.Array]):
    """A dK/dV program's sums by the backward's derived forms (`derived_backward`, walked over the query tiles
    `lo` to `hi`, in the forward's units `how`), its keys `k_tile` and values `v_tile` its own:
    `formed(index)`
    a query tile's operands as the forms read them (`q`, `dout`, its mask, its logits as the forward formed
    them, its rows' maximum, denominator, anchor and D, its hook's slopes and dropout's multipliers) and its
    slopes and mask; each tile's score cotangents pulled through the hook (`gradient.pulled`) into `hooks` by
    `pulled(index, hooks, pieces)`; each product issued through the mainloop's `dot`. dK and dV `[bk, ...]`
    and the hooks' sums."""
    forms = derived_backward(grad_scale, "rows", hooked, dropped, how, accurate, anchored=True,
                             normalized_values=v_tile.dtype == jnp.bfloat16)

    def tiles_of(index):
        tiles, slopes, seen = formed(index)
        return {**tiles, "k": k_tile, "values": v_tile}, slopes, seen

    def step(index, carry):
        sums, held = carry
        tiles, slopes, seen = tiles_of(index)
        if slopes is None:
            return forms.step(sums, tiles, product=dot), held
        sums, (cotangents,) = forms.step(sums, tiles, product=dot, outputs=(4,))
        return sums, pulled(index, held, gradient.pulled(cotangents, slopes, seen, score_kinds)[1])

    shapes = {name: (value.shape, value.dtype) for name, value in tiles_of(lo)[0].items()}
    sums, hooks = jax.lax.fori_loop(lo, hi, step, (forms.init(shapes), hooks))
    dk, dv = forms.finish(sums, shapes=shapes)[1:3]
    return dk, dv, hooks
