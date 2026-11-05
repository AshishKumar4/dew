"""The attention backward's declared algebra (theme A's declarations, docs/design/algebra.md §6.9) and its
derived forms behind the switch (`dew.algebra.forms.derived`): a backward tile's softmax cotangents
(`softmax_cotangents`, `derived_cotangents`), each mainloop's backward launch's, and the whole backward's
gradients walked over key or query tiles (`softmax_gradients`, `backward_declaration`, `derived_backward`).
The template's own forms are `attention_algebra`'s."""

import functools

import jax.numpy as jnp

from dew.algebra import SATURATION, derive
from dew.nn.kernels.templates import attention_algebra as algebra, attention_gradient as gradient
from dew.nn.precision import float32_einsum


def softmax_cotangents(logits, products, peak, total, anchor, delta, keep=None, seen=None, at_anchor=None,
                       *, keys_first: bool = False, accurate: bool = False):
    """A backward tile's softmax algebra (theme A's tile declaration, `derived_cotangents`): each query's
    unnormalized weights `e = exp(s - m)` from its `logits` (the hooked, scaled logits the template forms,
    natural, -inf where unseen) at its maximum `peak` (at least the least finite where it saw nothing), those
    weights over its denominator `total` (`T`, 1 where it saw nothing), and its score cotangents
    `e ((dP - c) - (D - c)) / T` about its anchor `c` (`anchor`, `D - c` its `delta`, `grad_scores`' order)
    with its products `dP = dO Vᵀ` times the dropout multipliers `keep` where the call drops, zero where the
    tile's mask `seen` says its query sees no key (`where_seen`, as `grad_scores`: an empty query's weight is
    0 while its product may overflow, `0 * inf`); the tile's queries its rows, or its columns where it holds
    its keys first (`Coords.keys_first`). The `accurate` form's (`normalized(safe=False)`): its maximum taken
    as given, and the anchor key's own `dP - c` exactly 0 where `at_anchor` marks it."""
    def per_query(x):
        return x[None, :] if keys_first else x[:, None]

    safe = peak if accurate else jnp.maximum(peak, jnp.finfo(logits.dtype).min)
    e = jnp.exp(logits - per_query(safe))
    over = per_query(1.0 / jnp.where(total == 0, 1.0, total))
    dropped = products if keep is None else products * keep
    about = dropped - per_query(anchor)
    if at_anchor is not None:
        about = jnp.where(at_anchor, 0.0, about)
    cotangents = e * ((about - per_query(delta)) * over)
    return e, e * over, gradient.where_seen(seen, cotangents, 0.0)


@functools.cache
def derived_cotangents(keys_first: bool, dropped: bool, masked: bool) -> derive.Derived:
    """The backward tile's derived softmax algebra (`softmax_cotangents`' forms, the fast class), the
    template's behind the switch (`dew.algebra.forms.derived`): each mainloop's backward launch forms its
    tile's products and logits and takes its weights and score cotangents from these forms, its mask an
    operand where the tile has one (`masked`)."""
    tile = ("keys", "rows") if keys_first else ("rows", "keys")
    axes = {"logits": tile, "products": tile, "peak": ("rows",), "total": ("rows",), "anchor": ("rows",),
            "delta": ("rows",), **({"keep": tile} if dropped else {}), **({"seen": tile} if masked else {})}
    declared = functools.partial(softmax_cotangents, keys_first=keys_first)
    return derive.derive(derive.Reduction(declared, axes, walked="keys", exponent="e"), accurate=False,
                         saturation=SATURATION)


def softmax_gradients(logits, q, k, values, dout, peak, total, anchor, delta, slope=None, seen=None,
                      keep=None, at_anchor=None, *, scale: float, accurate: bool = False):
    """Attention's gradients from its residuals, whole (theme A's backward declaration,
    `backward_declaration`): each tile's weights and score cotangents (`softmax_cotangents`) with
    `dP = dO Vᵀ`, the cotangents through the score hook by its `slope` (its derivative, `scored`) where a hook
    reads the logits, each masked where the tile's queries do not see its keys (`seen`: an empty query's
    product, or a padded query's hook derivative, need not be finite), and the queries', keys' and values'
    gradients their products at `scale` (dV from the dropped unnormalized weights against `dO / T`,
    `value_operands`' order), each product Dew's (`float32_einsum`), the `accurate` form's cotangents where
    it asks (`softmax_cotangents`). Returns dQ, dK, dV, and each tile's unnormalized weights and score
    cotangents, which the template's hook parameters and D's refinement read (`gradient.pulled`, `dq_rows`)."""
    products = float32_einsum("qd,kd->qk", dout, values, preferred_element_type=jnp.float32)
    e, _, ds = softmax_cotangents(logits, products, peak, total, anchor, delta, keep, seen, at_anchor,
                                  accurate=accurate)
    over = 1.0 / jnp.where(total == 0, 1.0, total)
    # The cotangents at the scale before their products, the reference's order: two keys of 2**127 whose
    # cotangents' quarter cancels in the product overflow it unscaled.
    scaled = (ds if slope is None else gradient.pulled(ds, (slope,), seen, ())[0]) * scale
    dropped = e if keep is None else e * keep
    return (float32_einsum("qk,kd->qd", scaled, k), float32_einsum("qk,qd->kd", scaled, q),
            float32_einsum("qk,qd->kd", dropped, dout * over[:, None]), e, ds)


@functools.cache
def derived_backward(scale: float, walked: str, hooked: bool = False, dropped: bool = False,
                     accurate: bool = False) -> derive.Derived:
    """The backward's derived forms (`backward_declaration`'s, of the fast class or `accurate`'s), the
    template's behind the switch (`dew.algebra.forms.derived`): walked over the keys for dQ, over the queries
    for dK and dV, each walked sum's product the mainloop's to issue where it takes the step
    (`Derived.step`'s `stepped`, `Accumulated`: the accurate form's keyed sums and split products)."""
    return derive.derive(backward_declaration(scale=scale, walked=walked, hooked=hooked, dropped=dropped,
                                              accurate=accurate),
                         accurate=accurate, saturation=SATURATION)


def backward_derived(how: algebra.Recompute, *, accurate: bool) -> bool:
    """Whether the backward's derived forms walk a call (behind the switch, `dew.algebra.forms.derived`):
    where its declaration states the walk. Its accurate form (the anchor test, the keyed sums, raw logits'
    recomputation) is a rounding choice the fast class's derivation does not make (§1.5's compensation rules,
    the next build): there the template's own forms run, under the replacement rule."""
    return algebra.derived_on() and not accurate and how == algebra.PLAIN


def backward_declaration(*, scale: float, walked: str, hooked: bool = False, dropped: bool = False,
                         accurate: bool = False):
    """Attention's backward as theme A's declaration (`softmax_gradients`), walked over its key tiles
    (`walked="keys"`, queries first: dQ the walk's sum, dK and dV each tile's own) or its query tiles
    (`"rows"`, keys first: dK and dV the walk's sums, dQ each tile's own), from the forward's natural
    residual rows and its D and the tile's mask, its hook's slopes an operand where a score hook reads the
    logits and its dropout multipliers where the call drops; the `accurate` form's with its anchor keys'
    mask (`at_anchor`, the tile's as `seen`)."""
    tile = ("rows", "keys")
    axes = {"logits": tile, "q": ("rows", "width"), "k": ("keys", "width"), "values": ("keys", "values"),
            "dout": ("rows", "values"), "peak": ("rows",), "total": ("rows",), "anchor": ("rows",),
            "delta": ("rows",), "seen": tile, **({"slope": tile} if hooked else {}),
            **({"keep": tile} if dropped else {}), **({"at_anchor": tile} if accurate else {})}
    return derive.Reduction(functools.partial(softmax_gradients, scale=scale, accurate=accurate), axes,
                            walked=walked, exponent="e")
