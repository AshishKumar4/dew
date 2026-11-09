"""A quantized model's attention on the template (row 4858).

Under a Qwix rule over activations (`dew.quantization`), the reference path's two products quantize each
operand by its channels' largest over the contraction (qwix/_src/core/dot_general_qt.py's `dot_general_qt`,
whose fast path sums the int8 products exactly and scales them by both operands' scales): the logits' query
and keys by each row, the weights by each row and the values by each column. Here the logits' operands
enter the template as their codes, exact in their dtype, and their scales a score hook on the products
(`Scaled`); the weights round at each row's maximum in the anchored walk (`Hooks.rounds`); the values enter
as their codes, their columns' scales over the bound taking the normalized output. The backward is Qwix's
straight-through one (its original residuals, `dot_general_qt_bwd`): the gradients of the unquantized
products at the quantized logits and weights, dense, as the reference path's.
"""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp

from dew._cache import unintercepted
from dew.nn.derivatives import KernelGlue, contracted, declared_derivative
from dew.nn.kernels import tuning
from dew.nn.kernels.templates import attention_route
from dew.nn.kernels.templates.attention import Hooks, KeySpans, Kind, Param, dense, score_modded
from dew.nn.precision import float32_einsum, products_quantized


def owner():
    """The Qwix rule that owns this call's products, a quantized model's matching `dot_general` at the
    calling module (`precision.products_quantized`, `qwix.pallas.get_current_rule`), or None."""
    if not products_quantized():
        return None
    from qwix import pallas as qwix_pallas

    return qwix_pallas.get_current_rule("dot_general")


def whole(rule) -> bool:
    """Whether `rule`'s products of two activations stay whole: a rule that quantizes no activation (Qwix's
    weight-only one, `act_qtype` None) leaves attention's products, which multiply no weight, as they are
    (qwix/_src/providers/qt.py's `_create_dot_general_qt_config`)."""
    return rule.act_qtype is None or rule.weight_qtype is None


def takes(rule) -> bool:
    """Whether this module computes `rule`'s attention: activations quantized to int8 by Qwix's dynamic
    absmax per channel (no static scale, no tile of the contraction), as Dew's serving rule
    (`qwix.QuantizationRule`) and its training rule (`qwix.QtRule`, `dew.quantization`'s) state it; a training
    rule's gradients unquantized too, so its backward is the straight-through one over the original residuals
    (`use_original_residuals`), no channel's axis disabled and no configuration of Qwix's own over it
    (`additional_qt_config`, which may replace any of these: qwix/_src/providers/qt.py's
    `_create_dot_general_qt_config`)."""
    from qwix import QtRule

    forward = (jnp.dtype(rule.act_qtype) == jnp.dtype(jnp.int8)
               and rule.act_calibration_method in (None, "absmax") and not rule.act_static_scale
               and rule.tile_size is None)
    if not isinstance(rule, QtRule):
        return forward
    return (forward and rule.bwd_qtype is None and not rule.disable_channelwise_axes
            and not rule.additional_qt_config)


BOUND = 127.5
"""Qwix's symmetric bound of int8 (`qwix/_src/core/numerics.py`'s `get_symmetric_bound`): an operand's
scale is its channel's largest over it, so its codes round to [-128, 127]."""


def codes(x: jax.Array, axis: int) -> tuple[jax.Array, jax.Array]:
    """`x`'s int8 codes in its own dtype, which holds them exactly, and their fp32 scales, by the largest
    magnitude along `axis` (kept), as Qwix's absmax calibration takes an operand's channels; a channel of
    nothing but zeros (below fp32's smallest normal's root) at scale 1, as Qwix's."""
    wide = x.astype(jnp.float32)
    scale = jnp.max(jnp.abs(wide), axis=axis, keepdims=True) / BOUND
    scale = jnp.where(scale < jnp.sqrt(jnp.finfo(jnp.float32).tiny), 1.0, scale)
    return jnp.clip(jnp.round(wide / scale), -128, 127).astype(x.dtype), scale


@dataclasses.dataclass(frozen=True)
class Scaled(tuning.FieldHook):
    """A quantized model's logits as a score hook: the products of the query's and keys' codes times each
    query's and key's scale (a `Kind.QUERY` and a `Kind.KEY` parameter), then `score_mod` and its own
    parameters (a softcap's, a bias's) on them: equal for equal inner hooks (`tuning.FieldHook`)."""

    score_mod: Callable | None = None

    def __call__(self, logits: jax.Array, q_pos: jax.Array, k_pos: jax.Array, query_scale: jax.Array,
                 key_scale: jax.Array, *params: jax.Array) -> jax.Array:
        scaled = logits * query_scale * key_scale
        return scaled if self.score_mod is None else self.score_mod(scaled, q_pos, k_pos, *params)


def _operands(query: jax.Array, key: jax.Array, value: jax.Array) -> tuple:
    """The call's operands as the template takes them under the rule: the query scaled by the inverse root of
    its width in its dtype, as the reference path scales it (`reference_attention`), and its codes, the
    keys' and the values' (`codes`), with the query's and keys' scales `[B, H, Q]` and `[B, H, K]` at
    their query heads and the values' columns' over the bound `[B, 1, H, Dv]`."""
    heads, width = query.shape[2:]
    group = heads // key.shape[2]
    scaled = query / jnp.sqrt(width).astype(query.dtype)
    query_codes, query_scale = codes(scaled, -1)
    key_codes, key_scale = codes(key, -1)
    value_codes, value_scale = codes(value, 1)
    by_head = jnp.repeat(jnp.swapaxes(key_scale[..., 0], 1, 2), group, axis=1)
    columns = jnp.repeat(value_scale, group, axis=2) / BOUND
    return (query_codes, key_codes, value_codes, jnp.swapaxes(query_scale[..., 0], 1, 2), by_head, columns)


def _attending(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
               hooks: Hooks) -> jax.Array:
    """The forward: the codes on the template's anchored walk (`attention_route.rounded`), the weights
    rounded at each row's maximum, the output taken by the values' columns' scales."""
    query_codes, key_codes, value_codes, query_scale, key_scale, columns = _operands(query, key, value)
    coded = Hooks(Scaled(hooks.score_mod),
                  (Param(query_scale, Kind.QUERY), Param(key_scale, Kind.KEY), *hooks.score_params),
                  rounds=BOUND)
    out = attention_route.rounded(query_codes, key_codes, value_codes, spans, coded)
    return (out.astype(jnp.float32) * columns).astype(query.dtype)


def _straight_through(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                      hooks: Hooks) -> jax.Array:
    """The call densely as Qwix's straight-through rule over its original residuals differentiates it, whose
    derivative alone the backward takes (`_attend_backward`): the logits the whole product plus its
    quantized one's difference from it, held constant, so the weights are the quantized logits'; the
    values' product the whole one, its weights' and values' gradients each the other's original operand."""
    batch, queries, heads, width = query.shape
    kv_heads, keys = key.shape[2], spans.keys
    query_codes, key_codes, _, query_scale, key_scale, _ = _operands(query, key, value)
    scaled = query / jnp.sqrt(width).astype(query.dtype)

    def products(left: jax.Array, right: jax.Array) -> jax.Array:
        grouped = left.reshape(batch, queries, kv_heads, heads // kv_heads, width)
        return float32_einsum("bqhgd,bkhd->bhgqk", grouped, right,
                              preferred_element_type=jnp.float32).reshape(batch, heads, queries, keys)

    whole = products(scaled, key)
    quantized = products(query_codes, key_codes) * query_scale[..., None] * key_scale[:, :, None, :]
    logits = whole + jax.lax.stop_gradient(quantized - whole)
    logits = jnp.where(dense(spans)[:, None], score_modded(logits, hooks), -jnp.inf)
    peak = jnp.max(logits, axis=-1, keepdims=True)
    weights = jnp.exp(logits - jnp.where(jnp.isfinite(peak), peak, 0.0))
    total = jnp.sum(weights, axis=-1, keepdims=True)

    def valued(left: jax.Array, right: jax.Array) -> jax.Array:
        grouped = left.reshape(batch, kv_heads, heads // kv_heads, queries, keys)
        return float32_einsum("bhgqk,bkhd->bhgqd", grouped, right.astype(jnp.float32),
                              preferred_element_type=jnp.float32).reshape(batch, heads, queries, -1)

    seen = jnp.where(total > 0, total, 1.0)
    return jnp.swapaxes(valued(weights / seen, value), 1, 2).astype(query.dtype)


def _attend_forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                    hooks: Hooks) -> tuple[jax.Array, tuple]:
    """The forward, its residuals the call's own operands, spans and hooks."""
    return _attending(query, key, value, spans, hooks), (query, key, value, spans, hooks)


def _attend_backward(saved: tuple, dout: jax.Array) -> tuple:
    """The straight-through gradients (`_straight_through`) of the query, key, value and score parameters;
    the spans take none. Traced where the derivative is taken, with any quantization provider's interception
    off too (`attend`'s)."""

    def pulled(saved, dout):
        query, key, value, spans, hooks = saved
        _, pullback = jax.vjp(lambda q, k, v, h: _straight_through(q, k, v, spans, h), query, key, value,
                              hooks)
        d_query, d_key, d_value, d_hooks = pullback(dout)
        return d_query, d_key, d_value, None, d_hooks

    return unintercepted(pulled)(saved, dout)


_attend = contracted("test_a_quantized_attention_holds_its_reverse_orders")(
    declared_derivative(KernelGlue(), _attending, _attend_forward, _attend_backward))
"""The template's forward over the codes (`_attending`), Qwix's straight-through backward
(`_straight_through`)."""


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks) -> jax.Array:
    """A quantized model's attention of `query` `[B, Q, H, D]` over `key` and `value` `[B, K, Hk, D]` and
    `[B, K, Hk, Dv]` under the visibility `spans` and score `hooks` (no initial state, no draw), forward and
    backward traced with any quantization provider's interception off (`unintercepted`): the template forms
    the rule's codes and scales itself, its walk's products none of the rule's, under Dew's wrapper and a
    provider built by hand alike."""
    return unintercepted(_attend)(query, key, value, spans, hooks)
