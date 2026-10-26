"""The attention template as a production route (`dew.nn.attention`'s `implementation='template'`): `attend`,
the template's forward and backward under one `jax.custom_vjp`, on the mainloops the process's calls run on.

On a GPU the process places calls on, of sm80 or later (`generation.bf16_dot_runs`), the forward and the
backward run on the Mosaic template, Mosaic GPU's mainloop, where it runs the call on the device
(`attention_mosaic.forward_takes`, `attention_mosaic.backward_takes`: its family has a bundle there, the call
is one it takes and a config fits), else by legality on the frozen Triton mainloop (`attention_triton`, which
walks no call's runs, `attention_triton.refused`). Everywhere else, or where neither takes the call, both run
the plain-JAX mainloop (`attention_portable`). Every mainloop's training forward gives the same residuals, so
the backward runs wherever it is fastest. The route takes score hooks and the initial state. The caller
applies an output hook past the kernel, because the template's backward takes the output before it
(`templates/attention.py`'s `gradients`).
"""

import dataclasses
import functools
from typing import Literal, overload

import jax

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention_mosaic, attention_portable, attention_triton
from dew.nn.kernels.templates.attention import Gradients, Hooks, KeySpans, Split

PLAIN_TILE = attention_triton.BLOCK
"""The plain-JAX mainloop's tile: the query and key blocks a call walks, and the padding it takes to whole
blocks. It is the Triton mainloop's largest tile (`attention_triton.BLOCK`)."""

Residuals = tuple[jax.Array, jax.Array, jax.Array, Split]
"""A training forward's residuals, which both backwards read: each query's running maximum and denominator
`[B, H, Q]`, its anchor key `[B, H, Q]` and its split sums (`Split`)."""


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
           *, scale: float | None = None) -> jax.Array:
    """Attention of `query` `[B, Q, H, D]` over `key` and `value` `[B, K, Hk, D]` and `[B, K, Hk, Dv]`.

    Each query reads the keys its `spans` leave it, under the score hooks and initial state of `hooks`. The
    arithmetic is the template's (`templates/attention.py`'s `reference`), and the result is differentiable in
    the query, key and value and in the hooks' parameters and state. `scale` is a Python float, `D ** -0.5` by
    default. An output hook is refused: a differentiated call applies `output_mod` to the output past the
    kernel, where its own derivative runs, so the caller applies it.
    """
    hooks = Hooks() if hooks is None else hooks
    if hooks.output_mod is not None:
        raise ValueError("the attention route takes score hooks and the initial state; its backward takes "
                         "the output before an output hook, so the caller applies `output_mod` to the result")
    if hooks.drop is not None and not drops():
        raise NotImplementedError(DROPS_NOT)
    return _attend(scale, query, key, value, spans, hooks)


def _on_gpu() -> bool:
    """Whether calls run on a GPU of sm80 or later, where the Triton mainloop runs (`bf16_dot_runs`)."""
    return generation.gpu_placed() and generation.bf16_dot_runs()


DROPS_NOT = ("the template's GPU mainloops take no probability dropout yet: the Triton mainloop is frozen, "
             "and Mosaic's takes the weights hook next (ledger row 1521)")
"""Why the route refuses a dropping call on a GPU (`drops`)."""


def drops() -> bool:
    """Whether the route's mainloops here take probability dropout: the plain-JAX mainloop's, where it runs
    every call (off a GPU, or on one before sm80, where the GPU mainloops do not run, `_on_gpu`). On an sm80
    or later GPU the route's mainloops are Triton's and Mosaic's, which take none yet (`DROPS_NOT`)."""
    return not _on_gpu()


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _attend(scale: float | None, query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
            hooks: Hooks) -> jax.Array:
    """The forward of a call that is not differentiated, which keeps no training state."""
    return _forward(query, key, value, spans, hooks, scale, residuals=False)


def _attend_forward(scale: float | None, query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                    hooks: Hooks) -> tuple[jax.Array, tuple]:
    """The forward of a differentiated call, which keeps the residuals its backward reads."""
    out, peak, total, anchors, split = _forward(query, key, value, spans, hooks, scale, residuals=True)
    return out, (query, key, value, spans, hooks, (peak, total, anchors, split))


@overload
def _forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks,
             scale: float | None, *, residuals: Literal[False]) -> jax.Array:
    ...


@overload
def _forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks,
             scale: float | None, *, residuals: Literal[True]) -> tuple[jax.Array, *Residuals]:
    ...


def _forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks,
             scale: float | None, *, residuals: bool):
    """The forward on the first mainloop that takes the call here: Mosaic GPU's (the template), Triton's,
    the plain-JAX one's."""
    if _on_gpu() and attention_mosaic.forward_takes(query, key, value, spans, hooks, scale=scale,
                                                    residuals=residuals):
        return attention_mosaic.forward(query, key, value, spans, hooks, scale=scale, residuals=residuals)
    if _on_gpu() and attention_triton.refused(spans) is None:
        return attention_triton.forward(query, key, value, spans, hooks, scale=scale, residuals=residuals)
    return attention_portable.attend(query, key, value, spans, hooks, block_q=PLAIN_TILE, block_k=PLAIN_TILE,
                                     pad=PLAIN_TILE, scale=scale, residuals=residuals)


def _attend_backward(scale: float | None, saved: tuple, dout: jax.Array) -> tuple:
    """The cotangents of the query, key, value and hooks, from the residuals; the spans take none."""
    query, key, value, spans, hooks, residuals = saved
    grads = _backward(query, key, value, spans, hooks, dout, residuals, scale)
    return grads.query, grads.key, grads.value, None, _hooks_cotangent(hooks, grads)


_attend.defvjp(_attend_forward, _attend_backward)


def _backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks,
              dout: jax.Array, residuals: Residuals, scale: float | None) -> Gradients:
    """The backward from the forward's residuals on the first mainloop that takes the call here: Mosaic
    GPU's, Triton's, the plain-JAX one's."""
    call = (query, key, value, spans, hooks, dout)
    if _on_gpu() and attention_mosaic.backward_takes(*call, residuals=residuals, scale=scale):
        return attention_mosaic.backward(*call, residuals=residuals, scale=scale)
    if _on_gpu() and attention_triton.refused(spans) is None:
        return attention_triton.backward(*call, residuals=residuals, scale=scale)
    return attention_portable.backward(*call, block_q=PLAIN_TILE, block_k=PLAIN_TILE, pad=PLAIN_TILE,
                                       residuals=residuals, scale=scale)


def _hooks_cotangent(hooks: Hooks, grads: Gradients) -> Hooks:
    """The cotangent of `hooks` in its own structure: each score parameter's gradient in its `Param`, and the
    initial state's where there is one. The callables are static, and the output hook is refused.
    """
    params = tuple(dataclasses.replace(param, value=grad)
                   for param, grad in zip(hooks.score_params, grads.score_params, strict=True))
    return Hooks(score_mod=hooks.score_mod, score_params=params, initial=grads.initial)
