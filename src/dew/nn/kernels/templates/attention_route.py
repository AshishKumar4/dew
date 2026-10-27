"""The attention template as a production route (`dew.nn.attention`'s `implementation='template'`): `attend`,
the template's forward and backward under one `jax.custom_vjp`, on the mainloops the process's calls run on.

On a GPU the process places calls on, of sm80 or later (`generation.bf16_dot_runs`), the forward and the
backward run on the Mosaic template, Mosaic GPU's mainloop, where it runs the call on the device
(`attention_mosaic.forward_takes`, `attention_mosaic.backward_takes`: its family has a bundle there, the call
is one it takes and a config fits), else by legality on the frozen Triton mainloop (`attention_triton`, which
walks no call's runs, `attention_triton.refused`). Everywhere else, or where neither takes the call, both run
the plain-JAX mainloop (`attention_portable`). Every mainloop's training forward gives the same residuals, so
the backward runs on whichever takes it by legality (a 16-bit call's fp16 backward on Triton's, which Mosaic's
bf16 backward does not take). The route takes score hooks and the initial state. The caller
applies an output hook past the kernel, because the template's backward takes the output before it
(`templates/attention.py`'s `gradients`).
"""

import dataclasses
import functools
from typing import Literal, overload

import jax
import jax.numpy as jnp

from dew.nn.kernels import generation, selection
from dew.nn.kernels.templates import attention_mosaic, attention_portable, attention_triton
from dew.nn.kernels.templates.attention import ACCURATE, NATURAL, Gradients, Hooks, KeySpans, Split

PLAIN_TILE = attention_triton.BLOCK
"""The plain-JAX mainloop's tile: the query and key blocks a call walks, and the padding it takes to whole
blocks. It is the Triton mainloop's largest tile (`attention_triton.BLOCK`)."""

Residuals = tuple[jax.Array, jax.Array, jax.Array, Split]
"""A training forward's residuals, which both backwards read: each query's running maximum and denominator
`[B, H, Q]`, its anchor key `[B, H, Q]` and its split sums (`Split`)."""


FORMS = ("fast", "accurate")
"""The template's forms (kernel-templates.md §1, testing.md): the default, fast (the base-2 fold and
FlashAttention's arithmetic, `NATURAL`, `ACCURATE`), and the accurate form one option away (natural logits and
the accurate products), which `generation.chosen(attention_form='accurate')` names for every template call in
its block."""


def attend(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
           *, scale: float | None = None, natural: bool | None = None, accurate: bool | None = None,
           operand_dtype=None) -> jax.Array:
    """Attention of `query` `[B, Q, H, D]` over `key` and `value` `[B, K, Hk, D]` and `[B, K, Hk, Dv]`.

    Each query reads the keys its `spans` leave it, under the score hooks and initial state of `hooks`. The
    arithmetic is the template's (`templates/attention.py`'s `reference`), and the result is differentiable in
    the query, key and value and in the hooks' parameters and state. `scale` is a Python float, `D ** -0.5` by
    default. An output hook is refused: a differentiated call applies `output_mod` to the output past the
    kernel, where its own derivative runs, so the caller applies it.

    `natural` and `accurate` name its form, each where None the one `generation.chosen(attention_form=...)`
    names (`FORMS`; 'accurate' takes both), else the default's. `operand_dtype`, where the key and value
    arrive wider than the mainloops compute them (a sharding wrapper's, so its partial gradients sum wide),
    narrows the operands to it and hands back the key's and value's gradients unrounded, in their own dtype:
    the wrapper sums them and rounds once.
    """
    hooks = Hooks() if hooks is None else hooks
    if hooks.output_mod is not None:
        raise ValueError("the attention route takes score hooks and the initial state; its backward takes "
                         "the output before an output hook, so the caller applies `output_mod` to the result")
    form = generation.measured_kernel('attention_form', 'fast')
    if form not in FORMS:
        raise ValueError(f"attention_form {form!r}: the template's forms are {', '.join(FORMS)}")
    wide = form == 'accurate'
    config = _Config(scale, NATURAL or wide if natural is None else natural,
                     ACCURATE or wide if accurate is None else accurate,
                     None if operand_dtype is None else jnp.dtype(operand_dtype).name)
    return _attend(config, query, key, value, spans, hooks)


@dataclasses.dataclass(frozen=True)
class _Config:
    """A call's static configuration, which its forward and backward share."""

    scale: float | None
    natural: bool
    accurate: bool
    operand_dtype: str | None

    def narrowed(self, query: jax.Array, key: jax.Array,
                 value: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """The query, key and value at `operand_dtype` where it is given."""
        if self.operand_dtype is None:
            return query, key, value
        dtype = self.operand_dtype
        return query.astype(dtype), key.astype(dtype), value.astype(dtype)


def _on_gpu() -> bool:
    """Whether calls run on a GPU of sm80 or later, where the Triton mainloop runs (`bf16_dot_runs`)."""
    return generation.gpu_placed() and generation.bf16_dot_runs()


DROPS_NOT = ("the template's GPU mainloops draw no probability dropout yet (the Triton mainloop is frozen, "
             "and Mosaic's takes the weights hook next, ledger row 1521): a dropping call there runs the "
             "plain-JAX mainloop, so the selector leaves it to the reference path")
"""Why 'auto' leaves a dropping call on a GPU to the reference path (`drops`)."""


def drops() -> bool:
    """Whether the route's own mainloop here draws probability dropout: the plain-JAX mainloop's, where it
    runs every call (off a GPU, or on one before sm80, where the GPU mainloops do not run, `_on_gpu`). On an
    sm80 or later GPU the route's mainloops are Triton's and Mosaic's, whose legality rules refuse a dropping
    call (`attention_triton.refused`, `attention_mosaic.refused`), so it runs the plain-JAX one there
    (`DROPS_NOT`)."""
    return not _on_gpu()


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _attend(config: _Config, query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
            hooks: Hooks) -> jax.Array:
    """The forward of a call that is not differentiated, which keeps no training state."""
    return _forward(*config.narrowed(query, key, value), spans, hooks, config, residuals=False)


def _attend_forward(config: _Config, query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans,
                    hooks: Hooks) -> tuple[jax.Array, tuple]:
    """The forward of a differentiated call, which keeps the residuals its backward reads."""
    out, peak, total, anchors, split = _forward(*config.narrowed(query, key, value), spans, hooks, config,
                                                residuals=True)
    return out, (query, key, value, spans, hooks, (peak, total, anchors, split))


@overload
def _forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks,
             config: _Config, *, residuals: Literal[False]) -> jax.Array:
    ...


@overload
def _forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks,
             config: _Config, *, residuals: Literal[True]) -> tuple[jax.Array, *Residuals]:
    ...


@dataclasses.dataclass(frozen=True)
class _Pass:
    """A pass of the template as its mainloop's selector reads it (`MAINLOOP`): whether it runs on a GPU,
    why the Mosaic mainloop declines it (`attention_mosaic.forward_declined`, `backward_declined`, None where
    it takes it) and why the Triton one refuses its spans (`attention_triton.refused`)."""

    on_gpu: bool
    mosaic: str | None
    triton: str | None


_ON_GPU = selection.Requirement[_Pass]('off_gpu', lambda case: case.on_gpu, "the call runs off a GPU",
                                       generation.INHERENT)
MAINLOOP = selection.Op[_Pass]('attention_mainloop', (
    selection.Kernel[_Pass]('mosaic', (_ON_GPU, selection.Requirement[_Pass](
        'declined', lambda case: case.mosaic is None, lambda case: str(case.mosaic),
        'roadmap: the Mosaic template takes every call'))),
    selection.Kernel[_Pass]('triton', (_ON_GPU, selection.Requirement[_Pass](
        'spans', lambda case: case.triton is None, lambda case: str(case.triton), generation.INHERENT))),
    selection.Kernel[_Pass]('portable', (), generic=True),
), preferred=lambda case: 'mosaic' if case.on_gpu else 'portable')
"""The template's mainloops, the first that takes a pass: Mosaic GPU's, which a GPU prefers, the frozen
Triton one's, the plain-JAX one's, which a CPU runs. `generation.chosen(attention_mainloop=...)` names one
for every pass in its block, which runs it or refuses it by name (`selection.select`); each pass reports its
route, so the kernel-route audit sees a call on a GPU that leaves the Mosaic template, with Mosaic's own
reason."""


def _mainloop(declined: str | None, spans: KeySpans) -> str:
    """The mainloop a pass runs, reported (`MAINLOOP`), from Mosaic's reason it `declined` the pass."""
    on_gpu = _on_gpu()
    case = _Pass(on_gpu, declined, attention_triton.refused(spans) if on_gpu else None)
    return selection.select(MAINLOOP, case).kernel


def _forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks,
             config: _Config, *, residuals: bool):
    """The forward on its mainloop (`_mainloop`)."""
    form = {"scale": config.scale, "natural": config.natural}
    declined = attention_mosaic.forward_declined(query, key, value, spans, hooks, residuals=residuals,
                                                 **form) if _on_gpu() else None
    mainloop = _mainloop(declined, spans)
    if mainloop == 'mosaic':
        return attention_mosaic.forward(query, key, value, spans, hooks, residuals=residuals, **form)
    if mainloop == 'triton':
        return attention_triton.forward(query, key, value, spans, hooks, residuals=residuals, **form)
    return attention_portable.attend(query, key, value, spans, hooks, block_q=PLAIN_TILE, block_k=PLAIN_TILE,
                                     pad=PLAIN_TILE, residuals=residuals, **form)


def _attend_backward(config: _Config, saved: tuple, dout: jax.Array) -> tuple:
    """The cotangents of the query, key, value and hooks, from the residuals, each in its primal's dtype;
    the spans take none."""
    query, key, value, spans, hooks, residuals = saved
    grads = _backward(*config.narrowed(query, key, value), spans, hooks, dout, residuals, config)
    return (grads.query.astype(query.dtype), grads.key.astype(key.dtype), grads.value.astype(value.dtype),
            None, _hooks_cotangent(hooks, grads))


_attend.defvjp(_attend_forward, _attend_backward)


def _backward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks,
              dout: jax.Array, residuals: Residuals, config: _Config) -> Gradients:
    """The backward from the forward's residuals on its mainloop (`_mainloop`); its key's and value's
    gradients unrounded where the call narrowed them (`_Config.operand_dtype`)."""
    call = (query, key, value, spans, hooks, dout)
    form = {"scale": config.scale, "natural": config.natural, "accurate": config.accurate}
    rounded = config.operand_dtype is None
    declined = (attention_mosaic.backward_declined(*call, residuals=residuals, **form) if _on_gpu()
                else None)
    mainloop = _mainloop(declined, spans)
    if mainloop == 'mosaic':
        return attention_mosaic.backward(*call, residuals=residuals, rounded=rounded, **form)
    if mainloop == 'triton':
        return attention_triton.backward(*call, residuals=residuals, rounded=rounded, **form)
    return attention_portable.backward(*call, block_q=PLAIN_TILE, block_k=PLAIN_TILE, pad=PLAIN_TILE,
                                       residuals=residuals, rounded=rounded, **form)


def _hooks_cotangent(hooks: Hooks, grads: Gradients) -> Hooks:
    """The cotangent of `hooks` in its own structure: each score parameter's gradient in its `Param`, and the
    initial state's where there is one. The callables are static, and the output hook is refused.
    """
    params = tuple(dataclasses.replace(param, value=grad)
                   for param, grad in zip(hooks.score_params, grads.score_params, strict=True))
    return Hooks(score_mod=hooks.score_mod, score_params=params, initial=grads.initial)
