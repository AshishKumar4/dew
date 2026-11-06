"""The attention template's Mosaic GPU mainloop on the wgmma family
(dew.nn.kernels.templates.attention_mosaic): its forward and its backward in both forms, check 1
(`reference_error.assert_as_exact`) against the template's reference over ordinary draws, under Mosaic GPU's
interpreter on the CPU (as an H100's launches, `generation.assumed`) and compiled on an H100. The interpreter
is coverage, not proof (it treats wgmma_wait as a no-op and takes the register operand's product as a dot);
the GPU gate runs the compiled kernels. Its backward reads its own training forward's residuals (its split
sums), and the Triton mainloop's, as the route feeds them. Its configs are its Template implementations'
(`MosaicForward`, `MosaicBackward`): generated from the family's facts and the call's extents and fitted to
the device's, a site pinning one where it names a schedule (`tuning.pinned`)."""

import contextlib
import dataclasses
import functools
import gc
import weakref
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import attention_terms
from float64_terms import float64_terms
from generation_support import COMPILES_ON_CPU, INTERPRETS, lowers
from jax.sharding import Mesh
from reference_error import as_given, assert_as_exact, widened
from test_attention_template import CASES, ROWS, RUN_CASES, arguments, geometry as core_geometry
from test_attention_triton import BACKWARD_SEEDS, HOOKS, SCORE_HOOKS, gate_of, hooks_for, score_hooks

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_mosaic_forward,
    attention_mosaic_program,
    attention_portable,
    attention_triton,
    resources,
)

pytestmark = pytest.mark.kernel

ON_HOPPER = pytest.mark.skipif(generation.device_generation() != "sm90",
                               reason="runs the sm90 mainloop compiled, on an H100")
CALLS = {"full": {}, "causal": {"causal": True}}
"""The call forms the mainloop takes."""
SHAPES = {"interpreted": (1, 160, 2, 1), "compiled": (2, 400, 4, 2)}
"""Rows, tokens (not whole tiles, so the call pads, over more than one key tile past the keys where a
program's two consumers pad it to 128), query and key heads (grouped), interpreted on the CPU and compiled
on an H100."""
SEEDS = 8
"""Draws a case takes, check 1's fewest (`reference_error.seeds_for`): it bounds the worst row over them
against the reference's own."""
ALL_HOOKS_SEEDS = 32
"""The draws of a site with every hook at once (`test_attention_triton.ALL_HOOKS_SEEDS`)."""
BACKWARD_HOOKS = list(SCORE_HOOKS)
"""The backward's score hooks: every one the backward differentiates (`test_attention_triton.SCORE_HOOKS`),
the hook indexed off the query's coordinates' column, which forms a unit axis, among them (ledger 1428)."""
FORWARD_HOOKS = {**{name: (SCORE_HOOKS[name], False) for name in BACKWARD_HOOKS},
                 "gate": ((), True),
                 "all": (("softcap", "custom", "sinks", "alibi", "temperature", "key", "tile"), True)}
"""The forward's hooks: each score hook the backward differentiates (`test_attention_triton.SCORE_HOOKS`:
every kind of parameter, the initial state as sinks or a resumed partial, a hook through a custom VJP), an
output gate, and all at once; each `(score hooks, gate)`."""
HOOK_FORMS = {"bfloat16": (jnp.bfloat16, False), "float16": (jnp.float16, False),
              "bfloat16-natural": (jnp.bfloat16, True), "float16-natural": (jnp.float16, True)}
"""The hooked forward's dtypes and forms: the fold and natural logits over both 16-bit dtypes."""
FED = {"dkv_family": "wgmma.fed", "dkv_consumers": 2}
"""The dK/dV launch on an H100's layout whose consumers feed their rings, its two consumers splitting each
slot's queries (`attention_mosaic_program.split_queries`), as the derivation takes it."""
RAISED_FORMS = ("fused-shared-dq", "fused-producer-dq")
"""The backward's forms whose consumers hold more registers than a warpgroup's even share, legal only where
the GPU's compile honors their raise (`resources.RAISES_HONORED`)."""
BACKWARD_FORMS = ("default", "fused", "fused-keys-first", "fused-shared-keys",
                  *(RAISED_FORMS if resources.RAISES_HONORED else ()), "accurate")
"""The backward's forms (`_backward_form`): the fast in two launches and in one each way, the raised ones
where raises hold (`RAISED_FORMS`), and the accurate."""
HOOKED_FORMS = ("default", "accurate")
"""The forms every hooked call takes: its fast form's one launch takes a hook of no parameters, an initial
state or a draw alone."""


FORM_WIDTHS = {"fused-keys-first": 64, "fused-shared-keys": 128, "fused-shared-dq": 128,
               "fused-producer-dq": 128}
"""The head width a backward form takes where the checks' own (128, a call form's 64) is not one it holds:
the keys-first one launch holds width 64 alone (its dK, dV and dQ tiles past a consumer's registers at
128), the shared keys' consumers 128 and wider (each its share of the columns whole swizzled rows)."""


def _backward_form(form: str) -> tuple[bool, attention_mosaic.BackwardTiles | None]:
    """The backward form `form` names, `(accurate, config)`: two launches, the dQ launch one consumer's
    beside a producer and the keys-first dK/dV launch two consumers sharing a key block (`FED`), the
    accurate form's at its derivation's tiles (None) or its sums in two column halves where it is
    `accurate-halves` (a width split's, `BackwardTiles.columns`); or the fast form's one launch (`fused`),
    queries first on the dQ launch's layout, or keys first on `FED`'s (`fused-keys-first`), each consumer
    64 of a slot's 128 queries, or queries first over keys its two consumers share on `FED`'s layout
    (`fused-shared-keys`), or keys first on the dQ launch's layout, each of its two consumers its own keys,
    their score cotangents exchanged for a shared dQ (`fused-shared-dq`, FlashAttention 3's width 128), its
    producer adding it (`fused-producer-dq`)."""
    if form == "fused":
        return False, attention_mosaic.BackwardTiles(64, 64, 2, 1, fused=True)
    if form == "fused-shared-keys":
        return False, attention_mosaic.BackwardTiles(64, 64, 2, 2, fused=True, family=FED["dkv_family"],
                                                     shared_keys=True)
    if form in ("fused-shared-dq", "fused-producer-dq"):
        return False, attention_mosaic.BackwardTiles(64, 64, 2, 1, fused=True, dkv_family="wgmma",
                                                     dkv_consumers=2, shared_dq=True, staged_logits=True,
                                                     producer_dq=form == "fused-producer-dq")
    if form == "fused-keys-first":
        return False, attention_mosaic.BackwardTiles(128, 64, 2, 1, fused=True, **FED)
    if form == "accurate":
        # Its derivation's: two launches, its dK/dV programs' column shares its dQ's two sums' choice.
        return True, None
    columns = 2 if form == "accurate-halves" else 1
    return form != "default", attention_mosaic.BackwardTiles(64, 64, 2, 1, columns=columns, **FED)


FORM_TILES = attention_mosaic.Tiles(64, 64, 2, 2)
"""The call forms' tiles: the template core's 100 queries over 96 keys (`test_attention_template.CASES`) in a
pair of query blocks over two key tiles, so a block's spans begin and end within the keys it streams."""
HEAD_WIDTHS = ((64, 128), (128, 64), (80, 48))
"""The (query, value) head widths the forward and the backward are checked at: a value wider than its
query, a narrower one, and both off a power of two (80 and 48 pad to 128 and 64, `attention.padded_call`)."""


def _drawn(shape: tuple[int, int, int, int], dtype, width: int = 128, keys: int | None = None,
           value_width: int | None = None):
    """A draw `(query, key, value, dout)` of unit Gaussians at `shape` (its keys `keys`, else its tokens) in
    `dtype`, by seed: the query and key `width` wide, the value and its cotangent `value_width` (`width` where
    None)."""
    rows, tokens, heads, kv_heads = shape
    keys = tokens if keys is None else keys
    value_width = width if value_width is None else value_width

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 4)
        dims = ((heads, tokens, width), (kv_heads, keys, width), (kv_heads, keys, value_width),
                (heads, tokens, value_width))
        return tuple(jax.random.normal(stream, (rows, count, heads_, depth), jnp.float32).astype(dtype)
                     for stream, (heads_, count, depth) in zip(streams, dims, strict=True))

    return draw


def _device(interpret: bool, target: generation.GpuTarget | None = None):
    """The device a site's launches are sized and run for: interpreted on the CPU, which has no instruction
    family of its own, `target`'s facts (an H100's where None, `generation.assumed`); the process's own
    compiled."""
    return generation.assumed(target or "sm90") if interpret else contextlib.nullcontext()


def _pinned(forward: attention_mosaic.Tiles | None = None,
            backward: attention_mosaic.BackwardTiles | None = None):
    """The launches pinned to `forward`'s and `backward`'s configs where given (`tuning.pinned`), each
    else at the cost model's pick."""
    pins = {attention_mosaic.MosaicForward: forward, attention_mosaic.MosaicBackward: backward}
    return tuning.pinned({op: config for op, config in pins.items() if config is not None})


def _check_forward(call: str, dtype, shape, *, natural: bool, interpret: bool, width: int = 128,
                   value_width: int = 128, tiles: attention_mosaic.Tiles | None = None,
                   training: template.Kept = template.Kept.NONE, target: generation.GpuTarget | None = None,
                   drop: bool = False, query_scale: float = 1.0, sinks: bool = False):
    """The forward, or with `training` (what it keeps, `template.Kept`) the training forward's output, over
    the call form `call` of `shape`,
    its query and key heads `width` wide and its value heads `value_width`, at `tiles` (the default when
    None), with `drop` under probability dropout at `DROP_RATE` (`_dropping`), with `sinks` a sink a query
    head (`template.sinks`, the initial state), its queries drawn times `query_scale` (a power of two,
    exact)."""
    spans = template.spans(shape[1], shape[1], shape[0], **CALLS[call])
    hooks = _dropping(shape) if drop else None
    if sinks:
        hooks = template.Hooks(initial=template.sinks(jnp.linspace(-1.0, 1.0, shape[2]), shape[0], shape[1],
                                                      value_width))
    unscaled = _drawn(shape, dtype, width=width, value_width=value_width)

    def draw(seed):
        query, *rest = unscaled(seed)
        return (query * jnp.asarray(query_scale, query.dtype), *rest)

    def ours(query, key, value, dout):
        del dout
        with _device(interpret, target), _pinned(tiles):
            out = attention_mosaic.forward(query, key, value, spans, hooks, natural=natural,
                                           residuals=training, interpret=interpret)
        return out if training is template.Kept.NONE else out[0]

    def reference(query, key, value, dout):
        del dout
        return template.reference(query, key, value, spans, hooks)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"mosaic {training.name.lower()} forward {call} {jnp.dtype(dtype).name} "
                    f"natural {natural} widths {width} {value_width} drop {drop}", seeds=SEEDS)


DROP_RATE = 0.25
"""The probability a dropping check drops an entry at."""


def _dropping(shape) -> template.Hooks:
    """Hooks of probability dropout alone at `DROP_RATE` for a call of `shape` `[B, Q, H, D]`, each batch row
    and query head its own global index, the seed fixed."""
    return template.Hooks(drop=template.Dropout(
        jax.random.bits(jax.random.PRNGKey(7), (2,), jnp.uint32), jnp.arange(shape[0], dtype=jnp.int32),
        jnp.arange(shape[2], dtype=jnp.int32), DROP_RATE))


def _mosaic_backward_takes(spans: template.KeySpans, query, key, hooks: template.Hooks | None, *,
                           accurate: bool) -> None:
    """Skips the check where the Mosaic backward declines a call of `query`'s and `key`'s shapes and dtype
    under `hooks` on the target in effect: a call it declines for its registers (`MosaicBackward.declined`'s
    resources) is the route's next mainloop's, whose own suites check it; it declines no call it takes
    otherwise."""
    query, key = (jnp.zeros(array.shape, array.dtype) for array in (query, key))
    reason = attention_mosaic.backward_declined(query, key, key, spans, hooks, query,
                                                residuals=_residuals(query.shape), accurate=accurate)
    assert reason is None or reason.code == "resources", reason
    if reason is not None:
        pytest.skip(f"the Mosaic backward declines it: {reason}")


def _check_spans(label: str, spans: template.KeySpans, shape, *, interpret: bool, backward: bool = False,
                 form: str = "default", score: tuple = (), gate: bool = False, dtype=jnp.bfloat16,
                 natural: bool = False, seeds: int | None = None, width: int = 64,
                 param_rows: int | None = None):
    """The forward, or the backward in the form `form` names (`_backward_form`), over `spans` at `FORM_TILES`
    (64 wide) under the score hooks `score` (`test_attention_triton.score_hooks`, their parameters and
    initial state drawn with the operands) and with `gate` an output gate (`hooks_for`), draws of `shape`
    (rows, queries, query and key heads, four and two under a hook) over the spans' keys in `dtype`, heads
    `width` wide, the forward's logits `natural` or folded. The backward differentiates the queries, keys,
    values, the hook's parameters and the initial state, over `seeds` draws where given (the site's own,
    else the template site's)."""
    rows, queries, heads, _ = shape
    # A hook's parameters drawn for `param_rows` batch rows where given (one every row shares).
    score_mod, kinds, draw_hooks = score_hooks(score, queries, spans.keys, width,
                                               rows=rows if param_rows is None else param_rows, heads=heads)
    output = hooks_for(HOOKS["gate" if gate else "none"], queries, width, gate_of(queries, width))
    draw_operands = _drawn(shape, dtype, width=width, keys=spans.keys)

    def draw(seed):
        return *draw_operands(seed), *draw_hooks(seed)

    def hooked(params, initial):
        return template.Hooks(score_mod, tuple(template.Param(value, kind)
                                               for value, kind in zip(params, kinds, strict=True)), initial,
                              output.output_mod, output.output_params)

    accurate, config = _backward_form(form)
    if backward:
        query, key = (jax.ShapeDtypeStruct((rows, queries, count, width), dtype) for count in shape[2:])
        with _device(interpret):
            _mosaic_backward_takes(spans, query, key, hooked(*draw_hooks(0)), accurate=accurate)

    def ours(query, key, value, dout, params, initial):
        hooks = hooked(params, initial)
        # The forward at the call forms' tiles 64 wide, else at its derivation's (two consumers' hooked sums
        # 128 wide pass their registers, and a hooked training forward's natural walk theirs 64 wide).
        with _device(interpret), _pinned(FORM_TILES if width == 64 and not (backward and score) else None,
                                         config):
            if not backward:
                return attention_mosaic.forward(query, key, value, spans, hooks, natural=natural,
                                                interpret=interpret)
            # The backward from this mainloop's own training forward's residuals.
            residuals = attention_mosaic.forward(query, key, value, spans, hooks, natural=natural,
                                                 residuals=template.Kept.SPLIT, interpret=interpret)[1:]
            grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                              accurate=accurate, interpret=interpret)
        return grads.query, grads.key, grads.value, grads.score_params, grads.initial

    def reference(query, key, value, dout, params, initial):
        if not backward:
            return template.reference(query, key, value, spans, hooked(params, initial))

        def attend(query, key, value, params, initial):
            return template.reference(query, key, value, spans, hooked(params, initial))

        _, pull = jax.vjp(attend, query, key, value, params, initial)
        return pull(dout)

    name = (f"mosaic {'backward' if backward else 'forward'} {label} {score}{' gate' if gate else ''} "
            f"{jnp.dtype(dtype).name}{f' {form}' if backward else ''}{' natural' if natural else ''}")
    if seeds is None:
        seeds = (BACKWARD_SEEDS.get((score, jnp.dtype(dtype).name), SEEDS) if backward
                 else ALL_HOOKS_SEEDS if len(score) + gate > 1 else SEEDS)
    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, name, seeds=seeds,
                    terms=_gradient_terms(spans, hooked, score, jax.jit(reference)) if backward else None)


def _check_hooks(hooks: str, dtype, natural: bool, *, interpret: bool):
    """The forward under the hooks `FORWARD_HOOKS[hooks]` over the template core's causal packed documents
    under a window (`test_attention_triton`'s hook site): a single hook over both rows, so each kind of
    parameter is read at a second row; several at once interpreted over the first row alone (its three
    documents and its padding: their 32 draws over a second row took the interpreter past the CPU suite's
    120 s target, row 2196), compiled over both."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    score, gate = FORWARD_HOOKS[hooks]
    rows = 1 if interpret and len(score) + gate > 1 else ROWS
    called = {name: value[:rows] if hasattr(value, "shape") else value
              for name, value in arguments(case).items()}
    spans = template.spans(queries, keys, rows, **called)
    _check_spans("causal-documents-window", spans, (rows, queries, 4, 2), interpret=interpret, score=score,
                 gate=gate, dtype=dtype, natural=natural)


INTERPRETED_HOOK_SITES = {"none": ((1, 2, 1), 8), "alibi": ((1, 2, 1), 128), "key-bias": ((1, 2, 1), 16),
                          "custom-vjp": ((1, 2, 1), 8), "softcap": ((1, 2, 1), 8), "sinks": ((1, 4, 2), 16),
                          "temperature": ((1, 2, 1), 8), "tile-bias": ((1, 2, 1), 8),
                          "resumed": ((1, 2, 1), 128), "softcap-sinks-alibi": ((2, 2, 1), 128),
                          "row-indexed": ((1, 2, 1), 8)}
"""The interpreted hooked backward's site under each hook: its rows, query heads and key heads, and its
seeds, sized by `seeds_for` at that shape (the reference against itself), never the compiled site's. The
claim interpreted is the hooked backward lowering and computing the template's gradients, so each site
keeps its paths: the first row's three documents and its padding, their boundaries inside key blocks, two
key blocks a query, more query heads than key heads, and the hook's every mode. Of the shapes that keep
them, each takes the fewest seeds times the interpreter's seconds a seed (a row of two heads 1.9 s, of four
3.8 s, two rows of four 7.5 s): a head's ALiBi slope or sink is one value a draw, so those hooks' seeds
grow as heads shrink. The H100 site (`_check_hooked_backward`, compiled) keeps the template site's shape
and seeds, at full resolution."""


def _check_hooked_backward(hooks: str, *, form: str, interpret: bool, width: int = 64, shared: bool = False):
    """The backward in the form `form` names (`_backward_form`) under the score hook `SCORE_HOOKS[hooks]` over
    the template core's causal packed documents under a window (`_check_hooks`' site), heads `width` wide:
    the gradients of the queries, keys, values, of the hook's parameters and of the initial state.
    Interpreted, at the hook's own site (`INTERPRETED_HOOK_SITES`); with `shared` over two rows, the hook's
    parameters one every row shares."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    (rows, heads, kv_heads), seeds = INTERPRETED_HOOK_SITES[hooks] if interpret else ((ROWS, 4, 2), None)
    rows = 2 if shared else rows
    # The call's documents of its first `rows` rows.
    called = {name: value[:rows] if hasattr(value, "shape") else value
              for name, value in arguments(case).items()}
    spans = template.spans(queries, keys, rows, **called)
    _check_spans("causal-documents-window", spans, (rows, queries, heads, kv_heads), interpret=interpret,
                 backward=True, form=form, score=SCORE_HOOKS[hooks], seeds=seeds, width=width,
                 param_rows=1 if shared else None)


def _check_call_form(case: str, *, interpret: bool, backward: bool = False, form: str = "default"):
    """`_check_spans` over the template core's call form `case` (`test_attention_template.CASES`, or its
    calls over runs, `RUN_CASES`), two query heads a key head, the backward in the form `form` names: over
    one key head interpreted (a second took the interpreter's accurate backward 7 s a draw more, past the
    CPU suite's 120 s target at 8 draws, row 2196; the hooked sinks site keeps two), two compiled."""
    call = CASES[case] if case in CASES else RUN_CASES[case]
    queries, keys = core_geometry(call)
    spans = template.spans(queries, keys, ROWS, **arguments(call))
    heads = (2, 1) if interpret else (4, 2)
    _check_spans(case, spans, (ROWS, queries, *heads), interpret=interpret, backward=backward, form=form,
                 width=FORM_WIDTHS.get(form, 64))


def _check_backward(call: str, shape, *, form: str | None, interpret: bool, keys: int | None = None,
                    residuals_from=None, target: generation.GpuTarget | None = None,
                    config: attention_mosaic.BackwardTiles | None = None, width: int = 128,
                    value_width: int | None = None, spans: template.KeySpans | None = None,
                    drop: bool = False, kept: template.Kept = template.Kept.SPLIT, dtype=jnp.bfloat16):
    """The backward in the form `form` names (`_backward_form`; None, the default form at `config`, else its
    derived one) over the call form `call` of `shape` (or `spans` where given; its keys `keys`, its query
    heads `width` wide and its value heads `value_width`) on `target` (`_device`), from the training
    forward's residuals at its pick: this mainloop's, those `kept` names (its anchor's split sums, or
    `template.Kept.OUTPUT`'s output, check 1 then judging D's rounding a term at a time), or
    `residuals_from`'s (another mainloop's `forward`, as the route feeds the Triton mainloop's), with `drop`
    under probability dropout (`_dropping`), on operands of `dtype`."""
    keys = shape[1] if keys is None else keys
    spans = template.spans(shape[1], keys, shape[0], **CALLS[call]) if spans is None else spans
    hooks = _dropping(shape) if drop else None
    draw = _drawn(shape, dtype, width, keys=keys, value_width=value_width)
    accurate, config = (False, config) if form is None else _backward_form(form)
    # Another mainloop's forward keeps its split sums (its `residuals` True).
    trained = (functools.partial(attention_mosaic.forward, residuals=kept) if residuals_from is None
               else functools.partial(residuals_from, residuals=True))

    def ours(query, key, value, dout):
        with _device(interpret, target):
            residuals = trained(query, key, value, spans, hooks, interpret=interpret)[1:]
            with _pinned(backward=config):
                grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                                  accurate=accurate, interpret=interpret)
        return grads.query, grads.key, grads.value

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, hooks), query, key, value)
        return pull(dout)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, f"mosaic backward {call} {form} {kept.name}",
                    seeds=SEEDS, terms=_gradient_terms(spans))


def _gradient_terms(spans: template.KeySpans, hooked: Callable | None = None, score: tuple = (),
                    reference: Callable | None = None):
    """The float64 terms of the query's, key's and value's gradients over `spans` (`attention_terms`): a
    logit's gradient `P (dP - D)` sized by `P (|dP| + sum |out dout|)`, the rounding a D formed from the
    rounded output carries (`template.Kept.OUTPUT`'s, and the fast form's: its anchor key's `dP - c` its
    product's rounding), so a row that cancels to zero in float64 is held to it, not to its own zero. Under
    score hooks (`hooked(params, initial)`, a draw's `(query, key, value, dout, params, initial)`), the
    logits are the hook's (`template.score_modded`), and its parameters' gradients take the terms check 1
    traces from `reference` (`float64_terms`, its static runs compiled once for every draw), as do the
    initial state's. Sinks are a head's value-free logit (`template.sinks`), which `attention_terms` joins to
    each denominator: a query of one key whose sink weighs 3e-8 beside it (ALiBi's slope lifting its logit
    to 17.7) cancels to its sink's share, and the fast form's anchor rounding, its terms' size, is 51 times
    that (row 3843). A resumed partial takes check 1's own terms (None): its weights start from a state of
    values, which `attention_terms` does not carry."""
    if "resumed" in score:
        return None
    keep = template.dense(spans)[:, None]
    # The parameters' terms over every draw, traced once and its static runs compiled (`float64_terms`).
    traced_terms = None if reference is None else functools.partial(float64_terms, reference)

    def terms(query, key, value, dout, *hooks):
        params, initial = hooks if hooks else ((), None)
        wide = tuple(jnp.asarray(np.asarray(param), np.float64) for param in params)
        # Each head's sink, the state's maximum at any row and query (`template.sinks`).
        sinks = None if initial is None else np.asarray(initial.m)[0, :, 0]
        modded = None if hooked is None else (lambda raw: template.score_modded(raw, hooked(wide, None)))
        found = attention_terms(query, key, value, dout, keep, sinks=sinks, score=modded)
        if not params and initial is None:
            # No hook parameter takes a gradient (a softcap's hook has none): the operands' terms alone, the
            # draw's hooks `((), None)` as they are.
            return (*found[1:], *hooks)
        assert traced_terms is not None, "a hooked call's parameters' terms trace its reference"
        with jax.enable_x64(new_val=True):
            _, traced = traced_terms(*widened((query, key, value, dout, *hooks)))
        return (*found[1:], *traced[3:])

    return terms


@INTERPRETS
@pytest.mark.parametrize("form", ["default", "fused"])
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_from_the_kept_output_is_on_par_interpreted(call, form):
    """The backward from a training forward that keeps its output for D (`template.Kept.OUTPUT`: the inference
    walk's rows, no anchor, `rowsum(dO ⊙ O)` of the rounded output, as FlashAttention's is), in two launches
    and in one: within check 1 of the reference from float64, its D's rounding judged a term at a time."""
    _check_backward(call, SHAPES["interpreted"], form=form, interpret=True, kept=template.Kept.OUTPUT)


@INTERPRETS
def test_the_plain_jax_backward_takes_the_kept_outputs_residuals():
    """The plain-JAX mainloop's backward from residuals a Mosaic training forward keeps its output in
    (`template.Kept.OUTPUT`: no anchor key, each query's -1, the seam's anchor value row zero there), as the
    route hands a Mosaic forward's residuals to whichever mainloop runs the backward: within check 1 of the
    reference from float64, a term at a time."""
    shape = SHAPES["interpreted"]
    spans = template.spans(shape[1], shape[1], shape[0], causal=True)

    def ours(query, key, value, dout):
        with _device(interpret=True):
            residuals = attention_mosaic.forward(query, key, value, spans, residuals=template.Kept.OUTPUT,
                                                 interpret=True)[1:]
        grads = attention_portable.backward(query, key, value, spans, None, dout, block_q=64, block_k=64,
                                            pad=64, residuals=residuals)
        return grads.query, grads.key, grads.value

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans), query, key, value)
        return pull(dout)

    assert_as_exact(jax.jit(ours), jax.jit(reference), _drawn(shape, jnp.bfloat16, 128),
                    "plain-JAX backward from the kept output", seeds=SEEDS, terms=_gradient_terms(spans))


@INTERPRETS
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_forward_is_on_par_interpreted(call, dtype):
    """The forward, the base-2 fold, under Mosaic GPU's interpreter."""
    _check_forward(call, dtype, SHAPES["interpreted"], natural=False, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("call", CALLS)
def test_a_dropping_backward_is_on_par_over_every_call_interpreted(call):
    """The default backward of each call that drops, the dropout multipliers an operand of its derived tile
    cotangents: on par with the reference (check 1)."""
    _check_backward(call, SHAPES["interpreted"], form="default", interpret=True, drop=True)


@INTERPRETS
@pytest.mark.parametrize("call", CALLS)
def test_the_fast_forward_is_on_par_interpreted(call):
    """The forward over fp16 operands (a plain walk, of no headroom guard or compensation) under the run's
    fast forms, the base-2 fold, its walk's step, merge and finish the derived forms
    (`attention_algebra.derived_forms`), its weighted values' sums in the mainloop's accumulator
    (`Keys.accumulate`), under Mosaic GPU's interpreter: on par with the reference (check 1)."""
    fast = dataclasses.replace(generation.numerics(), forms="fast")
    with generation.run_precision(fast):
        _check_forward(call, jnp.float16, SHAPES["interpreted"], natural=False, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
@pytest.mark.parametrize("call", CALLS)
def test_the_training_forward_is_on_par_interpreted(call, natural):
    """The training forward's output, bf16, under Mosaic GPU's interpreter: the weighted values it
    normalizes are its split sums joined with the anchor's term (`algebra.split_output`), the anchor's value
    row the one it kept; its residuals are the backward sites' (`_check_backward`)."""
    _check_forward(call, jnp.bfloat16, SHAPES["interpreted"], natural=natural, interpret=True,
                   training=template.Kept.SPLIT)


@ON_HOPPER
@pytest.mark.parametrize("call", CALLS)
def the_training_forward_is_on_par_on_an_h100(call):
    """The training forward's output, bf16, compiled on an H100."""
    _check_forward(call, jnp.bfloat16, SHAPES["compiled"], natural=False, interpret=False,
                   training=template.Kept.SPLIT)


@INTERPRETS
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_natural_forward_is_on_par_interpreted(call, dtype):
    """The forward with natural logits under the interpreter, bf16's under its headroom guard
    (`algebra.walked_launches`)."""
    _check_forward(call, dtype, SHAPES["interpreted"], natural=True, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("form", CASES)
def test_the_forward_is_on_par_over_every_call_form_interpreted(form):
    """The forward over each of the template core's call forms under the interpreter."""
    _check_call_form(form, interpret=True)


@INTERPRETS
@pytest.mark.parametrize(("width", "value_width"), HEAD_WIDTHS)
def test_the_forward_is_on_par_at_head_widths_interpreted(width, value_width):
    """The forward of query heads `width` wide and value heads `value_width` over a causal call at
    `FORM_TILES`, under the interpreter."""
    _check_forward("causal", jnp.bfloat16, SHAPES["interpreted"], natural=False, interpret=True, width=width,
                   value_width=value_width, tiles=FORM_TILES)


@INTERPRETS
def test_the_backward_at_a_width_of_256_is_on_par_under_bidirectional_blocks_interpreted():
    """The derived backward of heads 256 wide (dK's and dV's widths split among programs,
    `BackwardTiles.columns`, where one program's sums pass a consumer's registers) over a packed row's two
    documents read causally, each document one block read both ways (row 2110's `--split whole`), under the
    interpreter."""
    rows, queries = SHAPES["interpreted"][:2]
    documents = jnp.asarray([[1] * (queries // 2 - 10) + [2] * (queries // 2 + 10)] * rows, jnp.int32)
    spans = template.spans(queries, queries, rows, causal=True, segment_ids=documents,
                           block_groups=documents)
    _check_backward("causal", SHAPES["interpreted"], form=None, interpret=True, width=256, value_width=256,
                    spans=spans)


@INTERPRETS
@pytest.mark.parametrize(("width", "value_width"), HEAD_WIDTHS)
def test_the_backward_is_on_par_at_head_widths_interpreted(width, value_width):
    """The backward in its derived form of query heads `width` wide and value heads `value_width` over a
    causal call, under the interpreter: dK and dV each at its own head width."""
    _check_backward("causal", SHAPES["interpreted"], form=None, interpret=True, width=width,
                    value_width=value_width)


@INTERPRETS
@pytest.mark.parametrize("training", [template.Kept.NONE, template.Kept.SPLIT], ids=["plain", "training"])
@pytest.mark.parametrize("call", CALLS)
def test_the_forward_is_on_par_dropping_interpreted(call, training):
    """The forward and the training forward under probability dropout, its draw the reference's bit for bit
    (`template.Draw`), under the interpreter."""
    _check_forward(call, jnp.bfloat16, SHAPES["interpreted"], natural=False, interpret=True,
                   training=training, drop=True)


@INTERPRETS
@pytest.mark.parametrize("form", [*HOOKED_FORMS, "fused"])
def test_the_backward_is_on_par_dropping_interpreted(form):
    """The backward in each form under probability dropout over a causal call, the fast form's one launch
    too (each consumer's tile its own draw), its D the dropped products' own pass
    (`attention_mosaic_backward._centre_launch`), under the interpreter."""
    _check_backward("causal", SHAPES["interpreted"], form=form, interpret=True, drop=True)


@INTERPRETS
@pytest.mark.parametrize(("dtype", "natural"), HOOK_FORMS.values(), ids=HOOK_FORMS)
@pytest.mark.parametrize("hooks", FORWARD_HOOKS)
def test_the_forward_is_on_par_under_every_hook_interpreted(hooks, dtype, natural):
    """The forward under each hook (`FORWARD_HOOKS`) over causal packed documents under a window, under the
    interpreter, in each of `HOOK_FORMS`."""
    _check_hooks(hooks, dtype, natural, interpret=True)


PADDED_CALLS = {"one-query-default": (1, "default"), "one-query-accurate": (1, "accurate"),
                "whole-padding-tile": (64, "default")}
"""The padded calls' query counts and backward forms (`_backward_form`): one query, its tile mostly padding,
in both forms; and a tile's worth of queries under two consumers, which pads to a pair of query tiles, its
second wholly padding (the accurate form's one consumer pads 64 queries to no more)."""


def _keys_first(spans, *, form: str, scale: float):
    """The two-launch backward's gradients in the form `form` names, interpreted, from this mainloop's own
    training forward at `scale`, as a function of the queries, keys, values and output cotangent."""
    accurate, config = _backward_form(form)

    def gradients(query, key, value, dout):
        with _device(interpret=True), _pinned(FORM_TILES, config):
            residuals = attention_mosaic.forward(query, key, value, spans, scale=scale,
                                                 residuals=template.Kept.SPLIT,
                                                 interpret=True)[1:]
            grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals,
                                              scale=scale, accurate=accurate, interpret=True)
        return grads.query, grads.key, grads.value

    return gradients


def _keys_first_gradients(query, key, value, dout, spans, *, form: str, scale: float):
    """`_keys_first`'s gradients at these arrays, and the float64 reference's (`template.reference`
    widened)."""
    grads = _keys_first(spans, form=form, scale=scale)(query, key, value, dout)
    with jax.enable_x64(new_val=True):
        wide = [jnp.asarray(np.asarray(array, np.float64)) for array in (query, key, value, dout)]
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, scale=scale), *wide[:3])
        truth = tuple(np.asarray(grad) for grad in pull(wide[3]))
    return tuple(np.asarray(grad, np.float64) for grad in grads), truth


@INTERPRETS
@pytest.mark.parametrize("form", HOOKED_FORMS)
def test_the_keys_first_backward_keeps_the_logits_the_forward_keeps_finite_interpreted(form):
    """Queries of 2^120 over keys of 4, one of 8, 64 wide at the scale 1/8: the forward's logits, the
    queries taken by the scale's power before their products (`attention.prescale`), are 2^125 and 2^126,
    where the unscaled products pass fp32's largest. The two-launch backward takes its logits from the
    queries taken by the power (`attention_mosaic_dkv._dkv_step`), so its weights are the forward's,
    one-hot on the key of 8, and every gradient is the float64 reference's to the bit: dV that key's the
    queries' summed cotangents, dQ and dK zero."""
    tokens, width = 64, 64
    query = jnp.full((1, tokens, 1, width), 2.0 ** 120, jnp.bfloat16)
    key = jnp.full((1, tokens, 1, width), 4.0, jnp.bfloat16).at[:, 1].set(8.0)
    value = jnp.broadcast_to(jnp.arange(tokens, dtype=jnp.bfloat16)[None, :, None, None], key.shape)
    dout = jnp.ones((1, tokens, 1, width), jnp.bfloat16)
    spans = template.spans(tokens, tokens, 1)
    found, truth = _keys_first_gradients(query, key, value, dout, spans, form=form, scale=width ** -0.5)
    for mine, true in zip(found, truth, strict=True):
        np.testing.assert_array_equal(mine, true)


@INTERPRETS
@pytest.mark.parametrize("form", HOOKED_FORMS)
def test_the_keys_first_backward_keeps_the_logits_of_keys_the_power_would_flush_interpreted(form):
    """Queries of 2^126 over keys of 2^-126 and 2^-125 in turn, 64 wide at the scale 2^-10: the forward's
    logits are 1/16 and 1/8, where keys taken by the scale's power would fall below bf16's least subnormal
    and flush to zero. The two-launch backward takes the power on its queries, as the forward does, so its
    gradients are on par (check 1) with the fp32 reference's from float64, over drawn values and
    cotangents."""
    tokens, width, scale = 64, 64, 2.0 ** -10
    query = jnp.full((1, tokens, 1, width), 2.0 ** 126, jnp.bfloat16)
    key = jnp.where((jnp.arange(tokens) % 2 == 0)[None, :, None, None], 2.0 ** -126, 2.0 ** -125)
    key = jnp.broadcast_to(key, (1, tokens, 1, width)).astype(jnp.bfloat16)
    spans = template.spans(tokens, tokens, 1)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return query, key, *(jnp.asarray(rng.standard_normal((1, tokens, 1, width)), jnp.bfloat16)
                             for _ in range(2))

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, scale=scale), query, key, value)
        return pull(dout)

    assert_as_exact(jax.jit(_keys_first(spans, form=form, scale=scale)), jax.jit(reference), draw,
                    f"keys-first backward {form} over keys the power would flush", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("form", HOOKED_FORMS)
def test_the_keys_first_backward_keeps_the_key_gradients_of_queries_the_power_would_flush_interpreted(form):
    """A query of 2^-126 over two zero keys, 64 wide at the scale 2^-10, whose values' first entries are
    2^126 and -2^126 under a cotangent on that entry alone: each logit is 0, each score cotangent ±2^125
    and each key's gradient ±2^-11 an entry, where the query taken by the scale's power (2^-136) flushes to
    zero. dK takes the call's own queries, so every gradient is the float64 reference's to the bit."""
    width = 64
    query = jnp.full((1, 1, 1, width), 2.0 ** -126, jnp.bfloat16)
    key = jnp.zeros((1, 2, 1, width), jnp.bfloat16)
    first = jnp.asarray([2.0 ** 126, -2.0 ** 126])
    value = jnp.zeros((1, 2, 1, width), jnp.bfloat16).at[0, :, 0, 0].set(first)
    dout = jnp.zeros((1, 1, 1, width), jnp.bfloat16).at[..., 0].set(1.0)
    found, truth = _keys_first_gradients(query, key, value, dout, template.spans(1, 2, 1), form=form,
                                         scale=2.0 ** -10)
    for mine, true in zip(found, truth, strict=True):
        np.testing.assert_array_equal(mine, true)


@INTERPRETS
@pytest.mark.parametrize(("queries", "form"), PADDED_CALLS.values(), ids=PADDED_CALLS)
def test_padding_rows_reach_no_hook_gradient_interpreted(queries, form):
    """A call of `queries` queries over as many keys, padded to whole programs, under a hook dividing by a
    query's parameter and adding a head's: the padding rows' parameter is 0, so the hook is 0/0 there, and a
    padding row that saw a key would carry its NaN into the values' gradient and the head's, a sum over the
    rows. Padding rows see no key, a tile's last ones or a whole tile of them (`algebra.Forward.queries`),
    and every gradient is the reference's, the head's 0."""
    spans = template.spans(queries, queries, 1)
    query, key, value = (jnp.zeros((1, queries, 1, 64), jnp.bfloat16) for _ in range(3))
    dout = jnp.ones((1, queries, 1, 64), jnp.bfloat16)
    accurate, config = _backward_form(form)

    def score_mod(s, q_pos, k_pos, temperature, bias):
        del q_pos, k_pos
        return s / temperature + bias

    def hooked(temperature, bias):
        return template.Hooks(score_mod, (template.Param(temperature, template.Kind.QUERY),
                                          template.Param(bias, template.Kind.HEAD)))

    params = (jnp.ones((1, 1, queries), jnp.float32), jnp.zeros((1,), jnp.float32))
    hooks = hooked(*params)
    with _device(interpret=True):
        # Its hook divides by a query's parameter, a temperature (`_mosaic_backward_takes`).
        _mosaic_backward_takes(spans, query, key, hooks, accurate=accurate)
    with _device(interpret=True), _pinned(backward=config):
        residuals = attention_mosaic.forward(query, key, value, spans, hooks, residuals=template.Kept.SPLIT,
                                             interpret=True)[1:]
        grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                          accurate=accurate, interpret=True)
    _, pull = jax.vjp(lambda q, k, v, t, b: template.reference(q, k, v, spans, hooked(t, b)), query, key,
                      value, *params)
    for found, expected in zip((grads.query, grads.key, grads.value, *grads.score_params), pull(dout),
                               strict=True):
        np.testing.assert_array_equal(np.asarray(found, np.float32), np.asarray(expected, np.float32))


@INTERPRETS
def test_the_training_forwards_overflowed_rows_keep_their_form_and_every_other_row_interpreted():
    """A call as many queries as heads, one consumer's tile of them, whose head 0 sees two keys of equal
    logits past the fold's reach over values near bf16's largest: the call walks again unfolded, and head
    0's rows, whose weighted values overflow, take the guarded walk in that form (`algebra.unfolded`), its
    residuals in its units. Head 0's output is its values' mean exactly, its one value row (where the
    reference's fp32 sums overflow); every other head's rows are the plain launch's bits, as where head 0's
    values do not overflow; and the backward from these residuals gives head 0's values a gradient wherever
    the reference does, finite where it is."""
    queries, keys, width = 64, 2, 64
    spans = template.spans(queries, keys, 1)
    streams = jax.random.split(jax.random.key(0), 4)
    query, key, value, dout = (jax.random.normal(stream, (1, count, queries, width), jnp.float32)
                               for stream, count in zip(streams, (queries, keys, keys, queries), strict=True))
    # Head 0's logits, 128 * 128 * 64 / 8, past the fold's reach; its values overflow a sum of two or not.
    query, key = (array.at[:, :, 0].set(128.0) for array in (query, key))
    query, key, dout = (array.astype(jnp.bfloat16) for array in (query, key, dout))
    tiles = attention_mosaic.Tiles(64, 64, 2, 1)

    def trained(value):
        with _device(interpret=True), _pinned(tiles):
            return attention_mosaic.forward(query, key, value, spans, residuals=template.Kept.SPLIT,
                                            interpret=True)

    largest = value.at[:, :, 0].set(3e38).astype(jnp.bfloat16)
    out, *residuals = trained(largest)
    ordinary_out, *ordinary = trained(value.at[:, :, 0].set(1.0).astype(jnp.bfloat16))
    head, row = np.asarray(out[:, :, 0], np.float32), np.asarray(largest[:, :1, 0], np.float32)
    np.testing.assert_array_equal(head, np.broadcast_to(row, head.shape))
    np.testing.assert_array_equal(np.asarray(out[:, :, 1:]), np.asarray(ordinary_out[:, :, 1:]))
    for found, plain in zip(residuals[:3], ordinary[:3], strict=True):
        np.testing.assert_array_equal(np.asarray(found[:, 1:]), np.asarray(plain[:, 1:]))
    with _device(interpret=True):
        grads = attention_mosaic.backward(query, key, largest, spans, None, dout, residuals=tuple(residuals),
                                          interpret=True)
    _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans), query, key, largest)
    reference_value = np.asarray(pull(dout)[2], np.float32)
    found_value = np.asarray(grads.value, np.float32)
    finite = np.isfinite(reference_value)
    assert np.isfinite(found_value[finite]).all()
    assert (found_value[finite & (reference_value != 0)] != 0).all()


UNSEEN = {
    "hopper-class": generation.GpuTarget("sm90", "Unmeasured Hopper-class GPU", shared_limit=180 * 1024,
                                         multiprocessors=78, clock_hz=1.6e9, memory_bandwidth=2.4e12),
}
"""A GPU no row measured of this mainloop's family (`test_grouped_gemm.UNSEEN`'s Hopper-class one): fewer
multiprocessors, a smaller block and another L2 bandwidth than an H100's."""


def _bound(op, *arrays, **options):
    """`op`'s arguments for a call of `arrays` (`forward`'s, or `backward`'s with its residuals), as its
    launch binds them compiled, as a GPU process runs it (`MosaicForward.eligible` takes an interpreted call
    in a CPU process alone)."""
    return op.bind(*arrays, interpret=False, **options)


def _kept(launch: str) -> template.Kept:
    """What a forward launch named `launch` keeps: its split sums where it is the training forward."""
    return template.Kept.SPLIT if launch == "training" else template.Kept.NONE


def _forward_call(shape, width: int, value_width: int, residuals: template.Kept):
    """A causal forward's op and bound arguments over `shape`'s operands, its heads `width` and its values
    `value_width` wide."""
    rows, tokens, heads, kv_heads = shape
    spans = template.spans(tokens, tokens, rows, causal=True)
    query = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)
    value = jax.ShapeDtypeStruct((rows, tokens, kv_heads, value_width), jnp.bfloat16)
    op = attention_mosaic.MosaicForward(form="causal", keys=tokens, scale=width ** -0.5)
    return op, _bound(op, query, key, value, spans.start, spans.end, None, None, (), (), None,
                      residuals=residuals)


@pytest.mark.parametrize("residuals", [template.Kept.NONE, template.Kept.SPLIT], ids=["forward", "training"])
@pytest.mark.parametrize(("width", "value_width"), [(128, 64), (128, 128), (128, 256), (256, 256)])
def test_the_derived_tiles_stage_within_a_hoppers_block(width, value_width, residuals):
    """On an H100 the forward's tiles derived from the facts (`MosaicForward.derived`) stage within its
    block (`generation.shared_budget`) and a consumer's registers (`register_budget`), whatever the widths,
    the training forward's too (its anchored walk keeps no value row): no table's tile, which at heads 256
    wide staged 320 KiB against its 227 (ledger 1582); the anchored training forward of values 256 wide
    rewalks a block of its value columns a launch (`Tiles.relaunch_columns`, row 2526)."""
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        op, ba = _forward_call((2, 4096, 32, 8), width, value_width, residuals)
        derived = op.derived(ba, target)
        assert derived is not None
        assert op.legal(ba, derived, target)
        assert op.staged_bytes(ba, derived, target) <= generation.shared_budget(target)
        assert op.registers(ba, derived) <= op.register_budget(derived)


@pytest.mark.parametrize("named", [{"relaunch_k": 64}, {"relaunch_columns": 64}], ids=["key-tile", "columns"])
def test_a_pinned_rewalk_runs_whichever_of_its_tiles_the_pin_names(named):
    """A pin naming any of its natural rewalk's tiles (its key tile, its value columns) runs that rewalk, the
    op deriving none over it (`MosaicForward.completed`, rows 2647, 2675); a pin naming none takes the
    derived one."""
    with generation.assumed("sm90"):
        op, ba = _forward_call((2, 4096, 32, 8), 256, 256, template.Kept.SPLIT)
        derived = op.derived(ba, generation.gpu_target())
        unnamed = dataclasses.replace(derived, relaunch_k=None, relaunch_consumers=None,
                                      relaunch_columns=None)
        pinned = dataclasses.replace(unnamed, **named)
        assert op.completed(ba, unnamed) == derived
        assert op.completed(ba, pinned) == pinned


def _backward_call(shape, width: int, accurate: bool):
    """A causal backward's op and bound arguments over `shape`'s operands, its heads `width` wide, with a
    training forward's residuals."""
    rows, tokens, heads, kv_heads = shape
    spans = template.spans(tokens, tokens, rows, causal=True)
    query = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)
    stats = jax.ShapeDtypeStruct((rows, heads, tokens), jnp.float32)
    residuals = (stats, stats, jax.ShapeDtypeStruct((rows, heads, tokens), jnp.int32),
                 template.Split(jax.ShapeDtypeStruct((rows, heads, tokens, width), jnp.float32), stats, stats,
                                stats))
    op = attention_mosaic.MosaicBackward(form="causal", keys=tokens, scale=width ** -0.5, accurate=accurate)
    return op, _bound(op, query, key, key, query, spans.start, spans.end, None, None, (), None, residuals)


def _softcapped(s, q_pos, k_pos):
    """A score hook a tuning run keys by what it computes (`AttentionLaunch.HOOKS`)."""
    del q_pos, k_pos
    return 30.0 * jnp.tanh(s / 30.0)


def _gated(o):
    """An output hook a tuning run keys by what it computes."""
    return o * 0.5


@pytest.mark.parametrize("launch", ["forward", "backward"])
def test_a_hooked_launchs_keyed_configs_are_its_derived_ones(launch):
    """A hooked launch as its picks key it (`tuning.KernelOp.keyed`: each hook its fingerprint, which a
    stored pick and a tuning run look its configs up by) gives the configs its live call does: its
    legality reads the fingerprint's lowering as the live call established it, never traces the string."""
    rows, tokens, heads, kv_heads = SHAPES["interpreted"]
    spans = template.spans(tokens, tokens, rows, causal=True)
    query = jax.ShapeDtypeStruct((rows, tokens, heads, 128), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, 128), jnp.bfloat16)
    with generation.assumed("sm90"):
        if launch == "forward":
            op = attention_mosaic.MosaicForward(score_mod=_softcapped, output_mod=_gated, form="causal",
                                                keys=tokens, scale=128 ** -0.5)
            ba = _bound(op, query, key, key, spans.start, spans.end, None, None, (), (), None,
                        residuals=template.Kept.NONE)
        else:
            stats = jax.ShapeDtypeStruct((rows, heads, tokens), jnp.float32)
            residuals = (stats, stats, jax.ShapeDtypeStruct((rows, heads, tokens), jnp.int32),
                         template.Split(jax.ShapeDtypeStruct((rows, heads, tokens, 128), jnp.float32), stats,
                                        stats, stats))
            op = attention_mosaic.MosaicBackward(score_mod=_softcapped, form="causal", keys=tokens,
                                                 scale=128 ** -0.5)
            ba = _bound(op, query, key, key, query, spans.start, spans.end, None, None, (), None, residuals)
        keyed = op.keyed(ba)
        assert isinstance(keyed.op.score_mod, str)
        assert set(keyed.autotuning_configs) == set(ba.autotuning_configs)
        assert keyed.heuristics_config == ba.heuristics_config


def test_a_pin_the_mosaic_program_cannot_take_is_illegal():
    """A config a pin or a stored pick names is held to the tiles a program takes (`whole_tiles`) as a
    derived one is: a 32-query tile under wgmma's 64-row roles, or three consumers where the family's walks
    take two, is not legal, so a pin of it is refused before any launch."""
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        op, ba = _forward_call(SHAPES["interpreted"], 128, 128, residuals=template.Kept.NONE)
        derived = op.derived(ba, target)
        assert derived is not None and op.legal(ba, derived, target)
        rows, tokens, _, _ = SHAPES["interpreted"]
        arrays = [jnp.zeros(ba.arguments[name].shape, ba.arguments[name].dtype)
                  for name in ("query", "key", "value")]
        for config in (dataclasses.replace(derived, block_q=32), dataclasses.replace(derived, consumers=3)):
            assert not op.legal(ba, config, target)
            with pytest.raises(tuning.Unpinnable), _pinned(config):
                attention_mosaic.forward(*arrays, template.spans(tokens, tokens, rows, causal=True),
                                         interpret=True)


@INTERPRETS
@pytest.mark.parametrize("launch", ["forward", "training", "backward"])
def test_every_tuning_variant_is_on_par_interpreted(launch):
    """Each config an opt-in tuning run may time on an H100 (`Template.variants`: the derived one and its
    legal neighbours, `_Mainloop.neighbours`) holds check 1 on the interpreted causal call: a tuner picks
    among kernels the gate already holds."""
    shape = SHAPES["interpreted"]
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        if launch == "backward":
            op, ba = _backward_call(shape, 128, accurate=False)
        else:
            op, ba = _forward_call(shape, 128, 128, residuals=_kept(launch))
        variants = op.variants(ba)
    assert len(variants) > 1
    for config in variants:
        if launch == "backward":
            _check_backward("causal", shape, form=None, interpret=True, config=config, target=target)
        else:
            _check_forward("causal", jnp.bfloat16, shape, natural=False, interpret=True, tiles=config,
                           training=_kept(launch), target=target)


@INTERPRETS
def test_a_guarded_launch_walks_its_own_key_tiles():
    """A spans call (four documents over 256 keys, their ids from 1: 0 is padding) whose weighted values
    pass fp32's largest, at a launch of 128-key tiles whose guarded walk takes its own key tiles
    (`guarded_tiles`): the guarded launch runs on tiles its program takes, reads each block's
    keys from its own block map, not the plain launch's, and every row's output is on par with jax's own
    attention, which normalizes its weights before their product (bf16 values near their largest, equal
    logits)."""
    tiles = attention_mosaic.Tiles(64, 128, 2, 1)
    rows, queries, heads, kv_heads = 1, 256, 2, 1
    documents = jnp.repeat(jnp.arange(1, 5, dtype=jnp.int32), queries // 4)[None]
    spans = template.spans(queries, queries, rows, segment_ids=documents)
    keep = (documents[:, :, None] == documents[:, None, :])[:, None]

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jnp.zeros((rows, queries, heads, 128), jnp.bfloat16)
        key = jnp.zeros((rows, queries, kv_heads, 128), jnp.bfloat16)
        value = jax.random.uniform(streams[2], (rows, queries, kv_heads, 128), jnp.float32, 2.9e38,
                                   3.3e38).astype(jnp.bfloat16)
        return query, key, value

    def ours(query, key, value):
        with _device(interpret=True), _pinned(tiles):
            return attention_mosaic.forward(query, key, value, spans, interpret=True)

    def reference(query, key, value):
        return jax.nn.dot_product_attention(query, key, value, mask=keep, implementation="xla")

    assert_as_exact(jax.jit(ours), as_given(jax.jit(reference)), draw, "mosaic guarded spans forward",
                    seeds=SEEDS)


@pytest.mark.parametrize("unseen", UNSEEN)
def test_a_gpu_no_one_measured_runs_the_tiles_its_facts_give(unseen):
    """On a GPU no row measured the forward's and the backward's tiles come from the family's facts and the
    call's extents and fit the GPU's block, and the forward, the training forward and the backward at the
    tiles derived from its facts alone are on par interpreted."""
    target = UNSEEN[unseen]
    shape = (1, 160, 2, 1)
    with generation.assumed(target):
        op, ba = _forward_call(shape, 128, 128, residuals=template.Kept.SPLIT)
        derived = op.derived(ba, target)
        assert derived is not None and op.legal(ba, derived, target)
    _check_forward("causal", jnp.bfloat16, shape, natural=False, interpret=True, target=target)
    _check_forward("causal", jnp.bfloat16, shape, natural=False, interpret=True,
                   training=template.Kept.SPLIT, target=target)
    _check_backward("causal", shape, form=None, interpret=True, target=target)


@INTERPRETS
@pytest.mark.parametrize("form", ["default", "accurate"])
def test_the_backward_is_on_par_from_the_triton_mainloops_residuals_interpreted(form):
    """The backward from the Triton mainloop's training forward's residuals, as the route feeds them on a GPU
    (`attention_route`), under the interpreter."""
    _check_backward("causal", SHAPES["interpreted"], form=form, interpret=True,
                    residuals_from=attention_triton.forward)


@INTERPRETS
def test_a_hook_shaped_to_the_launchs_tile_is_taken():
    """An output hook that reshapes its gate to the value width, on the launch's `[64, 128]` output tile as
    on the reference's rows: the forward is on par under it."""
    queries = 128
    spans = template.spans(queries, queries, 1, causal=True)
    draw_operands = _drawn((1, queries, 2, 1), jnp.bfloat16, width=128, keys=queries)

    def draw(seed):
        query, key, value, _ = draw_operands(seed)
        gate = jax.random.normal(jax.random.PRNGKey(seed), (1, queries, 2, 128), jnp.float32)
        return query, key, value, gate

    def hooked(gate):
        return template.Hooks(output_mod=lambda out, g: out * jax.nn.sigmoid(g.reshape((-1, 128))),
                              output_params=(template.Param(gate, template.Kind.OUTPUT),))

    def ours(query, key, value, gate):
        with _device(interpret=True):
            return attention_mosaic.forward(query, key, value, spans, hooked(gate), interpret=True)

    def reference(query, key, value, gate):
        return template.reference(query, key, value, spans, hooked(gate))

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "mosaic forward, a gate shaped to the tile",
                    seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("form", BACKWARD_FORMS)
@pytest.mark.parametrize("case", CASES)
def test_the_backward_is_on_par_over_every_call_form_interpreted(case, form):
    """The backward in both forms over each of the template core's call forms under the interpreter."""
    _check_call_form(case, interpret=True, backward=True, form=form)


@INTERPRETS
@pytest.mark.parametrize("form", ["forward", *BACKWARD_FORMS])
@pytest.mark.parametrize("case", RUN_CASES)
def test_a_call_over_runs_is_on_par_interpreted(case, form):
    """A decision row's runs (`test_attention_template.RUN_CASES`), several key intervals a query, under the
    interpreter: each block walks its runs' envelope (`attention.block_map`), each tile masked by the union
    of its queries' runs (`algebra.seen`), the forward and the backward in each form."""
    backward = form != "forward"
    _check_call_form(case, interpret=True, backward=backward, form=form if backward else "default")


@INTERPRETS
@pytest.mark.parametrize("score", [(), ("softcap",)], ids=["plain", "softcap"])
@pytest.mark.parametrize("call", CALLS)
def test_an_fp32_forward_walks_its_operands_bf16_parts_interpreted(call, score):
    """An fp32 call's forward on the one mainloop (`MosaicForward.parts`): its operands as three bf16 parts
    each, every product its parts' six (`product.terms`), on par with the reference path in fp32 (check
    1), under no hook and a score hook."""
    spans = template.spans(128, 128, 1, **CALLS[call])
    _check_spans(f"{call} fp32 parts", spans, (1, 128, 2, 1), interpret=True, score=score, dtype=jnp.float32)


@INTERPRETS
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
@pytest.mark.parametrize("call", CALLS)
def test_a_call_of_whole_key_tiles_reads_no_padding_interpreted(call, backward):
    """64 keys under 128 queries at 64-key tiles: the call pads its keys to a pair's 128, a tile wholly past
    them, which a causal query past them reaches too; the keys are whole tiles, so no tile the walk reads
    takes the keys' bound (`algebra.seen`): the walk reads none wholly past them."""
    spans = template.spans(128, 64, 1, **CALLS[call])
    _check_spans(f"{call} of whole key tiles", spans, (1, 128, 2, 1), interpret=True, backward=backward)


@INTERPRETS
@pytest.mark.parametrize("form", BACKWARD_FORMS)
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_is_on_par_interpreted(call, form):
    """The backward in every form under the interpreter, each at its width (`FORM_WIDTHS`)."""
    _check_backward(call, SHAPES["interpreted"], form=form, interpret=True, width=FORM_WIDTHS.get(form, 128))


@INTERPRETS
@pytest.mark.parametrize("accurate", [False, True], ids=["fast", "accurate"])
@pytest.mark.parametrize("call", CALLS)
def test_an_fp16_backward_is_on_par_interpreted(call, accurate):
    """fp16's backward in two launches under the interpreter (row 3497): its weights' and score cotangents'
    parts scaled (`product.Scaled`), each product its own accumulator restored into the carried dK, dV or
    dQ (`operands.Issuer.summed`), on par with the reference (check 1)."""
    _check_backward(call, SHAPES["interpreted"], form="accurate" if accurate else None, interpret=True,
                    dtype=jnp.float16)


@INTERPRETS
def test_a_causal_backward_of_more_keys_than_queries_is_on_par_interpreted():
    """A causal call of 64 queries over 256 keys, whose last key tiles no query sees: the fast form's
    programs each take a key block, the last ones wholly past the queries' diagonal, and walk no query tile,
    the producer loading none for them."""
    _check_backward("causal", (1, 64, 2, 1), form="default", interpret=True, keys=256)


def _shared_key_backward(spans, query, key, value, dout):
    """The fast backward's one launch whose two consumers share a key tile (`fused-shared-keys`, each 32 of
    its 64 keys) under the interpreter: the query's, key's and value's gradients."""
    accurate, config = _backward_form("fused-shared-keys")
    with _device(interpret=True):
        residuals = attention_mosaic.forward(query, key, value, spans, None, interpret=True,
                                             residuals=template.Kept.SPLIT)[1:]
        with _pinned(backward=config):
            grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals,
                                              accurate=accurate, interpret=True)
    return grads.query, grads.key, grads.value


@INTERPRETS
def test_a_shared_key_backward_over_a_partial_key_tile_is_on_par_with_the_reference():
    """The shared-key launch over 96 keys, whose last tile's second piece is all padding, under the
    interpreter: the gradients on par with the reference over drawn operands (check 1)."""
    shape, keys, width = (1, 128, 2, 1), 96, 256
    spans = template.spans(shape[1], keys, shape[0])

    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.standard_normal(extent), jnp.bfloat16)
                     for extent in ((shape[0], shape[1], shape[2], width), (shape[0], keys, shape[3], width),
                                    (shape[0], keys, shape[3], width), (shape[0], shape[1], shape[2], width)))

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, None), query, key, value)
        return pull(dout)

    assert_as_exact(jax.jit(functools.partial(_shared_key_backward, spans)), jax.jit(reference), draw,
                    "a shared key tile past the keys", seeds=SEEDS, terms=_gradient_terms(spans))


@INTERPRETS
def test_a_common_walks_holder_wholly_past_the_keys_reads_none():
    """The fast backward's one launch fed by its consumers (`wgmma.fed`, each consumer 64 of a slot's
    queries, walking the slot's common range) over 64 keys at 64-key tiles, its second key holder wholly
    padding, under the interpreter: every real logit -128 and the padding's 0, which weighs exp(128), past
    fp32's largest, against zero keys and values. The gradients are finite, as the reference's, the holder
    past the call's keys masked (`attention_algebra.Forward.walked_keys`)."""
    query = jnp.zeros((1, 64, 1, 64), jnp.bfloat16).at[..., 0].set(128.0)
    key = jnp.zeros((1, 64, 1, 64), jnp.bfloat16).at[..., 0].set(-1.0)
    value = jnp.zeros_like(key)
    dout = jnp.zeros_like(query).at[..., 0].set(1.0)
    spans = template.spans(64, 64, 1)
    fast = dataclasses.replace(generation.numerics(), forms="fast")
    with generation.run_precision(fast), _device(interpret=True):
        residuals = attention_mosaic.forward(query, key, value, spans, None, interpret=True, scale=1.0,
                                             residuals=template.Kept.SPLIT)[1:]
        with _pinned(backward=attention_mosaic.BackwardTiles(64, 64, 2, 2, fused=True, family="wgmma.fed")):
            grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals,
                                              accurate=False, interpret=True, scale=1.0)
    for grad in (grads.query, grads.key, grads.value):
        assert np.isfinite(np.asarray(grad, np.float32)).all()


@INTERPRETS
def test_a_shared_key_backward_over_a_partial_key_tile_reads_no_padded_key():
    """The shared-key launch over 96 keys under the interpreter, every real logit near -128 (each query 32
    and each key -64 along one feature, the scale 1/16 at width 256) and the output's cotangent zero: a
    padded key's logit of 0 would weigh exp(128), past fp32's largest, whose zero cotangent makes NaN of
    every query's gradient. The gradients are the reference's, zero, and finite (the stability rule's
    claims on the fast form, docs/design/testing.md)."""
    shape, keys, width = (1, 128, 2, 1), 96, 256
    spans = template.spans(shape[1], keys, shape[0])
    rng = np.random.default_rng(0)
    query = rng.standard_normal((shape[0], shape[1], shape[2], width)) * 1e-2
    key = rng.standard_normal((shape[0], keys, shape[3], width)) * 1e-2
    query[..., 0], key[..., 0] = 32.0, -64.0
    value = rng.standard_normal((shape[0], keys, shape[3], width))
    operands = tuple(jnp.asarray(array, jnp.bfloat16) for array in (query, key, value, np.zeros_like(query)))
    for grad in jax.jit(functools.partial(_shared_key_backward, spans))(*operands):
        np.testing.assert_array_equal(np.asarray(grad, np.float32), 0.0)


@INTERPRETS
def test_a_backward_whose_dkv_launch_holds_more_keys_than_its_dq_launch_writes_every_key_interpreted():
    """The backward in two launches, the dK/dV launch's two consumers each their own key block beside one
    dQ consumer (`dkv_consumers=2` on a producer's layout), over 192 keys, under the interpreter: the dK/dV
    launch's last program holds a partial key block, and every key's gradients are written, on par with the
    reference (check 1)."""
    config = attention_mosaic.BackwardTiles(64, 64, 2, 1, dkv_family="wgmma", dkv_consumers=2)
    _check_backward("full", (1, 192, 2, 1), form=None, config=config, interpret=True, width=64)


@INTERPRETS
def test_a_causal_backward_whose_consumers_stage_shared_logits_walks_each_query_tile_interpreted():
    """The backward's dK/dV launch whose two consumers each hold their own causal key block and share a staged
    logits tile (`staged_logits`), over 128 causal queries and keys, under the interpreter: each consumer's
    query walk is its own block's, and the gradients are on par with the reference (check 1)."""
    config = attention_mosaic.BackwardTiles(64, 64, 2, 1, dkv_family="wgmma", dkv_consumers=2,
                                            staged_logits=True)
    _check_backward("causal", (1, 128, 2, 1), form=None, config=config, interpret=True, width=64)


HOPPER_HOOK_SITES = ("alibi", "resumed", "softcap-sinks-alibi")
"""The hooked backward's sites the H100 alone runs (`…_under_every_score_hook_on_an_h100`, compiled, at the
template site's shape and seeds): a head's ALiBi slope or a resumed state is one value a draw, so check 1
takes 128 draws at any head count, past the CPU suite's 240 s target interpreted (1.9 s a draw a row of two
heads: 270 to 650 s, row 1849); the CPU keeps every other hook, sinks and softcap among them."""


@INTERPRETS
@pytest.mark.parametrize("form", ["default", "accurate"])
@pytest.mark.parametrize("hooks", [hooks for hooks in BACKWARD_HOOKS if hooks not in HOPPER_HOOK_SITES])
def test_the_backward_is_on_par_under_every_score_hook_interpreted(hooks, form):
    """The backward in both forms under each score hook over causal packed documents under a window, under
    the interpreter: the gradients of the queries, keys and values, of each hook parameter and of the initial
    state."""
    _check_hooked_backward(hooks, form=form, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("kept", [template.Kept.NONE, template.Kept.SPLIT], ids=["forward", "training"])
def test_a_folded_forward_whose_products_pass_fp32_below_zero_rewalks_natural_interpreted(kept):
    """A full call whose raw products pass fp32's range below zero (a query 2^126 against keys -8, scale
    1/4: its power of two taken into the launch's scale, `_prepared`): every row's maximum reads -inf, which
    a call whose every row sees a key takes as past the fold's reach (`reach_of`), so it walks again
    natural on the query taken by the power, where its products stay finite (-2^127), and every equal
    logit weighs its values alike: the output, the training forward's too, the values' mean, 1 (as an empty
    row's -inf it read nothing: review of the FA3-gap batch's F3)."""
    shape = (1, 64, 1, 16)
    query = jnp.zeros(shape, jnp.bfloat16).at[..., 0].set(2.0 ** 126)
    key = jnp.zeros(shape, jnp.bfloat16).at[..., 0].set(-8.0)
    value = jnp.ones(shape, jnp.bfloat16)
    spans = template.spans(shape[1], shape[1], shape[0])
    with _device(interpret=True):
        out = attention_mosaic.forward(query, key, value, spans, scale=0.25, residuals=kept, interpret=True)
    out = out if kept is template.Kept.NONE else out[0]
    assert np.array_equal(np.asarray(out, np.float32), np.ones(shape, np.float32))


@INTERPRETS
def test_a_forward_kernel_built_for_its_facts_runs_each_calls_own_interpreted():
    """The forward's kernel is lowered once for every call of its static facts (`tuning.lowered_once`, row
    2440): calls alike but for their scale, past their queries' one power of two (0.2 and 0.24, both 2^-3
    times a scale left of 1.6 and 1.92), each run their own though their kernels' names agree (launched
    outside an op's report, which names them by its launch, `report.named`), each on par (check 1) with the
    reference's at its scale from float64."""
    shape = SHAPES["interpreted"]
    spans = template.spans(shape[1], shape[1], shape[0], causal=True)
    for scale in (0.2, 0.24):
        def ours(query, key, value, dout, scale=scale):
            del dout
            with _device(interpret=True):
                return attention_mosaic_forward._plain_forward(
                    query, key, value, spans, template.Hooks(), scale=scale, natural=attention_mosaic.NATURAL,
                    tiles=FORM_TILES, interpret=True)

        def reference(query, key, value, dout, scale=scale):
            del dout
            return template.reference(query, key, value, spans, scale=scale)

        assert_as_exact(jax.jit(ours), jax.jit(reference), _drawn(shape, jnp.bfloat16),
                        f"mosaic forward at scale {scale}", seeds=SEEDS)


@INTERPRETS
def test_a_forward_kernel_built_for_its_facts_holds_no_calls_arrays_interpreted():
    """The forward's kernel, lowered once for every call of its static facts (`tuning.lowered_once`, rows
    2440, 2808), holds none of a call's arrays, its operands' nor its hooks' (an initial state's sums, which
    its static hooks describe by shape and dtype): each is released once the call is."""
    shape = SHAPES["interpreted"]
    spans = template.spans(shape[1], shape[1], shape[0], causal=True)
    query, key, value, _ = _drawn(shape, jnp.bfloat16)(0)
    hooks = template.Hooks(initial=template.sinks(jnp.linspace(-1.0, 1.0, shape[2]), shape[0], shape[1], 128))
    refs = [weakref.ref(array) for array in (query, key, value, hooks.initial.m, hooks.initial.acc)]
    with _device(interpret=True):
        jax.block_until_ready(attention_mosaic.forward(query, key, value, spans, hooks, interpret=True))
    del query, key, value, hooks
    gc.collect()
    assert [ref() is None for ref in refs] == [True] * len(refs)


@INTERPRETS
@pytest.mark.parametrize("call", ["causal", "full"])
def test_a_backward_256_wide_shares_its_keys_and_is_on_par_interpreted(call):
    """The fast backward 256 wide, whose consumers' own keys' whole dK and dV sums pass their registers, at
    the tiles the derivation takes for it: one launch whose two consumers share a key block, each its share
    of the columns (`BackwardTiles.shared_keys`), every gradient on par (check 1) with the reference's,
    under the interpreter."""
    _check_backward(call, SHAPES["interpreted"], form="fused-shared-keys", interpret=True, width=256)


@INTERPRETS
@pytest.mark.parametrize("sinks", [False, True], ids=["plain", "sinks"])
@pytest.mark.parametrize("call", ["causal", "full"])
def test_a_training_forward_256_wide_past_the_folds_reach_is_on_par_interpreted(call, sinks):
    """An anchored training forward 256 wide whose logits pass the fold's reach (`algebra.FOLD_REACH`, its
    queries drawn 2^16 times larger) walks again in the natural form, whose launches take a block of the
    value columns each (`Tiles.relaunch_columns`, row 2526: no tile holds its whole sums), with attention
    sinks each block from its own columns of the initial state's sums (row 2671), its output on par
    (check 1) with the reference's from float64, under the interpreter."""
    _check_forward(call, jnp.bfloat16, SHAPES["interpreted"], natural=False, interpret=True, width=256,
                   value_width=256, training=template.Kept.SPLIT, query_scale=2.0 ** 16, sinks=sinks)


@INTERPRETS
@pytest.mark.parametrize(("hooks", "deterministic"), [("key-bias", False), ("tile-bias", False),
                                                      ("tile-bias", True)])
def test_a_parameter_every_row_shares_takes_each_rows_gradient_interpreted(hooks, deterministic, request):
    """A key's or a tile's parameter one for every batch row (`[1, H, ...]`, T5's shared bias): the dK/dV
    programs read it at its shared place and add each row's gradient into its own shape in the program
    (`shared_place`, `Family.added`, row 2577), or under --xla_gpu_deterministic_ops each program writes its
    own place, summed after (row 2645), every gradient the reference's, under the interpreter."""
    request.getfixturevalue("with_deterministic_ops" if deterministic else "without_deterministic_ops")
    _check_hooked_backward(hooks, form="default", interpret=True, shared=True)


@INTERPRETS
@pytest.mark.parametrize(("hooks", "form"), [("softcap", "fused"), ("softcap", "fused-shared-keys"),
                                             ("sinks", "fused")])
def test_a_hooked_backward_in_one_launch_holds_check_1_interpreted(hooks, form, without_deterministic_ops):
    """The fast form's one launch queries first under a softcap, a score hook of no parameters, each
    consumer its own keys or its share of a shared key block (`attention_mosaic_coop`), its score
    cotangents taken back through the cap's slope (`gradient.pulled`) before dQ's and dK's products; or
    under sinks, the softmax's initial state, whose gradient the first pass's D and the keys' share of the
    weights give: every gradient the reference's (check 1) over causal packed documents under a window,
    under the interpreter, heads as wide as the form takes (`FORM_WIDTHS`)."""
    _check_hooked_backward(hooks, form=form, interpret=True, width=FORM_WIDTHS.get(form, 64))


@INTERPRETS
@pytest.mark.parametrize("hooks", ["key-bias", "tile-bias"])
def test_a_width_split_backward_writes_its_score_parameters_gradients_once_interpreted(hooks):
    """The accurate backward 128 wide, its sums in two column halves (`BackwardTiles.columns`), under a hook
    whose gradient the dK/dV programs write (a key's parameter, a tile's): the first half's program writes
    it, and every gradient is the reference's, under the interpreter."""
    _check_hooked_backward(hooks, form="accurate-halves", interpret=True, width=128)


@lowers("sm90")
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("call", ["causal", "causal-documents-window", "runs-documents-valid"])
def test_the_launches_lower_for_an_h100(call, accurate):
    """The forward and the backward's launches lower for an H100 on any host, as the GPU gate runs them: a
    causal call, a spans one of documents, whose programs read its block map and ids, and one over a
    decision row's runs, whose programs read each run's bounds; the forward under every hook too, whose
    programs read each kind of parameter and the initial state."""
    if call == "causal":
        rows, tokens, heads, kv_heads = SHAPES["compiled"]
        spans = template.spans(tokens, tokens, rows, causal=True)
    else:
        case = CASES[call] if call in CASES else RUN_CASES[call]
        (tokens, _), rows, heads, kv_heads = core_geometry(case), ROWS, 4, 2
        spans = template.spans(tokens, tokens, rows, **arguments(case))
    q = jax.ShapeDtypeStruct((rows, tokens, heads, 128), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((rows, tokens, kv_heads, 128), jnp.bfloat16)
    rows_shape = jax.ShapeDtypeStruct((rows, heads, tokens), jnp.float32)
    others = jax.ShapeDtypeStruct((rows, heads, tokens, 128), jnp.float32)
    split = template.Split(others, rows_shape, rows_shape, rows_shape)
    residuals = (rows_shape, rows_shape, jax.ShapeDtypeStruct((rows, heads, tokens), jnp.int32), split)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    score_mod, kinds, draw_hooks = score_hooks(FORWARD_HOOKS["all"][0], tokens, tokens, 128)
    params, initial = draw_hooks(0)
    gate = hooks_for(HOOKS["gate"], tokens, 128, gate_of(tokens, 128))
    hooks = template.Hooks(score_mod, tuple(template.Param(value, kind)
                                            for value, kind in zip(params, kinds, strict=True)), initial,
                           gate.output_mod, gate.output_params)
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        for forward_hooks in (None, hooks):
            for training in (template.Kept.NONE, template.Kept.SPLIT):
                entry = functools.partial(attention_mosaic.forward, spans=spans, hooks=forward_hooks,
                                          residuals=training)
                jax.jit(entry).trace(q, kv, kv).lower(lowering_platforms=("cuda",))
        jax.jit(lambda q, k, v, dout, residuals: attention_mosaic.backward(
            q, k, v, spans, None, dout, residuals=residuals, accurate=accurate)).trace(
            q, kv, kv, q, residuals).lower(lowering_platforms=("cuda",))


APPROXIMATED_BY_MOSAIC = ("exp", "exp2", "log", "log2", "log1p", "tanh", "logistic", "sin", "cos", "sqrt",
                          "rsqrt")
"""The unary operations whose Mosaic GPU lowering takes `approx_math` (jax's Pallas Mosaic GPU lowering rules
for exp, exp2, log, log2, tanh, sin, cos, sqrt and rsqrt, which pass it to the lane implementation and set
the fast-math flag; logistic and log1p lower through them), read from the lowering, not from the
mainloop's own list."""


@pytest.mark.parametrize("operation", APPROXIMATED_BY_MOSAIC)
def test_every_approximated_operation_in_a_hook_takes_exact_math(operation):
    """A score hook and an output hook each computing one operation Mosaic GPU approximates under
    `approx_math` (`APPROXIMATED_BY_MOSAIC`) take exact math, in the forward and differentiated: no such
    operation of a hook passes unseen."""
    def applied(x):
        return getattr(jax.lax, operation)(jnp.abs(x) + 1.0)

    def score_mod(s, q_pos, k_pos):
        del q_pos, k_pos
        return s + applied(s)

    def output_mod(out):
        return applied(out)

    for hooks in (template.Hooks(score_mod), template.Hooks(output_mod=output_mod)):
        for differentiated in (False, True):
            assert not attention_mosaic_program.approximate(hooks, (64, 64), 64,
                                                            differentiated=differentiated)


def _fullest(op, ba, target):
    """The fitting config of `op`'s launch of `ba` on `target` that stages the most shared memory."""
    return max(op.fitting(ba, target), key=lambda config: op.staged_bytes(ba, config, target))


@lowers("sm90")
@pytest.mark.parametrize("launch", ["forward", "training", "default", "accurate"])
def test_the_fullest_fitting_tiles_lower_within_a_hoppers_block(launch):
    """Mosaic GPU's own lowering, which refuses a kernel staging past a block's shared memory, takes each
    launch at the fitting config that stages the most (`staged_bytes`), the one nearest the H100's block: the
    model counts every array and barrier the launch allocates (`_staged`), an independent witness of it."""
    rows, tokens, heads, kv_heads = SHAPES["compiled"]
    width = 128
    spans = template.spans(tokens, tokens, rows, causal=True)
    q = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)
    stats = jnp.zeros((rows, heads, tokens), jnp.float32)
    residuals = (stats, stats, jnp.zeros((rows, heads, tokens), jnp.int32),
                 template.Split(jnp.zeros((rows, heads, tokens, width), jnp.float32), stats, stats, stats))
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        target = generation.gpu_target()
        if launch in ("forward", "training"):
            training = _kept(launch)
            op, ba = _forward_call((rows, tokens, heads, kv_heads), width, width, training)
            fullest = _fullest(op, ba, target)
            assert op.staged_bytes(ba, fullest, target) <= generation.shared_budget(target)
            with _pinned(fullest):
                entry = functools.partial(attention_mosaic.forward, spans=spans, residuals=training)
                jax.jit(entry).trace(q, kv, kv).lower(lowering_platforms=("cuda",))
            return
        accurate = launch == "accurate"
        op = attention_mosaic.MosaicBackward(form="causal", keys=tokens, scale=width ** -0.5,
                                             accurate=accurate)
        ba = _bound(op, q, kv, kv, q, spans.start, spans.end, None, None, (), None, residuals)
        fullest = _fullest(op, ba, target)
        assert op.staged_bytes(ba, fullest, target) <= generation.shared_budget(target)
        with _pinned(backward=fullest):
            jax.jit(lambda q, k, v, dout, residuals: attention_mosaic.backward(
                q, k, v, spans, None, dout, residuals=residuals, accurate=accurate)).trace(
                q, kv, kv, q, residuals).lower(lowering_platforms=("cuda",))


@lowers("sm90")
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("hooks", BACKWARD_HOOKS)
def test_the_backward_under_every_score_hook_lowers_for_an_h100(hooks, accurate):
    """The backward under each score hook lowers for an H100 on any host, as the GPU gate runs it: each hook
    parameter and the initial state read per tile by the dQ and dK/dV programs."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    q = jax.ShapeDtypeStruct((ROWS, queries, 4, 128), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((ROWS, keys, 2, 128), jnp.bfloat16)
    rows_shape = jnp.zeros((ROWS, 4, queries), jnp.float32)
    split = template.Split(jnp.zeros((ROWS, 4, queries, 128), jnp.float32), rows_shape, rows_shape,
                           rows_shape)
    residuals = (rows_shape, rows_shape, jnp.zeros((ROWS, 4, queries), jnp.int32), split)
    score_mod, kinds, draw_hooks = score_hooks(SCORE_HOOKS[hooks], queries, keys, 128)
    params, initial = draw_hooks(0)

    def backward_of(q, k, v, dout, residuals, params, initial):
        hooked = template.Hooks(score_mod, tuple(template.Param(value, kind)
                                                 for value, kind in zip(params, kinds, strict=True)),
                                initial)
        return attention_mosaic.backward(q, k, v, spans, hooked, dout, residuals=residuals, accurate=accurate)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        hooked = template.Hooks(score_mod, tuple(template.Param(value, kind)
                                                 for value, kind in zip(params, kinds, strict=True)), initial)
        _mosaic_backward_takes(spans, q, kv, hooked, accurate=accurate)
        jax.jit(backward_of).trace(q, kv, kv, q, residuals, params, initial).lower(
            lowering_platforms=("cuda",))


@ON_HOPPER
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def the_forward_is_on_par_on_an_h100(call, dtype):
    """The forward, the base-2 fold, compiled on an H100."""
    _check_forward(call, dtype, SHAPES["compiled"], natural=False, interpret=False)


@ON_HOPPER
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def the_natural_forward_is_on_par_on_an_h100(call, dtype):
    """The forward with natural logits compiled on an H100."""
    _check_forward(call, dtype, SHAPES["compiled"], natural=True, interpret=False)


@ON_HOPPER
@pytest.mark.parametrize("form", CASES)
def the_forward_is_on_par_over_every_call_form_on_an_h100(form):
    """The forward over each of the template core's call forms compiled on an H100."""
    _check_call_form(form, interpret=False)


@ON_HOPPER
@pytest.mark.parametrize(("width", "value_width"), HEAD_WIDTHS)
def the_forward_is_on_par_at_head_widths_on_an_h100(width, value_width):
    """The forward of query heads `width` wide and value heads `value_width` over a causal call at
    `FORM_TILES`, compiled on an H100."""
    _check_forward("causal", jnp.bfloat16, SHAPES["compiled"], natural=False, interpret=False, width=width,
                   value_width=value_width, tiles=FORM_TILES)


@ON_HOPPER
@pytest.mark.parametrize(("width", "value_width"), HEAD_WIDTHS)
def the_backward_is_on_par_at_head_widths_on_an_h100(width, value_width):
    """The backward in its derived form of query heads `width` wide and value heads `value_width` over a
    causal call, compiled on an H100."""
    _check_backward("causal", SHAPES["compiled"], form=None, interpret=False, width=width,
                    value_width=value_width)


@ON_HOPPER
@pytest.mark.parametrize(("dtype", "natural"), HOOK_FORMS.values(), ids=HOOK_FORMS)
@pytest.mark.parametrize("hooks", FORWARD_HOOKS)
def the_forward_is_on_par_under_every_hook_on_an_h100(hooks, dtype, natural):
    """The forward under each hook compiled on an H100."""
    _check_hooks(hooks, dtype, natural, interpret=False)


@ON_HOPPER
@pytest.mark.parametrize("form", BACKWARD_FORMS)
@pytest.mark.parametrize("case", CASES)
def the_backward_is_on_par_over_every_call_form_on_an_h100(case, form):
    """The backward in both forms over each of the template core's call forms compiled on an H100."""
    _check_call_form(case, interpret=False, backward=True, form=form)


@ON_HOPPER
@pytest.mark.parametrize("form", ["forward", *BACKWARD_FORMS])
@pytest.mark.parametrize("case", RUN_CASES)
def a_call_over_runs_is_on_par_on_an_h100(case, form):
    """A decision row's runs compiled on an H100, the forward and the backward in each form."""
    backward = form != "forward"
    _check_call_form(case, interpret=False, backward=backward, form=form if backward else "default")


@ON_HOPPER
def a_causal_backward_of_more_keys_than_queries_is_on_par_on_an_h100():
    """A causal call of 64 queries over 256 keys compiled on an H100, where a consumer waiting for a load
    the producer never issues hangs (the interpreted test's case)."""
    _check_backward("causal", (2, 64, 4, 2), form="default", interpret=False, keys=256)


@ON_HOPPER
@pytest.mark.parametrize("form", BACKWARD_FORMS)
@pytest.mark.parametrize("call", CALLS)
def the_backward_is_on_par_on_an_h100(call, form):
    """The backward in both forms compiled on an H100."""
    _check_backward(call, SHAPES["compiled"], form=form, interpret=False, width=FORM_WIDTHS.get(form, 128))


@ON_HOPPER
@pytest.mark.parametrize("form", ["default", "accurate"])
@pytest.mark.parametrize("hooks", BACKWARD_HOOKS)
def the_backward_is_on_par_under_every_score_hook_on_an_h100(hooks, form):
    """The backward in both forms under each score hook compiled on an H100."""
    _check_hooked_backward(hooks, form=form, interpret=False)


def _counted(fn, *arrays) -> tuple[float, float]:
    """The model's share of the Mosaic launches' work `fn` lowers for an H100 each declares
    (`instrumentation.MODEL_FLOPS`) and the hardware's work each states (its cost estimate), as the lowered
    program's structured reader reads them (`lowered.kernel_launches`, each launch times the calls a call
    of the program makes)."""
    from dew.nn.kernels.lowered import kernel_launches

    # Lowered deviceless for an H100 whatever the host, so a GPU host of another generation lowers no TMA
    # of its own.
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(fn).trace(*arrays).lower(lowering_platforms=("cuda",))
    launches = [(call, times) for call, times in kernel_launches(lowered) if call.kind == "mosaic_gpu"]
    assert launches and all(call.flops is not None and call.model_flops is not None for call, _ in launches)
    return (sum((call.model_flops or 0.0) * times for call, times in launches),
            sum((call.flops or 0.0) * times for call, times in launches))


@pytest.mark.parametrize("causal, queries, keys, pairs", [
    (True, 256, 256, 256 * 257 // 2), (True, 128, 256, 128 * 129 // 2),
    (True, 256, 128, 128 * 129 // 2 + 128 * 128), (True, 64, 192, 64 * 65 // 2),
    (False, 256, 256, 256 * 256), (False, 64, 192, 64 * 192)],
    ids=["causal", "causal-fewer-queries", "causal-more-queries", "causal-padded", "full", "full-padded"])
def test_the_mosaic_launches_state_the_models_attention_flops(causal, queries, keys, pairs):
    """A bf16 call's plain forward, training forward and backward through the Mosaic mainloop, each lowered
    for an H100: every launch states its hardware work (`attention_mosaic_program.stated`), which Dew's FLOP
    count reads to size it (`instrumentation._stated_flops`: no launch unsized), and declares the model's
    share of it; together the model's are the library's fused attention's own count, a multiply-add a
    reached pair, head and product column (a causal call's `pairs` each query's keys up to its own place,
    written out: the first key aligned to the first query, as the mainloop masks them): the forward's two
    products, the training forward's two and the backward's four (`instrumentation._fused_attention_flops`).
    The hardware's work is every tile its walk issues, whole: each 64 by 64 tile holding a reached pair over
    the padded rows (a padded consumer's and a causal diagonal tile's masked half computed with the rest, as
    an H100's tensor-pipe counts read them, row 3948's receipt), and one recomputed `Q K^T` more in the
    training forward and in the backward; a rerun (the guarded launch, the fold's natural walk) adds none. The
    launches' own statements are read: the training forward's overflowed rows rerun the plain-JAX walk in a
    conditional, whose products the count bounds apart."""
    from dew.telemetry.instrumentation import hlo_flops

    rows, heads, kv_heads, width = 1, 4, 2, 64
    query = jax.ShapeDtypeStruct((rows, queries, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, keys, kv_heads, width), jnp.bfloat16)
    spans = template.spans(queries, keys, rows, causal=causal)
    forward = 2 * rows * heads * pairs * (width + width)
    # The 64 by 64 tiles the pinned launches walk that hold a pair a causal call reaches (a key at or before
    # its query), each computed whole, over the queries padded to a program's two consumers' rows (a padded
    # consumer walks its keys as a real one does) and the call's keys: the keys pad to two tiles too, a tile
    # wholly past them walked by none.
    block, consumers = 64, 2
    padded = -(-queries // (consumers * block)) * consumers * block
    walked = sum(block * block for first_q in range(0, padded, block) for first_k in range(0, keys, block)
                 if not causal or first_k < first_q + block)
    issued = 2 * rows * heads * walked * (width + width)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")

    def trained(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, residuals=template.Kept.SPLIT)

    with generation.assumed("sm90"):
        residuals = jax.eval_shape(trained, query, key, key)[1:]

    def backward(query, key, value, dout, residuals):
        return attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals)

    # The training forward recomputes its walk's `Q K^T`; the fast backward its `S` beside its four.
    product = issued // 2
    pins = (attention_mosaic.Tiles(block, block, 2, consumers),
            attention_mosaic.BackwardTiles(block, block, 2, consumers, fused=True))
    with _pinned(*pins):
        plain = _counted(lambda q, k, v: attention_mosaic.forward(q, k, v, spans), query, key, key)
    assert plain == (forward, issued)
    # A forward whose launches walk its value columns a block apiece: each recomputes `Q K^T`, which the
    # model owns once. The blocks' launches are one kernel called once a block, as the step's FLOP count
    # weighs it (`hlo_flops`, against the whole walk's count).
    def weighed(fn):
        with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
            found = hlo_flops(jax.jit(fn).trace(query, key, key).lower(lowering_platforms=("cuda",))
                              .as_text(dialect="hlo"))
        assert found is not None
        return found.model, found.hardware

    with _pinned(*pins):
        whole = weighed(lambda q, k, v: attention_mosaic.forward(q, k, v, spans))
    with _pinned(attention_mosaic.Tiles(block, block, 2, consumers, columns=width // 2)):
        blocked = weighed(lambda q, k, v: attention_mosaic.forward(q, k, v, spans))
    assert blocked == (whole[0], whole[1] + product)
    with _pinned(*pins):
        assert _counted(trained, query, key, key) == (forward, issued + product)
        assert _counted(backward, query, key, key, query, residuals) == (2 * forward, 2 * issued + product)


def test_a_common_walk_states_every_tile_each_holder_walks():
    """The fast backward's one launch fed by its consumers (`wgmma.fed`, a program two key blocks, each
    consumer walking every query tile either block is read by, `Family.walk_own`) over a causal call of 128
    queries and 192 keys, its keys padded to 256, lowered for an H100: the launch states the tiles each
    consumer walks (`attention_mosaic_program.issued`), one its own keys are not seen by too. The first
    program's consumers walk both 64-row query tiles each, the second keys' tile read by the second alone;
    the second program's none, its first keys past every query's and its second past the keys."""
    rows, heads, width, block = 1, 2, 64, 64
    query = jax.ShapeDtypeStruct((rows, 128, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, 192, heads, width), jnp.bfloat16)
    spans = template.spans(128, 192, rows, causal=True)
    with generation.assumed("sm90"):
        residuals = jax.eval_shape(lambda q, k, v: attention_mosaic.forward(
            q, k, v, spans, residuals=template.Kept.SPLIT), query, key, key)[1:]

    def backward(query, key, value, dout, residuals):
        return attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals)

    # The backward's five products (`S` recomputed, dP's, dV's, dK's and dQ's) over four tiles where each
    # consumer walked its own keys' readers alone: three.
    walked, pairs = 4 * block * block, 128 * 129 // 2
    with _pinned(backward=attention_mosaic.BackwardTiles(block, block, 2, 2, fused=True, family="wgmma.fed")):
        assert _counted(backward, query, key, key, query, residuals) == (
            2 * rows * heads * pairs * 4 * width, 2 * rows * heads * walked * 5 * width)


@pytest.mark.parametrize("axis", [0, 1])
def test_a_hook_that_forms_a_unit_axis_compiles_for_an_h100(axis):
    """A score hook that forms a unit column (`q_pos[:, :1]`, `axis` 0) or a unit row (`k_pos[:1, :]`, 1,
    which Mosaic GPU selects by a reduction across warps, its scratch the launch's warpgroups reserve,
    ledger 1428), the training forward and the backward lowered and compiled by ptxas for an H100 on this
    CPU (tests/kernel_compile.py), within its shared memory (`failures`: no lowering error, no block past
    it). Its registers at the derived tiles are the template's register model's (ledger 1617), hooked or
    not."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, failures

    queries = 256
    spans = template.spans(queries, queries, 1, causal=True)
    def score_mod(s, q_pos, k_pos):
        return s + 0.01 * (q_pos[:, :1] if axis == 0 else k_pos[:1, :]).astype(s.dtype)

    hooks = template.Hooks(score_mod)
    query = jax.ShapeDtypeStruct((1, queries, 4, 128), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, queries, 2, 128), jnp.bfloat16)

    def trained(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, hooks, residuals=template.Kept.SPLIT)

    def backward(query, key, value, dout):
        residuals = trained(query, key, value)[1:]
        grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals)
        return grads.query, grads.key, grads.value

    found = compile_mosaic_kernels([KernelCase("forward", lambda: (trained, (query, key, key))),
                                    KernelCase("backward", lambda: (backward, (query, key, key, query)))],
                                   "sm90")
    assert not failures(found), "\n".join(failures(found))


GATED = ("sm90",)
"""The GPUs whose families the mainloop qualifies (`attention_mosaic_program.QUALIFIED`): an H100's
wgmma."""

BACKWARD_GATE = [(generation_name, width, accurate, "none") for generation_name in GATED
                 for width in (64, 128, 256) for accurate in (False, True)]
BACKWARD_GATE += [(generation_name, width, False, hooks) for generation_name in GATED for width in (64, 128)
                  for hooks in BACKWARD_HOOKS if hooks != "none"]
"""The backward resource gate's cases: each GPU whose family the mainloop qualifies (`GATED`), a causal
call's head widths, each form, hookless; and each score hook at the widths a hooked call's derivation takes
whole (row 2281: its live set beside the step's)."""


def _gated_hooks(hooks: str, queries: int, width: int):
    """The gate's score hook `SCORE_HOOKS[hooks]` over `queries` queries and keys, four query heads, its
    parameters' first draw (`score_hooks`), as `template.Hooks`; None for none."""
    if hooks == "none":
        return None
    score_mod, kinds, draw_hooks = score_hooks(SCORE_HOOKS[hooks], queries, queries, width, rows=1, heads=4)
    params, initial = draw_hooks(0)
    return template.Hooks(score_mod, tuple(template.Param(value, kind) for value, kind in zip(params, kinds,
                                                                                               strict=True)),
                          initial)


def _gated_backward(generation_name: str, width: int, accurate: bool, hooks: str = "none"):
    """The resource gate's backward op, bound arguments and derived config: a causal call of 512 queries,
    four query heads over two key heads `width` wide, bf16, in the form `accurate` names, under the score
    hook `hooks` names (`_gated_hooks`), for `generation_name`."""
    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jnp.zeros((1, queries, 4, width), jnp.bfloat16)
    key = jnp.zeros((1, queries, 2, width), jnp.bfloat16)
    hooked = _gated_hooks(hooks, queries, width)
    with generation.assumed(generation_name):
        op, args = attention_mosaic._backward_call(query, key, key, spans, hooked, query,
                                                   _residuals(query.shape), accurate=accurate, scale=None,
                                                   natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, interpret=False)
        return op, ba, op.derived(ba, generation.gpu_target())


@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
def test_an_a100_derives_no_mosaic_backward(accurate):
    """The mma.sync family is not one this mainloop qualifies (`attention_mosaic_program.QUALIFIED`, row
    2078): an A100's backward derives no config, so the route takes another mainloop there."""
    assert _gated_backward("sm80", 128, accurate)[2] is None


FORWARD_GATE = [(generation_name, width, training, "none") for generation_name in GATED
                for width in (64, 128, 256)
                for training in (template.Kept.NONE, template.Kept.OUTPUT, template.Kept.SPLIT)]
FORWARD_GATE += [(generation_name, width, training, hooks) for generation_name in GATED for width in (64, 128)
                 for training in (template.Kept.NONE, template.Kept.OUTPUT, template.Kept.SPLIT)
                 for hooks in BACKWARD_HOOKS if hooks != "none"]
FORWARD_GATE += [(generation_name, 256, template.Kept.NONE, "softcap") for generation_name in GATED]
"""The forward resource gate's cases: each GPU whose family the mainloop qualifies (`GATED`), a causal call's
head widths, inference and both training residual forms (the 16-bit fast path's output and the anchor's
split sums), hookless, each score hook at the widths a hooked call's derivation takes (row 2281: its live set
beside the walk's), and a softcap's inference forward 256 wide (Gemma's)."""


@COMPILES_ON_CPU
def test_the_public_template_route_runs_each_shards_rows_and_heads_with_no_collective():
    """The public call on the template (`scaled_dot_product_attention`), causal, forward and backward, its
    batch split over a compile-only H100 mesh's data axis and its heads over its tensor axis, compiled for
    them: the forward launches, reading its shard's rows and heads (four query heads of eight, two key heads
    of four, two rows of four), and the program moves and sums nothing, its backward's launches whichever
    the derivation takes (`kernel_compile.Expected`, its route evidence): a head's softmax and its gradients
    are its own."""
    import fla_gdn_support as support
    from jax.sharding import PartitionSpec
    from kernel_compile import Expected, compile_mosaic_kernels, route_failures

    from dew.nn.attention import scaled_dot_product_attention

    rows, tokens, heads, kv_heads, width = 4, 256, 8, 4, 64

    def call():
        def step(query, key, value):
            def loss(query, key, value):
                out = scaled_dot_product_attention(query, key, value, implementation="template", causal=True)
                return out.astype(jnp.float32).sum()
            return jax.grad(loss, argnums=(0, 1, 2))(query, key, value)

        def shaped(kv):
            return jax.ShapeDtypeStruct((rows, tokens, kv, width), jnp.bfloat16)
        return step, (shaped(heads), shaped(kv_heads), shaped(kv_heads))

    spec = PartitionSpec("data", None, "tensor", None)
    expected = Expected(axes=(("data", 2), ("tensor", 2)), specs=(spec, spec, spec),
                        launches=frozenset({"attention_forward"}),
                        shards={"attention_forward": (rows // 2, tokens, heads // 2, width)})
    cases = [support.KernelCase("template route", call, expected=expected)]
    assert route_failures(cases, compile_mosaic_kernels(cases, "sm90")) == []


@COMPILES_ON_CPU
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("kept", [template.Kept.NONE, template.Kept.OUTPUT], ids=["inference", "training"])
def test_an_a100_forward_counts_the_shared_memory_its_lowering_asks_for(dtype, kept):
    """An A100's D256 forward (the receipt's: 1024 causal queries, 8 heads over 4) at 64 by 64 two stages
    deep on mma.sync, inference and the 16-bit training path's: the model counts exactly the shared bytes
    its lowering asks for, the reductions' scratch and both barriers included, past the A100's limit, so no
    derivation that reads the model takes it."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    queries, width = 1024, 256
    spans = template.spans(queries, queries, 1, causal=True)
    with generation.assumed("sm80"):
        query = jnp.zeros((1, queries, 8, width), dtype)
        key = jnp.zeros((1, queries, 4, width), dtype)
        op, args = attention_mosaic._forward_call(query, key, key, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=kept, interpret=False)
        config = attention_mosaic.Tiles(64, 64, 2, 1, "mma.sync", narrow=True, relaunch_k=64,
                                        relaunch_consumers=1, relaunch_columns=128)
        target = generation.gpu_target()
        shared = op.staged_bytes(ba, config, target)
        (found,) = compile_mosaic_kernels([KernelCase("forward", lambda: op.lowering(ba, config))], "sm80")
        assert found["shared"] == shared
        assert shared > target.shared_limit


@COMPILES_ON_CPU
def test_an_a100_fused_backward_counts_the_shared_memory_its_lowering_asks_for():
    """An A100's fused backward at 64 by 64 two stages deep on mma.sync, whose adds read no stage
    (`Transport.adds_through_shared`), so its dK/dV output stage is scoped after its walks: the model counts
    exactly the shared bytes its binary takes, the stage within the walks' slots, not beside them, and no
    reduction scratch (the launch reduces across no warps)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    op, ba, _ = _gated_backward("sm80", 128, accurate=False)
    config = attention_mosaic.BackwardTiles(64, 64, 2, 1, fused=True, family="mma.sync")
    with generation.assumed("sm80"):
        shared = op.staged_bytes(ba, config, generation.gpu_target())
        (found,) = compile_mosaic_kernels([KernelCase("backward", lambda: op.lowering(ba, config))], "sm80")
    (binary,) = found["binaries"]
    assert binary["shared"] == shared


@COMPILES_ON_CPU
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
def test_an_a100_backward_256_wide_compiles_on_32_row_tiles_within_its_block(accurate):
    """An A100's backward of heads 256 wide (Gemma's) on 32 x 32 tiles, two launches on mma.sync arranged by
    their 32-row shares (`backward_family`), each register operand crossing its warps relayed through its
    consumer's tile (`_relay_tiles`): within the A100's block, as its model sizes it, spilling nothing, every
    launch's model ptxas's, lowered and compiled for an sm80 on this CPU, in each form."""
    import kernel_compile
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    op, ba, _ = _gated_backward("sm80", 256, accurate)
    config = attention_mosaic.BackwardTiles(32, 32, 2, 1, family="mma.sync")
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        assert op.staged_bytes(ba, config, target) <= target.shared_limit
        models = _backward_models(op, ba, config)
        found = compile_mosaic_kernels([KernelCase("backward", lambda: op.lowering(ba, config), models)],
                                       "sm80")
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
    assert set(models) <= kernel_compile.compiled_kernels(found[0]), (set(models), found[0].get("kernels"))


def _forward_kernels(op, ba, config) -> dict:
    """The forward's register model by each name a launch of it takes (`attention_mosaic_forward._launch`:
    `attention_{forward, training, peaks}`, then `_natural`, `_kept`, `_guarded`, `_columns{width}` and
    `_split` where it is so), each the op's (`MosaicForward.registers`): a peak walk's its own, a natural
    launch's at its own tiles (`Tiles.relaunch`), a launch over a block of the values' columns at that
    block's width."""
    family = attention_mosaic_program.bundle_of(config, jnp.bfloat16)
    found = {}
    for natural, tiles in (("", config), ("_natural", config.relaunch)):
        for kind in ("forward", "training", "peaks"):
            for columns in (None, 64, 128, 256):
                registers = op.registers(ba, tiles, peaks=kind == "peaks", columns=columns)
                model = resources.register_model(family, tiles.consumers, registers)
                named = "" if columns is None else f"_columns{columns}"
                found.update({f"attention_{kind}{natural}{kept}{guarded}{named}{split}": model
                              for kept in ("", "_kept") for guarded in ("", "_guarded")
                              for split in ("", "_split")})
    return found


@COMPILES_ON_CPU
@pytest.mark.parametrize("stored", ["bfloat16", "int8"])
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("generation_name", GATED)
def test_a_decode_step_compiles_spill_free_at_its_plans_tiles(generation_name, width, stored):
    """A decode step (`attention_decode.attend`: its plain launch at the plan's streaming key tile and
    slices, `MosaicForward.streaming`, and its masked rewalk's launches), 32 query heads over 8 key heads
    `width` wide, two rows of 4096 keys, its cache bf16 or int8 codes with their scales (each slot converted
    in registers, `_Codes`), lowered and compiled by ptxas for `generation_name` on this CPU
    (tests/kernel_compile.py): every launch within its shared memory and spilling nothing (`mosaic_failures`),
    where the forward gate's prefill calls derive their own tiles."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_decode

    rows, keys = 2, 4096
    query = jax.ShapeDtypeStruct((rows, 1, 32, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, keys, 8, width), jnp.dtype(stored))
    scale = jax.ShapeDtypeStruct((rows, 8, keys), jnp.float32)
    quantized = stored != "bfloat16"
    spans = template.spans(1, keys, rows, lengths=jnp.asarray([keys, keys - 77], jnp.int32))

    def decode(query, key, value, *scales):
        return attention_decode.attend(query, key, value, spans, scales=scales or None)

    operands = (query, key, key, *((scale, scale) if quantized else ()))
    with generation.assumed(generation_name):
        # The plan's op over the packed rows, its launches' models (`_forward_kernels`).
        zeros = [jnp.zeros(array.shape, array.dtype) for array in operands]
        scales = tuple(zeros[3:]) or None
        rows, row_spans, _ = attention_decode.packed(zeros[0], spans, key.shape[2])
        op, args = attention_mosaic._forward_call(rows, *zeros[1:3], row_spans, None, scale=None,
                                                  natural=False, out_dtype=jnp.float32, masked=True,
                                                  scales=scales)
        ba = op.bind(*args, residuals=template.Kept.OUTPUT, interpret=False)
        config = attention_decode.plan(rows, *zeros[1:3], row_spans, None, scale=None,
                                       target=generation.gpu_target(), scales=scales).config
        found = compile_mosaic_kernels(
            [KernelCase("decode", lambda: (decode, operands), _forward_kernels(op, ba, config))],
            generation_name)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width", "training", "hooks"), FORWARD_GATE)
def test_the_forward_compiles_spill_free_as_its_model_decides(generation_name, width, training, hooks):
    """The forward, the training forward where `training`, at its derived tiles of a causal call `width`
    wide under the score hook `hooks` names, lowered and compiled by ptxas for `generation_name` on this CPU
    (tests/kernel_compile.py): every launch within its shared memory, spilling nothing, and its register
    model's decision ptxas's (`mosaic_failures`)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), jnp.bfloat16)
    hooked = _gated_hooks(hooks, queries, width)

    def forward(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, hooked, residuals=training)

    with generation.assumed(generation_name):
        zeros = [jnp.zeros(array.shape, array.dtype) for array in (query, key, key)]
        op, args = attention_mosaic._forward_call(*zeros, spans, hooked, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=training, interpret=False)
        config = op.derived(ba, generation.gpu_target())
        assert config is not None, f"the Mosaic forward derives no config {width} wide"
        found = compile_mosaic_kernels(
            [KernelCase("forward", lambda: (forward, (query, key, key)), _forward_kernels(op, ba, config))],
            generation_name)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("generation_name", ["sm90"])
def test_an_fp32_forward_compiles_spill_free_as_its_model_decides(generation_name, width):
    """An fp32 call's inference forward (`MosaicForward.parts`: its operands bf16 parts) at its derived tiles
    of a causal call `width` wide, lowered and compiled by ptxas for `generation_name` on this CPU: every
    launch within its shared memory, spilling nothing, its register model's decision ptxas's."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), jnp.float32)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), jnp.float32)

    def forward(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, None)

    with generation.assumed(generation_name):
        zeros = [jnp.zeros(array.shape, array.dtype) for array in (query, key, key)]
        op, args = attention_mosaic._forward_call(*zeros, spans, None, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        ba = op.bind(*args, residuals=template.Kept.NONE, interpret=False)
        config = op.derived(ba, generation.gpu_target())
        assert config is not None, f"the Mosaic forward derives no fp32 config {width} wide"
        found = compile_mosaic_kernels(
            [KernelCase("forward", lambda: (forward, (query, key, key)), _forward_kernels(op, ba, config))],
            generation_name)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


@INTERPRETS
@pytest.mark.parametrize("call", ["full", "causal"])
def test_a_deterministic_backward_in_one_launch_holds_check_1_interpreted(call, with_deterministic_ops):
    """The backward under --xla_gpu_deterministic_ops on a full or causal call, its one launch's adds each
    in its turn (`Family.ticketed`: dQ's in its key tiles' order, grouped dK's and dV's in their query
    heads'), its gradients the reference's (check 1), under the interpreter, whose programs run in turn."""
    _check_backward(call, SHAPES["interpreted"], form=None, interpret=True)


@COMPILES_ON_CPU
@pytest.mark.parametrize("width", [64, 128])
def test_a_deterministic_backward_compiles_spill_free_as_its_model_decides(width, with_deterministic_ops):
    """The backward under --xla_gpu_deterministic_ops at its derived tiles of a causal call `width` wide,
    its adds ordered across programs by semaphores (`Family.ticketed`), lowered and compiled by ptxas for an
    H100 on this CPU: within its shared memory, spilling nothing, its register model's decision ptxas's."""
    test_the_backward_compiles_spill_free_as_its_model_decides(generation_name="sm90", width=width,
                                                               accurate=False, hooks="none")


@COMPILES_ON_CPU
@pytest.mark.parametrize(("generation_name", "width", "accurate", "hooks"), BACKWARD_GATE)
def test_the_backward_compiles_spill_free_as_its_model_decides(generation_name, width, accurate, hooks):
    """The backward at its derived tiles of a causal call `width` wide in the form `accurate` names under the
    score hook `hooks` names, lowered and compiled by ptxas for `generation_name` on this CPU
    (tests/kernel_compile.py): every launch within its shared memory, spilling nothing, and its register
    model's decision ptxas's (`mosaic_failures`, each launch's model `resources.register_model` at its own
    layout and consumers, `MosaicBackward._launch_registers`)."""
    import kernel_compile
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 512
    spans = template.spans(queries, queries, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, queries, 4, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, queries, 2, width), jnp.bfloat16)
    residuals = _residuals(query.shape)

    hooked = _gated_hooks(hooks, queries, width)

    def backward(query, key, value, dout, residuals):
        grads = attention_mosaic.backward(query, key, value, spans, hooked, dout, residuals=residuals,
                                          accurate=accurate)
        return grads.query, grads.key, grads.value

    op, ba, config = _gated_backward(generation_name, width, accurate, hooks)
    assert config is not None, f"the Mosaic backward derives no config {width} wide on {generation_name}"
    with generation.assumed(generation_name):
        models = _backward_models(op, ba, config)
        shapes = jax.tree.map(lambda array: jax.ShapeDtypeStruct(array.shape, array.dtype), residuals)
        found = compile_mosaic_kernels(
            [KernelCase("backward", lambda: (backward, (query, key, key, query, shapes)), models)],
            generation_name)
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
    # Every launch's model names a kernel it compiled: a launch renamed past its model is compared with none.
    assert set(models) <= kernel_compile.compiled_kernels(found[0]), (set(models), found[0].get("kernels"))


def _residuals(shape) -> tuple:
    """A training forward's residuals' shapes for a call of `shape` `[B, Q, H, D]`, zeros (`backward` binds
    its arguments only)."""
    batch, queries, heads, width = shape
    stats = jnp.zeros((batch, heads, queries), jnp.float32)
    return (stats, stats, jnp.zeros((batch, heads, queries), jnp.int32),
            template.Split(jnp.zeros((batch, heads, queries, width), jnp.float32), stats, stats, stats))


def _backward_models(op, ba, config) -> dict:
    """Each backward launch's register model by its kernel's name (`attention_dq`, `attention_dkv`, a width
    share's `_columnsN`, the one launch's `attention_dkv_fused`, or `attention_dkv_shared` over shared
    keys), at its own layout and consumers (`BackwardTiles.dkv`)."""
    dq, dkv = op._launch_registers(ba, config)
    if config.fused:
        family = attention_mosaic_program.bundle_of(config, jnp.bfloat16)
        name = "attention_dkv_shared" if config.shared_keys else "attention_dkv_fused"
        return {name: resources.register_model(family, config.consumers, dkv)}
    models = {}
    for kind, tiles, registers in (("dq", config, dq), ("dkv", config.dkv, dkv)):
        family = attention_mosaic_program.bundle_of(tiles, jnp.bfloat16)
        model = resources.register_model(family, tiles.consumers, registers)
        names = ([f"attention_{kind}"] if config.columns == 1
                 else [f"attention_{kind}_columns{share}" for share in range(config.columns)])
        models.update(dict.fromkeys(names, model))
    return models
