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
import contextvars
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
from generation_support import INTERPRETS, lowers
from jax.sharding import AbstractDevice, AbstractMesh, AxisType, Mesh
from reference_error import assert_as_exact, widened
from test_attention_template import CASES, ROWS, RUN_CASES, arguments, geometry as core_geometry
from test_attention_triton import BACKWARD_SEEDS, HOOKS, SCORE_HOOKS, gate_of, hooks_for, score_hooks

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_mosaic_forward,
    attention_mosaic_program,
    attention_portable,
    attention_quantized,
    attention_route,
    attention_triton,
    bundles,
    mosaic_sm90,
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
WGMMA = mosaic_sm90.WGMMA.name
"""An H100's warpgroup MMA layout with a producer, the tiles' family where a site states its own."""
INTERPRETED_ON = {"sm90": WGMMA, "sm80": "mma.sync"}
"""The GPUs an interpreted cell runs as (`_device`), each family's own walk: an H100's warpgroup MMA and an
A100's mma.sync (Main 11-08: an interpreted check of one family alone left the A100's backward unchecked on
a CPU). A cell of one family's mechanism names its GPU itself; the others run on each (`ON_EACH_FAMILY`)."""
_INTERPRETED = contextvars.ContextVar("interpreted_on", default="sm90")
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
WGMMA_FORMS = ("fused", "fused-keys-first", "fused-shared-keys", *RAISED_FORMS)
"""The backward's forms on an H100's layouts alone: its one launch, whose tile of the call forms an A100's
registers hold at no extent (64 x 64 holds 258 a thread, 32 x 128 in two column halves 258, et-a100bwd2),
and `FED`'s fed consumers, its shared keys and dQ, which an A100's mma.sync, one warpgroup, takes none of."""
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
    producer adding it (`fused-producer-dq`); or the fast form at its derivation's tiles (`derived`)."""
    family = INTERPRETED_ON[_INTERPRETED.get()]
    if family != WGMMA:
        # An A100's forms: its own layout's two launches, the accurate and the derived.
        if form in WGMMA_FORMS:
            raise ValueError(f"{form} is an H100's layout's form alone")
        if form in ("default", "accurate", "derived"):
            # Its derived tiles, which its registers hold (a 64 x 64 tile of the call forms' at one column
            # block holds 280 a thread, past its 255).
            return form == "accurate", None
        # Its layout's narrow form in two column halves (`BackwardTiles.narrow`).
        return True, attention_mosaic.BackwardTiles(64, 64, 2, 1, columns=2, family=family, narrow=True)
    if form == "fused":
        return False, attention_mosaic.BackwardTiles(64, 64, 2, 1, fused=True, family=WGMMA)
    if form == "fused-shared-keys":
        return False, attention_mosaic.BackwardTiles(64, 64, 2, 2, fused=True, family=FED["dkv_family"],
                                                     shared_keys=True)
    if form in ("fused-shared-dq", "fused-producer-dq"):
        return False, attention_mosaic.BackwardTiles(64, 64, 2, 1, fused=True, dkv_family="wgmma",
                                                     dkv_consumers=2, shared_dq=True, staged_logits=True,
                                                     producer_dq=form == "fused-producer-dq", family=WGMMA)
    if form == "fused-keys-first":
        return False, attention_mosaic.BackwardTiles(128, 64, 2, 1, fused=True, family=WGMMA, **FED)
    if form == "accurate":
        # Its derivation's: two launches, its dK/dV programs' column shares its dQ's two sums' choice.
        return True, None
    if form == "derived":
        return False, None
    columns = 2 if form == "accurate-halves" else 1
    return form != "default", attention_mosaic.BackwardTiles(64, 64, 2, 1, columns=columns, family=WGMMA,
                                                             **FED)


def _form_tiles(generation_name: str | None = None) -> attention_mosaic.Tiles:
    """The call forms' tiles on `generation_name`'s family (the interpreted cell's where None,
    `INTERPRETED_ON`): the template core's 100 queries over 96 keys (`test_attention_template.CASES`) in a
    pair of query blocks over two key tiles, so a block's spans begin and end within the keys it streams; an
    H100's pair two consumers' blocks, an A100's one consumer's each (mma.sync's one warpgroup)."""
    family = INTERPRETED_ON[generation_name or _INTERPRETED.get()]
    return attention_mosaic.Tiles(64, 64, 2, 2 if family == WGMMA else 1, family=family)
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


ON_EACH_FAMILY = pytest.mark.parametrize("interpreted_on", list(INTERPRETED_ON), indirect=True)
"""An interpreted cell run once a GPU of `INTERPRETED_ON`, as its family's own walk (`interpreted_on`); a cell
of one family's mechanism (a TPU core's, an H100's three consumers or pinned layouts, a B200's tcgen05 chunks,
an fp32 call's bf16 parts, which an A100's pipeline loads in no pieces, row 4475) names its GPU itself."""


def on_each_family_for(names: str | tuple[str, ...], values, ids=None) -> pytest.MarkDecorator:
    """`ON_EACH_FAMILY` crossed with `values` of the parameters `names`, one of them `form`: the cells each
    GPU's family takes, a backward form of an H100's layouts alone (`WGMMA_FORMS`) on its family alone."""
    names = (names,) if isinstance(names, str) else tuple(names)
    values = list(values)
    if callable(ids):
        labels = ["-".join(ids(each) for each in (value if len(names) > 1 else (value,))) for value in values]
    else:
        labels = [str(value) for value in values] if ids is None else list(ids)
    at = names.index("form")
    cells = [(generation_name, *(value if len(names) > 1 else (value,)), f"{generation_name}-{label}")
             for generation_name in INTERPRETED_ON for value, label in zip(values, labels, strict=True)
             if INTERPRETED_ON[generation_name] == WGMMA
             or (value[at] if len(names) > 1 else value) not in WGMMA_FORMS]
    return pytest.mark.parametrize(("interpreted_on", *names), [cell[:-1] for cell in cells],
                                   ids=[cell[-1] for cell in cells], indirect=["interpreted_on"])


@pytest.fixture(autouse=True)
def interpreted_on(request):
    """The GPU this cell runs as interpreted (`_device`, `_form_tiles`, `_backward_form`): its parameter's,
    an H100 where it has none (`on_each_family_for` generates no cell of a form its family lacks)."""
    generation_name = getattr(request, "param", "sm90")
    token = _INTERPRETED.set(generation_name)
    yield generation_name
    _INTERPRETED.reset(token)


def _device(interpret: bool, target: generation.GpuTarget | None = None):
    """The device a site's launches are sized and run for: interpreted on the CPU, which has no instruction
    family of its own, `target`'s facts and bundles, run where the process places them (where None, the GPU
    the interpreted cell runs as, `INTERPRETED_ON`; `generation.assumed`'s `interpreted`); the process's own
    compiled."""
    return (generation.assumed(target or _INTERPRETED.get(), interpreted=True) if interpret
            else contextlib.nullcontext())


def _pinned(forward: attention_mosaic.Tiles | None = None,
            backward: attention_mosaic.BackwardTiles | None = None):
    """The launches pinned to `forward`'s and `backward`'s configs where given (`tuning.pinned`), each
    else at the cost model's pick."""
    pins = {attention_mosaic.MosaicForward: forward, attention_mosaic.MosaicBackward: backward}
    return tuning.pinned({op: config for op, config in pins.items() if config is not None})


def _check_forward(call: str, dtype, shape, *, natural: bool, interpret: bool, width: int = 128,
                   value_width: int = 128, tiles: attention_mosaic.Tiles | None = None,
                   training: template.Kept = template.Kept.NONE, target: generation.GpuTarget | None = None,
                   drop: bool = False, query_scale: float = 1.0, sinks: bool = False,
                   rounds: float | None = None):
    """The forward, or with `training` (what it keeps, `template.Kept`) the training forward's output, over
    the call form `call` of `shape`,
    its query and key heads `width` wide and its value heads `value_width`, at `tiles` (the default when
    None), with `drop` under probability dropout at `DROP_RATE` (`_dropping`), with `sinks` a sink a query
    head (`template.sinks`, the initial state), its queries drawn times `query_scale` (a power of two,
    exact), with `rounds` its weights rounded to a quantized model's codes (`template.Hooks.rounds`)."""
    spans = template.spans(shape[1], shape[1], shape[0], **CALLS[call])
    hooks = _dropping(shape) if drop else None
    if rounds is not None:
        hooks = template.Hooks(rounds=rounds)
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


def _check_spans(label: str, spans: template.KeySpans, shape, *, interpret: bool, backward: bool = False,
                 form: str = "default", score: tuple = (), gate: bool = False, dtype=jnp.bfloat16,
                 natural: bool = False, seeds: int | None = None, width: int = 64,
                 param_rows: int | None = None):
    """The forward, or the backward in the form `form` names (`_backward_form`), over `spans` at `_form_tiles`
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

    def ours(query, key, value, dout, params, initial):
        hooks = hooked(params, initial)
        # The forward at the call forms' tiles 64 wide, else at its derivation's (two consumers' hooked sums
        # 128 wide pass their registers, and a hooked training forward's natural walk theirs 64 wide).
        with _device(interpret), _pinned(_form_tiles() if width == 64 and not (backward and score) else None,
                                         config):
            if not backward:
                return attention_mosaic.forward(query, key, value, spans, hooks, natural=natural,
                                                interpret=interpret)
            # The backward from this mainloop's own training forward's residuals.
            residuals = attention_mosaic.forward(query, key, value, spans, hooks, natural=natural,
                                                 residuals=template.Kept.SPLIT, interpret=interpret)[1:]
            # On the mainloop the route takes it to (`attention_route.backward`): this one's, else the
            # Pallas-Triton one's where this one's registers hold no tile of it.
            grads = attention_route.backward(query, key, value, spans, hooks, dout, residuals=residuals,
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
            return (*found[1:4], *hooks)
        assert traced_terms is not None, "a hooked call's parameters' terms trace its reference"
        with jax.enable_x64(new_val=True):
            _, traced = traced_terms(*widened((query, key, value, dout, *hooks)))
        return (*found[1:4], *traced[3:])

    return terms


@INTERPRETS
@on_each_family_for("form", ["default", "fused"])
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_from_the_kept_output_is_on_par_interpreted(call, form):
    """The backward from a training forward that keeps its output for D (`template.Kept.OUTPUT`: the inference
    walk's rows, no anchor, `rowsum(dO ⊙ O)` of the rounded output, as FlashAttention's is), in two launches
    and in one: within check 1 of the reference from float64, its D's rounding judged a term at a time."""
    _check_backward(call, SHAPES["interpreted"], form=form, interpret=True, kept=template.Kept.OUTPUT)


def test_a_slope_reaches_a_custom_vjp_only_through_the_operands_it_feeds():
    """A score hook's slope along an input counts as reaching a `jax.custom_vjp` only where that input
    feeds one, through a jit's program operand by operand (`_reaches_custom_vjp`): a score added to a
    custom VJP of a parameter has the constant slope one, traced, and the parameter's slope is counted."""
    from dew.nn.kernels.templates.attention_mosaic_program import _reaches_custom_vjp

    @jax.custom_vjp
    def held(x):
        return x

    held.defvjp(lambda x: (x, None), lambda _, cotangent: (cotangent,))
    traced = jax.make_jaxpr(lambda s, p: jax.jit(lambda s, p: s + held(p))(s, p))(1.0, 2.0).jaxpr
    assert not _reaches_custom_vjp(traced, {traced.invars[0]})
    assert _reaches_custom_vjp(traced, {traced.invars[1]})
    # A jit's outputs reached one by one: its second output, independent of `s`, feeds the custom VJP.
    paired = jax.make_jaxpr(lambda s, p: (lambda pair: pair[0] + held(pair[1]))(
        jax.jit(lambda s, p: (s, p))(s, p)))(1.0, 2.0).jaxpr
    assert not _reaches_custom_vjp(paired, {paired.invars[0]})
    assert _reaches_custom_vjp(paired, {paired.invars[1]})
    # No tangent passes a stop_gradient.
    stopped = jax.make_jaxpr(lambda s: held(jax.lax.stop_gradient(s)))(1.0).jaxpr
    assert not _reaches_custom_vjp(stopped, {stopped.invars[0]})


def test_a_hook_reads_its_places_as_a_vector_only_where_every_run_of_it_reads_them():
    """A score hook's places are a vector broadcast only where it surely reads them (`score_places`): through
    a jitted helper input by input, and never through a loop's body, which may run no step (a scan of none)
    or read its carries a step later, an unread vector's iota finding no layout (row 2977); a tile of
    places serves every hook (row 4025: a call read where every reader reads it, `dew.nn.programs.called`)."""
    def helper(score, q_pos, k_pos):
        return jax.jit(lambda score, places: score + places.astype(score.dtype))(score, q_pos)

    def looped(score, q_pos, k_pos):
        return jax.lax.scan(lambda carry, _: (carry + q_pos.astype(carry.dtype), None), score, None,
                            length=0)[0]

    assert attention_mosaic_program.score_places(template.Hooks(helper)) == (True, False)
    assert attention_mosaic_program.score_places(template.Hooks(looped)) == (False, False)


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
        grads = attention_portable.backward(attention_portable.Tiles(64, 64, 64), query, key, value, spans,
                                            None, dout, residuals=residuals)
        return grads.query, grads.key, grads.value

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans), query, key, value)
        return pull(dout)

    assert_as_exact(jax.jit(ours), jax.jit(reference), _drawn(shape, jnp.bfloat16, 128),
                    "plain-JAX backward from the kept output", seeds=SEEDS, terms=_gradient_terms(spans))


@ON_EACH_FAMILY
@INTERPRETS
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_forward_is_on_par_interpreted(call, dtype):
    """The forward, the base-2 fold, under Mosaic GPU's interpreter."""
    _check_forward(call, dtype, SHAPES["interpreted"], natural=False, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("form", ["fused", "fused-shared-keys"])
@pytest.mark.parametrize("call", CALLS)
def test_an_fp16_backward_in_one_launch_holds_check_1_interpreted(call, form):
    """An fp16 call's backward in the fast form's one launch (row 3886): its weights and score cotangents as
    two fp16 parts at each walk's running scale (`algebra.running_parts`), dV's and dK's accumulators taken
    to the new scale as it grows and back at the end, each tile's dQ by its own; every gradient the
    reference's (check 1), under the interpreter: each consumer its own keys, heads 64 wide, or two sharing
    a key block, each its share's exponents exchanged before its parts (`attention_mosaic_coop`), heads 128
    wide, where each launch's registers hold fp16's parts. (A dropping fp16 call takes no Mosaic walk at
    all: its dropping walk takes bf16 operands.)"""
    _check_backward(call, SHAPES["interpreted"], form=form, interpret=True, dtype=jnp.float16,
                    width=FORM_WIDTHS.get(form, 64))


TPU_PASSES = {"fold": {"natural": False}, "natural": {"natural": True},
              "sinks": {"natural": False, "sinks": True},
              "training": {"natural": False, "training": template.Kept.SPLIT}}
"""The forward passes a TPU core's launches run (`mosaic_tpu.MXU`): the base-2 fold, the natural form, an
initial state, and the training forward's output beside its residuals."""


@INTERPRETS
@pytest.mark.parametrize("forward", TPU_PASSES)
@pytest.mark.parametrize("call", CALLS)
def test_a_tpu_cores_forward_is_on_par_interpreted(call, forward):
    """The forward on a v5e's MXU (`mosaic_tpu.MXU`), the one mainloop's launches through the family's
    seams (its staging, reads, grid coordinates and turns), in each pass, under Pallas TPU's interpreter: on
    par with the reference (check 1)."""
    _check_forward(call, jnp.bfloat16, SHAPES["interpreted"], interpret=True,
                   target=generation.gpu_target("v5e"), **TPU_PASSES[forward])


@INTERPRETS
@pytest.mark.parametrize("call", CALLS)
def test_a_tpu_cores_backward_is_on_par_interpreted(call):
    """The backward on a v5e's MXU, its fused launch's dQ adds each program's in turn on the core's one
    TensorCore (`mosaic_tpu.VmemRings.added`), under Pallas TPU's interpreter: every gradient on par with
    the reference (check 1)."""
    _check_backward(call, SHAPES["interpreted"], form=None, interpret=True,
                    target=generation.gpu_target("v5e"))


@ON_EACH_FAMILY
@INTERPRETS
@pytest.mark.parametrize("call", CALLS)
def test_a_dropping_backward_is_on_par_over_every_call_interpreted(call):
    """The default backward of each call that drops, the dropout multipliers an operand of its derived tile
    cotangents: on par with the reference (check 1)."""
    _check_backward(call, SHAPES["interpreted"], form="default", interpret=True, drop=True)


@ON_EACH_FAMILY
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


@ON_EACH_FAMILY
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


@ON_EACH_FAMILY
@INTERPRETS
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_natural_forward_is_on_par_interpreted(call, dtype):
    """The forward with natural logits under the interpreter, bf16's under its headroom guard
    (`algebra.walked_launches`)."""
    _check_forward(call, dtype, SHAPES["interpreted"], natural=True, interpret=True)


@INTERPRETS
@on_each_family_for("form", CASES)
def test_the_forward_is_on_par_over_every_call_form_interpreted(form):
    """The forward over each of the template core's call forms under the interpreter."""
    _check_call_form(form, interpret=True)


@ON_EACH_FAMILY
@INTERPRETS
@pytest.mark.parametrize(("width", "value_width"), HEAD_WIDTHS)
def test_the_forward_is_on_par_at_head_widths_interpreted(width, value_width):
    """The forward of query heads `width` wide and value heads `value_width` over a causal call at
    `_form_tiles`, under the interpreter."""
    _check_forward("causal", jnp.bfloat16, SHAPES["interpreted"], natural=False, interpret=True, width=width,
                   value_width=value_width, tiles=_form_tiles())


@ON_EACH_FAMILY
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


@ON_EACH_FAMILY
@INTERPRETS
@pytest.mark.parametrize(("width", "value_width"), HEAD_WIDTHS)
def test_the_backward_is_on_par_at_head_widths_interpreted(width, value_width):
    """The backward in its derived form of query heads `width` wide and value heads `value_width` over a
    causal call, under the interpreter: dK and dV each at its own head width."""
    _check_backward("causal", SHAPES["interpreted"], form=None, interpret=True, width=width,
                    value_width=value_width)


DROPPING_FORWARDS = [(call, training, jnp.bfloat16) for call in CALLS
                     for training in (template.Kept.NONE, template.Kept.SPLIT)]
DROPPING_FORWARDS += [("causal", training, jnp.float16)
                      for training in (template.Kept.NONE, template.Kept.SPLIT)]
"""The dropping forward's cases: each call form, plain and training, over bf16; over fp16 a causal call's,
whose dropped weights narrow as they are (`algebra.dropped_parts`), the walk bf16's otherwise."""


@ON_EACH_FAMILY
@INTERPRETS
@pytest.mark.parametrize(("call", "training", "dtype"), DROPPING_FORWARDS,
                         ids=[f"{call}-{training.name}-{jnp.dtype(dtype).name}"
                              for call, training, dtype in DROPPING_FORWARDS])
def test_the_forward_is_on_par_dropping_interpreted(call, training, dtype):
    """The forward and the training forward under probability dropout, its draw the reference's bit for bit
    (`template.Draw`), over bf16 or fp16 operands, fp16's dropped weights narrowed as they are and summed in
    place (`algebra.dropped_parts`), under the interpreter."""
    _check_forward(call, dtype, SHAPES["interpreted"], natural=False, interpret=True, training=training,
                   drop=True)


@INTERPRETS
@on_each_family_for(("form", "dtype"),
                    [*((form, jnp.bfloat16) for form in (*HOOKED_FORMS, "fused")), ("derived", jnp.float16)],
                    ids=lambda value: value if isinstance(value, str) else jnp.dtype(value).name)
def test_the_backward_is_on_par_dropping_interpreted(form, dtype):
    """The backward in each form under probability dropout over a causal call, the fast form's one launch
    too (each consumer's tile its own draw), its D the forward's dropped split sums' about the anchor's
    value as its draw keeps it (`algebra.anchor_multipliers`), under the interpreter; over fp16 operands at
    its derivation's tiles."""
    _check_backward("causal", SHAPES["interpreted"], form=form, interpret=True, drop=True, dtype=dtype)


@ON_EACH_FAMILY
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
        with _device(interpret=True), _pinned(_form_tiles(), config):
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
@on_each_family_for("form", HOOKED_FORMS)
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
@on_each_family_for("form", HOOKED_FORMS)
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
@on_each_family_for("form", HOOKED_FORMS)
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
@on_each_family_for(("queries", "form"), PADDED_CALLS.values(), ids=PADDED_CALLS)
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
    with _device(interpret=True), _pinned(backward=config):
        residuals = attention_mosaic.forward(query, key, value, spans, hooks, residuals=template.Kept.SPLIT,
                                             interpret=True)[1:]
        grads = attention_route.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                         accurate=accurate, interpret=True)
    _, pull = jax.vjp(lambda q, k, v, t, b: template.reference(q, k, v, spans, hooked(t, b)), query, key,
                      value, *params)
    for found, expected in zip((grads.query, grads.key, grads.value, *grads.score_params), pull(dout),
                               strict=True):
        np.testing.assert_array_equal(np.asarray(found, np.float32), np.asarray(expected, np.float32))


@INTERPRETS
def test_the_training_forwards_rows_past_fp32s_largest_keep_their_mean_and_every_other_row_interpreted():
    """A call as many queries as heads, one consumer's tile of them, whose head 0 sees two keys of equal
    logits past the fold's reach over values near bf16's largest: the call walks again unfolded, and head
    0's rows, whose weighted values' sum would pass fp32's largest, hold it at their scale in that walk
    (`algebra.SCALED_WALK`), their residuals at it. Head 0's output is its values' mean exactly, its one
    value row (where the reference's fp32 sums overflow); every other head's rows are the bits they take
    where head 0's values are ordinary (a row's scale its own: the scaled op's kernel,
    `MosaicForward.scaled`, takes them through its scaled arithmetic at the scale `2^0`, the unscaled
    kernel's bits at one config, the fed mma.sync's both take, `tuning.pinned_where_taken`: the scaled
    walk's small weights' parts hold a warpgroup MMA consumer's registers past a thread's); and the
    backward from these residuals gives head 0's values a gradient wherever the reference does, finite
    where it is."""
    queries, keys, width = 64, 2, 64
    spans = template.spans(queries, keys, 1)
    streams = jax.random.split(jax.random.key(0), 4)
    query, key, value, dout = (jax.random.normal(stream, (1, count, queries, width), jnp.float32)
                               for stream, count in zip(streams, (queries, keys, keys, queries), strict=True))
    # Head 0's logits, 128 * 128 * 64 / 8, past the fold's reach; its values overflow a sum of two or not.
    query, key = (array.at[:, :, 0].set(128.0) for array in (query, key))
    query, key, dout = (array.astype(jnp.bfloat16) for array in (query, key, dout))
    tiles = attention_mosaic.Tiles(64, 32, 2, 1, family=bundles.MmaFed.name, columns=32)

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


@pytest.mark.parametrize("held", ["keys", "start"])
def test_a_training_forwards_output_scales_exactly_with_its_values_however_large(held):
    """A training forward's head whose rows see 512 keys, the row's largest magnitude near fp32's largest
    under a weight near fp32's least normal at the row's maximum (`held`: a key's value, its logit 86.5
    below the row's peak; or a resumed state's sums, its maximum 120 binades below it), every other key
    weighing nothing or holding zeros: the row's scale (`algebra.SCALED_WALK`) is a power of two, and a
    weight it takes below fp32's least normal enters the small weights' product raised (`algebra.at_scale`),
    so its small weight keeps its bits. Its output is exactly 2^64 times the call's over its values and sums
    at 2^-64."""
    queries, keys, width = 64, 512, 64
    spans = template.spans(queries, keys, 1)
    streams = jax.random.split(jax.random.key(1), 3)
    query, key, value = (jax.random.normal(stream, (1, count, 2, width), jnp.float32)
                         for stream, count in zip(streams, (queries, keys, keys), strict=True))
    # Head 0's logits at a scale of 1/8: 128 at key 0, 41.5 at key 1, -72 at every other key.
    query = query.at[:, :, 0].set(0.0).at[:, :, 0, 0].set(16.0)
    key = key.at[:, :, 0].set(0.0).at[:, :, 0, 0].set(-36.0)
    key = key.at[:, 0, 0, 0].set(64.0).at[:, 1, 0, 0].set(20.75)
    value = value.at[:, :, 0].set(0.0)
    if held == "keys":
        value = value.at[:, 1, 0].set(3e38)
    query, key, value = (array.astype(jnp.bfloat16) for array in (query, key, value))
    # Head 0's resumed sums at 2^127 under a maximum 120 binades below its keys' (head 1's at 0, none).
    m = jnp.zeros((1, 2, queries)).at[:, 0].set(128.0 - 120 * np.log(2.0))
    acc = jnp.zeros((1, 2, queries, width)).at[:, 0].set(2.0 ** 127 if held == "start" else 0.0)
    tiles = attention_mosaic.Tiles(64, 64, 2, 1, family=WGMMA)

    def trained(scale):
        hooks = template.Hooks(initial=template.Initial(m, jnp.ones((1, 2, queries)), acc * scale))
        with _device(interpret=True), _pinned(tiles):
            out = attention_mosaic.forward(query, key, value * jnp.bfloat16(scale), spans, hooks,
                                           residuals=template.Kept.SPLIT, interpret=True)[0]
        return np.asarray(out[:, :, 0], np.float32)

    np.testing.assert_array_equal(trained(1.0), trained(2.0 ** -64) * np.float32(2.0 ** 64))


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


@pytest.mark.parametrize(("target", "width", "hooks", "queries"), [
    ("sm90", 64, ("softcap",), 512), ("sm90", 128, ("softcap",), 512), ("sm90", 128, ("alibi",), 512),
    ("sm100", 128, (), 512), ("sm100", 64, ("softcap",), 512), ("sm80", 128, ("softcap",), 512),
    ("sm120", 128, (), 512), ("sm80", 32, (), 48), ("sm86", 64, (), 33), ("sm90", 64, ("softcap",), 48),
    ("sm120", 128, (), 48)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
def test_a_scaled_forward_derives_wherever_its_common_one_does(target, width, hooks, queries, dtype):
    """A rare call's scaled op (`MosaicForward.scaled`) derives its own config, legal on the target with
    its natural rewalk where it folds (`MosaicForward.legal`), for every call whose common op derives one:
    its small weights' parts and their products priced in its register model (an H100's fp32 call under a
    softcap and a B200's fp32 call at width 128 derived none, their scaled walk's terms priced beside tiles
    they are not live with), narrower blocks of its value columns where a family stages them swizzled
    (`_column_blocks_of`), and a call of fewer queries than a role's at fewer of them a tile (a 48-query
    training walk's on mma.sync held no tile of 48, and its trace failed), so no call leaves the mainloop."""
    spans = template.spans(queries, queries, 1, causal=True)
    score_mod, kinds, draw = score_hooks(hooks, queries, queries, width, rows=1, heads=4)
    params, initial = draw(0)
    hooked = template.Hooks(score_mod, tuple(map(template.Param, params, kinds)), initial)
    arrays = [jnp.zeros((1, queries, heads, width), dtype) for heads in (4, 2, 2)]
    with generation.assumed(target):
        op, args = attention_mosaic._forward_call(*arrays, spans, hooked, scale=None,
                                                  natural=attention_mosaic.NATURAL)
        scaled = dataclasses.replace(op, scaled=True)
        devices = generation.gpu_target()
        for kept in (template.Kept.NONE, template.Kept.SPLIT):
            if op.derived(op.bind(*args, residuals=kept, interpret=False), devices) is None:
                continue
            ba = scaled.bind(*args, residuals=kept, interpret=False)
            config = scaled.derived(ba, devices)
            assert config is not None
            assert scaled.legal(ba, config, devices)


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


@pytest.mark.parametrize("accurate", [False, True], ids=["fast", "accurate"])
def test_a_backward_config_on_a_family_the_device_lacks_is_illegal(accurate):
    """An A100's backward at a config of an H100's families (wgmma, its dK/dV launch on wgmma.fed) is not
    legal (`MosaicBackward.legal`): the device has no bundle of them, so its legality refuses it, as the
    attention matrix's cells of two launches ask of every generation (tests/attention_matrix.py), and its
    derived config is legal there."""
    two = attention_mosaic.BackwardTiles(64, 64, 2, 1, family=WGMMA, dkv_family="wgmma.fed", dkv_consumers=2)
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        op, ba = _backward_call(SHAPES["interpreted"], 128, accurate=accurate)
        derived = op.derived(ba, target)
        assert derived is not None and op.legal(ba, derived, target)
        assert not op.legal(ba, two, target)


def test_a_pin_the_mosaic_program_cannot_take_is_illegal():
    """A config a pin or a stored pick names is held to the tiles a program takes (`whole_tiles`) as a
    derived one is: an 8-query tile under a family's 16- or 64-row products, or three consumers where the
    family's walks take two at most, is not legal, so a pin of it is refused before any launch."""
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        op, ba = _forward_call(SHAPES["interpreted"], 128, 128, residuals=template.Kept.NONE)
        derived = op.derived(ba, target)
        assert derived is not None and op.legal(ba, derived, target)
        rows, tokens, _, _ = SHAPES["interpreted"]
        arrays = [jnp.zeros(ba.arguments[name].shape, ba.arguments[name].dtype)
                  for name in ("query", "key", "value")]
        for config in (dataclasses.replace(derived, block_q=8), dataclasses.replace(derived, consumers=3)):
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
@on_each_family_for("form", ["default", "accurate"])
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
@on_each_family_for("form", BACKWARD_FORMS)
@pytest.mark.parametrize("case", CASES)
def test_the_backward_is_on_par_over_every_call_form_interpreted(case, form):
    """The backward in both forms over each of the template core's call forms under the interpreter."""
    _check_call_form(case, interpret=True, backward=True, form=form)


@INTERPRETS
@on_each_family_for("form", ["forward", *BACKWARD_FORMS])
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


@ON_EACH_FAMILY
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
@on_each_family_for("form", BACKWARD_FORMS)
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_is_on_par_interpreted(call, form):
    """The backward in every form under the interpreter, each at its width (`FORM_WIDTHS`)."""
    _check_backward(call, SHAPES["interpreted"], form=form, interpret=True, width=FORM_WIDTHS.get(form, 128))


@ON_EACH_FAMILY
@INTERPRETS
@pytest.mark.parametrize("accurate", [False, True], ids=["fast", "accurate"])
@pytest.mark.parametrize("call", CALLS)
def test_an_fp16_backward_is_on_par_interpreted(call, accurate):
    """fp16's backward in two launches under the interpreter (row 3497): its weights' and score cotangents'
    parts scaled (`product.Scaled`), each product its own accumulator restored into the carried dK, dV or
    dQ (`operands.Issuer.summed`), on par with the reference (check 1)."""
    _check_backward(call, SHAPES["interpreted"], form="accurate" if accurate else None, interpret=True,
                    dtype=jnp.float16)


@ON_EACH_FAMILY
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
@pytest.mark.parametrize("training", [template.Kept.NONE, template.Kept.SPLIT], ids=["plain", "training"])
@pytest.mark.parametrize("call", ["full", "causal"])
def test_the_forward_on_three_consumers_holds_check_1_interpreted(call, training):
    """The forward and the training forward on wgmma's three consumers a program (FlashAttention 3's 192-row
    tile, `Wgmma.consumers`), under the interpreter, on par with the reference (check 1)."""
    tiles = attention_mosaic.Tiles(64, 64, 2, 3, family=WGMMA)
    _check_forward(call, jnp.bfloat16, (1, 192, 2, 1), natural=False, interpret=True, width=64,
                   value_width=64, tiles=tiles, training=training)


@INTERPRETS
@pytest.mark.parametrize("training", [template.Kept.NONE, template.Kept.SPLIT], ids=["plain", "training"])
@pytest.mark.parametrize("call", ["full", "causal"])
def test_the_forward_on_tcgen05s_chunks_holds_check_1_interpreted(call, training):
    """The forward and the training forward on a B200's tcgen05 chunks (`mosaic_sm100.CHUNKS`: consumers of
    64 rows, three a program for the forward and two for the training forward, whose registers three do
    not hold, each product into its scoped tensor memory and read back into registers, the weights staged
    into the issuer's tile for the values' product), under the interpreter, which models tensor memory: on
    par with the reference (check 1)."""
    consumers = 3 if training == template.Kept.NONE else 2
    tiles = attention_mosaic.Tiles(64, 64, 2, consumers, family="tcgen05.chunks")
    _check_forward(call, jnp.bfloat16, (1, 192, 2, 1), natural=False, interpret=True, width=64,
                   value_width=64, tiles=tiles, training=training, target=generation.gpu_target("sm100"))


@pytest.mark.parametrize("training", [template.Kept.NONE, template.Kept.SPLIT], ids=["plain", "training"])
@pytest.mark.parametrize("consumers", [2, 3])
def test_the_forwards_schedule_holds_every_property_in_every_interleaving(consumers, training):
    """The forward's launch on wgmma at `consumers` a program over 192 queries and 128 keys, traced as an
    H100 builds it: every interleaving of its warpgroups and its loads' landings explored
    (`ring_protocol.explore`), no deadlock and every barrier's phase whole. Its consumers' schedule turns
    balance at any count (row 4840: at three the third made one fewer, and its partners' last turn waited
    on a phase that never completed, the interpreted hang)."""
    from ring_protocol import explore, programs

    query, key = jnp.ones((1, 192, 1, 64), jnp.bfloat16), jnp.ones((1, 128, 1, 64), jnp.bfloat16)
    spans = template.spans(192, 128, 1)
    tiles = attention_mosaic.Tiles(64, 64, 2, consumers, family=WGMMA)

    def forward(query, key):
        with generation.assumed("sm90"), tuning.pinned({attention_mosaic.MosaicForward: tiles}):
            return attention_mosaic.forward(query, key, key, spans, residuals=training, interpret=False)

    threads, arrivals = programs(forward, query, key, threads=consumers + 1)
    assert explore(threads, arrivals) > 0


@INTERPRETS
def test_a_backward_whose_dkv_launch_holds_more_keys_than_its_dq_launch_writes_every_key_interpreted():
    """The backward in two launches, the dK/dV launch's two consumers each their own key block beside one
    dQ consumer (`dkv_consumers=2` on a producer's layout), over 192 keys, under the interpreter: the dK/dV
    launch's last program holds a partial key block, and every key's gradients are written, on par with the
    reference (check 1)."""
    config = attention_mosaic.BackwardTiles(64, 64, 2, 1, dkv_family="wgmma", dkv_consumers=2, family=WGMMA)
    _check_backward("full", (1, 192, 2, 1), form=None, config=config, interpret=True, width=64)


@INTERPRETS
def test_a_causal_backward_whose_consumers_stage_shared_logits_walks_each_query_tile_interpreted():
    """The backward's dK/dV launch whose two consumers each hold their own causal key block and share a staged
    logits tile (`staged_logits`), over 128 causal queries and keys, under the interpreter: each consumer's
    query walk is its own block's, and the gradients are on par with the reference (check 1)."""
    config = attention_mosaic.BackwardTiles(64, 64, 2, 1, dkv_family="wgmma", dkv_consumers=2,
                                            staged_logits=True, family=WGMMA)
    _check_backward("causal", (1, 128, 2, 1), form=None, config=config, interpret=True, width=64)


HOPPER_HOOK_SITES = ("alibi", "resumed", "softcap-sinks-alibi")
"""The hooked backward's sites the H100 alone runs (`…_under_every_score_hook_on_an_h100`, compiled, at the
template site's shape and seeds): a head's ALiBi slope or a resumed state is one value a draw, so check 1
takes 128 draws at any head count, past the CPU suite's 240 s target interpreted (1.9 s a draw a row of two
heads: 270 to 650 s, row 1849); the CPU keeps every other hook, sinks and softcap among them."""


@INTERPRETS
@on_each_family_for("form", ["default", "accurate"])
@pytest.mark.parametrize("hooks", [hooks for hooks in BACKWARD_HOOKS if hooks not in HOPPER_HOOK_SITES])
def test_the_backward_is_on_par_under_every_score_hook_interpreted(hooks, form):
    """The backward in both forms under each score hook over causal packed documents under a window, under
    the interpreter: the gradients of the queries, keys and values, of each hook parameter and of the initial
    state."""
    _check_hooked_backward(hooks, form=form, interpret=True)


@ON_EACH_FAMILY
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


@ON_EACH_FAMILY
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
                    tiles=_form_tiles(), interpret=True)

        def reference(query, key, value, dout, scale=scale):
            del dout
            return template.reference(query, key, value, spans, scale=scale)

        assert_as_exact(jax.jit(ours), jax.jit(reference), _drawn(shape, jnp.bfloat16),
                        f"mosaic forward at scale {scale}", seeds=SEEDS)


@ON_EACH_FAMILY
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


@ON_EACH_FAMILY
@INTERPRETS
@pytest.mark.parametrize("call", CALLS)
def test_a_quantized_calls_weights_round_at_each_rows_maximum_interpreted(call):
    """A quantized model's call (`template.Hooks.rounds`, int8's bound, row 4858): the anchored training
    walk's weights rounded to their codes at each row's maximum in the value product, its denominator the
    weights themselves, the output on par (check 1) with the reference's, which rounds them so from its whole
    rows, under the interpreter."""
    _check_forward(call, jnp.bfloat16, SHAPES["interpreted"], natural=False, interpret=True, width=64,
                   value_width=64, training=template.Kept.SPLIT, rounds=attention_quantized.BOUND)


@ON_EACH_FAMILY
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


@ON_EACH_FAMILY
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
@on_each_family_for(("hooks", "form"),
                    [("softcap", "fused"), ("softcap", "fused-shared-keys"), ("sinks", "fused")])
def test_a_hooked_backward_in_one_launch_holds_check_1_interpreted(hooks, form, without_deterministic_ops):
    """The fast form's one launch queries first under a softcap, a score hook of no parameters, each
    consumer its own keys or its share of a shared key block (`attention_mosaic_coop`), its score
    cotangents taken back through the cap's slope (`gradient.pulled`) before dQ's and dK's products; or
    under sinks, the softmax's initial state, whose gradient the first pass's D and the keys' share of the
    weights give: every gradient the reference's (check 1) over causal packed documents under a window,
    under the interpreter, heads as wide as the form takes (`FORM_WIDTHS`)."""
    _check_hooked_backward(hooks, form=form, interpret=True, width=FORM_WIDTHS.get(form, 64))


@ON_EACH_FAMILY
@INTERPRETS
@pytest.mark.parametrize("hooks", ["key-bias", "tile-bias"])
def test_a_width_split_backward_writes_its_score_parameters_gradients_once_interpreted(hooks):
    """The accurate backward 128 wide, its sums in two column halves (`BackwardTiles.columns`), under a hook
    whose gradient the dK/dV programs write (a key's parameter, a tile's): the first half's program writes
    it, and every gradient is the reference's, under the interpreter."""
    _check_hooked_backward(hooks, form="accurate-halves", interpret=True, width=128)


def _lowered_call(call: str) -> tuple:
    """A lowering test's call form `call` as abstract operands: its spans, its query and key `[B, S, H, 128]`
    bf16 and a training forward's residuals' shapes, the compiled shape's causal call or a template core
    case's (`CASES`, `RUN_CASES`) over four query heads and two key heads."""
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
    anchors = jax.ShapeDtypeStruct((rows, heads, tokens), jnp.int32)
    return spans, q, kv, (rows_shape, rows_shape, anchors, split)


@lowers("sm90")
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("call", ["causal", "causal-documents-window", "runs-documents-valid"])
def test_the_launches_lower_for_an_h100(call, accurate):
    """The forward and the backward's launches lower for an H100 on any host, as the GPU gate runs them: a
    causal call, a spans one of documents, whose programs read its block map and ids, and one over a
    decision row's runs, whose programs read each run's bounds; the forward under every hook too, whose
    programs read each kind of parameter and the initial state."""
    spans, q, kv, residuals = _lowered_call(call)
    tokens = q.shape[1]
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


@pytest.mark.parametrize("call", ["causal", "causal-documents-window"])
def test_the_launches_lower_for_a_v5e(call):
    """The forward, the training forward and the backward on a v5e's MXU (`mosaic_tpu.MXU`, the route's Mosaic
    mainloop on a TPU) lower for a TPU on this CPU, Pallas TPU's Mosaic lowering, as a TPU compiles them: a
    causal call and a spans one of documents under a window, whose programs read its block map and ids."""
    spans, q, kv, residuals = _lowered_call(call)
    # An abstract mesh of one v5e device, which Pallas TPU's lowering reads its target from on any host.
    device = AbstractDevice(device_kind=generation.gpu_target("v5e").device_kind, num_cores=1, platform="tpu")
    mesh = AbstractMesh((1,), ("data",), axis_types=(AxisType.Auto,), abstract_device=device)
    with generation.assumed("v5e"), jax.sharding.use_abstract_mesh(mesh):
        for training in (template.Kept.NONE, template.Kept.SPLIT):
            entry = functools.partial(attention_mosaic.forward, spans=spans, hooks=None, residuals=training)
            jax.jit(entry).trace(q, kv, kv).lower(lowering_platforms=("tpu",))
        jax.jit(lambda q, k, v, dout, residuals: attention_mosaic.backward(
            q, k, v, spans, None, dout, residuals=residuals)).trace(q, kv, kv, q, residuals).lower(
            lowering_platforms=("tpu",))


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
        return attention_route.backward(q, k, v, spans, hooked, dout, residuals=residuals, accurate=accurate)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(backward_of).trace(q, kv, kv, q, residuals, params, initial).lower(
            lowering_platforms=("cuda",))


@lowers("sm90")
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("natural", [False, True], ids=["folded", "natural"])
@pytest.mark.parametrize("hooks", BACKWARD_HOOKS)
def test_the_training_forward_under_every_score_hook_lowers_for_an_h100(hooks, natural, width):
    """The training forward that supplies each hooked backward's residuals lowers for an H100 at the GPU
    cells' head widths, in either form: a row-indexed hook's unit column of its query places found no layout
    beside the walk's split on mma.sync (row 5279)."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    q = jax.ShapeDtypeStruct((ROWS, queries, 4, width), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((ROWS, keys, 2, width), jnp.bfloat16)
    score_mod, kinds, draw_hooks = score_hooks(SCORE_HOOKS[hooks], queries, keys, width)
    params, initial = draw_hooks(0)

    def trained(q, k, v, params, initial):
        hooked = template.Hooks(score_mod, tuple(template.Param(value, kind)
                                                 for value, kind in zip(params, kinds, strict=True)),
                                initial)
        return attention_mosaic.forward(q, k, v, spans, hooked, natural=natural,
                                        residuals=template.Kept.SPLIT)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(trained).trace(q, kv, kv, params, initial).lower(lowering_platforms=("cuda",))


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
    `_form_tiles`, compiled on an H100."""
    _check_forward("causal", jnp.bfloat16, SHAPES["compiled"], natural=False, interpret=False, width=width,
                   value_width=value_width, tiles=_form_tiles("sm90"))


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
def the_backward_at_its_faulting_geometry_is_on_par_on_an_h100():
    """The full backward at the geometry whose two launches faulted on an H100 (row 2254: two rows of 4096
    tokens, 32 query heads over 8 key heads, 128 wide, pinned at 64 x 64, two stages, one consumer), compiled
    on an H100 on the published tip: finite and on par with the reference (check 1, row 2269)."""
    with _pinned(backward=attention_mosaic.BackwardTiles(64, 64, 2, 1, family=WGMMA, **FED)):
        _check_backward("full", (2, 4096, 32, 8), form="default", interpret=False, width=128)


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


@ON_HOPPER
def the_backward_256_wide_under_a_softcap_is_on_par_on_an_h100():
    """Gemma 2's backward, a softcap 256 wide, at its derivation's tiles (row 3704: one launch whose
    consumers share a key block, `BackwardTiles.shared_keys`) compiled on an H100: on par with the reference
    (check 1)."""
    _check_hooked_backward("softcap", form="derived", interpret=False, width=256)


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
    training forward and in the backward; a rerun (the fold's natural walk) adds none. The launches' own
    statements are read."""
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
    pins = (attention_mosaic.Tiles(block, block, 2, consumers, family=WGMMA),
            attention_mosaic.BackwardTiles(block, block, 2, consumers, fused=True, family=WGMMA))
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
    with _pinned(attention_mosaic.Tiles(block, block, 2, consumers, family=WGMMA, columns=width // 2)):
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


def _residuals(shape) -> tuple:
    """A training forward's residuals' shapes for a call of `shape` `[B, Q, H, D]`, zeros (`backward` binds
    its arguments only)."""
    batch, queries, heads, width = shape
    stats = jnp.zeros((batch, heads, queries), jnp.float32)
    return (stats, stats, jnp.zeros((batch, heads, queries), jnp.int32),
            template.Split(jnp.zeros((batch, heads, queries, width), jnp.float32), stats, stats, stats))


@INTERPRETS
def test_an_unrounded_fused_backward_keeps_its_key_and_value_gradients_fp32_interpreted():
    """The fast form's one launch of a bf16 causal call (`BackwardTiles.fused`) whose caller takes dK and dV
    unrounded (`attention_mosaic.backward`'s `rounded` False: a sequence exchange's, summed across shards
    and rounded once): its dK and dV are its fp32 sums, bits past bf16's among them, where the launch stored
    them in bf16 when its block held the typed stage (`attention_mosaic_fused.stores_typed`) and the caller
    took them rounded twice."""
    shape = (2, 128, 4, 4)
    spans = template.spans(shape[1], shape[1], shape[0], causal=True)
    query, key, value, dout = _drawn(shape, jnp.bfloat16, 64)(0)
    _, config = _backward_form("fused")
    with _device(interpret=True):
        residuals = attention_mosaic.forward(query, key, value, spans, None, residuals=template.Kept.SPLIT,
                                             interpret=True)[1:]
        with _pinned(backward=config):
            grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals,
                                              interpret=True, rounded=False)
    for grad in (grads.key, grads.value):
        assert grad.dtype == jnp.float32
        assert np.any(np.asarray(grad) != np.asarray(grad.astype(jnp.bfloat16).astype(jnp.float32)))
