"""Attention's check-1 matrix (tests/kernel_matrix.py): the route's training step on par with the reference.

Each cell runs the template's route (`attention_route.attend`) as a user's training step does, the forward and
its backward in one program, so XLA builds what it builds around the launches there: row 3180's dK and dV held
one buffer only under a dynamic-slice fusion the route's own program made, while the backward alone in its own
program was right. The output and every gradient are held to the reference path's (check 1) with their float64
terms (`attention_oracle.attention_terms`), so a causal query keeping few keys is held to its terms' rounding,
not its own cancelled size. The backward's launch form is pinned where the Mosaic mainloop serves it: the one
launch, or the one over shared keys, at the config the derivation takes, which must be of that form; two
launches at the derivation's where it takes two, else (a call whose derivation takes one, two launches its
deterministic mode's) at the dK/dV launch on the layout whose consumers feed their rings (`FED`).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import attention_terms, kept
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import assert_as_exact
from test_attention_mosaic import FED, _residuals

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention as template, attention_mosaic, attention_route

MAINLOOP = "attention_mainloop"
"""The choice of the mainloop the route's passes run on (`attention_route.MAINLOOP`)."""
CHOICES = frozenset({MAINLOOP})
"""The `generation.CHOICES` entries the cells run (tests/test_kernel_matrix.py's union, row 184); the public
op's kernels, the template's among them, are tests/attention_kernels_matrix.py's."""
KERNELS = {MAINLOOP: frozenset({"mosaic", "triton"})}
"""The mainloops the cells run under test (row 4513): each cell asserts its passes take them, the Mosaic one
where it pins the backward's launch form; which takes an unpinned pass is the route's to derive."""
LENGTH = 1024
DOCUMENTS = 4
CAP = 2.0
"""A softcap's cap: the scaled logits of unit Gaussians are of its size, so its slope varies across them."""
SEEDS = 8
HEADS = {"mha": (8, 8), "gqa-8-2": (8, 2), "gqa-8-4": (8, 4)}
"""Query heads over key heads, by the `heads` axis's value."""
BIAS = {"key-bias": template.Kind.KEY, "tile-bias": template.Kind.TILE}
"""The kind of the two parameters a bias hook adds, `s + a + 2 b`."""


@functools.cache
def _masked(mask: str):
    """A call's spans under `mask`, and the keys each query keeps."""
    segments = (np.repeat(np.arange(DOCUMENTS, dtype=np.int32), LENGTH // DOCUMENTS)[None]
                if mask == "documents" else None)
    causal = mask != "full"
    spans = template.spans(LENGTH, LENGTH, 1, causal=causal,
                           segment_ids=None if segments is None else jnp.asarray(segments))
    return spans, kept(LENGTH, LENGTH, causal=causal, segment_ids=segments)


def _case(case: dict[str, str]):
    """`case`'s call: its operands' shapes and dtype, its spans and the keys they keep."""
    heads, kv_heads = HEADS[case["heads"]]
    return heads, kv_heads, int(case["width"]), jnp.dtype(case["dtype"]), *_masked(case["mask"])


def _softcapped(s, q_pos, k_pos):
    """The softcap hook's logits."""
    return CAP * jnp.tanh(s / CAP)


def _biased(s, q_pos, k_pos, a, b):
    """A bias hook's logits, its two parameters added."""
    return s + a.astype(s.dtype) + 2.0 * b.astype(s.dtype)


def _hooks(case: dict[str, str], params: tuple) -> template.Hooks | None:
    """`case`'s hook over its drawn `params`: one function a hook, so every case's call of it is the one the
    template's analyses of a hook (its legality, traced at each tile shape) are kept for."""
    if case["hook"] == "softcap":
        return template.Hooks(_softcapped)
    if case["hook"] in BIAS:
        return template.Hooks(_biased, tuple(template.Param(param, BIAS[case["hook"]]) for param in params))
    return None


def _param_shape(kind: template.Kind, heads: int) -> tuple:
    """A parameter of `kind`'s shape: a key's `[B, H, K]`, a tile's `[B, H, Q, K]`."""
    return (1, heads, LENGTH) if kind is template.Kind.KEY else (1, heads, LENGTH, LENGTH)


def terms_of(keep, heads: int, *, kind: template.Kind | None = None, softcap: float | None = None):
    """The output's and the gradients' float64 terms (`attention_terms`) of a step `s + a + 2 b` under two
    parameters of `kind` (a bias of the logits, its terms an entry's, summed over the queries a key's
    shares, twice them for `b`), or under a `softcap`, or neither."""
    def terms(query, key, value, dout, *params):
        bias = None
        if kind is not None:
            def tile(param):
                param = np.asarray(param, np.float64)
                return param[:, :, None, :] if kind is template.Kind.KEY else param

            bias = np.broadcast_to(tile(params[0]) + 2.0 * tile(params[1]), (1, heads, LENGTH, LENGTH))
        out, dq, dk, dv, *entries = attention_terms(query, key, value, dout, keep, softcap=softcap, bias=bias)
        if kind is None:
            return out, dq, dk, dv
        shared = np.sqrt(np.sum(np.square(entries[0]), axis=2)) if kind is template.Kind.KEY else entries[0]
        return out, dq, dk, dv, shared, 2.0 * shared

    return terms


def _form(case: dict[str, str]) -> tuple[bool | None, bool]:
    """`case`'s form, `(natural, accurate)`: the accurate form's natural logits and accurate sums
    (docs/design/testing.md), or the fast form's default fold (None) and fast sums."""
    return (True, True) if case["form"] == "accurate" else (None, False)


def _backward_op(case: dict[str, str]):
    """`case`'s Mosaic backward op, its bound arguments and the device it is read for, under the case's
    generation (`attention_mosaic._backward_call`)."""
    heads, kv_heads, width, dtype, spans, _ = _case(case)
    query = jnp.zeros((1, LENGTH, heads, width), dtype)
    key = jnp.zeros((1, LENGTH, kv_heads, width), dtype)
    params = tuple(jnp.zeros(_param_shape(BIAS[case["hook"]], heads), jnp.float32) for _ in range(2)
                   ) if case["hook"] in BIAS else ()
    residuals = _residuals(query.shape)
    natural, accurate = _form(case)
    natural = attention_mosaic.NATURAL if natural is None else natural
    op, args = attention_mosaic._backward_call(query, key, key, spans, _hooks(case, params), query, residuals,
                                               scale=None, natural=natural, accurate=accurate)
    return op, op.bind(*args, interpret=False), generation.gpu_target()


@functools.cache
def _route(call: tuple[tuple[str, str], ...]):
    """What the Mosaic backward does with a call (`case` but its launch): whether it declines it, and else the
    config it derives and the two launches' pin (`FED`) where that is legal, under the call's generation."""
    case = dict(call)
    with generation.assumed(case["generation"]):
        op, ba, target = _backward_op(case)
        if op.declined(ba, target) is not None:
            return True, None, None
        two = attention_mosaic.BackwardTiles(64, 64, 2, 1, columns=2 if case["width"] == "256" else 1, **FED)
        return False, op.derived(ba, target), two if op.legal(ba, two, target) else None


def pin_of(case: dict[str, str]) -> attention_mosaic.BackwardTiles | None:
    """The backward config `case` pins (its launch form on the Mosaic mainloop, the module's docstring), or
    None where another mainloop runs its backward; raises where the route serves no such form."""
    declined, derived, two = _route(tuple((axis, value) for axis, value in case.items() if axis != "launch"))
    if declined:
        if case["launch"] != "two":
            raise ValueError(f"the Mosaic backward declines {case}: no form but another mainloop's")
        return None
    if derived is None:
        raise ValueError(f"the Mosaic backward derives no config for {case}")
    if case["launch"] == "two":
        config = derived if not derived.fused else two
    elif derived.fused and derived.shared_keys == (case["launch"] == "shared-keys"):
        config = derived
    else:
        config = None
    if config is None:
        raise ValueError(f"the route serves no {case['launch']} launch for {case}")
    return config


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its backward on the Mosaic mainloop in the
    case's launch form (the one launch only where the derivation takes one, `pin_of`), or on another mainloop
    in two."""
    try:
        pin_of(case)
    except ValueError:
        return False
    return True


def _step(case: dict[str, str], spans, *, reference: bool):
    """`case`'s training step: the output, and the gradients of `sum(out * dout)` of its query, key, value
    and hook parameters, through the route (its launch form pinned) or the reference path."""
    natural, accurate = _form(case)

    def step(query, key, value, dout, *params):
        def attended(query, key, value, *params):
            hooks = _hooks(case, params)
            if reference:
                out = template.reference(query, key, value, spans, hooks)
            else:
                out = attention_route.attend(query, key, value, spans, hooks, natural=natural,
                                             accurate=accurate)
            return jnp.asarray(out)

        out, pull = jax.vjp(attended, query, key, value, *params)
        return (out, *pull(dout.astype(out.dtype)))

    return step


def operands(width: int, heads: int, kv_heads: int, dtype, kind: template.Kind | None = None):
    """A draw of an attention training step's operands by seed: unit Gaussians in `dtype` (and, under a hook
    of `kind`, its two parameters, fp32, of half that size), for the matrix's cells and the in-place
    regressions (attention_inplace_operands) alike."""
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 6)
        query, dout = (jax.random.normal(stream, (1, LENGTH, heads, width), dtype) for stream in streams[:2])
        key, value = (jax.random.normal(stream, (1, LENGTH, kv_heads, width), dtype)
                      for stream in streams[2:4])
        params = (() if kind is None else
                  tuple(0.5 * jax.random.normal(stream, _param_shape(kind, heads), jnp.float32)
                        for stream in streams[4:]))
        return query, key, value, dout, *params

    return draw


def _drawn(case: dict[str, str]):
    """A draw of `case`'s operands by seed (`operands`)."""
    heads, kv_heads, width, dtype, _, _ = _case(case)
    return operands(width, heads, kv_heads, dtype, BIAS.get(case["hook"]))


CASES = (
    ('sm90', '256', 'gqa-8-4', 'causal', 'plain', 'two', 'float32', 'fast'),
    ('sm80', '64', 'mha', 'full', 'key-bias', 'two', 'bfloat16', 'accurate'),
    ('sm90', '128', 'gqa-8-2', 'documents', 'softcap', 'shared-keys', 'bfloat16', 'fast'),
    ('sm80', '128', 'gqa-8-2', 'causal', 'tile-bias', 'two', 'float16', 'accurate'),
    ('sm80', '64', 'gqa-8-4', 'documents', 'softcap', 'two', 'float32', 'accurate'),
    ('sm90', '64', 'mha', 'causal', 'plain', 'one', 'bfloat16', 'fast'),
    ('sm90', '128', 'gqa-8-4', 'full', 'key-bias', 'two', 'float16', 'fast'),
    ('sm90', '256', 'mha', 'documents', 'tile-bias', 'two', 'float16', 'accurate'),
    ('sm80', '256', 'gqa-8-2', 'full', 'plain', 'two', 'float32', 'fast'),
    ('sm90', '64', 'gqa-8-4', 'full', 'tile-bias', 'two', 'bfloat16', 'fast'),
    ('sm90', '128', 'mha', 'documents', 'plain', 'two', 'float32', 'accurate'),
    ('sm90', '64', 'gqa-8-2', 'full', 'softcap', 'one', 'bfloat16', 'fast'),
    ('sm90', '256', 'mha', 'causal', 'plain', 'shared-keys', 'bfloat16', 'fast'),
    ('sm90', '64', 'mha', 'causal', 'softcap', 'two', 'float16', 'fast'),
    ('sm90', '256', 'gqa-8-2', 'causal', 'key-bias', 'two', 'float32', 'fast'),
    ('sm90', '128', 'gqa-8-4', 'documents', 'plain', 'one', 'bfloat16', 'fast'),
    ('sm90', '128', 'gqa-8-4', 'full', 'softcap', 'shared-keys', 'bfloat16', 'fast'),
    ('sm90', '64', 'mha', 'causal', 'plain', 'two', 'float16', 'fast'),
    ('sm90', '64', 'mha', 'causal', 'tile-bias', 'two', 'float32', 'fast'),
    ('sm90', '64', 'mha', 'documents', 'key-bias', 'two', 'bfloat16', 'fast'),
    ('sm90', '256', 'mha', 'causal', 'softcap', 'two', 'bfloat16', 'fast'),
)
"""The cells' cases, each its axis values in `AXES`' order: the cases ledger rows name and every pair of axis
values the route serves (`kernel_matrix.covering("attention")`, regenerated where the axes or the route's
support change)."""


@cell("attention", covers=lambda case: dict(zip(AXES["attention"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=pytest.mark.skipif(
    generation.device_generation() != case[0], reason=f"an {case[0]} cell runs on its own GPU alone"))
    for case in CASES], ids="-".join)
def a_routes_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The route's training step of `case` on its GPU on par with the reference path's (check 1), its launch
    form pinned (`pin_of`): its passes on the Mosaic mainloop where it pins one, and on none but the GPU's
    mainloops (`KERNELS`)."""
    named = dict(zip(AXES["attention"], case, strict=True))
    heads, _, _, _, spans, keep = _case(named)
    config = pin_of(named)

    def ours(*operands):
        if config is None:
            return _step(named, spans, reference=False)(*operands)
        with tuning.pinned({attention_mosaic.MosaicBackward: config}):
            return _step(named, spans, reference=False)(*operands)

    terms = terms_of(keep, heads, kind=BIAS.get(named["hook"]),
                     softcap=CAP if named["hook"] == "softcap" else None)
    with generation.routes() as routes:
        assert_as_exact(jax.jit(ours), jax.jit(_step(named, spans, reference=True)), _drawn(named),
                        "-".join(case), seeds=SEEDS, terms=terms)
    recorded = {found.kernel for found in routes if found.choice == MAINLOOP}
    assert recorded <= KERNELS[MAINLOOP] and (config is None or "mosaic" in recorded), sorted(recorded)

