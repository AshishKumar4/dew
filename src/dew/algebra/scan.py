"""A recurrence's chunk forms derived from its token update (docs/design/algebra.md §6.9, `Scan`): a state `h`
updated token by token, `h = update(h, x_t)`, read out each token, `o_t = readout(h, x_t)`.

The derivation reads the update's and the readout's jaxprs at one token as an affine map of the state by
rules (`linear_scan.parts`): a decay `D_t = exp(z_t)` (or `exp2`) on the state, a scalar or each row's own
(a per-key vector); one input `B_t (x) u_t` free of it; and corrections along the input's own vector,
reading the decayed state, `-s_t B_t (c_t^T D_t h)`. So `h_t = (I - s_t B_t c_t^T) D_t h_{t-1} + B_t u_t^T`:
the gated delta rule (a scalar decay, `s = beta`, `B = c = k`, `u = beta v`), Kimi's delta attention (a
per-key decay), or with no correction Mamba-2's state space. The readout must contract the
state with a vector `q_t` along `B_t`'s axis, plus a part free of it. Over a chunk, with `G` the decay
exponents' inclusive prefix (per key), `Gamma[i, j, k] = exp(G_ik - G_jk)` for `j <= i` and
`gamma = exp(G)`, the inputs each token's correction leaves, `w`, solve the unit lower-triangular system
(the compact WY form, its inverse by doubling, products alone), each overlap a decayed contraction over keys
(a scalar decay's the keys' product decayed after it):

    (I + diag(s) strictly_lower(sum_k C_ik B_jk Gamma_ijk)) w = U - diag(s) (C * gamma) h_in,
    o = (Q * gamma) h_in + lower(sum_k Q_ik B_jk Gamma_ijk) w + r,
    h_out = gamma_last h_in + sum_j (B_j * exp(G_last - G_j)) w_j^T,

Mamba-2's state-space dual where no token corrects (`w = U`), derived from the token update rather than
written. The prefix is taken compensated (each partial a float and its rounding error, `two_sum`), and every
difference of two partials from both parts: a plain prefix loses about 1e-6 of a deep decay (ledger 2098). A
chunk's adjoint (`backward`) is its forms' derivative, each exponent's from the decays whose sums hold it,
summed without one term added and taken away again (`_decays_backward`). A token a reset marks (`Scan.reset`,
a packed document's first) takes the exponent `RESET_DECAY` in place of its own, so every decay spanning it is
zero and every other its own: the forms, their merge and their adjoint unchanged, the exponent its reset
replaced given no gradient.
Every other update or readout is refused, named.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core as jcore

from dew._cache import ALGEBRA
from dew.algebra import carry, cost, derive, linear_scan, program, terms
from dew.algebra.forms import Derived, Refused, aval, environment, evaluate, read_by
from dew.nn.derivatives import KernelGlue, Protected, contracted, declared_derivative
from dew.nn.precision import at_least_fp32, float32_einsum, float32_matmul, two_sum, vanishing

Axes = tuple[str, ...]

RESET_DECAY = -1e4
"""The exponent a reset token's decay takes in place of its own (natural or base 2): the state entering it
is multiplied by `exp(-1e4)`, which is 0 in fp32. A token's own exponent is never positive, so every span
holding a reset sums to at most this and decays to exactly 0, and a span without one never sees it. Finite,
so a span's exponent is a difference of finite partials (an infinity less an infinity is a NaN), and a
kernel's segment sums formed as a matmul against a 0/1 triangle stay finite (an infinity times 0 is a NaN);
thousands of resets in a walk's sum stay far inside fp32's range, and the compensated prefix keeps each
difference of two partials exact past them."""

@dataclasses.dataclass(frozen=True)
class Scan:
    """A recurrence over its `walked` axis (time): `update(h, **token)` the state after one token,
    `readout(h, **token)` that token's output from the state after it, each in plain JAX on one token's
    operands (`operands`' axes by name, the walked one dropped), the state of `state`'s shape and dtype.
    `reset`, where given, names an operand of `operands`, booleans along the walked axis alone that neither
    reads, marking each token a zero state enters (a packed document's first). `ranges` bounds operands by
    name, a magnitude, each element's (a sigmoid's gate 1, keys 1) or an interval (a log decay's `(-inf, 0)`),
    the range facts a reordering's new intermediates are held under (docs/design/algebra.md §6.3); an operand
    it names none of is unbounded, the state is where it names "state" (a bound on its entering value), and
    "overlap" bounds two tokens' corrections' overlap, `|s_i C_i . B_j|`, a vector bound the elementwise ones
    do not state (keys of unit norm under a write gate at most `b`: `b`, by Cauchy-Schwarz; else the
    elementwise bounds' product times the keys' count).
    Its decays (the exponentials of its update's exponents) are at most 1: its
    exponents are at most zero, as a gated recurrence's log decays are."""

    update: Callable
    readout: Callable
    operands: Mapping[str, Axes]
    walked: str
    state: tuple[tuple[int, ...], jnp.dtype]
    reset: str | None = None
    ranges: Mapping[str, float | tuple[float, float]] = dataclasses.field(default_factory=dict)


def _traced(declaration: Scan, function: Callable, names: tuple[str, ...], chunk: Mapping):
    """`function` (the declaration's update or readout) traced at one token of `chunk`: the state's type and
    each operand's without its walked axis, jit disabled so a jitted library function's equations are the
    declaration's own."""
    shape, dtype = declaration.state

    def named(h, *xs):
        return function(h, **dict(zip(names, xs, strict=True)))

    def token(name: str) -> jax.ShapeDtypeStruct:
        axes = zip(chunk[name].shape, declaration.operands[name], strict=True)
        return jax.ShapeDtypeStruct(tuple(extent for extent, axis in axes if axis != declaration.walked),
                                    chunk[name].dtype)

    tokens = [token(name) for name in names]
    with jax.disable_jit():
        return jax.make_jaxpr(named)(jax.ShapeDtypeStruct(shape, dtype), *tokens)


def derive_scan(declaration: Scan, *, saturation: program.Saturation,
                splits_held: int = derive.FORMS_HELD) -> Derived:
    """`declaration`'s chunk forms (the module's account), split into what every chunk computes at once and
    what its carry computes in sequence (`carry.split`, under `saturation`'s limits, the latest
    `splits_held` contexts' and shapes' splits kept: the bound `derive.FORMS_HELD` sets a declaration's
    forms, given apart only so a test reaches eviction without 65 derivations, never a knob to tune):
    `init(shapes)` a zero state and no decay, `prepare(chunk)` a chunk's values that read no state
    (`Prepared`), `step(state, prepared)` the state after the chunk, `mapped(prepared, state)` the chunk's
    outputs from the state entering it, `merge(a, b)` two consecutive walks' states joined (the later walk's
    from zero), `finish(state, operands)` the final state; `backward` a chunk's adjoint, a cotangent for
    each operand but the reset's. Refused (`Refused`) where the update or the readout is not of the
    module's form. `init(shapes)` given a chunk's operands (or their shapes and dtypes) starts a walk that
    merges (`Walk`), its transfer carried where its tokens correct the state; given none, one that steps
    alone. `prepare`, `step`, `mapped` and `merge` each take a mainloop's issuer, `issue(role, lhs, rhs)`
    (`lhs @ rhs`), which makes every matrix product of the forms, each by its role: `scores`, `gram`,
    `inverse_merge`, `apply` and `carry` prepared; `correct` and `carry` stepped (`carry.CARRY_ROLES`);
    `readout_memory` and `readout_local` mapped (`carry.EMIT_ROLES`); `transfer_compose` (a chunk's prepared
    transition on the transfer, and two transfers joined), `transfer_apply` and `transfer_decay` (`rhs *
    lhs`, a decay's column scale, the one role no product) forming and joining a walk's transfer. The same
    forms without one take Dew's products, the portable route's; a chunk's transition, which a walk that
    carries its transfer composes (`prepare(chunk, transition=True)`), is always Dew's, formed
    chunk-parallel."""
    names = tuple(name for name in declaration.operands if name != declaration.reset)
    shape, dtype = declaration.state
    if len(shape) != 2:
        raise Refused("a state of rank other than two (keys by values) has no chunk form here")

    decomposed: dict = {}  # the token's affine map by its operands' types, its e-graph saturated once

    def affine_of(chunk: Mapping):
        types = tuple((tuple(chunk[name].shape), str(chunk[name].dtype)) for name in names)
        if types not in decomposed:
            decomposed[types] = linear_scan.parts(_traced(declaration, declaration.update, names, chunk),
                                                   _traced(declaration, declaration.readout, names, chunk))
        return decomposed[types]

    def parts(chunk: Mapping):
        affine = affine_of(chunk)
        return affine, _token_parts(declaration, names, affine, chunk)

    # A walk's decay and state carried in its chunks' accumulation (`parts`' float32 at least; a float64
    # state's, float64), the state rounded to its declared dtype once, at the finish: a 16-bit state carried
    # as declared left each chunk wider than it entered.
    carried = at_least_fp32(jnp.dtype(dtype))

    def init(shapes):
        empty = jnp.zeros((shape[0],), carried), jnp.zeros(shape, carried)
        if not shapes:
            return empty
        corrected = affine_of(shapes).correction is not None
        return Walk(*empty, jnp.eye(shape[0], dtype=carried) if corrected else None)

    splits: collections.OrderedDict = collections.OrderedDict()  # the least recently prepared first

    def prepare(chunk, issue=None, transition: bool = False, inverse=None):
        affine, (exponent, *values) = parts(chunk)
        decays = _decays(exponent, affine.base2)
        natural = _walked(exponent, affine.base2)
        tokens = dict(zip(TOKENS, values, strict=True))
        extents = linear_scan.Extents(tokens["B"].shape[0], shape[0], shape[1], exponent.shape[-1])
        corrected = affine.correction is not None
        form = _form(extents, corrected)
        # The form's largest values that read no state, prepared with the chunk's decays and tokens: the
        # operands of its function, whose state the carry reads (`carry.split`).
        held, solved = _prepared(form, tokens, decays, issue or _issued, inverse)
        given = (*decays, *(tokens[name] for name in TOKENS), *held.values())
        # A split is priced and traced where it is derived (`derive.context`): one derived under another
        # target, precision, numerics or mesh is none of this one's.
        key = (derive.context(), affine.base2, corrected,
               *((value.shape, str(value.dtype)) for value in given))
        split = splits.get(key)
        if split is None:
            split = splits[key] = _split(declaration, names, chunk, affine, form, tuple(held), given,
                                         jax.ShapeDtypeStruct(shape, carried), extents.tokens, saturation)
            if len(splits) > splits_held:
                splits.popitem(last=False)
        splits.move_to_end(key)
        prepared = Prepared((natural, split.prepare(*given, issue=issue or _issued)), split, corrected,
                            inverse=solved)
        if not (corrected and transition):
            return prepared
        return dataclasses.replace(prepared, transition=_transition(prepared, jnp.zeros(shape, carried)))

    def step(state, prepared, issue=None):
        decay, h = state[:2]
        natural, values = prepared.values
        (leaving,) = prepared.split.carry(values, h, issue)
        decay = _held(decay + natural.astype(decay.dtype))
        if not isinstance(state, Walk):
            return decay, leaving
        if prepared.corrected and state.transfer is None:
            raise Refused("a walk started for tokens that do not correct the state steps a chunk whose do")
        if state.transfer is not None and prepared.transition is None:
            raise Refused("a walk that carries its transfer steps a chunk prepared without its transition "
                          "(`prepare(chunk, transition=True)`)")
        transfer = (None if state.transfer is None
                    else (issue or _issued)("transfer_compose", prepared.transition, state.transfer))
        return Walk(decay, leaving, transfer)

    def mapped(prepared, state, issue=None):
        # Dew's products where no mainloop issues them (`_issued`): a readout's the portable math's.
        (outputs,) = prepared.split.emit(prepared.values[1], state[1], issue or _issued)
        return outputs

    def merge(a, b, operands, issue=None):
        del operands
        return _merged(a, b, None if issue else carried, issue or _issued)

    def finish(state, operands):
        del operands
        return state[1].astype(dtype)

    def backward(chunk, state, d_state, d_outputs, issue=None, inverse=None):
        return _backward((prepare, mapped, step), names, chunk, state, d_state, d_outputs, issue, inverse)

    return Derived(init, step, merge, finish, mapped,
                   {"rounding": "compensated prefix", "walked": declaration.walked}, backward, prepare)


def _token_parts(declaration: Scan, names: Sequence[str], affine, chunk: Mapping) -> tuple:
    """A chunk's tokens' exponents and the affine map's parts at each (`_token`), the exponents in float32
    at least: a narrower prefix loses a decay's digits to its partials' magnitude, and float16's overflows
    past six resets (65504); a reset token's decay `RESET_DECAY`."""
    # An operand with no walked axis (a rate, a skip) is every token's.
    axes = [declaration.operands[name].index(declaration.walked)
            if declaration.walked in declaration.operands[name] else None for name in names]
    exponent, *values = jax.vmap(lambda *xs: _token(affine, xs), in_axes=axes)(
        *[chunk[name] for name in names])
    exponent = exponent.astype(at_least_fp32(exponent.dtype))
    if declaration.reset is not None:
        exponent = jnp.where(chunk[declaration.reset][:, None], RESET_DECAY, exponent)
    return exponent, *values


def _backward(forms: tuple, names: Sequence[str], chunk, state, d_state, d_outputs, issue, inverse):
    """A chunk's forms' adjoint (`derive_scan`'s `backward`, `forms` its prepare, mapped and step): the
    reverse scan's step, each chunk's transition transposed, as the forward forms' derivative (their prefix,
    substitution and Dew products differentiated), each product's adjoint the issuer's where one is given
    (`_adjoined`), the solve the forward's kept `inverse` where given. The reset's booleans held, no operand
    of the derivative."""
    prepare, mapped, step = forms
    issued = None if issue is None else _adjoined(issue)

    def forward(h, operands):
        entering, prepared = (state[0], h), prepare({**chunk, **operands}, issued, inverse=inverse)
        return mapped(prepared, entering, issued), step(entering, prepared, issued)[1]

    _, pullback = jax.vjp(forward, state[1], {name: chunk[name] for name in names})
    d_h, d_chunk = pullback((d_outputs, d_state))
    return d_h, d_chunk


@functools.partial(jax.tree_util.register_dataclass, data_fields=["values", "transition", "inverse"],
                   meta_fields=["split", "corrected"])
@dataclasses.dataclass(frozen=True)
class Prepared:
    """A chunk's values that read no state (`carry.Split.prepare`), batched as a walk computes them for every
    chunk at once, the split that reads them and whether its tokens correct the state (static); where they
    do, the chunk's transition (`transition`, keys by keys: its carry's linear part, `_transition`), which a
    walk that merges composes into its transfer."""

    values: tuple
    split: carry.Split
    corrected: bool
    transition: jax.Array | None = None
    inverse: jax.Array | None = None
    """The chunk's solve's inverse less its unit diagonal (`[C, C]`, None where the form solves nothing),
    which a training forward keeps and its backward reads (`Derived.backward`'s `inverse`) in place of
    solving again."""


class Walk(NamedTuple):
    """A derived scan's state that merges (`derive_scan`'s `init` given a chunk's operands): its decay (the
    exponent sum per key in natural units), its state from the one entering it (zero), and where its tokens
    correct the state its transfer, the keys-by-keys matrix it applied to the entering state, else None (the
    decay's diagonal)."""

    decay: jax.Array
    h: jax.Array
    transfer: jax.Array | None


def _merged(a, b, carried, issue: Callable) -> Walk:
    """Two consecutive walks' states joined (`derive_scan`'s `merge`), the later's from zero applying its
    transfer to the state before it: its decay per key (its exponent sum in natural units) where no token
    corrects the state, else the transfer it carried; each in `carried`, the walks' accumulation dtype (as
    `issue` makes them where `carried` is None: a mainloop's issuer, its products float32 and its operands
    its own, each touched by it alone)."""
    if not (isinstance(a, Walk) and isinstance(b, Walk)):
        raise Refused("a merge of walks init did not start from their operands' shapes")
    if b.transfer is None:
        h = jnp.exp(b.decay)[:, None] * a.h + b.h
        transfer = None if a.transfer is None else jnp.exp(b.decay)[:, None] * a.transfer
    else:
        h = issue("transfer_apply", b.transfer, a.h) + b.h
        transfer = (issue("transfer_decay", jnp.exp(a.decay), b.transfer) if a.transfer is None
                    else issue("transfer_compose", b.transfer, a.transfer))
    if carried is None:
        return Walk(_held(a.decay + b.decay), h, transfer)
    return Walk(_held(a.decay + b.decay), h.astype(carried),
                None if transfer is None else transfer.astype(carried))


def _transition(prepared: Prepared, h: jax.Array) -> jax.Array:
    """The chunk's transition (keys by keys): its carry's linear part in the state (`jax.linearize` at a
    state `h`, the carry affine in it) over the identity's columns a state's width at a time, Dew's products,
    prepared for every chunk at once."""
    keys, width = h.shape
    _, linear = jax.linearize(lambda state: prepared.split.carry(prepared.values[1], state)[0], h)
    blocks = -(-keys // width)
    padded = jnp.pad(jnp.eye(keys, dtype=h.dtype), ((0, 0), (0, blocks * width - keys)))
    columns = jax.vmap(linear, in_axes=1, out_axes=1)(padded.reshape(keys, blocks, width))
    return columns.reshape(keys, blocks * width)[:, :keys]


DECAYS = 4
"""The chunk function's leading operands: its decays (`_Decays`' four fields), its tokens' values after."""

TOKENS = ("B", "U", "s", "C", "Q", "R")
"""A token's values (`_token`'s), as the chunk's form names them (`linear_scan.Chunk.token`)."""


def _form(extents: linear_scan.Extents, corrected: bool) -> linear_scan.Form:
    """The chunk's form at its extents, derived by the identities of affine scans and extracted by the
    work-span model under XLA's schedule (`linear_scan.derive`) on the target the trace compiles for and at
    the products' configured precision, each of which prices the forms (`cost.target`, `cost.product_work`):
    a derivation under one target is no other's."""
    return _derived_form(extents, corrected, cost.target(), jax.config.jax_default_matmul_precision,
                         ALGEBRA.value.checked)


@functools.cache
def _derived_form(extents: linear_scan.Extents, corrected: bool, target, precision,
                  checked: bool) -> linear_scan.Form:
    """`_form`'s, once for each of its facts, `precision` the configured one its products read, `checked`
    whether the extractor oracle holds its extraction (`Passes.checked`)."""
    del precision
    return linear_scan.derive(extents, corrects=corrected, schedule=cost.fused(target), checked=checked)


def _magnitude(bound: float | tuple[float, float] | None) -> float | None:
    """A declared range's magnitude (`Scan.ranges`): a magnitude itself, an interval its endpoints' larger."""
    return None if bound is None else max(map(abs, bound)) if isinstance(bound, tuple) else bound


def _split(declaration: Scan, names: Sequence[str], chunk: Mapping, affine, form: linear_scan.Form,
           held: tuple, given: tuple, state: jax.ShapeDtypeStruct, chunk_size: int,
           saturation: program.Saturation) -> carry.Split:
    """The chunk function's split (`carry.split`) at the shapes of `given`, its decays, tokens and the form's
    values that read no state (`held`, prepared), its operands bounded as the declaration ranges them
    (`_ranges`), a solve's inverse by its bound."""
    ranges = _ranges(declaration, names, chunk, affine, chunk_size)
    bound = ranges.pop("inverse", None)
    ranges.update({DECAYS + len(TOKENS) + place: bound for place, node in enumerate(held)
                   if bound is not None and linear_scan.inverse(node)})
    avals = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in given]
    return carry.split(functools.partial(_chunk, form, held), state, avals, (1,), saturation=saturation,
                       ranges=ranges, state_range=_magnitude(declaration.ranges.get("state")))


def _ranges(declaration: Scan, names: Sequence[str], chunk: Mapping, affine,
            chunk_size: int) -> dict:
    """The chunk function's operands' ranges (`_chunk`'s, `carry.split`'s): its decays at most 1, each token
    value's (`_token`'s) as the declared operands bound it (`terms.Region`'s magnitudes), and the solve's
    inverse's entries' bound under "inverse" (`prepare` places it); a value an unbounded operand's has
    none."""
    tokens = [jax.ShapeDtypeStruct(tuple(extent for extent, axis in zip(chunk[name].shape,
                                                                          declaration.operands[name],
                                                                          strict=True)
                                         if axis != declaration.walked), chunk[name].dtype)
              for name in names]
    closed = jax.make_jaxpr(lambda *xs: _token(affine, xs)[1:])(*tokens)
    declared = {index: declaration.ranges[name] for index, name in enumerate(names)
                if name in declaration.ranges}
    region = terms.region(closed, declared)
    bounded = {}
    for index, (output, literal) in enumerate(zip(region.outputs, region.literal_outputs, strict=True)):
        if output is None:
            # A literal output (a delta rule's unit correction scale) bounds itself.
            bounded[DECAYS + index] = float(np.max(np.abs(np.asarray(literal, np.float64))))
            continue
        magnitude = region.magnitudes.get(terms.key(output))
        if magnitude is not None:
            # No bound where it passes float64's range (a step of 2**1023 declared, `prepare`'s values).
            bounded[DECAYS + index] = math.inf if magnitude >= 1024 else 2.0 ** magnitude
    # The solve's inverse less its diagonal: the inverse of a unit lower-triangular matrix whose entries
    # are at most `a` in magnitude is at most `(1 + a)^(C - 1)` (by induction on the substitution); its
    # entries `s_i C_i . B_j` times a decay at most 1, a contraction over the keys: at most the declared
    # overlap (`Scan.ranges`' "overlap"), else the corrections' elementwise bounds' product times the keys'
    # count (two tokens of 16 keys each element at most 1 overlap by 16).
    places = TOKENS
    tokens_at = {name: DECAYS + place for place, name in enumerate(places)}
    entry = _magnitude(declaration.ranges.get("overlap"))
    if entry is None and all(tokens_at[name] in bounded for name in ("B", "s", "C")):
        B = region.outputs[places.index("B")]
        keys = math.prod(np.shape(region.literal_outputs[0]) if B is None else region.shape(B))
        entry = bounded[tokens_at["s"]] * bounded[tokens_at["C"]] * bounded[tokens_at["B"]] * keys
    if entry is not None:
        # No bound where it passes float64's range (16 unit keys over 256 tokens: 17**255).
        power = max(chunk_size - 1, 0) * math.log2(1.0 + entry)
        bounded["inverse"] = math.inf if power >= 1024 else 2.0 ** power
    return {**dict.fromkeys(range(DECAYS), 1.0), **bounded}


def _chunk(form: linear_scan.Form, held: tuple, h, entering, to_last, whole, between, *values):
    """A chunk's outputs and the state it leaves, from the state `h` entering it, its decays (`_Decays`'
    fields, each at most 1), its tokens' values (`_token`'s, `TOKENS`) and the form's prepared values
    (`held`'s nodes, `_prepared`): the extracted form evaluated (`_evaluated`), which `carry.split`
    splits."""
    decays = _Decays(entering=entering, to_last=to_last, whole=whole, between=between)
    tokens = dict(zip(TOKENS, values[:len(TOKENS)], strict=True))
    known = {terms.key(node): value for node, value in zip(held, values[len(TOKENS):], strict=True)}
    # The language's own products (`_plain`), which the split's forms name and issue themselves
    # (`carry.split`'s roles): a declared product's tangent in this trace reads the state opaquely.
    return (_evaluated(form.outputs, tokens, decays, h, known, _plain),
            _evaluated(form.state, tokens, decays, h, known, _plain))


def _plain(name: str, left, right):
    """A chunk's products as the language's plain ones (`float32_matmul`), as the trace `carry.split`
    derives from takes them (`_chunk`)."""
    del name
    return float32_matmul(left, right)


def _prepared(form: linear_scan.Form, tokens: Mapping, decays: _Decays, issue: Callable,
              inverse: jax.Array | None = None) -> tuple[dict, jax.Array | None]:
    """The form's largest values that read no state but its tokens and decays (a solve's inverse, a gram,
    a token decayed), by node, evaluated, each product `issue`'s (`_evaluated`), and the solve's inverse
    they read (None where the form solves nothing): a training forward's kept `inverse`, where given, in
    place of the solve (`_kept`)."""
    known: dict = {} if inverse is None else {INVERSE: inverse}
    nodes = linear_scan.prepared((form.outputs, form.state))
    values = {node: _evaluated(node, tokens, decays, None, known, issue) for node in nodes}
    return values, known.get(INVERSE)


INVERSE = object()
"""`_evaluated`'s key of a chunk's solve's inverse (less its unit diagonal) among the values it knows: a
training forward's kept one, which takes the solve's place, or the one it solved."""


def _evaluated(node, tokens: Mapping, decays: _Decays, h, known: dict, issue: Callable | None = None):
    """A node of the chunk's language (`linear_scan.Chunk`) at the chunk's tokens, decays and entering state
    `h`, each node once (`known`, by node), each matrix product `issue(role, lhs, rhs)`'s (`_issued`, Dew's,
    where none is given): a pair's scores (`scores`) or a strict gram (`gram`), the inverse's doubling
    (`inverse_merge`), an application (`apply`) and a write's `Kᵀ V` (`carry`)."""
    issue = issue or _issued
    from egglog import get_callable_args

    Chunk = linear_scan.Chunk
    named = terms.key(node)
    if named in known:
        return known[named]

    def of(child):
        return _evaluated(child, tokens, decays, h, known, issue)

    if (args := get_callable_args(node, Chunk.token)) is not None:
        value = tokens[terms.text(args[0])]
    elif get_callable_args(node, Chunk.entering) is not None:
        value = h
    elif (args := get_callable_args(node, Chunk.decay)) is not None:
        value = getattr(decays, terms.text(args[0]))
    elif (args := get_callable_args(node, Chunk.loop)) is not None:
        value = _looped(tokens, decays, h, of(args[0]), corrected=True)
    elif (args := get_callable_args(node, Chunk.plain)) is not None:
        value = _looped(tokens, decays, h, of(args[0]), corrected=False)
    elif (args := get_callable_args(node, Chunk.pair)) is not None:
        value = (of(args[0]), of(args[1]))
    elif (args := get_callable_args(node, Chunk.outputs)) is not None:
        value = of(args[0])[0]
    elif (args := get_callable_args(node, Chunk.state)) is not None:
        value = of(args[0])[1]
    elif (args := get_callable_args(node, Chunk.gram)) is not None:
        strict = bool(terms.integer(args[2]))
        value = _paired(of(args[0]), of(args[1]), decays.between, issue, "gram" if strict else "scores",
                        strict=strict)
    elif (args := get_callable_args(node, Chunk.rows)) is not None:
        value = of(args[1])[:, None] * of(args[0])
    elif (args := get_callable_args(node, Chunk.times)) is not None:
        value = of(args[0]) * of(args[1])
    elif (args := get_callable_args(node, Chunk.matmul)) is not None:
        # The chunk's scores (a pairing of its tokens through their decays, not the solve's) applied to its
        # tokens' values is its local readout, `readout_local` (`carry.EMIT_ROLES`), as the hand form's.
        pairing = get_callable_args(args[0], Chunk.gram)
        local = pairing is not None and not terms.integer(pairing[2])
        value = issue("readout_local" if local else "apply", of(args[0]), of(args[1]))
    elif (args := get_callable_args(node, Chunk.plus)) is not None:
        value = of(args[0]) + of(args[1])
    elif (args := get_callable_args(node, Chunk.minus)) is not None:
        value = of(args[0]) - of(args[1])
    elif (args := get_callable_args(node, Chunk.decayed)) is not None:
        value = decays.whole[:, None] * of(args[0])
    elif (args := get_callable_args(node, Chunk.written)) is not None:
        value = issue("carry", (of(args[0]) * decays.to_last).T, of(args[1]))
    elif (args := get_callable_args(node, Chunk.doubled)) is not None:
        # Differentiated by its own rule where Dew's products make it (`_lowered`); a mainloop's are forward.
        if INVERSE in known:
            value = _kept(of(args[0]), known[INVERSE])
        else:
            value = _lowered(of(args[0])) if issue is _issued else _doubled(of(args[0]), issue)
        known[INVERSE] = value
    elif (args := get_callable_args(node, Chunk.substituted)) is not None:
        value = _substituted(of(args[0]), of(args[1]))
    else:
        raise Refused(f"a chunk form's node of no evaluation: {node}")
    known[named] = value
    return value


def _looped(tokens: Mapping, decays: _Decays, h, values, *, corrected: bool):
    """The token loop over the chunk, each token's step in turn (its decay `exp(z_t)` the pair of it and the
    token before, `between[t, t - 1]`, the first's `entering[0]`): its outputs and the state it leaves. Each
    token's output is placed at its row by a constant one-hot column and the rows summed, the carry's own
    language (`carry.split` reads products and sums, not a concatenation)."""
    B, s, C, Q, R = (tokens[name] for name in ("B", "s", "C", "Q", "R"))
    placed = jnp.eye(B.shape[0], dtype=jnp.float32)
    outputs = 0.0
    for t in range(B.shape[0]):
        decay = decays.entering[0] if t == 0 else decays.between[t, t - 1]
        h = decay[:, None] * h
        if corrected:
            h = h - s[t] * B[t][:, None] * float32_matmul(C[t][None, :], h)
        h = h + B[t][:, None] * values[t][None, :]
        outputs = outputs + placed[:, t][:, None] * (float32_matmul(Q[t][None, :], h)[0] + R[t])[None, :]
    return outputs, h


def _substituted(lower, given):
    """`(I + lower)^-1 given` by forward substitution, `lower` strictly lower: a row a step, each reading the
    rows before it."""
    def row(index, solved):
        return solved.at[index].set(given[index] - float32_matmul(lower[index][None, :], solved)[0])

    return lax.fori_loop(0, given.shape[0], row, jnp.zeros_like(given))


def _token(parts: linear_scan.Parts, token: tuple) -> tuple:
    """One token's exponent, input vector `B` and values `U`, correction strength `s` and vector `C`, reader
    `Q` and free readout `R` (zeros where the readout has none): the affine map's parts
    (`linear_scan.Parts`) at the token's values, each in float32 at least."""
    shape = aval(parts.update.jaxpr.invars[0]).shape
    zero = jnp.zeros(shape, aval(parts.update.jaxpr.invars[0]).dtype)
    envs: dict = {}

    def read(index: int):
        traced, atom = parts.atoms[index]
        if isinstance(atom, jcore.Literal):
            return jnp.asarray(atom.val)
        env = envs.setdefault(id(traced), environment(traced, (zero, *token), {}))
        evaluate(traced, env, read_by(traced.jaxpr, {atom}))
        return env[atom].astype(at_least_fp32(env[atom].dtype))

    def along(term, extent: int):
        value = linear_scan.value_of(term, read, shape[0])
        return jnp.broadcast_to(jnp.reshape(value, (-1,)) if jnp.ndim(value) == 2 else value, (extent,))

    keys, values = shape
    B, U = along(parts.left, keys), along(parts.right, values)
    corrects = parts.correction is not None
    # A correction `B (X^T D h)` is the transition's `-s B C^T D h`: `C = -X`, of unit strength.
    C = -along(parts.correction, keys) if corrects else jnp.zeros_like(B)
    s = jnp.float32(1.0 if corrects else 0.0)
    exponent = along(parts.exponent, keys if parts.per_key else 1)
    return exponent, B, U, s, C, along(parts.reader, keys), along(parts.free, values)


def _paired(left, right, between, issue: Callable | None = None, role: str = "scores", *,
            strict: bool = False):
    """Each pair's decayed contraction over keys, `sum_k left_ik right_jk between_ijk` `[C, C]`: a decay a
    key dimension's over the pairs' every key; a scalar decay's (`between` `[C, C, 1]`) the keys'
    product decayed after it, `(left right^T) between`, which spans no `[C, C, K]` tensor (Qwen3.5-27B's
    derived forward took 66 ms on an H100 spanning it, derived-cells-h100-6d105a908). A per-key decay's
    pairs factor around each sub-chunk's reference (`_factored`): a product of the rows decayed to it by the
    keys decayed from it, the diagonal sub-chunks' alone elementwise. A pair no decay joins (`between` zero,
    a token's future) is zero, selected, not multiplied: its product, or its cotangent, may be infinite near
    float32's largest (a future corrected write's), whose product by a zero decay is NaN in every query's
    gradient the pair reads (the chunked op's reverse mode, near the largest) and in the decays'. A `strict`
    pairing's diagonal (a gram's own token) is no pair."""
    joined = between
    if strict:
        joined = jnp.where(jnp.tril(jnp.ones(between.shape[:2], bool), -1)[:, :, None], between, 0.0)
    active = jnp.any(joined != 0, axis=-1)
    issue = issue or _issued
    if between.shape[-1] == 1:
        # The product selected before its decay multiplies it: an inactive pair's may overflow, whose zero
        # decay's derivative would read it.
        return jnp.where(active, issue(role, left, right.T), 0.0) * joined[..., 0]
    # The factors read the reference's own decay (1, on the diagonal a strict pairing excludes).
    return jnp.where(active, _factored(left, right, between, joined, issue, role), 0.0)


def _factored(left, right, between, joined, issue: Callable, role: str):
    """A per-key decay's pairs `sum_k left_ik right_jk between_ijk` over sub-chunks of `SUB` rows
    (`recurrence_algebra.SUB`, fla's BC): a pair of an earlier sub-chunk's key `j` and a later one's row `i`
    is the product of the row decayed to the reference `p` before its sub-chunk and the key decayed from it,
    `(left_i between_ip) (right_j between_pj)`, each factor at most 1 (every exponent at most zero, each span
    across a vanished token zero, `_decays_value`), a matrix product `issue` makes; a sub-chunk's own pairs
    elementwise over its keys (at most SUB^2 K a sub-chunk, not C^2 K), by the pairing's own decays
    (`joined`: `between` less a strict pairing's diagonal)."""
    from dew.nn.kernels import recurrence_algebra

    count = left.shape[0]
    sub = recurrence_algebra.SUB
    starts = range(0, count, sub)
    rows = []
    for first in starts:
        own = slice(first, min(first + sub, count))
        blocks = []
        for start in starts:
            keys = slice(start, min(start + sub, count))
            if start > first:
                blocks.append(jnp.zeros((own.stop - own.start, keys.stop - keys.start), jnp.float32))
            elif start == first:
                blocks.append(float32_einsum("ik,ijk->ij", left[own], right[None, keys] * joined[own, keys]))
            else:
                reference = first - 1
                blocks.append(issue(role, left[own] * between[own, reference],
                                    (right[keys] * between[reference, keys]).T))
        rows.append(jnp.concatenate(blocks, axis=1))
    return jnp.concatenate(rows, axis=0)


def _lowered(overlaps):
    """`(I + overlaps)^-1` less its unit diagonal, `overlaps` strictly lower: the template's inverse by
    doubling (`recurrence_algebra.doubled` of `-overlaps`), each product a Dew product (`float32_matmul`),
    log2(C) levels of products where a forward substitution took C dependent row steps (the derived forward's
    every chunk 64 launches in sequence). The solve applies it with the unit diagonal outside the product
    (`_applied`, `given + L given`, as `weighted` adds it), so the product rounds the off-diagonal terms
    alone. Differentiated at every order (`_lowered_rule`): its reverse derivative, with `X = I + L`,
    `-X^T dL X^T` below the diagonal, its tangent the doubling's own."""
    return _lowered_rule()(overlaps)


def _doubled(overlaps, issue: Callable | None = None):
    """`_lowered`'s doubling, its products `issue`'s under the role `inverse_merge` (Dew's where none is
    given)."""
    # The template's algebra imports the compile point, which imports this package: read at a call.
    from dew.nn.kernels import recurrence_algebra

    if issue is None:
        return _below(recurrence_algebra.doubled(-overlaps, _issued))
    return _below(recurrence_algebra.doubled(-overlaps, lambda name, left, right: issue(
        "inverse_" + name, left, right)))


def _adjoined(issue: Callable) -> Callable:
    """`issue` (a mainloop's issuer) as its products' adjoints take it too: each `issue(role, lhs, rhs)`
    differentiated by the issuer's own products, `d lhs = issue(role + "_dlhs", d, rhsᵀ)` and `d rhs =
    issue(role + "_drhs", lhsᵀ, d)` (`transfer_decay`'s, a column scale, by its own scales), a declared
    derivative of a kernel's glue (`dew.nn.derivatives`)."""
    @functools.cache
    def rule(role: str) -> Callable:
        def forward(left, right):
            return issue(role, left, right), (left, right)

        def backward(saved, d):
            left, right = saved
            if role == "transfer_decay":
                return (jnp.sum(d * right, axis=0), d * left[None, :])
            return issue(role + "_dlhs", d, right.T), issue(role + "_drhs", left.T, d)

        return declared_derivative(KernelGlue(), functools.partial(issue, role), forward, backward)

    def issued(role: str, left, right):
        return rule(role)(left, right)

    return issued


def _issued(name: str, left, right):
    """A derived scan's products as Dew's (`float32_matmul`), the doubling's among them
    (`recurrence_algebra.doubled`'s issuer); `transfer_decay` a transfer's columns scaled, `right * left`;
    a readout (`carry.EMIT_ROLES`) the portable math's (`recurrence_algebra.remembered` by its issuer), its
    gradient in what it reads summed from the chunk's last token (row 4466: from the first, the entering
    memory's gradient 1.9 times the portable math's distance from float64)."""
    if name == "transfer_decay":
        return right * left[None, :]
    if name in carry.EMIT_ROLES:
        from dew.nn.kernels import recurrence_algebra

        return recurrence_algebra.remembered(left, right, recurrence_algebra.portable, name)
    return float32_matmul(left, right)


def _inverse_cotangent(lower, d_lower):
    """The overlaps' cotangent of `(I + overlaps)^-1 - I` (`lower`) from its own: with `X = I + lower`,
    `-X^T d_lower X^T` below the diagonal."""
    back = _applied(lower.T, d_lower)
    return _below(-_applied(lower, back.T).T)


def _kept(overlaps, lower):
    """A chunk's solve's inverse as a training forward kept it (`lower`, `_lowered`'s), in place of solving
    `overlaps` again: its value `lower`, its overlaps' cotangent the solve's (`_inverse_cotangent`), so a
    backward that reads the kept inverse redoes no doubling."""
    return _kept_rule()(overlaps, lower)


@functools.cache
def _kept_rule() -> Callable:
    def forward(overlaps, lower):
        del overlaps
        return lower, lower

    def backward(lower, d_lower):
        return _inverse_cotangent(lower, d_lower), jnp.zeros_like(lower)

    return declared_derivative(Protected(), lambda overlaps, lower: lower, forward, backward)


def _applied(lower, given):
    """`(I + lower) @ given`, the identity outside the product."""
    return given + float32_matmul(lower, given)


def _below(square):
    """`square`'s strictly lower part."""
    return jnp.where(jnp.tril(jnp.ones(square.shape, bool), -1), square, 0.0)


@functools.cache
def _lowered_rule() -> Callable:
    """`_doubled` differentiated at every order (`generation.linear_tangent`): reverse mode by its inverse's
    own transpose (`_lowered`'s), forward mode and every order above it by the doubling's tangent, so a
    `jax.jvp` of the derived forms runs where a custom VJP alone refused it."""
    from dew.nn.kernels import generation

    def forward(overlaps):
        lower = _doubled(overlaps)
        return lower, lower

    def backward(lower, operands, d_lower):
        del operands
        return (_inverse_cotangent(lower, d_lower),)

    return contracted("test_the_derived_delta_rule_chunks_hold_their_orders")(
        generation.linear_tangent(forward, backward, _doubled, kind=Protected()))


def _compensated(values, *, reverse: bool = False, restarts=None) -> tuple[jax.Array, jax.Array]:
    """`values`' inclusive prefix sums along their first axis (suffix sums where `reverse`), each a float and
    its rounding error (a compensated scan), each restarting where `restarts` holds (a segmented scan: an
    element of it the first of its segment's)."""
    def combine(left, right):
        total, error = two_sum(left[0], right[0])
        summed = total, error + left[1] + right[1]
        if restarts is None:
            return summed
        # Where the right part restarts, the left's sums are not its segment's.
        kept = right[2]
        return (jnp.where(kept, right[0], summed[0]), jnp.where(kept, right[1], summed[1]),
                left[2] | right[2])

    if restarts is None:
        return lax.associative_scan(combine, (values, jnp.zeros_like(values)), reverse=reverse)
    high, low, _ = lax.associative_scan(combine, (values, jnp.zeros_like(values), restarts), reverse=reverse)
    return high, low


class _Decays(NamedTuple):
    """A chunk's decays from its tokens' exponents (`_decays`), each per key: `entering`, the decay from the
    chunk's start through each token, `exp(G_i)`; `to_last`, from after each token to the chunk's end,
    `exp(G_last - G_j)`; `whole`, the chunk's, `exp(G_last)`; `between`, from after token `j` through token
    `i`, `exp(G_i - G_j)` for `j <= i` and zero above the diagonal."""

    entering: jax.Array
    to_last: jax.Array
    whole: jax.Array
    between: jax.Array


def _decays(exponents, base2: bool) -> _Decays:
    """`exponents`' decays (`_Decays`, `exp` or `exp2` of the sums), each exponent a partial sum of the
    compensated prefix of its segment differenced from both parts (`_decays_value`): every exponent taken is
    at most zero, so each stays finite at any decay, at `tokens^2 keys` elementwise work a chunk (a sub-chunk
    factoring around a reference prefix is the faster equal form, the extraction's choice under a kernel
    cost)."""
    return _decays_of(exponents, base2)


def _walked(exponents, base2: bool) -> jax.Array:
    """A chunk's whole decay in natural units, per key, as a walk's state carries it (`Derived.step`,
    `merge`): its exponents' compensated sum, or its dtype's vanished decay where a token's decay vanishes
    (`vanishing`; two of -2**127 summed are -inf, and its residual NaN)."""
    vanished = exponents <= vanishing(exponents.dtype, base2)
    live = _scanned(jnp.where(vanished, 0.0, exponents))[-1] * (math.log(2.0) if base2 else 1.0)
    return _held(jnp.where(jnp.any(vanished, axis=0), vanishing(exponents.dtype, base2=False), live))


def _held(decay: jax.Array) -> jax.Array:
    """A walk's decay in natural units held from below at its dtype's vanished one (`vanishing`): zero
    there, and a walk's sums of vanished chunks finite."""
    return jnp.maximum(decay, jnp.asarray(vanishing(decay.dtype, base2=False), decay.dtype))


def _decays_value(exponents, base2: bool) -> _Decays:
    """The decays (`_Decays`) of a chunk's tokens' exponents, each from the compensated prefix of its
    segment: a token whose exponent vanishes (`vanishing`, its decay and every decay through it zero) ends
    one and starts the next, taken as 0 within it, so no span's sum holds a vanished token's (a prefix
    holding -2**127 twice is -inf; one holding -2**70 loses a later -1 to its rounding, its decay 1 in place
    of `exp(-1)`), and each span through one is zero."""
    vanished = exponents <= vanishing(exponents.dtype, base2)
    high, low = _compensated(jnp.where(vanished, 0.0, exponents), restarts=vanished)
    segments = jnp.cumsum(vanished.astype(jnp.int32), axis=0)
    count = exponents.shape[0]
    below = (jnp.arange(count)[:, None] >= jnp.arange(count)[None, :])[:, :, None]
    gaps = (high[:, None] - high[None, :]) + (low[:, None] - low[None, :])
    # A span holds a vanished token where its ends lie in two segments, or its start in the first's.
    joined = below & (segments[:, None] == segments[None, :])
    started, ending = segments == 0, segments == segments[-1]
    # Each exponent masked before its exponential, as each span's: one across a vanished token is no sum
    # of a live span's (`exp(100)` unselected is infinite, its derivative's zero times it NaN).
    return _Decays(entering=jnp.where(started, _exp(base2, jnp.where(started, high + low, 0.0)), 0.0),
                   to_last=jnp.where(ending, _exp(base2, jnp.where(ending,
                                                                   (high[-1] - high) + (low[-1] - low), 0.0)),
                                     0.0),
                   whole=jnp.where(started[-1], _exp(base2, jnp.where(started[-1], high[-1] + low[-1], 0.0)),
                                   0.0),
                   between=jnp.where(joined, _exp(base2, jnp.where(joined, gaps, 0.0)), 0.0))


def _decays_of(exponents, base2: bool) -> _Decays:
    return _decays_rule(base2)(exponents)


@functools.cache
def _decays_rule(base2: bool) -> Callable:
    """`_decays_value` differentiated at every order (`generation.linear_tangent`): reverse mode by
    `_decays_backward`, forward mode and every order above it by the decays' own tangent, so a `jax.jvp` of
    the derived forms runs where a custom VJP alone refused it."""
    from dew.nn.kernels import generation

    def forward(exponents):
        decays = _decays_value(exponents, base2)
        return decays, decays

    def backward(decays, operands, cotangents):
        del operands
        return _decays_backward(base2, decays, cotangents)

    return contracted("test_the_derived_delta_rule_chunks_hold_their_orders")(
        generation.linear_tangent(forward, backward, functools.partial(_decays_value, base2=base2),
                                  kind=Protected()))


def _decays_backward(base2: bool, decays: _Decays, cotangents: _Decays):
    """Each exponent's cotangent as the sum of the terms whose exponent sum holds it, never one added and
    taken away again (each decay's own derivative its value times its cotangent): `entering`'s from each
    token on, `to_last`'s from each token before, `whole`'s every one, `between`'s each pair spanning it,
    `j < s <= i`, summed over the rectangle by compensated scans. A difference of two prefix sums of these
    terms loses the deep decays' gradients near zero (the oracle's `summed_ranges`)."""
    weight = math.log(2.0) if base2 else 1.0
    entering = cotangents.entering * decays.entering
    to_last = cotangents.to_last * decays.to_last
    between = cotangents.between * decays.between
    from_on = _scanned(entering, reverse=True)
    before = _earlier(_scanned(to_last))
    # Rectangle sums: each row's terms left of a column, then the rows from a column's own row down.
    left = jnp.swapaxes(_earlier(_scanned(jnp.swapaxes(between, 0, 1))), 0, 1)
    spanning = jnp.diagonal(_scanned(left, reverse=True), axis1=0, axis2=1).T
    whole = (cotangents.whole * decays.whole)[None]
    return ((from_on + before + spanning + whole) * weight,)



def _earlier(sums) -> jax.Array:
    """Inclusive prefix sums along the first axis made exclusive by a shift (zero first), so no term is
    added and taken away again."""
    return jnp.concatenate([jnp.zeros_like(sums[:1]), sums[:-1]])


def _scanned(values, *, reverse: bool = False) -> jax.Array:
    """`values`' inclusive prefix sums along their first axis (suffix sums where `reverse`), compensated."""
    high, low = _compensated(values, reverse=reverse)
    return high + low


def _exp(base2: bool, x):
    return jnp.exp2(x) if base2 else jnp.exp(x)
