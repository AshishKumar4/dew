"""A recurrence's chunk forms derived from its token update (docs/design/algebra.md §6.9, `Scan`): a state `h`
updated token by token, `h = update(h, x_t)`, read out each token, `o_t = readout(h, x_t)`.

The derivation reads the update's and the readout's jaxprs at one token as an affine map of the state by
rules (`linear_scan.parts`): a decay `D_t = exp(z_t)` (or `exp2`) on the state, a scalar or each row's own
(a per-key vector); one input `B_t (x) u_t` free of it; and corrections along the input's own vector,
reading the decayed state, `-s_t B_t (c_t^T D_t h)`. So `h_t = (I - s_t B_t c_t^T) D_t h_{t-1} + B_t u_t^T`:
the gated delta rule (a scalar decay, `s = beta`, `B = c = k`, `u = beta v`), Kimi's delta attention (a
per-key decay), or with no correction Mamba-2's state space; its corrections may read the state the token
enters, before its decay, along the input's vector or an erase's own (RWKV7's, `linear_scan.Affine.
erased_before`), through the decays a token back (`_Decays.entering_before`). The readout must contract the
state with a vector `q_t` along `B_t`'s axis, plus a part free of it. Over a chunk, with `G` the decay
exponents' inclusive prefix (per key), `Gamma[i, j, k] = exp(G_ik - G_jk)` for `j <= i` and
`gamma = exp(G)`, the inputs each token's correction leaves, `w`, solve the unit lower-triangular system
(the compact WY form, its inverse by doubling, products alone), each overlap a decayed contraction over keys
(a scalar decay's the keys' product decayed after it):

    (I + diag(s) strictly_lower(sum_k C_ik B_jk Gamma_ijk)) w = U - diag(s) (C * gamma) h_in,
    o = (Q * gamma) h_in + lower(sum_k Q_ik B_jk Gamma_ijk) w + r,
    h_out = gamma_last h_in + sum_j (B_j * exp(G_last - G_j)) w_j^T,

Mamba-2's state-space dual where no token corrects (`w = U`), derived from the token update rather than
written. The prefix is taken compensated (each partial a float and its rounding error, `compensated_add`),
and every difference of two partials from both parts: a plain prefix loses about 1e-6 of a deep decay
(ledger 2098). A chunk's adjoint (`backward`) is its token update's, replayed per chunk (`replay`, the
owner's ruling of 2026-11-07): the chunk walked token by token by the affine map's parts from the state
entering it and differentiated, each token's state cotangent erased before its decay's is formed, where the
forms' derivative cancelled independently rounded inverse, decay and readout paths (row 1966: an aligned
key's decay gradient 2.04 times the token scan's error, its exact zero lost, NaNs near float32's largest).
A token a reset marks (`Scan.reset`, a packed document's first) takes the exponent `RESET_DECAY` in place of
its own, so every decay spanning it is zero and every other its own: the forms, their merge and their
adjoint unchanged, the exponent its reset replaced given no gradient.
Every other update or readout is refused, named.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.custom_batching import custom_vmap
from jax.custom_derivatives import SymbolicZero
from jax.extend import core as jcore

from dew import _cache
from dew._cache import ALGEBRA
from dew.algebra import carry, cost, derive, linear_scan, program, ratios, scaled, terms, windows
from dew.algebra.forms import Derived, Refused, aval, environment, evaluate, scalar_case_batch
from dew.nn import float_bits
from dew.nn.derivatives import Protected, contracted, declared_tangent
from dew.nn.kernels.templates.axes import laid
from dew.nn.nested import nested
from dew.nn.precision import (
    accurate_exp,
    at_least_fp32,
    compensated_add,
    renormalized_add,
    rounding_alone,
    twice_scaled,
    uncontracted,
    vanishing,
)
from dew.nn.programs import holes, read_by

Axes = tuple[str, ...]

RESET_DECAY = -1e4
"""The exponent a reset token's decay takes in place of its own (natural or base 2): the state entering it
is multiplied by `exp(-1e4)`, which is 0 in fp32. A token's own exponent is never positive, so every span
holding a reset sums to at most this and decays to exactly 0, and a span without one never sees it. Finite,
so a span's exponent is a difference of finite partials (an infinity less an infinity is a NaN), and a
kernel's segment sums formed as a matmul against a 0/1 triangle stay finite (an infinity times 0 is a NaN);
thousands of resets in a walk's sum stay far inside fp32's range, and the compensated prefix keeps each
difference of two partials exact past them."""

State = tuple[tuple[int, ...], jnp.dtype]
"""A state's shape and dtype (`Scan.state`)."""

Reads = Literal["after", "before", "apart"]
"""Where a scan's readout reads its state (`Scan.reads`): after its token's update, as it enters, or after its
decay apart from its group's writes (`Scan.group`)."""


@dataclasses.dataclass(frozen=True)
class Scan:
    """A recurrence over its `walked` axis (time): `update(h, **token)` the state after one token,
    `readout(h, **token)` that token's output from the state after it, or where `reads` is "before" from the
    state entering it (before its decay and its write: RWKV-6's `q_t S_{t-1}`), each in plain JAX on one
    token's operands (`operands`' axes by name, the walked one dropped), the state of `state`'s shape and
    dtype.
    `reset`, where given, names an operand of `operands`, booleans along the walked axis alone that neither
    reads, marking each token a zero state enters (a packed document's first). `ranges` bounds operands by
    name, a magnitude, each element's (a sigmoid's gate 1, keys 1) or an interval (a log decay's `(-inf, 0)`),
    the range facts a reordering's new intermediates are held under (docs/design/algebra.md §6.3); an operand
    it names none of is unbounded, the state is where it names "state" (a bound on its entering value), and
    "overlap" bounds two tokens' corrections' overlap, `|s_i C_i . B_j|`, a vector bound the elementwise ones
    do not state (keys of unit norm under a write gate at most `b`: `b`, by Cauchy-Schwarz; else the
    elementwise bounds' product times the keys' count).
    Its decays (the exponentials of its update's exponents) are at most 1: its
    exponents are at most zero, as a gated recurrence's log decays are.
    Where `reads` is "apart", each token reads the state after its decay (and every earlier token's) less
    every write of its group, the `group` consecutive tokens it lies among (its own write among them, a
    group of one): Mamba-3's ranks (`group` the MIMO rank R, a token's R ranks the walked axis's R entries,
    mamba_ssm's tilelang mamba3_mimo_fwd excluding a token's whole R x R block), whose own block a model adds
    itself. A rank reading a lower rank of its own token at its gate and taking the gate's part back off
    cancelled catastrophically (one row 13 times its reference in 48 seeds, row 7402). A correction reads
    apart from a group of one alone: each token reads the state after its decay, before its own corrected
    write, its pairs with the chunk's earlier tokens' corrected writes (row 7608); a group past one beside a
    correction is refused, its tokens' writes correcting the state its group's earlier writes left.
    A `state` that is a pair of one shape and dtype is a coupled pair's (`coupled_scan`): `update((h, m),
    **token)` both states after the token and `readout((h, m), **token)` its output, their forms derived
    apart (`coupled_scan.derive_coupled`)."""

    update: Callable
    readout: Callable
    operands: Mapping[str, Axes]
    walked: str
    state: State | tuple[State, State]
    reset: str | None = None
    ranges: Mapping[str, float | tuple[float, float]] = dataclasses.field(default_factory=dict)
    reads: Reads = "after"
    group: int = 1


def _traced(declaration: Scan, function: Callable, names: tuple[str, ...], chunk: Mapping):
    """`function` (the declaration's update or readout) traced at one token of `chunk`: the state's type (a
    pair's two, `paired`) and each operand's without its walked axis, jit disabled so a jitted library
    function's equations are the declaration's own."""
    def named(h, *xs):
        return function(h, **dict(zip(names, xs, strict=True)))

    states = declaration.state
    typed = (tuple(jax.ShapeDtypeStruct(*state) for state in states) if paired(states)
             else jax.ShapeDtypeStruct(*states))
    tokens = [token_type(declaration, chunk, name) for name in names]
    with jax.disable_jit():
        return jax.make_jaxpr(named)(typed, *tokens)


def paired(states) -> bool:
    """Whether a declaration's `Scan.state` is a pair's, two shapes and dtypes, rather than one's."""
    return len(states) == 2 and all(
        isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], tuple) for state in states)


def token_type(declaration: Scan, chunk: Mapping, name: str) -> jax.ShapeDtypeStruct:
    """`chunk`'s operand `name` at one token: its type without the walked axis."""
    axes = zip(chunk[name].shape, declaration.operands[name], strict=True)
    return jax.ShapeDtypeStruct(tuple(extent for extent, axis in axes if axis != declaration.walked),
                                chunk[name].dtype)


def derive_scan(declaration: Scan, *, saturation: program.Saturation,
                splits_held: int = derive.FORMS_HELD) -> Derived:
    """`declaration`'s chunk forms (the module's account), split into what every chunk computes at once and
    what its carry computes in sequence (`carry.split`, under `saturation`'s limits, the latest `splits_held`
    contexts' and shapes' splits kept: the bound `derive.FORMS_HELD` sets a declaration's forms, given apart
    only so a test reaches eviction without 65 derivations, never a knob to tune): `init(shapes)` a zero state
    and no decay, `prepare(chunk)` a chunk's values that read no state (`Prepared`), `step(state, prepared)`
    the state after the chunk, `mapped(prepared, state)` the chunk's outputs from the state entering it,
    `merge(a, b)` two consecutive walks' states joined (the later walk's from zero), `finish(state, operands)`
    the final state; `replay(chunk, h)` the chunk walked token by token from the state `h` entering it, its
    outputs and the state leaving it (`_replayed`), and `backward(chunk, state, d_state, d_outputs)` its
    derivative, the chunk's adjoint: the cotangent of the state entering it and of each operand but the
    reset's. `prepare(chunk, window=(origin, through))` prepares a document's segment of the chunk, a
    memory entering its token `origin` and its state left after `through` (`_windowed`: every other token
    the identity, its resets cleared), whose `mapped` and `step` from that memory are the segment's
    outputs and the state it leaves, as a forked walk loads and stores a document's memory mid-chunk: its
    values differ from the chunk's in content alone, so one split serves both (the split's key is its
    operands' shapes). `replay` and `backward` take a walk's own hooks around each token (`walked`,
    `_replayed`), `backward`'s with the cotangent of what the walk carries after the chunk, and give its
    cotangent before it besides. Refused (`Refused`) where the update or the readout is not of the module's
    form. `init(shapes)`
    given a chunk's operands (or their shapes and dtypes) starts a walk that merges (`Walk`), its transfer
    carried where its tokens correct the state; given none, one that steps alone. `prepare`, `step`, `mapped`
    and `merge` each take a mainloop's issuer, `issue(role, lhs, rhs)` (`lhs @ rhs`), which makes every matrix
    product of the forms, each by its role: `scores`, `gram`, `inverse_merge`, `apply` and `carry` prepared;
    `correct` and `carry` stepped (`carry.CARRY_ROLES`); `readout_memory` and `readout_local` mapped
    (`carry.EMIT_ROLES`); `transfer_compose` (a chunk's prepared transition on the transfer, and two transfers
    joined), `transfer_apply` and `transfer_decay` (`rhs * lhs`, a decay's column scale, the one role no
    product) forming and joining a walk's transfer. The same forms without one take Dew's products, the
    portable route's; a chunk's transition, which a walk that carries its transfer composes (`prepare(chunk,
    transition=True)`), is always Dew's, formed chunk-parallel."""
    if paired(declaration.state):
        return _coupled(declaration, saturation)
    names = tuple(name for name in declaration.operands if name != declaration.reset)
    shape, dtype = _keyed_state(declaration)

    # The token's affine map by its operands' types and what its derivation reads where it runs
    # (`derive.context`: the mesh, the target, the numerics, the matmul precision its products' parameters
    # hold), its e-graph saturated once for each, the latest `derive.FORMS_HELD` kept.
    decomposed: collections.OrderedDict = collections.OrderedDict()

    def affine_of(chunk: Mapping):
        key = (tuple((tuple(chunk[name].shape), str(chunk[name].dtype)) for name in names), derive.context())
        return _cache.holding(decomposed, key, lambda: linear_scan.parts(
            _traced(declaration, declaration.update, names, chunk),
            _traced(declaration, declaration.readout, names, chunk), saturation), derive.FORMS_HELD)

    # A walk's decay and state carried in its chunks' accumulation (`parts`' float32 at least; a float64
    # state's, float64), the state rounded to its declared dtype once, at the finish: a 16-bit state carried
    # as declared left each chunk wider than it entered.
    carried = at_least_fp32(jnp.dtype(dtype))

    def init(shapes):
        empty = jnp.zeros((shape[0],), carried), jnp.zeros(shape, carried)
        if not shapes:
            return empty
        affine = affine_of(shapes)
        corrected = affine.correction is not None
        across = None if affine.across is None else jnp.zeros((shape[1],), carried)
        return Walk(*empty, jnp.eye(shape[0], dtype=carried) if corrected else None, across)

    splits: collections.OrderedDict = collections.OrderedDict()  # the least recently prepared first

    def prepare(chunk, issue=None, transition: bool = False, window=None):
        affine = affine_of(chunk)
        (exponent, *values), held_scale = _segment_parts(declaration, names, affine, chunk, window)
        read = declaration if window is None else _unreset(declaration)
        decays = _started(_decays(exponent, affine.base2), None if read.reset is None else chunk[read.reset])
        natural = _walked(exponent, affine.base2)
        tokens, entering = _read_entering(read, chunk, dict(zip(TOKENS, values, strict=True)))
        apart = _apart(declaration, affine, tokens["B"].shape[0])
        extents = linear_scan.Extents(tokens["B"].shape[0], shape[0], shape[1], exponent.shape[-1],
                                      affine.values_read, apart, affine.across is not None, affine.before,
                                      affine.erase is not None)
        corrected = affine.correction is not None
        if held_scale is not None:
            split = _unlanded_split(extents, corrected=corrected, base2=affine.base2, scale=held_scale)
            given, entering = (tokens, decays, entering, _token_factors(exponent, affine.base2)), None
        else:
            form = _form(extents, corrected, saturation)
            # The largest values that read no state: operands of the function the carry reads.
            held = _prepared(form, tokens, decays, issue or _issued)
            given = (*_flat_decays(decays), *(tokens[name] for name in TOKENS), *held.values())
            # A split is priced and traced in its own target, precision, numerics and mesh.
            key = (derive.context(), affine.base2, corrected, affine.values_read, apart, extents.across,
                   extents.before, extents.erased, *((value.shape, str(value.dtype)) for value in given))
            split = _cache.holding(splits, key, lambda: _split(
                declaration, names, chunk, affine, form, tuple(held), given,
                jax.ShapeDtypeStruct(shape, carried), extents.tokens, saturation), splits_held)
        across = _walked(tokens["E"], affine.base2) if extents.across else None
        prepared = Prepared((natural, split.prepare(*given, issue=issue or _issued)), split, corrected,
                            entering=entering, values_read=affine.values_read, across=across)
        if not (corrected and transition):
            return prepared
        return dataclasses.replace(prepared, transition=_transition(prepared, jnp.zeros(shape, carried)))

    def mapped(prepared, state, kept=None, issue=None):
        # Dew's products where no mainloop issues them (`_issued`): the portable math's; `kept` None.
        (outputs,) = prepared.split.emit(prepared.values[1], state[1], issue or _issued)
        entering = prepared.entering
        return outputs if entering is None else _entered(outputs, entering, prepared.values_read, state[1])

    def merge(a, b, operands, issue=None):
        del operands
        return _merged(a, b, None if issue else carried, issue or _issued)

    def finish(state, operands):
        del operands
        return state[1].astype(dtype)

    def replay(chunk, h, walked=None):
        if walked is None:
            return _replayed(declaration, names, affine_of(chunk), chunk, h)
        # A walk's hooks take its documents' memories up where a reset would have zeroed them.
        return _replayed(_unreset(declaration), names, affine_of(chunk), chunk, h, walked)

    return Derived(init, _stepped, merge, finish, mapped,
                   {"rounding": "compensated prefix", "walked": declaration.walked},
                   functools.partial(_replay_backward, replay, names), prepare, replay=replay)


def _keyed_state(declaration: Scan) -> tuple:
    """A declaration's one state's shape and dtype, refused unless it is keys by values."""
    shape, dtype = declaration.state
    if len(shape) != 2:
        raise Refused("a state of rank other than two (keys by values) has no chunk form here")
    return shape, dtype


def _coupled(declaration: Scan, saturation: program.Saturation) -> Derived:
    """A pair's forms (`coupled_scan.derive_coupled`)."""
    from dew.algebra.coupled_scan import derive_coupled

    return derive_coupled(declaration, saturation=saturation)


def _segment_parts(declaration: Scan, names: Sequence[str], affine, chunk: Mapping, window=None):
    """A chunk's token parts (`_token_parts`), or where `window` is given (`(origin, through)`) a document's
    segment of it (`_windowed`, `_unreset`'s): every token before its origin the identity, so a token
    reading the state entering it (`Scan.reads`) at the origin reads the memory the window enters with."""
    if window is None:
        return _token_parts(declaration, names, affine, chunk)
    parts, unlanded = _token_parts(_unreset(declaration), names, affine, chunk)
    return _windowed(parts, window), unlanded


def _unreset(declaration: Scan) -> Scan:
    """`declaration` with no reset, as a window takes it: its segment ends before the next document's first
    token, and its origin takes its own exponent and reads the memory entering it, decayed and read as a
    row's entering state is, not zero."""
    return dataclasses.replace(declaration, reset=None)


def _replay_backward(replay: Callable, names: Sequence[str], chunk, state, d_state, d_outputs, walked=None):
    """A chunk's adjoint (`derive_scan`'s `backward`) through its token walk (`replay`): the cotangent of the
    state entering it and of each operand but the reset's, and with a walk's hooks (`walked`, theirs and the
    cotangent of what the walk carries after the chunk) the cotangent of what it carries before."""
    # The reset's booleans held, no operand of the derivative.
    operands = {name: chunk[name] for name in names}
    if walked is None:
        _, pullback = jax.vjp(lambda h, operands: replay({**chunk, **operands}, h), state[1], operands)
        return pullback((d_outputs, d_state))
    before, after, carried, d_carried = walked
    _, pullback = jax.vjp(lambda h, operands, carried: replay({**chunk, **operands}, h,
                                                               (before, after, carried)),
                          state[1], operands, carried)
    return pullback((d_outputs, d_state, d_carried))


def _apart(declaration: Scan, affine, tokens: int) -> int:
    """The tokens of the group a readout reads apart from (`Scan.group`, 0 where it reads after its write or
    before its token): refused where its tokens correct the state in a group past one (`Scan.reads`), a chunk
    of `tokens` cuts a group (its memory would hold its own group's earlier writes), or a group of more than
    one token is declared beside another readout, which reads none of it (train 1004's review F3), or its
    update decays the state along its values too (`linear_scan.Parts.across`), whose apart state no form
    decays along them; and an erase along a vector of its own (`linear_scan.Parts.erase`) beside a readout
    along the values or apart from its writes, which no form takes: refused here, which preparing and
    replaying a chunk both ask."""
    if affine.erase is not None and (affine.values_read or declaration.reads == "apart"):
        raise Refused("an erase along a vector of its own beside a readout along the values or apart from "
                      "its group's writes has no chunk form here")
    if declaration.reads == "apart" and affine.across is not None:
        raise Refused("a readout apart from its group's writes beside a decay along the state's values has "
                      "no chunk form here")
    if declaration.reads != "apart":
        if declaration.group != 1:
            raise Refused(f"a group of {declaration.group} tokens under a readout {declaration.reads!r} its "
                          "token: a group is what a readout apart from its writes reads none of")
        return 0
    if affine.correction is not None and declaration.group != 1:
        raise Refused(f"a readout apart from a group of {declaration.group} tokens' writes under a "
                      "correction has no chunk form here: a group of one alone")
    if declaration.group < 1 or tokens % declaration.group:
        raise Refused(f"a chunk of {tokens} tokens cuts a group of {declaration.group}")
    return declaration.group


def _stepped(state, prepared: Prepared, issue=None):
    """A walk's state after a prepared chunk and what it keeps for the chunk's outputs (`derive_scan`'s
    `step`, `Derived`'s scans): its decay summed, its state the chunk's carry from it, and where the walk
    carries its transfer, the chunk's transition composed on it; a single state keeps nothing (None)."""
    decay, h = state[:2]
    natural, values = prepared.values
    (leaving,) = prepared.split.carry(values, h, issue)
    decay = _held(decay + natural.astype(decay.dtype))
    if not isinstance(state, Walk):
        return (decay, leaving), None
    if prepared.corrected and state.transfer is None:
        raise Refused("a walk started for tokens that do not correct the state steps a chunk whose do")
    if state.transfer is not None and prepared.transition is None:
        raise Refused("a walk that carries its transfer steps a chunk prepared without its transition "
                      "(`prepare(chunk, transition=True)`)")
    transfer = (None if state.transfer is None
                else (issue or _issued)("transfer_compose", prepared.transition, state.transfer))
    return Walk(decay, leaving, transfer, _summed(state.across, prepared.across)), None


def _at_token(declaration: Scan, names: Sequence[str], affine, token: Mapping) -> tuple:
    """One token's exponent and the affine map's parts at it (`_token`), `token` its operands without the
    walked axis, the exponent in float32 at least: a narrower prefix loses a decay's digits to its partials'
    magnitude, and float16's overflows past six resets (65504); a reset token's decay `RESET_DECAY`."""
    exponent, *values = _token(affine, tuple(token[name] for name in names))
    exponent = exponent.astype(at_least_fp32(exponent.dtype))
    if declaration.reset is not None:
        exponent = jnp.where(token[declaration.reset], RESET_DECAY, exponent)
    return exponent, *values


def _walked_tokens(declaration: Scan, chunk: Mapping) -> tuple[dict, dict]:
    """`chunk`'s operands split by the walked axis: those along it, the axis first, a token each; and those
    with none (a rate, a skip), every token's."""
    along = {name: jnp.moveaxis(chunk[name], axes.index(declaration.walked), 0)
             for name, axes in declaration.operands.items() if declaration.walked in axes}
    return along, {name: chunk[name] for name in declaration.operands if name not in along}


def _token_parts(
    declaration: Scan, names: Sequence[str], affine, chunk: Mapping,
) -> tuple[tuple, np.floating | None]:
    """A chunk's parts and its held subnormal reader scale, if any."""
    along, every = _walked_tokens(declaration, chunk)
    parts = jax.vmap(lambda token: _at_token(declaration, names, affine, {**token, **every}))(along)
    reader = parts[1 + TOKENS.index("Q")]
    held = subnormal_reader_scale(affine.reader_scale, jnp.result_type(affine.reader_scale, reader))
    return (parts if held is not None else _with_reader_scale(affine, parts)), held


def _with_reader_scale(affine, parts: tuple) -> tuple:
    """The chunk forms' reader includes its scalar, as their products and range bounds read it."""
    exponent, b, u, strength, c, q, r, across, erase = parts
    return exponent, b, u, strength, c, affine.reader_scale * q, r, across, erase


def _windowed(parts: tuple, window) -> tuple:
    """A chunk's token parts (`_token_parts`) with every token outside `window` (`(origin, through)`, its
    first and last, inclusive) the identity token: exponent 0 (a unit decay), no input, correction, reader or
    free readout, so its update leaves the state as it was and its output is zero."""
    origin, through = window
    place = jnp.arange(parts[0].shape[0])
    inside = (place >= origin) & (place <= through)
    return tuple(jnp.where(inside.reshape(-1, *([1] * (jnp.ndim(part) - 1))), part, 0) for part in parts)


def _replayed(declaration: Scan, names: Sequence[str], affine, chunk: Mapping, h, walked=None) -> tuple:
    """`chunk` walked token by token from the state `h` entering it by the affine map's parts at each token
    (`_at_token`): `(outputs, h)`, each token's output and the state leaving the chunk, in the state's
    carried dtype. A scalar decay follows the transition, so its cotangent reads the corrected state; a
    per-key decay's correction and update read one decayed state, `D h`, so a state's cotangent meets the
    decay once, summed before it does. The scalar reader contracts each outer product before it is formed,
    so its gradient's tangent avoids a state's overflowing unscaled contraction. A token whose exponent
    vanishes (a reset's, `RESET_DECAY`) enters a zero state, selected, so no NaN of the state before it
    reaches the one after (row 1966's overflow orders). Each token's parts are formed inside its step
    (`_at_token`), as the token's own update forms them: a factor applied to an operand (a reader's scale,
    a write's gate) then meets the operand's cotangent inside the step, so a second derivative applies it
    before the step's products, never after them, which near float32's largest overflowed (row 5426).
    The chunk's shared operands (`every`) are laid a copy a token, so their cotangents leave the scan a
    token each and are summed outside it, as fla sums RWKV6's bonus gradient (chunk_rwkv6's du): closed
    over, the scan's reverse added them a token at a time in its carry, 1.67 times fla's error from
    float64.

    A walk's own hooks (`walked`, `(before, after, carried)`) take part in the tokens' steps:
    `before(t, h, carried)` the state token `t` enters and the walk's own values (a forked walk's document
    taking its memory up from its table), `after(t, h, carried)` its values after the token (a document
    leaving its memory in a slot); then `(outputs, h, carried)`."""
    along, every = _walked_tokens(declaration, chunk)
    count = next(iter(along.values())).shape[0]
    along.update({name: jnp.broadcast_to(value, (count, *value.shape)) for name, value in every.items()})

    group = _apart(declaration, affine, count)

    def token(carry, step):
        h, apart, index = carry
        z, b, u, strength, c, q, r, across, erase = _at_token(declaration, names, affine, step)
        # A token that reads the state entering it (`Scan.reads`) reads it before its decay and its write: a
        # reset's zero, any other's whole, a decay that vanishes at this token erasing nothing it reads.
        entering = None
        if declaration.reads == "before":
            read = h if declaration.reset is None else jnp.where(step[declaration.reset], 0.0, h)
            entering = _read(read, q, affine.values_read, affine.reader_scale) + r
        # A correction reading the state the token enters (`linear_scan.Parts.before`) reads a document's
        # start's zero and a key whose decay alone vanishes whole, as its forms read them (`_started`).
        entered = (None if not affine.before else h if declaration.reset is None
                   else jnp.where(step[declaration.reset], 0.0, h))
        _, cut = vanished(z, affine.base2)
        h = jnp.where(cut[:, None], 0.0, h)
        scale = affine.reader_scale
        split_read = subnormal_reader_scale(scale, jnp.result_type(scale, q)) is not None
        decay = (_token_factors(z, affine.base2) if split_read
                 else jnp.broadcast_to(_exp(affine.base2, z), (h.shape[0],)))
        factors = jax.tree.map(lambda factor: factor[:, None], decay)
        if affine.across is not None:
            # A decay along the values too (two-sided GLA's, `D h E`), at most 1 each.
            h = h * _exp(affine.base2, across)[None, :]
        if group:
            # A token apart from its group's writes reads the state its group entered, decayed by every token
            # since it: the group's first takes the state as it enters (`h`, its vanishing decay's zero).
            apart = jnp.where(index % group == 0, h, jnp.where(cut[:, None], 0.0, apart))
            apart = _scaled(apart, factors)
            entering = _read(apart, q, affine.values_read, affine.reader_scale) + r
        if affine.before:
            # The correction reads the state the token enters, before its decay (RWKV7's erase); its adjoint
            # meets the entering state's cotangent beside the decay's, mirrored. The token's own update, its
            # write less its erase (along its own vector, `linear_scan.Parts.erase`, where it has one, else
            # its write's), is read apart from the state it enters, whose decay a read along the keys takes
            # on the query, as the decode step reads them (`dew.nn.kernels.recurrence.update`): read through
            # the decayed state's own rounding, a packed additive erase's worst gradient row was 2.7 times
            # the decode step's (jsf-assoc3).
            correction = jnp.sum(c[:, None] * entered, axis=0)
            erased = (erase if affine.erase is not None else b)[:, None] * (strength * correction)[None, :]
            update = b[:, None] * u[None, :] - erased
            along = affine.values_read
            if along or split_read:
                h = _scaled(h, factors) + update
                output = _read(h, q, along, scale) + r
            else:
                output = _read(h, _scaled(q, decay), along, scale) + _read(update, q, along, scale) + r
                h = _scaled(h, factors) + update
        elif not affine.per_key:
            correction = jnp.sum(c[:, None] * h, axis=0)
            transitioned = _scaled(h - b[:, None] * (strength * correction)[None, :], factors)
            h = transitioned + b[:, None] * u[None, :]
            output = (_read(transitioned, q, affine.values_read, affine.reader_scale)
                      + _read((b, u), q, affine.values_read, affine.reader_scale) + r)
        else:
            decayed = _scaled(h, factors)
            correction = jnp.sum(c[:, None] * decayed, axis=0)
            h = decayed + b[:, None] * (u - strength * correction)[None, :]
            output = _read(h, q, affine.values_read, affine.reader_scale) + r
        return (h, apart, index + 1), output if entering is None else entering

    # The state carried in the type its update's parts widen it to (`_carry_type`, the token loop's), its
    # token's parts typed from the operands' token shapes and types, so a walk of no tokens is one too.
    def carry_typed(step):
        z, b, u, strength, c, _, _, across, erase = _at_token(declaration, names, affine, step)
        return jnp.zeros((), _carry_type(h, z, b, u, strength, c, across, erase))

    one = {name: jax.ShapeDtypeStruct(x.shape[1:], x.dtype, weak_type=jax.typeof(x).weak_type)
           for name, x in along.items()}
    carried = jax.eval_shape(carry_typed, one).dtype
    h = h.astype(carried)
    if walked is None:
        (h, _, _), outputs = lax.scan(token, (h, h, 0), along)
        return outputs, h
    before, after, own = walked

    def hooked(state, step):
        # A document's memory taken up at its first token, which begins a group (a position's ranks, a
        # group, are one document's), so its group's state is that memory (`token`'s `apart`).
        ((h, apart, index), own), (t, step) = state, step
        h, own = before(t, h, own)
        (h, apart, index), output = token((h.astype(carried), apart, index), step)
        return ((h, apart, index), after(t, h, own)), output

    ((h, _, _), own), outputs = lax.scan(hooked, ((h, h, 0), own), (jnp.arange(count), along))
    return outputs, h, own


@functools.partial(jax.tree_util.register_dataclass,
                   data_fields=["values", "transition", "entering", "across"],
                   meta_fields=["split", "corrected", "values_read"])
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
    entering: tuple | None = None
    """Where its tokens read the state entering them (`Scan.reads`), its first token's reader and every
    token's part free of the state, which its outputs take (`derive_scan`'s `mapped`); else None."""
    values_read: bool = False
    """Whether its readout reads the state along its values (`linear_scan.Parts.values_read`)."""
    across: jax.Array | None = None
    """Where its update decays the state along its values (`linear_scan.Parts.across`), the chunk's whole
    decay along them in natural units, which a walk that merges sums (`Walk.across`); else None."""


def rotated(x, cos, sin, *, axis: int = -1):
    """`x` with each adjacent pair `(2j, 2j + 1)` of its first `2 A` entries along `axis`, the complex number
    `x_2j + i x_2j+1`, multiplied by `cos_j + i sin_j` (`cos`, `sin` `[..., A]`, broadcast against `x`'s pairs
    with `axis` moved last), the rest as given: a turned state's rotation (`Hooks.turns`), RoPE's adjacent
    pairs' arithmetic (`dew.nn.rope.rotate`), on every pair's row at once."""
    from dew.nn import rope

    moved = jnp.moveaxis(x, axis, -1)
    pairs = cos.shape[-1]
    rows = moved.reshape(-1, moved.shape[-1])
    cos, sin = (jnp.broadcast_to(part, (*moved.shape[:-1], pairs)).reshape(-1, pairs) for part in (cos, sin))
    return jnp.moveaxis(rope.rotate(rows, cos, sin, pairs="adjacent").reshape(moved.shape), -1, axis)


def turned(x, turn, *, axis: int = -1):
    """`x` turned by `turn` `[..., A]` (`Hooks.turns`): each of its first `A` pairs along `axis` times
    `e^{-i turn}`, `(a, b) -> (a cos + b sin, b cos - a sin)` (`rotated`), at any size of turn its
    backend's sine and cosine take (their full-range reduction)."""
    return rotated(x, jnp.cos(turn), -jnp.sin(turn), axis=axis)


def within_pi(turns):
    """Each turn as an angle within pi of zero (a turned call's frames take them so,
    `templates.recurrence.turned.frames`): itself where it is, else read back from its rotation (`atan2` of
    its sine and cosine, their backend's full-range reduction), so a prefix of them stays within its `n pi`
    however far a turn reaches. Summed whole, a turn of 2^26 left every later prefix 2.4e-7 radians off, a
    double float's precision there (the chunk's final state 6 times the token scan's worst row); taken off
    by 2 pi's float32 pair, a turn of 2^127 overflowed its product's split (NaN), and one of 2^50 drifted
    7.7 radians by the pair's own rounding."""
    return jnp.where(jnp.abs(turns) <= np.pi, turns, jnp.arctan2(jnp.sin(turns), jnp.cos(turns)))


def rotation(high, low) -> tuple[jax.Array, jax.Array]:
    """`(cos, sin)` of the angle `high + low` (a double float, a chunk's prefix of turns within pi of zero,
    `within_pi`), the rotation by it composed of its two parts' (`e^{i hi} e^{i lo}`), each unit: read as
    `cos(high) - low sin(high)`, a low part past a radian left the rotation's magnitude 1.41."""
    cos, sin, cos_low, sin_low = jnp.cos(high), jnp.sin(high), jnp.cos(low), jnp.sin(low)
    return cos * cos_low - sin * sin_low, sin * cos_low + cos * sin_low


def _read_entering(declaration: Scan, chunk: Mapping, tokens: dict) -> tuple[dict, tuple | None]:
    """A chunk's token values (`TOKENS`) as the after-forms read them, and what its outputs take besides
    (`Prepared.entering`), None where each token reads the state after it. A token that reads the state
    entering it (`Scan.reads`) reads the one the token before it left: the forms' reader of each token its
    next token's (none where that one enters a zero state, a reset's) and no part free of the state, so each
    form's output is the next token's; the first token's reader, whose read of the state entering the chunk
    `_entered` makes, and every token's free part, which it adds."""
    if declaration.reads != "before":
        # An apart readout's forms are the after-forms', each pairing apart (`_apart`).
        return tokens, None
    readers = tokens["Q"]
    if declaration.reset is not None:
        readers = jnp.where(_walked_tokens(declaration, chunk)[0][declaration.reset][:, None], 0, readers)
    following = jnp.concatenate([readers[1:], jnp.zeros_like(readers[:1])])
    return {**tokens, "Q": following, "R": jnp.zeros_like(tokens["R"])}, (readers[0], tokens["R"])


def _entered(outputs, entering: tuple, values_read: bool, h, *, scale: float | np.floating = 1.0):
    """A chunk's outputs whose tokens read the state entering them (`_read_entering`, its `entering`): the
    forms' outputs each the next token's, the first token's read of the state `h` entering the chunk before
    them, and every token's part free of the state added."""
    reader, free = entering
    first = _read(h, reader, values_read, scale)
    return jnp.concatenate([first[None].astype(outputs.dtype), outputs[:-1]]) + free


def _read(h, reader, values_read: bool, scale: float | np.floating, *, row_sum: bool = False):
    """A scaled read along the state's keys or values, its rank-one part contracted before it is formed."""
    # The coupled token's ordinary read is a sum of elementwise products, not a dot.
    if row_sum and subnormal_reader_scale(scale, jnp.result_type(scale, reader)) is None:
        return jnp.sum((scale * reader)[:, None] * h, axis=0)
    dtype = np.asarray(scale).dtype if isinstance(scale, np.floating) else None
    if isinstance(h, tuple):
        left, right = h
        return (_scaled_contract("j,j,i->i", scale, dtype)(reader, right, left) if values_read
                else _scaled_contract("i,i,j->j", scale, dtype)(reader, left, right))
    return (_scaled_contract("j,ij->i", scale, dtype)(reader, h) if values_read
            else _scaled_contract("i,ij->j", scale, dtype)(reader, h))


class Walk(NamedTuple):
    """A derived scan's state that merges (`derive_scan`'s `init` given a chunk's operands): its decay (the
    exponent sum per key in natural units), its state from the one entering it (zero), and where its tokens
    correct the state its transfer, the keys-by-keys matrix it applied to the entering state, else None (the
    decay's diagonal)."""

    decay: jax.Array
    h: jax.Array
    transfer: jax.Array | None
    across: jax.Array | None = None
    """Where its update decays the state along its values too, its exponent sum per value in natural units;
    else None."""


def _merged(a, b, carried, issue: Callable) -> Walk:
    """Two consecutive walks' states joined (`derive_scan`'s `merge`), the later's from zero applying its
    transfer to the state before it: its decay per key (its exponent sum in natural units) where no token
    corrects the state, else the transfer it carried; each in `carried`, the walks' accumulation dtype (as
    `issue` makes them where `carried` is None: a mainloop's issuer, its products float32 and its operands
    its own, each touched by it alone)."""
    if not (isinstance(a, Walk) and isinstance(b, Walk)):
        raise Refused("a merge of walks init did not start from their operands' shapes")
    entered = a.h if b.across is None else scaled_exp(a.h, b.across[None, :])
    if b.transfer is None:
        h = scaled_exp(entered, b.decay[:, None]) + b.h
        transfer = None if a.transfer is None else scaled_exp(a.transfer, b.decay[:, None])
    else:
        h = issue("transfer_apply", b.transfer, entered) + b.h
        transfer = (issue("transfer_decay", accurate_exp(a.decay), b.transfer) if a.transfer is None
                    else issue("transfer_compose", b.transfer, a.transfer))
    across = _summed(a.across, b.across)
    if carried is None:
        return Walk(_held(a.decay + b.decay), h, transfer, across)
    return Walk(_held(a.decay + b.decay), h.astype(carried),
                None if transfer is None else transfer.astype(carried), across)


def _summed(a, b):
    """Two walks' exponent sums along the state's values (`Walk.across`), None the identity's: a walk that
    decays none composes with one that does as the zero sum."""
    if a is None or b is None:
        return b if a is None else a
    return _held(a + b.astype(a.dtype))


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


TOKENS = ("B", "U", "s", "C", "Q", "R", "E", "A")
"""A token's values (`_token`'s), as the chunk's form names them (`linear_scan.Chunk.token`): "E" its
exponents along the state's values (`linear_scan.Parts.across`), zero where its update decays none; "A" its
erase's own vector (`linear_scan.Parts.erase`), zero where its corrections run along "B"."""


def _form(extents: linear_scan.Extents, corrected: bool,
          saturation: program.Saturation) -> linear_scan.Form:
    """The chunk's form at its extents, derived by the identities of affine scans within `saturation`'s limits
    and extracted by the
    work-span model under XLA's schedule (`linear_scan.derive`) on the target the trace compiles for, which
    prices the forms (`cost.pricing_key`, `cost.product_work`): different prices require a new derivation. Its
    products are priced at the precision they run at (`_portable`'s, IEEE float32 whatever the configured
    one), so the configured precision is no fact of the form."""
    return _derived_form(extents, corrected, cost.pricing_key(cost.target()), ALGEBRA.value.checked,
                         saturation)


@functools.cache
def _derived_form(extents: linear_scan.Extents, corrected: bool, target, checked: bool,
                  saturation: program.Saturation) -> linear_scan.Form:
    """`_form`'s, once for each of its facts, `checked` whether the extractor oracle holds its extraction
    (`Passes.checked`)."""
    from dew.nn.kernels import recurrence_algebra

    return linear_scan.derive(extents, corrects=corrected, schedule=cost.fused(target),
                              precision=recurrence_algebra.STATE_PRECISION, saturation=saturation,
                              checked=checked)


def _magnitude(bound: float | tuple[float, float] | None) -> float | None:
    """A declared range's magnitude (`Scan.ranges`): a magnitude itself, an interval its endpoints' larger."""
    return None if bound is None else max(map(abs, bound)) if isinstance(bound, tuple) else bound


def _split(declaration: Scan, names: Sequence[str], chunk: Mapping, affine, form: linear_scan.Form,
           held: tuple, given: tuple, state: jax.ShapeDtypeStruct, chunk_size: int,
           saturation: program.Saturation) -> carry.Split:
    """The chunk function's split (`carry.split`) at the shapes of `given`, its decays, tokens and the form's
    values that read no state (`held`, prepared), its operands bounded as the declaration ranges them
    (`_ranges`), a solve's inverse by its bound."""
    if linear_scan.looped((form.outputs, form.state)):
        return _looped_split(form, held, affine.values_read, affine.base2)
    ranges = _ranges(declaration, names, chunk, affine, chunk_size)
    bound = ranges.pop("inverse", None)
    ranges.update({DECAYS + len(TOKENS) + place: bound for place, node in enumerate(held)
                   if bound is not None and linear_scan.inverse(node)})
    avals = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in given]
    return carry.split(functools.partial(_chunk, form, held, affine.values_read, affine.base2), state, avals,
                       (1,),
                       saturation=saturation, ranges=ranges,
                       state_range=_magnitude(declaration.ranges.get("state")))


def _looped_split(form: linear_scan.Form, held: tuple, values_read: bool, base2: bool) -> carry.Split:
    """A form that runs the token loop split by its construction, never by extraction: every value it reads
    but the state's is a token's or a decay's, so what it prepares is the chunk function's operands, and its
    carry and its emitted outputs each run the chunk function (`_chunk`, the loop a scan of one token's step)
    from the state entering the chunk. Its chunk function unrolled was a chain of every token's step, which
    the split's e-graph saturated and extracted whole: a 256-token chunk's 6,917 equations, 20 minutes, and
    egglog's extraction past Python's stack (row 7201)."""
    return _direct_split(lambda prepared, h: _chunk(form, held, values_read, base2, h, *prepared))


def _unlanded_split(extents: linear_scan.Extents, *, corrected: bool, base2: bool,
                    scale: float | np.floating) -> carry.Split:
    """Keep a subnormal reader scale in each read, not in the prepared query vector."""
    def run(prepared, prepared_state):
        tokens, decays, entering, steps = prepared
        across = _exp(base2, tokens["E"]) if extents.across else None
        out, h = _looped(tokens, decays, prepared_state, tokens["U"], corrected=corrected,
                         values_read=extents.values_read, apart=extents.apart, across=across,
                         before=extents.before, erased=extents.erased, reader_scale=scale, token_decays=steps)
        if entering is not None:
            out = _entered(out, entering, extents.values_read, prepared_state, scale=scale)
        return out, h

    return _direct_split(run)


def _direct_split(ran: Callable) -> carry.Split:
    """A token loop reads its operands and entering state without a derived carry split."""
    def prepare(*given, issue=None):
        del issue
        return given

    def carried(prepared: tuple, h, issue=None) -> tuple:
        del issue
        return (ran(prepared, h)[1],)

    def emitted(prepared: tuple, h, issue=None) -> tuple:
        del issue
        return (ran(prepared, h)[0],)

    return carry.Split(prepare, carried, emitted)


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
    closed = jax.make_jaxpr(lambda *xs: _with_reader_scale(affine, _token(affine, xs))[1:])(*tokens)
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
    # entries `s_i C_i . B_j` (an erase of its own's `s_i C_i . A_j`, `linear_scan.Parts.erase`) times a
    # decay at most 1, a contraction over the keys: at most the declared overlap (`Scan.ranges`' "overlap",
    # the writes' keys', which bounds no erase), else the corrections' elementwise bounds' product times the
    # keys' count (two tokens of 16 keys each element at most 1 overlap by 16).
    places = TOKENS
    tokens_at = {name: DECAYS + place for place, name in enumerate(places)}
    along = "B" if affine.erase is None else "A"
    entry = _magnitude(declaration.ranges.get("overlap")) if along == "B" else None
    if entry is None and all(tokens_at[name] in bounded for name in (along, "s", "C")):
        at = places.index(along)
        vector = region.outputs[at]
        keys = math.prod(np.shape(region.literal_outputs[at]) if vector is None else region.shape(vector))
        entry = bounded[tokens_at["s"]] * bounded[tokens_at["C"]] * bounded[tokens_at[along]] * keys
    if entry is not None:
        # No bound where it passes float64's range (16 unit keys over 256 tokens: 17**255).
        power = max(chunk_size - 1, 0) * math.log2(1.0 + entry)
        bounded["inverse"] = math.inf if power >= 1024 else 2.0 ** power
    return {**dict.fromkeys(range(DECAYS), 1.0), **bounded}


def _chunk(form: linear_scan.Form, held: tuple, values_read: bool, base2: bool, h, *operands):
    """A chunk's outputs and the state it leaves, from the state `h` entering it, its decays (`_Decays`'
    leaves, each at most 1), its tokens' values (`_token`'s, `TOKENS`) and the form's prepared values
    (`held`'s nodes, `_prepared`): the extracted form evaluated (`_evaluated`), its readout along the state's
    values where `values_read`, its decays along the values (`linear_scan.Chunk.across`) of its tokens'
    exponents "E" (`exp2` where `base2`), which `carry.split` splits."""
    decays, values = _decays_from(operands[:DECAYS]), operands[DECAYS:]
    tokens = dict(zip(TOKENS, values[:len(TOKENS)], strict=True))
    known = {terms.key(node): value for node, value in zip(held, values[len(TOKENS):], strict=True)}
    # The language's own products (`_plain`), which the split's forms name and issue themselves
    # (`carry.split`'s roles): a declared product's tangent in this trace reads the state opaquely.
    return (_evaluated(form.outputs, tokens, decays, h, known, _plain, values_read=values_read,
                       apart=form.apart, base2=base2, across=form.across, before=form.before,
                       erased=form.erased),
            _evaluated(form.state, tokens, decays, h, known, _plain, values_read=values_read,
                       apart=form.apart, base2=base2, across=form.across, before=form.before,
                       erased=form.erased))


def _plain(name: str, left, right):
    """A chunk's products as the language's plain ones, `dot_general`s, as the trace `carry.split` derives
    from takes them (`_chunk`): a mainloop's forms issue every product themselves (its roles); where none
    issues it, the portable walk's, the split's evaluator takes the product at the precision this trace
    states (`forms.evaluator`'s `float32_dot_general`), the portable math's (`_portable`): stated nowhere,
    the walk's state products ran at JAX's configured precision, TF32 on an A100 (M-JS-96). `float32_matmul`
    here traced a float64 run's product inside a protected span the language cannot read, and the split
    refused the state it read (Kimi Linear's float64 truth)."""
    del name
    from dew.nn.kernels import recurrence_algebra

    return jnp.matmul(left, right, precision=recurrence_algebra.STATE_PRECISION)


def _prepared(form: linear_scan.Form, tokens: Mapping, decays: _Decays, issue: Callable) -> dict:
    """The form's largest values that read no state but its tokens and decays (a solve's inverse, a gram,
    a token decayed), by node, evaluated, each product `issue`'s (`_evaluated`)."""
    known: dict = {}
    nodes = linear_scan.prepared((form.outputs, form.state))
    return {node: _evaluated(node, tokens, decays, None, known, issue) for node in nodes}


def _evaluated(node, tokens: Mapping, decays: _Decays, h, known: dict, issue: Callable | None = None, *,
               values_read: bool = False, apart: int = 0, base2: bool = False, across: bool = False,
               before: bool = False, erased: bool = False):
    """A node of the chunk's language (`linear_scan.Chunk`) at the chunk's tokens, decays and entering state
    `h`, each node once (`known`, by node), each matrix product `issue(role, lhs, rhs)`'s (`_issued`, Dew's,
    where none is given): a pair's scores (`scores`) or a strict gram (`gram`), the inverse's doubling
    (`inverse_merge`), an application (`apply`) and a write's `Kᵀ V` (`carry`). A token loop's reads are
    apart from its group's writes where `apart` (`Scan.group`). Where the form decays the state along its
    values (`across`), its decays along them are the tokens' exponents "E"'s (`_decays`, `exp2` where
    `base2`; `linear_scan.Chunk.across_prefix`), and its token loop's each token's own. Its token loop's
    corrections read the state a token enters where `before` (`linear_scan.Parts.before`)."""
    issue = issue or _issued
    from egglog import get_callable_args

    Chunk = linear_scan.Chunk
    named = terms.key(node)
    if named in known:
        return known[named]

    def of(child):
        return _evaluated(child, tokens, decays, h, known, issue, values_read=values_read, apart=apart,
                          base2=base2, across=across, before=before, erased=erased)

    if (args := get_callable_args(node, Chunk.token)) is not None:
        value = tokens[terms.text(args[0])]
    elif get_callable_args(node, Chunk.entering) is not None:
        value = h
    elif (args := get_callable_args(node, Chunk.decay)) is not None:
        value = getattr(decays, terms.text(args[0]))
    elif get_callable_args(node, Chunk.across_prefix) is not None:
        value = _decays(tokens["E"], base2)
    elif (args := get_callable_args(node, Chunk.across)) is not None:
        value = getattr(of(args[1]), terms.text(args[0]))
    elif (args := get_callable_args(node, Chunk.loop)) is not None:
        value = _looped(tokens, decays, h, of(args[0]), corrected=True, values_read=values_read, apart=apart,
                        before=before, erased=erased)
    elif (args := get_callable_args(node, Chunk.plain)) is not None:
        value = _looped(tokens, decays, h, of(args[0]), corrected=False, values_read=values_read,
                        apart=apart, across=_exp(base2, tokens["E"]) if across else None)
    elif (args := get_callable_args(node, Chunk.pair)) is not None:
        value = (of(args[0]), of(args[1]))
    elif (args := get_callable_args(node, Chunk.outputs)) is not None:
        value = of(args[0])[0]
    elif (args := get_callable_args(node, Chunk.state)) is not None:
        value = of(args[0])[1]
    elif (args := get_callable_args(node, Chunk.gram)) is not None:
        # Through decays a token back (`between_before`), a pair factors at the reference's next row.
        through = get_callable_args(args[3], Chunk.decay)
        value = _paired(of(args[0]), of(args[1]), of(args[3]), issue, _pairing_role(args[0]),
                        apart=terms.integer(args[2]),
                        back=through is not None and terms.text(through[0]) == "between_before")
    elif (args := get_callable_args(node, Chunk.spread)) is not None:
        value = spread(of(args[0]), of(args[1]), of(args[2]), of(args[3]), issue=issue)
    elif (args := get_callable_args(node, Chunk.weighed)) is not None:
        # Each pair's score times the earlier token's values decayed to the reader along them, summed.
        value = _each_decayed(of(args[0]), of(args[1]), of(args[2]))
    elif (args := get_callable_args(node, Chunk.applied)) is not None:
        value = issue("apply", of(args[1]), of(args[0]).T)
    elif (value := _elementwise(node, of)) is not None:
        pass
    elif (args := get_callable_args(node, Chunk.matmul)) is not None:
        # The chunk's scores (a pairing of its tokens through their decays, not the solve's) applied to its
        # tokens' values is its local readout, `readout_local` (`carry.EMIT_ROLES`), as the hand form's.
        pairing = get_callable_args(args[0], Chunk.gram)
        local = pairing is not None and _pairing_role(pairing[0]) == "scores"
        value = issue("readout_local" if local else "apply", of(args[0]), of(args[1]))
    elif (args := get_callable_args(node, Chunk.written)) is not None:
        value = issue("carry", _scaled(of(args[0]), of(args[2])).T, of(args[1]))
    elif (args := get_callable_args(node, Chunk.doubled)) is not None:
        # Differentiated by its own rule where Dew's products make it (`_lowered`); a mainloop's are forward.
        value = _lowered(of(args[0])) if issue is _issued else _doubled(of(args[0]), issue)
    elif (args := get_callable_args(node, Chunk.substituted)) is not None:
        value = _substituted(of(args[0]), of(args[1]))
    else:
        raise Refused(f"a chunk form's node of no evaluation: {node}")
    known[named] = value
    return value


def _elementwise(node, of: Callable):
    """`node`'s value where it is elementwise in its operands (`of`'s values) or a decay's scaling of them;
    None for any other."""
    from egglog import get_callable_args

    Chunk = linear_scan.Chunk
    if (args := get_callable_args(node, Chunk.rows)) is not None:
        return of(args[1])[:, None] * of(args[0])
    if (args := get_callable_args(node, Chunk.times)) is not None:
        # The decay (or the values' decay, `across`) on the right, the value it scales on the left.
        first_decays = get_callable_args(args[0], Chunk.decay) is not None
        value, decay = (of(args[1]), of(args[0])) if first_decays else (of(args[0]), of(args[1]))
        # A decay across a document's start (`_crossed`) takes its factor out by select before the product,
        # so a memory past the largest read before its reset is no 0 times inf, nor its gradient (row 8560);
        # a live span's that underflows multiplies as the recurrence does, a split decay's factors in turn.
        if first_decays or get_callable_args(args[1], Chunk.decay) is not None:
            value = jnp.where(_crossed(decay.first if isinstance(decay, _Split) else decay), 0.0, value)
        return _scaled(value, decay)
    if (args := get_callable_args(node, Chunk.plus)) is not None:
        return of(args[0]) + of(args[1])
    if (args := get_callable_args(node, Chunk.minus)) is not None:
        return of(args[0]) - of(args[1])
    if (args := get_callable_args(node, Chunk.decayed)) is not None:
        # A state carried across a document's start taken out by select before its decay, as `times` takes
        # a read's: a memory past the largest before the reset is no 0 times inf.
        whole = of(args[1])
        crossed = _crossed(whole.first if isinstance(whole, _Split) else whole)
        state = jnp.where(crossed[:, None], 0.0, of(args[0]))
        return _scaled(state, jax.tree.map(lambda x: x[:, None], whole))
    if (args := get_callable_args(node, Chunk.columns)) is not None:
        return _scaled(of(args[0]), jax.tree.map(lambda x: x[None, :], of(args[1])))
    return None


def _looped(tokens: Mapping, decays: _Decays, h, values, *, corrected: bool, values_read: bool = False,
            apart: int = 0, across=None, before: bool = False, erased: bool = False,
            reader_scale: float | np.floating | None = None, token_decays: _Split | None = None):
    """The token loop over the chunk, a scan of one token's step (its decay `exp(z_t)` the pair of it and the
    token before, `between[t, t - 1]`, the first's `entering[0]`): its outputs, a row a token, and the state
    it leaves, each token's read apart from its group's writes where `apart` (`Scan.group`), its corrections
    reading the state it enters where `before` (`linear_scan.Parts.before`), erasing along their own vector
    where `erased` (`linear_scan.Parts.erase`). Split by its construction (`_looped_split`), never unrolled
    into an e-graph."""
    B, s, C, Q, R = (tokens[name] for name in ("B", "s", "C", "Q", "R"))
    A = tokens["A"] if erased else B
    count = B.shape[0]
    if token_decays is None:
        between = decays.between[jnp.arange(1, count), jnp.arange(count - 1)]
        first = _scaled(1.0, jax.tree.map(lambda x: x[:1], decays.entering))
        stepped = jnp.concatenate([first, between])
    else:
        stepped = token_decays
    leading = stepped.first if isinstance(stepped, _Split) else stepped
    # A decay along the values too (each token's gates [C, V]).
    valued = jnp.ones((count, values.shape[-1]), leading.dtype) if across is None else across
    h = h.astype(_carry_type(h, leading, B, values, s, C, valued, A))
    # The decays a token back, cut where a document starts alone (`_decays_value`): what a correction reading
    # the state a token enters reads it through.
    back = jnp.concatenate([_scaled(1.0, jax.tree.map(lambda x: x[:1], decays.entering_before)),
                            decays.between_before[jnp.arange(1, count), jnp.arange(count - 1)]])

    def step(carry, token):
        h, held, index = carry
        decay, b, strength, c, q, r, value, along, a, started = token
        # Read before the decay where the token's corrections read the state it enters, as the replay reads
        # it: a document's start's zero, a key whose decay alone vanishes whole.
        entered = (_portable(c[None, :], jnp.where(_crossed(started)[:, None], 0.0, h))
                   if corrected and before else None)
        # A cut key's state selected out before its decay, so a memory past the largest before a document's
        # start is no 0 times inf.
        leading = decay.first if isinstance(decay, _Split) else decay
        h = jnp.where(_crossed(leading)[:, None], 0.0, h)
        factors = jax.tree.map(lambda part: part[:, None], decay)
        held = _scaled(jnp.where(index % apart == 0, h, held) if apart else h, factors)
        h = _scaled(h, factors) * along[None, :]
        if corrected:
            h = h - strength * a[:, None] * (_portable(c[None, :], h) if entered is None else entered)
        h = h + b[:, None] * value[None, :]
        read_from = held if apart else h
        if reader_scale is not None:
            read = _read(read_from, q, values_read, reader_scale)
        else:
            read = (_portable(read_from, q[:, None])[:, 0] if values_read
                    else _portable(q[None, :], read_from)[0])
        return (h, held, index + 1), read + r

    (h, _, _), outputs = lax.scan(step, (h, h, 0), (stepped, B, s, C, Q, R, values, valued, A, back))
    return outputs, h


def _carry_type(h, *parts) -> jnp.dtype:
    """The type a token walk carries its state in, the token loop's and the replay's alike: the state's and
    its update's parts' (each token's decay, input and values, its correction's strength and vector, its
    decay along the values and its erase's vector; its readout's none), float32 at least, so operands wider
    than the state widen it once (a float32 state's walk over float64 operands, x64's)."""
    return at_least_fp32(jnp.result_type(h, *parts))


def _substituted(lower, given):
    """`(I + lower)^-1 given` by forward substitution, `lower` strictly lower: a row a step, each reading the
    rows before it."""
    def row(index, solved):
        return solved.at[index].set(given[index] - _portable(lower[index][None, :], solved)[0])

    return lax.fori_loop(0, given.shape[0], row, jnp.zeros_like(given))


def _token(parts: linear_scan.Parts, token: tuple) -> tuple:
    """One token's exponent, input vector `B` and values `U`, correction strength `s` and vector `C`, reader
    `Q`, free readout `R` (zeros where the readout has none) and exponent along the values `E` (zeros where
    its update decays none, `linear_scan.Parts.across`): the affine map's parts
    (`linear_scan.Parts`) at the token's values, each in float32 at least."""
    shape, along = token_terms(parts, token)
    carried = at_least_fp32(aval(parts.update.jaxpr.invars[0]).dtype)
    keys, values = shape
    B, U = along(parts.left, keys), along(parts.right, values)
    corrects = parts.correction is not None
    # A correction `B (X^T D h)` is the transition's `-s B C^T D h`: `C = -X`, of unit strength.
    C = -along(parts.correction, keys) if corrects else jnp.zeros_like(B)
    s = jnp.asarray(1.0 if corrects else 0.0, carried)
    exponent = along(parts.exponent, keys if parts.per_key else 1)
    reader, out = (values, keys) if parts.values_read else (keys, values)
    across = jnp.zeros((values,), carried) if parts.across is None else along(parts.across, values)
    erase = jnp.zeros_like(B) if parts.erase is None else along(parts.erase, keys)
    return exponent, B, U, s, C, along(parts.reader, reader), along(parts.free, out), across, erase


def token_terms(parts, token: tuple, states: int = 1) -> tuple[tuple[int, ...], Callable]:
    """The state's shape and `along(term, extent)`: a term of the update's or readout's parts free of the
    state (`linear_scan.Parts`', `Coupled`'s) at one token's values, `[extent]` in float32 at least, its
    traced programs evaluated from `states` zero states (a pair's two) and `token`."""
    shape = aval(parts.update.jaxpr.invars[0]).shape
    zero = jnp.zeros(shape, aval(parts.update.jaxpr.invars[0]).dtype)
    carried = at_least_fp32(zero.dtype)  # the parts' constants': a float64 run's are float64
    envs: dict = {}

    def read(index: int):
        traced, atom = parts.atoms[index]
        if isinstance(atom, jcore.Literal):
            return jnp.asarray(atom.val)
        env = envs.setdefault(id(traced), environment(traced, (*(zero,) * states, *token), {}))
        evaluate(traced, env, read_by(traced.jaxpr, {atom}))
        return env[atom].astype(at_least_fp32(env[atom].dtype))

    def along(term, extent: int | None):
        """`term`'s value broadcast to `[extent]`, or as it is where `extent` is None (a scalar)."""
        value = linear_scan.value_of(term, read, shape[0], carried)
        if extent is None:
            return value
        return jnp.broadcast_to(jnp.reshape(value, (-1,)) if jnp.ndim(value) == 2 else value, (extent,))

    return shape, along


def subnormal_reader_scale(scale: float | np.floating, dtype) -> np.floating | None:
    """The typed scale to hold apart if multiplying it into the reader could flush it."""
    if jnp.issubdtype(dtype, jnp.floating) and 0 < abs(float(scale)) < float(jnp.finfo(dtype).tiny):
        return dtype.type(scale)
    return None


def _subnormal_contract(axes: Sequence[str], output: str, operands: tuple, scale, dtype):
    """Contract operands and a subnormal scale as unlanded factors, not as an underflowed input."""
    reduced = tuple(dict.fromkeys(tag for used in axes for tag in used if tag not in output))
    layout = (*output, *reduced)
    factors = [laid(operand.astype(dtype), used, layout)
               for used, operand in zip(axes, operands, strict=True)]
    factors.append(jnp.asarray(np.asarray(scale, dtype)))
    over = tuple(range(len(output), len(layout)))
    terms = ((1.0, tuple(("v", i) for i in range(len(factors)))),)
    form = ratios.Ratio(terms, (), 0, reduced=False, units=windows.Units(), over=over, ieee_products=True)
    value = ratios.ratio(factors, form)
    return jnp.squeeze(value, axis=over) if over else value


@functools.cache
def _scaled_contract(subscripts: str, scale: float | np.floating, scale_dtype: np.dtype | None):
    """A multilinear contraction whose transposes apply its scalar before their products. Equal-valued
    literals of different dtypes promote differently, so the cache keeps their dtype too."""
    from jax.custom_derivatives import SymbolicZero

    from dew.nn.kernels import generation
    from dew.nn.sharding import varied_alike

    inputs, output = subscripts.split("->")
    axes = inputs.split(",")

    def value(*operands):
        factor = scale if scale_dtype is None else jnp.asarray(scale, dtype=scale_dtype)
        first = jnp.result_type(factor, operands[0])
        if subnormal_reader_scale(scale, first) is not None:
            return _subnormal_contract(axes, output, operands, scale, jnp.result_type(factor, *operands))
        return _portable_einsum(subscripts, factor * operands[0], *operands[1:])

    def tangent(primals, tangents):
        out = value(*primals)
        live = tuple(index for index, part in enumerate(tangents) if not isinstance(part, SymbolicZero))
        if not live:
            return out, jnp.zeros_like(out)
        moving = tuple(tangents[index] for index in live)

        def applied(held, directions):
            return sum(rule(*(direction if at == index else part for at, part in enumerate(held)))
                       for index, direction in zip(live, directions, strict=True))

        def pulled(held, cotangent):
            cotangents = []
            for index in live:
                others = tuple(part for at, part in enumerate(held) if at != index)
                remaining = [axis for at, axis in enumerate(axes) if at != index]
                transposed = ",".join((*remaining, output)) + "->" + axes[index]
                # A strong scalar can widen a contraction whose operand's cotangent stays narrower.
                cotangents.append(_scaled_contract(transposed, scale, scale_dtype)(
                    *others, cotangent).astype(held[index].dtype))
            return tuple(cotangents)

        return out, generation.linear_map(applied, pulled, primals, moving)

    rule = contracted("test_the_chunked_delta_rule_holds_every_order")(
        declared_tangent(Protected(), value, tangent, symbolic_zeros=True))

    def contraction(*operands):
        # The casts' transposes sum a varying cotangent back to each whole operand.
        return rule(*varied_alike(operands))

    return contraction


def _pairing_role(left) -> str:
    """A pairing's products' role (`linear_scan.Chunk.gram`), by its rows: a correction's (the token `C`) a
    system's `gram`, a reader's the chunk's `scores`."""
    from egglog import get_callable_args

    token = get_callable_args(left, linear_scan.Chunk.token)
    return "gram" if token is not None and terms.text(token[0]) == "C" else "scores"


def _paired(left, right, between, issue: Callable | None = None, role: str = "scores", *, apart: int = 0,
            back: bool = False):
    """Each pair's decayed contraction over keys, `sum_k left_ik right_jk between_ijk` `[C, C]`: a decay a
    key dimension's over the pairs' every key; a scalar decay's (`between` `[C, C, 1]`) the keys'
    product decayed after it, `(left right^T) between`, which spans no `[C, C, K]` tensor (Qwen3.5-27B's
    derived forward took 66 ms on an H100 spanning it, derived-cells-h100-6d105a908). A per-key decay's
    pairs factor around each sub-chunk's reference (`_factored`): a product of the rows decayed to it by the
    keys decayed from it, the diagonal sub-chunks' alone elementwise. A pair no decay joins (`between` zero,
    a token's future) is zero, selected, not multiplied: its product, or its cotangent, may be infinite near
    float32's largest (a future corrected write's), whose product by a zero decay is NaN in every query's
    gradient the pair reads (the chunked op's reverse mode, near the largest) and in the decays'. A pairing
    `apart` by groups of that many tokens pairs no two of one group (`j // apart < i // apart`: a gram's own
    token, a group of one; a readout's group, `Scan.group`). A pair is out where its decay is crossed
    (`_crossed`), not where it rounds to zero: a live pair's that underflows reads a NaN as the recurrence
    does (train 1079's review F1), and any other product as zero (`_kept`). Decays a token `back`
    (`_Decays.between_before`) factor as `_factored` states."""
    joined = between
    if apart:
        rows, columns = (jnp.arange(between.shape[axis])[index] // apart
                         for axis, index in ((0, (slice(None), None)), (1, (None, slice(None)))))
        joined = jnp.where((rows > columns)[:, :, None], between, -0.0)
    active = jnp.any(~_crossed(joined), axis=-1)
    issue = issue or _issued
    if between.shape[-1] == 1:
        # The product selected before its decay multiplies it: an inactive pair's may overflow, whose zero
        # decay's derivative would read it.
        products = jnp.where(active, issue(role, left, right.T), 0.0)
        return _kept(products, joined, ~jnp.isnan(products))[..., 0] * joined[..., 0]
    # The factors read the reference's own decay (1, on the diagonal a strict pairing excludes).
    return jnp.where(active, _factored(left, right, between, joined, issue, role, back=back), 0.0)


def _pair_sum(factors: tuple[jax.Array, ...]) -> jax.Array:
    """Contract the leading key axis, keeping the crossed mask as a held last factor."""
    term = (*(("v", i) for i in range(len(factors) - 1)), ("m", len(factors) - 1))
    form = ratios.Ratio(((1.0, term),), (), 0, reduced=True, units=windows.Units(), ieee_products=True)
    return ratios.ratio(list(factors), form)


def _tiny_factor(operand, decay, product):
    def nonzero(value):
        _, _, exponent, fraction = float_bits.raw_fields(value)
        return (exponent != 0) | (fraction != 0)

    return (jnp.abs(product) < jnp.finfo(product.dtype).tiny) & nonzero(operand) & nonzero(decay)


def _pair_factors(left, right, decay, after):
    """Contract channels first; a crossed key contributes neither a value nor a tangent."""
    left, right = (jnp.moveaxis(part, -1, 0) for part in (left, right))
    first, second = jnp.moveaxis(decay, -1, 0), None if after is None else jnp.moveaxis(after, -1, 0)
    if second is None:
        active, factors = ~_crossed(first), (first,)
    else:
        factors = (first[..., :, None], second[..., None, :])
        active = ~(_crossed(factors[0]) | _crossed(factors[1]))
    # Keep the broadcast axes on the numeric inputs: the ratio's transpose reduces them before landing.
    return left[..., :, None], right[..., None, :], *factors, active.astype(left.dtype)


def _portable_pair_jvp(primals, tangents):
    live = tuple(t is not None and not isinstance(t, SymbolicZero) for t in tangents)
    value = _portable_pairs(*primals)
    if not any(live):
        return value, SymbolicZero(jax.typeof(value).to_tangent_aval())
    operands, fill = holes(primals, live, is_leaf=lambda arg: arg is None)

    def exact(*active):
        return _pair_sum(_pair_factors(*fill(active)))

    directions = tuple(t for t, take in zip(tangents, live, strict=True) if take)
    return value, jax.jvp(exact, tuple(operands), directions)[1]


@custom_vmap
def _pair_dispatch(left, right, decay, after):
    if after is None:
        decayed = right[..., None, :, :] * decay
        tiny = jnp.any(_tiny_factor(right[..., None, :, :], decay, decayed))

        ordinary = _portable_einsum("...ik,...ijk->...ij", left, decayed)
    else:
        leading, trailing = left * decay, right * after
        tiny = jnp.any(_tiny_factor(left, decay, leading)) | jnp.any(_tiny_factor(right, after, trailing))
        ordinary = _portable_einsum("...ik,...jk->...ij", leading, trailing)

    # A dot can overflow one channel before another cancels it. Keep the ordinary finite path,
    # but let the existing scaled sum decide any nonfinite result or flushed pre-product.
    repair = tiny | jnp.any(~jnp.isfinite(ordinary))
    return lax.cond(repair, lambda: _pair_sum(_pair_factors(left, right, decay, after)), lambda: ordinary)


_pair_dispatch.def_vmap(functools.partial(scalar_case_batch, _pair_dispatch))


_portable_pairs = contracted("test_a_decayed_pair_holds_its_derivatives")(
    declared_tangent(Protected(), _pair_dispatch, _portable_pair_jvp, symbolic_zeros=True))


def _factored(left, right, between, joined, issue: Callable, role: str, *, back: bool = False):
    """A per-key decay's pairs `sum_k left_ik right_jk between_ijk` over sub-chunks of `SUB` rows
    (`recurrence_algebra.SUB`, fla's BC): a pair of an earlier sub-chunk's key `j` and a later one's row `i`
    is the product of the row decayed to the reference `p` before its sub-chunk and the key decayed from it,
    with a dot unless a factor underflows; then the portable contraction holds all factors until the sum
    lands (row 12888). A sub-chunk's own pairs are elementwise over its keys (at most SUB^2 K a sub-chunk,
    not C^2 K), by the pairing's own decays (joined less a strict pairing's diagonal). Decays a token back
    (between_before, row i
    `between`'s row `i - 1`) factor as `between_back_ip between_back_(p+1)j`, the key's factor read at the
    reference's next row, its own sub-chunk's first: `between_back_pj` is a token short of `p`, which the
    pair's whole decay spans (`exp(G_{i-1} - G_p) exp(G_p - G_j)`)."""
    from dew.nn.kernels import recurrence_algebra

    count = left.shape[0]
    sub = recurrence_algebra.SUB
    work = at_least_fp32(jnp.result_type(left, right, joined))
    starts = range(0, count, sub)
    rows = []
    for first in starts:
        own = slice(first, min(first + sub, count))
        blocks = []
        for start in starts:
            keys = slice(start, min(start + sub, count))
            if start > first:
                blocks.append(jnp.zeros((own.stop - own.start, keys.stop - keys.start), work))
            elif start == first:
                decay = joined[own, keys]
                blocks.append(_portable_pairs(left[own].astype(work), right[keys].astype(work),
                                              decay.astype(work), None) if issue is _issued else
                              _portable_einsum("ik,ijk->ij", left[own], right[None, keys] * decay))
            else:
                reference = first - 1
                before, after = between[own, reference], between[reference + back, keys]
                if issue is _issued:
                    blocks.append(_portable_pairs(left[own].astype(work), right[keys].astype(work),
                                                  before.astype(work), after.astype(work)))
                else:
                    blocks.append(issue(role, left[own] * before, (right[keys] * after).T))
        rows.append(jnp.concatenate(blocks, axis=1))
    return jnp.concatenate(rows, axis=0)


def spread(readers, values, keys, between, *, weights=None, issue: Callable | None = None):
    """Each token's earlier writes read along the state's values and decayed to it along its keys, `sum_{j <=
    i} (readers_i . values_j) keys_jk between_ijk` `[C, K]` (`linear_scan.Chunk.spread`): the reads a
    product `issue` makes (`scores`; the portable product where None, `_issued`), a pair no decay joins (a
    token's future, or one through a vanished decay) selected out. `weights` `[C, C]`, each pair's read
    multiplied before it is applied (log-linear's level weights, kernel-templates.md Appendix B), none where
    None. A scalar decay's pairs decay the reads, then a product applies them; a decay a key dimension's
    factor around each sub-chunk's reference as `_factored`'s (`recurrence_algebra.SUB`): an earlier
    sub-chunk's keys decayed to the reference `p` before the row's sub-chunk and read, the row's decay from it
    after, each factor at most 1; a sub-chunk's own pairs elementwise over its keys.

    A read that overflows, formed before its decay (the token order's, row 5426's contract), reads zero
    through a decay that underflows to zero, each key's by the pair's whole decay at that key (`_kept`): the
    term the recurrence decays to its range there, which float32's scores-then-decay order cannot form,
    where infinity times that zero was NaN (rows 9547, 9631 and train 1079's review F1). A NaN read reads
    NaN."""
    issue = issue or _issued
    active, scalar = jnp.any(~_crossed(between), axis=-1), between.shape[-1] == 1
    reads = jnp.where(active, issue("scores", readers, values.T), 0.0)
    if scalar:
        # Out before a weight multiplies it: an overflowing product times a level's zero weight is NaN.
        reads = _kept(reads, between, ~jnp.isnan(reads))[..., 0]
    if weights is not None:
        reads = reads * weights
    if scalar:
        return issue("apply", reads * between[..., 0], keys)
    return _each_decayed(reads, keys, between)


def _each_decayed(reads, rows, between):
    """`sum_{j <= i} reads_ij rows_jk between_ijk` `[C, K]`: pairs' reads `[C, C]` applied to the earlier
    tokens' `rows` `[C, K]`, each column decayed to the reader by its own `between` `[C, C, K]` (`spread`'s
    keys, `weighed`'s values along a decay across them), factored around each sub-chunk's reference
    (`recurrence_algebra.SUB`): an earlier sub-chunk's rows decayed to the reference `p` before the row's
    sub-chunk and read a column at a time, the row's decay from it after, each factor at most 1, which keeps a
    pair whose whole decay underflows; an overflowed read zero at each column its whole decay underflows
    (`_kept`)."""
    from dew.nn.kernels import recurrence_algebra

    count = rows.shape[0]
    sub = recurrence_algebra.SUB
    found = []
    for first in range(0, count, sub):
        own = slice(first, min(first + sub, count))
        mine = reads[own, own]
        row = _summed_pairs(_kept(mine, between[own, own], jnp.isinf(mine)), rows[own], between[own, own])
        if first:
            # The earlier sub-chunks' rows decayed to the reference before the row's sub-chunk, each pair's
            # read then the row's decay from the reference, each factor at most 1.
            reference, earlier = first - 1, reads[own, :first]
            kept = _kept(earlier, between[own, :first], jnp.isinf(earlier))
            row = row + _summed_pairs(kept, rows[:first] * between[reference, :first],
                                      between[own, reference][:, None, :])
        found.append(row)
    return jnp.concatenate(found, axis=0)


def _summed_pairs(reads, rows, decays):
    """`sum_j reads_ijk rows_jk decays_ijk` `[R, K]`, each pair's product formed before the sum in the order
    whose intermediate is a normal float32 (`precision.twice_scaled`): its read times its row, or times its
    decay first, so a pair's term overflows only where it does. Its rows read and summed first, two reads of
    2^127 under a decay of 1/4 to the reader summed past the largest before it, where each pair's term is
    2^125 (row 9029's head review)."""
    return jnp.sum(twice_scaled(reads, rows[None], decays), axis=1)


def _kept(reads, between, dropped):
    """Pairs' reads `[R, C]` a key at a time `[R, C, K]`, each of `dropped` `[R, C]` zero where that key's
    decay (`between` `[R, C, K]`, a scalar decay's `K` one) is zero, selected before the decay multiplies it:
    a live pair's decay that underflows (`_crossed`'s positive zero) weighs its read to zero, which, formed
    before that decay, may overflow, infinity times zero (a query's and a key's 2^70 under decays of
    exp(-160), head review of 512810acf F7). A scalar decay's pairs drop every read but a NaN, which reads NaN
    as the recurrence's (train 1079's review F1); a decay a key's the overflowed reads alone, each key's other
    reads its factors' product, which its whole decay loses (row 9549), across a sub-chunk too, where a read
    zeroed only at a decay zero at every key was NaN at a key whose decay alone was (row 9631)."""
    return jnp.where(dropped[..., None] & (between == 0), 0.0, reads[..., None])


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


def _issued(name: str, left, right):
    """A derived scan's products as the portable math's (`recurrence_algebra.portable`), the doubling's
    among them (`recurrence_algebra.doubled`'s issuer); `transfer_decay` a transfer's columns scaled,
    `right * left`; a readout (`carry.EMIT_ROLES`) the portable math's (`recurrence_algebra.remembered` by
    its issuer), its gradient in what it reads summed from the chunk's last token (row 4466: from the first,
    the entering memory's gradient 1.9 times the portable math's distance from float64)."""
    if name == "transfer_decay":
        return right * left[None, :]
    if name in carry.EMIT_ROLES:
        from dew.nn.kernels import recurrence_algebra

        return recurrence_algebra.remembered(left, right, recurrence_algebra.portable, name)
    return _portable(left, right)


def _portable(left, right):
    """`left @ right` as the portable math issues it (`recurrence_algebra.portable`, at its
    `STATE_PRECISION`): the derived scan's products carry the recurrent state."""
    # The template's algebra imports the compile point, which imports this package: read at a call.
    from dew.nn.kernels import recurrence_algebra

    return recurrence_algebra.portable("product", left, right)


def _portable_einsum(subscripts: str, *operands):
    """`_portable`'s contraction as an einsum (`recurrence_algebra.portable_einsum`)."""
    from dew.nn.kernels import recurrence_algebra

    return recurrence_algebra.portable_einsum(subscripts, *operands)


def _inverse_cotangent(lower, d_lower):
    """The overlaps' cotangent of `(I + overlaps)^-1 - I` (`lower`) from its own: with `X = I + lower`,
    `-X^T d_lower X^T` below the diagonal."""
    back = _applied(lower.T, d_lower)
    return _below(-_applied(lower, back.T).T)


def _applied(lower, given):
    """`(I + lower) @ given`, the identity outside the product."""
    return given + _portable(lower, given)


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


@functools.partial(nested, static_argnames=("axis", "reverse", "wrap", "chunk"), inline=True)
def compensated(values, *, low=None, axis: int = 0, reverse: bool = False, restarts=None,
                wrap: Callable | None = None, chunk: int | None = None) -> tuple[jax.Array, jax.Array]:
    """`values`' inclusive prefix sums along `axis` (suffix sums where `reverse`), each a float and its
    rounding error (a compensated scan), each restarting where `restarts` holds (a segmented scan: an
    element of it the first of its segment's, in either direction: a suffix sums to its segment's last).
    `low`, where given, is each value's own low part, the values then double floats themselves (an exact
    product's pair, `two_product`: Mamba-3's turns). Each pair joined renormalized (`renormalized_add`), the
    hand math's sum: a pair left unrenormalized held a residual that a product XLA:CPU fused into its sum (a
    rated decay's `rate * beta`, an FMA) had left wrong, 1.5e-5 off a 16-token prefix (row 1966).
    A nested program (`dew.nn.nested`), traced once for its arguments' types within a trace: the associative
    scan otherwise traces its protected additions afresh at every level on every call, each chunk's walked
    sum among them. `wrap`, where given, maps each sum of two pairs to an equal one modulo its period (an
    angle less its whole turns, `templates.recurrence.turned`'s frames). `chunk`, where given, walks the
    prefixes in chunks of that many elements (`_chunk_walked`), every element read once, where an associative
    scan reads each at every level (Mamba-3's frames took a step 13% past its kernels on a GH200 so, row
    7402): prefixes alone, a reverse scan's none."""
    low = jnp.zeros_like(values) if low is None else low
    # Each value read as one where it is made (`uncontracted`, or its additions marked to round alone), once:
    # the scan's partial sums are its additions', which its combine reads unfenced (`renormalized_add`), where
    # a fence at every level split its fusions (row 8498). A wrapped sum is no addition's, so a scan that
    # wraps its sums fences each combine's (`compensated_add`).
    with rounding_alone():
        (values,) = uncontracted(values)

    def renormalized(left, right):
        with rounding_alone():
            return renormalized_add(left, right)

    joined = renormalized if wrap is None else (lambda left, right: wrap(compensated_add(left, right)))
    if chunk is not None:
        if reverse:
            raise ValueError("compensated's chunk walks prefixes alone: a reverse scan takes chunk=None")
        return _chunk_walked(values, low, axis, restarts, joined, chunk)

    def combine(left, right):
        summed = joined(left[:2], right[:2])
        if restarts is None:
            return summed
        # Where the right part restarts, the left's sums are not its segment's.
        kept = right[2]
        return (jnp.where(kept, right[0], summed[0]), jnp.where(kept, right[1], summed[1]),
                left[2] | right[2])

    if restarts is None:
        return lax.associative_scan(combine, (values, low), reverse=reverse, axis=axis)
    if reverse:
        # A reverse scan combines its elements from the last, so a segment restarts there at its last
        # element: the one before each segment's first, and the last of all (none of an empty axis).
        before = (slice(None),) * (axis % jnp.ndim(restarts))
        restarts = jnp.concatenate([restarts[(*before, slice(1, None))],
                                    jnp.ones_like(restarts[(*before, slice(0, 1))])], axis=axis)
    high, low, _ = lax.associative_scan(combine, (values, low, restarts), reverse=reverse, axis=axis)
    return high, low


def _chunk_walked(values, low, axis: int, restarts, joined: Callable,
                  chunk: int) -> tuple[jax.Array, jax.Array]:
    """`compensated`'s prefixes walked in chunks of `chunk` elements along `axis`, each pair `joined` to the
    next: each chunk's own a walk of its elements, each chunk's entering the walk of the chunks' ends, a
    segment's first (`restarts`) starting its sums from zero within its chunk and the chunks' walk from its
    chunk's end."""
    restarts = (jnp.zeros(jnp.shape(values), bool) if restarts is None
                else jnp.broadcast_to(jnp.asarray(restarts, bool), jnp.shape(values)))
    values, low, restarts = (jnp.moveaxis(x, axis, 0) for x in (values, low, restarts))
    length, rest = values.shape[0], values.shape[1:]
    if length == 0:
        return jnp.moveaxis(values, 0, axis), jnp.moveaxis(low, 0, axis)
    pad = (-length) % chunk
    chunks = (length + pad) // chunk

    def laid(x):
        x = jnp.pad(x, [(0, pad)] + [(0, 0)] * len(rest))
        return jnp.moveaxis(jnp.reshape(x, (chunks, chunk, *rest)), 1, 0)

    def within(total, token):
        value, part, restart = token
        total = tuple(jnp.where(restart, jnp.zeros((), each.dtype), each) for each in total)
        total = joined(total, (value, part))
        return total, total

    zero = jnp.zeros((chunks, *rest), values.dtype)
    flags = laid(restarts)
    _, local = lax.scan(within, (zero, zero), (laid(values), laid(low), flags))

    def across(entering, end):
        sums, broke = end
        leaving = joined(entering, sums)
        return tuple(jnp.where(broke, e, part) for e, part in zip(sums, leaving, strict=True)), entering

    ends = tuple(part[-1] for part in local)
    _, offsets = lax.scan(across, (zero[0], zero[0]), (ends, jnp.any(flags, axis=0)))
    seen = jnp.cumsum(flags.astype(jnp.int32), axis=0) > 0
    summed = joined(local, tuple(part[None] for part in offsets))
    high, low = (jnp.where(seen, part, whole) for part, whole in zip(local, summed, strict=True))

    def unlaid(part):
        return jnp.moveaxis(jnp.reshape(jnp.moveaxis(part, 0, 1), (chunks * chunk, *rest))[:length], 0, axis)

    return unlaid(high), unlaid(low)


class _Split(NamedTuple):
    """A decay `exp(x)` as three factors, each a normal float32 or one: `first`, `exp(max(x, -L))`, `second`,
    `exp(max(x - first's, -L))`, and `rest`, the remainder's (`L` its dtype's least normal's exponent,
    `_normal_floor`; `_exponents`), applied in turn (`_scaled`): each at most 1, so a value's product with
    them falls monotonically to its own. A decay alone past fp32's least normal (three gates of -40,
    `e^-120`) never zeroes a write of 2^100 whose decayed value (`e^-51`) is normal (row 9660), nor one past
    two normal factors' reach (2^-252: three gates of -58.375, `e^-175.125`) a state of 2^127 whose decayed
    value (`e^-87.09`, 1.5e-38) is normal, where two factors' second was subnormal and flushed to zero.
    Three normal factors reach 2^-378, past any finite value's way to a nonzero float32 (2^-277)."""

    first: jax.Array
    second: jax.Array
    rest: jax.Array


class _Decays(NamedTuple):
    """A chunk's decays from its tokens' exponents (`_decays`), each per key: `entering`, the decay from the
    chunk's start through each token, `exp(G_i)`; `to_last`, from after each token to the chunk's end,
    `exp(G_last - G_j)`; `whole`, the chunk's, `exp(G_last)`, each a decay a value takes alone (`_Split`);
    `between`, from after token `j` through token `i`, `exp(G_i - G_j)` for `j <= i` and zero above the
    diagonal, which pairs take beside their reads; and `entering_before` and `between_before`, `entering`'s
    and `between`'s a token back, through the token before each (`exp(G_{i-1})`, one at the first;
    `exp(G_{i-1} - G_j)` for `j < i`), as corrections reading the state a token enters read them
    (`linear_scan.Parts.before`): shifted from them, never a token's exponent taken away."""

    entering: _Split
    to_last: _Split
    whole: _Split
    between: jax.Array
    entering_before: _Split
    between_before: jax.Array


DECAYS = len(_Split._fields) * 4 + 2
"""The chunk function's leading operands: its decays (`_flat_decays`' leaves: `entering`, `to_last`, `whole`
and `entering_before` each a split's factors, then `between` and `between_before`), its tokens' values
after."""


def _flat_decays(decays: _Decays) -> tuple:
    """`decays`' arrays in the chunk function's order (`DECAYS` of them): each split decay's two factors,
    `entering`, `to_last`, `whole` and `entering_before`, then `between` and `between_before`."""
    return (*decays.entering, *decays.to_last, *decays.whole, *decays.entering_before, decays.between,
            decays.between_before)


def _decays_from(leaves: Sequence) -> _Decays:
    """`_flat_decays`' arrays as their decays."""
    width = len(_Split._fields)
    entering, to_last, whole, before = (_Split(*leaves[at * width:(at + 1) * width]) for at in range(4))
    return _Decays(entering, to_last, whole, leaves[4 * width], before, leaves[4 * width + 1])


def _scaled(value, decay):
    """`value` times a decay (`_Split`'s factors in turn, else the array)."""
    for factor in decay if isinstance(decay, _Split) else (decay,):
        value = value * factor
    return value


def _normal_floor(dtype, base2: bool) -> float:
    """The least exponent whose decay (`exp`, or `exp2` with `base2`) is a normal number of `dtype`."""
    return math.ceil(math.log(float(jnp.finfo(dtype).tiny)) / (math.log(2.0) if base2 else 1.0))


def _exponents(exponent, floor: float) -> tuple:
    """`exponent` (at most 0) as `_Split`'s three, each held from below at `floor` but the last (the rest),
    summing to it exactly (each a difference of a multiple of `exponent`'s ulp and an integer `floor`, its
    magnitude at most `exponent`'s)."""
    first = jnp.maximum(exponent, floor)
    second = jnp.maximum(exponent - first, floor)
    return first, second, exponent - first - second


def _token_factors(exponent, base2: bool) -> _Split:
    live, cut = vanished(exponent, base2)
    if not base2:
        return _factors(base2, live, ~cut)
    # A native exp2 at the least normal can round below it before multiplying the state.
    # Put back an exact integral power after exponentiating just the fractional remainder.
    def formed(part):
        power = jnp.floor(part)
        return _exp(base2, part - power) * float_bits.exact_power2(power.astype(jnp.int32), part.dtype)

    first, second, rest = map(formed, _exponents(live, jnp.finfo(exponent.dtype).minexp))
    return _Split(jnp.where(cut, -0.0, first), second, rest)


def _factors(base2: bool, exponent, kept) -> _Split:
    """`exp(exponent)` (`exp2` where `base2`) as its factors (`_Split`, `_exponents`) where `kept`, else a
    crossed decay's negative zero (`_crossed`) and ones; each exponent masked before its exponential."""
    exponent = jnp.where(kept, exponent, 0.0)
    first, *rest = (_exp(base2, part) for part in _exponents(exponent, _normal_floor(exponent.dtype, base2)))
    return _Split(jnp.where(kept, first, -0.0), *(jnp.where(kept, factor, 1.0) for factor in rest))


def scaled_exp(value, exponent):
    """`value` times `exp(exponent)`, `exponent` in natural units and at most 0 (a decay's: a walk's, a
    span's): the factor applied as three in turn (`_Split`'s, `_exponents`), the first two held at fp32's
    least normal exponent and the last the rest, each to fp32's rounding (`precision.accurate_exp`), so the
    product falls monotonically to the value's decayed one and keeps it where that is normal (row 9660: a
    decay of `e^-120` takes 2^100 to `e^-51`, which one exponential's underflow zeroed); an exponent of -inf
    zeroes a finite value and keeps a NaN or an infinite one NaN. Its derivatives are declared
    (`_scaled_exp_tangent`): the value's tangent and cotangent decayed alike, and the exponent's the value,
    the direction and the decay together, balanced so no partial product leaves fp32's range where the
    whole is in it (`_decayed`): autodiff through the factors overflowed (`2^127 * 2^127 e^-87`
    where `2^254 e^-90` is finite), and the direction times the decayed value read zero where that value
    underflowed (`2^120` times `1 e^-120`, its derivative 1.0e-16)."""
    return _scaled_exp(value, exponent)


def _scaled_exp_value(value, exponent):
    """`scaled_exp`'s value: the factors applied in turn."""
    for part in _exponents(exponent, _normal_floor(exponent.dtype, base2=False)):
        value = value * accurate_exp(part)
    return value


def _scaled_exp_tangent(primals, tangents):
    """`scaled_exp`'s value and tangent, a linear map of the moving tangents with its transpose declared
    (`generation.linear_map`): the value's tangent and cotangent decayed by the same factors, and the
    exponent's each the value, the direction and the decay together (`_decayed`, one rounding over the three
    as a scaled value's), never the decayed value rounded first nor a factor's partial product. The
    exponent's cotangent times the decayed value read zero where that value underflowed (1 decayed by
    `e^-120` under a cotangent of 2^120, its derivative 1.0e-16; train 1383's reviews), and autodiff through
    the factors overflowed (2^127 by 2^127 under `e^-90`, its derivative 2.4e37; train 1379's review K-F2).
    A symbolic zero tangent moves nothing (an overflowed value's zero tangent times it NaN)."""
    value, exponent = primals
    # The declared function again, so a derivative of this one meets this rule, no factor's product.
    out = _scaled_exp(value, exponent)

    def moved(held, index, direction):
        value, exponent = held
        return _scaled_exp(direction, exponent) if index == 0 else _decayed(exponent, value, direction)

    def pulled(held, index, cotangent):
        value, exponent = held
        return (_summed_to(_scaled_exp(cotangent, exponent), value) if index == 0
                else _summed_to(_decayed(exponent, value, cotangent), exponent))

    return out, _linear_tangent(primals, tangents, out, moved, pulled)


def _linear_tangent(primals, tangents, out, moved, pulled):
    """The tangent of a declared rule whose tangent is a sum of linear maps, one each moving tangent:
    `moved(primals, index, direction)` the map of tangent `index`, `pulled(primals, index, cotangent)` its
    transpose (`generation.linear_map`), a symbolic zero tangent none (`out`'s symbolic zero where every
    tangent is one)."""
    from dew.nn.kernels import generation

    live = tuple(index for index, part in enumerate(tangents) if not isinstance(part, SymbolicZero))
    if not live:
        return SymbolicZero(jax.typeof(out).to_tangent_aval())

    def applied(held, directions):
        parts = [moved(held, index, direction) for index, direction in zip(live, directions, strict=True)]
        return functools.reduce(jnp.add, parts).astype(out.dtype)

    # The map's working dtype, the rule's operands' promoted with fp32: each moving tangent enters the map in
    # it and its cotangent leaves in it, narrowed once outside, so a batch's or a mesh's sum of a broadcast
    # operand's cotangent (the map's transpose under `vmap`, a manual axis's) is taken in it: fp32 values
    # 256 and -255.5 under a shared bf16 exponent summed in bf16 read 0, not 0.5, and float64 values 2^130
    # and -2^130 + 2^126 summed in fp32 read inf less inf, not 2^126 (train 1398's and a head review).
    work = jnp.promote_types(jnp.result_type(*primals), jnp.float32)

    def transposed(held, cotangent):
        return tuple(pulled(held, index, cotangent.astype(work)).astype(work) for index in live)

    moving = tuple(tangents[index].astype(work) for index in live)
    return generation.linear_map(applied, transposed, primals, moving)


def _decayed_value(exponent, *factors):
    """`factors`' product times `exp(exponent)` (`exponent` at most 0, every array broadcast together) with
    one rounding into the format, a scaled value's (`dew.algebra.scaled`): each factor exact as a
    significand and a power of two, the decay a power of two and the exponential of its remainder
    (`scaled.base2`), their product's significands and exponents formed apart, landed once. No partial
    product leaves the format's range where the whole is in it: 2^127 by 2^127 under `e^-90`, 2^120 by 1
    under `e^-120`, 2^100 by 2^-120 under `e^-25`. An exponent past the cutoff, -inf or fp32's most negative
    among them, decays a finite product to zero (a non-finite one to NaN, no product of the factors formed:
    2^127 by 2^127 at -inf is zero)."""
    dtype = jnp.result_type(exponent, *factors)
    # The decomposition in at least fp32, each operand widened exactly (a 16-bit subnormal a normal fp32),
    # the product narrowed once at the end: in fp16 the decay's base-two split doubled -10 and its
    # two-product splitter (4097) overflowed, NaN (train 1398's review of 9b6609de0).
    work = jnp.promote_types(dtype, jnp.float32)
    exponent = jnp.asarray(exponent).astype(work)
    factors = [jnp.asarray(factor).astype(work) for factor in factors]
    form = scaled.format_of(work)
    # The least power of two the decay holds: beneath it every product of `factors` in the format (each
    # below `2^(largest + 1)`) decays below the format's least subnormal (`2^-reach`).
    cutoff = len(factors) * (form.largest + 1) + form.reach + 1
    # A decay past the cutoff (-inf among them) decays every finite product to zero, detected in natural
    # units before the base-two conversion, whose head overflowed at fp32's most negative exponent (its
    # remainder then -inf less -inf, NaN).
    gone = exponent < -(cutoff + 2) * math.log(2.0)
    high, low = scaled.base2(jnp.where(gone, 0.0, exponent))
    whole = jnp.round(high)
    # The remainder's exponential in [2^-1/2, 2^1/2], its power of two apart.
    remainder = accurate_exp(((high - whole) + low) * np.asarray(math.log(2.0), work))
    decay = scaled.normalized(remainder, jnp.clip(whole, -cutoff, 0).astype(jnp.int32), cutoff + 1)
    parts = [scaled.scaled(factor) for factor in factors]
    found = scaled.landed(functools.reduce(scaled.mul, parts, decay))
    unbounded = functools.reduce(jnp.logical_or, [~jnp.isfinite(factor) for factor in factors])
    return jnp.where(gone, jnp.where(unbounded, np.asarray(np.nan, work), 0), found).astype(dtype)


def _decayed_tangent(primals, tangents):
    """`_decayed`'s value and tangent, linear maps of the moving tangents (`_linear_tangent`): a factor's
    tangent in its place, and the exponent's beside the factors, each another `_decayed`, so every order
    meets this rule again and no order forms a partial product."""
    exponent, *factors = primals
    out = _decayed(exponent, *factors)

    def moved(held, index, direction):
        exponent, *factors = held
        if index == 0:
            return _decayed(exponent, direction, *factors)
        return _decayed(exponent, *factors[:index - 1], direction, *factors[index:])

    def pulled(held, index, cotangent):
        exponent, *factors = held
        if index == 0:
            return _summed_to(_decayed(exponent, cotangent, *factors), exponent)
        others = (*factors[:index - 1], *factors[index:])
        return _summed_to(_decayed(exponent, cotangent, *others), factors[index - 1])

    return out, _linear_tangent(primals, tangents, out, moved, pulled)


def _summed_to(cotangent, like):
    """`cotangent` summed over the axes `like` broadcast across and over each manual mesh axis it varies over
    that `like` does not (`sharding.summed_to`: a replicated exponent's cotangent each shard's partial of a
    value split across them, which JAX's accumulator refuses unsummed), in `like`'s shape and in the
    cotangent's dtype promoted with `like`'s and fp32, never narrowed, so every sum is taken wide (the dtype
    `_linear_tangent`'s map works in); and placed as `like` is where an Explicit axis places it, as
    `templates.gemm.factored` places its rows' (a replicated value's cotangent split as a split exponent's
    was; train 1398's review)."""
    from dew.nn import sharding

    summing = jnp.promote_types(jnp.promote_types(cotangent.dtype, like.dtype), jnp.float32)
    extra = cotangent.ndim - like.ndim
    axes = (*range(extra), *(extra + axis for axis, size in enumerate(like.shape)
                              if size == 1 and cotangent.shape[extra + axis] != 1))
    total = sharding.summed_to(jnp.sum(cotangent.astype(summing), axis=axes).reshape(like.shape), like)
    placed = sharding.explicit_placement(jax.typeof(like))
    return total if placed is None else jax.sharding.reshard(total, placed)


_decayed = contracted("test_a_decayed_product_holds_every_order")(
    declared_tangent(Protected(), _decayed_value, _decayed_tangent, symbolic_zeros=True))


_scaled_exp = contracted("test_scaled_exp_holds_every_order")(
    declared_tangent(Protected(), _scaled_exp_value, _scaled_exp_tangent, symbolic_zeros=True))


def _decays(exponents, base2: bool) -> _Decays:
    """`exponents`' decays (`_Decays`, `exp` or `exp2` of the sums), each exponent a partial sum of the
    compensated prefix of its segment differenced from both parts (`_decays_value`): every exponent taken is
    at most zero, so each stays finite at any decay, at `tokens^2 keys` elementwise work a chunk (a sub-chunk
    factoring around a reference prefix is the faster equal form, the extraction's choice under a kernel
    cost)."""
    return _decays_of(exponents, base2)


def between(exponents, *, base2: bool = False) -> jax.Array:
    """`exponents` `[T, K]`'s decays of the spans after token `j` through token `i`, `[T, T, K]`, zero above
    the diagonal (`_decays`' `between`): compensated within each segment, a vanished token ending one, so no
    span's sum holds a -2**70 or two -2**127 (`_decays_value`)."""
    return _decays(exponents, base2).between


def vanished(exponents, base2: bool = False) -> tuple[jax.Array, jax.Array]:
    """`(live, cut)` of `exponents` (decays' natural logarithms, base 2's where `base2`): `live` the exponents
    with each vanished token's (`precision.vanishing`'s threshold: its decay zero) set to 0, and `cut` where
    a span restarts, at each vanished token. A prefix sums `live` and restarts at a `cut`, and a span reaching
    through a cut is zero, so no sum holds a vanished token's (two of -2**127 summed are -inf; one of -2**70
    loses a later -1 to its rounding). The one reading of the threshold: a walk's chunk (`_walked`), a
    chunk's spans (`_decays_value`) and a turned call's turns' gradient (`templates.recurrence.turned`) go
    through it, where three copies of it left the turned gradient's prefixes summing vanished tokens (row
    8426, audit KPM-02)."""
    cut = exponents <= vanishing(exponents.dtype, base2)
    return jnp.where(cut, 0.0, exponents), cut


def _walked(exponents, base2: bool) -> jax.Array:
    """A chunk's whole decay in natural units, per key, as a walk's state carries it (`Derived.step`,
    `merge`): its exponents' compensated sum, or the walk's vanished decay where a token's decay vanishes
    (`_held`; two of -2**127 summed are -inf, and its residual NaN)."""
    live, cut = vanished(exponents, base2)
    summed = _scanned(live)[-1] * (math.log(2.0) if base2 else 1.0)
    return _held(jnp.where(jnp.any(cut, axis=0), _vanished_walk(exponents.dtype), summed))


def _vanished_walk(dtype) -> float:
    """A walk's vanished decay in natural units: twice its dtype's vanished exponent (`vanishing`), so its
    factors applied in turn (`scaled_exp`) are zero, where a walk's decay of `e^-120` (three gates of -40)
    takes a state of 2^100 to a normal `e^-51`, which a floor at the vanished exponent itself (-105) zeroed
    (row 9660); its last factor at it normal (`e^-36` in fp32), so every decay above it is kept."""
    return 2.0 * vanishing(dtype, base2=False)


def _held(decay: jax.Array) -> jax.Array:
    """A walk's decay in natural units held from below at its vanished one (`_vanished_walk`): zero there,
    and a walk's sums of vanished chunks finite."""
    return jnp.maximum(decay, jnp.asarray(_vanished_walk(decay.dtype), decay.dtype))


def _decays_value(exponents, base2: bool) -> _Decays:
    """The decays (`_Decays`) of a chunk's tokens' exponents, each from the compensated prefix of its
    segment: a token whose exponent vanishes (`vanishing`, its decay and every decay through it zero) ends
    one and starts the next, taken as 0 within it, so no span's sum holds a vanished token's (a prefix
    holding -2**127 twice is -inf; one holding -2**70 loses a later -1 to its rounding, its decay 1 in place
    of `exp(-1)`), and each span through one is zero."""
    live, cut = vanished(exponents, base2)
    high, low = compensated(live, restarts=cut)
    segments = jnp.cumsum(cut.astype(jnp.int32), axis=0)
    count = exponents.shape[0]
    below = (jnp.arange(count)[:, None] >= jnp.arange(count)[None, :])[:, :, None]
    gaps = (high[:, None] - high[None, :]) + (low[:, None] - low[None, :])
    # A span holds a vanished token where its ends lie in two segments, or its start in the first's.
    joined = below & (segments[:, None] == segments[None, :])
    started, ending = segments == 0, segments == segments[-1]
    # Each exponent masked before its exponential, as each span's: one across a vanished token is no sum
    # of a live span's (`exp(100)` unselected is infinite, its derivative's zero times it NaN), and its
    # decay negative zero, which no exponential is (`_crossed`).
    between = jnp.where(joined, _exp(base2, jnp.where(joined, gaps, 0.0)), -0.0)
    entering = _factors(base2, high + low, started)
    return _Decays(entering=entering,
                   to_last=_factors(base2, (high[-1] - high) + (low[-1] - low), ending),
                   whole=_factors(base2, high[-1] + low[-1], started[-1]), between=between,
                   # A token back, `entering`'s and `between`'s shifted a row: through the tokens before the
                   # reader's alone, so a key whose decay vanishes at the reader is read whole (a document's
                   # start cuts them where the walk declares its resets, `_started`).
                   entering_before=_Split(*(_back(factor, first=1.0) for factor in entering)),
                   between_before=_back(between, first=-0.0))


def _started(decays: _Decays, resets) -> _Decays:
    """`decays` with their token-back decays (`entering_before`, `between_before`) cut at each document's
    start, `resets` `[C]` the walk's declared reset operand (`Scan.reset`; none where None): a correction
    reading the state a token enters reads a reset's zero, where its own exponent (`RESET_DECAY`) is no
    mark of one, a saturated gate's as deep."""
    if resets is None:
        return decays
    at = jnp.asarray(resets, bool)[:, None]
    entering = decays.entering_before
    return decays._replace(entering_before=_Split(jnp.where(at, -0.0, entering.first),
                                                  *(jnp.where(at, 1.0, factor) for factor in entering[1:])),
                           between_before=jnp.where(at[:, None], -0.0, decays.between_before))


def _back(array, first):
    """`array` a token back along its first axis (row `i` its row `i - 1`), its first row `first`."""
    return jnp.concatenate([jnp.full_like(array[:1], first), array[:-1]], axis=0)


def _forward_one(array):
    """`_back`'s transpose: row `i` row `i + 1`'s, its last row zero (its first row's dropped)."""
    return jnp.concatenate([array[1:], jnp.zeros_like(array[:1])], axis=0)


def _crossed(decay: jax.Array) -> jax.Array:
    """Where a chunk's decay (`_decays_value`'s) is cut: a span through a vanished token (a document's start,
    or a decay that alone underflows), or a pair's future, negative zero, which a live span's decay that
    underflows (positive zero) is not: the one reading the readout's select and the pairs' take."""
    # The sign bit read as the integer of its width's: `jnp.signbit` shifts by a constant, which a checked
    # `shard_map` refuses beside a varying decay.
    bits = jax.lax.bitcast_convert_type(decay, jnp.dtype(f"int{jnp.finfo(decay.dtype).bits}"))
    return (decay == 0) & (bits < 0)


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
    entering, to_last, whole = (_split_term(decay, cotangent) for decay, cotangent in
                                ((decays.entering, cotangents.entering), (decays.to_last, cotangents.to_last),
                                 (decays.whole, cotangents.whole)))
    # A decay a token back (`entering_before`, `between_before`) is its row before's: its term that row's.
    entering = entering + _forward_one(_split_term(decays.entering_before, cotangents.entering_before))
    between = (cotangents.between * decays.between
               + _forward_one(cotangents.between_before * decays.between_before))
    if between.shape[-1] > 1:
        # A decay a key's pairs' terms `[C, C, K]` formed once, ahead of their scans: fused into the first,
        # XLA:CPU formed them again in its every step, Kimi delta attention's derived forward and backward
        # 14.7 s against 1.24 at 2 x 1024 tokens, 4 heads of 128 (row 1966's probe at ea6a839c5ba, a CPU
        # host); a decay a head's, its pairs a key's alone, ran 147 ms fused against 161 so.
        between = lax.optimization_barrier(between)
    from_on = _scanned(entering, reverse=True)
    before = _earlier(_scanned(to_last))
    # Rectangle sums: each row's terms left of a column, then the rows from a column's own row down.
    left = jnp.swapaxes(_earlier(_scanned(jnp.swapaxes(between, 0, 1))), 0, 1)
    spanning = jnp.diagonal(_scanned(left, reverse=True), axis1=0, axis2=1).T
    return ((from_on + before + spanning + whole[None]) * weight,)


def _split_term(decay: _Split, cotangent: _Split):
    """A split decay's term of its exponent's gradient: the factor whose exponent moves with it
    (`_exponents`), its first's where its exponent is above the floor (the others one), its second's where
    the remainder is (the first the floor's, a constant), its rest's below both."""
    moved_first = jnp.where(decay.second < 1, cotangent.second * decay.second, cotangent.first * decay.first)
    return jnp.where(decay.rest < 1, cotangent.rest * decay.rest, moved_first)



def _earlier(sums) -> jax.Array:
    """Inclusive prefix sums along the first axis made exclusive by a shift (zero first), so no term is
    added and taken away again."""
    return jnp.concatenate([jnp.zeros_like(sums[:1]), sums[:-1]])


def _scanned(values, *, reverse: bool = False) -> jax.Array:
    """`values`' inclusive prefix sums along their first axis (suffix sums where `reverse`), compensated."""
    high, low = compensated(values, reverse=reverse)
    return high + low


def _exp(base2: bool, x):
    """A scan's decays, `2^x` or `e^x`, to fp32's rounding under every form, not the run's
    (`precision.Numerics.forms`, `precision.accurate_exp`): the default path must meet its upstream's bound,
    and a v5e's default fp32 exp, 1.5e-6 from float64, held Mamba-2's portable scan at 12.8 times its float32
    on a CPU and its layer cell past transformers' bound, where accurate the scan is 1.25 times (row 7062); a
    CPU's default is as accurate."""
    return accurate_exp(x, base2=base2)
