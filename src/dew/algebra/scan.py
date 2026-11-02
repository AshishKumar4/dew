"""A recurrence's chunk forms derived from its token update (docs/design/algebra.md §6.9, `Scan`): a state `h`
updated token by token, `h = update(h, x_t)`, read out each token, `o_t = readout(h, x_t)`.

The derivation reads the update's and the readout's jaxprs at one token, each value that reads the state as
a sum of terms (`_Term`): the state scaled, a rank-one correction of it (`a (c^T h)`), and a rank-one input
free of it (`B (x) u`), each scaled by a product of the token's scalars. The update must be affine in the
state: a decay `D_t = exp(z_t)` (or `exp2`) on the state, a scalar or each row's own (a per-key vector);
corrections, each `-s_t B_t (c_t^T D_t h)`, reading the decayed state along the input's own vector `B_t`;
and one input `B_t (x) u_t`. So `h_t = (I - s_t B_t c_t^T) D_t h_{t-1} + B_t u_t^T`: the gated delta rule
(a scalar decay, `s = beta`, `B = c = k`, `u = beta v`), Kimi's delta attention (a per-key decay), or with
no correction Mamba-2's state space. The readout must contract the
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

from dew.algebra import carry, program, terms
from dew.algebra.forms import Derived, Program, Refused, aval, environment, evaluate, read_by
from dew.nn.precision import at_least_fp32, float32_einsum, float32_matmul, two_sum

Axes = tuple[str, ...]
Atom = jcore.Var | jcore.Literal

RESET_DECAY = -1e4
"""The exponent a reset token's decay takes in place of its own (natural or base 2): the state entering it
is multiplied by `exp(-1e4)`, which is 0 in fp32. A token's own exponent is never positive, so every span
holding a reset sums to at most this and decays to exactly 0, and a span without one never sees it. Finite,
so a span's exponent is a difference of finite partials (an infinity less an infinity is a NaN), and a
kernel's segment sums formed as a matmul against a 0/1 triangle stay finite (an infinity times 0 is a NaN);
thousands of resets in a walk's sum stay far inside fp32's range, and the compensated prefix keeps each
difference of two partials exact past them."""

_KEEPING = frozenset({"convert_element_type", "reshape", "expand_dims", "squeeze", "copy", "copy_p"})
"""Equations that keep a value's elements and shape class (a cast, a unit-axis reshape)."""


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


@dataclasses.dataclass(frozen=True)
class _Term:
    """One term of a value that reads the state, or of a row it reads: `kind` "state" (`h`), "corrected"
    (`left (right^T h)`), "input" (`left (x) right`, free of the state), "read" (`right^T h`, a row) or
    "free" (`right`, a row's part free of the state); scaled by `sign` times the product of `factors`, the
    token's scalars. `left` and `right` are vectors of the traced jaxpr, along the state's first axis but
    an input's or a free row's `right`, along its second. A state term's `diag`, a first-axis vector, scales
    the state's rows (a per-key decay), and a read or a correction of it reads the rows so scaled."""

    kind: str
    factors: tuple = ()
    sign: float = 1.0
    left: Atom | None = None
    right: Atom | None = None
    diag: Atom | None = None

    def scaled(self, factor=None, sign: float = 1.0) -> _Term:
        return dataclasses.replace(self, factors=self.factors + ((factor,) if factor is not None else ()),
                                   sign=self.sign * sign)


@dataclasses.dataclass(frozen=True)
class _Symbolic:
    """A value that reads the state: `shape` "matrix" (the state's), "row" (a vector along its second axis)
    or "spread" (a row broadcast along the first axis), the sum of `terms`."""

    shape: str
    terms: tuple[_Term, ...]


@dataclasses.dataclass(frozen=True)
class _Affine:
    """A token's affine update and linear readout as the traced jaxprs' values: the decay's exponent
    (`exponent`, base 2 where `base2`, a scalar or each key's own where `per_key`), the input (`input`), the
    corrections (`corrections`, each with the decay among its factors, which `strength` leaves out), the
    readout's contraction (`read`) and its free terms (`free`)."""

    update: Program
    readout: Program
    exponent: Atom
    base2: bool
    per_key: bool
    decay: Atom
    input: _Term
    corrections: tuple[_Term, ...]
    read: _Term
    free: tuple[_Term, ...]


def derive_scan(declaration: Scan, *, saturation: program.Saturation) -> Derived:
    """`declaration`'s chunk forms (the module's account), split into what every chunk computes at once and
    what its carry computes in sequence (`carry.split`, under `saturation`'s limits): `init(shapes)` a zero
    state and no decay, `prepare(chunk)` a chunk's values that read no state (`Prepared`), `step(state,
    prepared)` the state after the chunk, `mapped(prepared, state)` the chunk's outputs from the state
    entering it, `merge(a, b)` two consecutive walks' states joined (the later walk's from zero),
    `finish(state, operands)` the final state; `backward` a chunk's adjoint, a cotangent for each operand
    but the reset's. Refused (`Refused`) where the update or the readout is not of the module's form."""
    names = tuple(name for name in declaration.operands if name != declaration.reset)
    shape, dtype = declaration.state
    if len(shape) != 2:
        raise Refused("a state of rank other than two (keys by values) has no chunk form here")

    def traced(function: Callable, chunk: Mapping):
        def named(h, *xs):
            return function(h, **dict(zip(names, xs, strict=True)))

        tokens = [jax.ShapeDtypeStruct(tuple(extent for extent, axis in zip(chunk[name].shape,
                                                                              declaration.operands[name],
                                                                              strict=True)
                                             if axis != declaration.walked), chunk[name].dtype)
                  for name in names]
        # Jit disabled, so a jitted library function's equations are the declaration's own.
        with jax.disable_jit():
            return jax.make_jaxpr(named)(jax.ShapeDtypeStruct(shape, dtype), *tokens)

    corrects: list = []  # whether a token corrects the state, once a chunk was read

    def parts(chunk: Mapping):
        affine = _affine(traced(declaration.update, chunk), traced(declaration.readout, chunk))
        corrects[:] = [bool(affine.corrections)]
        # An operand with no walked axis (a rate, a skip) is every token's.
        axes = [declaration.operands[name].index(declaration.walked)
                if declaration.walked in declaration.operands[name] else None for name in names]
        exponent, *values = jax.vmap(lambda *xs: _token(affine, xs), in_axes=axes)(
            *[chunk[name] for name in names])
        # The exponents summed in float32 at least: a narrower prefix loses a decay's digits to its partials'
        # magnitude, and float16's overflows past six resets (65504).
        exponent = exponent.astype(at_least_fp32(exponent.dtype))
        if declaration.reset is not None:
            exponent = jnp.where(chunk[declaration.reset][:, None], RESET_DECAY, exponent)
        return affine, (exponent, *values)

    # A walk's decay carried in its exponents' accumulation (`parts`' float32 at least; a float64 state's,
    # float64).
    carried = at_least_fp32(jnp.dtype(dtype))

    def init(shapes):
        del shapes
        return jnp.zeros((shape[0],), carried), jnp.zeros(shape, dtype)

    splits: dict = {}

    def prepare(chunk):
        affine, (exponent, *values) = parts(chunk)
        decays = _decays(exponent, affine.base2)
        natural = _walked(exponent, affine.base2)
        B, _, s, C, _, _ = values
        # The solve's inverse reads no state: an operand of the chunk's function, prepared with the rest.
        lower = _inverse_of(decays, B, s, C) if affine.corrections else jnp.zeros((B.shape[0],) * 2, B.dtype)
        given = (*decays, lower, *values)
        key = (affine.base2, bool(affine.corrections), *((value.shape, str(value.dtype)) for value in given))
        if key not in splits:
            h = jax.ShapeDtypeStruct(shape, dtype)
            avals = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in given]
            splits[key] = carry.split(functools.partial(_chunk, bool(affine.corrections)), h, avals, (1,),
                                      saturation=saturation,
                                      ranges=_ranges(declaration, names, chunk, affine, B.shape[0]),
                                      state_range=declaration.ranges.get("state"))
        return Prepared((natural, splits[key].prepare(*given)), splits[key])

    def step(state, prepared):
        decay, h = state
        natural, values = prepared.values
        (leaving,) = prepared.split.carry(values, h)
        return _held(decay + natural.astype(decay.dtype)), leaving

    def mapped(prepared, state):
        (outputs,) = prepared.split.emit(prepared.values[1], state[1])
        return outputs

    def merge(a, b, operands):
        del operands
        # A state's decay is its walk's exponent sum per key in natural units. A later walk from zero leaves
        # the state before it decayed alone only where no token corrects it: refused where one does.
        if corrects != [False]:
            raise Refused("a merge of walks whose tokens correct the state, or before any chunk was read")
        (decay_a, h_a), (decay_b, h_b) = a, b
        return _held(decay_a + decay_b), jnp.exp(decay_b)[:, None] * h_a + h_b

    def finish(state, operands):
        del operands
        return state[1]

    def backward(chunk, state, d_state, d_outputs):
        # The chunk's forms' adjoint: the reverse scan's step, each chunk's transition transposed, as the
        # forward forms' derivative (their prefix, substitution and Dew products differentiated).
        # The reset's booleans held, no operand of the derivative.
        def forward(h, operands):
            entering, prepared = (state[0], h), prepare({**chunk, **operands})
            return mapped(prepared, entering), step(entering, prepared)[1]

        _, pullback = jax.vjp(forward, state[1], {name: chunk[name] for name in names})
        d_h, d_chunk = pullback((d_outputs, d_state))
        return d_h, d_chunk

    return Derived(init, step, merge, finish, mapped,
                   {"rounding": "compensated prefix", "walked": declaration.walked}, backward, prepare)


@functools.partial(jax.tree_util.register_dataclass, data_fields=["values"], meta_fields=["split"])
@dataclasses.dataclass(frozen=True)
class Prepared:
    """A chunk's values that read no state (`carry.Split.prepare`), batched as a walk computes them for every
    chunk at once, and the split that reads them (static)."""

    values: tuple
    split: carry.Split


DECAYS = 4
"""The chunk function's leading operands: its decays (`_Decays`' four fields), the solve's inverse after."""


def _ranges(declaration: Scan, names: Sequence[str], chunk: Mapping, affine,
            chunk_size: int) -> dict[int, float]:
    """The chunk function's operands' ranges (`_chunk`'s, `carry.split`'s): its decays at most 1, each token
    value's (`_token`'s) as the declared operands bound it (`terms.Region`'s magnitudes), the solve's inverse
    by its entries' bound; a value an unbounded operand's has none."""
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
            bounded[DECAYS + 1 + index] = float(np.max(np.abs(np.asarray(literal, np.float64))))
            continue
        magnitude = region.magnitudes.get(terms.key(output))
        if magnitude is not None:
            # No bound where it passes float64's range (a step of 2**1023 declared, `prepare`'s values).
            bounded[DECAYS + 1 + index] = math.inf if magnitude >= 1024 else 2.0 ** magnitude
    # The solve's inverse less its diagonal: the inverse of a unit lower-triangular matrix whose entries
    # are at most `a` in magnitude is at most `(1 + a)^(C - 1)` (by induction on the substitution); its
    # entries `s_i C_i . B_j` times a decay at most 1, a contraction over the keys: at most the declared
    # overlap (`Scan.ranges`' "overlap"), else the corrections' elementwise bounds' product times the keys'
    # count (two tokens of 16 keys each element at most 1 overlap by 16).
    places = ("B", "U", "s", "C", "Q", "R")
    tokens_at = {name: DECAYS + 1 + place for place, name in enumerate(places)}
    overlap = declaration.ranges.get("overlap")
    entry = None if overlap is None else max(map(abs, overlap)) if isinstance(overlap, tuple) else overlap
    if entry is None and all(tokens_at[name] in bounded for name in ("B", "s", "C")):
        B = region.outputs[places.index("B")]
        keys = math.prod(np.shape(region.literal_outputs[0]) if B is None else region.shape(B))
        entry = bounded[tokens_at["s"]] * bounded[tokens_at["C"]] * bounded[tokens_at["B"]] * keys
    if entry is not None:
        # No bound where it passes float64's range (16 unit keys over 256 tokens: 17**255).
        power = max(chunk_size - 1, 0) * math.log2(1.0 + entry)
        bounded[DECAYS] = math.inf if power >= 1024 else 2.0 ** power
    return {**dict.fromkeys(range(DECAYS), 1.0), **bounded}


def _chunk(corrects: bool, h, entering, to_last, whole, between, lower, B, U, s, C, Q, R):
    """A chunk's outputs and the state it leaves, from the state `h` entering it, its decays (`_Decays`'
    fields, each at most 1), its solve's inverse less its diagonal (`_inverse_of`, zeros where no token
    corrects the state) and its tokens' values (`_token`): the module's forms, which `carry.split`
    splits."""
    decays = _Decays(entering=entering, to_last=to_last, whole=whole, between=between)
    w = _corrected(decays, lower, U, s, C, h) if corrects else U
    carried = float32_einsum("tk,tv->kv", B * decays.to_last, w)
    outputs = float32_matmul(Q * decays.entering, h) + float32_matmul(_paired(Q, B, decays.between), w) + R
    return outputs, decays.whole[:, None] * h + carried


def _producer(jaxpr: jcore.Jaxpr, var):
    for equation in jaxpr.eqns:
        if var in equation.outvars:
            return equation
    return None


def _scalar(atom) -> bool:
    return isinstance(atom, jcore.Literal) or aval(atom).shape == ()


def _vector_along(jaxpr: jcore.Jaxpr, atom, axis: int, rank: int):
    """The vector `atom` broadcasts along `axis` of a rank-`rank` value (through broadcasts and
    element-keeping equations), or None."""
    position, base = None, atom
    chain = []
    while isinstance(base, jcore.Var) and len(aval(base).shape) != 1:
        equation = _producer(jaxpr, base)
        if equation is None or equation.primitive.name not in _KEEPING | {"broadcast_in_dim"}:
            return None
        chain.append(equation)
        base = equation.invars[0]
    if not isinstance(base, jcore.Var) or len(aval(base).shape) != 1:
        return None
    position = 0
    for equation in reversed(chain):
        if equation.primitive.name == "broadcast_in_dim":
            position = equation.params["broadcast_dimensions"][position]
        else:
            before, after = aval(equation.invars[0]).shape, aval(equation.outvars[0]).shape
            long_before = [index for index, extent in enumerate(before) if extent != 1]
            long_after = [index for index, extent in enumerate(after) if extent != 1]
            if [before[index] for index in long_before] != [after[index] for index in long_after]:
                return None
            position = long_after[long_before.index(position)]
    return base if position == axis and len(aval(atom).shape) == rank else None


def _symbolic(jaxpr: jcore.Jaxpr, state) -> dict:
    """Each value of `jaxpr` that reads `state`, as a `_Symbolic` sum of terms; refused where one is not of
    the module's form."""
    env = {state: _Symbolic("matrix", (_Term("state"),))}
    for equation in jaxpr.eqns:
        read = [var for var in equation.invars if isinstance(var, jcore.Var) and var in env]
        if not read:
            continue
        (out,) = equation.outvars
        env[out] = _rule(jaxpr, equation, env)
    return env


def _rule(jaxpr: jcore.Jaxpr, equation: jcore.JaxprEqn, env: dict) -> _Symbolic:
    """One equation's output that reads the state (`_symbolic`)."""
    name, operands = equation.primitive.name, equation.invars
    symbolic = [env.get(var) if isinstance(var, jcore.Var) else None for var in operands]
    first = symbolic[0]
    if first is not None and name in _KEEPING:
        return first
    if first is not None and name == "neg":
        return _Symbolic(first.shape, tuple(term.scaled(sign=-1.0) for term in first.terms))
    if name in ("add", "sub"):
        return _summed(jaxpr, operands, symbolic, minus=name == "sub")
    if name == "mul":
        return _product(jaxpr, operands, symbolic)
    if first is not None and name == "broadcast_in_dim":
        dimensions = tuple(equation.params["broadcast_dimensions"])
        if first.shape == "row" and dimensions == (1,) and len(equation.params["shape"]) == 2:
            return _Symbolic("spread", first.terms)
        if first.shape in ("spread", "matrix") and dimensions == (0, 1):
            return first
        raise Refused("a value of the state broadcast other than a row along the state's first axis")
    if name == "dot_general":
        (left_axes, right_axes), batch = equation.params["dimension_numbers"]
        second = symbolic[1]
        if first is None and second is not None:
            return _contracted(operands[0], second, left_axes, right_axes, batch)
        if first is not None and second is None:
            return _contracted(operands[1], first, right_axes, left_axes, batch)
    raise Refused(f"the state read through {name}, which no chunk form here holds")


def _summed(jaxpr, operands, symbolic, *, minus: bool) -> _Symbolic:
    """A sum of two values, at least one of which reads the state; a free one is an input (a matrix's) or a
    free row (a row's)."""
    shape = next(value.shape for value in symbolic if value is not None)
    terms = []
    for index, (operand, value) in enumerate(zip(operands, symbolic, strict=True)):
        sign = -1.0 if minus and index == 1 else 1.0
        if value is None:
            value = _free(jaxpr, operand, shape)
        elif value.shape != shape:
            raise Refused("a sum of values of the state of different shapes")
        terms.extend(term.scaled(sign=sign) for term in value.terms)
    return _Symbolic(shape, tuple(terms))


def _free(jaxpr, operand, shape: str) -> _Symbolic:
    """A value free of the state, as the terms of a sum of `shape`: a row's vector, or a matrix's outer
    product of two vectors (scaled by scalars)."""
    if shape == "row":
        if _scalar(operand) or len(aval(operand).shape) != 1:
            raise Refused("a row of the state plus a value that is not a row")
        return _Symbolic("row", (_Term("free", right=operand),))
    factors, left, right = _outer_factors(jaxpr, operand)
    return _Symbolic("matrix", (_Term("input", tuple(factors), left=left, right=right),))


def _outer_factors(jaxpr, atom) -> tuple[list, Atom, Atom]:
    """A free matrix as scalars times the outer product of a first-axis and a second-axis vector."""
    factors, left, right = _factored(jaxpr, atom)
    if left is None or right is None:
        raise Refused("a free term of the update that is not an outer product of two vectors")
    return factors, left, right


def _factored(jaxpr, atom) -> tuple[list, Atom | None, Atom | None]:
    """A free rank-two value as a product of scalars, at most one first-axis vector and at most one
    second-axis vector, each broadcast; refused where it is not."""
    factors, left, right = [], None, None
    pending = [atom]
    while pending:
        value = pending.pop()
        producer = _producer(jaxpr, value) if isinstance(value, jcore.Var) else None
        if _scalar(value):
            factors.append(value)
        elif (found := _vector_along(jaxpr, value, 0, 2)) is not None and left is None:
            left = found
        elif (found := _vector_along(jaxpr, value, 1, 2)) is not None and right is None:
            right = found
        elif producer is not None and producer.primitive.name == "mul":
            pending.extend(producer.invars)
        elif producer is not None and _passes(producer):
            pending.append(producer.invars[0])
        else:
            raise Refused("a free term of the update that is not a product of scalars and two vectors")
    return factors, left, right


def _product(jaxpr, operands, symbolic) -> _Symbolic:
    """A product with a value that reads the state: by a scalar, or a spread row by a first-axis vector (a
    rank-one correction, or an input where the row is free)."""
    if all(value is not None for value in symbolic):
        raise Refused("a product of two values of the state (the state read twice)")
    index = 0 if symbolic[0] is not None else 1
    value, other = symbolic[index], operands[1 - index]
    if _scalar(other) or value.shape == "row":
        if not _scalar(other):
            scalar = _scalar_broadcast(jaxpr, other)
            if scalar is None:
                raise Refused("a row of the state scaled by a value that is not a scalar")
            other = scalar
        return _Symbolic(value.shape, tuple(term.scaled(other) for term in value.terms))
    factors, left, right = _factored(jaxpr, other)
    if right is not None:
        raise Refused("the state scaled by a value along its second axis (no chunk form here)")
    scaled = list(value.terms)
    for factor in factors:
        scaled = [term.scaled(factor) for term in scaled]
    if left is None:
        return _Symbolic(value.shape, tuple(scaled))
    if value.shape == "matrix":
        # The state's rows scaled by a first-axis vector: a per-key decay, once.
        if any(term.kind != "state" or term.diag is not None for term in scaled):
            raise Refused("a value of the state other than the state itself scaled per row")
        return _Symbolic("matrix", tuple(dataclasses.replace(term, diag=left) for term in scaled))
    kinds = {"read": "corrected", "free": "input"}
    return _Symbolic("matrix", tuple(_Term(kinds[term.kind], term.factors, term.sign, left=left,
                                           right=term.right, diag=term.diag) for term in scaled))


def _passes(producer: jcore.JaxprEqn) -> bool:
    """Whether a factor passes through `producer` unchanged in its role: an element-keeping equation, a
    scalar's broadcast, or a broadcast that keeps a rank-two value's axes."""
    name = producer.primitive.name
    return name in _KEEPING or (name == "broadcast_in_dim" and (
        _scalar(producer.invars[0]) or tuple(producer.params["broadcast_dimensions"]) == (0, 1)))


def _scalar_broadcast(jaxpr, atom):
    producer = _producer(jaxpr, atom) if isinstance(atom, jcore.Var) else None
    while producer is not None and producer.primitive.name in _KEEPING | {"broadcast_in_dim"}:
        atom = producer.invars[0]
        if _scalar(atom):
            return atom
        producer = _producer(jaxpr, atom) if isinstance(atom, jcore.Var) else None
    return None


def _contracted(vector, value: _Symbolic, vector_axes, state_axes, batch) -> _Symbolic:
    """`vector^T value`: a row read from the state where `value` is the state scaled, contracting its first
    axis with a vector."""
    del vector_axes
    if batch[0] or len(aval(vector).shape) != 1 or tuple(state_axes) != (0,) or value.shape != "matrix":
        raise Refused("the state contracted other than along its first axis with a vector")
    if any(term.kind != "state" for term in value.terms):
        raise Refused("a row read from a corrected state (a correction of a correction)")
    return _Symbolic("row", tuple(_Term("read", term.factors, term.sign, right=vector, diag=term.diag)
                                  for term in value.terms))


def _affine(update: Program, readout: Program) -> _Affine:
    """The token's parts (`_Affine`), refused where the update or the readout is not of the module's form."""
    (out,) = update.jaxpr.outvars
    symbolic = _symbolic(update.jaxpr, update.jaxpr.invars[0]).get(out)
    if symbolic is None or symbolic.shape != "matrix":
        raise Refused("the update does not read the state")
    held = [term for term in symbolic.terms if term.kind == "state"]
    inputs = [term for term in symbolic.terms if term.kind == "input"]
    corrections = tuple(term for term in symbolic.terms if term.kind == "corrected")
    if len(held) != 1 or len(inputs) != 1:
        raise Refused("the update is not one decayed state plus one input")
    (state,) = held
    decay, exponent, base2 = _decay(update.jaxpr, state)
    (given,) = inputs
    for correction in corrections:
        decayed = correction.diag is decay if state.diag is not None else (
            correction.diag is None and decay in correction.factors)
        if correction.left is not given.left or not decayed:
            raise Refused("a correction along a vector other than the input's, or not decayed with the state")
    (read_out,) = readout.jaxpr.outvars
    read = _symbolic(readout.jaxpr, readout.jaxpr.invars[0]).get(read_out)
    if read is None or read.shape != "row":
        raise Refused("the readout does not read the state as a row")
    reads = [term for term in read.terms if term.kind == "read"]
    if len(reads) != 1:
        raise Refused("the readout reads the state other than once")
    free = tuple(term for term in read.terms if term.kind == "free")
    return _Affine(update, readout, exponent, base2, state.diag is not None, decay, given, corrections,
                   reads[0], free)


def _decay(jaxpr, state: _Term) -> tuple[Atom, Atom, bool]:
    """The state term's decay `exp(z)` (or `exp2`): the factor (a scalar, or the rows' per-key vector) and
    its exponent."""
    if state.diag is not None:
        producer = _producer(jaxpr, state.diag) if isinstance(state.diag, jcore.Var) else None
        if state.sign != 1.0 or state.factors or producer is None or producer.primitive.name not in (
                "exp", "exp2"):
            raise Refused("the state's per-key decay is not one exponential, whose exponent the prefix sums")
        return state.diag, producer.invars[0], producer.primitive.name == "exp2"
    exponentials = []
    for factor in state.factors:
        producer = _producer(jaxpr, factor) if isinstance(factor, jcore.Var) else None
        if producer is not None and producer.primitive.name in ("exp", "exp2"):
            exponentials.append((factor, producer))
    if state.sign != 1.0 or len(state.factors) != 1 or len(exponentials) != 1:
        raise Refused("the state's decay is not one exponential, whose exponent the prefix sums")
    ((factor, producer),) = exponentials
    return factor, producer.invars[0], producer.primitive.name == "exp2"


def _token(affine: _Affine, token: tuple) -> tuple:
    """One token's exponent, input vector `B` and values `U`, correction strength `s` and vector `C`, reader
    `Q` and free readout `R` (zeros where the readout has none)."""
    update, readout = affine.update, affine.readout
    zero = jnp.zeros(aval(update.jaxpr.invars[0]).shape, aval(update.jaxpr.invars[0]).dtype)

    def values(closed, atoms):
        env = environment(closed, (zero, *token), {})
        evaluate(closed, env, read_by(closed.jaxpr, {atom for atom in atoms if isinstance(atom, jcore.Var)}))
        return lambda atom: atom.val if isinstance(atom, jcore.Literal) else env[atom]

    def coefficient(term: _Term, read, leaving=None):
        product = jnp.float32(term.sign)
        for factor in term.factors:
            if factor is not leaving:
                product = product * read(factor)
        return product

    terms = (affine.input, *affine.corrections)
    read = values(update, [affine.exponent, *(atom for term in terms for atom in
                                              (*term.factors, term.left, term.right))])
    B = read(affine.input.left)
    U = coefficient(affine.input, read) * read(affine.input.right)
    # The corrections, each decay * (-strength) * B (c^T h), are one: B ((sum of strength c)^T h), of unit
    # strength where any corrects.
    s = jnp.float32(1.0 if affine.corrections else 0.0)
    C = jnp.zeros_like(B)
    for correction in affine.corrections:
        C = C - coefficient(correction, read, leaving=affine.decay) * read(correction.right)
    weights = () if affine.read.diag is None else (affine.read.diag,)
    out = values(readout, [*affine.read.factors, affine.read.right, *weights,
                           *(atom for term in affine.free for atom in (*term.factors, term.right))])
    # A readout that weighs the state's rows first (`_Term.diag`) reads them through its vector so weighed.
    Q = coefficient(affine.read, out) * out(affine.read.right)
    for weight in weights:
        Q = Q * out(weight)
    R = sum((coefficient(term, out) * out(term.right) for term in affine.free),
            jnp.zeros((aval(update.jaxpr.invars[0]).shape[1],), jnp.float32))
    keys = aval(update.jaxpr.invars[0]).shape[0] if affine.per_key else 1
    return jnp.broadcast_to(read(affine.exponent), (keys,)), B, U, s, C, Q, R


def _inverse_of(decays: _Decays, B, s, C):
    """The unit lower-triangular solve's inverse less its diagonal (`_lowered`) of the corrections'
    decayed overlaps: it reads no state, so a chunk's walk prepares it with its other values."""
    strictly = jnp.tril(jnp.ones(decays.between.shape[:2], bool), -1)[:, :, None]
    return _lowered(s[:, None] * _paired(C, B, jnp.where(strictly, decays.between, 0.0)))


def _corrected(decays: _Decays, lower, U, s, C, h):
    """The inputs each token's correction leaves (the module's `w`): the solve (its inverse less its
    diagonal `lower`, `_inverse_of`) of the corrections, from the state entering the chunk."""
    given = U - float32_matmul(s[:, None] * C * decays.entering, h)
    return _applied(lower, given)


def _paired(left, right, between):
    """Each pair's decayed contraction over keys, `sum_k left_ik right_jk between_ijk` `[C, C]`: a decay a
    key dimension's over the pairs' every key; a scalar decay's (`between` `[C, C, 1]`) the keys'
    product decayed after it, `(left right^T) between`, which spans no `[C, C, K]` tensor (Qwen3.5-27B's
    derived forward took 66 ms on an H100 spanning it, derived-cells-h100-6d105a908). A pair no decay joins
    (`between` zero, a token's future) is zero, selected, not multiplied: its product, or its cotangent, may
    be infinite near float32's largest (a future corrected write's), whose product by a zero decay is NaN in
    every query's gradient the pair reads (the chunked op's reverse mode, near the largest) and in the
    decays'."""
    active = jnp.any(between != 0, axis=-1)
    if between.shape[-1] == 1:
        # The product selected before its decay multiplies it: an inactive pair's may overflow, whose zero
        # decay's derivative would read it.
        return jnp.where(active, float32_matmul(left, right.T), 0.0) * between[..., 0]
    return jnp.where(active, float32_einsum("ik,ijk->ij", left, right[None] * between), 0.0)


def _lowered(overlaps):
    """`(I + overlaps)^-1` less its unit diagonal, `overlaps` strictly lower: the template's inverse by
    doubling (`recurrence_algebra.doubled` of `-overlaps`), each product a Dew product (`float32_matmul`),
    log2(C) levels of products where a forward substitution took C dependent row steps (the derived forward's
    every chunk 64 launches in sequence). The solve applies it with the unit diagonal outside the product
    (`_applied`, `given + L given`, as `weighted` adds it), so the product rounds the off-diagonal terms
    alone. Differentiated at every order (`_lowered_rule`): its reverse derivative, with `X = I + L`,
    `-X^T dL X^T` below the diagonal, its tangent the doubling's own."""
    return _lowered_rule()(overlaps)


def _doubled(overlaps):
    # The template's algebra imports the compile point, which imports this package: read at a call.
    from dew.nn.kernels import recurrence_algebra

    return _below(recurrence_algebra.doubled(-overlaps, _issued))


def _issued(name: str, left, right):
    """The doubling's products (`recurrence_algebra.doubled`'s issuer) as Dew's (`float32_matmul`)."""
    del name
    return float32_matmul(left, right)


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
        back = _applied(lower.T, d_lower)
        return (_below(-_applied(lower, back.T).T),)

    return generation.linear_tangent(forward, backward, _doubled)


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


def _vanishing(dtype, base2: bool) -> float:
    """The exponent at and below which a decay is zero in `dtype`, whatever exponents join it (each at most
    zero): below the base's logarithm of its least subnormal (`exp(-105)` and `exp2(-151)` are zero in
    fp32)."""
    least = math.log(float(jnp.finfo(dtype).smallest_subnormal))
    return math.floor(least / (math.log(2.0) if base2 else 1.0)) - 1


def _walked(exponents, base2: bool) -> jax.Array:
    """A chunk's whole decay in natural units, per key, as a walk's state carries it (`Derived.step`,
    `merge`): its exponents' compensated sum, or its dtype's vanished decay where a token's decay vanishes
    (`_vanishing`; two of -2**127 summed are -inf, and its residual NaN)."""
    vanished = exponents <= _vanishing(exponents.dtype, base2)
    live = _scanned(jnp.where(vanished, 0.0, exponents))[-1] * (math.log(2.0) if base2 else 1.0)
    return _held(jnp.where(jnp.any(vanished, axis=0), _vanishing(exponents.dtype, base2=False), live))


def _held(decay: jax.Array) -> jax.Array:
    """A walk's decay in natural units held from below at its dtype's vanished one (`_vanishing`): zero
    there, and a walk's sums of vanished chunks finite."""
    return jnp.maximum(decay, jnp.asarray(_vanishing(decay.dtype, base2=False), decay.dtype))


def _decays_value(exponents, base2: bool) -> _Decays:
    """The decays (`_Decays`) of a chunk's tokens' exponents, each from the compensated prefix of its
    segment: a token whose exponent vanishes (`_vanishing`, its decay and every decay through it zero) ends
    one and starts the next, taken as 0 within it, so no span's sum holds a vanished token's (a prefix
    holding -2**127 twice is -inf; one holding -2**70 loses a later -1 to its rounding, its decay 1 in place
    of `exp(-1)`), and each span through one is zero."""
    vanished = exponents <= _vanishing(exponents.dtype, base2)
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

    return generation.linear_tangent(forward, backward, functools.partial(_decays_value, base2=base2))


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
