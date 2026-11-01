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
from collections.abc import Callable, Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core as jcore

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
    reads, marking each token a zero state enters (a packed document's first)."""

    update: Callable
    readout: Callable
    operands: Mapping[str, Axes]
    walked: str
    state: tuple[tuple[int, ...], jnp.dtype]
    reset: str | None = None


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


def derive_scan(declaration: Scan) -> Derived:
    """`declaration`'s chunk forms (the module's account): `init(shapes)` a zero state and no decay,
    `step(state, chunk)` the state after a chunk of tokens, `mapped(chunk, state)` the chunk's outputs from
    the state entering it, `merge(a, b)` two consecutive walks' states joined (the later walk's from zero),
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

    def init(shapes):
        del shapes
        return jnp.zeros((shape[0],), jnp.float32), jnp.zeros(shape, dtype)

    def step(state, chunk):
        affine, (exponent, B, U, s, C, _, _) = parts(chunk)
        decay, h = state
        decays = _decays(exponent, affine.base2)
        w = _corrected(decays, B, U, s, C, h)
        carried = float32_einsum("tk,tv->kv", B * decays.to_last, w)
        natural = _scanned(exponent)[-1] * (math.log(2.0) if affine.base2 else 1.0)
        return decay + natural, decays.whole[:, None] * h + carried

    def mapped(chunk, state):
        affine, (exponent, B, U, s, C, Q, R) = parts(chunk)
        _, h = state
        decays = _decays(exponent, affine.base2)
        w = _corrected(decays, B, U, s, C, h)
        scores = _paired(Q, B, decays.between)
        return float32_matmul(Q * decays.entering, h) + float32_matmul(scores, w) + R

    def merge(a, b):
        # A state's decay is its walk's exponent sum per key in natural units. A later walk from zero leaves
        # the state before it decayed alone only where no token corrects it: refused where one does.
        if corrects != [False]:
            raise Refused("a merge of walks whose tokens correct the state, or before any chunk was read")
        (decay_a, h_a), (decay_b, h_b) = a, b
        return decay_a + decay_b, jnp.exp(decay_b)[:, None] * h_a + h_b

    def finish(state, operands):
        del operands
        return state[1]

    def backward(chunk, state, d_state, d_outputs):
        # The chunk's forms' adjoint: the reverse scan's step, each chunk's transition transposed, as the
        # forward forms' derivative (their prefix, substitution and Dew products differentiated).
        # The reset's booleans held, no operand of the derivative.
        def forward(h, operands):
            entering, tiles = (state[0], h), {**chunk, **operands}
            return mapped(tiles, entering), step(entering, tiles)[1]

        _, pullback = jax.vjp(forward, state[1], {name: chunk[name] for name in names})
        d_h, d_chunk = pullback((d_outputs, d_state))
        return d_h, d_chunk

    return Derived(init, step, merge, finish, mapped,
                   {"rounding": "compensated prefix", "walked": declaration.walked}, backward)


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
    out = values(readout, [*affine.read.factors, affine.read.right,
                           *(atom for term in affine.free for atom in (*term.factors, term.right))])
    Q = coefficient(affine.read, out) * out(affine.read.right)
    R = sum((coefficient(term, out) * out(term.right) for term in affine.free),
            jnp.zeros((aval(update.jaxpr.invars[0]).shape[1],), jnp.float32))
    keys = aval(update.jaxpr.invars[0]).shape[0] if affine.per_key else 1
    return jnp.broadcast_to(read(affine.exponent), (keys,)), B, U, s, C, Q, R


def _corrected(decays: _Decays, B, U, s, C, h):
    """The inputs each token's correction leaves (the module's `w`): the unit lower-triangular solve of the
    corrections' decayed overlaps, from the state entering the chunk."""
    strictly = jnp.tril(jnp.ones(decays.between.shape[:2], bool), -1)[:, :, None]
    overlaps = s[:, None] * _paired(C, B, jnp.where(strictly, decays.between, 0.0))
    given = U - s[:, None] * float32_matmul(C * decays.entering, h)
    return _solved(overlaps, given)


def _paired(left, right, between):
    """Each pair's decayed contraction over keys, `sum_k left_ik right_jk between_ijk` `[C, C]`: a decay a
    key dimension's over the pairs' every key; a scalar decay's (`between` `[C, C, 1]`) the keys'
    product decayed after it, `(left right^T) between`, which spans no `[C, C, K]` tensor (Qwen3.5-27B's
    derived forward took 66 ms on an H100 spanning it, derived-cells-h100-6d105a908)."""
    if between.shape[-1] == 1:
        return float32_matmul(left, right.T) * between[..., 0]
    return float32_einsum("ik,ijk->ij", left, right[None] * between)


@jax.custom_vjp
def _solved(overlaps, given):
    """`w` with `(I + overlaps) w = given`, `overlaps` strictly lower: the inverse by doubling (`_inverse`),
    then one product, each a Dew product (`float32_matmul`): the template's own form (recurrence_algebra's
    `doubled` and `weighted`), log2(C) levels of products where a forward substitution took C dependent row
    steps (the derived forward's every chunk 64 launches in sequence). The unit diagonal is added outside
    the product (`given + L given`, `L` the inverse less it), as `weighted` adds it, so the product rounds
    the off-diagonal terms alone. Its derivative `back = X^T dw` alike and the overlaps' cotangent
    `-back w^T` below the diagonal."""
    return _applied(_below(_inverse(overlaps)), given)


def _applied(lower, given):
    """`(I + lower) @ given`, the identity outside the product."""
    return given + float32_matmul(lower, given)


def _below(square):
    """`square`'s strictly lower part."""
    return jnp.where(jnp.tril(jnp.ones(square.shape, bool), -1), square, 0.0)


def _solved_forward(overlaps, given):
    lower = _below(_inverse(overlaps))
    solved = _applied(lower, given)
    return solved, (lower, solved)


def _solved_backward(residuals, d_solved):
    lower, solved = residuals
    back = _applied(lower.T, d_solved)
    return _below(-float32_matmul(back, solved.T)), back


_solved.defvjp(_solved_forward, _solved_backward)


def _inverse(overlaps):
    """`(I + overlaps)^-1` of a strictly lower `overlaps` `[C, C]` by doubling: from the two-wide blocks, each
    level joins pairs of `w`-blocks as `X_2w = X_w + (M_w * (X_w A)) X_w`, `A = -overlaps` and `M_w` each
    pair's lower block (recurrence_algebra's `doubled`), until one block spans the chunk: a chunk of any
    length, its last blocks shorter."""
    a = -overlaps
    size = a.shape[0]
    rows, cols = jnp.arange(size)[:, None], jnp.arange(size)[None, :]
    inverse = jnp.where(rows == cols, 1.0, 0.0) + jnp.where((rows % 2 == 1) & (cols == rows - 1), a, 0.0)
    width = 2
    while width < size:
        lower = ((rows // width) % 2 == 1) & (cols // width == rows // width - 1)
        inverse = inverse + float32_matmul(jnp.where(lower, float32_matmul(inverse, a), 0.0), inverse)
        width *= 2
    return inverse


def _compensated(values, *, reverse: bool = False) -> tuple[jax.Array, jax.Array]:
    """`values`' inclusive prefix sums along their first axis (suffix sums where `reverse`), each a float and
    its rounding error (a compensated scan)."""
    def combine(left, right):
        total, error = two_sum(left[0], right[0])
        return total, error + left[1] + right[1]

    return lax.associative_scan(combine, (values, jnp.zeros_like(values)), reverse=reverse)


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
    compensated prefix differenced from both parts: every exponent taken is at most zero, so each stays
    finite at any decay, at `tokens^2 keys` elementwise work a chunk (a sub-chunk factoring around a
    reference prefix is the faster equal form, the extraction's choice under a kernel cost)."""
    return _decays_of(exponents, base2)


def _decays_value(exponents, base2: bool) -> _Decays:
    high, low = _compensated(exponents)
    count = exponents.shape[0]
    below = (jnp.arange(count)[:, None] >= jnp.arange(count)[None, :])[:, :, None]
    gaps = (high[:, None] - high[None, :]) + (low[:, None] - low[None, :])
    return _Decays(entering=_exp(base2, high + low),
                   to_last=_exp(base2, (high[-1] - high) + (low[-1] - low)),
                   whole=_exp(base2, high[-1] + low[-1]),
                   between=jnp.where(below, _exp(base2, jnp.where(below, gaps, 0.0)), 0.0))


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _decays_of(exponents, base2: bool) -> _Decays:
    return _decays_value(exponents, base2)


def _decays_forward(exponents, base2: bool):
    decays = _decays_value(exponents, base2)
    return decays, decays


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


_decays_of.defvjp(_decays_forward, _decays_backward)


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
