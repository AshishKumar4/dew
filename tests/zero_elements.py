"""Check 2 at the elements whose float64 truth is exactly zero (a prototype the suite does not judge by yet):
each such element of Dew's native output held to the rigorous bound of Dew's own computation's rounding
there, the specification approved to code on 2026-10-21 (the gate's zero-element-certificate-plan).

Dew's function is traced on the draw's native inputs and its program evaluated primitive by primitive
(`Interpreter`), each value carrying its native value, a bound on its distance from the exact result of
the program on those inputs, whether it is that exact result, and whether its bound is certified at all.
A primitive is certified only by a verified contract of op_certificates.py:

- add, sub, mul, div and a cast are IEEE operations, correctly rounded, their native value checked
  against the exact one element by element; one is exact where a residual proves it (TwoSum, the exact
  float64 product of fp32 values, b·q = a), its rounding u|v̂| otherwise, and the least normal where
  the backend may flush;
- a sum (reduce_sum, dot_general) of m nonzero native terms rounds by at most γ(m - 1) of their
  magnitudes in any order, a dot's products by γ(m), on the CPU at float32 or float64 (XLA:CPU's dot
  accumulates in the operands' format, at any precision);
- max, min, moves and selections by a decided predicate carry bounds exactly, and exp carries them by
  its derivative, its native value held to a verified enclosure of the exact one at every element;
- anything else is NOT_CERTIFIED, named, and so is every value it reaches.

Bounds are rounded outward: every bound expression of a few terms widened by 2^-50 and rounded up, a
product of nonzero bounds never below the least float64, a float64 sum of n terms widened by γ64(n).
An element is certified only where its interpretation's native value equals Dew's native output bit for
bit: the bound is the interpretation's, and an execution that differs is not described by it.

The truth's own distance from its program's exact result (τ) is the same propagation over the
reference's program at float64. An element of truth exactly zero is held to `|dew - truth| <= e + τ`;
a certified round-off budget, not a proof that Dew computes the reference's function: a defect smaller
than Dew's own rounding there passes.
"""

import dataclasses
import fractions
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jcore
from op_certificates import (
    EXP_BOUND,
    FLOAT32,
    Format,
    _exp_enclosed,
    _two_product,
    fraction_rounded,
    gamma,
    interval_ends,
    interval_precision,
    two_sum,
)

FLOAT64 = Format("float64", 53, -1022, 1023)
FORMATS = {np.dtype(np.float32): FLOAT32, np.dtype(np.float64): FLOAT64}
TINIEST = 5e-324
"""The least positive float64: a product of nonzero bounds is never taken below it."""
DEKKER_FLOOR = 2.0 ** -968
"""The least float64 product whose Dekker residual cannot underflow (2^(emin + p + 1))."""


def _least_normal(fmt: Format) -> float:
    return 2.0 ** fmt.emin


def _up(value: np.ndarray) -> np.ndarray:
    """A bound expression of a few float64 terms rounded outward: widened by 2^-50 past every rounding it
    took, then rounded up."""
    value = np.asarray(value, np.float64)
    return np.where(value > 0, np.nextafter(value * (1 + 2.0 ** -50), np.inf), value)


def _times(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """An upper bound of a·b for nonnegative bounds, never below the least float64 where both are nonzero."""
    product = _up(np.asarray(a, np.float64) * np.asarray(b, np.float64))
    return np.where((a > 0) & (b > 0), np.maximum(product, TINIEST), product)


def _widened_sum(total: np.ndarray, terms: int, positive) -> np.ndarray:
    """A float64 sum of `terms` nonnegative bounds (products of them included), taken in any order,
    widened to an upper bound of the exact sum: by γ64(terms) relatively, and by two least normal float64
    values for each of its `positive` terms, which a product or a partial sum may lose in underflow, a
    flushing backend's included (the bound dots run in XLA), so no positive bound sums to zero."""
    widen = 1 + gamma(max(terms, 1), FLOAT64)
    least = _least_normal(FLOAT64)
    with np.errstate(over="ignore"):
        return _up(np.asarray(total, np.float64) * widen + 2 * np.asarray(positive, np.float64) * least)


@dataclasses.dataclass
class Bounded:
    """A value of the interpreted program: its native value, a bound on its distance from the exact
    result of the program on its native inputs, whether it is that result, and whether the bound holds
    at all (False where a primitive without a contract reaches it)."""
    value: np.ndarray
    error: np.ndarray
    exact: np.ndarray
    certified: np.ndarray

    @classmethod
    def exactly(cls, value) -> "Bounded":
        value = np.asarray(value)
        shape = value.shape
        return cls(value, np.zeros(shape), np.ones(shape, bool), np.ones(shape, bool))


def _input(value) -> Bounded:
    """A native input, constant or literal: the computation's own, exact by definition, but for a
    subnormal float, which a backend may read as zero, held to its own size; a value that is not a
    number, or of a type without contracts (complex, or a float format other than float32's and
    float64's), is not certified."""
    value = np.asarray(value)
    bounded = Bounded.exactly(value)
    if np.issubdtype(value.dtype, np.complexfloating):
        bounded.exact = bounded.certified = np.zeros(value.shape, bool)
        return bounded
    fmt = FORMATS.get(value.dtype)
    if fmt is not None:
        wide = value.astype(np.float64)
        tiny = (wide != 0) & (np.abs(wide) < _least_normal(fmt))
        bounded.error = np.where(tiny, np.abs(wide), 0.0)
        bounded.certified = ~np.isnan(wide)
        bounded.exact = ~tiny & bounded.certified
    elif np.issubdtype(value.dtype, np.inexact):
        bounded.exact = bounded.certified = np.zeros(value.shape, bool)
    return bounded


MOVES = {"broadcast_in_dim", "reshape", "transpose", "squeeze", "expand_dims", "slice", "rev", "copy",
         "copy_p", "concatenate", "stack", "split", "pad", "dynamic_slice", "gather"}
"""Primitives that only move or select elements, by operands the moved values do not decide: each
element's bound, exactness and certification move with it."""
CALLS = {"jit": "jaxpr", "pjit": "jaxpr", "closed_call": "call_jaxpr", "core_call": "call_jaxpr",
         "custom_jvp_call": "call_jaxpr", "custom_vjp_call": "call_jaxpr",
         "custom_vjp_call_jaxpr": "fun_jaxpr", "checkpoint": "jaxpr", "remat": "jaxpr", "remat2": "jaxpr"}
"""Primitives that call a program of their own: interpreted through it."""
COMPARISONS = {"lt", "le", "gt", "ge", "eq", "ne"}


class Interpreter:
    """Dew's program evaluated primitive by primitive at its native inputs (`run`), each value `Bounded`;
    `blocked` counts, by primitive, the values a primitive without a contract left uncertified."""

    def __init__(self) -> None:
        self.blocked: dict[str, int] = {}

    def run(self, closed, *inputs) -> list[Bounded]:
        return self._program(closed.jaxpr, [_input(c) for c in closed.consts], [_input(x) for x in inputs])

    def _program(self, jaxpr, consts: list[Bounded], arguments: list[Bounded]) -> list[Bounded]:
        env: dict = dict(zip(jaxpr.constvars, consts, strict=True))
        env.update(zip(jaxpr.invars, arguments, strict=True))

        def read(atom):
            if isinstance(atom, jcore.Literal):
                return _input(np.asarray(atom.val, dtype=atom.aval.dtype))
            return env[atom]

        for eqn in jaxpr.eqns:
            outs = self._equation(eqn, [read(atom) for atom in eqn.invars])
            env.update(zip(eqn.outvars, outs, strict=True))
        return [read(atom) for atom in jaxpr.outvars]

    def _native(self, eqn, operands: list[Bounded]) -> list[np.ndarray]:
        outs = eqn.primitive.bind(*[jnp.asarray(operand.value) for operand in operands], **eqn.params)
        return [np.asarray(out) for out in (outs if eqn.primitive.multiple_results else [outs])]

    def _blocked(self, name: str, values: list[np.ndarray]) -> list[Bounded]:
        self.blocked[name] = self.blocked.get(name, 0) + sum(int(value.size) for value in values)
        return [Bounded(value, np.zeros(value.shape), *(np.zeros(value.shape, bool) for _ in range(2)))
                for value in values]

    def _equation(self, eqn, operands: list[Bounded]) -> list[Bounded]:
        name = eqn.primitive.name
        if eqn.effects:
            return self._blocked(f"{name} (effects)", self._native(eqn, operands))
        if name in {"scan", "while", "cond"}:
            return getattr(self, f"_{name}_loop" if name != "cond" else "_branch")(eqn, operands)
        if name == "linear_call":
            # The callee on the call's whole arguments (its constants, residuals and linear arguments, in
            # that order); the transpose is not run forward.
            callee = eqn.params["callee"]
            return self._program(callee.jaxpr, [_input(c) for c in callee.consts], operands)
        if name in CALLS:
            inner = eqn.params[CALLS[name]]
            closed = inner if hasattr(inner, "consts") else None
            jaxpr = closed.jaxpr if closed is not None else inner
            consts = [_input(c) for c in closed.consts] if closed is not None else []
            return self._program(jaxpr, consts, operands[:len(jaxpr.invars)])
        natives = self._native(eqn, operands)
        if any(np.issubdtype(array.dtype, np.complexfloating)
               for array in [*natives, *(operand.value for operand in operands)]):
            return self._blocked(f"{name} (complex)", natives)
        handler = getattr(self, f"_{name}", None)
        if name not in MOVES | {"stop_gradient", "copy"} and handler is None:
            return self._blocked(name, natives)
        # Bound arithmetic that overflows or has no value leaves a bound that is not finite, which the
        # check below refuses; numpy's warnings of it say nothing more.
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            if name in {"stop_gradient", "copy"}:
                outs = [operands[0]]
            elif name in MOVES:
                outs = self._moved(eqn, operands, natives)
            else:
                outs = handler(eqn, operands, natives)
        for out in outs:
            # A bound past float64's range certifies nothing.
            out.certified = out.certified & np.isfinite(out.error)
            out.exact = out.exact & out.certified
            out.error = np.where(out.certified, out.error, 0.0)
        return outs

    # Moves and selections.

    # Control flow: interpreted step by step at the native values, every predicate and index decided.

    def _uncertified(self, name: str, eqn, operands) -> list[Bounded]:
        return self._blocked(name, self._native(eqn, operands))

    @staticmethod
    def _decided(value: Bounded) -> bool:
        """A predicate or an index certified and exact throughout: decided, the bounds of what it
        compares parted (`_compare`), or integers exact."""
        return bool(value.certified.all() and value.exact.all())

    def _scan_loop(self, eqn, operands):
        """`scan`, its body interpreted once a step in its order (`reverse`), the carry bounded through
        it; its stacked outputs in step order. Its consts, carry and stacked inputs told apart by their
        shapes, a stacked one's the body's with the scan's length before it."""
        body, length, reverse = eqn.params["jaxpr"], eqn.params["length"], eqn.params["reverse"]
        inner = body.jaxpr
        outer_out, inner_out = eqn.outvars, inner.outvars
        carried = 0
        while carried < len(outer_out) and outer_out[carried].aval.shape == inner_out[carried].aval.shape:
            carried += 1
        stacked = 0
        while stacked < len(eqn.invars) and (
                eqn.invars[-1 - stacked].aval.shape == (length, *inner.invars[-1 - stacked].aval.shape)):
            stacked += 1
        held = len(eqn.invars) - carried - stacked
        consts, carry, xs = operands[:held], operands[held:held + carried], operands[held + carried:]
        steps: list[list[Bounded] | None] = [None] * length
        for step in (range(length - 1, -1, -1) if reverse else range(length)):
            sliced = [Bounded(x.value[step], x.error[step], x.exact[step], x.certified[step]) for x in xs]
            outs = self._program(inner, [_input(c) for c in body.consts], [*consts, *carry, *sliced])
            carry, steps[step] = outs[:carried], outs[carried:]
        stacked_outs = []
        for index, var in enumerate(eqn.outvars[carried:]):
            if length == 0:
                empty = np.zeros(var.aval.shape, var.aval.dtype)
                stacked_outs.append(Bounded(empty, np.zeros(empty.shape), np.ones(empty.shape, bool),
                                            np.ones(empty.shape, bool)))
                continue
            parts = [steps[step][index] for step in range(length)]

            def stacked_field(field, parts=parts):
                return np.stack([np.broadcast_to(getattr(part, field), part.value.shape) for part in parts])

            stacked_outs.append(Bounded(np.stack([part.value for part in parts]), stacked_field("error"),
                                        stacked_field("exact"), stacked_field("certified")))
        return [*carry, *stacked_outs]

    WHILE_STEPS = 1 << 20
    """The most steps a `while` is followed before its outputs are taken as not certified."""

    def _while_loop(self, eqn, operands):
        """`while`, its predicate interpreted at every step with its operands' bounds, the last one that
        stops it included, each required decided (`_decided`); else every output not certified."""
        cond, body = eqn.params["cond_jaxpr"], eqn.params["body_jaxpr"]
        held_cond, held_body = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
        cond_consts = operands[:held_cond]
        body_consts = operands[held_cond:held_cond + held_body]
        carry = operands[held_cond + held_body:]
        for _ in range(self.WHILE_STEPS):
            [going] = self._program(cond.jaxpr, [_input(c) for c in cond.consts], [*cond_consts, *carry])
            if not self._decided(going):
                return self._uncertified("while (its predicate undecided)", eqn, operands)
            if not bool(going.value):
                return carry
            carry = self._program(body.jaxpr, [_input(c) for c in body.consts], [*body_consts, *carry])
        return self._uncertified("while (past its step limit)", eqn, operands)

    def _branch(self, eqn, operands):
        """`cond` or `switch`, the branch its decided index selects (clamped, as XLA does); an index not
        decided leaves every output not certified."""
        index, *rest = operands
        if not self._decided(index):
            return self._uncertified("cond (its index undecided)", eqn, operands)
        branches = eqn.params["branches"]
        chosen = branches[int(np.clip(int(index.value), 0, len(branches) - 1))]
        return self._program(chosen.jaxpr, [_input(c) for c in chosen.consts], rest)

    def _moved(self, eqn, operands: list[Bounded], natives: list[np.ndarray]) -> list[Bounded]:
        """A move applied to each operand's bound, exactness and certification, its index operands (an
        integer exact and certified) moved as they are."""
        def moved(field):
            with jax.enable_x64(new_val=True):
                arrays = [jnp.asarray(getattr(operand, field), jnp.float64)
                          if np.issubdtype(operand.value.dtype, np.inexact) else jnp.asarray(operand.value)
                          for operand in operands]
                outs = eqn.primitive.bind(*arrays, **eqn.params)
                return [np.asarray(out) for out in (outs if eqn.primitive.multiple_results else [outs])]

        indices_certified = all(operand.certified.all() and operand.exact.all() for operand in operands
                                if not np.issubdtype(operand.value.dtype, np.inexact))
        if not any(np.issubdtype(native.dtype, np.inexact) for native in natives):
            certified = all(operand.certified.all() for operand in operands)
            return [Bounded(native, np.zeros(native.shape), np.full(native.shape, certified),
                            np.full(native.shape, certified)) for native in natives]
        results = []
        fields = zip(natives, moved("error"), moved("exact"), moved("certified"), strict=True)
        for native, error, exact, certified in fields:
            certified = (certified > 0.5) & indices_certified
            results.append(Bounded(native, error, (exact > 0.5) & certified, certified))
        return results

    def _select_n(self, eqn, operands, natives):
        which, *cases = operands
        native = natives[0]
        index = which.value.astype(np.int64)
        decided = which.certified & which.exact
        def pick(field):
            return np.choose(index, [np.broadcast_to(getattr(case, field), native.shape) for case in cases])

        certified = decided & pick("certified")
        error = np.where(certified, pick("error"), 0.0)
        return [Bounded(native, error, pick("exact") & certified, certified)]

    # Exact or IEEE elementwise operations.

    def _formatted(self, name, natives, *operands):
        fmt = FORMATS.get(natives[0].dtype)
        if fmt is None or any(FORMATS.get(operand.value.dtype) is not fmt for operand in operands):
            return None, self._blocked(f"{name} ({natives[0].dtype})", natives)
        return fmt, None

    def _rounded(self, fmt: Format, native: np.ndarray, hi: np.ndarray, lo: np.ndarray, transfer: np.ndarray,
                 operands, *, exact_from=None) -> Bounded:
        """An IEEE operation's value: exact where its exact result `hi + lo` is its native value and every
        operand exact; its rounding otherwise, checked against the native value; not certified where the
        native value is not within that rounding of the exact result, or not finite."""
        value = native.astype(np.float64)
        representable = (lo == 0) & (hi == value)
        if exact_from is not None:
            representable = exact_from
        u = fmt.unit_roundoff / (1 - fmt.unit_roundoff)
        with np.errstate(invalid="ignore", over="ignore"):
            rounding = _up(u * np.maximum(np.abs(value), np.abs(hi)) + _least_normal(fmt))
            apart = _up(np.abs(value - hi) + np.abs(lo))
        local = np.where(representable, 0.0, rounding)
        operands_exact = np.logical_and.reduce([np.broadcast_to(o.exact, value.shape) for o in operands])
        certified = (np.logical_and.reduce([np.broadcast_to(o.certified, value.shape) for o in operands])
                     & np.isfinite(value) & np.isfinite(transfer) & ((apart <= local) | representable))
        error = np.where(certified, _up(transfer + local), 0.0)
        return Bounded(native, error, certified & operands_exact & representable & (error == 0), certified)

    def _add(self, eqn, operands, natives, sign=1.0):
        if not np.issubdtype(natives[0].dtype, np.inexact):
            return self._integers(operands, natives)
        fmt, blocked = self._formatted("add", natives, *operands)
        if blocked:
            return blocked
        x, y = (operand.value.astype(np.float64) for operand in operands)
        finite = np.isfinite(x) & np.isfinite(y)
        with np.errstate(invalid="ignore"):
            hi, lo = two_sum(np.where(finite, x, 0.0), np.where(finite, sign * y, 0.0))
        transfer = _up(operands[0].error + operands[1].error)
        result = self._rounded(fmt, natives[0], hi, lo, transfer, operands)
        return [self._infinite(result, natives[0], x, sign * y, operands, finite)]

    def _infinite(self, result: Bounded, native, x, y, operands, finite) -> Bounded:
        """An exact infinity added to a certified finite value is that infinity exactly (IEEE), whatever the
        finite value's bound; any other sum holding an infinity or not a number is not certified."""
        (first, second), shape = operands, native.shape
        # The infinite operand exact, the finite one certified whatever its bound: its exact value is finite.
        held = ((np.isinf(x) & np.isfinite(y) & np.broadcast_to(first.exact & first.certified, shape)
                 & np.broadcast_to(second.certified, shape))
                | (np.isinf(y) & np.isfinite(x) & np.broadcast_to(second.exact & second.certified, shape)
                   & np.broadcast_to(first.certified, shape)))
        held = held & (native.astype(np.float64) == np.where(np.isinf(x), x, y))
        result.certified = np.where(finite, result.certified, held)
        result.exact = np.where(finite, result.exact, held)
        result.error = np.where(finite, result.error, 0.0)
        return result

    def _sub(self, eqn, operands, natives):
        return self._add(eqn, operands, natives, sign=-1.0)

    def _mul(self, eqn, operands, natives):
        if not np.issubdtype(natives[0].dtype, np.inexact):
            return self._integers(operands, natives)
        fmt, blocked = self._formatted("mul", natives, *operands)
        if blocked:
            return blocked
        (x, ex), (y, ey) = ((o.value.astype(np.float64), o.error) for o in operands)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = np.where(finite, x, 0.0), np.where(finite, y, 0.0)
        if fmt is FLOAT64:
            hi, lo = _two_product(x, y)
            # Dekker's residual is exact only where it cannot underflow; below, one least normal holds it.
            low = (np.abs(hi) < DEKKER_FLOOR) & (x != 0) & (y != 0)
            lo = np.where(low, _least_normal(FLOAT64), lo)
        else:
            hi, lo = x * y, np.zeros(np.broadcast(x, y).shape)
        transfer = _up(_times(np.abs(x), ey) + _times(np.abs(y), ex) + _times(ex, ey))
        # A product with an exact zero is that zero, whatever the other operand's bound.
        zero = ((x == 0) & np.broadcast_to(operands[0].exact, hi.shape)) | (
            (y == 0) & np.broadcast_to(operands[1].exact, hi.shape))
        result = self._rounded(fmt, natives[0], hi, lo, np.where(zero, 0.0, transfer), operands)
        result.exact = result.exact | (zero & result.certified)
        result.error = np.where(zero & result.certified, 0.0, result.error)
        # An infinity in a product is not certified: 0·inf has no value, and a product's overflow no bound.
        result.certified = result.certified & finite
        result.exact = result.exact & finite
        return [result]

    def _div(self, eqn, operands, natives):
        if not np.issubdtype(natives[0].dtype, np.inexact):
            return self._blocked("div (integer)", natives)
        fmt, blocked = self._formatted("div", natives, *operands)
        if blocked:
            return blocked
        (x, ex), (y, ey) = ((o.value.astype(np.float64), o.error) for o in operands)
        q = natives[0].astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(q) & (y != 0)
        x, y, q = np.where(finite, x, 0.0), np.where(finite, y, 1.0), np.where(finite, q, 0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            if fmt is FLOAT64:
                # The residual x - y·q exactly, where Dekker's product cannot underflow and x - p is exact
                # (Sterbenz: p within a factor of two of x).
                p, e = _two_product(y, q)
                near = (np.abs(p) >= np.abs(x) / 2) & (np.abs(p) <= 2 * np.abs(x))
                sterbenz = (q == 0) | ((x * p > 0) & near)
                valid = ((np.abs(p) >= DEKKER_FLOOR) | (q == 0)) & sterbenz
                residual = (x - p) - e
                representable = valid & (x - p == 0) & (e == 0)
                apart = np.where(valid, _up(np.abs(residual) / np.abs(y) * (1 + 2.0 ** -50)), np.inf)
            else:
                # Products of fp32 values are exact in float64, and x/y there within 2^-53 of its value.
                representable = y * q == x
                apart = _up(np.abs(q - x / y) + 2.0 ** -53 * np.abs(x / y))
            margin = np.abs(y) - ey
            lower = np.where(margin > 0, np.nextafter(np.abs(y) * margin * (1 - 2.0 ** -50), 0.0), 0.0)
            moved = _up(_times(np.abs(x), ey) + _times(np.abs(y), ex))
            divided = _up(moved / np.where(lower > 0, lower, 1.0))
            quotient = np.where(moved > 0, np.maximum(divided, TINIEST), 0.0)
            transfer = np.where(lower > 0, quotient, np.inf)
        result = self._rounded(fmt, natives[0], q, apart, transfer, operands, exact_from=representable)
        result.certified = result.certified & finite
        result.exact = result.exact & finite
        return [result]

    def _neg(self, eqn, operands, natives):
        operand = operands[0]
        return [Bounded(natives[0], operand.error, operand.exact, operand.certified)]

    def _abs(self, eqn, operands, natives):
        return self._neg(eqn, operands, natives)

    def _max(self, eqn, operands, natives):
        native = natives[0]
        if not np.issubdtype(native.dtype, np.inexact):
            return self._integers(operands, natives)
        error = _up(np.maximum(*(np.broadcast_to(o.error, native.shape) for o in operands)))
        exact = np.logical_and(*(np.broadcast_to(o.exact, native.shape) for o in operands))
        certified = np.logical_and(*(np.broadcast_to(o.certified, native.shape) for o in operands)) & (
            ~np.isnan(native.astype(np.float64)))
        return [Bounded(native, np.where(certified, error, 0.0), exact & certified, certified)]

    def _min(self, eqn, operands, natives):
        return self._max(eqn, operands, natives)

    def _integers(self, operands, natives):
        """An integer or boolean result of integer or boolean operands: exact where they all are."""
        native = natives[0]
        certified = np.logical_and.reduce([np.broadcast_to(o.certified & o.exact, native.shape)
                                           for o in operands])
        return [Bounded(native, np.zeros(native.shape), certified, certified)]

    def _and(self, eqn, operands, natives):
        return self._integers(operands, natives)

    _or = _not = _xor = _and

    def _iota(self, eqn, operands, natives):
        return [Bounded.exactly(natives[0])]

    def _convert_element_type(self, eqn, operands, natives):
        operand, native = operands[0], natives[0]
        source, target = operand.value.dtype, native.dtype
        if not np.issubdtype(target, np.inexact):
            if np.issubdtype(source, np.inexact):
                return self._blocked(f"convert_element_type ({source} to {target})", natives)
            return self._integers(operands, natives)
        fmt = FORMATS.get(target)
        if fmt is None:
            return self._blocked(f"convert_element_type (to {target})", natives)
        value = operand.value.astype(np.float64) if np.issubdtype(source, np.inexact) else operand.value
        hi = np.asarray(value, np.float64)
        lo = np.zeros(hi.shape)
        if not np.issubdtype(source, np.inexact):
            # An integer wider than the format's significand is checked back through it.
            representable = native.astype(source) == operand.value
            return [self._rounded(fmt, native, hi, lo, operand.error, [operand], exact_from=representable)]
        finite = np.isfinite(hi)
        result = self._rounded(fmt, native, np.where(finite, hi, 0.0), lo, operand.error, [operand])
        # An exact infinity casts to itself.
        held = ~finite & np.isinf(hi) & operand.exact & operand.certified & (native.astype(np.float64) == hi)
        result.certified = np.where(finite, result.certified, held)
        result.exact = np.where(finite, result.exact, held)
        result.error = np.where(finite, result.error, 0.0)
        return [result]

    # Comparisons: decided where the operands' bounds part, or both are exact.

    def _compare(self, operands, natives):
        native = natives[0]
        (x, ex), (y, ey) = ((o.value.astype(np.float64), o.error) for o in operands)
        exact = np.logical_and(*(o.exact for o in operands))
        # Parted bounds decide every relation, equality's as false.
        decided = exact | (x + ex < y - ey) | (x - ex > y + ey)
        certified = np.logical_and(*(o.certified for o in operands)) & decided & ~np.isnan(x) & ~np.isnan(y)
        return [Bounded(native, np.zeros(native.shape), certified, certified)]

    def _lt(self, eqn, operands, natives):
        return self._compare(operands, natives)

    def _le(self, eqn, operands, natives):
        return self._compare(operands, natives)

    def _gt(self, eqn, operands, natives):
        return self._compare(operands, natives)

    def _ge(self, eqn, operands, natives):
        return self._compare(operands, natives)

    def _eq(self, eqn, operands, natives):
        return self._compare(operands, natives)

    def _ne(self, eqn, operands, natives):
        return self._compare(operands, natives)

    # Pointwise functions at their exact points.

    def _exp(self, eqn, operands, natives):
        """exp, its native value held to an enclosure of the exact one at every element, so no library exp
        is trusted (the validated double-double of op_certificates for fp32 arguments, within EXP_BOUND;
        mpmath's interval exp for float64 ones): its rounding the native value's distance from the
        enclosure. Its bound carried by |exp(t) - exp(x̂)| <= exp(x̂)·e·(1 + e) for |t - x̂| <= e <= 1.
        exp(-inf) = 0 and exp(0) = 1 exactly where the native value is so; an argument past exp's range
        (a native infinity), not a number, or carrying a bound over one is not certified."""
        operand, native = operands[0], natives[0]
        fmt, blocked = self._formatted("exp", natives, operand)
        if blocked:
            return blocked
        x, value, e = operand.value.astype(np.float64), native.astype(np.float64), operand.error
        ordinary = np.isfinite(x) & np.isfinite(value) & (e <= 1) & operand.certified
        upper, local = np.zeros(x.shape), np.zeros(x.shape)
        if fmt is FLOAT32:
            inside = ordinary & (x >= -110) & (x <= 90)
            hi, lo = _exp_enclosed(np.where(inside, x, 0.0))
            slack = np.abs(hi) * EXP_BOUND
            upper = np.where(inside, _up(np.abs(hi) + np.abs(lo) + slack), 0.0)
            local = np.where(inside, _up(np.abs(value - hi) + np.abs(lo) + slack), 0.0)
            # Below -110 exp is under 2^-158, which the format holds as zero or flushes.
            below = ordinary & (x < -110)
            upper = np.where(below, 2.0 ** -150, upper)
            local = np.where(below, _up(np.abs(value) + 2.0 ** -150), local)
            ordinary = inside | below
        else:
            for index in np.flatnonzero(ordinary):
                try:
                    with interval_precision(128) as iv:
                        low, high = interval_ends(iv.exp(iv.mpf(float(x.flat[index]))), 128)
                    native_value = fractions.Fraction(float(value.flat[index]))
                    apart = max(abs(native_value - low), abs(high - native_value))
                    upper.flat[index] = math.nextafter(float(high), math.inf)
                    local.flat[index] = math.nextafter(float(apart), math.inf)
                except (ValueError, OverflowError):
                    ordinary.flat[index] = False
        transfer = _up(_times(upper, _up(e * (1 + e))))
        point = operand.exact & operand.certified & (
            ((x == -np.inf) & (value == 0.0)) | ((x == 0.0) & (value == 1.0)))
        certified = (ordinary | point) & np.isfinite(transfer + local)
        error = np.where(point, 0.0, np.where(certified, _up(transfer + local), 0.0))
        if (~certified).any():
            self.blocked["exp (past its range, or not a number)"] = (
                self.blocked.get("exp (past its range, or not a number)", 0) + int((~certified).sum()))
        return [Bounded(native, error, point & certified, certified)]

    def _enclosed(self, name, natives, operand, function) -> list[Bounded]:
        """A unary function held at every element to mpmath's interval enclosure of it over the operand's
        whole interval [x̂ - e, x̂ + e] (`function` of an mpmath interval): its bound the native value's
        farthest distance from that enclosure, which covers its own rounding and its operand's bound at
        once; exact where the enclosure is the native value alone. Outside its domain, or not finite,
        not certified."""
        native = natives[0]
        _, blocked = self._formatted(name, natives, operand)
        if blocked:
            return blocked
        x, value, e = operand.value.astype(np.float64), native.astype(np.float64), operand.error
        error, exact = np.zeros(x.shape), np.zeros(x.shape, bool)
        certified = operand.certified & np.isfinite(x) & np.isfinite(value)
        for index in np.flatnonzero(certified):
            try:
                with interval_precision(128) as iv:
                    radius = float(e.flat[index])
                    around = iv.mpf(float(x.flat[index])) + iv.mpf([-radius, radius])
                    low, high = interval_ends(function(iv, around), 128)
            except (ValueError, ZeroDivisionError, ArithmeticError):
                certified.flat[index] = False
                continue
            native_value = fractions.Fraction(float(value.flat[index]))
            apart = max(abs(native_value - low), abs(high - native_value))
            try:
                error.flat[index] = math.nextafter(float(apart), math.inf) if apart else 0.0
            except OverflowError:
                certified.flat[index] = False
                continue
            exact.flat[index] = low == high == native_value and bool(operand.exact.flat[index])
        certified &= np.isfinite(error)
        return [Bounded(native, np.where(certified, error, 0.0), exact & certified, certified)]

    def _log(self, eqn, operands, natives):
        return self._enclosed("log", natives, operands[0], lambda iv, t: iv.log(t))

    def _log1p(self, eqn, operands, natives):
        return self._enclosed("log1p", natives, operands[0], lambda iv, t: iv.log1p(t))

    def _expm1(self, eqn, operands, natives):
        return self._enclosed("expm1", natives, operands[0], lambda iv, t: iv.expm1(t))

    def _sqrt(self, eqn, operands, natives):
        return self._enclosed("sqrt", natives, operands[0], lambda iv, t: iv.sqrt(t))

    def _rsqrt(self, eqn, operands, natives):
        return self._enclosed("rsqrt", natives, operands[0], lambda iv, t: 1 / iv.sqrt(t))

    def _sin(self, eqn, operands, natives):
        return self._enclosed("sin", natives, operands[0], lambda iv, t: iv.sin(t))

    def _cos(self, eqn, operands, natives):
        return self._enclosed("cos", natives, operands[0], lambda iv, t: iv.cos(t))

    def _tanh(self, eqn, operands, natives):
        # Increasing: its ends' enclosures, each 1 - 2/(exp(2t) + 1).
        return self._enclosed("tanh", natives, operands[0], lambda iv, t: _monotone(
            iv, t, lambda point: 1 - 2 / (iv.exp(2 * point) + 1)))

    def _logistic(self, eqn, operands, natives):
        return self._enclosed("logistic", natives, operands[0], lambda iv, t: _monotone(
            iv, t, lambda point: 1 / (1 + iv.exp(-point))))

    def _integer_pow(self, eqn, operands, natives):
        """x², a correctly rounded product of x with itself (XLA lowers it so): |t² - x̂²| <= 2|x̂|e + e²."""
        if eqn.params["y"] != 2:
            return self._blocked(f"integer_pow (y = {eqn.params['y']})", natives)
        return self._mul(eqn, [operands[0], operands[0]], natives)

    def _square(self, eqn, operands, natives):
        return self._mul(eqn, [operands[0], operands[0]], natives)

    # Sums.

    def _reduce_sum(self, eqn, operands, natives):
        operand, native = operands[0], natives[0]
        axes = tuple(eqn.params["axes"])
        if not np.issubdtype(native.dtype, np.inexact):
            certified = np.all(operand.certified & operand.exact, axis=axes)
            return [Bounded(native, np.zeros(native.shape), certified, certified)]
        fmt, blocked = self._formatted("reduce_sum", natives, operand)
        if blocked:
            return blocked
        x = np.abs(operand.value.astype(np.float64))
        terms = int(np.prod([operand.value.shape[a] for a in axes])) if axes else 1
        nonzero = np.asarray(np.sum(operand.value != 0, axis=axes))
        with np.errstate(over="ignore", invalid="ignore"):
            magnitude = _widened_sum(np.sum(x, axis=axes), terms, nonzero)
        positive = np.sum(operand.error > 0, axis=axes)
        propagated = _widened_sum(np.sum(operand.error, axis=axes), terms, positive)
        rounding = np.array([gamma(int(m) - 1, fmt) if m > 1 else 0.0 for m in nonzero.reshape(-1)]).reshape(
            nonzero.shape)
        local = np.where(nonzero > 1, _up(_times(rounding, magnitude) + nonzero * _least_normal(fmt)), 0.0)
        certified = np.all(operand.certified, axis=axes) & np.isfinite(native.astype(np.float64))
        exact = certified & np.all(operand.exact, axis=axes) & (nonzero <= 1)
        return [Bounded(native, np.where(certified, _up(propagated + local), 0.0), exact, certified)]

    def _dot_general(self, eqn, operands, natives):
        native = natives[0]
        fmt, blocked = self._formatted("dot_general", natives, *operands)
        if blocked:
            return blocked
        if jax.default_backend() != "cpu":
            return self._blocked(f"dot_general (no contract on {jax.default_backend()})", natives)
        precision, preferred = eqn.params.get("precision"), eqn.params.get("preferred_element_type")
        if not _probed_dot(precision, preferred, operands[0].value.dtype):
            return self._blocked(f"dot_general (precision {precision} unprobed)", natives)
        (a, ea, ca), (b, eb, cb) = ((o.value.astype(np.float64), o.error, o.certified) for o in operands)
        numbers = eqn.params["dimension_numbers"]

        def dot(left, right):
            # XLA may read a subnormal operand as zero: each positive one raised to the least normal, an
            # upper bound of it, so no bound product is lost.
            least = _least_normal(FLOAT64)
            left, right = (np.where((v > 0) & (v < least), least, v)
                           for v in (np.asarray(left, np.float64), np.asarray(right, np.float64)))
            with jax.enable_x64(new_val=True):
                left, right = jnp.asarray(left, jnp.float64), jnp.asarray(right, jnp.float64)
                highest = jax.lax.Precision.HIGHEST
                return np.asarray(jax.lax.dot_general(left, right, numbers, precision=highest))

        terms = int(np.prod([a.shape[axis] for axis in numbers[0][0]])) if numbers[0][0] else 1
        nonzero = np.asarray(dot(a != 0, b != 0))
        magnitude = _widened_sum(dot(np.abs(a), np.abs(b)), terms, nonzero)
        positive = dot(a != 0, eb > 0) + dot(ea > 0, b != 0) + dot(ea > 0, eb > 0)
        propagated = _widened_sum(dot(np.abs(a), eb) + dot(ea, np.abs(b)) + dot(ea, eb), 3 * terms, positive)
        rounding = np.array([gamma(int(m), fmt) if m > 0 else 0.0 for m in nonzero.reshape(-1)]).reshape(
            nonzero.shape)
        local = np.where(nonzero > 0, _up(_times(rounding, magnitude) + nonzero * _least_normal(fmt)), 0.0)
        certified = (dot(~ca, np.ones(b.shape)) == 0) & (dot(np.ones(a.shape), ~cb) == 0) & np.isfinite(
            native.astype(np.float64))
        exact = certified & (nonzero == 0) & (propagated == 0)
        return [Bounded(native, np.where(certified, _up(propagated + local), 0.0), exact, certified)]

    def _reduce_max(self, eqn, operands, natives):
        operand, native = operands[0], natives[0]
        axes = tuple(eqn.params["axes"])
        certified = np.all(operand.certified, axis=axes) & ~np.isnan(native.astype(np.float64))
        exact = certified & np.all(operand.exact, axis=axes)
        return [Bounded(native, np.where(certified, np.max(operand.error, axis=axes), 0.0), exact, certified)]

    def _reduce_min(self, eqn, operands, natives):
        return self._reduce_max(eqn, operands, natives)


def _monotone(iv, interval, function):
    """An increasing function's enclosure over `interval`: its enclosures at the two ends joined."""
    low, high = function(iv.mpf(interval.a)), function(iv.mpf(interval.b))
    return iv.mpf([low.a, high.b])


_PROBES: dict = {}


def _probed_dot(precision, preferred, dtype) -> bool:
    """Whether this backend's dot at `precision` (none, or a precision per operand; never an algorithm,
    which may convert its operands) rounds each product and its accumulation in the operands' own format:
    probed natively, a product (1 + 2^-12)² that a narrower product loses, and a sum 2^20 + 1.125 that a
    narrower accumulator loses (for float64, (1 + 2^-27)² and 2^40 + (1 + 2^-10)), each its correctly
    rounded value exactly."""
    if precision is not None and not (isinstance(precision, tuple)
                                      and all(isinstance(p, jax.lax.Precision) for p in precision)):
        return False
    if preferred is not None and np.dtype(preferred) != np.dtype(dtype):
        return False
    key = (precision, np.dtype(dtype))
    if key not in _PROBES:
        wide = np.dtype(dtype) == np.float64
        small, big, rest = (2.0 ** -27, 2.0 ** 40, 1 + 2.0 ** -10) if wide else (2.0 ** -12, 2.0 ** 20, 1.125)
        fmt = FLOAT64 if wide else FLOAT32
        with jax.enable_x64(new_val=wide):
            numbers = (((0,), (0,)), ((), ()))

            def run(left, right):
                left, right = jnp.asarray(left, dtype), jnp.asarray(right, dtype)
                return float(jax.lax.dot_general(left, right, numbers, precision=precision))

            square = fractions.Fraction(1 + small) ** 2
            product = run([1 + small], [1 + small]) == fraction_rounded(square, fmt)
            summed = run([1.0, 1.0], [big, rest]) == big + rest
        _PROBES[key] = product and summed
    return _PROBES[key]


@dataclasses.dataclass
class Coverage:
    """What the certificate judged of a site's zero elements: how many were certified and held to their
    bound, how many were not certified and why (a primitive without a contract on Dew's or the truth's
    side, or an interpretation its native execution does not reproduce), the primitives that left values
    uncertified, and each certified element past its bound."""
    certified: int
    uncertified: int
    causes: dict[str, int]
    blocked: dict[str, int]
    over: list[str]


UNTRACEABLE = (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError,
               jax.errors.TracerBoolConversionError, jax.errors.TracerIntegerConversionError)
"""A function that reads its arguments' values in Python is no program to interpret: its elements
are not certified, and named so."""


def _interpreted(function, inputs, outputs: list[np.ndarray]) -> tuple[list[Bounded], dict[str, int]]:
    """`function`'s program interpreted at `inputs`; where it cannot be traced, its `outputs` (as it ran)
    each not certified, the cause named."""
    try:
        closed = jax.make_jaxpr(function)(*inputs)
    except UNTRACEABLE as error:
        name = f"not a traceable program ({type(error).__name__})"
        return [Bounded(out, np.zeros(out.shape), *(np.zeros(out.shape, bool) for _ in range(2)))
                for out in outputs], {name: sum(int(out.size) for out in outputs)}
    interpreter = Interpreter()
    return interpreter.run(closed, *jax.tree_util.tree_leaves(inputs)), interpreter.blocked


def _widened(value) -> np.ndarray:
    value = np.asarray(value)
    return value.astype(np.float64) if np.issubdtype(value.dtype, np.floating) else value


def _bitwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Where two arrays hold the same bits, element by element (False throughout where their dtypes or
    shapes differ)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return np.zeros(b.shape, bool)
    if not np.issubdtype(a.dtype, np.inexact):
        return a == b
    unsigned = np.dtype(f"uint{8 * a.dtype.itemsize}")
    return a.view(unsigned) == b.view(unsigned)


def zero_elements(dew_fn, reference_fn, inputs, label: str, truth_fn=None) -> Coverage:
    """Check 2 at the elements of Dew's output whose float64 truth is exactly zero (the module's
    certificate), the truth the reference's executed float64 output: each where Dew's interpretation and
    the truth's are certified and each reproduces its native execution bit for bit, held to
    `|dew - truth| <= e + τ`; `inputs` the tuple of the functions' arguments. Its coverage; an element
    past its bound is listed in `over`."""
    native = [np.asarray(out) for out in jax.tree_util.tree_leaves(dew_fn(*inputs))]
    mine, blocked = _interpreted(dew_fn, inputs, native)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(_widened, inputs)
        executed = [np.asarray(out) for out in jax.tree_util.tree_leaves((truth_fn or reference_fn)(*wide))]
        truths, blocked_truth = _interpreted(truth_fn or reference_fn, wide, executed)
    causes = {"Dew's interpretation not certified": 0, "the truth's interpretation not certified": 0,
              "Dew's execution not the interpretation": 0, "the truth's execution not the interpretation": 0}
    certified = 0
    over = []
    for index, (bounded, out, truth, value) in enumerate(zip(mine, native, truths, executed, strict=True)):
        zero = value.astype(np.float64) == 0.0
        reasons = [(~bounded.certified, "Dew's interpretation not certified"),
                   (~truth.certified, "the truth's interpretation not certified"),
                   (~_bitwise(bounded.value, out), "Dew's execution not the interpretation"),
                   (~_bitwise(truth.value, value), "the truth's execution not the interpretation")]
        held = zero.copy()
        for failed, cause in reasons:
            causes[cause] += int((held & failed).sum())
            held &= ~failed
        certified += int(held.sum())
        apart = np.abs(out.astype(np.float64) - value.astype(np.float64))
        allowed = _up(bounded.error + truth.error)
        for at in np.argwhere(held & ~(apart <= allowed))[:4]:
            at = tuple(int(i) for i in at)
            over.append(f"{label}: leaf {index} at {at}: {apart[at]:.3e} from a truth of exactly zero, past "
                        f"Dew's own rounding there {bounded.error[at]:.3e} and the truth's "
                        f"{truth.error[at]:.3e}")
    merged = dict(blocked)
    merged.update({f"{name} (truth)": count for name, count in blocked_truth.items()})
    found = {cause: count for cause, count in causes.items() if count}
    return Coverage(certified, sum(found.values()), found, merged, over)


def assert_zero_elements(dew_fn, reference_fn, inputs, label: str, truth_fn=None) -> Coverage:
    """`zero_elements`, failing on any certified element past its bound."""
    coverage = zero_elements(dew_fn, reference_fn, inputs, label, truth_fn)
    assert not coverage.over, "\n".join(coverage.over)
    return coverage


__all__ = ["Bounded", "Coverage", "Interpreter", "assert_zero_elements", "zero_elements"]
