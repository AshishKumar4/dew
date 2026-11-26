"""Magnitude, sign, floor, and structural provenance proofs for `dew.algebra.terms` (`terms.proven` and
`terms.region`): `Apart`, `Proven`, lossless float cast checks, L2-norm denominator tracking (`_NormTrack`),
softmax max-shifted exponential/sum tracking (`_SoftmaxTrack`), and deep slice refinement."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax.core import ShapedArray
from jax.extend import core as jcore

from dew.nn import programs

type Magnitude = int
"""A power of two bounding a value's magnitude, `|x| <= 2**bound` (`NOTHING` for 0, an integer for every
other bound; `None` where no number bounds the value)."""

NOTHING = -(2 ** 20)
"""A zero's magnitude: under every other."""

UNIT = 0
"""The power of two for `2**0 = 1`."""


class Apart(NamedTuple):
    """What a value is known to hold beside its magnitude, as its dtype holds it: `floor`, a power with
    `2**floor <= |x|` (None where nothing holds it away from zero; a float's alone); `sign`, 1 where
    `x >= 0`, -1 where `x <= 0`, 0 where either may be; `number`, that it is no NaN; and `finite`, that it
    is a finite value its magnitude bound holds of as computed; each proven of it from the program's inputs
    taken as finite reals within their declared ranges.

    A zero of either sign is on both sides: only a floor says a value is none, so a reciprocal's sign is
    read where its operand has one. A sign and a floor are a number's alone, and the bounds read from a
    floor (`_apart_bound`) read an operand's magnitude of a finite one alone. A value that may have
    overflowed is not finite, nor is one an equation makes by keeping or scaling it, a conversion to a
    wider float or to an integer among them, whatever its bound reads there; a value is finite again where
    its operation bounds it whatever its operand is (a boolean, a logistic, a zeroth power) or makes a
    zero of an infinity (a quotient by one, its reciprocal root, its negative power). The bounds that read
    no floor keep their own convention, every value a finite real (`_real_bound`)."""

    floor: int | None = None
    sign: int = 0
    number: bool = False
    finite: bool = False
    unit_norm: bool = False

    @property
    def above(self) -> int | None:
        """Its floor where `x >= 2**floor`, above zero; None where it is not known so."""
        return self.floor if self.sign > 0 else None

    @property
    def positive(self) -> bool:
        """Whether `x >= 2**floor`: above zero."""
        return self.above is not None


NOWHERE = Apart()
"""A value nothing is proven of: one an equation outside the language made, or one with no rule here."""
WHOLE = Apart(number=True, finite=True)
"""A finite number of no known side: an input of no declared interval, an integer or a boolean before its
sign."""


class Proven(NamedTuple):
    """An input's declaration as a program around it proves it (`dew.algebra.program._declared`): its
    magnitude's bound where one holds (a power of two, 0 for nothing; None for none) and what else it holds
    (`Apart`). A window cut from a program takes each value it reads from the program so, not as every
    input is taken, a finite real: a value the program proves no number (a NaN's tanh made outside the
    window) bounds nothing within it (spec alignment, 8993)."""

    bound: float | None
    apart: Apart


def typed_key(value) -> bool:
    """Whether `value` holds typed PRNG keys (`jax.random.key`), whose bits NumPy cannot read."""
    dtype = getattr(value, "dtype", None)
    return (dtype is not None and not isinstance(dtype, (str, np.dtype))
            and jnp.issubdtype(dtype, jax.dtypes.prng_key))


def _shape(atom) -> tuple[int, ...]:
    """An array value's shape; a value of no shape (a token, a key's) has none the rules read."""
    return tuple(atom.aval.shape) if isinstance(atom.aval, ShapedArray) else ()


def _reduced(equation: jcore.JaxprEqn) -> int:
    """The elements a reduction along axes reads for each it makes."""
    shape = _shape(equation.invars[0])
    return math.prod(shape[axis] for axis in equation.params["axes"])


def _host_array(value) -> np.ndarray | None:
    """Return `np.asarray(value)` for host constants, or None for PRNG keys, tracers, or concrete device
    `jax.Array`s (avoiding device transfers and `TracerArrayConversionError`)."""
    if typed_key(value) or isinstance(value, (jax_core.Tracer, jax.Array)):
        return None
    return np.asarray(value)


def _log2(value: float) -> int:
    """`ceil(log2(value))`, `NOTHING` for 0: a magnitude's exponent bounding `value` from above, exactly, by
    the float's own exponent (a logarithm rounds, and `log2` of the float under 8 is 3)."""
    if value == 0:
        return NOTHING
    mantissa, exponent = math.frexp(value)
    return exponent - 1 if mantissa == 0.5 else exponent


def _floor_log2(value: float) -> int:
    """`floor(log2(value))` of a positive float, exactly, by its exponent."""
    return math.frexp(value)[1] - 1


def _magnitude(value) -> Magnitude | None:
    """A constant's magnitude, its elements' largest (a complex one's modulus); None for one of no number (a
    key's, tracer's, or concrete device array's). A number of any dtype, ml_dtypes' (bfloat16, float8) among
    them, which numpy's type tree omits."""
    array = _host_array(value)
    if array is None:
        return None
    if not (jnp.issubdtype(array.dtype, jnp.number) or array.dtype == bool):
        return None
    # In complex128, which holds every integer's absolute value (int32's least's too) and a complex modulus.
    largest = float(np.max(np.abs(array.astype(np.complex128)))) if array.size else 0.0
    return _log2(largest) if math.isfinite(largest) else None


def _declared(declared: float | tuple[float, float] | Proven | None) -> Magnitude | None:
    """An input's magnitude from its declared range (an interval's larger end's), None where it declares
    none."""
    if isinstance(declared, Proven):
        declared = declared.bound
    if declared is None:
        return None
    largest = max(abs(declared[0]), abs(declared[1])) if isinstance(declared, tuple) else declared
    return _log2(largest) if math.isfinite(largest) else None


def _held_apart(apart: Apart, dtype) -> Apart:
    """`apart` as `dtype` holds it: a sign and a floor of a number alone; a floor only of a float, and only
    a normal power of it, which a result at or above it rounds to nothing under (rounding to nearest is
    monotone; a subnormal one, or one past the dtype's largest, is dropped); an integer a number whatever
    made it, and a boolean a finite one; a complex value nothing."""
    if jnp.issubdtype(dtype, jnp.integer) or dtype == np.bool_:
        return Apart(None, apart.sign if apart.number else 0, number=True,
                     finite=apart.finite or dtype == np.bool_)
    if not jnp.issubdtype(dtype, jnp.floating):
        return NOWHERE
    if not apart.number:
        return NOWHERE
    floats = jnp.finfo(dtype)
    unit = apart.unit_norm and jnp.dtype(dtype).itemsize >= 2
    floor = (apart.floor if apart.floor is not None and int(floats.minexp) <= apart.floor < int(floats.maxexp)
             else None)
    return apart if (floor == apart.floor and unit == apart.unit_norm) else apart._replace(
        floor=floor, unit_norm=unit)


def _constant_apart(value) -> Apart:
    """A real constant's `Apart`: whether it holds no NaN and no infinity, its sign where its elements share
    one, and a float's least magnitude's power where no element is zero; none of a key's, tracer's, or
    concrete device array's."""
    array = _host_array(value)
    if array is None:
        return NOWHERE
    real = jnp.issubdtype(array.dtype, jnp.floating) or jnp.issubdtype(array.dtype, jnp.integer)
    if not real or not array.size:
        return NOWHERE
    if not jnp.issubdtype(array.dtype, jnp.floating):
        return WHOLE._replace(sign=1 if (array >= 0).all() else -1 if (array <= 0).all() else 0)
    values = array.astype(np.float64)
    if np.isnan(values).any():
        return NOWHERE
    sign = 1 if (values >= 0).all() else -1 if (values <= 0).all() else 0
    least = float(np.min(np.abs(values)))
    return Apart(_floor_log2(least) if 0 < least < math.inf else None, sign, number=True,
                 finite=bool(np.isfinite(values).all()))


def _declared_apart(declared: float | tuple[float, float] | Proven | None) -> Apart:
    """An input's `Apart`: a finite number, as every input is taken to be (a real, within its declared range
    where it has one); and, of an interval, its end nearer zero where the interval holds none and the side
    of zero it lies on."""
    if declared is None:
        return WHOLE
    if isinstance(declared, Proven):
        return declared.apart
    if not isinstance(declared, tuple):
        return Apart(number=True, finite=math.isfinite(declared))
    low, high = declared
    nearest = low if low > 0 else -high if high < 0 else 0.0
    return Apart(_floor_log2(nearest) if 0 < nearest < math.inf else None,
                 1 if low >= 0 else -1 if high <= 0 else 0, number=True,
                 finite=math.isfinite(low) and math.isfinite(high))


def _within(bound: Magnitude | None, dtype) -> bool:
    """Whether a float of `dtype` bounded by `2**bound` in the reals is finite as computed: its bound is
    known and under the dtype's largest power, which a value under it cannot round past."""
    return bound is not None and (bound == NOTHING or bound < int(jnp.finfo(dtype).maxexp))


def _apart_of(atom, apart: Mapping[jcore.Var, Apart]) -> Apart:
    """`atom`'s proven `Apart` from `apart`, or its constant `Apart` for a literal."""
    if isinstance(atom, jcore.Literal):
        if isinstance(atom.val, (jax_core.Tracer, jax.Array)) and not typed_key(atom.val):
            return _held_apart(WHOLE, atom.aval.dtype) if isinstance(atom.aval, ShapedArray) else NOWHERE
        dtype = atom.val.dtype if typed_key(atom.val) else np.asarray(atom.val).dtype
        return _held_apart(_constant_apart(atom.val), dtype)
    return apart.get(atom, NOWHERE)


def _side_matches(held: Apart, bound: Magnitude | None, side: int) -> bool:
    """Whether a value proven `(bound, held)` lies on `side` (`>= 0` for `1`, `<= 0` for `-1`), treating an
    exact zero (`bound == NOTHING` and `held.number`) as on both sides."""
    return held.sign == side or (held.number and bound == NOTHING)


def _shared_sign(operands: Sequence[Apart], bounds: Sequence[Magnitude | None]) -> int:
    """Common sign (`1` if all `>= 0`, `-1` if all `<= 0`, else `0`) honoring exact zero `NOTHING`."""
    if all(_side_matches(h, b, 1) for h, b in zip(operands, bounds, strict=True)):
        return 1
    if all(_side_matches(h, b, -1) for h, b in zip(operands, bounds, strict=True)):
        return -1
    return 0


def _reshape_kept(kept: tuple[int | None, ...], in_shape: tuple[int, ...],
                  out_shape: tuple[int, ...]) -> tuple[int | None, ...] | None:
    """Propagate tracked source axes across a reshape that only inserts or drops size-1 axes."""
    in_non1 = [(axis, size) for axis, size in enumerate(in_shape) if size != 1]
    out_non1 = [(axis, size) for axis, size in enumerate(out_shape) if size != 1]
    if [size for _, size in in_non1] != [size for _, size in out_non1]:
        return None
    mapped: list[int | None] = [None] * len(out_shape)
    for (in_ax, _), (out_ax, _) in zip(in_non1, out_non1, strict=True):
        mapped[out_ax] = kept[in_ax]
    return tuple(mapped)


def _broadcast_kept(kept: tuple[int | None, ...], in_shape: tuple[int, ...],
                    out_shape: tuple[int, ...], dims: Sequence[int]) -> tuple[int | None, ...]:
    """Propagate tracked source axes across `broadcast_in_dim_p`, clearing broadcast size-1 axes to None."""
    mapped: list[int | None] = [None] * len(out_shape)
    for in_ax, out_ax in enumerate(dims):
        if in_shape[in_ax] == out_shape[out_ax]:
            mapped[out_ax] = kept[in_ax]
    return tuple(mapped)


def _move_kept(equation: jcore.JaxprEqn, src: jcore.Var, out: jcore.Var,
               kept: tuple[int | None, ...]) -> tuple[int | None, ...] | None:
    """Propagate tracked source axes `kept` across `broadcast_in_dim_p` or a non-permuting `reshape_p`."""
    in_shape, out_shape = _shape(src), _shape(out)
    if equation.primitive is jax.lax.broadcast_in_dim_p:
        return _broadcast_kept(kept, in_shape, out_shape, equation.params["broadcast_dimensions"])
    dims = equation.params.get("dimensions")
    if dims is not None and tuple(dims) != tuple(range(len(in_shape))):
        return None
    return _reshape_kept(kept, in_shape, out_shape)


def _aligned_unit_axes(kept: tuple[int | None, ...], rank: int) -> bool:
    """Whether `kept` aligns 1-to-1 with every non-reduced axis of a rank-`rank` tensor and reduces at least
    one axis."""
    return (len(kept) == rank and any(entry is None for entry in kept)
            and all(entry in (axis, None) for axis, entry in enumerate(kept)))


def dtype_lossless_in(src_dtype, dst_dtype) -> bool:
    """Whether every finite value (normal or subnormal) of floating `src_dtype` is exact in `dst_dtype`."""
    src_dt, dst_dt = jnp.dtype(src_dtype), jnp.dtype(dst_dtype)
    if src_dt == dst_dt:
        return True
    if not (jnp.issubdtype(src_dt, jnp.floating) and jnp.issubdtype(dst_dt, jnp.floating)):
        return False
    src, dst = jnp.finfo(src_dt), jnp.finfo(dst_dt)
    return (int(src.nmant) <= int(dst.nmant)
            and int(src.minexp) >= int(dst.minexp)
            and int(src.minexp) - int(src.nmant) >= int(dst.minexp) - int(dst.nmant)
            and int(src.maxexp) <= int(dst.maxexp)
            and float(src.min) >= float(dst.min) and float(src.max) <= float(dst.max))


def _lossless_float_cast(source, out) -> bool:
    """Whether `convert_element_type` preserves every finite floating value losslessly."""
    return (isinstance(source.aval, ShapedArray) and isinstance(out.aval, ShapedArray)
            and dtype_lossless_in(source.aval.dtype, out.aval.dtype))


class _NormTrack:
    """Tracks verified L2-norm denominator provenance (`sum(x^2, axes) + eps` with `eps > 0`) for
    `proven(..., deep=True)` so `x * rsqrt(sum(x^2) + eps)` and `x / sqrt(sum(x^2) + eps)` prove unit norm
    (`bound = UNIT + 1`, finite number) without false positives across mismatched axes or operands."""

    def __init__(self, sq_sum: dict | None = None, inv_norm: dict | None = None,
                 pos_norm: dict | None = None) -> None:
        self.origin: dict[jcore.Var, jcore.Var] = {}
        self.squared: dict[jcore.Var, tuple[jcore.Var, int]] = {}
        self.sq_sum: dict[jcore.Var, tuple[jcore.Var, tuple[int | None, ...], bool, int]] = dict(sq_sum or {})
        self.inv_norm: dict[jcore.Var, tuple[jcore.Var, tuple[int | None, ...]]] = dict(inv_norm or {})
        self.pos_norm: dict[jcore.Var, tuple[jcore.Var, tuple[int | None, ...]]] = dict(pos_norm or {})

    def _canon(self, var: jcore.Var) -> jcore.Var:
        return self.origin.get(var, var)

    def _propagate_layout(self, equation: jcore.JaxprEqn, src: jcore.Var, out: jcore.Var) -> None:
        if ((entry := self.sq_sum.get(src)) is not None
                and (moved := _move_kept(equation, src, out, entry[1])) is not None):
            self.sq_sum[out] = (entry[0], moved, entry[2], entry[3])
        for table in (self.inv_norm, self.pos_norm):
            if ((entry := table.get(src)) is not None
                    and (moved := _move_kept(equation, src, out, entry[1])) is not None):
                table[out] = (entry[0], moved)

    def _unit_check(self, scaled: jcore.Var | jcore.Literal, divisor: jcore.Var | jcore.Literal,
                    table: Mapping[jcore.Var, tuple[jcore.Var, tuple[int | None, ...]]],
                    out_shape: tuple[int, ...]) -> tuple[bool, bool]:
        if not (isinstance(scaled, jcore.Var) and isinstance(divisor, jcore.Var)) or divisor not in table:
            return False, False
        src, kept = table[divisor]
        if (self._canon(scaled) is not src or _shape(scaled) != out_shape
                or not _aligned_unit_axes(kept, len(out_shape))):
            return False, False
        return True, bool(kept) and kept[-1] is None

    def _step_cast(self, src: jcore.Var, out: jcore.Var) -> None:
        if not _lossless_float_cast(src, out):
            return
        self.origin[out] = self._canon(src)
        if src in self.sq_sum:
            self.sq_sum[out] = self.sq_sum[src]
        for table in (self.inv_norm, self.pos_norm):
            if src in table:
                table[out] = table[src]

    def _step_add_max(self, prim: jcore.Primitive, invars: Sequence, out: jcore.Var,
                      apart_of: Callable[..., Apart]) -> None:
        for i in (0, 1):
            cand, other = invars[i], invars[1 - i]
            if not isinstance(cand, jcore.Var):
                continue
            if (entry := self.sq_sum.get(cand)) is not None and (oa := apart_of(other)).sign > 0:
                floored = entry[2] or (oa.above is not None and oa.above >= entry[3])
                self.sq_sum[out] = (entry[0], entry[1], floored, entry[3])
            if (prim is jax.lax.max_p and (pos := self.pos_norm.get(cand)) is not None
                    and apart_of(other).sign > 0):
                self.pos_norm[out] = pos

    def step(self, equation: jcore.JaxprEqn,
             apart_of: Callable[..., Apart]) -> tuple[Magnitude | None, Apart] | None:
        """Update norm provenance for `equation`, returning unit-norm facts when `equation` divides/scales
        `x` by a verified positive-floored L2 norm of `x`."""
        if len(equation.outvars) != 1 or not isinstance(out_aval := equation.outvars[0].aval, ShapedArray):
            return None
        prim, out, invars = equation.primitive, equation.outvars[0], equation.invars
        if prim is jax.lax.convert_element_type_p and isinstance(invars[0], jcore.Var):
            self._step_cast(invars[0], out)
            return None
        if (prim is jax.lax.square_p
                or (prim is jax.lax.integer_pow_p and equation.params.get("y") == 2)
                or (prim is jax.lax.mul_p and len(invars) == 2 and invars[0] is invars[1])):
            if isinstance(invars[0], jcore.Var) and jnp.issubdtype(out_aval.dtype, jnp.floating):
                self.squared[out] = (self._canon(invars[0]), int(jnp.finfo(out_aval.dtype).minexp))
            return None
        if prim is jax.lax.reduce_sum_p and isinstance(invars[0], jcore.Var):
            if ((sq := self.squared.get(invars[0])) is not None
                    and jnp.issubdtype(out_aval.dtype, jnp.floating)):
                red = frozenset(int(ax) for ax in equation.params["axes"])
                kept = tuple(ax for ax in range(len(_shape(invars[0]))) if ax not in red)
                self.sq_sum[out] = (sq[0], kept, False, max(sq[1], int(jnp.finfo(out_aval.dtype).minexp)))
            return None
        if prim in (jax.lax.broadcast_in_dim_p, jax.lax.reshape_p) and isinstance(invars[0], jcore.Var):
            self._propagate_layout(equation, invars[0], out)
            return None
        if prim in (jax.lax.add_p, jax.lax.max_p) and len(invars) == 2:
            self._step_add_max(prim, invars, out, apart_of)
            return None
        if prim in (jax.lax.rsqrt_p, jax.lax.sqrt_p) and isinstance(invars[0], jcore.Var):
            if (entry := self.sq_sum.get(invars[0])) is not None and entry[2]:
                (self.inv_norm if prim is jax.lax.rsqrt_p else self.pos_norm)[out] = (entry[0], entry[1])
            return None
        if prim in (jax.lax.mul_p, jax.lax.div_p) and len(invars) == 2:
            out_shape = tuple(out_aval.shape)
            if prim is jax.lax.mul_p:
                hit, last = self._unit_check(invars[0], invars[1], self.inv_norm, out_shape)
                if not hit:
                    hit, last = self._unit_check(invars[1], invars[0], self.inv_norm, out_shape)
            else:
                hit, last = self._unit_check(invars[0], invars[1], self.pos_norm, out_shape)
            if hit:
                return UNIT + 1, _held_apart(
                    Apart(None, 0, number=True, finite=True, unit_norm=last), out_aval.dtype)
        return None

    def child(self, outer_invars: Sequence, inner_invars: Sequence[jcore.Var]) -> _NormTrack:
        """Map tracked norm entries into a peeled call body when both source and sum enter the call."""
        pairs = [(o, i) for o, i in zip(outer_invars, inner_invars, strict=True) if isinstance(o, jcore.Var)]
        by_origin: dict[jcore.Var, jcore.Var] = {}
        for outer, inner in pairs:
            by_origin.setdefault(self._canon(outer), inner)
        return _NormTrack(*({i: (by_origin[e[0]], *e[1:]) for o, i in pairs
                             if (e := t.get(o)) is not None and e[0] in by_origin}
                            for t in (self.sq_sum, self.inv_norm, self.pos_norm)))


def _is_neg_inf(atom) -> bool:
    """Whether `atom` is a host floating literal equal to `-inf` everywhere (`jnp.max(..., initial=-inf)`)."""
    if not isinstance(atom, jcore.Literal):
        return False
    values = _host_array(atom.val)
    return (values is not None and values.size > 0 and jnp.issubdtype(values.dtype, jnp.floating)
            and bool(np.all(values == -np.inf)))


class _SoftmaxTrack:
    """Tracks max-shifted softmax provenance across `reduce_max -> sub -> exp/exp2 -> reduce_sum` for both
    `terms.region` and `terms.proven`:
    - `x - max(x, axes)` of a finite float `x` is a non-NaN value `<= 0` with an exact `0.0` along `axes`;
    - `exp(x - max(x))` / `exp2(x - max(x))` is therefore finite in `[0, 1]` with an exact `1.0` along `axes`
      (`_prove_equation` gives `bound = UNIT + 1`, `sign = 1`, `finite = True`);
    - `reduce_sum(exp(x - max(x)), S)` for any non-empty `S >= axes` sums at least one exact `1.0` beside
      nonnegative terms, proving `floor = 0` (`s >= 1`), `positive = True`, finite `UNIT + 1 + log2(N)`."""

    def __init__(self, max_of: dict | None = None, nonpos_max: dict | None = None,
                 unit_exp: dict | None = None, neg_inf: set[jcore.Var] | None = None) -> None:
        self.origin: dict[jcore.Var, jcore.Var] = {}
        self.max_of: dict[jcore.Var, tuple[jcore.Var, tuple[int | None, ...]]] = dict(max_of or {})
        self.nonpos_max: dict[jcore.Var, frozenset[int]] = dict(nonpos_max or {})
        self.unit_exp: dict[jcore.Var, frozenset[int]] = dict(unit_exp or {})
        self.neg_inf: set[jcore.Var] = set(neg_inf or ())

    def _canon(self, var: jcore.Var) -> jcore.Var:
        return self.origin.get(var, var)

    def _is_neg_inf_atom(self, atom) -> bool:
        return _is_neg_inf(atom) or (isinstance(atom, jcore.Var) and atom in self.neg_inf)

    def passthrough(self, src: jcore.Var, out: jcore.Var) -> None:
        """Forward softmax provenance across a value-preserving barrier (`stop_gradient_p`)."""
        self.origin[out] = self._canon(src)
        if src in self.max_of:
            self.max_of[out] = self.max_of[src]
        if src in self.nonpos_max:
            self.nonpos_max[out] = self.nonpos_max[src]
        if src in self.unit_exp:
            self.unit_exp[out] = self.unit_exp[src]
        if src in self.neg_inf:
            self.neg_inf.add(out)

    def step(self, equation: jcore.JaxprEqn, bound: Magnitude | None, proved: Apart,
             apart_of: Callable[..., Apart]) -> tuple[Magnitude | None, Apart]:
        """Update softmax provenance for `equation` and refine `(bound, proved)` when `sub(x, max(x))` or
        `reduce_sum(exp(x - max(x)))` fires."""
        if len(equation.outvars) != 1 or not isinstance(out_aval := equation.outvars[0].aval, ShapedArray):
            return bound, proved
        prim, out, invars = equation.primitive, equation.outvars[0], equation.invars
        if (prim in (jax.lax.convert_element_type_p, jax.lax.broadcast_in_dim_p, jax.lax.reshape_p)
                and self._is_neg_inf_atom(invars[0]) and jnp.issubdtype(out_aval.dtype, jnp.floating)):
            self.neg_inf.add(out)
        if prim is jax.lax.convert_element_type_p and isinstance(invars[0], jcore.Var):
            if _lossless_float_cast(invars[0], out):
                self.passthrough(invars[0], out)
            return bound, proved
        if prim is jax.lax.reduce_max_p and isinstance(invars[0], jcore.Var):
            if (jnp.issubdtype(out_aval.dtype, jnp.floating) and apart_of(invars[0]).finite
                    and _reduced(equation) >= 1):
                red = frozenset(int(ax) for ax in equation.params["axes"])
                kept = tuple(ax for ax in range(len(_shape(invars[0]))) if ax not in red)
                self.max_of[out] = (self._canon(invars[0]), kept)
            return bound, proved
        if prim is jax.lax.max_p and len(invars) == 2:
            for i in (0, 1):
                cand, other = invars[i], invars[1 - i]
                if (isinstance(cand, jcore.Var) and cand in self.max_of and self._is_neg_inf_atom(other)
                        and _shape(cand) == out_aval.shape):
                    self.max_of[out] = self.max_of[cand]
            return bound, proved
        if prim in (jax.lax.broadcast_in_dim_p, jax.lax.reshape_p) and isinstance(invars[0], jcore.Var):
            if ((entry := self.max_of.get(invars[0])) is not None
                    and (moved := _move_kept(equation, invars[0], out, entry[1])) is not None):
                self.max_of[out] = (entry[0], moved)
            return bound, proved
        if prim is jax.lax.sub_p and len(invars) == 2:
            left, right = invars
            if (isinstance(left, jcore.Var) and isinstance(right, jcore.Var)
                    and (entry := self.max_of.get(right)) is not None
                    and jnp.issubdtype(out_aval.dtype, jnp.floating) and apart_of(left).finite):
                src, kept = entry
                out_shape = tuple(out_aval.shape)
                if (self._canon(left) is src and _shape(left) == out_shape
                        and _aligned_unit_axes(kept, len(out_shape))):
                    self.nonpos_max[out] = frozenset(ax for ax, k in enumerate(kept) if k is None)
                    refined = _held_apart(
                        Apart(proved.floor, -1, number=True, finite=proved.finite), out_aval.dtype)
                    return (bound if refined.finite else None), refined
            return bound, proved
        if prim in (jax.lax.exp_p, jax.lax.exp2_p) and isinstance(invars[0], jcore.Var):
            if ((red := self.nonpos_max.get(invars[0])) is not None
                    and jnp.issubdtype(out_aval.dtype, jnp.floating)):
                self.unit_exp[out] = red
            return bound, proved
        if (prim is jax.lax.reduce_sum_p and isinstance(invars[0], jcore.Var)
                and (red := self.unit_exp.get(invars[0])) is not None
                and jnp.issubdtype(out_aval.dtype, jnp.floating) and _reduced(equation) >= 1):
            sum_axes = frozenset(int(ax) for ax in equation.params["axes"])
            if red.issubset(sum_axes):
                floor = max(proved.floor, 0) if proved.floor is not None else 0
                refined = _held_apart(
                    Apart(floor, 1, number=True, finite=proved.finite), out_aval.dtype)
                return (bound if refined.finite else None), refined
        return bound, proved

    def child(self, outer_invars: Sequence, inner_invars: Sequence[jcore.Var]) -> _SoftmaxTrack:
        """Map tracked softmax entries into a peeled call body."""
        pairs = [(o, i) for o, i in zip(outer_invars, inner_invars, strict=True) if isinstance(o, jcore.Var)]
        by_origin: dict[jcore.Var, jcore.Var] = {}
        for outer, inner in pairs:
            by_origin.setdefault(self._canon(outer), inner)
        max_child = {i: (by_origin[e[0]], e[1]) for o, i in pairs
                     if (e := self.max_of.get(o)) is not None and e[0] in by_origin}
        nonpos_child = {i: self.nonpos_max[o] for o, i in pairs if o in self.nonpos_max}
        exp_child = {i: self.unit_exp[o] for o, i in pairs if o in self.unit_exp}
        neg_inf_child = {i for o, i in pairs if o in self.neg_inf}
        return _SoftmaxTrack(max_child, nonpos_child, exp_child, neg_inf_child)


def _same_atom(a, b) -> bool:
    """Literal-safe identity/value match between two Jaxpr atoms (JAX 0.12 `Literal` is unhashable)."""
    if a is b:
        return True
    if isinstance(a, jcore.Literal) and isinstance(b, jcore.Literal):
        ha, hb = _host_array(a.val), _host_array(b.val)
        return ha is not None and hb is not None and ha.dtype == hb.dtype and bool(np.array_equal(ha, hb))
    return False


def _nan_guard_apart(equation: jcore.JaxprEqn, made_eq: Mapping[jcore.Var, jcore.JaxprEqn],
                     magnitude_of: Callable[..., Magnitude | None],
                     apart_of: Callable[..., Apart]) -> Apart | None:
    """Recognize JAX's `logaddexp(u, v)` NaN-propagation guard `select_n(ne(u - v, u - v), safe, u + v)`:
    when one operand of `sub(u, v)` is statically bounded and finite, `u - v` is NaN only if the other is
    NaN, so every non-NaN value takes `safe` (`false_val`)."""
    if equation.primitive is not jax.lax.select_n_p or len(equation.invars) != 3:
        return None
    pred, false_val, true_val = equation.invars
    pred_eq = made_eq.get(pred) if isinstance(pred, jcore.Var) else None
    if pred_eq is None or pred_eq.primitive is not jax.lax.ne_p or pred_eq.invars[0] is not pred_eq.invars[1]:
        return None
    diff = pred_eq.invars[0]
    diff_eq = made_eq.get(diff) if isinstance(diff, jcore.Var) else None
    if diff_eq is None or diff_eq.primitive is not jax.lax.sub_p:
        return None
    left, right = diff_eq.invars
    if not ((apart_of(left).finite and magnitude_of(left) is not None)
            or (apart_of(right).finite and magnitude_of(right) is not None)):
        return None
    if _same_atom(true_val, left) or _same_atom(true_val, right):
        return apart_of(false_val)
    true_eq = made_eq.get(true_val) if isinstance(true_val, jcore.Var) else None
    if true_eq is not None and true_eq.primitive is jax.lax.add_p and len(true_eq.invars) == 2:
        a0, a1 = true_eq.invars
        if ((_same_atom(a0, left) and _same_atom(a1, right))
                or (_same_atom(a0, right) and _same_atom(a1, left))):
            return apart_of(false_val)
    return None


def _concat_facts(operands: Sequence[Apart], bounds: Sequence[Magnitude | None], dtype,
                  *, keep_unit: bool = False) -> tuple[Magnitude | None, Apart]:
    """Intersect sign, floor, finiteness, unit-norm, and magnitude bounds across `concatenate_p` operands."""
    if not operands:
        return None, NOWHERE
    sign = _shared_sign(operands, bounds)
    floors = [h.floor for h in operands if h.floor is not None]
    floor = min(floors) if len(floors) == len(operands) else None
    known_bounds = [b for b in bounds if b is not None]
    bound = max(known_bounds) if len(known_bounds) == len(bounds) else None
    finite = all(h.finite for h in operands) and (
        not jnp.issubdtype(dtype, jnp.floating) or _within(bound, dtype))
    unit = keep_unit and all(h.unit_norm for h in operands)
    proved = _held_apart(Apart(floor, sign, all(h.number for h in operands), finite, unit_norm=unit), dtype)
    return (bound if proved.finite else None), proved


def _deep_refine(equation: jcore.JaxprEqn, bound: Magnitude | None, proved: Apart,
                 made_eq: Mapping[jcore.Var, jcore.JaxprEqn],
                 magnitude_of: Callable[..., Magnitude | None], apart_of: Callable[..., Apart],
                 norm_track: _NormTrack) -> tuple[Magnitude | None, Apart]:
    """Refine an in-language equation's `(bound, apart)` with unit-norm, NaN-guard, and `log(logistic)`."""
    out = equation.outvars[0]
    if (unit := norm_track.step(equation, apart_of)) is not None:
        bound, proved = unit
    if (equation.primitive is jax.lax.convert_element_type_p and proved.finite
            and apart_of(equation.invars[0]).unit_norm
            and isinstance(out.aval, ShapedArray) and jnp.issubdtype(out.aval.dtype, jnp.floating)
            and jnp.dtype(out.aval.dtype).itemsize >= 2):
        proved = proved._replace(unit_norm=True)
    if equation.primitive is jax.lax.select_n_p:
        guarded = _nan_guard_apart(equation, made_eq, magnitude_of, apart_of)
        if guarded is not None:
            proved = _held_apart(guarded, out.aval.dtype) if isinstance(out.aval, ShapedArray) else guarded
    if (equation.primitive in (jax.lax.log_p, jax.lax.log2_p)
            and isinstance(equation.invars[0], jcore.Var)
            and (src_eq := made_eq.get(equation.invars[0])) is not None
            and src_eq.primitive is jax.lax.logistic_p):
        proved = proved._replace(sign=-1)
    return bound, proved


def _deep_nonlang(equation: jcore.JaxprEqn, held: dict[jcore.Var, Magnitude | None],
                  apart: dict[jcore.Var, Apart], magnitude_of: Callable[..., Magnitude | None],
                  apart_of: Callable[..., Apart], norm_track: _NormTrack, softmax_track: _SoftmaxTrack,
                  depth: int, walk_fn: Callable) -> bool:
    """Handle slice/stop_gradient, `concatenate_p`, and peeled called bodies for `proven(..., deep=True)`."""
    prim = equation.primitive
    if (prim in (jax.lax.slice_p, jax.lax.dynamic_slice_p, jax.lax.stop_gradient_p)
            and len(equation.outvars) == 1):
        out = equation.outvars[0]
        held[out], apart[out] = magnitude_of(equation.invars[0]), apart_of(equation.invars[0])
        if prim is jax.lax.stop_gradient_p and isinstance(equation.invars[0], jcore.Var):
            softmax_track.passthrough(equation.invars[0], out)
        return True
    if (prim is jax.lax.concatenate_p and len(equation.outvars) == 1
            and isinstance(out_aval := equation.outvars[0].aval, ShapedArray)):
        out = equation.outvars[0]
        keep_unit = out_aval.ndim > 0 and equation.params.get("dimension") != out_aval.ndim - 1
        held[out], apart[out] = _concat_facts(
            [apart_of(atom) for atom in equation.invars],
            [magnitude_of(atom) for atom in equation.invars], out_aval.dtype, keep_unit=keep_unit)
        return True
    if (depth < 6 and (body := programs.called(equation)) is not None
            and len(body.consts) == len(body.constvars)):
        sub_init = {inner: (magnitude_of(outer), apart_of(outer))
                    for outer, inner in zip(equation.invars, body.invars, strict=True)}
        sub_consts = dict(zip(body.constvars, body.consts, strict=True))
        sub_targets = [atom for atom in body.outvars if isinstance(atom, jcore.Var)]
        sub_held, sub_apart = walk_fn(
            body.jaxpr, consts=sub_consts, targets=sub_targets, deep=True,
            init=sub_init, track=norm_track.child(equation.invars, body.invars),
            softmax=softmax_track.child(equation.invars, body.invars), depth=depth + 1)
        for out_var, body_out in zip(equation.outvars, body.outvars, strict=True):
            if isinstance(body_out, jcore.Literal):
                held[out_var], apart[out_var] = _magnitude(body_out.val), _apart_of(body_out, {})
            else:
                held[out_var] = sub_held.get(body_out)
                apart[out_var] = sub_apart.get(body_out, NOWHERE)
        return True
    return False
