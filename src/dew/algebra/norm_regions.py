"""The fused norms at the compile point (row 3450, theme K40): a plain program's RMSNorm and SwiGLU, as Flax
and plain JAX spell them, found in a body (`regions`) and written as Dew's (`dew.nn.norms.rms_normalized`,
`dew.nn.activations.gated_product`, each at implementation 'auto'), so the kernel is the one their selector
gives the call (`dew.nn.kernels.norm_swiglu`'s `NORM` and `SWIGLU`) where that is not XLA's (`taken`).

An RMSNorm is a row's root mean square, its inverse times the row and an optional weight, cast once:

    m = sum(x * x, -1) / D       its statistics in float32 at least (`x` widened before it), the square a
                                 `square`, `integer_pow` of 2 or a product of the row with itself
    r = rsqrt(m + eps)           eps a constant
    y = x * r                    the inverse broadcast along the row
    y = y * w   or   (1 + w)     a weight of the row's width (`scale_offset`), before the cast or after it
                                 (`scale_after_cast`), or none
    y = cast(y)

A SwiGLU is `silu(a) * b`, silu `a * sigmoid(a)`. Each value between the region's inputs and its output is
read by the region alone. The call computes as Dew's norms do, on par with the program's own (check 1), not
bitwise.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import numpy as np
from jax.extend import core as jcore

from dew.algebra.attention_regions import Graph, widening
from dew.nn.programs import jax_name


class Norm(NamedTuple):
    """An RMSNorm region (the module's account): its equations' places, its row and weight as the program
    holds them (the weight None where there is none), its output, and Dew's norm's arguments."""

    places: tuple[int, ...]
    row: jcore.Var
    weight: jcore.Var | None
    output: jcore.Var
    epsilon: float
    offset: bool
    after_cast: bool

    @property
    def operands(self) -> tuple[jcore.Var, ...]:
        return (self.row,) if self.weight is None else (self.row, self.weight)


class Gated(NamedTuple):
    """A SwiGLU region: its equations' places, its gate and up projections, its output."""

    places: tuple[int, ...]
    gate: jcore.Var
    up: jcore.Var
    output: jcore.Var

    @property
    def operands(self) -> tuple[jcore.Var, ...]:
        return self.gate, self.up


def regions(index) -> list[Norm | Gated]:
    """Every RMSNorm and SwiGLU region of the body `index` reads (`fusion.Index`), the first found where two
    would share an equation."""
    graph = Graph(index)
    found, claimed = [], set()
    for place, equation in enumerate(index.jaxpr.eqns):
        name = jax_name(equation.primitive)
        gated = name == "logistic" or (name in ("jit", "pjit") and equation.params.get("name") == "silu")
        region = _rms(graph, place) if name == "rsqrt" else _gated(graph, place) if gated else None
        if region is not None and not claimed & set(region.places):
            found.append(region)
            claimed.update(region.places)
    return found


def written(region: Norm | Gated) -> Callable:
    """The region's call: Dew's RMSNorm or SwiGLU over its operands, at implementation 'auto'."""
    from dew.nn.activations import gated_product
    from dew.nn.norms import rms_normalized

    dtype = region.output.aval.dtype
    if isinstance(region, Gated):
        return lambda gate, up: gated_product("swiglu")(gate, up).astype(dtype)
    return lambda row, weight=None: rms_normalized(row, weight, region.epsilon, dtype, region.offset,
                                                   region.after_cast, fp32_statistics=True)


def taken(region: Norm | Gated) -> bool:
    """Whether the norms' selector gives the region's call Dew's fused kernel rather than XLA's arithmetic."""
    from dew.nn.kernels import norm_swiglu

    shapes = [jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype) for var in region.operands]
    if isinstance(region, Gated):
        return norm_swiglu.swiglu_kernel("auto", *shapes) == "pallas"
    weight = shapes[1] if len(shapes) > 1 else None
    return norm_swiglu.rms_kernel("auto", shapes[0], weight, region.output.aval.dtype,
                                  fp32_statistics=True) == "pallas"


def _rms(graph: Graph, place: int) -> Norm | None:
    """The RMSNorm around the inverse square root at `place`, None where the body holds none there."""
    eqns, places = graph.jaxpr.eqns, [place]
    added = graph.made(graph.uncast(eqns[place].invars[0], places, widening), "add")
    if added is None:
        return None
    places.append(added)
    epsilon, mean = _constant_beside(graph, eqns[added].invars)
    if epsilon is None:
        return None
    # The mean, its kept axis laid before or after its division (`jnp.mean`'s keepdims after the sum).
    mean = _unkept(graph, graph.uncast(mean, places, widening), places)
    scaled = graph.made(mean, "div")
    if scaled is not None:
        summed, width = eqns[scaled].invars
        width = graph.constant(width)
    else:
        scaled = graph.made(mean, "mul")
        if scaled is None:
            return None
        inverse_width, summed = _constant_beside(graph, eqns[scaled].invars)
        width = None if inverse_width is None else 1 / inverse_width
    if summed is None:
        return None
    reduced = graph.made(_unkept(graph, graph.uncast(summed, places, widening), places), "reduce_sum")
    if width is None or np.ndim(width) != 0 or reduced is None:
        return None
    places += [scaled, reduced]
    squared = eqns[reduced].invars[0]
    row = _squared(graph, squared, places)
    if (row is None or tuple(eqns[reduced].params["axes"]) != (row.aval.ndim - 1,)
            or not np.isclose(float(width), row.aval.shape[-1]) or np.dtype(row.aval.dtype).itemsize < 4):
        return None
    # The inverse broadcast along the row and multiplied into it.
    normed = _times_row(graph, eqns[place].outvars[0], row, places)
    if normed is None:
        return None
    weight, offset, after_cast, output = _weighted(graph, normed, places)
    source = graph.uncast(row, places, widening)
    places = sorted(set(places))
    if not _closed(graph, places, output):
        return None
    return Norm(tuple(places), source, weight, output, float(epsilon), offset, after_cast)


def _constant_beside(graph: Graph, operands) -> tuple[float | None, object]:
    """`(c, other)` where one of a binary equation's `operands` is a constant scalar `c`; `(None, None)`
    else."""
    for constant, other in (operands, operands[::-1]):
        value = graph.constant(constant)
        if value is not None and np.ndim(value) == 0 and graph.constant(other) is None:
            return float(value), other
    return None, None


def _unkept(graph: Graph, atom, places: list[int]):
    """`atom` past the layout a reduction's kept axis takes (`keepdims`: a broadcast or reshape adding a unit
    axis), casts between."""
    while (place := graph.made(atom, "broadcast_in_dim", "reshape")) is not None:
        places.append(place)
        atom = graph.uncast(graph.jaxpr.eqns[place].invars[0], places, widening)
    return atom


def _squared(graph: Graph, atom, places: list[int]):
    """The row `atom` squares (`square`, `integer_pow` of 2, or its product with itself), None where it is
    none."""
    eqns = graph.jaxpr.eqns
    place = graph.made(atom, "square", "integer_pow", "mul")
    if place is None:
        return None
    eqn = eqns[place]
    name = jax_name(eqn.primitive)
    if eqn.invars[0] is not eqn.invars[-1] or (name == "integer_pow" and eqn.params["y"] != 2):
        return None
    places.append(place)
    return eqn.invars[0]


def _times_row(graph: Graph, inverse, row, places: list[int]):
    """The product of `row` and `inverse` laid along it (a broadcast, casts between), its value; None where
    the inverse's one reader chain reaches none."""
    eqns, atom = graph.jaxpr.eqns, inverse
    while True:
        readers = graph.readers[atom]
        if len(readers) != 1 or atom in graph.escaping:
            return None
        (reader,) = readers
        name = jax_name(eqns[reader].primitive)
        if name == "mul" and row in eqns[reader].invars:
            places.append(reader)
            return eqns[reader].outvars[0]
        if name not in ("broadcast_in_dim", "reshape", "convert_element_type"):
            return None
        places.append(reader)
        atom = eqns[reader].outvars[0]


def _weighted(graph: Graph, normed, places: list[int]) -> tuple[jcore.Var | None, bool, bool, jcore.Var]:
    """The weight the normalized row takes and how (`(weight, offset, after_cast, output)`): a product with a
    weight broadcast along the row (one more than it where `offset`), before the row's last cast or after it,
    or none; the region's output its last value."""
    eqns, atom, cast = graph.jaxpr.eqns, normed, False
    weight = offset = None
    while len(graph.readers[atom]) == 1 and atom not in graph.escaping:
        (reader,) = graph.readers[atom]
        eqn = eqns[reader]
        name = jax_name(eqn.primitive)
        if name == "convert_element_type" and not cast:
            cast = True
        elif name == "mul" and weight is None:
            other = next((var for var in eqn.invars if var is not atom), None)
            found = None if other is None else _row_weight(graph, other, atom.aval.shape[-1], places)
            if found is None:
                break
            weight, offset = found
            weight_after_cast = cast
        else:
            break
        places.append(reader)
        atom = eqn.outvars[0]
    if weight is None:
        return None, False, False, atom
    return weight, bool(offset), weight_after_cast, atom


def _row_weight(graph: Graph, atom, width: int, places: list[int]) -> tuple[jcore.Var, bool] | None:
    """`(w, offset)` where `atom` is a weight `w` of the row's `width` broadcast along it (`1 + w` where
    `offset`), casts between; None else."""
    eqns, found = graph.jaxpr.eqns, []
    atom = graph.uncast(atom, found, widening)
    if graph.made(atom, "broadcast_in_dim") is None:
        return None
    # Laid along the row: broadcast, reshaped to the row's rank first (Flax's `feature_shape`).
    while (laid := graph.made(atom, "broadcast_in_dim", "reshape")) is not None:
        found.append(laid)
        atom = graph.uncast(eqns[laid].invars[0], found, widening)
    offset = False
    if (added := graph.made(atom, "add")) is not None:
        one, atom = _constant_beside(graph, eqns[added].invars)
        if one != 1.0:
            return None
        found.append(added)
        offset = True
        atom = graph.uncast(atom, found, widening)
    if not isinstance(atom, jcore.Var) or atom.aval.shape != (width,):
        return None
    places += found
    return atom, offset


def _gated(graph: Graph, place: int) -> Gated | None:
    """The SwiGLU around the sigmoid at `place` (`silu(a) * b`, silu `a * sigmoid(a)`, or `jax.nn.silu`'s
    nested program of it), None where the body holds none there."""
    eqns, places = graph.jaxpr.eqns, [place]
    gate = eqns[place].invars[0]
    activated = eqns[place].outvars[0]
    if jax_name(eqns[place].primitive) == "logistic":
        if len(graph.readers[activated]) != 1:
            return None
        (silu,) = graph.readers[activated]
        if jax_name(eqns[silu].primitive) != "mul" or gate not in eqns[silu].invars:
            return None
        places.append(silu)
        activated = eqns[silu].outvars[0]
    if len(graph.readers[activated]) != 1:
        return None
    (product,) = graph.readers[activated]
    if jax_name(eqns[product].primitive) != "mul":
        return None
    up = next((var for var in eqns[product].invars if var is not activated), None)
    if not isinstance(up, jcore.Var) or up.aval.shape != gate.aval.shape or up.aval.dtype != gate.aval.dtype:
        return None
    places.append(product)
    output = eqns[product].outvars[0]
    places = sorted(places)
    if not _closed(graph, places, output):
        return None
    return Gated(tuple(places), gate, up, output)


def _closed(graph: Graph, places: list[int], output) -> bool:
    """Whether every value the region's equations make but its `output` is read inside it alone."""
    inside = set(places)
    return all(var is output or (var not in graph.escaping and graph.readers[var] <= inside)
               for place in places for var in graph.jaxpr.eqns[place].outvars)
