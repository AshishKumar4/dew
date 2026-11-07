"""The fused norms at the compile point (row 3450, theme K40): a plain program's RMSNorm and SwiGLU, as Flax
and plain JAX spell them, found in a body (`regions`) and written as Dew's (`dew.nn.norms.rms_normalized`,
`dew.nn.activations.gated_product`, each at implementation 'auto'), so the kernel is the one their selector
gives the call (`dew.nn.kernels.norm_swiglu`'s `NORM` and `SWIGLU`) where that is not XLA's (`taken`).

An RMSNorm is a row's root mean square, its inverse times the row and an optional weight, cast once:

    m = sum(x * x, -1) / D       its statistics in float32 at least (`x` widened before it), the square a
                                 `square`, `integer_pow` of 2 or a product of the row with itself
    r = rsqrt(m + eps)           eps a constant
    y = x * r                    the inverse broadcast along the row, or the inverse times the weight first
                                 (Flax's `mul *= scale`)
    y = y * w   or   (1 + w)     a weight of the row's width (`scale_offset`), before the cast or after it
                                 (`scale_after_cast`), or none
    y = cast(y)

Its row may be a residual stream's sum `x + branch`, which the program reads on (a pre-norm block's seam,
`dew.nn.norms.stream_rms_normalized`, its sum one of the region's outputs), and its weighted row may be
gated, `y * act(g)` before the cast with `act` silu or sigmoid (Qwen3-Next's,
`dew.nn.norms.gated_rms_normalized`). A SwiGLU is `silu(a) * b`, silu `a * sigmoid(a)`, its two halves read
from one packed projection's `[..., 2, H]` where they are its two slices along that axis
(`dew.nn.activations.packed_gated_product`). Each value between the region's inputs and its outputs is read
by the region alone. The call computes as Dew's norms do, on par with the program's own (check 1), not
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
    holds them (the weight None where there is none), its output, and Dew's norm's arguments; a stream's
    branch and the sum the program reads on (`stream`), and a gate and its activation, where it has them."""

    places: tuple[int, ...]
    row: jcore.Var
    weight: jcore.Var | None
    output: jcore.Var
    epsilon: float
    offset: bool
    after_cast: bool
    branch: jcore.Var | None = None
    stream: jcore.Var | None = None
    gate: jcore.Var | None = None
    activation: str | None = None

    @property
    def operands(self) -> tuple[jcore.Var, ...]:
        """The row, then the branch, weight and gate it has, in that order (`written`'s)."""
        return tuple(var for var in (self.row, self.branch, self.weight, self.gate) if var is not None)

    @property
    def outputs(self) -> tuple[jcore.Var, ...]:
        """The stream's sum the program reads on, where it does, then the normalized row."""
        return (self.output,) if self.stream is None else (self.stream, self.output)


class Gated(NamedTuple):
    """A SwiGLU region: its equations' places, its gate and up projections (or the one packed projection
    `[..., 2, H]` both are slices of, `packed`), its output."""

    places: tuple[int, ...]
    gate: jcore.Var
    up: jcore.Var
    output: jcore.Var
    packed: jcore.Var | None = None

    @property
    def operands(self) -> tuple[jcore.Var, ...]:
        return (self.packed,) if self.packed is not None else (self.gate, self.up)

    @property
    def outputs(self) -> tuple[jcore.Var, ...]:
        return (self.output,)


def regions(index) -> list[Norm | Gated]:
    """Every RMSNorm and SwiGLU region of the body `index` reads (`fusion.Index`): the norms first, so a gated
    norm's activation is its own rather than a SwiGLU's, the first found where two would share an equation."""
    graph = Graph(index)
    found, claimed = [], set()
    for norms in (True, False):
        for place, equation in enumerate(index.jaxpr.eqns):
            name = jax_name(equation.primitive)
            if norms:
                region = _rms(graph, place) if name == "rsqrt" else None
            else:
                region = _gated(graph, place) if name == "logistic" or _silu(equation) else None
            if region is not None and not claimed & set(region.places):
                found.append(region)
                claimed.update(region.places)
    return found


def written(region: Norm | Gated) -> Callable:
    """The region's call over its operands (`operands`), its outputs (`outputs`) its results: Dew's RMSNorm,
    stream norm, gated norm, SwiGLU or packed SwiGLU, at implementation 'auto'."""
    from dew.nn.activations import gated_product, packed_gated_product
    from dew.nn.norms import gated_rms_normalized, stream_rms_normalized

    dtype = region.output.aval.dtype
    if isinstance(region, Gated):
        if region.packed is not None:
            return lambda packed: packed_gated_product("swiglu")(packed).astype(dtype)
        return lambda gate, up: gated_product("swiglu")(gate, up).astype(dtype)

    def call(*operands):
        given = iter(operands)
        row = next(given)
        branch = next(given) if region.branch is not None else None
        weight = next(given) if region.weight is not None else None
        if region.gate is not None:
            return gated_rms_normalized(row, next(given), weight, region.epsilon, dtype, region.activation)
        stream, normed = stream_rms_normalized(row, branch, weight, region.epsilon, dtype, region.offset,
                                               region.after_cast, fp32_statistics=True)
        return normed if region.stream is None else (stream, normed)

    return call


def taken(region: Norm | Gated) -> bool:
    """Whether the norms' selector gives the region's call Dew's fused kernel rather than XLA's arithmetic."""
    from dew.nn.kernels import norm_swiglu

    def shape(var):
        return None if var is None else jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype)

    dtype = region.output.aval.dtype
    if isinstance(region, Gated):
        if region.packed is not None:
            return norm_swiglu.packed_swiglu_kernel("auto", shape(region.packed)) == "pallas"
        return norm_swiglu.swiglu_kernel("auto", shape(region.gate), shape(region.up)) == "pallas"
    if region.gate is not None:
        return norm_swiglu.gated_rms_kernel("auto", shape(region.row), shape(region.gate),
                                            shape(region.weight), dtype, region.activation) == "pallas"
    return norm_swiglu.rms_kernel("auto", shape(region.row), shape(region.weight), dtype,
                                  fp32_statistics=True, branch=shape(region.branch)) == "pallas"


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
        width = None if inverse_width is None or inverse_width <= 0 else 1 / inverse_width
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
    # The statistics laid back along the row and multiplied into it, a weight beside them where Flax's is.
    along = _along_row(graph, reduced, place, row, places)
    if along is None:
        return None
    normed, first = along
    weight, offset, after_cast, gate, activation, output = _weighted(graph, normed, places, first)
    if gate is not None and (offset or after_cast):
        return None  # Qwen3-Next's gated norm weighs before its one cast, by the weight itself
    source = graph.uncast(row, places, widening)
    # A residual stream's sum: its addends the call's, the sum one of its outputs where the program reads it
    # on, after the call's place.
    branch = stream = None
    summed = graph.made(source, "add")
    if summed is not None and gate is None:
        addends = graph.jaxpr.eqns[summed].invars
        others = graph.readers[source] - set(places)
        if (all(isinstance(var, jcore.Var) and var.aval.shape == source.aval.shape
                and var.aval.dtype == source.aval.dtype for var in addends)
                and all(reader > max(places) for reader in others)):
            places.append(summed)
            (source, branch), stream = addends, (source if others or source in graph.escaping else None)
    places = sorted(set(places))
    region = Norm(tuple(places), source, weight, output, float(epsilon), offset, after_cast, branch, stream,
                  gate, activation)
    if not _closed(graph, places, region.outputs, region.operands):
        return None
    return region


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


def _along_row(graph: Graph, reduced: int, inverse: int, row, places: list[int]):
    """`(product, weight)`: the product of `row` and its statistics laid back along it, and the weight
    multiplied into the statistics before it (Flax's, `(w, offset)`; None where none is), the row reduction
    at `reduced` followed forward through the statistics' equations (`places`, the inverse square root at
    `inverse` among them) and then layouts, casts and that one weight alone (added to `places`), each value
    read once, its axes tracked (`_relabeled`) so each of the row's leading axes lies where it did and its
    feature axis is broadcast; None where they do not (another row's statistics, or the statistics laid
    along the features)."""
    eqns, known = graph.jaxpr.eqns, set(places)
    atom, labels, after = eqns[reduced].outvars[0], list(range(row.aval.ndim - 1)), False
    weight = None
    while True:
        readers = graph.readers[atom]
        if len(readers) != 1 or atom in graph.escaping:
            return None
        (reader,) = readers
        eqn = eqns[reader]
        name = jax_name(eqn.primitive)
        other = next((var for var in eqn.invars if var is not atom), None)
        # The row itself, or the same input widened again for the product less a mean of zero (Flax's
        # statistics and product each widen it, its RMSNorm's `x - mean` at no mean).
        cast: list[int] = []
        if (after and name == "mul" and isinstance(other, jcore.Var) and other.aval.shape == row.aval.shape
                and graph.uncast(_unshifted(graph, other, cast), cast, widening)
                is graph.uncast(row, [], widening)):
            rank = row.aval.ndim
            if len(labels) != rank or labels[-1] is not None or not all(
                    labels[a] in (a, None) and (labels[a] == a or row.aval.shape[a] == 1)
                    for a in range(rank - 1)):
                return None
            places += [*cast, reader]
            return eqn.outvars[0], weight
        if after and name == "mul" and weight is None:
            weight = None if other is None else _row_weight(graph, other, row.aval.shape[-1], places)
            if weight is None or eqn.outvars[0].aval.shape != row.aval.shape:
                return None
            # The weight lies along the features alone (`_row_weight`), the statistics along the rest.
            places.append(reader)
            atom = eqn.outvars[0]
            continue
        if (reader not in known if not after else name not in LAYOUTS):
            return None
        labels = _relabeled(graph, eqn, atom, labels)
        if labels is None:
            return None
        if after:
            places.append(reader)
        after = after or reader == inverse
        atom = eqn.outvars[0]


def _unshifted(graph: Graph, atom, places: list[int]):
    """`atom` past a subtraction of zeros (a constant every entry of which is 0), its place added to
    `places`; `atom` itself where it is none."""
    place = graph.made(atom, "sub")
    if place is None:
        return atom
    minuend, subtrahend = graph.jaxpr.eqns[place].invars
    zeros = graph.constant(subtrahend)
    if zeros is None or np.any(zeros != 0):
        return atom
    places.append(place)
    return minuend


LAYOUTS = frozenset({"broadcast_in_dim", "reshape", "convert_element_type"})
"""The equations that lay a value out again or cast it, each entry the same number."""


def _relabeled(graph: Graph, eqn: jcore.JaxprEqn, atom, labels: list) -> list | None:
    """`labels` (for each of `atom`'s axes, the axis of the row or weight it is, None for a broadcast or unit
    axis) for `eqn`'s output: through a broadcast or a reshape that adds or drops unit axes, as laid; through
    a cast, or an entrywise equation whose other operands are constant scalars, as they are; None through
    any other."""
    name = jax_name(eqn.primitive)
    shape, out = atom.aval.shape, eqn.outvars[0].aval.shape
    if name == "broadcast_in_dim":
        laid: list = [None] * len(out)
        for source, target in enumerate(eqn.params["broadcast_dimensions"]):
            laid[int(target)] = labels[source] if shape[source] == out[int(target)] else None
        return laid
    if name == "reshape":
        kept = [a for a in range(len(shape)) if shape[a] != 1]
        made = [a for a in range(len(out)) if out[a] != 1]
        if [shape[a] for a in kept] != [out[a] for a in made]:
            return None
        laid = [None] * len(out)
        for source, target in zip(kept, made, strict=True):
            laid[target] = labels[source]
        return laid
    others = [var for var in eqn.invars if var is not atom]
    if name == "convert_element_type" or all(
            graph.constant(var) is not None and np.ndim(graph.constant(var)) == 0 for var in others):
        return labels
    return None


def _weighted(graph: Graph, normed, places: list[int], first=None):
    """The weight and gate the normalized row takes and how (`(weight, offset, after_cast, gate, activation,
    output)`): a product with a weight broadcast along the row (one more than it where `offset`), before the
    row's last cast or after it, or none, or `first`, the one the statistics took before the row
    (`_along_row`); a product with an activation of a gate of the row's shape before the cast (`_activated`),
    or none; the region's output its last value."""
    eqns, atom, cast = graph.jaxpr.eqns, normed, False
    weight, offset = first if first is not None else (None, None)
    weight_after_cast = False
    gate = activation = None
    while len(graph.readers[atom]) == 1 and atom not in graph.escaping:
        (reader,) = graph.readers[atom]
        eqn = eqns[reader]
        name = jax_name(eqn.primitive)
        if name == "convert_element_type" and not cast:
            cast = True
        elif name == "mul":
            other = next((var for var in eqn.invars if var is not atom), None)
            weighed = (None if other is None or weight is not None
                       else _row_weight(graph, other, atom.aval.shape[-1], places))
            activated = (None if other is None or gate is not None or cast
                         else _activated(graph, other, atom.aval.shape, places))
            if weighed is not None:
                weight, offset = weighed
                weight_after_cast = cast
            elif activated is not None:
                gate, activation = activated
            else:
                break
        else:
            break
        places.append(reader)
        atom = eqn.outvars[0]
    return weight, bool(offset), weight is not None and weight_after_cast, gate, activation, atom


def _activated(graph: Graph, atom, shape, places: list[int]) -> tuple[jcore.Var, str] | None:
    """`(g, activation)` where `atom` is silu or sigmoid of a gate `g` of the row's `shape` (a logistic, its
    product with its input, or `jax.nn`'s nested programs of either), casts between, its places added to
    `places`; None else."""
    eqns, found = graph.jaxpr.eqns, []
    atom = graph.uncast(atom, found, widening)
    place = graph.made(atom, "mul", "logistic", "jit", "pjit")
    if place is None:
        return None
    eqn = eqns[place]
    name = jax_name(eqn.primitive)
    if name == "mul":
        logistic = next((graph.made(var, "logistic", "jit", "pjit") for var in eqn.invars
                         if graph.made(var, "logistic", "jit", "pjit") is not None), None)
        if logistic is None or not _logistic(eqns[logistic]):
            return None
        argument = eqns[logistic].invars[0]
        if argument not in eqn.invars or len(graph.readers[eqns[logistic].outvars[0]]) != 1:
            return None
        found += [logistic, place]
        activation = "silu"
    elif _silu(eqn):
        argument, activation = eqn.invars[0], "silu"
        found.append(place)
    elif _logistic(eqn):
        argument, activation = eqn.invars[0], "sigmoid"
        found.append(place)
    else:
        return None
    gate = graph.uncast(argument, found, widening)
    if not isinstance(gate, jcore.Var) or gate.aval.shape != tuple(shape):
        return None
    places += found
    return gate, activation


def _row_weight(graph: Graph, atom, width: int, places: list[int]) -> tuple[jcore.Var, bool] | None:
    """`(w, offset)` where `atom` is a weight `w` of the row's `width` laid along its feature axis alone (`1 +
    w` where `offset`), through casts, reshapes and broadcasts (Flax's `feature_shape`); None else."""
    eqns, chain = graph.jaxpr.eqns, []
    offset = False
    while True:
        place = graph.made(atom, *LAYOUTS, "add")
        if place is None:
            break
        if jax_name(eqns[place].primitive) == "add":
            one, inner = _constant_beside(graph, eqns[place].invars)
            if one != 1.0 or offset:
                return None
            offset = True
        else:
            inner = eqns[place].invars[0]
        chain.append(place)
        atom = inner
    if not isinstance(atom, jcore.Var) or atom.aval.shape != (width,) or graph.constant(atom) is not None:
        return None
    # Its one axis followed forward to where the product reads it: the row's feature axis alone.
    labels, value = [0], atom
    for place in reversed(chain):
        labels = _relabeled(graph, eqns[place], value, labels)
        if labels is None:
            return None
        value = eqns[place].outvars[0]
    if labels[-1] != 0 or any(label is not None for label in labels[:-1]):
        return None
    places += chain
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
    packed = _halves(graph, gate, up, places)
    places = sorted(places)
    region = Gated(tuple(places), gate, up, output, packed)
    if not _closed(graph, places, region.outputs, region.operands):
        return None
    return region


def _halves(graph: Graph, gate, up, places: list[int]) -> jcore.Var | None:
    """The packed projection `[..., 2, H]` whose two slices along its pair axis (the second last) are `gate`
    and `up`, first and second, each read by the region alone (a static slice, or `jnp.split`'s, its unit
    axis squeezed or reshaped away); its slicing's places added to `places`. None where they are none."""
    eqns, found, sources = graph.jaxpr.eqns, [], []
    for half, index in ((gate, 0), (up, 1)):
        dropped = graph.made(half, "squeeze", "reshape")
        if dropped is None or len(graph.readers[half]) != 1:
            return None
        sliced = eqns[dropped].invars[0]
        place = graph.made(sliced, "slice", "split")
        if place is None or len(graph.readers[sliced]) != 1:
            return None
        eqn = eqns[place]
        source = eqn.invars[0]
        shape = source.aval.shape
        if len(shape) < 2 or shape[-2] != 2 or sliced.aval.shape != (*shape[:-2], 1, shape[-1]):
            return None
        if jax_name(eqn.primitive) == "slice":
            start = eqn.params["start_indices"]
            if start[-2] != index or any(start[a] for a in range(len(shape)) if a != len(shape) - 2):
                return None
        elif eqn.params["axis"] % len(shape) != len(shape) - 2 or eqn.outvars.index(sliced) != index:
            return None
        found += [dropped, place]
        sources.append(source)
    if sources[0] is not sources[1] or not isinstance(sources[0], jcore.Var):
        return None
    places += sorted(set(found))
    return sources[0]


def _closed(graph: Graph, places: list[int], outputs, operands) -> bool:
    """Whether the region's `operands` are made outside it and every value its equations make but its
    `outputs` is read inside it alone, so the splice removes no value another equation reads."""
    inside = set(places)
    return (all(graph.producer.get(var) not in inside for var in operands)
            and all(var in outputs or (var not in graph.escaping and graph.readers[var] <= inside)
                    for place in places for var in graph.jaxpr.eqns[place].outvars))


def _silu(eqn: jcore.JaxprEqn) -> bool:
    """Whether `eqn` is a nested program computing `x * logistic(x)` of its one input (`jax.nn.silu`'s), the
    logistic itself or a nested program of it alone: its body read, not its name."""
    body = _body(eqn)
    if body is None or len(body.invars) != 1 or len(body.eqns) != 2 or len(body.outvars) != 1:
        return False
    (x,), (first, second) = body.invars, body.eqns
    return (_logistic(first) and first.invars == [x] and jax_name(second.primitive) == "mul"
            and set(second.invars) == {x, first.outvars[0]} and second.outvars == body.outvars)


def _logistic(eqn: jcore.JaxprEqn) -> bool:
    """Whether `eqn` is a logistic, or a nested program of one of its input alone (`jax.nn.sigmoid`'s)."""
    if jax_name(eqn.primitive) == "logistic":
        return True
    body = _body(eqn)
    return (body is not None and len(body.eqns) == 1 and jax_name(body.eqns[0].primitive) == "logistic"
            and body.eqns[0].invars == body.invars and body.eqns[0].outvars == body.outvars)


def _body(eqn: jcore.JaxprEqn) -> jcore.Jaxpr | None:
    """A nested program's body with no constants (a `jit`'s), None for any other equation."""
    if jax_name(eqn.primitive) not in ("jit", "pjit") or eqn.params["jaxpr"].consts:
        return None
    return eqn.params["jaxpr"].jaxpr
