"""Attention at the compile point (row 3450, theme K40): a plain program's softmax attention, as
`jax.nn.dot_product_attention`'s XLA form and Flax's `dot_product_attention` spell it, found in a body
(`regions`) and written as Dew's attention (`dew.nn.attention.scaled_dot_product_attention`, its
implementation 'auto'), so its kernel is the one Dew's selector gives the call (the template, FlashAttention,
cuDNN) where that is not XLA's (`taken`).

A region is a logits product, its scale, an optional causal mask, a safe softmax over the keys and the
weights' product with the values, each value between them read by the region alone:

    L = q · k            a `dot_general` of one contracted axis, the head width D
    s = L * D^-0.5       or q divided by sqrt(D) before it (Flax's), the default scale only
    x = where(m, s, f)   a mask built from no input of the program (iotas: jnp.tril's, make_causal_mask's),
                         causal (each query row keeps the keys up to its own index) or keeping every key
    p = exp(x - max_S x) / sum_S exp(x - max_S x)   over the keys axis S, casts between
    o = p · v            a `dot_general` contracting S, the values' one free axis their width

The call it is written as folds every axis q and k share (the product's batch: rows, and the key heads a
grouped call shares) into the call's batch, the query rows into its queries (the causal mask's row axis, or
every free axis of q where there is no mask) and any other free axis of q (a grouped call's queries a key
head) into its query heads over one key head, so every query reads exactly the keys it read as written; its
output laid back out as the second product's. The call computes attention as Dew's kernels do, on par
with the program's own (check 1), not bitwise.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jcore

from dew.nn.programs import jax_name

MASK_ENTRIES = 1 << 26
"""The most entries a mask is evaluated at to read whether it is causal; a larger one keeps its region XLA's."""

CASTS = frozenset({"convert_element_type"})


class Attention(NamedTuple):
    """A region's equations (their places, in order), its query, key and value operands as the program holds
    them, and the call written for it (`written`): whether causal, and the product's axes as the fold reads
    them."""

    places: tuple[int, ...]
    query: jcore.Var
    key: jcore.Var
    value: jcore.Var
    output: jcore.Var
    causal: bool
    shared: tuple[tuple[int, int, int], ...]
    """Each axis q and k share, its value axis beside: (q's axis, k's axis, v's axis), in the product's order."""
    rows: tuple[int, ...]
    """q's free axes read as query rows (the causal row axis first), then its query heads (`heads`)."""
    heads: tuple[int, ...]
    query_width: int
    key_axis: int
    key_width: int
    value_keys: int
    value_width: int
    out_order: tuple[int, ...]
    """The written call's output axes (shared, rows, heads, value width) as the second product lays them."""
    out_dtype: np.dtype


def regions(jaxpr: jcore.Jaxpr) -> list[Attention]:
    """Every attention region of `jaxpr` (the module's account), the first found where two would share an
    equation."""
    graph = _Graph(jaxpr)
    found, claimed = [], set()
    for place, equation in enumerate(jaxpr.eqns):
        if jax_name(equation.primitive) != "exp":
            continue
        region = _matched(graph, place)
        if region is not None and not claimed & set(region.places):
            found.append(region)
            claimed.update(region.places)
    return found


def written(region: Attention) -> Callable:
    """The region's call: its query, key and value folded as the module's account says, Dew's attention over
    them (implementation 'auto'), its output laid out as the second product's."""
    from dew.nn.attention import scaled_dot_product_attention

    def call(query, key, value):
        batch = [query.shape[q] for q, _, _ in region.shared]
        row_sizes, heads = [query.shape[a] for a in region.rows], [query.shape[a] for a in region.heads]
        q = jnp.transpose(query, [q for q, _, _ in region.shared] + list(region.rows) + list(region.heads)
                          + [region.query_width])
        q = q.reshape(math.prod(batch), math.prod(row_sizes), math.prod(heads), q.shape[-1])
        k = jnp.transpose(key, [k for _, k, _ in region.shared] + [region.key_axis, region.key_width])
        k = k.reshape(math.prod(batch), k.shape[-2], 1, k.shape[-1])
        v = jnp.transpose(value, [v for _, _, v in region.shared] + [region.value_keys, region.value_width])
        v = v.reshape(math.prod(batch), v.shape[-2], 1, v.shape[-1])
        out = scaled_dot_product_attention(q, k, v, causal=region.causal)
        out = out.reshape(*batch, *row_sizes, *heads, out.shape[-1])
        return jnp.transpose(out, region.out_order).astype(region.out_dtype)

    return call


def taken(region: Attention) -> bool:
    """Whether Dew's selector gives the region's call a kernel other than XLA's: the call traced over its
    operands' shapes (`written`), its attention route read (`generation.routes`)."""
    from dew.nn.kernels import generation

    shapes = [jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype) for var in (region.query, region.key, region.value)]
    with generation.routes() as found:
        jax.eval_shape(written(region), *shapes)
    return any(route.choice == "attention" and route.kernel != "xla" for route in found)


class _Graph:
    """A body's producers and readers."""

    def __init__(self, jaxpr: jcore.Jaxpr):
        self.jaxpr = jaxpr
        self.producer = {var: place for place, eqn in enumerate(jaxpr.eqns) for var in eqn.outvars}
        self.readers: dict[jcore.Var, set[int]] = collections.defaultdict(set)
        for place, eqn in enumerate(jaxpr.eqns):
            for var in eqn.invars:
                if isinstance(var, jcore.Var):
                    self.readers[var].add(place)
        self.escaping = {var for var in jaxpr.outvars if isinstance(var, jcore.Var)}
        self._constants: dict[jcore.Var, object] = {}

    def made(self, atom, *names: str):
        """The place of the equation making `atom`, where its primitive is one of `names`; None else."""
        place = self.producer.get(atom) if isinstance(atom, jcore.Var) else None
        if place is None or jax_name(self.jaxpr.eqns[place].primitive) not in names:
            return None
        return place

    def uncast(self, atom, places: list[int]):
        """`atom` past the casts that made it, their places added to `places`."""
        while (place := self.made(atom, *CASTS)) is not None:
            places.append(place)
            atom = self.jaxpr.eqns[place].invars[0]
        return atom

    def constant(self, atom):
        """`atom`'s value where no input of the program makes it (literals and iotas through any equation
        without effects), None else, nor where it holds more than MASK_ENTRIES entries."""
        if isinstance(atom, jcore.Literal):
            return np.asarray(atom.val)
        if atom in self._constants:
            return self._constants[atom]
        place = self.producer.get(atom)
        value = None
        if place is not None and math.prod(getattr(atom.aval, "shape", ())) <= MASK_ENTRIES:
            eqn = self.jaxpr.eqns[place]
            inputs = [self.constant(var) for var in eqn.invars]
            if not eqn.effects and all(value is not None for value in inputs):
                with jax.ensure_compile_time_eval():
                    made = eqn.primitive.bind(*inputs, **eqn.params)
                value = np.asarray((made if eqn.primitive.multiple_results else [made])[eqn.outvars.index(atom)])
        self._constants[atom] = value
        return value


def _matched(graph: _Graph, exp_place: int) -> Attention | None:
    """The attention region around the softmax exponential at `exp_place`, None where the body holds no
    region there (the module's account)."""
    eqns, places = graph.jaxpr.eqns, [exp_place]
    exponential = eqns[exp_place]
    sub = graph.made(exponential.invars[0], "sub")
    if sub is None:
        return None
    places.append(sub)
    logits, centre = eqns[sub].invars
    axis = _rows_maximum(graph, centre, logits, places)
    if axis is None:
        return None
    # The normalizer: a reader of the exponential (or its cast) sums it over the keys and divides.
    division = _normalized(graph, exponential.outvars[0], axis, places)
    if division is None:
        return None
    weights = eqns[division].outvars[0]
    if len(graph.readers[weights]) != 1:
        return None
    (reader,) = graph.readers[weights]
    if jax_name(eqns[reader].primitive) in CASTS:
        places.append(reader)
        weights = eqns[reader].outvars[0]
        if len(graph.readers[weights]) != 1:
            return None
        (reader,) = graph.readers[weights]
    if jax_name(eqns[reader].primitive) != "dot_general":
        return None
    product = (reader, weights)
    second, weights_read = product
    places.append(second)
    logits_places: list[int] = []
    scaled, causal = _unmasked(graph, logits, logits_places)
    if causal is None:
        return None
    places += logits_places
    first, query, key, query_width, key_width, layout = _scaled_product(graph, scaled, places)
    if first is None:
        return None
    region = _laid_out(graph, first, second, weights_read, query, key, layout, query_width, key_width, axis,
                       causal)
    if region is None:
        return None
    places = sorted(set(places))
    inside = set(places)
    for place in places:
        for var in eqns[place].outvars:
            if place != second and (var in graph.escaping or not graph.readers[var] <= inside):
                return None
    return region._replace(places=tuple(places))


def _rows_maximum(graph: _Graph, centre, logits, places: list[int]) -> int | None:
    """The keys axis where `centre` is `logits`' largest over it, broadcast back (and through stop_gradient
    and a maximum with a constant, jax.nn's `initial`); None where it is not."""
    eqns = graph.jaxpr.eqns
    broadcast = graph.made(centre, "broadcast_in_dim")
    if broadcast is None:
        return None
    places.append(broadcast)
    atom = eqns[broadcast].invars[0]
    if (held := graph.made(atom, "stop_gradient")) is not None:
        places.append(held)
        atom = eqns[held].invars[0]
    if (initial := graph.made(atom, "max")) is not None:
        a, b = eqns[initial].invars
        constant, atom = (a, b) if graph.constant(a) is not None else (b, a)
        if graph.constant(constant) is None:
            return None
        places.append(initial)
    reduced = graph.made(atom, "reduce_max")
    if reduced is None or eqns[reduced].invars[0] is not logits or len(eqns[reduced].params["axes"]) != 1:
        return None
    places.append(reduced)
    return int(eqns[reduced].params["axes"][0])


def _normalized(graph: _Graph, exponential, axis: int, places: list[int]) -> int | None:
    """The place of the division of the exponential (or its cast) by its sum over `axis`, broadcast back
    (casts between)."""
    eqns = graph.jaxpr.eqns
    for reader in sorted(graph.readers[exponential]):
        if jax_name(eqns[reader].primitive) != "div":
            continue
        numerator, denominator = eqns[reader].invars
        found: list[int] = []
        if graph.uncast(numerator, found) is not exponential:
            continue
        atom = graph.uncast(denominator, found)
        broadcast = graph.made(atom, "broadcast_in_dim")
        if broadcast is None:
            continue
        found.append(broadcast)
        summed = graph.made(graph.uncast(eqns[broadcast].invars[0], found), "reduce_sum")
        if (summed is None or tuple(eqns[summed].params["axes"]) != (axis,)
                or graph.uncast(eqns[summed].invars[0], found) is not exponential):
            continue
        places += [*found, summed, reader]
        return reader
    return None


def _unmasked(graph: _Graph, logits, places: list[int]):
    """`logits` past a constant mask (`jnp.where(mask, s, fill)`: a nested `_where`, or a `select_n`), and the
    mask's array with the logits' rank (`_aligned`), True where there is none; (None, None) where a mask is
    made from the program's inputs, holds more than MASK_ENTRIES entries, or lies otherwise than broadcast."""
    eqns = graph.jaxpr.eqns
    place = graph.made(logits, "jit", "pjit", "select_n")
    if place is None:
        return logits, True
    eqn = eqns[place]
    if jax_name(eqn.primitive) in ("jit", "pjit"):
        if eqn.params.get("name") != "_where":
            return logits, True
        mask, kept, fill = eqn.invars
        axes = _broadcast_axes(eqn.params["jaxpr"].jaxpr, 0, logits.aval.ndim)
    else:
        if len(eqn.invars) != 3:
            return None, None
        mask, fill, kept = eqn.invars  # select_n(predicate, on_false, on_true)
        axes = tuple(range(logits.aval.ndim))
    mask_value = graph.constant(mask)
    if mask_value is None or axes is None or graph.constant(fill) is None:
        return None, None
    places.append(place)
    aligned = np.reshape(mask_value.astype(bool), [mask_value.shape[axes.index(a)] if a in axes else 1
                                                   for a in range(logits.aval.ndim)])
    return kept, aligned


def _broadcast_axes(inner: jcore.Jaxpr, index: int, rank: int) -> tuple[int, ...] | None:
    """The output axes `inner`'s input `index` lies along where it is broadcast to the output's `rank`
    (`broadcast_in_dim`'s dimensions, through casts), every axis where it is read at the output's rank;
    None where it is laid out otherwise."""
    atom, axes = inner.invars[index], None
    for eqn in inner.eqns:
        if atom not in eqn.invars:
            continue
        name = jax_name(eqn.primitive)
        if name == "broadcast_in_dim" and eqn.invars[0] is atom:
            dims = tuple(eqn.params["broadcast_dimensions"])
            axes = dims if axes is None else tuple(dims[a] for a in axes)
            atom = eqn.outvars[0]
        elif name in CASTS:
            atom = eqn.outvars[0]
        elif name == "select_n":
            break
    if atom.aval.ndim != rank:
        return None
    return tuple(range(rank)) if axes is None else axes


def _scaled_product(graph: _Graph, scaled, places: list[int]):
    """The logits product making `scaled` at the default scale: (its place, the query and key operands it
    reads (a query divided by sqrt(D) read before its division), their contracted axes, its layout); Nones
    where the product is scaled otherwise or is no `dot_general`."""
    eqns, nothing = graph.jaxpr.eqns, (None,) * 6
    scale = None
    atom = graph.uncast(scaled, places)
    if (multiplied := graph.made(atom, "mul")) is not None:
        a, b = eqns[multiplied].invars
        if graph.constant(b) is not None and np.ndim(graph.constant(b)) == 0:
            scale, atom = float(graph.constant(b)), a
        elif graph.constant(a) is not None and np.ndim(graph.constant(a)) == 0:
            scale, atom = float(graph.constant(a)), b
        else:
            return nothing
        places.append(multiplied)
        atom = graph.uncast(atom, places)
    first = graph.made(atom, "dot_general")
    if first is None:
        return nothing
    eqn = eqns[first]
    (contracted_q, contracted_k), _ = eqn.params["dimension_numbers"]
    if len(contracted_q) != 1:
        return nothing
    query, key = eqn.invars
    width = query.aval.shape[contracted_q[0]]
    if scale is None:
        # Flax's: the query divided by sqrt(D) before the product, its quotient read by the product alone.
        divided = graph.made(query, "div")
        if divided is None or graph.readers[query] != {first}:
            return nothing
        numerator, denominator = eqns[divided].invars
        root = graph.constant(denominator)
        if root is None or np.ndim(root) != 0 or not math.isclose(float(root), math.sqrt(width), rel_tol=1e-3):
            return nothing
        query = numerator
    elif not math.isclose(scale, width ** -0.5, rel_tol=1e-6):
        return nothing
    places.append(first)
    return first, query, key, contracted_q[0], contracted_k[0], eqn.params["dimension_numbers"]


def _laid_out(graph: _Graph, first: int, second: int, weights, query, key, layout, query_width: int,
              key_width: int, axis: int, mask) -> Attention | None:
    """The written call's folds (the module's account) from the two products' layouts and the mask; None
    where they do not hold one: the keys not a free axis of k, the values' product contracting another axis
    than the keys or sharing other axes than q and k do, the values more than one free axis, or a mask
    neither causal over the query rows and the keys nor keeping every key."""
    eqns = graph.jaxpr.eqns
    (_, _), (shared_q, shared_k) = layout
    q_free = [a for a in range(query.aval.ndim) if a != query_width and a not in shared_q]
    k_free = [a for a in range(key.aval.ndim) if a != key_width and a not in shared_k]
    # The logits' axes: the shared, then q's free, then k's free (dot_general's order).
    logits_rank = len(shared_q) + len(q_free) + len(k_free)
    if len(k_free) != 1 or axis != logits_rank - 1:
        return None
    (left_c, right_c), (left_b, right_b) = eqns[second].params["dimension_numbers"]
    left, right = eqns[second].invars
    if left is weights:
        weights_c, value_c, weights_b, value_b, value = left_c, right_c, left_b, right_b, right
    elif right is weights:
        weights_c, value_c, weights_b, value_b, value = right_c, left_c, right_b, left_b, left
    else:
        return None
    if tuple(weights_c) != (axis,) or sorted(weights_b) != list(range(len(shared_q))) or len(value_c) != 1:
        return None
    value_free = [a for a in range(value.aval.ndim) if a not in value_c and a not in value_b]
    if len(value_free) != 1:
        return None
    value_shared = {w: v for w, v in zip(weights_b, value_b, strict=True)}
    shared = tuple((q, k, value_shared[i]) for i, (q, k) in enumerate(zip(shared_q, shared_k, strict=True)))
    causal = _causal(mask, len(shared_q), len(q_free), logits_rank)
    if causal is None:
        return None
    is_causal, row = causal
    rows = (q_free[row],) if is_causal else tuple(q_free)
    heads = tuple(a for a in q_free if a not in rows)
    # The call's output axes: shared (their logits order), rows, heads, the value width; the second
    # product's: its batch (the weights' order), then the first operand's free axes, then the second's.
    ours = ([("shared", i) for i in range(len(shared_q))] + [("free", q_free.index(a)) for a in rows]
            + [("free", q_free.index(a)) for a in heads] + [("width", 0)])
    weights_free = [("free", a - len(shared_q)) for a in range(logits_rank) if a >= len(shared_q) and a != axis]
    theirs = [("shared", w) for w in weights_b]
    theirs += (weights_free + [("width", 0)]) if left is weights else ([("width", 0)] + weights_free)
    out_order = tuple(ours.index(label) for label in theirs)
    return Attention((), query, key, value, eqns[second].outvars[0], is_causal, shared, rows, heads,
                     query_width, k_free[0], key_width, value_c[0], value_free[0], out_order,
                     np.dtype(eqns[second].outvars[0].aval.dtype))


def _causal(mask, shared: int, free: int, rank: int) -> tuple[bool, int] | None:
    """(True, the q-free axis it reads as query rows) where `mask` (the logits' rank, `_unmasked`) keeps each
    query row's keys up to its own index (jax's `is_causal`, top left aligned) and is the same along every
    other axis; (False, 0) where it keeps every key (True for none); None else."""
    if mask is True or bool(np.all(mask)):
        return False, 0
    del rank

    def varies(a: int) -> bool:
        return mask.shape[a] > 1 and not np.all(mask == np.take(mask, [0], axis=a))

    varying = [a for a in range(shared, shared + free) if varies(a)]
    if len(varying) != 1 or any(varies(a) for a in range(shared)):
        return None
    row = varying[0]
    plane = np.moveaxis(mask, (row, mask.ndim - 1), (-2, -1))
    plane = plane.reshape(-1, *plane.shape[-2:])[0]
    if not np.array_equal(plane, np.tril(np.ones(plane.shape, bool))):
        return None
    return True, row - shared
