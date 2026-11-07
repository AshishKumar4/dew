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

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jcore

from dew.nn.programs import jax_name

MASK_ENTRIES = 1 << 26
"""The most entries a mask is evaluated at to read whether it is causal; a larger one keeps its region
XLA's."""

CASTS = frozenset({"convert_element_type"})


WIDENINGS = frozenset({("bfloat16", "float32"), ("float16", "float32"), ("bfloat16", "float64"),
                       ("float16", "float64"), ("float32", "float64")})
"""The float casts that keep every value: a wider exponent range and a longer significand at once (bfloat16 to
float16 keeps neither: its range past 65504 is lost)."""


def _widening(source: np.dtype, target: np.dtype) -> bool:
    """A float cast that keeps every value (`WIDENINGS`), or none."""
    return source == target or (np.dtype(source).name, np.dtype(target).name) in WIDENINGS


def _floating(source: np.dtype, target: np.dtype) -> bool:
    """A cast between floats (the weights rounded once to the values' dtype for their product)."""
    return all(jnp.issubdtype(dtype, jnp.floating) for dtype in (source, target))


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
    """Each axis q and k share, its value axis beside: (q's axis, k's axis, v's axis), in the product's
    order."""
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


def regions(index) -> list[Attention]:
    """Every attention region of the body `index` reads (`fusion.Index`; the module's account), the first
    found where two would share an equation."""
    graph, jaxpr = _Graph(index), index.jaxpr
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


KEPT = frozenset({"xla", "reference"})
"""The attention kernels that compute as the program's own XLA does (Dew's reference attention among them):
a region the selector gives one of keeps the program's math (`taken`)."""


def taken(region: Attention) -> bool:
    """Whether Dew's selector gives the region's call a kernel of its own (the template, FlashAttention,
    cuDNN; none of `KEPT`): the call traced over its operands' shapes (`written`), its attention route read
    (`generation.routes`)."""
    from dew.nn.kernels import generation

    shapes = [jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype)
              for var in (region.query, region.key, region.value)]
    with generation.routes() as found:
        jax.eval_shape(written(region), *shapes)
    return any(route.choice == "attention" and route.kernel not in KEPT for route in found)


class _Graph:
    """A body's index (`fusion.Index`: its producers, readers and returned values) with the reads this
    module's matcher takes: an equation by its primitive, a value past the casts a rule admits, and a value
    the program makes from no input."""

    def __init__(self, index):
        self.jaxpr, self.producer, self.readers, self.escaping = (index.jaxpr, index.producer, index.readers,
                                                                  index.escaping)
        self._constants: dict[jcore.Var, object] = {}

    def made(self, atom, *names: str):
        """The place of the equation making `atom`, where its primitive is one of `names`; None else."""
        place = self.producer.get(atom) if isinstance(atom, jcore.Var) else None
        if place is None or jax_name(self.jaxpr.eqns[place].primitive) not in names:
            return None
        return place

    def uncast(self, atom, places: list[int], admits: Callable[[np.dtype, np.dtype], bool]):
        """`atom` past the casts that made it, each from the dtype its input holds to its own where `admits`
        the pair, their places added to `places`; at the first cast it does not admit, that cast's output."""
        while (place := self.made(atom, *CASTS)) is not None:
            source = self.jaxpr.eqns[place].invars[0]
            if not admits(np.dtype(source.aval.dtype), np.dtype(atom.aval.dtype)):
                break
            places.append(place)
            atom = source
        return atom

    def excluding(self, atom, dtype) -> bool:
        """Whether `atom` is a constant scalar a fill or a maximum's initial value excludes with: -inf, or a
        sentinel at least half `dtype`'s most negative finite value (jax's and Flax's `finfo.min` fills).
        The written call and the program agree while every visible logit lies above the fill, as jax's and
        Flax's own fills assume: a visible logit near `finfo.min` would weigh as a masked one there."""
        value = self.constant(atom)
        if value is None or np.ndim(value) != 0:
            return False
        value = float(value)
        return value == -math.inf or value <= 0.5 * float(jnp.finfo(dtype).min)

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
                value = _evaluated(eqn, inputs, eqn.outvars.index(atom))
        self._constants[atom] = value
        return value


def _evaluated(eqn: jcore.JaxprEqn, inputs: list, index: int):
    """`eqn`'s output `index` over concrete `inputs`, evaluated on this host's CPU (a compile-only mesh's
    devices run nothing); None where the host evaluates it to no value (no CPU backend, or the primitive has
    no rule there), so a region whose mask is not read keeps its program's own math."""
    try:
        cpu = jax.devices("cpu")[0]
        with jax.ensure_compile_time_eval(), jax.default_device(cpu):
            made = eqn.primitive.bind(*inputs, **eqn.params)
        return np.asarray((made if eqn.primitive.multiple_results else [made])[index])
    except (RuntimeError, ValueError, TypeError, AttributeError, NotImplementedError):
        return None


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
    logits_places: list[int] = []
    scaled, mask, perm = _unmasked(graph, logits, logits_places)
    if perm is None:
        return None
    axis = _rows_maximum(graph, centre, logits, scaled, perm, places)
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
        # The weights cast once, to the values' dtype, as both spellings narrow them for the product.
        cast_to = np.dtype(eqns[reader].outvars[0].aval.dtype)
        if not _floating(np.dtype(weights.aval.dtype), cast_to):
            return None  # the weights rounded once to the values' dtype: the product's operand, as written
        places.append(reader)
        weights = eqns[reader].outvars[0]
        if len(graph.readers[weights]) != 1:
            return None
        (reader,) = graph.readers[weights]
    if jax_name(eqns[reader].primitive) != "dot_general" or any(
            np.dtype(var.aval.dtype) != np.dtype(weights.aval.dtype) for var in eqns[reader].invars):
        return None
    product = (reader, weights)
    second, weights_read = product
    places.append(second)
    places += logits_places
    first, query, key, query_width, key_width, layout = _scaled_product(graph, scaled, places)
    if first is None:
        return None
    region = _laid_out(graph, second, weights_read, query, key, layout, query_width, key_width, axis, mask,
                       perm)
    if region is None:
        return None
    places = sorted(set(places))
    inside = set(places)
    for place in places:
        for var in eqns[place].outvars:
            if place != second and (var in graph.escaping or not graph.readers[var] <= inside):
                return None
    return region._replace(places=tuple(places))


def _rows_maximum(graph: _Graph, centre, logits, scaled, perm: tuple[int, ...],
                  places: list[int]) -> int | None:
    """The keys axis (the softmax's layout, `logits`') where `centre` is the logits' largest over it, each
    row's own (`_back_to_rows`): broadcast back, held from its gradient, raised to an excluding initial
    value (jax.nn's -inf), laid out again and widened, in any order; a maximum of `logits` itself, or of
    `scaled` before the layout `perm` moved it (jax.nn's vmapped softmax over a grouped call). None where it
    is not."""
    eqns, atom, chain = graph.jaxpr.eqns, centre, []
    while True:
        place = graph.made(atom, "broadcast_in_dim", "stop_gradient", "max", "transpose", *CASTS)
        if place is None:
            break
        eqn = eqns[place]
        name = jax_name(eqn.primitive)
        if name == "max":
            a, b = eqn.invars
            constant, atom = (a, b) if graph.constant(a) is not None else (b, a)
            if not graph.excluding(constant, logits.aval.dtype):
                return None
        else:
            if name in CASTS and not _widening(np.dtype(eqn.invars[0].aval.dtype), np.dtype(atom.aval.dtype)):
                return None
            atom = eqn.invars[0]
        chain.append(place)
    reduced = graph.made(atom, "reduce_max")
    if reduced is None or len(eqns[reduced].params["axes"]) != 1:
        return None
    reduce_axis, read = int(eqns[reduced].params["axes"][0]), eqns[reduced].invars[0]
    if read is logits:
        axis, labels = reduce_axis, [a for a in range(read.aval.ndim) if a != reduce_axis]
    elif read is scaled:
        axis = perm.index(reduce_axis)
        labels = [perm.index(a) for a in range(read.aval.ndim) if a != reduce_axis]
    else:
        return None
    if not _back_to_rows(eqns, chain[::-1], labels, centre, axis):
        return None
    places += [*chain, reduced]
    return axis


def _back_to_rows(eqns, chain: list[int], labels: list[int], value, axis: int) -> bool:
    """Whether a row reduction's value (its axes `labels`: the softmax axes they hold, the reduced `axis`
    gone), through `chain`'s equations in order (transposes, broadcasts, casts, holds and maxima of a
    scalar), reaches `value` laid along the softmax's axes again: each of its axes but `axis` the same axis,
    `axis` a broadcast one. A sum or maximum broadcast to another row's place (`[None, :]` of a reduced
    vector) does not."""
    labels = list(labels)
    for place in chain:
        eqn = eqns[place]
        name = jax_name(eqn.primitive)
        if name == "transpose":
            labels = [labels[d] for d in eqn.params["permutation"]]
        elif name == "broadcast_in_dim":
            laid: list[int | None] = [None] * len(eqn.params["shape"])
            for source, target in enumerate(eqn.params["broadcast_dimensions"]):
                laid[int(target)] = labels[source]
            labels = laid
    rank = value.aval.ndim
    return len(labels) == rank and all(labels[a] == (None if a == axis else a) for a in range(rank))


def _normalized(graph: _Graph, exponential, axis: int, places: list[int]) -> int | None:
    """The place of the division of the exponential (no cast between) by its sum over `axis` (the sum read
    from the exponential widened, `WIDENINGS`, and narrowed back to exactly its dtype, as Flax's, nothing
    below it), broadcast back to its own rows (`_back_to_rows`)."""
    eqns = graph.jaxpr.eqns
    for reader in sorted(graph.readers[exponential]):
        if jax_name(eqns[reader].primitive) != "div":
            continue
        numerator, denominator = eqns[reader].invars
        if numerator is not exponential:
            continue
        chain: list[int] = []
        held = np.dtype(exponential.aval.dtype)

        def back(source: np.dtype, target: np.dtype, held=held) -> bool:
            # The sum narrowed back to exactly the exponential's dtype from a widening of it, or kept.
            return target == held and _widening(held, source)

        atom = graph.uncast(denominator, chain, back)
        broadcast = graph.made(atom, "broadcast_in_dim")
        if broadcast is None:
            continue
        chain.append(broadcast)
        summed = graph.made(graph.uncast(eqns[broadcast].invars[0], chain, back), "reduce_sum")
        if summed is None or tuple(eqns[summed].params["axes"]) != (axis,):
            continue
        widened: list[int] = []
        if graph.uncast(eqns[summed].invars[0], widened, _widening) is not exponential:
            continue
        labels = [a for a in range(exponential.aval.ndim) if a != axis]
        if not _back_to_rows(eqns, chain[::-1], labels, denominator, axis):
            continue
        places += [*chain, *widened, summed, reader]
        return reader
    return None


def _unmasked(graph: _Graph, logits, places: list[int]):
    """`logits` past a constant mask (`jnp.where(mask, s, fill)`: a nested `_where`, or a `select_n`): the
    scaled logits `s`, the mask's array at the logits' rank (True where there is none), and the logits' axes
    as `s` lays them (`perm`: the logits' axis i is `s`'s `perm[i]`, a batched `_where` moving its batch axis
    first); (None, None, None) where a mask is made from the program's inputs, holds more than MASK_ENTRIES
    entries, or is laid out otherwise than broadcast."""
    eqns, rank, nothing = graph.jaxpr.eqns, logits.aval.ndim, (None, None, None)
    identity = tuple(range(rank))
    if (moved := graph.made(logits, "transpose")) is not None:
        places.append(moved)
        return eqns[moved].invars[0], True, tuple(int(d) for d in eqns[moved].params["permutation"])
    place = graph.made(logits, "jit", "pjit", "select_n")
    if place is None:
        return logits, True, identity
    eqn = eqns[place]
    if jax_name(eqn.primitive) in ("jit", "pjit"):
        if eqn.params.get("name") != "_where":
            return logits, True, identity
        if len(eqn.invars) != 3:
            return nothing
        mask, kept, fill = eqn.invars
        inner = eqn.params["jaxpr"].jaxpr
        axes, perm = _laid(inner, 0, rank), _laid(inner, 1, rank)
        if perm is None or sorted(perm) != list(identity):
            return nothing
    else:
        if len(eqn.invars) != 3:
            return nothing
        mask, fill, kept = eqn.invars  # select_n(predicate, on_false, on_true)
        axes, perm = identity, identity
    mask_value = graph.constant(mask)
    if mask_value is None or axes is None or not graph.excluding(fill, logits.aval.dtype):
        return nothing
    places.append(place)
    aligned = np.reshape(mask_value.astype(bool), [mask_value.shape[axes.index(a)] if a in axes else 1
                                                   for a in range(rank)])
    return kept, aligned, perm


def _laid(inner: jcore.Jaxpr, index: int, rank: int) -> tuple[int, ...] | None:
    """Where `inner`'s input `index` lies in its select's output (rank `rank`): for each of the input's axes,
    the output axis it is laid along (`broadcast_in_dim`'s dimensions), and through a transpose, for each
    output axis the input axis it reads (an input of the output's rank: its permutation); None where it
    reaches the select otherwise."""
    atom, axes = inner.invars[index], tuple(range(inner.invars[index].aval.ndim))
    for eqn in inner.eqns:
        if atom not in eqn.invars:
            continue
        name = jax_name(eqn.primitive)
        if name == "broadcast_in_dim" and eqn.invars[0] is atom and index == 0:
            dims = tuple(int(d) for d in eqn.params["broadcast_dimensions"])
            axes = tuple(dims[a] for a in axes)
        elif name == "transpose" and eqn.invars[0] is atom and index == 1:
            permutation = tuple(int(d) for d in eqn.params["permutation"])
            axes = tuple(axes[d] for d in permutation)
        elif name == "select_n":
            return axes if atom.aval.ndim == rank or index == 0 else None
        elif (name == "ne" and index == 0
              and any(isinstance(v, jcore.Literal) and v.val == 0 for v in eqn.invars)):
            pass  # a float mask's predicate, x != 0: the mask's value read as booleans (`_unmasked`'s astype)
        elif name not in CASTS:
            return None
        atom = eqn.outvars[0]
    return None


def _scaled_product(graph: _Graph, scaled, places: list[int]):
    """The logits product making `scaled` at the default scale: (its place, the query and key operands it
    reads (a query divided by sqrt(D) read before its division), their contracted axes, its layout); Nones
    where the product is scaled otherwise or is no `dot_general`."""
    eqns, nothing = graph.jaxpr.eqns, (None,) * 6
    scale = None
    atom = graph.uncast(scaled, places, _widening)
    if (multiplied := graph.made(atom, "mul")) is not None:
        a, b = eqns[multiplied].invars
        if graph.constant(b) is not None and np.ndim(graph.constant(b)) == 0:
            scale, atom = float(graph.constant(b)), a
        elif graph.constant(a) is not None and np.ndim(graph.constant(a)) == 0:
            scale, atom = float(graph.constant(a)), b
        else:
            return nothing
        places.append(multiplied)
        atom = graph.uncast(atom, places, _widening)
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
        if (root is None or np.ndim(root) != 0
                or not math.isclose(float(root), math.sqrt(width), rel_tol=1e-3)):
            return nothing
        query = numerator
    elif not math.isclose(scale, width ** -0.5, rel_tol=1e-6):
        return nothing
    places.append(first)
    return first, query, key, contracted_q[0], contracted_k[0], eqn.params["dimension_numbers"]


def _laid_out(graph: _Graph, second: int, weights, query, key, layout, query_width: int, key_width: int,
              axis: int, mask, perm: tuple[int, ...]) -> Attention | None:
    """The written call's folds (the module's account) from the two products' layouts and the mask, the
    softmax's axes as the logits product lays them through `perm` (`_unmasked`'s); None where they do not
    hold one: the keys not k's one free axis, the values' product contracting another axis than the keys or
    sharing other axes than q and k do, the values more than one free axis, or a mask neither causal over the
    query rows and the keys nor keeping every key."""
    eqns = graph.jaxpr.eqns
    _, (shared_q, shared_k) = layout
    q_free = [a for a in range(query.aval.ndim) if a != query_width and a not in shared_q]
    k_free = [a for a in range(key.aval.ndim) if a != key_width and a not in shared_k]
    # The product's logits: the shared axes, then q's free, then k's free (dot_general's order).
    shared_n, rank = len(shared_q), len(shared_q) + len(q_free) + len(k_free)
    if len(k_free) != 1 or len(perm) != rank or perm[axis] != rank - 1:
        return None
    (left_c, right_c), (left_b, right_b) = eqns[second].params["dimension_numbers"]
    left, right = eqns[second].invars
    if left is weights:
        weights_c, value_c, weights_b, value_b, value = left_c, right_c, left_b, right_b, right
    elif right is weights:
        weights_c, value_c, weights_b, value_b, value = right_c, left_c, right_b, left_b, left
    else:
        return None
    shared_b = [perm[w] for w in weights_b]
    if tuple(weights_c) != (axis,) or sorted(shared_b) != list(range(shared_n)) or len(value_c) != 1:
        return None
    value_free = [a for a in range(value.aval.ndim) if a not in value_c and a not in value_b]
    if len(value_free) != 1:
        return None
    value_shared = dict(zip(shared_b, value_b, strict=True))
    shared = tuple((q, k, value_shared[i]) for i, (q, k) in enumerate(zip(shared_q, shared_k, strict=True)))
    mask = mask if mask is True else np.transpose(mask, [perm.index(a) for a in range(rank)])
    causal = _causal(mask, shared_n, len(q_free))
    if causal is None:
        return None
    is_causal, row = causal
    rows = (q_free[row],) if is_causal else tuple(q_free)
    heads = tuple(a for a in q_free if a not in rows)
    # The call's output axes: shared (the product's order), rows, heads, the value width; the second
    # product's: its batch (as the weights pair them), then its first operand's free axes, then its second's.
    ours = ([("shared", i) for i in range(shared_n)] + [("free", q_free.index(a)) for a in rows]
            + [("free", q_free.index(a)) for a in heads] + [("width", 0)])
    weights_free = [("free", perm[i] - shared_n) for i in range(rank) if i not in weights_b and i != axis]
    theirs = [("shared", s) for s in shared_b]
    theirs += [*weights_free, ("width", 0)] if left is weights else [("width", 0), *weights_free]
    out_order = tuple(ours.index(label) for label in theirs)
    return Attention((), query, key, value, eqns[second].outvars[0], is_causal, shared, rows, heads,
                     query_width, k_free[0], key_width, value_c[0], value_free[0], out_order,
                     np.dtype(eqns[second].outvars[0].aval.dtype))


def _causal(mask, shared: int, free: int) -> tuple[bool, int] | None:
    """(True, the q-free axis it reads as query rows) where `mask` (the logits' rank, `_unmasked`) keeps each
    query row's keys up to its own index (jax's `is_causal`, top left aligned) and is the same along every
    other axis; (False, 0) where it keeps every key (True for none); None else."""
    if mask is True or bool(np.all(mask)):
        return False, 0

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
