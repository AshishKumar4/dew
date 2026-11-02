"""Whether a traced forward is causal over its tokens, decided from its program alone, with no weights
(ledger 1595, the 1493 design's H7): tier 3 admits a source whose shrunken probe cannot run by a trace of
its full config over abstract weights, and reads that trace's causality here instead of from the
probe's logits.

The walk follows every value the tokens reach through the program (`jax.make_jaxpr`'s), keeping the axis
that holds their sequence position. An operation that combines values across that axis mixes positions,
and is admitted only where it is causal by its structure: Dew's causal attention (`CAUSAL_ATTENTION`, the
named call tier 3's attention runs in), a computation Dew states is token by token (`TOKENWISE`, its
experts' dispatch), a convolution whose padding over the sequence is all on the left, a scan or a
cumulative operation running forward over it, and a gather whose indices are each token's position or an
earlier one. Anything else that mixes positions refuses the program, naming the operation."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence

from jax.extend import core

CAUSAL_ATTENTION = "dew_causal_attention"
"""The name of the jitted call tier 3's attention runs in (`torchax_fallback._dew_attention`), causal over
its keys by construction, which the walk admits whole."""

TOKENWISE = "dew_tokenwise"
"""The name of a jitted call Dew computes token by token (tier 3's experts' dispatch,
`torchax_fallback.functional_experts`), whose outputs at a token read that token's inputs alone."""


class SequenceMixed(ValueError):
    """A traced program whose outputs read later positions, or that the walk cannot show reads none,
    naming the operation."""


MIXED = -1
"""A value that depends on the tokens but holds no position axis (a scan's last carry): its every entry
may read every position."""


@dataclasses.dataclass(frozen=True)
class Positions:
    """A value the tokens do not reach that holds, along `axis`, an index no later than each position
    plus `offset`: `iota` over the sequence, shifted or clamped, the indices a causal gather reads by;
    `nonnegative` where none is below zero (a clamp from below at zero holds it), so a wrap of negative
    indices (`jnp`'s indexing) leaves it as it is."""

    axis: int
    offset: int
    nonnegative: bool = True


type Held = int | Positions | None
"""What the walk knows of a value: its sequence axis (`int`), `MIXED`, a position index (`Positions`),
or None for a value the tokens do not reach."""


def causality_refusal(closed: core.ClosedJaxpr, tokens: int, length: int) -> str | None:
    """Why `closed` is not provably causal over the sequence of its input `tokens` (`[B, S]`, `length`
    tokens a row, the sequence on axis 1), or None where each of its outputs at a position reads that
    position and earlier ones alone."""
    held: list[Held] = [1 if index == tokens else None for index in range(len(closed.jaxpr.invars))]
    try:
        outputs = _Walk(length).run(closed.jaxpr, held)
    except SequenceMixed as mixed:
        return str(mixed)
    if any(output == MIXED for output in outputs):
        return "an output reads every position (a scan's last state, or a value pooled over the sequence)"
    return None


class _Walk:
    def __init__(self, length: int):
        self.length = length

    def run(self, jaxpr: core.Jaxpr, inputs: Sequence[Held]) -> list[Held]:
        held: dict[core.Var, Held] = dict(zip(jaxpr.invars, inputs, strict=True))
        for var in jaxpr.constvars:
            held[var] = None
        self.producers: dict[core.Var, core.JaxprEqn] = {}
        for eqn in jaxpr.eqns:
            given = [None if isinstance(var, core.Literal) else held.get(var) for var in eqn.invars]
            held.update(zip(eqn.outvars, self.equation(eqn, given), strict=True))
            self.producers.update(dict.fromkeys(eqn.outvars, eqn))
        return [None if isinstance(var, core.Literal) else held.get(var) for var in jaxpr.outvars]

    def equation(self, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
        name = eqn.primitive.name
        reached = [value for value in given if isinstance(value, int)]
        rule = _RULES.get(name)
        if rule is not None:
            return rule(self, eqn, given)
        if not reached:
            # The tokens reach none of its inputs: a position index survives elementwise steps alone.
            return [_positions_through(self, eqn, given)] * len(eqn.outvars)
        if name in _MIXING:
            raise SequenceMixed(f"{name} combines the tokens' values across positions")
        return self.elementwise(eqn, given)

    def elementwise(self, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
        """An operation each output entry of which reads the inputs' entries at its own index: the
        tokens' axis passes through, if every input the tokens reach holds it at one place."""
        shape = _shape(eqn.outvars[0])
        axes = set()
        for var, value in zip(eqn.invars, given, strict=True):
            if not isinstance(value, int):
                continue
            if value == MIXED or _shape(var) != shape:
                if any(isinstance(other, int) and other >= 0 for other in given):
                    raise SequenceMixed(f"{eqn.primitive.name} combines a value read from every position "
                                        "with each position's")
                return [MIXED] * len(eqn.outvars)
            axes.add(value)
        if len(axes) > 1:
            raise SequenceMixed(f"{eqn.primitive.name} pairs positions along two axes {sorted(axes)}")
        return [axes.pop()] * len(eqn.outvars)


def _shape(var) -> tuple[int, ...]:
    return tuple(getattr(var.aval, "shape", ()))


def _single(given: Sequence[Held]) -> int | None:
    """The one input axis the tokens reach, None where they reach none; refused where they reach a
    value without one."""
    values = [value for value in given if isinstance(value, int)]
    if any(value == MIXED for value in values):
        return MIXED
    return values[0] if values else None


def _positions_through(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> Positions | None:
    """A position index carried through an operation the tokens do not reach: `iota` over the sequence
    makes one; a shift by a literal moves its offset; a cast keeps it; a clamp from below at zero or
    more makes it nonnegative, and any clamp from above keeps it no later than it was; a wrap of
    negative indices (`select_n` of `lt(index, 0)`, the index, and the index plus the sequence's
    length) leaves a nonnegative one as it is; anything else loses it."""
    name = eqn.primitive.name
    if name == "iota":
        dimension = int(eqn.params["dimension"])
        return Positions(dimension, 0) if _shape(eqn.outvars[0])[dimension] == walk.length else None
    indices = [value for value in given if isinstance(value, Positions)]
    literals = [float(var.val) for var in eqn.invars if isinstance(var, core.Literal)]
    if name == "select_n" and len(given) == 3 and isinstance(given[1], Positions):
        return given[1] if _wrapped(walk, eqn, given) else None
    if len(indices) != 1:
        return None
    index = indices[0]
    if name in ("convert_element_type", "copy", "copy_p"):
        return index
    if name in ("add", "sub") and len(literals) == 1 and literals[0].is_integer():
        if name == "sub" and isinstance(given[1], Positions):
            return None  # a literal less the position
        shift = int(literals[0]) * (1 if name == "add" else -1)
        return Positions(index.axis, index.offset + shift, index.nonnegative and shift >= 0)
    # A bound from below at zero or less keeps each index no later than its position (the positions
    # count from zero), at zero making it nonnegative; one from above keeps it no later than it was.
    if name == "max" and len(literals) == 1 and literals[0] <= 0:
        return Positions(index.axis, max(index.offset, 0), index.nonnegative or literals[0] == 0)
    if name == "min" and len(literals) == 1:
        return Positions(index.axis, index.offset, index.nonnegative and literals[0] >= 0)
    if name == "clamp" and isinstance(eqn.invars[0], core.Literal) and float(eqn.invars[0].val) <= 0:
        return Positions(index.axis, max(index.offset, 0), index.nonnegative or float(eqn.invars[0].val) == 0)
    return None


def _wrapped(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> bool:
    """Whether `eqn`, a `select_n`, wraps negative indices of a nonnegative position index: its
    predicate `lt` of its first case and zero, its second case the first plus the sequence's length."""
    first, second = given[1], given[2]
    if not (isinstance(first, Positions) and first.nonnegative and isinstance(second, Positions)
            and second.axis == first.axis and second.offset == first.offset + walk.length):
        return False
    test, shifted = (None if isinstance(var, core.Literal) else walk.producers.get(var)
                     for var in (eqn.invars[0], eqn.invars[2]))
    if (test is None or test.primitive.name != "lt" or test.invars[0] is not eqn.invars[1]
            or shifted is None or shifted.primitive.name != "add" or shifted.invars[0] is not eqn.invars[1]):
        return False
    zero, length = test.invars[1], shifted.invars[1]
    return (isinstance(zero, core.Literal) and float(zero.val) == 0 and isinstance(length, core.Literal)
            and float(length.val) == walk.length)


def _call(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    """A call of a jaxpr of its own: Dew's causal attention and its token-by-token computations admitted
    whole, any other walked inside."""
    name = eqn.params.get("name")
    if name in (CAUSAL_ATTENTION, TOKENWISE):
        axis = _single(given)
        if axis == MIXED:
            raise SequenceMixed(f"{name} reads a value read from every position")
        if axis is None:
            return [None] * len(eqn.outvars)
        size = _shape(eqn.invars[next(index for index, value in enumerate(given) if value == axis)])[axis]
        return [axis if len(_shape(var)) > axis and _shape(var)[axis] == size else MIXED
                for var in eqn.outvars]
    inner = _inner(eqn)
    return walk.run(inner.jaxpr if isinstance(inner, core.ClosedJaxpr) else inner, given)


def _inner(eqn: core.JaxprEqn):
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if key in eqn.params:
            return eqn.params[key]
    raise SequenceMixed(f"{eqn.primitive.name} calls a program the walk does not read")


def _broadcast_in_dim(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    value, dims = given[0], eqn.params["broadcast_dimensions"]
    if isinstance(value, Positions):
        return [Positions(dims[value.axis], value.offset)]
    if value is None or value == MIXED:
        return [value]
    return [dims[value]]


def _reshape(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    value = given[0]
    if value is None or value == MIXED:
        return [value]
    axis = value.axis if isinstance(value, Positions) else value
    before, after = _shape(eqn.invars[0]), tuple(eqn.params["new_sizes"])
    lead, size = math.prod(before[:axis]), before[axis]
    for index, width in enumerate(after):
        if width == size and math.prod(after[:index]) == lead:
            return [Positions(index, value.offset) if isinstance(value, Positions) else index]
    if isinstance(value, Positions):
        return [None]
    raise SequenceMixed(f"reshape {before} -> {after} folds the sequence axis into another")


def _squeeze(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    value, dims = given[0], eqn.params["dimensions"]
    if value is None or value == MIXED:
        return [value]
    axis = value.axis if isinstance(value, Positions) else value
    moved = axis - sum(1 for dim in dims if dim < axis)
    return [Positions(moved, value.offset) if isinstance(value, Positions) else moved]


def _transpose(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    value, order = given[0], tuple(eqn.params["permutation"])
    if value is None or value == MIXED:
        return [value]
    if isinstance(value, Positions):
        return [Positions(order.index(value.axis), value.offset)]
    return [order.index(value)]


def _reduce(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    axis, axes = _single(given), tuple(eqn.params["axes"])
    if axis is None or axis == MIXED:
        return [axis] * len(eqn.outvars)
    if axis in axes:
        raise SequenceMixed(f"{eqn.primitive.name} reduces over the sequence axis")
    return [axis - sum(1 for dim in axes if dim < axis)] * len(eqn.outvars)


def _cumulative(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    axis = _single(given)
    if axis is None or axis == MIXED:
        return [axis]
    if axis == eqn.params["axis"] and eqn.params.get("reverse", False):
        raise SequenceMixed(f"{eqn.primitive.name} runs backward over the sequence axis")
    return [axis]


def _dot_general(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params["dimension_numbers"]
    lhs_rank, rhs_rank = len(_shape(eqn.invars[0])), len(_shape(eqn.invars[1]))
    lhs_free = [dim for dim in range(lhs_rank) if dim not in lhs_contract and dim not in lhs_batch]
    rhs_free = [dim for dim in range(rhs_rank) if dim not in rhs_contract and dim not in rhs_batch]
    placed = set()
    for value, contract, batch, free, start in ((given[0], lhs_contract, lhs_batch, lhs_free, len(lhs_batch)),
                                                (given[1], rhs_contract, rhs_batch, rhs_free,
                                                 len(lhs_batch) + len(lhs_free))):
        if not isinstance(value, int):
            continue
        if value == MIXED:
            placed.add(MIXED)
        elif value in contract:
            raise SequenceMixed("dot_general contracts the sequence axis outside Dew's attention")
        else:
            placed.add(list(batch).index(value) if value in batch else start + free.index(value))
    if MIXED in placed and len(placed) > 1:
        raise SequenceMixed("dot_general combines a value read from every position with each position's")
    if len(placed) > 1:
        raise SequenceMixed("dot_general pairs the positions of two token-reached operands")
    return [placed.pop() if placed else None]


def _conv(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    lhs, rhs = given[0], given[1]
    if isinstance(rhs, int):
        raise SequenceMixed("a convolution's kernel is read from the tokens")
    if not isinstance(lhs, int):
        return [None]
    if lhs == MIXED:
        return [MIXED]
    numbers = eqn.params["dimension_numbers"]
    lhs_spec, rhs_spec, out_spec = numbers.lhs_spec, numbers.rhs_spec, numbers.out_spec
    if lhs == lhs_spec[0]:
        return [out_spec[0]]
    if lhs == lhs_spec[1]:
        raise SequenceMixed("a convolution takes the sequence axis for its features")
    spatial = list(lhs_spec[2:]).index(lhs)
    low, high = eqn.params["padding"][spatial]
    width = _shape(eqn.invars[1])[rhs_spec[2 + spatial]]
    dilation = eqn.params["rhs_dilation"][spatial]
    if (high != 0 or low != dilation * (width - 1) or eqn.params["window_strides"][spatial] != 1
            or eqn.params["lhs_dilation"][spatial] != 1):
        raise SequenceMixed(f"a convolution over the sequence pads it ({low}, {high}) for a window of "
                            f"{width}: not all on the left, so a position reads later ones")
    return [out_spec[2 + spatial]]


def _gather(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    operand, indices = given[0], given[1]
    numbers = eqn.params["dimension_numbers"]
    out_rank = len(_shape(eqn.outvars[0]))
    offset_dims = tuple(numbers.offset_dims)
    batch_out = [dim for dim in range(out_rank) if dim not in offset_dims]
    rank = len(_shape(eqn.invars[0]))
    collapsed, operand_batching = tuple(numbers.collapsed_slice_dims), tuple(numbers.operand_batching_dims)
    kept = [dim for dim in range(rank) if dim not in collapsed and dim not in operand_batching]
    placed = set()
    if operand == MIXED or indices == MIXED:
        return [MIXED]
    if isinstance(operand, int):
        if operand in operand_batching:
            paired = numbers.start_indices_batching_dims[operand_batching.index(operand)]
            placed.add(batch_out[paired])
        elif operand in numbers.start_index_map:
            if not (isinstance(indices, Positions) and indices.offset <= 0):
                raise SequenceMixed("gather reads the sequence axis by indices no later than nothing proves "
                                    "each token's position")
            placed.add(batch_out[indices.axis])
        elif eqn.params["slice_sizes"][operand] != _shape(eqn.invars[0])[operand]:
            raise SequenceMixed("gather slices the sequence axis")
        else:
            placed.add(offset_dims[kept.index(operand)])
    if isinstance(indices, int):
        placed.add(batch_out[indices])
    if len(placed) > 1:
        raise SequenceMixed("gather pairs the positions of its operand and its indices")
    return [placed.pop() if placed else None]


def _sort(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    axis = _single(given)
    if isinstance(axis, int) and axis >= 0 and axis == eqn.params["dimension"]:
        raise SequenceMixed("sort orders values across positions")
    return [axis] * len(eqn.outvars)


def _top_k(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    axis = _single(given)
    if isinstance(axis, int) and axis >= 0 and axis == len(_shape(eqn.invars[0])) - 1:
        raise SequenceMixed("top_k selects across positions")
    return [axis] * len(eqn.outvars)


def _sliced(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    """A slice, pad, reversal or concatenation: admitted where the sequence axis passes whole."""
    reached = {value for value in given if isinstance(value, int)}
    if not reached or MIXED in reached:
        return [MIXED if reached else None] * len(eqn.outvars)
    if len(reached) > 1:
        raise SequenceMixed(f"{eqn.primitive.name} pairs positions along two axes {sorted(reached)}")
    axis = reached.pop()
    size = _shape(eqn.outvars[0])[axis]
    if any(isinstance(value, int) and _shape(var)[axis] != size for var, value in zip(eqn.invars, given,
                                                                                         strict=True)) or (
            eqn.primitive.name == "rev" and axis in eqn.params["dimensions"]) or (
            eqn.primitive.name == "concatenate" and axis == eqn.params["dimension"]):
        raise SequenceMixed(f"{eqn.primitive.name} moves values along the sequence axis")
    return [axis] * len(eqn.outvars)


def _merged(held: Held, found: Held) -> Held:
    """A carried value's axis across two passes of a loop's body: one it keeps, or one it takes on from
    none; refused where it moves."""
    if held == found or found is None:
        return held
    if held is None or found == MIXED:
        return found
    raise SequenceMixed("a loop's carried value moves its sequence axis from step to step")


def _fixed(walk: _Walk, body: core.Jaxpr, consts: list[Held], carried: list[Held],
           rest: list[Held]) -> list[Held]:
    """`body`'s outputs once its carried values' axes stop changing between passes."""
    for _ in range(len(carried) + 2):
        out = walk.run(body, [*consts, *carried, *rest])
        found = [_merged(held, value) for held, value in zip(carried, out[:len(carried)], strict=True)]
        if found == carried:
            return out
        carried = found
    raise SequenceMixed("a loop's carried values do not settle on their sequence axes")


def _scan(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    """A scan: over the sequence (a scanned input the tokens reach on its axis 0), forward, each step's
    output reads that step and the carry of the steps before it, and its last carry every step; over
    another axis, its body walked with its carried values' axes held fixed."""
    params = eqn.params
    consts, carries = params["num_consts"], params["num_carry"]
    body = params["jaxpr"].jaxpr
    xs = list(given[consts + carries:])
    if any(value == 0 for value in xs if isinstance(value, int)):
        if params["reverse"]:
            raise SequenceMixed("a scan runs backward over the sequence")
        if any(isinstance(value, int) for value in given[:consts]):
            raise SequenceMixed("a scan over the sequence reads at each step a value the tokens reach whole")
        if any(isinstance(value, int) and value > 0 for value in xs):
            raise SequenceMixed("a scan over the sequence steps through a value holding it on another axis")
        steps: list[Held] = [MIXED if isinstance(value, int) else None for value in xs]
        out = _fixed(walk, body, list(given[:consts]), [MIXED if isinstance(value, int) else value
                                                         for value in given[consts:consts + carries]],
                     steps)
        last = [MIXED if isinstance(value, int) else value for value in out[:carries]]
        return [*last, *[0 if isinstance(value, int) else None for value in out[carries:]]]
    steps = [value - 1 if isinstance(value, int) and value > 0 else value for value in xs]
    out = _fixed(walk, body, list(given[:consts]), list(given[consts:consts + carries]), steps)
    return [*out[:carries], *[value + 1 if isinstance(value, int) and value >= 0 else value
                              for value in out[carries:]]]


def _while(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    params = eqn.params
    cond_consts, body_consts = params["cond_nconsts"], params["body_nconsts"]
    out = _fixed(walk, params["body_jaxpr"].jaxpr, list(given[cond_consts:cond_consts + body_consts]),
                 list(given[cond_consts + body_consts:]), [])
    cond = walk.run(params["cond_jaxpr"].jaxpr, [*given[:cond_consts], *out])
    if any(isinstance(value, int) for value in cond):
        raise SequenceMixed("a loop's condition reads the tokens")
    return out


def _cond(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    if isinstance(given[0], int):
        raise SequenceMixed("a branch is chosen by the tokens")
    branches = [walk.run(branch.jaxpr, given[1:]) for branch in eqn.params["branches"]]
    merged = []
    for values in zip(*branches, strict=True):
        reached = {value for value in values if value is not None}
        if len(reached) > 1:
            raise SequenceMixed("a branch's outputs hold their positions on different axes")
        merged.append(reached.pop() if reached else None)
    return merged


def _scatter(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    if any(isinstance(value, int) for value in given):
        raise SequenceMixed(f"{eqn.primitive.name} writes values the tokens reach at indices the walk does "
                            "not hold to their positions")
    return [None]


def _dynamic_slice(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    if any(isinstance(value, int) for value in given[1:]):
        raise SequenceMixed(f"{eqn.primitive.name} places by indices the tokens reach")
    return _sliced(walk, eqn, given[:1] + [None] * (len(given) - 1))


def _split(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    axis = _single(given)
    if isinstance(axis, int) and axis >= 0 and axis == eqn.params["axis"]:
        raise SequenceMixed("split cuts the sequence axis")
    return [axis] * len(eqn.outvars)


def _each(walk: _Walk, eqn: core.JaxprEqn, given: list[Held]) -> list[Held]:
    """An operation passing each operand through as it is (`optimization_barrier`)."""
    return list(given)


_MIXING = frozenset({
    "reduce_window", "reduce_window_sum", "reduce_window_max", "reduce_window_min", "select_and_scatter_add",
    "select_and_gather_add", "fft", "triangular_solve", "cholesky", "lu", "qr", "eigh", "svd",
    "all_gather", "all_to_all", "ppermute", "psum", "pmax", "pmin", "rng_bit_generator"})
"""Operations with no rule of their own that read their inputs across an axis: refused where the tokens
reach them."""

type _Rule = Callable[[_Walk, core.JaxprEqn, list[Held]], list[Held]]

_RULES: Mapping[str, _Rule] = {
    **dict.fromkeys(("pjit", "jit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call",
                     "custom_vjp_call_jaxpr", "remat", "checkpoint", "custom_lin"), _call),
    "broadcast_in_dim": _broadcast_in_dim, "reshape": _reshape, "squeeze": _squeeze, "transpose": _transpose,
    **dict.fromkeys(("reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "reduce_and", "reduce_or",
                     "reduce_xor", "argmax", "argmin"), _reduce),
    **dict.fromkeys(("cumsum", "cumprod", "cummax", "cummin", "cumlogsumexp"), _cumulative),
    "dot_general": _dot_general, "conv_general_dilated": _conv, "gather": _gather, "sort": _sort,
    "top_k": _top_k, **dict.fromkeys(("slice", "pad", "rev", "concatenate"), _sliced),
    **dict.fromkeys(("dynamic_slice", "dynamic_update_slice"), _dynamic_slice),
    **dict.fromkeys(("scatter", "scatter-add", "scatter_add", "scatter-mul", "scatter_mul", "scatter-min",
                     "scatter_min", "scatter-max", "scatter_max"), _scatter),
    "scan": _scan, "while": _while, "cond": _cond, "split": _split, "optimization_barrier": _each,
}
