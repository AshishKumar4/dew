"""Template forms derived from a declaration (docs/design/algebra.md §6.9): a reduction's math stated on
whole operands becomes the state a kernel carries along its walked axis, that state's empty value, one tile's
step into it, the merge of two states (a split walk's, the work scheduler's fix-up) and the outputs a finished
state gives.

The derivation reads the declaration's jaxpr. A reduction along the walked axis is a monoid's: a maximum's
(`reduce_max`) or a sum's (`reduce_sum`, and a `dot_general` contracting the walked axis, a sum of products).
A sum whose terms read a maximum along the same axis only through `exp(x - m)` (or `exp2`) is decoupled from
it (Neptune's correction terms): for any `m'` along the walk, `sum exp(x - m) f = exp(m' - m) sum exp(x - m')
f`, so a tile's sum is taken at the running maximum and the running sum rescaled when the maximum grows, which
is online softmax and its split-walk merge, derived rather than written. Every other dependence of a walked
reduction on another is refused, named. The decoupling reorders rounding (`reorders`): a template whose
numerics admit only exact forms keeps its hand-written ones.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core as jcore

from dew.algebra.forms import Derived, Program, Refused, aval, environment, evaluate, read_by

Axes = tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Reduction:
    """A walk whose state is monoids' (attention): `function(**operands)` on whole operands in plain JAX,
    each operand's axes by name (`operands`), the axis the kernel walks in tiles (`walked`), and the
    exponent's base the template takes (`exponent`: 2 where it folds `log2(e)` into its logits, `"e"`
    otherwise), which the derivation keeps."""

    function: Callable
    operands: Mapping[str, Axes]
    walked: str
    exponent: Literal[2, "e"]


@dataclasses.dataclass(frozen=True)
class _Walked:
    """One reduction along the walked axis: its equation, its monoid (`max` or `sum`), the walked position
    of the axis it reduces in its input, and, for a sum, the maximum it is decoupled from (its index among
    the walk's reductions) and whether its exponent is base 2."""

    equation: jcore.JaxprEqn
    monoid: str
    decoupled: int | None
    base2: bool


def derive(declaration: Reduction) -> Derived:
    """`declaration`'s monoid forms along its walked axis (the module's account); refused (`Refused`) where a
    walked sum reads a walked reduction other than through a decoupled exponent."""
    names = tuple(declaration.operands)
    walked = {name: axes.index(declaration.walked) for name, axes in declaration.operands.items()
              if declaration.walked in axes}

    def traced(shapes: Mapping[str, tuple[tuple[int, ...], jnp.dtype]]):
        # Traced with jit disabled, so a jitted library function's equations (jnp.where's, matmul's) are
        # the declaration's own, not a nested program.
        avals = [jax.ShapeDtypeStruct(*shapes[name]) for name in names]
        def named(*values):
            return declaration.function(**dict(zip(names, values, strict=True)))

        with jax.disable_jit():
            return jax.make_jaxpr(named)(*avals)

    kinds: list = []  # each reduction's (monoid, decoupled maximum, base 2), shape-independent

    def analysed(shapes):
        closed = traced(shapes)
        positions = _walked_positions(closed.jaxpr, [walked.get(name) for name in names])
        reductions = _reductions(closed.jaxpr, positions)
        reduced = _reading(closed.jaxpr, {reduction.equation.outvars[0] for reduction in reductions})
        if any(isinstance(var, jcore.Var) and var in positions and var in reduced
               for var in closed.jaxpr.outvars):
            raise Refused("an output along the walked axis reads a walked reduction: no tile holds it")
        kinds[:] = [(reduction.monoid, reduction.decoupled, reduction.base2) for reduction in reductions]
        return closed, positions, reductions

    def shaped(values):
        return {name: (value.shape, value.dtype) for name, value in values.items()}

    def init(shapes):
        _, _, reductions = analysed(shapes)
        return tuple(_empty(reduction) for reduction in reductions)

    def step(state, tiles):
        closed, _, reductions = analysed(shaped(tiles))
        return _stepped(closed, reductions, state, [tiles[name] for name in names])

    def merge(a, b):
        if not kinds:
            raise Refused("a merge before the walk's state was formed (`init`)")
        return tuple(_merged(index, a, b, kind) for index, kind in enumerate(kinds))

    def finish(state, operands):
        closed, positions, reductions = analysed(shaped(operands))
        return _finished(closed, positions, reductions, state, [operands[name] for name in names])

    def mapped(tiles):
        closed, positions, _ = analysed(shaped(tiles))
        env = environment(closed, [tiles[name] for name in names], {})
        wanted = {var for var in closed.jaxpr.outvars if isinstance(var, jcore.Var) and var in positions}
        evaluate(closed, env, needed=read_by(closed.jaxpr, wanted))
        chosen = [env[var] if isinstance(var, jcore.Var) and var in wanted else None
                  for var in closed.jaxpr.outvars]
        return chosen[0] if len(chosen) == 1 else tuple(chosen)

    return Derived(init, step, merge, finish, mapped,
                   {"rounding": "reorders", "walked": declaration.walked, "exponent": declaration.exponent})


def _reading(jaxpr: jcore.Jaxpr, sources: set) -> set:
    """The values that read any of `sources`, themselves included."""
    reading = set(sources)
    for equation in jaxpr.eqns:
        if any(isinstance(var, jcore.Var) and var in reading for var in equation.invars):
            reading.update(equation.outvars)
    return reading


def _walked_positions(jaxpr: jcore.Jaxpr, inputs: Sequence[int | None]) -> dict:
    """Each value's position of the walked axis (absent where it does not vary along the walk): the inputs'
    given, carried through elementwise equations, broadcasts and layout changes that keep the axis, and
    lost where a reduction or a contraction takes it."""
    positions = {var: position for var, position in zip(jaxpr.invars, inputs, strict=True)
                 if position is not None}
    producers = {out: equation for equation in jaxpr.eqns for out in equation.outvars}
    for equation in jaxpr.eqns:
        name = equation.primitive.name
        if (branch := _platform_branch(equation, producers)) is not None:
            # A product chosen by the platform it lowers for (Dew's wide CPU sums): its default branch's.
            inner = _walked_positions(branch.jaxpr, _given(positions, equation.invars[1:]))
            for out, branch_out in zip(equation.outvars, branch.jaxpr.outvars, strict=True):
                if isinstance(branch_out, jcore.Var) and branch_out in inner:
                    positions[out] = inner[branch_out]
            continue
        if any(_program(value) or (isinstance(value, tuple) and any(_program(program) for program in value))
               for value in equation.params.values()):
            raise Refused(f"{name}: a nested program, whose equations the derivation does not read")
        found = [positions[var] for var in equation.invars if isinstance(var, jcore.Var) and var in positions]
        if not found:
            continue
        if name in ("reduce_max", "reduce_sum", "reduce_min", "argmax", "argmin"):
            # A reduction over the walked axis is the walk's (`_reductions`); over others, the walked axis
            # moves down past each axis reduced before it.
            axes = equation.params["axes"]
            if found[0] not in axes:
                positions[equation.outvars[0]] = found[0] - sum(axis < found[0] for axis in axes)
            continue
        if name == "dot_general":
            _contracted_away(equation, positions)
            continue
        if name == "broadcast_in_dim":
            position = equation.params["broadcast_dimensions"][found[0]]
        elif name == "transpose":
            position = list(equation.params["permutation"]).index(found[0])
        elif len(set(found)) == 1 and name not in ("reshape", "squeeze", "expand_dims"):
            position = found[0]
        else:
            raise Refused(f"the walked axis through {name} is not read here")
        for out in equation.outvars:
            positions[out] = position
    return positions


def _contracted_away(equation: jcore.JaxprEqn, positions: dict) -> None:
    """A contraction's output's walked axis: none where it contracts the walked axis (a walked sum), else the
    walked operand's free axis in the output's order (batch axes, then each operand's free ones)."""
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = equation.params["dimension_numbers"]
    lhs, rhs = equation.invars
    walked_lhs, walked_rhs = positions.get(lhs), positions.get(rhs)
    if walked_lhs in lhs_contract or walked_rhs in rhs_contract:
        return
    lhs_free = [axis for axis in range(_rank(lhs)) if axis not in lhs_contract and axis not in lhs_batch]
    rhs_free = [axis for axis in range(_rank(rhs)) if axis not in rhs_contract and axis not in rhs_batch]
    if walked_lhs is not None:
        position = (list(lhs_batch).index(walked_lhs) if walked_lhs in lhs_batch
                    else len(lhs_batch) + lhs_free.index(walked_lhs))
    elif walked_rhs is not None:
        position = (list(rhs_batch).index(walked_rhs) if walked_rhs in rhs_batch
                    else len(lhs_batch) + len(lhs_free) + rhs_free.index(walked_rhs))
    else:
        return
    positions[equation.outvars[0]] = position


def _program(value) -> bool:
    """Whether `value` is a nested program: a jaxpr, or a closed one (a jaxpr with its constants, its type
    JAX's own `ClosedJaxpr`, which `jax.extend` does not name)."""
    return isinstance(value, (jcore.Jaxpr, _CLOSED))


_CLOSED = type(jax.make_jaxpr(lambda: None)())
"""JAX's closed jaxpr's type, as `jax.make_jaxpr` returns it."""


def _given(positions: Mapping, atoms) -> list:
    """Each of `atoms`' walked position, None where it has none (a literal's, or a value's along no walk)."""
    return [positions.get(atom) if isinstance(atom, jcore.Var) else None for atom in atoms]


def _chosen_reduction(branch: Program, given: list) -> str | None:
    """The monoid of a platform choice's one walked reduction (its default branch's, whose value the choice
    gives): None where it reduces nothing along the walk; refused where it reduces more than once."""
    inner = _reductions(branch.jaxpr, _walked_positions(branch.jaxpr, given))
    if not inner:
        return None
    if len(inner) != 1 or inner[0].decoupled is not None:
        raise Refused("a platform's choice of several walked reductions, or of a decoupled one")
    return inner[0].monoid


def _rank(atom) -> int:
    return len(aval(atom).shape)


def _platform_branch(equation: jcore.JaxprEqn, producers: Mapping):
    """The default branch of a choice by the platform a program lowers for (`lax.platform_dependent`: a
    `cond` on a `platform_index`), whose branches compute one value each their own way (Dew's products sum
    in float64 on a CPU under its wide arithmetic, `dew.nn.precision.accumulated`); None for any other
    equation. Its structure is read from the default branch, and the equation itself evaluated, so each
    platform keeps its own arithmetic."""
    branches = equation.params.get("branches") if equation.primitive.name == "cond" else None
    if not branches:
        return None
    chooser = producers.get(equation.invars[0])
    return branches[-1] if chooser is not None and chooser.primitive.name == "platform_index" else None


def _reductions(jaxpr: jcore.Jaxpr, positions: dict) -> list[_Walked]:
    """The walk's reductions in program order, each sum's decoupled maximum found (`_Homogeneity`)."""
    found: list[_Walked] = []
    maxima: dict = {}
    sums: set = set()
    producers = {out: equation for equation in jaxpr.eqns for out in equation.outvars}
    for equation in jaxpr.eqns:
        name = equation.primitive.name
        if name in ("reduce_max", "reduce_sum"):
            (operand,) = equation.invars
            if positions.get(operand) not in equation.params["axes"]:
                continue
            if len(equation.params["axes"]) != 1:
                raise Refused(f"{name} over the walked axis and others at once")
            monoid = "max" if name == "reduce_max" else "sum"
        elif name == "dot_general":
            (lhs_contract, rhs_contract), _ = equation.params["dimension_numbers"]
            lhs, rhs = equation.invars
            if positions.get(lhs) not in lhs_contract and positions.get(rhs) not in rhs_contract:
                continue
            monoid = "sum"
        elif (branch := _platform_branch(equation, producers)) is not None:
            monoid = _chosen_reduction(branch, _given(positions, equation.invars[1:]))
            if monoid is None:
                continue
        else:
            continue
        decoupled, base2 = None, False
        classified = _Homogeneity(maxima, sums, producers)
        kinds = [classified(operand) for operand in equation.invars
                 if _platform_branch(equation, producers) is None or operand is not equation.invars[0]]
        kind = _Homogeneity.product(kinds, name) if name in ("dot_general", "cond") else kinds[0]
        if monoid == "max":
            if kind is not None:
                raise Refused("a walked maximum of values a walked maximum scales")
            maxima[equation.outvars[0]] = len(found)
        else:
            if kind is not None:
                decoupled, base2 = kind[0], kind[1]
            sums.add(equation.outvars[0])
        found.append(_Walked(equation, monoid, decoupled, base2))
    return found


_PASSING = frozenset({"broadcast_in_dim", "convert_element_type", "reshape", "expand_dims", "squeeze"})
"""The equations a maximum or a zero passes through unchanged in value: a broadcast, a cast, a reshape."""


class _Homogeneity:
    """Each value's dependence on the walk's maxima (None, free of every one, or `(maximum, base2)`: it scales
    by `exp(m - m')` where its maximum `m` is read as `m'`), refused where a value reads a maximum otherwise:
    a maximum is read only as `exp(x - m)` or `exp2(x - m)` with `x` free of it; a product or a quotient by a
    free value, a sum of two values of the same maximum, a selection among them or a zero, and a change of
    layout or type keep it; every other operation's operands are free."""

    LAYOUT = frozenset({"convert_element_type", "broadcast_in_dim", "transpose", "reshape", "squeeze",
                        "expand_dims", "neg", "reduce_precision"})

    def __init__(self, maxima: dict, sums: set, producers: dict):
        self.maxima, self.sums, self.producers, self.seen = maxima, sums, producers, {}

    def __call__(self, atom):
        if not isinstance(atom, jcore.Var):
            return None
        if atom in self.seen:
            return self.seen[atom]
        if atom in self.sums:
            raise Refused("a walked reduction reads a walked sum: no monoid of the walk holds it")
        if atom in self.maxima:
            raise Refused("a walked sum reads a walked maximum other than through exp(x - m)")
        equation = self.producers.get(atom)
        kind = None if equation is None else self._kind(equation)
        self.seen[atom] = kind
        return kind

    def _kind(self, equation: jcore.JaxprEqn):
        name = equation.primitive.name
        if name in ("exp", "exp2"):
            shifted = self._shifted(equation.invars[0], base2=name == "exp2")
            if shifted is not None:
                return shifted
        if name in self.LAYOUT:
            return self(equation.invars[0])
        chosen = _platform_branch(equation, self.producers) is not None
        kinds = [self(operand) for operand in (equation.invars[1:] if chosen else equation.invars)]
        if all(kind is None for kind in kinds):
            return None
        if name in ("mul", "dot_general") or chosen:
            # A platform's choice is one of Dew's products, a product of its operands each way.
            return self.product(kinds, name)
        if name == "div":
            if kinds[1] is not None:
                raise Refused("a walked sum divides by a value scaled by a walked maximum")
            return kinds[0]
        if name in ("add", "sub"):
            if kinds[0] != kinds[1]:
                raise Refused(f"a walked sum's terms {name} values of different walked maxima")
            return kinds[0]
        if name == "select_n":
            if kinds[0] is not None:
                raise Refused("a walked sum selects by a value scaled by a walked maximum")
            cases = {kind for operand, kind in zip(equation.invars[1:], kinds[1:], strict=True)
                     if not _zero(operand, self.producers)}
            if len(cases) != 1:
                raise Refused("a walked sum selects among values of different walked maxima")
            return next(iter(cases))
        raise Refused(f"a walked sum reads a walked maximum through {name}")

    def _shifted(self, operand, *, base2: bool):
        """`(maximum, base2)` where `operand` is `x - m`, `m` a walked maximum or a safe shift of one
        (`_maximum_of`) and `x` free of every maximum; None where it is not of that form."""
        equation = self.producers.get(operand) if isinstance(operand, jcore.Var) else None
        if equation is None or equation.primitive.name != "sub":
            return None
        minuend, subtrahend = equation.invars
        maximum = self._maximum_of(subtrahend)
        if maximum is None or self(minuend) is not None:
            return None
        return (self.maxima[maximum], base2)

    def _maximum_of(self, atom):
        """The walked maximum `atom` is, through value-preserving equations (`_PASSING`), or is a safe shift
        of: a selection of it against values free of every maximum, by a predicate that reads no other
        (`where(m == -inf, 0, m)`, a row that has seen nothing shifted by zero, whose terms are zero
        either way); None otherwise. The step evaluates the selection at the running maximum, so a tile's
        terms are taken at the shift the declaration states."""
        while isinstance(atom, jcore.Var) and atom not in self.maxima:
            producer = self.producers.get(atom)
            if producer is None:
                return None
            if producer.primitive.name in _PASSING:
                atom = producer.invars[0]
                continue
            if producer.primitive.name != "select_n":
                return None
            predicate, *cases = producer.invars
            found = [(case, self._maximum_of(case)) for case in cases]
            leading = {maximum for _, maximum in found if maximum is not None}
            if len(leading) != 1:
                return None
            (maximum,) = leading
            if any(maximum_of is None and self(case) is not None for case, maximum_of in found):
                return None
            return maximum if self._reads_at_most(predicate, maximum) else None
        return atom if isinstance(atom, jcore.Var) else None

    def _reads_at_most(self, atom, maximum) -> bool:
        """Whether `atom` reads no walked maximum but `maximum`, and no walked sum."""
        if not isinstance(atom, jcore.Var):
            return True
        if atom in self.maxima:
            return atom is maximum
        if atom in self.sums:
            return False
        producer = self.producers.get(atom)
        return producer is None or all(self._reads_at_most(operand, maximum) for operand in producer.invars)

    @staticmethod
    def product(kinds, name: str):
        scaled = [kind for kind in kinds if kind is not None]
        if len(scaled) > 1:
            raise Refused(f"a walked sum's {name} of two values scaled by walked maxima")
        return scaled[0] if scaled else None


def _zero(atom, producers: dict) -> bool:
    """Whether `atom` is a literal zero, or a broadcast or cast of one."""
    while isinstance(atom, jcore.Var):
        producer = producers.get(atom)
        if producer is None or producer.primitive.name not in _PASSING:
            return False
        atom = producer.invars[0]
    return isinstance(atom, jcore.Literal) and float(atom.val) == 0


def _empty(reduction: _Walked):
    """A reduction's empty partial: a maximum's -inf, a sum's zero, in its output's shape and dtype."""
    typed = aval(reduction.equation.outvars[0])
    return jnp.full(typed.shape, -jnp.inf if reduction.monoid == "max" else 0, typed.dtype)


def _shift(base2: bool):
    return jnp.exp2 if base2 else jnp.exp


def _rescale(old_max, new_max, base2: bool, shape):
    """The factor a sum taken at `old_max` takes to `new_max`, zero where nothing was seen (both -inf)."""
    safe = jnp.where(jnp.isneginf(new_max), 0, new_max)
    factor = _shift(base2)(jnp.where(jnp.isneginf(old_max), -jnp.inf, old_max - safe))
    return lax.broadcast_in_dim(factor, shape, tuple(range(factor.ndim))) if factor.shape != shape else factor


def _stepped(closed, reductions, state, values):
    """The state after one tile: each maximum the larger of its running value and the tile's; each sum its
    running value rescaled to its maximum's new value (decoupled) plus the tile's sum, its terms read at the
    new maximum."""
    current = {}
    for index, reduction in enumerate(reductions):
        if reduction.monoid == "max":
            tile = _evaluated(closed, reduction.equation, values, current)
            current[reduction.equation.outvars[0]] = jnp.maximum(state[index], tile)
    stepped = []
    for index, reduction in enumerate(reductions):
        if reduction.monoid == "max":
            stepped.append(current[reduction.equation.outvars[0]])
            continue
        running = state[index]
        if reduction.decoupled is not None:
            grown = current[reductions[reduction.decoupled].equation.outvars[0]]
            running = running * _rescale(state[reduction.decoupled], grown, reduction.base2, running.shape)
        stepped.append(running + _evaluated(closed, reduction.equation, values, current))
    return tuple(stepped)


def _merged(index: int, a, b, kind):
    """Two states' entry `index` joined: a maximum's the larger; a sum's each rescaled to the joined maximum
    of the one it is decoupled from."""
    monoid, decoupled, base2 = kind
    if monoid == "max":
        return jnp.maximum(a[index], b[index])
    left, right = a[index], b[index]
    if decoupled is not None:
        joined = jnp.maximum(a[decoupled], b[decoupled])
        left = left * _rescale(a[decoupled], joined, base2, left.shape)
        right = right * _rescale(b[decoupled], joined, base2, right.shape)
    return left + right


def _evaluated(closed, target: jcore.JaxprEqn, values, known: dict):
    """`target`'s output on one tile: the declaration's equations up to it evaluated on `values`, each walked
    reduction's output already `known` (a maximum's running value) read in place of its own."""
    env = environment(closed, values, known)

    def read(atom):
        return atom.val if isinstance(atom, jcore.Literal) else env[atom]

    for equation in closed.jaxpr.eqns:
        if all(out in env for out in equation.outvars):
            if equation is target:
                break
            continue
        outs = equation.primitive.bind(*map(read, equation.invars), **equation.params)
        outs = outs if equation.primitive.multiple_results else [outs]
        if equation is target:
            return outs[0]
        env.update(zip(equation.outvars, outs, strict=True))
    raise Refused("a walked reduction the evaluation never reached")


def _finished(closed, positions, reductions, state, values):
    """The declaration's outputs from a finished state: its equations evaluated with each walked reduction's
    output its state's value, on the whole operands for every value the walk does not reduce."""
    env = environment(closed, values, {reduction.equation.outvars[0]: state[index]
                                        for index, reduction in enumerate(reductions)})

    def read(atom):
        return atom.val if isinstance(atom, jcore.Literal) else env[atom]

    for equation in closed.jaxpr.eqns:
        if all(out in env for out in equation.outvars):
            continue
        if any(isinstance(var, jcore.Var) and var in positions for var in equation.outvars):
            continue
        outs = equation.primitive.bind(*map(read, equation.invars), **equation.params)
        outs = outs if equation.primitive.multiple_results else [outs]
        env.update(zip(equation.outvars, outs, strict=True))
    outputs = [None if isinstance(var, jcore.Var) and var in positions else read(var)
               for var in closed.jaxpr.outvars]
    return outputs[0] if len(outputs) == 1 else tuple(outputs)
