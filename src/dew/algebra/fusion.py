"""Fusion search (docs/design/algebra.md §2 "Fusion"): a program's products fused with their elementwise
producers and consumers into the GEMM template's prologue and epilogue, where the template takes them and
fusing is the cheaper.

The compile-time pass (`program._rewritten`) hands each body here after its algebra: the compile point at
which a plain JAX program gets the template (row 3450). Each matrix product of the body the template takes
(`Form`: a `dot_general` of rank-2 operands, x @ W, dy @ Wᵀ or a kernel gradient's xᵀ @ dy; a
`ragged_dot_general` of each group's rows, an MoE's; floating operands, no batch) and the elementwise
equations that follow it, up to the first equation outside them that reads one of their values, with the
broadcasts only they read of their captured operands (a bias's), are one region (`_regions`), a product
alone among them: its inputs the values it reads from before it, its outputs the values read after it. A
product whose left operand is its rows (x @ W, dy @ Wᵀ, a group's rows) takes as its prologue the elementwise
equations of the operand's shape that make it, back to a value of another shape or another kind (an RMSNorm's
row statistic, a reduction, stays outside, a captured input; its normalization, x · inv · scale and its cast,
the prologue), with the broadcasts only they read: replayed in the region, and removed from the program where
nothing else reads them (q, k and v projections of one normalized row each replay it, and the row it made
then goes unread). A
region is substituted where its launch is the template's, its selector's choice (`grouped_matmul.DECLARED`,
or a call's, `generation.chosen`), and where it has an epilogue, where the template takes it and its price
says fused wins (`_takes`): its equations replaced by one nested program of `fused_gemm.fused` over the
region as the program wrote it, its product the template's (`_product`: `fused_gemm.product` at the dot's
own precision and result dtype, so the fused region computes what the region did; a kernel gradient's
`gemm.gemm_t`, its consumers after it in XLA), written where its first reader stood (`_spliced`). Every
other region, and every body with none, is kept as traced: the body itself, so an unchanged body keeps its
identity (`program._within`'s). A region's decision is held by its structure
and the context it was weighed in (`_decided`), so every layer of a stack, and every retrace, weighs it once.
A softmax attention, RMSNorm or SwiGLU is written as Dew's call where the e-graph proves the region computes
it and a kernel of Dew's takes it (`recognize`), through the same splice (`_spliced`): attention first, so a
product inside it is none of the products' regions, the norms after the products, so an epilogue's norm is
the product's.

A region's written program is Dew's call traced over the region's inputs' types (`_traced`): a program the
step's own trace does not hold (the template's launch, Dew's attention or norm), which only a trace of that
call makes. It is traced once a process for the region's identity (`Written.key`: its structure, the values
of the constants its call reads, and which call writes it) in the context a kernel is chosen in
(`choice_context`: the derivation's context and `options.held_context`), and written
into every later lowering of a program holding it: a step's
relowerings, its planner's candidates and every layer of a stack share one trace, which reports its kernels'
routes and launches where each program is written as it did when it traced.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Hashable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax._src import core as jax_core
from jax.core import ShapedArray
from jax.extend import core as jcore

from dew import _cache
from dew.nn import derivatives
from dew.nn.kernels import generation
from dew.nn.nested import nested
from dew.nn.programs import RaggedLayout, entrywise, ragged_layout

if TYPE_CHECKING:
    from dew.nn.kernels import tuning

CAPTURED = (jax.lax.broadcast_in_dim_p, jax.lax.reshape_p)
"""The layout equations a region takes of a captured operand only it reads (a bias broadcast over rows), so
the epilogue captures the operand itself."""

DECISIONS_HELD = 256
"""How many regions' decisions are held (`_decided`), and their written programs (`_traced`), the least
recently weighed dropped past them: a model's distinct projections, each shape once."""


class Region(NamedTuple):
    """A product, the elementwise equations making its rows (its prologue) and its elementwise consumers in
    a body (`_regions`): the places of the equations the fused region replaces (`places`, in the body's
    order: its prologue's equations nothing outside it reads, the product, its consumers), those of its
    prologue (`prologue`, each replayed in the region, the ones something else reads kept in the program
    too), the values they read from before them (`inputs`), the values read after them (`outputs`), and
    the place of the first equation after them that reads one (where the fused region is written)."""

    places: tuple[int, ...]
    inputs: tuple[jcore.Var, ...]
    outputs: tuple[jcore.Var, ...]
    at: int
    prologue: tuple[int, ...] = ()

    @property
    def replayed(self) -> tuple[int, ...]:
        """The places the fused region computes, in the body's order: its prologue's and its own."""
        return tuple(sorted({*self.prologue, *self.places}))


def fused_program(jaxpr: jcore.Jaxpr) -> jcore.Jaxpr:
    """`jaxpr` with each attention region a kernel of Dew's takes written as Dew's attention, each product
    region the template takes fused (the module's account), and each RMSNorm and SwiGLU region the fused
    norms take written as Dew's, every call by `recognize`; `jaxpr` itself where none is."""
    from dew.algebra import recognize
    from dew.nn.kernels import fused_gemm

    index = Index(jaxpr)
    attention = recognize.recognized(jaxpr, index, recognize.ATTENTION)
    held = {place for region in attention for place in region.places}
    identities = {}
    for found in _regions(index):
        if held & set(found.places):
            continue
        taken = _taken(index, found, lambda region: _decided(
            _identity(jaxpr, region), lambda: _takes(fused_gemm, jaxpr, region)))
        if taken is not None:
            identities[taken] = _identity(jaxpr, taken)
    products = list(identities)
    # A norm a product's prologue replays is the product's (`_prologue`), as an epilogue's norm is.
    held |= {place for region in products for place in region.replayed}
    written = attention + recognize.recognized(jaxpr, index, recognize.NORMS, frozenset(held))
    if not products and not written:
        return jaxpr
    written += [Written(region.places, region.inputs, region.outputs, region.at,
                        _fusing(_replayed(jaxpr, region, fused_gemm), fused_gemm),
                        ("product", identities[region])) for region in products]
    return _spliced(jaxpr, written)


def _taken(index: Index, found: Region, decided: Callable[[Region], RegionDecision]) -> Region | None:
    """`found`, or where a stage keeps it from fusing (a prologue or an epilogue the template refuses, or one
    that does not pay), the region with the most stages of fewer the template takes (`_fewer`, each
    `decided` once for its structure), its product the template's where it can be and the stages it drops
    left as the program wrote them, XLA's beside the launch; None where none is taken (row 8996: a refused
    epilogue left the product a plain dot)."""
    level, seen = [found], set()
    while level:
        fewer = []
        for region in level:
            decision = decided(region)
            if decision.taken:
                return region
            fewer += _fewer(index, region, decision)
        # One region each, a product alone reached without its prologue and without its epilogue alike.
        level = []
        for region in fewer:
            if (region.places, region.prologue) not in seen:
                seen.add((region.places, region.prologue))
                level.append(region)
    return None


def _fewer(index: Index, region: Region, decision: RegionDecision) -> list[Region]:
    """`region` with one stage fewer, as its `decision` allows each: without its prologue (its product and
    its epilogue, reading the product's rows as given) where its launch is XLA's for no reason a prologue
    enters (`RegionDecision.launchable`); without its epilogue (its prologue and its product, before the
    first of its consumers, which read the product's value as the program wrote them) where its launch is
    the template's or undecided (`RegionDecision.launched`: an epilogue keys no launch's cell). Each where the
    region has the stage it drops."""
    jaxpr = index.jaxpr
    (product,) = (place for place in region.places if _form(jaxpr.eqns[place]) is not None)
    fewer = []
    if region.prologue and decision.launchable:
        places = tuple(place for place in region.places if place not in region.prologue)
        fewer.append(Region(places, region_inputs(jaxpr, places), region.outputs, region.at))
    if decision.launched and any(place > product for place in region.places):
        made = tuple(jaxpr.eqns[product].outvars)
        # The prologue's equations it replaces read again: one a consumer it leaves reads stays.
        computed = {*region.prologue, product}
        places = tuple(sorted({*_alone(index, region.prologue, computed), product}))
        first = min(place for var in made for place in index.readers[var])
        fewer.append(Region(places, region_inputs(jaxpr, sorted(computed)), made, first, region.prologue))
    return fewer


def region_inputs(jaxpr: jcore.Jaxpr, places: Sequence[int]) -> tuple[jcore.Var, ...]:
    """The values the equations at `places` read that none of them makes, in the order they are read."""
    made = {var for place in places for var in jaxpr.eqns[place].outvars}
    return tuple(dict.fromkeys(var for place in places for var in jaxpr.eqns[place].invars
                               if isinstance(var, jcore.Var) and var not in made))


def _fusing(replay: Callable, fused_gemm) -> Callable:
    """A product region's call: `fused_gemm.fused` over the region as the program wrote it (`_replayed`)."""
    return lambda *inputs: fused_gemm.fused(replay, *inputs)


class Index:
    """A body's equations by the values they make (`producer`), each value's readers (`readers`: the places
    of the equations that read it), the values the body returns (`escaping`) and its constants' values where
    it holds them (`consts`): what the region finders read (`_regions`, `recognize.recognized`); and what
    recognition finds once of the body, held as long as the index lives: its key (`key`), its values made
    from constants alone (`constants`) and its walks from a value by direction and reach (`walks`)."""

    def __init__(self, jaxpr: jcore.Jaxpr):
        self.jaxpr = jaxpr
        held = len(jaxpr.consts) == len(jaxpr.constvars)
        self.consts = dict(zip(jaxpr.constvars, jaxpr.consts, strict=True)) if held else {}
        self.producer = {var: place for place, equation in enumerate(jaxpr.eqns) for var in equation.outvars}
        self.readers: dict[jcore.Var, set[int]] = collections.defaultdict(set)
        for place, equation in enumerate(jaxpr.eqns):
            for var in equation.invars:
                if isinstance(var, jcore.Var):
                    self.readers[var].add(place)
        self.escaping = {var for var in jaxpr.outvars if isinstance(var, jcore.Var)}
        self.key: Hashable | None = None
        self.constants: frozenset[jcore.Var] | None = None
        self.walks: dict[tuple[str, jcore.Var, int], dict[jcore.Var, int]] = {}


class Written(NamedTuple):
    """A region's equations (their places) replaced by one nested program of `call` over its `inputs`, making
    its `outputs`, written at place `at` (`_spliced`); `key` its identity, which its written program is held
    by (`_traced`): every value `call`'s trace reads of the region, its structure and the values of the
    constants it closes over, and which call it is."""

    places: tuple[int, ...]
    inputs: tuple[jcore.Var, ...]
    outputs: tuple[jcore.Var, ...]
    at: int
    call: Callable
    key: Hashable


class Form(NamedTuple):
    """A product the template takes (`_form`): its right operand held columns by depth (`transpose_rhs`),
    its left one contracted over its rows (`transpose_lhs`: a kernel gradient's xᵀ @ dy), and its groups
    (`grouped`: a `ragged_dot_general`'s, its group sizes its third operand)."""

    transpose_rhs: bool
    transpose_lhs: bool
    grouped: bool


FORWARD = Form(transpose_rhs=False, transpose_lhs=False, grouped=False)
"""x @ W: rows by depth times depth by columns."""

DENSE_FORMS = {((1,), (0,)): FORWARD, ((1,), (1,)): FORWARD._replace(transpose_rhs=True),
               ((0,), (0,)): FORWARD._replace(transpose_lhs=True)}
"""A rank-2 `dot_general`'s contracted axes, no batch axes, to its form: x @ W, dy @ Wᵀ and xᵀ @ dy."""

GROUPED_FORMS = {RaggedLayout.ROWS: FORWARD._replace(grouped=True),
                 RaggedLayout.ROWS_TRANSPOSED: FORWARD._replace(transpose_rhs=True, grouped=True),
                 RaggedLayout.GROUPS: FORWARD._replace(transpose_lhs=True, grouped=True)}
"""A `ragged_dot_general`'s layout (`dew.nn.programs.ragged_layout`) to its form: rows times their group's
matrix (`[groups, depth, columns]`, or columns by depth), and the kernel gradient's rows contracted within
each group (`[groups, depth, columns]` out)."""


def _form(equation: jcore.JaxprEqn) -> Form | None:
    """`equation`'s form as a product the template takes (`Form`): a `dot_general` of rank-2 operands (x @ W,
    dy @ Wᵀ, xᵀ @ dy) or a `ragged_dot_general` of rows ragged by group (`GROUPED_FORMS`), each of floating
    operands, no batch axes; None for any other equation (an integer product is XLA's own,
    `fused_gemm.product`), and of one dtype: the template casts its right operand to its left's
    (`gemm.gemm`), so a product of two dtypes stays JAX's, which multiplies each as it is (a float16 of 2^-8
    by a bfloat16 of 2^16 cast to float16 was infinite: base review 7754e5fde F1)."""
    primitive = equation.primitive
    if primitive not in (jax.lax.dot_general_p, jcore.primitives.ragged_dot_general_p):
        return None
    dtypes = {var.aval.dtype for var in equation.invars[:2] if isinstance(var.aval, ShapedArray)}
    if len(dtypes) != 1 or not all(jnp.issubdtype(dtype, jnp.inexact) for dtype in dtypes):
        return None
    if primitive is jax.lax.dot_general_p:
        (left, right), batch = equation.params["dimension_numbers"]
        if batch != ((), ()) or any(len(_shape(var)) != 2 for var in equation.invars):
            return None
        return DENSE_FORMS.get((tuple(left), tuple(right)))
    layout = ragged_layout(len(_shape(equation.invars[0])), len(_shape(equation.invars[1])),
                           equation.params["ragged_dot_dimension_numbers"])
    return None if layout is None else GROUPED_FORMS[layout]


def _shape(atom) -> tuple[int, ...]:
    """An equation's operand's or result's shape; () for a value of no array's type (a token's)."""
    return tuple(atom.aval.shape) if isinstance(atom.aval, ShapedArray) else ()


def _follows(operand: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    """Whether an operand of `operand`'s shape follows the product's tiles of `shape`: a scalar, or of its
    rank with each axis the product's or one (a bias's `[1, columns]`, a row scale's `[rows, 1]`)."""
    return not operand or (len(operand) == len(shape)
                           and all(axis in (1, extent) for axis, extent in zip(operand, shape, strict=True)))


def _elementwise(equation: jcore.JaxprEqn, shape: tuple[int, ...]) -> bool:
    """Whether `equation` computes each entry of a value of `shape` from the same entry of its operands, each
    following its tiles (`_follows`): a member of a product's prologue or epilogue."""
    return (entrywise(equation.primitive) and not equation.effects
            and all(_shape(var) == shape for var in equation.outvars)
            and all(_follows(_shape(var), shape) for var in equation.invars))


def _named_value(equation: jcore.JaxprEqn, shape: tuple[int, ...]) -> bool:
    """Whether `equation` is a remat name (`checkpoint_name`) on a value of `shape`: an epilogue's member
    (`_regions`), its name lifted off the launch's consumer as a residual (`fused_gemm._named`)."""
    return (equation.primitive is jcore.primitives.name_p
            and all(_shape(var) == shape for var in (*equation.invars, *equation.outvars)))


def _activation(equation: jcore.JaxprEqn, shape: tuple[int, ...]) -> bool:
    """Whether `equation` is a gated product's region (`derivatives.Activation`, `activations.gated_product`'s
    span over its gate and up projections) on values of `shape`: an epilogue's member (`_regions`), replayed
    as the activation's arithmetic on the tile (`activations.gated_hook`), never as its body, which on an
    H100 is SwiGLU's Pallas kernel (a call no epilogue runs)."""
    kind = equation.params.get("kind") if equation.primitive is derivatives.marked_call_p else None
    return (isinstance(kind, derivatives.Activation) and kind.gated and len(equation.invars) == 2
            and all(_shape(var) == shape for var in (*equation.invars, *equation.outvars)))


def _captured(index: Index, scanned: Sequence[int], members: set[int], claimed: set[int]) -> set[int]:
    """The layout equations of the captured operands of the equations at `scanned` that only `members` read
    (`CAPTURED`: a bias's broadcast, a row statistic's over the depth), no other region holds (`claimed`)
    and none escaping the program: a product's epilogue's (`scanned` its consumers, `members` them and the
    product) and its prologue's (`_prologue`)."""
    jaxpr, producer = index.jaxpr, index.producer
    # Each layout equation taken joins the members its later readers are checked against.
    held, found = set(members), set()
    for place in scanned:
        for var in jaxpr.eqns[place].invars:
            if not isinstance(var, jcore.Var):
                continue
            source = producer.get(var)
            if (source is not None and source not in held and source not in claimed
                    and jaxpr.eqns[source].primitive in CAPTURED and index.readers[var] <= held
                    and var not in index.escaping):
                held.add(source)
                found.add(source)
    return found


def _prologue(index: Index, start: int, claimed: set[int]) -> tuple[int, ...]:
    """The places of the product at `start`'s prologue (`_regions`): the elementwise equations of its left
    operand's shape (`_elementwise`) that make it, walked back from it to a value of another shape or kind or
    an equation another region holds (`claimed`), with the layout equations of their captured operands only
    they read (`CAPTURED`: a row statistic broadcast over the depth, a scale over the rows); none for a
    product whose left operand is not its rows (a kernel gradient's xᵀ @ dy)."""
    jaxpr, producer = index.jaxpr, index.producer
    equation = jaxpr.eqns[start]
    form, lhs = _form(equation), equation.invars[0]
    if form is None or form.transpose_lhs or not isinstance(lhs, jcore.Var):
        return ()
    shape, members, frontier = _shape(lhs), set(), [lhs]
    while frontier:
        place = producer.get(frontier.pop())
        if place is None or place in members or place in claimed or not _elementwise(jaxpr.eqns[place],
                                                                                      shape):
            continue
        members.add(place)
        frontier.extend(var for var in jaxpr.eqns[place].invars if isinstance(var, jcore.Var))
    return tuple(sorted(members | _captured(index, sorted(members), members, claimed)))


def _regions(index: Index) -> list[Region]:
    """Each product of `jaxpr` (`_form`), its prologue (`_prologue`) and the elementwise equations after it
    that read its values, each of the product's shape over operands that follow its tiles (`_elementwise`),
    up to the first equation outside them that reads one of their values, with the layout equations of its
    captured operands that no other equation reads (`CAPTURED`). A prologue's equation nothing outside the
    region reads is replaced with it (`Region.places`); one something else reads stays in the program too."""
    jaxpr, escaping = index.jaxpr, index.escaping
    found: list[Region] = []
    claimed: set[int] = set()
    for start, equation in enumerate(jaxpr.eqns):
        if _form(equation) is None or start in claimed:
            continue
        prologue = _prologue(index, start, claimed)
        shape = _shape(equation.outvars[0])
        places, made, at = [start], set(equation.outvars), len(jaxpr.eqns)
        for place in range(start + 1, len(jaxpr.eqns)):
            later = jaxpr.eqns[place]
            if not any(isinstance(var, jcore.Var) and var in made for var in later.invars):
                continue
            # A product's epilogue takes each entry of its output from the same entry of its inputs: an
            # entrywise operation, or a remat name on such a value (`checkpoint_name`, a gate's projection
            # before its SwiGLU), which the launch's consumer hands out under its name (`fused_gemm._named`,
            # its residual, row 7131), where it ended the region (row 6102). A name on a prologue's value
            # stays its boundary: a producer's stage runs no name. A gated product's region (`_activation`) is
            # one too, its other projection a captured operand.
            if place in claimed or not (_elementwise(later, shape) or _named_value(later, shape)
                                        or _activation(later, shape)):
                at = place
                break
            places.append(place)
            made.update(later.outvars)
        captured = _captured(index, places[1:], set(places), claimed)
        made.update(var for place in captured for var in jaxpr.eqns[place].outvars)
        places = sorted({*places, *captured})
        read_after = {var for later in jaxpr.eqns[places[-1] + 1:] for var in later.invars
                      if isinstance(var, jcore.Var)} | escaping
        outputs = tuple(var for place in places for var in jaxpr.eqns[place].outvars if var in read_after)
        computed = {*prologue, *places}
        made.update(var for place in prologue for var in jaxpr.eqns[place].outvars)
        inputs = tuple(dict.fromkeys(var for place in sorted(computed) for var in jaxpr.eqns[place].invars
                                     if isinstance(var, jcore.Var) and var not in made))
        alone = _alone(index, prologue, computed)
        if outputs:
            replaced = tuple(sorted({*places, *alone}))
            found.append(Region(replaced, inputs, outputs, min(at, places[-1] + 1), prologue))
            claimed.update(replaced)
    return found


def _alone(index: Index, prologue: Sequence[int], computed: set[int]) -> tuple[int, ...]:
    """The equations of `prologue` a region computing `computed` replaces (`Region.places`): a prologue's
    equation goes with the region where only the region reads it and nothing kept reads it; one something
    else reads stays, and every one of the prologue it reads with it."""
    jaxpr, readers, escaping, producer = index.jaxpr, index.readers, index.escaping, index.producer
    kept = [place for place in prologue if any(not readers[var] <= computed or var in escaping
                                               for var in jaxpr.eqns[place].outvars)]
    held = set(kept)
    while kept:
        for var in jaxpr.eqns[kept.pop()].invars:
            source = producer.get(var) if isinstance(var, jcore.Var) else None
            if source is not None and source in prologue and source not in held:
                held.add(source)
                kept.append(source)
    return tuple(place for place in prologue if place not in held)


def _replayed(jaxpr: jcore.Jaxpr, region: Region, fused_gemm):
    """The region as the program wrote it, a function of its inputs returning its outputs, its product
    `fused_gemm.product` declared as its dot (its precision, its result's dtype)."""
    def replay(*inputs):
        values = dict(zip(region.inputs, inputs, strict=True))

        def read(atom):
            return atom.val if isinstance(atom, jcore.Literal) else values[atom]

        for place in region.replayed:
            equation = jaxpr.eqns[place]
            operands = [read(atom) for atom in equation.invars]
            form = _form(equation)
            if form is not None:
                # At its traced result's dtype, the dot's own (JAX's rule, not a promotion of its operands').
                (traced,) = _shapes(equation.outvars)
                made = [_product(form, operands, equation.params["precision"], traced.dtype, fused_gemm)]
            elif _activation(equation, _shape(equation.outvars[0])):
                # A gated product's arithmetic as an epilogue traces it, whatever route its span's body took.
                from dew.nn import activations

                made = [activations.gated_hook(equation.params["kind"].name)(*operands)]
            else:
                # As written, a manual mesh's casts among them (`pvary`, its axes the program's): on whole
                # arrays a cast is the program's own, and a kernel's tile reads it as the value it casts (the
                # Pallas fuser's rule for it).
                bound = equation.primitive.bind(*operands, **equation.params)
                made = list(bound) if equation.primitive.multiple_results else [bound]
            values.update(zip(equation.outvars, made, strict=True))
        return tuple(values[var] for var in region.outputs)

    return replay


def _product(form: Form, operands: list, precision, out_dtype, fused_gemm) -> jax.Array:
    """A product of `form` over its equation's `operands` as the template computes it, at the dot's own
    precision and result dtype: `fused_gemm.product` where its consumers may be the launch's epilogue (x @
    W and dy @ Wᵀ, each group's rows times its matrix where grouped), `gemm.gemm_t` for a kernel gradient
    (its consumers after it in XLA), whose dense product is its one group's."""
    from dew.nn.kernels import gemm

    assumed = generation.assumption()
    interpret_on_cpu = assumed is not None and assumed.interpreted
    lhs, rhs, *sizes = operands
    if form.transpose_lhs:
        made = gemm.gemm_t(lhs, rhs, sizes[0] if form.grouped else None, out_dtype=out_dtype,
                           precision=precision, interpret_on_cpu=interpret_on_cpu)
        return made if form.grouped else made[0]
    return fused_gemm.product(lhs, rhs, *sizes, transpose_rhs=form.transpose_rhs, precision=precision,
                              out_dtype=out_dtype, interpret_on_cpu=interpret_on_cpu)


def _shapes(variables: Sequence[jcore.Var]) -> list[jax.ShapeDtypeStruct]:
    """Each of `variables` as its type (each a region's value, an array's, `typed`)."""
    return [typed(var.aval) for var in variables if isinstance(var.aval, ShapedArray)]


def typed(aval: ShapedArray, *, weak: bool = False) -> jax.ShapeDtypeStruct:
    """`aval` as a type a trace takes: its shape and dtype (its weak type with `weak`), and inside a manual
    mesh's body, its variance checked or not, its sharding, its manual axis type (the axes it varies over, a
    reduced or unreduced placement, an outer checked map's an unchecked one inherits) and its weak type, as
    the body holds them, so a region's call or a window's form traced over them binds its inputs as the
    body does (row 6578: binders typed the same on every device beside the body's varying values, which JAX's
    check of the body refused)."""
    if not jax.sharding.get_abstract_mesh().manual_axes:
        return jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=weak and aval.weak_type)
    return jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type, sharding=aval.sharding,
                                manual_axis_type=aval.mat)


class RegionDecision(NamedTuple):
    """A product region's weighing (`_takes`): whether it is fused; whether its product's launch may be the
    template's without its prologue, so the region without it could be (`fused_program`): unless its launch
    is XLA's for a reason no prologue enters (`_xla_whatever_its_prologue`). A launch XLA's because the
    prologue's resources refused the template, or one no analysis reached (a stage refused first,
    `fused_gemm.Analysis.refused`), may be. And whether its launch is the template's or undecided
    (`launched`), so the region without its epilogue could be: an epilogue keys no launch's cell, so a launch
    the selector gave XLA is XLA's without it too."""

    taken: bool
    launchable: bool
    launched: bool


_held: collections.OrderedDict[Hashable, RegionDecision] = collections.OrderedDict()


def choice_context() -> Hashable:
    """The context a kernel of Dew's is chosen in, which every held kernel decision of the algebra's keys
    (`_cache.holding`: a region's fusion, its written program, a recognition, an attention call's kernel):
    the derivation's (`derive.context`: the mesh, the target priced for, the numerics and precision) beside
    the key a trace kept outside the compile point is held by (`options.held_context`: what a trace reads of
    this thread's context, XLA's deterministic-ops flag among it, and the compiles' refused launches)."""
    from dew.algebra import derive
    from dew.nn.kernels.options import held_context

    return derive.context(), held_context()


def _identity(jaxpr: jcore.Jaxpr, region: Region) -> Hashable:
    """A product region's structure (`program.structure`'s, the algebra's windows' key): all its call reads
    of it (`_replayed` takes its constants as inputs, and its literals by their values)."""
    from dew.algebra import program

    return program.structure(jcore.Jaxpr([], list(region.inputs), list(region.outputs),
                                         [jaxpr.eqns[place] for place in region.replayed]))


def _decided(identity: Hashable, decide: Callable[[], RegionDecision]) -> RegionDecision:
    """A region's decision (`decide`), weighed once for its structure (`_identity`) in the context a kernel
    is chosen in (`choice_context`), held (`holding`)."""
    def weighed() -> RegionDecision:
        # Its probes' routes are no program's: the routes a block collects are the kernels it runs.
        with generation.routes(alone=True):
            return decide()
    return _cache.holding(_held, (identity, choice_context()), weighed, DECISIONS_HELD)


def _takes(fused_gemm, jaxpr: jcore.Jaxpr, region: Region) -> RegionDecision:
    """Whether the region is fused (`fused_gemm.analysis`): its product's launch the template's (its
    selector's choice, or a call's, `generation.chosen`), its prologue and epilogue the kernel's, nothing
    refused, its fused seconds less than its unfused ones, and its launch keeping with them the residency it
    takes without (`fused_gemm.Analysis.resident`: KERNEL-TEMPLATES-13, row 9032); and whether its launch may
    be the template's without its prologue (`RegionDecision`). The prices are a roofline on the part's rates
    (each stage's units overlapped, plus the fixed launch), not the derived tile's walk: on a cell where the
    launch measured slower than XLA's dot (`grouped_matmul.DECLARED`, every dense cell: DENSE1, 1.05 to 2.9
    times XLA's) they favour fusion more than measurement does, so a region whose launch the selector gives
    XLA is never fused by price alone; a call naming the template (an A/B's arm) fuses it."""
    inputs = _shapes(region.inputs)
    # A region of a program staged under x64 (the CPU arithmetic's float64 sums) is analyzed under it, its
    # 64-bit values as it states them.
    wide = any(shape.dtype.itemsize == 8 for shape in (*inputs, *_shapes(region.outputs)))
    with jax.enable_x64(new_val=wide):
        found = fused_gemm.analysis(_replayed(jaxpr, region, fused_gemm), *inputs)
    def kernel(route) -> bool:
        return route is not None and route.refusal is None and route.kernel != "xla"

    launchable = not _xla_whatever_its_prologue(found.launch)
    launched = found.launch is None or kernel(found.launch)
    routes = [found.launch, *([found.prologue] if region.prologue else [])]
    if found.epilogue is None and not region.prologue:
        # No stage to weigh (a bare product, a kernel gradient's): its launch's selector decides alone.
        return RegionDecision(all(map(kernel, routes)), launchable, launched)
    # A prologue the template refuses (a row normalized from float32 to bfloat16, `dtype_changes`) is no
    # region's: the region without it is weighed instead (`fused_program`). Each stage it holds is weighed by
    # price, a prologue's as an epilogue's (`fused_gemm.priced`: the operand formed in the launch, not
    # written whole and read back).
    stages = (*routes, *([found.epilogue] if found.epilogue is not None else []))
    priced = found.fused_seconds is not None and found.unfused_seconds is not None
    # A neighbour joins the kernel only where the kernel with it keeps its residency (KERNEL-TEMPLATES-13, row
    # 9032: `fused_gemm.Analysis.resident`); refused for it, the region is still weighed without its stages.
    return RegionDecision(all(map(kernel, stages)) and priced and found.fused_seconds < found.unfused_seconds
                          and found.resident, launchable, launched)


def _xla_whatever_its_prologue(launch) -> bool:
    """Whether a product's `launch` (a `generation.Route`) is XLA's for a reason its region's prologue does
    not enter: the call's naming XLA's. A cell's measurement is no such reason: a product's cell is keyed by
    whether its prologue forms its operand (`grouped_matmul`'s formed cells), so the product without it may
    be another cell's, the template's (a 4B normed projection's on an H100: its formed call XLA's, the call
    without its prologue FOUR_B's Mosaic, train 977's F1)."""
    return launch is not None and launch.kernel == "xla" and launch.forced == "xla"


class Recorded(NamedTuple):
    """What a trace of Dew's reported as it ran (`recorded`): the routes it took, the launches it made (each
    its op and keyed arguments, and the config it took, `tuning.Seen.held`) and the facts it read
    (`generation.recording`). A program written from that trace again, never traced again, reports them
    again (`replay`), so a recording around the program that writes it holds them as one around the trace
    did (row 6708), and is reused only while each fact answers alike (`still`)."""

    routes: tuple
    launches: tuning.Seen
    reads: dict

    def still(self) -> bool:
        """Whether each fact the trace read answers alike now (`generation.still`)."""
        return generation.still(self.reads)

    def replay(self) -> None:
        """The trace's routes and launches reported, and its facts read, where its program is written."""
        from dew.nn.kernels.tuning import noted

        generation.reported(self.routes)
        noted(self.launches)
        generation.replay(self.reads)


def recorded[Made](make: Callable[[], Made]) -> tuple[Made, Recorded]:
    """`make()`, a trace of Dew's, and what it reported (`Recorded`): the one record-and-replay of a held
    trace, a region's (`_traced`) and a nested body's (`dew.algebra.program._closed`)."""
    # Imported where used: tuning imports the algebra's terms, which this module's package imports.
    from dew.nn.kernels.tuning import seen as launched

    with (generation.routes(alone=True) as routes, launched() as seen, generation.recording() as reads):
        made = make()
    return made, Recorded(tuple(routes), seen.held(), reads)


class _Trace(NamedTuple):
    """A region's written program (`_traced`): its one equation, the inputs it reads them by, and what its
    trace reported (`Recorded`)."""

    equation: jcore.JaxprEqn
    invars: tuple[jcore.Var, ...]
    record: Recorded


_traces: collections.OrderedDict[Hashable, _Trace] = collections.OrderedDict()


def _traced(region: Written) -> _Trace | None:
    """`region`'s call traced as one nested program over its inputs' types (`nested`), once a process for
    its identity (`Written.key`) in the context a kernel is chosen in (`choice_context`), held
    (`_cache.holding`) while each fact its trace read answers alike (`generation.still`); its routes and
    launches reported where it is written, as its trace reported them (`generation.reported`,
    `tuning.noted`), and the facts it read read again there (`generation.replay`), so a recording around the
    program that writes it holds them as one around its trace did (row 6708)."""
    def traced_once() -> _Trace:
        traced, record = recorded(lambda: jax.make_jaxpr(nested(region.call))(*_shapes(region.inputs)))
        (equation,) = traced.jaxpr.eqns
        return _Trace(equation, tuple(traced.jaxpr.invars), record)

    held = _cache.holding(_traces, (region.key, choice_context()), traced_once, DECISIONS_HELD,
                          still=lambda trace: trace.record.still())
    # Its outputs the very types the region made (`jax_core.typematch`: shape, dtype, sharding, the manual
    # axes they vary over), else it is not written, nothing of its trace reported: a value of another type
    # in its place is a program JAX refuses.
    if not all(jax_core.typematch(out.aval, made.aval)
               for out, made in zip(region.outputs, held.equation.outvars, strict=True)):
        return None
    held.record.replay()
    return held


def _spliced(jaxpr: jcore.Jaxpr, regions: Sequence[Written]) -> jcore.Jaxpr:
    """`jaxpr` with each of `regions`' equations replaced by one nested program of its call over its inputs
    (`_traced`), its outputs the very values the region made, written at its place."""
    traces = [(region, trace) for region in regions if (trace := _traced(region)) is not None]
    if not traces:
        # No region written: the program itself, whose identity JAX shares its lowering by.
        return jaxpr
    removed = {place for region, _ in traces for place in region.places}
    written: dict[int, list[jcore.JaxprEqn]] = {}
    for region, (equation, invars, *_) in traces:
        renamed = dict(zip(invars, region.inputs, strict=True))
        written.setdefault(region.at, []).append(equation.replace(
            invars=[renamed.get(var, var) if isinstance(var, jcore.Var) else var for var in equation.invars],
            outvars=list(region.outputs)))
    equations = []
    for place, equation in enumerate(jaxpr.eqns):
        equations.extend(written.get(place, []))
        if place not in removed:
            equations.append(equation)
    equations.extend(written.get(len(jaxpr.eqns), []))
    return jaxpr.replace(eqns=equations)
