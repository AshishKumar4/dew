"""Fusion search (docs/design/algebra.md §2 "Fusion"): a program's products fused with their elementwise
consumers into the GEMM template's epilogue, where the template takes them and fusing is the cheaper.

The compile-time pass (`program._rewritten`) hands each body here after its algebra: the compile point at
which a plain JAX program gets the template (row 3450). Each matrix product of the body the template takes
(`Form`: a `dot_general` of rank-2 operands, x @ W, dy @ Wᵀ or a kernel gradient's xᵀ @ dy; a
`ragged_dot_general` of each group's rows, an MoE's; floating operands, no batch) and the elementwise
equations that follow it, up to the first equation outside them that reads one of their values, with the
broadcasts only they read of their captured operands (a bias's), are one region (`_regions`), a product
alone among them: its inputs the values it reads from before it, its outputs the values read after it. A
region is substituted where its launch is the template's, its selector's choice (`grouped_matmul.DECLARED`,
or a call's, `generation.chosen`), and where it has an epilogue, where the template takes it and its price
says fused wins (`_takes`): its equations replaced by one nested program of `fused_gemm.fused` over the
region as the program wrote it, its product the template's (`_product`: `fused_gemm.product` at the dot's
own precision and result dtype, so the fused region computes what the region did; a kernel gradient's
`gemm.gemm_t`, its consumers after it in XLA), written where its first reader stood (`_spliced`). Every
other region, and every body with none, is kept as traced: the body itself, so an unchanged body keeps its
identity (`program._within`'s). A region's decision is held by its structure
and the context it was weighed in (`_decided`), so every layer of a stack, and every retrace, weighs it once.
A softmax attention (`attention_regions`) is found first and written as Dew's attention where a kernel of
Dew's takes it, through the same splice (`_spliced`); a product inside it is none of the products' regions.
An RMSNorm or SwiGLU no product's epilogue took (`norm_regions`) is written as Dew's where its fused kernel
takes it, through the same splice.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Hashable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend import core as jcore

from dew.nn.kernels import generation
from dew.nn.nested import nested
from dew.nn.programs import ENTRYWISE, jax_name

CONSUMERS = ENTRYWISE | {"convert_element_type"}
"""The equations a product's epilogue takes: each entry of its output from the same entry of its inputs
(`dew.nn.programs.ENTRYWISE`) or a cast."""

CAPTURED = frozenset({"broadcast_in_dim", "reshape"})
"""The layout equations a region takes of a captured operand only it reads (a bias broadcast over rows), so
the epilogue captures the operand itself."""

DECISIONS_HELD = 256
"""How many regions' decisions are held (`_decided`), the least recently weighed dropped past them: a
model's distinct projections, each shape once."""


class Region(NamedTuple):
    """A product and its elementwise consumers in a body (`_regions`): their equations' places in the body,
    in order, the values they read from before them (`inputs`), the values read after them (`outputs`), and
    the place of the first equation after them that reads one (where the fused region is written)."""

    places: tuple[int, ...]
    inputs: tuple[jcore.Var, ...]
    outputs: tuple[jcore.Var, ...]
    at: int


def fused_program(jaxpr: jcore.Jaxpr) -> jcore.Jaxpr:
    """`jaxpr` with each attention region a kernel of Dew's takes written as Dew's attention
    (`attention_regions`), each product region the template takes fused (the module's account), and each
    RMSNorm and SwiGLU region the fused norms take written as Dew's (`norm_regions`); `jaxpr` itself where
    none is."""
    from dew.algebra import attention_regions, norm_regions
    from dew.nn.kernels import fused_gemm

    index = Index(jaxpr)
    attention = [region for region in attention_regions.regions(index)
                 if _decided(jaxpr, region.places, _read(jaxpr, region.places), region.outputs,
                             lambda region=region: attention_regions.taken(region))]
    held = {place for region in attention for place in region.places}
    products = [region for region in _regions(index) if not held & set(region.places)
                and _decided(jaxpr, region.places, region.inputs, region.outputs,
                             lambda region=region: _takes(fused_gemm, jaxpr, region))]
    held |= {place for region in products for place in region.places}
    norms = [region for region in norm_regions.regions(index) if not held & set(region.places)
             and _decided(jaxpr, region.places, _read(jaxpr, region.places), region.outputs,
                          lambda region=region: norm_regions.taken(region))]
    if not attention and not products and not norms:
        return jaxpr
    written = [Written(region.places, region.operands, region.outputs, region.places[-1],
                       module.written(region))
               for module, found in ((attention_regions, attention), (norm_regions, norms))
               for region in found]
    written += [Written(region.places, region.inputs, region.outputs, region.at,
                        _fusing(_replayed(jaxpr, region, fused_gemm), fused_gemm)) for region in products]
    return _spliced(jaxpr, written)


def _read(jaxpr: jcore.Jaxpr, places: Sequence[int]) -> tuple[jcore.Var, ...]:
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
    it holds them (`consts`): what the region finders read (`_regions`, `attention_regions.regions`,
    `norm_regions.regions`)."""

    def __init__(self, jaxpr: jcore.Jaxpr):
        self.jaxpr = jaxpr
        values = getattr(jaxpr, "consts", ())
        held = len(values) == len(jaxpr.constvars)
        self.consts = dict(zip(jaxpr.constvars, values, strict=True)) if held else {}
        self.producer = {var: place for place, equation in enumerate(jaxpr.eqns) for var in equation.outvars}
        self.readers: dict[jcore.Var, set[int]] = collections.defaultdict(set)
        for place, equation in enumerate(jaxpr.eqns):
            for var in equation.invars:
                if isinstance(var, jcore.Var):
                    self.readers[var].add(place)
        self.escaping = {var for var in jaxpr.outvars if isinstance(var, jcore.Var)}


class Written(NamedTuple):
    """A region's equations (their places) replaced by one nested program of `call` over its `inputs`, making
    its `outputs`, written at place `at` (`_spliced`)."""

    places: tuple[int, ...]
    inputs: tuple[jcore.Var, ...]
    outputs: tuple[jcore.Var, ...]
    at: int
    call: Callable


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

GROUPED_FORMS = {((1,), (1,), 3): FORWARD._replace(grouped=True),
                 ((1,), (2,), 3): FORWARD._replace(transpose_rhs=True, grouped=True),
                 ((0,), (0,), 2): FORWARD._replace(transpose_lhs=True, grouped=True)}
"""A `ragged_dot_general`'s contracted axes and its right operand's rank, its left operand's rows ragged and
no batch axes, to its form: rows times their group's matrix (`[groups, depth, columns]`, or columns by
depth), and the kernel gradient's rows contracted within each group (`[groups, depth, columns]` out)."""


def _form(equation: jcore.JaxprEqn) -> Form | None:
    """`equation`'s form as a product the template takes (`Form`): a `dot_general` of rank-2 operands (x @ W,
    dy @ Wᵀ, xᵀ @ dy) or a `ragged_dot_general` of rows ragged by group (`GROUPED_FORMS`), each of floating
    operands, no batch axes; None for any other equation (an integer product is XLA's own,
    `fused_gemm.product`)."""
    name = jax_name(equation.primitive)
    if name not in ("dot_general", "ragged_dot_general"):
        return None
    if not all(jnp.issubdtype(var.aval.dtype, jnp.inexact) for var in equation.invars[:2]
               if isinstance(var.aval, ShapedArray)):
        return None
    if name == "dot_general":
        (left, right), batch = equation.params["dimension_numbers"]
        if batch != ((), ()) or any(len(_shape(var)) != 2 for var in equation.invars):
            return None
        return DENSE_FORMS.get((tuple(left), tuple(right)))
    numbers = equation.params["ragged_dot_dimension_numbers"]
    # Its axes as lists (`lax.ragged_dot`'s), a dot_general's as tuples.
    (left, right), batch = numbers.dot_dimension_numbers
    if (any(batch) or tuple(numbers.lhs_ragged_dimensions) != (0,)
            or len(_shape(equation.invars[0])) != 2):
        return None
    form = GROUPED_FORMS.get((tuple(left), tuple(right), len(_shape(equation.invars[1]))))
    groups = tuple(numbers.rhs_group_dimensions)
    return form if form is not None and groups == ((0,) if not form.transpose_lhs else ()) else None


def _shape(atom) -> tuple[int, ...]:
    """An equation's operand's or result's shape; () for a value of no array's type (a token's)."""
    return tuple(atom.aval.shape) if isinstance(atom.aval, ShapedArray) else ()


def _follows(operand: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    """Whether an operand of `operand`'s shape follows the product's tiles of `shape`: a scalar, or of its
    rank with each axis the product's or one (a bias's `[1, columns]`, a row scale's `[rows, 1]`)."""
    return not operand or (len(operand) == len(shape)
                           and all(axis in (1, extent) for axis, extent in zip(operand, shape, strict=True)))


def _regions(index: Index) -> list[Region]:
    """Each product of `jaxpr` (`_form`) and the elementwise equations after it that read its values,
    each of the product's shape over operands that follow its tiles (`_follows`, `CONSUMERS`), up to the
    first equation outside them that reads one of their values, with the layout equations of its captured
    operands that no other equation reads (`CAPTURED`)."""
    jaxpr, readers, escaping, producer = index.jaxpr, index.readers, index.escaping, index.producer
    found: list[Region] = []
    claimed: set[int] = set()
    for start, equation in enumerate(jaxpr.eqns):
        if _form(equation) is None or start in claimed:
            continue
        shape = _shape(equation.outvars[0])
        places, made, at = [start], set(equation.outvars), len(jaxpr.eqns)
        for place in range(start + 1, len(jaxpr.eqns)):
            later = jaxpr.eqns[place]
            if not any(isinstance(var, jcore.Var) and var in made for var in later.invars):
                continue
            if not (jax_name(later.primitive) in CONSUMERS and not later.effects and place not in claimed
                    and all(_shape(var) == shape for var in later.outvars)
                    and all(_follows(_shape(var), shape) for var in later.invars)):
                at = place
                break
            places.append(place)
            made.update(later.outvars)
        members = set(places)
        for place in list(places[1:]):
            for var in jaxpr.eqns[place].invars:
                if not isinstance(var, jcore.Var):
                    continue
                source = producer.get(var)
                if (source is not None and source not in members and source not in claimed
                        and jax_name(jaxpr.eqns[source].primitive) in CAPTURED
                        and readers[var] <= members and var not in escaping):
                    members.add(source)
                    made.update(jaxpr.eqns[source].outvars)
        places = sorted(members)
        read_after = {var for later in jaxpr.eqns[places[-1] + 1:] for var in later.invars
                      if isinstance(var, jcore.Var)} | escaping
        outputs = tuple(var for place in places for var in jaxpr.eqns[place].outvars if var in read_after)
        inputs = tuple(dict.fromkeys(var for place in places for var in jaxpr.eqns[place].invars
                                     if isinstance(var, jcore.Var) and var not in made))
        if outputs:
            found.append(Region(tuple(places), inputs, outputs, min(at, places[-1] + 1)))
            claimed.update(places)
    return found


def _replayed(jaxpr: jcore.Jaxpr, region: Region, fused_gemm):
    """The region as the program wrote it, a function of its inputs returning its outputs, its product
    `fused_gemm.product` declared as its dot (its precision, its result's dtype)."""
    def replay(*inputs):
        values = dict(zip(region.inputs, inputs, strict=True))

        def read(atom):
            return atom.val if isinstance(atom, jcore.Literal) else values[atom]

        for place in region.places:
            equation = jaxpr.eqns[place]
            operands = [read(atom) for atom in equation.invars]
            form = _form(equation)
            if form is not None:
                made = [_product(form, operands, equation.params["precision"],
                                 equation.params["preferred_element_type"] or jnp.result_type(*operands[:2]),
                                 fused_gemm)]
            else:
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

    interpret_on_cpu = generation.assumption() is not None
    lhs, rhs, *sizes = operands
    if form.transpose_lhs:
        made = gemm.gemm_t(lhs, rhs, sizes[0] if form.grouped else None, out_dtype=out_dtype,
                           precision=precision, interpret_on_cpu=interpret_on_cpu)
        return made if form.grouped else made[0]
    return fused_gemm.product(lhs, rhs, *sizes, transpose_rhs=form.transpose_rhs, precision=precision,
                              out_dtype=out_dtype, interpret_on_cpu=interpret_on_cpu)


def _shapes(variables: Sequence[jcore.Var]) -> list[jax.ShapeDtypeStruct]:
    """Each of `variables` as its shape and dtype (each a region's value, an array's)."""
    return [jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype) for var in variables
            if isinstance(var.aval, ShapedArray)]


_held: collections.OrderedDict[Hashable, bool] = collections.OrderedDict()


def _decided(jaxpr: jcore.Jaxpr, places: Sequence[int], inputs: Sequence[jcore.Var],
             outputs: Sequence[jcore.Var], decide: Callable[[], bool]) -> bool:
    """A region's decision (`decide`), weighed once for its structure (`program.structure`'s, the algebra's
    windows' key) in the context it is weighed in (`derive.context`: the mesh, the target, the numerics and
    precision; and the kernel choices a call names), the latest `DECISIONS_HELD` held."""
    from dew import _cache
    from dew.algebra import derive, program

    window = jcore.Jaxpr([], list(inputs), list(outputs), [jaxpr.eqns[place] for place in places])
    key = (program.structure(window), derive.context(), _cache.KERNEL_CHOICES.value)
    if key not in _held:
        # Its probes' routes are no program's: the routes a block collects are the kernels it runs.
        with generation.routes(alone=True):
            _held[key] = decide()
        if len(_held) > DECISIONS_HELD:
            _held.popitem(last=False)
    _held.move_to_end(key)
    return _held[key]


def _takes(fused_gemm, jaxpr: jcore.Jaxpr, region: Region) -> bool:
    """Whether the region is fused (`fused_gemm.analysis`): its product's launch the template's (its
    selector's choice, or a call's, `generation.chosen`), its epilogue the kernel's, nothing refused, and
    its fused seconds less than its unfused ones. The prices are a roofline on the part's rates (each
    stage's units overlapped, plus the fixed launch), not the derived tile's walk: on a cell where the
    launch measured slower than XLA's dot (`grouped_matmul.DECLARED`, every dense cell: DENSE1, 1.05 to 2.9
    times XLA's) they favour fusion more than measurement does, so a region whose launch the selector gives
    XLA is never fused by price alone; a call naming the template (an A/B's arm) fuses it."""
    found = fused_gemm.analysis(_replayed(jaxpr, region, fused_gemm), *_shapes(region.inputs))
    if found.epilogue is None:
        # No epilogue to weigh (a bare product, a kernel gradient's): its launch's selector decides alone.
        return found.launch is not None and found.launch.refusal is None and found.launch.kernel != "xla"
    taken = all(route is not None and route.refusal is None and route.kernel != "xla"
                for route in (found.launch, found.epilogue))
    return (taken and found.fused_seconds is not None and found.unfused_seconds is not None
            and found.fused_seconds < found.unfused_seconds)


def _spliced(jaxpr: jcore.Jaxpr, regions: Sequence[Written]) -> jcore.Jaxpr:
    """`jaxpr` with each of `regions`' equations replaced by one nested program of its call over its inputs,
    its outputs the very values the region made, written at its place."""
    removed = {place for region in regions for place in region.places}
    written: dict[int, list[jcore.JaxprEqn]] = {}
    for region in regions:
        traced = jax.make_jaxpr(nested(region.call))(*_shapes(region.inputs))
        (equation,) = traced.jaxpr.eqns
        renamed = dict(zip(traced.jaxpr.invars, region.inputs, strict=True))
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
