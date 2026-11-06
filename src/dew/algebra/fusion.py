"""Fusion search (docs/design/algebra.md §2 "Fusion"): a program's products fused with their elementwise
consumers into the GEMM template's epilogue, where the template takes them.

The compile-time pass (`program._rewritten`) hands each body here after its algebra. Each matrix product of
the body (`dot_general` of a rows-by-depth operand and a matrix, no batch) and the elementwise equations that
follow it, up to the first equation outside them that reads one of their values (`_regions`), are one region:
its inputs the values it reads from before it, its outputs the values read after it. A region is fused where
the template's selectors take it (`fused_gemm.analysis`: the product's launch is the template's, its
epilogue the kernel's, no refusal), its equations replaced by one nested program of `fused_gemm.fused` over
the region as the program wrote it, its dot `fused_gemm.product` (the dot's own precision and rounding, so
the fused region computes what the region did), written where its first reader stood (`_spliced`). Every
other region, and every body with none, is kept as traced: the body itself, so an unchanged body keeps its
identity (`program._within`'s).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
from jax.extend import core as jcore

from dew.nn.kernels import generation
from dew.nn.nested import nested
from dew.nn.programs import ENTRYWISE, jax_name

__all__ = ["fused_program"]

CONSUMERS = ENTRYWISE | {"convert_element_type"}
"""The equations a product's epilogue takes: each entry of its output from the same entry of its inputs
(`dew.nn.programs.ENTRYWISE`) or a cast."""


class Region(NamedTuple):
    """A product and its elementwise consumers in a body (`_regions`): their equations' places in the body,
    the values they read from before them (`inputs`), the values read after them (`outputs`), and the place
    of the first equation after them that reads one (where the fused region is written)."""

    places: tuple[int, ...]
    inputs: tuple[jcore.Var, ...]
    outputs: tuple[jcore.Var, ...]
    at: int


def fused_program(jaxpr: jcore.Jaxpr) -> jcore.Jaxpr:
    """`jaxpr` with each region the template takes fused (the module's account); `jaxpr` itself where none
    is."""
    from dew.nn.kernels import fused_gemm

    taken = [region for region in _regions(jaxpr) if _takes(fused_gemm, jaxpr, region)]
    return _spliced(jaxpr, taken, fused_gemm) if taken else jaxpr


class Product(NamedTuple):
    """A product the template's epilogue follows (`_product`): its dot's equation, whether its right operand
    is laid columns by depth (its transpose), and the dtype its result is rounded to."""

    dot: jcore.JaxprEqn
    transposed: bool
    out_dtype: object


def _product(equation: jcore.JaxprEqn) -> Product | None:
    """`equation` as a product the template's epilogue follows: a `dot_general` of a rows-by-depth left
    operand and a depth-by-columns right one (or columns by depth, its transpose), no batch axes, or a call
    of one (`jnp.matmul`'s jit), its result cast once after it; None for any other equation."""
    dot, out_dtype = equation, equation.outvars[0].aval.dtype if equation.outvars else None
    if jax_name(equation.primitive) in ("pjit", "jit", "closed_call"):
        body = equation.params.get("jaxpr") or equation.params.get("call_jaxpr")
        inner = getattr(body, "jaxpr", body)
        names = [jax_name(eqn.primitive) for eqn in inner.eqns]
        if names not in (["dot_general"], ["dot_general", "convert_element_type"]):
            return None
        dot = inner.eqns[0]
    if jax_name(dot.primitive) != "dot_general" or len(equation.invars) != 2:
        return None
    (left, right), batch = dot.params["dimension_numbers"]
    shapes = [tuple(var.aval.shape) for var in equation.invars]
    transposed = {(0,): False, (1,): True}.get(tuple(right))
    if batch != ((), ()) or tuple(left) != (1,) or transposed is None or any(len(shape) != 2
                                                                             for shape in shapes):
        return None
    return Product(dot, transposed, out_dtype)


def _regions(jaxpr: jcore.Jaxpr) -> list[Region]:
    """Each product of `jaxpr` (`_product`) and the elementwise equations after it that read its values,
    each of the product's shape over them and values from before the region (`CONSUMERS`), up to the first
    equation outside them that reads one of their values."""
    found: list[Region] = []
    claimed: set[int] = set()
    for start, equation in enumerate(jaxpr.eqns):
        if _product(equation) is None or start in claimed:
            continue
        shape = tuple(equation.outvars[0].aval.shape)
        places, made = [start], set(equation.outvars)
        at = len(jaxpr.eqns)
        for place in range(start + 1, len(jaxpr.eqns)):
            later = jaxpr.eqns[place]
            reads = [var for var in later.invars if isinstance(var, jcore.Var) and var in made]
            if not reads:
                continue
            joins = (jax_name(later.primitive) in CONSUMERS and not later.effects and place not in claimed
                     and all(tuple(var.aval.shape) in (shape, ()) for var in (*later.invars, *later.outvars)))
            if not joins:
                at = place
                break
            places.append(place)
            made.update(later.outvars)
        members = [jaxpr.eqns[place] for place in places]
        read_after = {var for later in jaxpr.eqns[places[-1] + 1:] for var in later.invars
                      if isinstance(var, jcore.Var)}
        read_after.update(var for var in jaxpr.outvars if isinstance(var, jcore.Var))
        outputs = tuple(var for member in members for var in member.outvars if var in read_after)
        inputs = tuple(dict.fromkeys(var for member in members for var in member.invars
                                     if isinstance(var, jcore.Var) and var not in made))
        if len(places) > 1 and outputs:
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
            product = _product(equation)
            if product is not None:
                results = [fused_gemm.product(*operands, transpose_rhs=product.transposed,
                                              precision=product.dot.params["precision"],
                                              out_dtype=product.out_dtype,
                                              interpret_on_cpu=generation.assumption() is not None)]
            else:
                made = equation.primitive.bind(*operands, **equation.params)
                results = list(made) if equation.primitive.multiple_results else [made]
            values.update(zip(equation.outvars, results, strict=True))
        return tuple(values[var] for var in region.outputs)

    return replay


def _shapes(variables: Sequence[jcore.Var]) -> list[jax.ShapeDtypeStruct]:
    return [jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype) for var in variables]


def _takes(fused_gemm, jaxpr: jcore.Jaxpr, region: Region) -> bool:
    """Whether the template takes `region` (`fused_gemm.analysis`): its product's launch the template's and
    its epilogue the kernel's, nothing refused."""
    found = fused_gemm.analysis(_replayed(jaxpr, region, fused_gemm), *_shapes(region.inputs))
    return all(route is not None and route.refusal is None and route.kernel != "xla"
               for route in (found.launch, found.epilogue))


def _spliced(jaxpr: jcore.Jaxpr, regions: Sequence[Region], fused_gemm) -> jcore.Jaxpr:
    """`jaxpr` with each of `regions`' equations replaced by one nested program of `fused_gemm.fused` over
    the region (`_replayed`), its inputs the region's and its outputs the very values the region made,
    written where the region's first reader stood."""
    removed = {place for region in regions for place in region.places}
    written: dict[int, list[jcore.JaxprEqn]] = {}
    for region in regions:
        replay = _replayed(jaxpr, region, fused_gemm)
        call = nested(lambda *inputs, replay=replay: fused_gemm.fused(replay, *inputs))
        traced = jax.make_jaxpr(call)(*_shapes(region.inputs))
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
