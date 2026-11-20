"""A program written back as Python (docs/design/algebra.md §6.11): a jaxpr printed as a plain JAX function,
each equation its primitive bound at its own parameters (`lax.<name>_p.bind`), so the function computes
exactly what the jaxpr does, bit for bit; and the same function with its float sums compensated
(`compensated`: the error-free derivation, §6.9), each addition's rounding carried beside it as a double word
(`dew.nn.precision.compensated_add`) and each sum's reduction compensated (`compensated_sum`), both protected
from the pass. What the generator writes and what dev time measures are one text (`source`, `loaded`)."""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import ShapedArray
from jax.extend import core as jcore

WIDTH = 110
"""The tree's line length (pyproject.toml's ruff), which every printed line keeps."""

INLINED = frozenset({"pjit", "jit", "closed_call", "core_call"})
"""The call primitives whose body is printed in their place: a call that computes its body, no rule of its
own (a custom derivative's call is refused, its rule not a value's)."""

IMPORTS = {"jnp": "import jax.numpy as jnp", "np": "import numpy as np", "lax": "from jax import lax",
           "precision": "from dew.nn import precision"}
"""The imports a printed function may read, by the name it reads, in the tree's order (`imports`)."""


def imports(text: str) -> list[str]:
    """The imports printed `text` reads (`IMPORTS`), each where it reads its name, in the tree's order."""
    return [line for name, line in IMPORTS.items() if f"{name}." in text]


class Unprintable(ValueError):
    """A program with an equation no plain JAX call writes as the jaxpr holds it (a custom derivative's call,
    a primitive JAX does not export, a parameter of no literal form)."""


@dataclasses.dataclass
class _Printer:
    """One function's text as it is printed: its lines, each value's name, and, compensated, each sum's
    double word by the value it computes and the values already read from one."""

    compensated: bool
    lines: list[str] = dataclasses.field(default_factory=list)
    names: dict[jcore.Var, str] = dataclasses.field(default_factory=dict)
    pairs: dict[jcore.Var, str] = dataclasses.field(default_factory=dict)
    read: set[jcore.Var] = dataclasses.field(default_factory=set)
    count: int = 0
    used: bool = False

    def fresh(self, prefix: str = "v") -> str:
        self.count += 1
        return f"{prefix}{self.count}"

    def emit(self, target: str, call: str, arguments: list[str]) -> None:
        self.lines.extend(_wrapped(f"    {target} = {call}(", arguments, ")"))

    def atom(self, atom) -> str:
        """An operand's text: a literal's value, or a value's name, a double word read as its high word."""
        if isinstance(atom, jcore.Literal):
            return _value(atom.val, atom.aval.dtype)
        if atom in self.pairs and atom not in self.read:
            self.read.add(atom)
            self.lines.append(f"    {self.names[atom]} = {self.pairs[atom]}[0]")
        return self.names[atom]

    def pair(self, atom, negated: bool = False) -> str:
        """An operand as a double word (its own, or the value beside a zero), negated where `negated`."""
        if isinstance(atom, jcore.Var) and atom in self.pairs:
            held = self.pairs[atom]
            return f"(-{held}[0], -{held}[1])" if negated else held
        value = self.atom(atom)
        return f"({'-' if negated else ''}{value}, jnp.zeros_like({value}))"


def source(closed: jcore.ClosedJaxpr, name: str, *, compensated: bool = False) -> tuple[str, bool]:
    """`closed` printed as a function `name` over its inputs returning its outputs as a tuple, and whether a
    sum was compensated (`compensated`: always False where it is not asked). Raises `Unprintable` for an
    equation no plain call writes."""
    printer = _Printer(compensated)
    inputs = [printer.fresh("x") for _ in closed.jaxpr.invars]
    printer.names.update(zip(closed.jaxpr.invars, inputs, strict=True))
    for var, value in zip(closed.jaxpr.constvars, closed.consts, strict=True):
        printer.names[var] = printer.fresh("c")
        printer.lines.extend(_constant(printer.names[var], np.asarray(value)))
    _body(printer, closed.jaxpr)
    outputs = [printer.atom(atom) for atom in closed.jaxpr.outvars]
    returned = _wrapped("    return (", outputs, ",)" if len(outputs) == 1 else ")")
    return "\n".join([f"def {name}({', '.join(inputs)}):", *printer.lines, *returned]) + "\n", printer.used


def _body(printer: _Printer, jaxpr: jcore.Jaxpr) -> None:
    """Each equation of `jaxpr` printed in order, a call's body in its place (`INLINED`)."""
    for equation in jaxpr.eqns:
        primitive = equation.primitive
        if primitive.name in INLINED:
            inner = equation.params.get("jaxpr", equation.params.get("call_jaxpr"))
            body = inner.jaxpr if isinstance(inner, jcore.ClosedJaxpr) else inner
            consts = inner.consts if isinstance(inner, jcore.ClosedJaxpr) else []
            for var, value in zip(body.constvars, consts, strict=True):
                printer.names[var] = printer.fresh("c")
                printer.lines.extend(_constant(printer.names[var], np.asarray(value)))
            for var, atom in zip(body.invars, equation.invars, strict=True):
                _alias(printer, var, atom)
            _body(printer, body)
            for var, atom in zip(equation.outvars, body.outvars, strict=True):
                _alias(printer, var, atom)
            continue
        if printer.compensated and _compensates(equation):
            _compensated(printer, equation)
            continue
        if getattr(lax, f"{primitive.name}_p", None) is not primitive:
            raise Unprintable(f"no plain call writes a {primitive.name} equation")
        outputs = [printer.fresh() for _ in equation.outvars]
        for var, named in zip(equation.outvars, outputs, strict=True):
            printer.names[var] = named
        arguments = [printer.atom(atom) for atom in equation.invars]
        arguments += [f"{key}={_param(value)}" for key, value in equation.params.items()]
        target = ", ".join(outputs) if primitive.multiple_results else outputs[0]
        printer.emit(target, f"lax.{primitive.name}_p.bind", arguments)


def _alias(printer: _Printer, var: jcore.Var, atom) -> None:
    """`var` named as `atom` is: a call's input as its operand, its output as its body's."""
    if isinstance(atom, jcore.Literal):
        printer.names[var] = printer.fresh()
        printer.lines.append(f"    {printer.names[var]} = {_value(atom.val, atom.aval.dtype)}")
        return
    printer.names[var] = printer.names[atom]
    if atom in printer.pairs:
        printer.pairs[var] = printer.pairs[atom]


def _floating(var) -> bool:
    return isinstance(var.aval, ShapedArray) and jnp.issubdtype(var.aval.dtype, jnp.floating)


def _compensates(equation: jcore.JaxprEqn) -> bool:
    """Whether a compensated program carries `equation` as a double word: a float addition or subtraction,
    or a float sum's reduction."""
    if equation.primitive in (lax.add_p, lax.sub_p):
        return all(_floating(var) for var in (*equation.invars, *equation.outvars))
    return equation.primitive is lax.reduce_sum_p and _floating(equation.invars[0])


def _compensated(printer: _Printer, equation: jcore.JaxprEqn) -> None:
    """`equation` printed compensated (`_compensates`): an addition's double word, its operands' own, or a
    reduction's compensated sum, rounded once."""
    printer.used = True
    (out,) = equation.outvars
    printer.names[out] = printer.fresh()
    if equation.primitive is lax.reduce_sum_p:
        axes = tuple(equation.params["axes"])
        printer.emit(printer.names[out], "precision.compensated_sum",
                     [printer.atom(equation.invars[0]), _param(axes)])
        return
    left, right = equation.invars
    pair = printer.fresh("p")
    printer.pairs[out] = pair
    printer.emit(pair, "precision.compensated_add",
                 [printer.pair(left), printer.pair(right, negated=equation.primitive is lax.sub_p)])


def loaded(text: str, name: str) -> Callable:
    """The function `name` a printed text defines (`source`), over the imports a printed module reads."""
    from dew.nn import precision

    namespace = {"jax": jax, "jnp": jnp, "np": np, "lax": lax, "precision": precision}
    exec(compile(text, f"<printed {name}>", "exec"), namespace)
    return namespace[name]


def _value(value, dtype) -> str:
    """A literal's text: a scalar exactly (a float by its hex digits), an array by its bytes."""
    held = np.asarray(value, dtype=dtype)
    if held.ndim:
        return _bytes(held)
    named = f"jnp.dtype({np.dtype(dtype).name!r})"
    if np.dtype(dtype) == np.bool_:
        return f"np.array({bool(held)}, dtype={named})"
    if jnp.issubdtype(dtype, jnp.integer):
        return f"np.array({int(held)}, dtype={named})"
    if jnp.issubdtype(dtype, jnp.floating):
        return f"np.array({_float(float(held))}, dtype={named})"
    raise Unprintable(f"no literal form of a {np.dtype(dtype).name} value")


def _float(value: float) -> str:
    if math.isnan(value):
        return "float('nan')"
    if math.isinf(value):
        return f"float('{'-' if value < 0 else ''}inf')"
    return f"float.fromhex({value.hex()!r})"


def _bytes(held: np.ndarray) -> str:
    return (f"np.frombuffer(bytes.fromhex({held.tobytes().hex()!r}), dtype=jnp.dtype({held.dtype.name!r}))"
            f".reshape({tuple(held.shape)!r})")


def _constant(name: str, held: np.ndarray) -> list[str]:
    """A constant's lines: its bytes in chunks each line holds."""
    digits = held.tobytes().hex()
    chunk = WIDTH - 12
    parts = [f"        {digits[start:start + chunk]!r}" for start in range(0, len(digits), chunk)]
    return [f"    {name} = np.frombuffer(bytes.fromhex(", *(parts or ["        ''"]),
            f"    ), dtype=jnp.dtype({held.dtype.name!r})).reshape({tuple(held.shape)!r})"]


def _param(value) -> str:
    """A parameter's literal text, refused where it has none JAX reads back as it is."""
    if value is None or isinstance(value, bool | int | str):
        return repr(value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, tuple | list):
        inner = ", ".join(_param(item) for item in value)
        return f"({inner}{',' if len(value) == 1 else ''})"
    if isinstance(value, np.dtype) or (isinstance(value, type) and issubclass(value, np.generic)):
        return f"jnp.dtype({np.dtype(value).name!r})"
    if isinstance(value, enum.Enum) and getattr(lax, type(value).__name__, None) is type(value):
        return f"lax.{type(value).__name__}.{value.name}"
    raise Unprintable(f"no literal form of a {type(value).__name__} parameter")


def _wrapped(head: str, arguments: list[str], tail: str) -> list[str]:
    """A call's lines within `WIDTH`: on one where it fits, else its arguments packed onto lines after its
    head, each indented four more."""
    whole = head + ", ".join(arguments) + tail
    if len(whole) <= WIDTH:
        return [whole]
    lines, current = [head], "        "
    for number, argument in enumerate(arguments):
        piece = argument + (", " if number < len(arguments) - 1 else tail)
        if current.strip() and len(current) + len(piece.rstrip()) > WIDTH:
            lines.append(current.rstrip())
            current = "        "
        current += piece
    lines.append(current.rstrip())
    return lines
