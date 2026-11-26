"""A program written back as Python (docs/design/algebra.md §6.11): a jaxpr printed as a plain JAX function,
each equation its primitive bound at its own parameters (`lax.<name>_p.bind`), so the function computes
exactly what the jaxpr does, bit for bit, but for a choice by the platform it lowers for, printed as its
default branch (a GPU's and a TPU's: Dew's products under a CPU's wide arithmetic choose the CPU's own,
`dew.nn.precision`); and a compensated candidate (`compensated`, §6.9) whose elementwise sums carry
their own and their direct products' rounding residuals, and whose reductions compensate their sums.
The transforms are protected from the pass. Generation and dev-time admission measure the same text.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import re
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import core
from jax.core import ShapedArray
from jax.extend import core as jcore

WIDTH = 110
"""The tree's line length (pyproject.toml's ruff), which every printed line keeps."""

INLINED = frozenset({"pjit", "jit", "closed_call", "core_call"})
"""The call primitives whose body is printed in their place: a call that computes its body, no rule of its
own (a custom derivative's call is refused, its rule not a value's)."""

IMPORTS = ({"functools": "import functools", "Callable": "from collections.abc import Callable"},
           {"jnp": "import jax.numpy as jnp", "np": "import numpy as np", "lax": "from jax import lax"},
           {"_generated": "from dew.algebra import _generated", "precision": "from dew.nn import precision"})
"""The imports a printed function or a generated module may read, by the name it reads, in the tree's
sections (the standard library's, the third parties', Dew's) and order (`imports`)."""


def imports(text: str) -> list[str]:
    """The imports `text` reads (`IMPORTS`), each where it reads its name, a blank line between sections."""
    # A name read as a module's (`np.`) or a generic's (`Callable[`), never a keyword argument's.
    found = [[line for name, line in section.items() if re.search(rf"(?<![\w.]){name}[.\[]", text)]
             for section in IMPORTS]
    return [line for number, lines in enumerate(section for section in found if section)
            for line in ([""] if number else []) + lines]


class Unprintable(ValueError):
    """A program with an equation no plain JAX call writes as the jaxpr holds it (a custom derivative's call,
    a primitive JAX does not export, a parameter of no literal form)."""


@dataclasses.dataclass
class _Printer:
    """One function's text as it is printed: its lines, each value's name, each platform choice's index
    (`platforms`: the platforms its `platform_index` equation names), and, compensated, each sum's double word
    by the value it computes and the values already read from one."""

    compensated: bool
    lines: list[str] = dataclasses.field(default_factory=list)
    names: dict[jcore.Var, str] = dataclasses.field(default_factory=dict)
    platforms: dict[jcore.Var, tuple] = dataclasses.field(default_factory=dict)
    pairs: dict[jcore.Var, str] = dataclasses.field(default_factory=dict)
    read: set[jcore.Var] = dataclasses.field(default_factory=set)
    dimensions: dict[int, str] = dataclasses.field(default_factory=dict)
    count: int = 0
    used: bool = False

    def fresh(self, prefix: str = "v") -> str:
        self.count += 1
        return f"{prefix}{self.count}"

    def emit(self, target: str, call: str, arguments: list[str]) -> None:
        self.lines.extend(wrapped(f"    {target} = {call}(", arguments, ")"))

    def atom(self, atom) -> str:
        """An operand's text: a literal's value, or a value's name, a double word read as its high word."""
        if isinstance(atom, jcore.Literal):
            return _value(atom.val, _dtype(atom))
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


def source(closed: core.Jaxpr, name: str, *, compensated: bool = False,
           borrow_shapes: bool = False) -> tuple[str, bool]:
    """`closed` printed as a function `name` over its inputs returning its outputs as a tuple, and whether a
    sum was compensated (`compensated`: always False where it is not asked). Raises `Unprintable` for an
    equation no plain call writes."""
    printer = _Printer(compensated)
    inputs = [printer.fresh("x") for _ in closed.jaxpr.invars]
    printer.names.update(zip(closed.jaxpr.invars, inputs, strict=True))
    if borrow_shapes:
        for var, arg in zip(closed.jaxpr.invars, inputs, strict=True):
            for axis, extent in enumerate(_shaped(var).shape):
                if isinstance(extent, int) and extent > 1:
                    printer.dimensions.setdefault(extent, f"{arg}.shape[{axis}]")
    for var, value in zip(closed.jaxpr.constvars, closed.consts, strict=True):
        printer.names[var] = printer.fresh("c")
        printer.lines.extend(_constant(printer.names[var], np.asarray(value)))
    _body(printer, closed.jaxpr)
    outputs = [printer.atom(atom) for atom in closed.jaxpr.outvars]
    returned = wrapped("    return (", outputs, ",)" if len(outputs) == 1 else ")")
    text = "\n".join([f"def {name}({', '.join(inputs)}):", *printer.lines, *returned]) + "\n"
    _held(text, name, closed)
    return text, printer.used


def _typed(aval: ShapedArray | jax.ShapeDtypeStruct) -> tuple:
    return tuple(aval.shape), np.dtype(aval.dtype), bool(aval.weak_type)


def _shaped(atom) -> ShapedArray:
    """A program's value's abstract value, an array's, else refused."""
    if not isinstance(atom.aval, ShapedArray):
        raise Unprintable(f"no plain call writes a {type(atom.aval).__name__} value")
    return atom.aval


def _held(text: str, name: str, closed: core.Jaxpr) -> None:
    """Refuses a printed text whose outputs are not the program's in shape, dtype and weak type (a literal
    the program holds weak, printed strong, that reaches an output): what a plain call cannot write as the
    jaxpr holds it."""
    shaped = jax.eval_shape(loaded(text, name), *(
        jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)
        for aval in map(_shaped, closed.jaxpr.invars)))
    found = [_typed(leaf) for leaf in shaped]
    expected = [_typed(_shaped(atom)) for atom in closed.jaxpr.outvars]
    if found != expected:
        raise Unprintable(f"its printed outputs are {found}, the program's {expected}")


def _body(printer: _Printer, jaxpr: jcore.Jaxpr) -> None:
    """Each equation of `jaxpr` printed in order, a call's body in its place (`INLINED`)."""
    summands = ({atom for eq in jaxpr.eqns if _sum_pair(eq) for atom in eq.invars
                 if isinstance(atom, jcore.Var)} if printer.compensated else set())
    for equation in jaxpr.eqns:
        primitive = equation.primitive
        if primitive is jcore.primitives.platform_index_p:
            printer.platforms[equation.outvars[0]] = tuple(equation.params["platforms"])
            continue
        inner, operands = _inlined(printer, equation)
        if inner is not None:
            # A program's own constants with it (JAX's closed program is its jaxpr's alias).
            body: core.Jaxpr = inner.jaxpr
            consts = inner.consts
            _forget(printer, body)
            for var, value in zip(body.constvars, consts, strict=True):
                printer.names[var] = printer.fresh("c")
                printer.lines.extend(_constant(printer.names[var], np.asarray(value)))
            for var, atom in zip(body.invars, operands, strict=True):
                _alias(printer, var, atom)
            _body(printer, body)
            for var, atom in zip(equation.outvars, body.outvars, strict=True):
                _alias(printer, var, atom)
            continue
        if printer.compensated and (transform := _eft(equation, summands)) is not None:
            _compensated(printer, equation, transform)
            continue
        if getattr(lax, f"{primitive.name}_p", None) is not primitive:
            raise Unprintable(f"no plain call writes a {primitive.name} equation")
        outputs = [printer.fresh() for _ in equation.outvars]
        for var, named in zip(equation.outvars, outputs, strict=True):
            printer.names[var] = named
        arguments = [printer.atom(atom) for atom in equation.invars]
        arguments += [f"{key}={_layout(printer, value) if key == 'shape' else _param(value)}"
                      for key, value in equation.params.items()]
        # A primitive of several results binds a sequence, unpacked even where it holds one.
        target = (", ".join(outputs) + ("," if len(outputs) == 1 else "") if primitive.multiple_results
                  else outputs[0])
        printer.emit(target, f"lax.{primitive.name}_p.bind", arguments)


def _layout(printer: _Printer, value) -> str:
    """Identical emitted bodies may share extents read from their declared input shapes."""
    if not printer.dimensions or not isinstance(value, tuple) or not all(isinstance(n, int) for n in value):
        return _param(value)
    sizes = [printer.dimensions.get(n, repr(n)) for n in value]
    return f"({', '.join(sizes)}{',' if len(sizes) == 1 else ''})"


def _inlined(printer: _Printer, equation: jcore.JaxprEqn) -> tuple:
    """The body an equation is printed as in its place and the operands its inputs are, or `(None, ())`: a
    call's body (`INLINED`), or a platform choice's default branch (a `cond` on a `platform_index` that names
    one, `printer.platforms`), its index dropped."""
    if equation.primitive.name in INLINED:
        return equation.params.get("jaxpr", equation.params.get("call_jaxpr")), equation.invars
    index = equation.invars[0] if equation.primitive is lax.cond_p and equation.invars else None
    platforms = printer.platforms.get(index) if isinstance(index, jcore.Var) else None
    if platforms is not None and None in platforms:
        return equation.params["branches"][platforms.index(None)], equation.invars[1:]
    return None, ()


def _forget(printer: _Printer, body: core.Jaxpr) -> None:
    """Every value of `body` unnamed before it is printed in place, so a body inlined again (one nested
    program called twice) binds its own names, pairs and reads afresh, none of an earlier call's."""
    for var in (*body.constvars, *body.invars, *(var for equation in body.eqns for var in equation.outvars)):
        printer.names.pop(var, None)
        printer.pairs.pop(var, None)
        printer.platforms.pop(var, None)
        printer.read.discard(var)


def _alias(printer: _Printer, var: jcore.Var, atom) -> None:
    """`var` named as `atom` is: a call's input as its operand, its output as its body's."""
    if isinstance(atom, jcore.Literal):
        printer.names[var] = printer.fresh()
        printer.lines.append(f"    {printer.names[var]} = {_value(atom.val, _dtype(atom))}")
        return
    printer.names[var] = printer.names[atom]
    if atom in printer.pairs:
        printer.pairs[var] = printer.pairs[atom]


def _dtype(atom) -> np.dtype:
    """A literal's or a value's dtype, as its abstract value states it."""
    aval = atom.aval
    if not isinstance(aval, ShapedArray):
        raise Unprintable(f"no literal form of a {type(aval).__name__} value")
    return np.dtype(aval.dtype)


def _floating(var) -> bool:
    return isinstance(var.aval, ShapedArray) and jnp.issubdtype(var.aval.dtype, jnp.floating)


def _sum_pair(equation: jcore.JaxprEqn) -> bool:
    return (equation.primitive in (lax.add_p, lax.sub_p)
            and all(_floating(var) for var in (*equation.invars, *equation.outvars)))


def _eft(equation: jcore.JaxprEqn, summands: set[jcore.Var]) -> str | None:
    """The dev-time transform rules, keeping a residual as a separate word, not in a real e-class."""
    primitive = equation.primitive
    if _sum_pair(equation):
        return "sum"
    if primitive is lax.reduce_sum_p and _floating(equation.invars[0]):
        return "reduction"
    if (primitive is lax.mul_p and equation.outvars[0] in summands
            and equation.params.get("out_dtype") is None):
        left, right = equation.invars
        out = equation.outvars[0].aval
        if (isinstance(out, ShapedArray) and not out.weak_type and out.dtype in (jnp.float32, jnp.float64)
                and all(isinstance(atom.aval, ShapedArray) and not atom.aval.weak_type
                        and atom.aval.dtype == out.dtype for atom in (left, right))):
            return "product"
    return None


def _compensated(printer: _Printer, equation: jcore.JaxprEqn, transform: str) -> None:
    """Emit protected TwoSum, TwoProduct, pair addition or compensated reduction."""
    printer.used = True
    (out,) = equation.outvars
    printer.names[out] = printer.fresh()
    if transform == "reduction":
        axes = tuple(equation.params["axes"])
        printer.emit(printer.names[out], "precision.compensated_sum",
                     [printer.atom(equation.invars[0]), _param(axes)])
        return
    left, right = equation.invars
    pair = printer.fresh("p")
    printer.pairs[out] = pair
    if transform == "product":
        hi, lo = printer.fresh(), printer.fresh()
        printer.emit(f"{hi}, {lo}", "precision.two_product",
                     [printer.atom(left), printer.atom(right), "bit_split=True"])
        # The residual restores the rounded product's real value, not another product's tangent.
        printer.lines.append(f"    {pair} = ({hi}, lax.stop_gradient({lo}))")
    elif any(isinstance(atom, jcore.Var) and atom in printer.pairs for atom in (left, right)):
        printer.emit(pair, "precision.compensated_add",
                     [printer.pair(left), printer.pair(right, negated=equation.primitive is lax.sub_p)])
    else:
        left, right = printer.atom(left), printer.atom(right)
        signed_right = f"-({right})" if equation.primitive is lax.sub_p else right
        printer.emit(pair, "precision.two_sum", [left, signed_right])


def refused(closed: core.Jaxpr, *, compensated: bool = False) -> str | None:
    """Why no plain call writes `closed` (`Unprintable`'s reason), compensated where asked (a compensated
    sum returns its result strong, so a weak one's print is refused), None where it is printable."""
    try:
        source(closed, "printable", compensated=compensated)
    except Unprintable as refusal:
        return str(refusal)
    return None


def reprinted(function: Callable, operands, tree) -> Callable | None:
    """`function` over operands of `operands`' shapes as its printed text computes it (`source`, `loaded`),
    returning its outputs in their structure `tree`: what a generated module of it would compute, where a
    platform's choice takes its default branch; None where it is not printable (`refused`)."""
    closed = jax.make_jaxpr(function)(*operands)
    if refused(closed) is not None:
        return None
    flat = loaded(source(closed, "reprinted")[0], "reprinted")
    return lambda *values: jax.tree.unflatten(tree, list(flat(*values)))


def loaded(text: str, name: str) -> Callable:
    """The function `name` a printed text defines (`source`), over the imports a printed module reads."""
    from dew.nn import precision

    namespace: dict[str, object] = {"Callable": Callable, "jax": jax, "jnp": jnp, "np": np, "lax": lax,
                                    "precision": precision}
    exec(compile(text, f"<printed {name}>", "exec"), namespace)
    function = namespace[name]
    if not callable(function):
        raise TypeError(f"the printed text defines no function {name!r}")
    return function


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
    if jnp.issubdtype(dtype, jnp.floating) and not np.isnan(held):
        return f"np.array({_float(float(held))}, dtype={named})"
    if jnp.issubdtype(dtype, jnp.floating):
        # A NaN by its bits, its payload kept.
        return _bytes(held)
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
        inner = ", ".join(_param(entry) for entry in value)
        return f"({inner}{',' if len(value) == 1 else ''})"
    if isinstance(value, np.dtype) or (isinstance(value, type) and issubclass(value, np.generic)):
        return f"jnp.dtype({np.dtype(value).name!r})"
    if isinstance(value, enum.Enum) and getattr(lax, type(value).__name__, None) is type(value):
        return f"lax.{type(value).__name__}.{value.name}"
    raise Unprintable(f"no literal form of a {type(value).__name__} parameter")


def wrapped(head: str, arguments: list[str], tail: str) -> list[str]:
    """A call's (or a literal's) lines within `WIDTH`: on one where it fits, else its arguments packed onto
    lines after its head, each at eight spaces' indent, the generated code's deepest."""
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
