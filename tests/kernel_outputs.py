"""Every Pallas launch's declared outputs as it writes them (the unwritten-output audit, Main 11-08: a
declared output left unwritten on some path trips `jax_debug_nans` wherever the launch runs alone, and holds
whatever memory held before). `launched_outputs` traces a call, finds each `pallas_call` in its program,
nested bodies included (a `pl.kernel`'s, Mosaic GPU's `plgpu.kernel`'s, binds `mpmd_map`: it is a launch
too), and runs that launch alone on the values that reach it, so its every output is seen,
those the caller slices away among them. Interpreted, an output the kernel never writes holds NaN (the
generic interpreter's `uninitialized_value`, Mosaic GPU's and Mosaic TPU's `uninitialized_memory='nan'`),
so on finite operands a NaN is an unwritten element."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from float64_terms import scanned_steps
from jax._src.pallas import mpmd, pallas_call as pallas_call_module
from jax.extend import core

from dew.nn import programs

LAUNCHES = (pallas_call_module.pallas_call_p, mpmd.mpmd_map_p)
"""The primitives a Pallas launch binds: `pallas_call`, and `mpmd_map` (`pl.kernel`, `plgpu.kernel`)."""


def launched_outputs(function, *args) -> list[tuple[str, list[np.ndarray]]]:
    """Each launch (`LAUNCHES`) `function(*args)` runs, by its name, with every output it declares: the
    launch run alone under `jax.jit` on the operands its program gives it."""
    closed = jax.make_jaxpr(function)(*args)
    found: list[tuple[str, list[np.ndarray]]] = []
    _evaluated(closed.jaxpr, closed.consts, jax.tree.leaves(args), found)
    return found


class Unwritten(NamedTuple):
    """A launch's output holding elements it never wrote (`unwritten`): the launch's name, the output's index
    and shape, and how many of its elements."""
    launch: str
    output: int
    shape: tuple[int, ...]
    elements: int

    def __str__(self) -> str:
        return f"{self.launch} output {self.output} {self.shape}: {self.elements} unwritten"


def unwritten(found: list[tuple[str, list[np.ndarray]]]) -> list[Unwritten]:
    """`launched_outputs`' launches with a floating output holding NaN, bfloat16 and the narrower floats
    among them (JAX's floating kinds, which NumPy's own hierarchy does not hold), each output with how many
    of its elements: an element the launch never wrote, on finite operands."""
    return [Unwritten(name, index, output.shape, int(np.isnan(output).sum()))
            for name, outputs in found for index, output in enumerate(outputs)
            if jnp.issubdtype(output.dtype, jnp.floating) and np.isnan(output).any()]


def _evaluated(jaxpr: core.Jaxpr, consts, args, found: list) -> list:
    env = dict(zip((*jaxpr.constvars, *jaxpr.invars), (*consts, *args), strict=True))

    def read(variable):
        return variable.val if isinstance(variable, core.Literal) else env[variable]

    for equation in jaxpr.eqns:
        values = [read(variable) for variable in equation.invars]
        name = equation.primitive.name
        if equation.primitive in LAUNCHES:
            launched = jax.jit(lambda *operands, equation=equation: equation.primitive.bind(
                *operands, **equation.params))(*values)
            label = equation.params.get("name_and_src_info", equation.params.get("name", name))
            found.append((str(label),
                          [np.asarray(jax.device_get(output)) for output in launched]))
            outs = list(launched)
        elif name in LOOPS and any(_launches(inner) for inner in programs.bodies(equation)):
            outs = LOOPS[name](equation, values, found)
        elif (body := programs.called(equation)) is not None:
            # A call runs its program on its operands one for one (`programs.called`): a jit's, a contracted
            # derivative's, a custom rule's, a Mosaic GPU kernel's `custom_vmap_call` around its `mpmd_map`.
            outs = _evaluated(body, body.consts, values, found)
        else:
            if any(_launches(inner) for inner in programs.bodies(equation)):
                raise NotImplementedError(f"a launch inside {name}'s body, which the audit does not walk "
                                          "(a loop or branch `LOOPS` runs, a call `programs.called` reads)")
            outs = equation.primitive.bind(*values, **equation.primitive.get_bind_params(equation.params))
            outs = list(outs) if equation.primitive.multiple_results else [outs]
        env.update(zip(equation.outvars, outs, strict=True))
    return [read(variable) for variable in jaxpr.outvars]


def _closed(body) -> tuple[core.Jaxpr, list]:
    """A body's jaxpr and its constants, closed or open."""
    return (body.jaxpr, body.consts) if isinstance(body, core.ClosedJaxpr) else (body, [])


def _while(equation, values: list, found: list) -> list:
    """A `while` loop run trip by trip, each trip's launches run alone (`_evaluated`)."""
    params = equation.params
    cond_consts, body_consts = params["cond_nconsts"], params["body_nconsts"]
    cond_args, body_args, carry = (values[:cond_consts], values[cond_consts:cond_consts + body_consts],
                                   values[cond_consts + body_consts:])
    cond, cond_closed = _closed(params["cond_jaxpr"])
    body, body_closed = _closed(params["body_jaxpr"])
    while bool(_evaluated(cond, cond_closed, [*cond_args, *carry], found)[0]):
        carry = _evaluated(body, body_closed, [*body_args, *carry], found)
    return carry


def _scan(equation, values: list, found: list) -> list:
    """A `scan` run step by step as the gate's one program walker steps it (`float64_terms.scanned_steps`:
    its operands split, its carries threaded and its outputs stacked there), each step's launches run alone
    (`_evaluated`)."""
    body, closed = _closed(equation.params["jaxpr"])

    def step(_, operands):
        return [(out,) for out in _evaluated(body, closed, [value for (value,) in operands], found)]

    return [value for (value,) in scanned_steps(equation, [(value,) for value in values], step, lanes=1)]


def _cond(equation, values: list, found: list) -> list:
    """A `cond`'s branch its index takes, its launches run alone (`_evaluated`)."""
    index, operands = values[0], values[1:]
    branches = equation.params["branches"]
    branch, closed = _closed(branches[int(np.clip(int(index), 0, len(branches) - 1))])
    return _evaluated(branch, closed, operands, found)


LOOPS = {"while": _while, "scan": _scan, "cond": _cond}
"""The control flow whose bodies the audit runs as the program would, by the primitive's name: a loop trip by
trip, a branch as its index chooses, each launch inside run alone; a launch in any other body is refused."""


def _launches(jaxpr: core.Jaxpr) -> bool:
    """Whether `jaxpr` runs a launch (`LAUNCHES`), at any depth."""
    return any(equation.primitive in LAUNCHES or any(_launches(inner) for inner in programs.bodies(equation))
               for equation in jaxpr.eqns)
