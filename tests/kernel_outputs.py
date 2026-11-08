"""Every Pallas launch's declared outputs as it writes them (the unwritten-output audit, Main 11-08: a
declared output left unwritten on some path trips `jax_debug_nans` wherever the launch runs alone, and holds
whatever memory held before). `launched_outputs` traces a call, finds each `pallas_call` in its program,
nested bodies included (a `pl.kernel`'s, Mosaic GPU's `plgpu.kernel`'s, binds `mpmd_map`: it is a launch
too), and runs that launch alone on the values that reach it, so its every output is seen,
those the caller slices away among them. Interpreted, an output the kernel never writes holds NaN (the
generic interpreter's `uninitialized_value`, Mosaic GPU's and Mosaic TPU's `uninitialized_memory='nan'`),
so on finite operands a NaN is an unwritten element."""

import jax
import numpy as np
from jax._src.pallas import mpmd, pallas_call as pallas_call_module
from jax.extend import core

LAUNCHES = (pallas_call_module.pallas_call_p, mpmd.mpmd_map_p)
"""The primitives a Pallas launch binds: `pallas_call`, and `mpmd_map` (`pl.kernel`, `plgpu.kernel`)."""

NESTED = {"jit": "jaxpr", "pjit": "jaxpr", "closed_call": "call_jaxpr", "core_call": "call_jaxpr",
          "marked_call": "call_jaxpr", "remat": "jaxpr", "checkpoint": "jaxpr",
          "custom_jvp_call": "call_jaxpr", "custom_vjp_call": "call_jaxpr",
          "custom_vjp_call_jaxpr": "fun_jaxpr", "custom_vmap_call": "call"}
"""The primitives whose body runs once on their operands, and the parameter holding it: walked into
(`jit` is a nested `jax.jit`'s; `marked_call` a contracted derivative's, `dew.nn.derivatives`;
`custom_vmap_call` wraps a Mosaic GPU kernel's `mpmd_map`). A launch inside any other body (a loop's, a
branch's) is refused, so no launch goes unread."""


def launched_outputs(function, *args) -> list[tuple[str, list[np.ndarray]]]:
    """Each launch (`LAUNCHES`) `function(*args)` runs, by its name, with every output it declares: the
    launch run alone under `jax.jit` on the operands its program gives it."""
    closed = jax.make_jaxpr(function)(*args)
    found: list[tuple[str, list[np.ndarray]]] = []
    _evaluated(closed.jaxpr, closed.consts, jax.tree.leaves(args), found)
    return found


def unwritten(found: list[tuple[str, list[np.ndarray]]]) -> list[str]:
    """`launched_outputs`' launches with a floating output holding NaN, each with the output's index and how
    many of its elements: an element the launch never wrote, on finite operands."""
    return [f"{name} output {index} {output.shape}: {int(np.isnan(output).sum())} unwritten"
            for name, outputs in found for index, output in enumerate(outputs)
            if np.issubdtype(output.dtype, np.floating) and np.isnan(output).any()]


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
        elif name in NESTED and NESTED[name] in equation.params:
            body = equation.params[NESTED[name]]
            inner, inner_consts = ((body.jaxpr, body.consts) if isinstance(body, core.ClosedJaxpr)
                                   else (body, ()))
            # A custom rule's body takes its operands past the rule's own (`num_consts` leading ones too).
            outs = _evaluated(inner, inner_consts, values[len(values) - len(inner.invars):], found)
        else:
            if any(_launches(inner) for inner in _bodies(equation)):
                raise NotImplementedError(f"a launch inside {name}'s body, which the audit does not walk "
                                          f"(`NESTED`)")
            outs = equation.primitive.bind(*values, **equation.primitive.get_bind_params(equation.params))
            outs = list(outs) if equation.primitive.multiple_results else [outs]
        env.update(zip(equation.outvars, outs, strict=True))
    return [read(variable) for variable in jaxpr.outvars]


def _bodies(equation) -> list[core.Jaxpr]:
    """The jaxprs an equation's parameters hold, closed or open, in tuples too (a branch's)."""
    found = []
    for value in equation.params.values():
        for each in value if isinstance(value, tuple | list) else (value,):
            if isinstance(each, core.ClosedJaxpr):
                found.append(each.jaxpr)
            elif isinstance(each, core.Jaxpr):
                found.append(each)
    return found


def _launches(jaxpr: core.Jaxpr) -> bool:
    """Whether `jaxpr` runs a launch (`LAUNCHES`), at any depth."""
    return any(equation.primitive in LAUNCHES or any(_launches(inner) for inner in _bodies(equation))
               for equation in jaxpr.eqns)
