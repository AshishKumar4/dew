"""A program's float64 truth by replay (`float64_replay`): its operations, as traced, each run in float64
from operands widened exactly, for a truth no float64 clone of the program gives (an initializer's
draws, whose float32 bits a float64 run would draw otherwise)."""

import collections
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import _floating, as_given, equations

REPLAYED = frozenset({
    "add", "sub", "mul", "div", "neg", "max", "min", "sqrt", "exp", "expm1", "log", "erf", "erf_inv",
    "broadcast_in_dim", "convert_element_type", "gather", "iota", "reshape", "slice", "squeeze",
    "stack", "stop_gradient",
})
"""The float operations `float64_replay` evaluates in float64: those an initializer's parameters
depend on (the parallelism matrix's models', surveyed). Any other operation on a float refuses."""
FORMATTED = frozenset({"bitcast_convert_type", "nextafter", "reduce_precision"})
"""Operations whose meaning is a float format's: `float64_replay` runs them at the traced dtypes
and widens a float they return."""


def float64_replay(fn):
    """`fn` as its float64 truth: its program, as traced, with every operation on a float run in
    float64 from operands widened exactly, so no float result is rounded but where the program
    names a format. Operations on integers, bits and keys run as traced, so a draw's random bits
    and the uniform built from them are the program's own (`FORMATTED`); a float literal or constant
    is the program's value widened. The program's calls (`jit`), custom derivatives' calls and
    scans are replayed inside, each equation under the context it was traced in; a float
    operation outside `REPLAYED`, a complex one, and any other call of a program refuse by name.
    Returns a function of `fn`'s arguments whose float results are float64, for check 1's
    `truth_fn` where a float64 run of `fn` would be another program (a float64 draw reads 64-bit
    random bits). It traces `fn` as the float32 program, x64 off, whatever the caller's mode and
    however widened its arguments arrive (check 1 hands a truth widened inputs under x64), and
    rounds each argument to its traced dtype before widening it, as the program would read it. It traces
    on the host and compiles its replay itself, once a traced program (`_replay_program`), so the gate
    runs it as given."""
    def run(*arguments):
        with jax.enable_x64(new_val=False):
            graph, shapes = jax.make_jaxpr(fn, return_shape=True)(*arguments)
        native = [value if jax.dtypes.issubdtype(aval.dtype, jax.dtypes.extended)
                  else np.asarray(value).astype(aval.dtype)
                  for value, aval in zip(jax.tree.leaves(arguments), graph.in_avals, strict=True)]
        with jax.enable_x64(new_val=True):
            outputs = _replay_program(graph)(list(graph.consts), native)
        return jax.tree.unflatten(jax.tree.structure(shapes), outputs)
    return as_given(run)


_REPLAYS: collections.OrderedDict[str, Callable] = collections.OrderedDict()
"""The compiled replays of the programs last replayed (`_replay_program`), by program text."""


def _replay_program(graph):
    """`_replayed` of `graph`'s program as one jitted function of its constants and arguments, cached by
    the program's text (its equations, literals and shapes; its constants' values are arguments), so a
    traced program compiles its float64 replay once."""
    key = str(graph.jaxpr)
    if key not in _REPLAYS:
        program = graph.jaxpr
        _REPLAYS[key] = jax.jit(lambda consts, arguments: _replayed(program, consts, arguments))
        while len(_REPLAYS) > 32:
            _REPLAYS.popitem(last=False)
    _REPLAYS.move_to_end(key)
    return _REPLAYS[key]


def _widened(value):
    """`value` with a float widened exactly to float64: `widened` for one value, kept on the device,
    since inside a replayed scan body it is a tracer, which a host array cannot hold."""
    dtype = value.dtype if hasattr(value, "dtype") else np.result_type(value)
    return jnp.asarray(value, jnp.float64) if _floating(dtype) else value


def _replayed(graph, consts, arguments):
    held = {variable: _widened(value) for variable, value in
            zip((*graph.constvars, *graph.invars), (*consts, *arguments), strict=True)}

    def read(variable):
        return _widened(variable.val) if hasattr(variable, "val") else held[variable]

    for equation in graph.eqns:
        with equation.ctx.manager:
            outputs = _replayed_equation(equation, [read(variable) for variable in equation.invars])
        held.update(zip(equation.outvars, outputs, strict=True))
    return [read(variable) for variable in graph.outvars]


def _replayed_equation(equation, values):
    name, params = equation.primitive.name, equation.params
    dtypes = [variable.aval.dtype for variable in (*equation.invars, *equation.outvars)
              if hasattr(variable.aval, "dtype")]
    if any(jnp.issubdtype(dtype, jnp.complexfloating) for dtype in dtypes):
        raise NotImplementedError(f"float64_replay has no float64 rule for complex {name}")
    if name == "jit":
        return _replayed(params["jaxpr"].jaxpr, params["jaxpr"].consts, values)
    if name in ("custom_jvp_call", "custom_vjp_call"):
        return _replayed(params["call_jaxpr"].jaxpr, params["call_jaxpr"].consts, values)
    if name == "scan":
        # The body, replayed in float64 for its operands' widened dtypes, as the scan's own.
        body = params["jaxpr"]
        widened = jax.make_jaxpr(lambda *operands: _replayed(body.jaxpr, body.consts, list(operands)))(
            *(jax.ShapeDtypeStruct(variable.aval.shape, jnp.float64 if _floating(variable.aval.dtype)
                                   else variable.aval.dtype) for variable in body.jaxpr.invars))
        return equation.primitive.bind(*values, **{**params, "jaxpr": widened})
    if next(equations(params), None) is not None:
        # A call this replay does not open may hold float arithmetic its integer boundary hides.
        raise NotImplementedError(f"float64_replay does not replay inside {name}")
    if name in FORMATTED or not any(_floating(dtype) for dtype in dtypes):
        traced = [jnp.asarray(value).astype(variable.aval.dtype) if hasattr(variable.aval, "dtype")
                  and _floating(variable.aval.dtype) else value
                  for value, variable in zip(values, equation.invars, strict=True)]
        outputs = equation.primitive.bind(*traced, **equation.primitive.get_bind_params(params))
        outputs = outputs if equation.primitive.multiple_results else [outputs]
        return [_widened(output) if _floating(output.dtype) else output for output in outputs]
    if name not in REPLAYED:
        raise NotImplementedError(f"float64_replay has no float64 rule for {name}")
    widened = {key: jnp.float64 if key in ("dtype", "new_dtype") and value is not None
               and _floating(value) else value for key, value in params.items()}
    outputs = equation.primitive.bind(*values, **equation.primitive.get_bind_params(widened))
    return outputs if equation.primitive.multiple_results else [outputs]
