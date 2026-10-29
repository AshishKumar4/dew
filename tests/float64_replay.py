"""A program's float64 truth by replay (`float64_replay`): its operations, as traced, each run in float64
from operands widened exactly, for a truth no float64 clone of the program gives (an initializer's
draws, whose float32 bits a float64 run would draw otherwise)."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import _floating, as_given, equations

REPLAYED = frozenset({
    "add", "sub", "mul", "div", "neg", "max", "min", "sqrt", "square", "exp", "expm1", "log", "erf",
    "erf_inv", "sin", "cos", "rsqrt", "integer_pow", "logistic", "ne", "is_finite", "reduce_sum",
    "reduce_max", "dot_general", "conv_general_dilated", "broadcast_in_dim", "dynamic_slice", "split",
    "concatenate", "convert_element_type", "gather", "iota", "reshape", "select_n", "slice", "squeeze",
    "stack", "stop_gradient", "transpose",
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
    scans are replayed inside, as are a rematerialized call's (a remat module's init), the branch a
    conditional takes and a reduction's combining body, each equation under the context it was
    traced in; a float
    operation outside `REPLAYED`, a complex one, and any other call of a program refuse by name.
    Only what the results read is replayed: an initializer's forward pass, which a module's init
    traces and whose outputs it drops, never runs (`_read`).
    Returns a function of `fn`'s arguments whose float results are float64, for check 1's
    `truth_fn` where a float64 run of `fn` would be another program (a float64 draw reads 64-bit
    random bits). It traces `fn` as the float32 program, x64 off, whatever the caller's mode and
    however widened its arguments arrive (check 1 hands a truth widened inputs under x64), and
    rounds each argument to its traced dtype before widening it, as the program would read it. It runs
    eagerly, each operation its own float64 dispatch, so its bits are each operation's own (a
    compiled replay's fusions would move them by ulps), and the gate runs it as given."""
    def run(*arguments):
        with jax.enable_x64(new_val=False):
            graph, shapes = jax.make_jaxpr(fn, return_shape=True)(*arguments)
        native = [value if jax.dtypes.issubdtype(aval.dtype, jax.dtypes.extended)
                  else np.asarray(value).astype(aval.dtype)
                  for value, aval in zip(jax.tree.leaves(arguments), graph.in_avals, strict=True)]
        with jax.enable_x64(new_val=True):
            outputs = _replayed(graph.jaxpr, graph.consts, native)
        return jax.tree.unflatten(jax.tree.structure(shapes), outputs)
    return as_given(run)


def _widened(value):
    """`value` with a float widened exactly to float64: `widened` for one value, kept on the device,
    since inside a replayed scan body it is a tracer, which a host array cannot hold."""
    dtype = value.dtype if hasattr(value, "dtype") else np.result_type(value)
    return jnp.asarray(value, jnp.float64) if _floating(dtype) else value


def _read(graph, used):
    """The equations of `graph` its results read, those of `used` (each result's), each with which
    of its own results are read, in order: the rest, dead code, is not replayed."""
    wanted = {variable for variable, kept in zip(graph.outvars, used, strict=True)
              if kept and not hasattr(variable, "val")}
    read = []
    for equation in reversed(graph.eqns):
        reading = [variable in wanted for variable in equation.outvars]
        if any(reading):
            read.append((equation, reading))
            wanted.update(variable for variable in equation.invars if not hasattr(variable, "val"))
    return read[::-1]


def _replayed(graph, consts, arguments, used=None):
    """`graph`'s results replayed (`_replayed_equation`), each of `used` (every one by default); a
    result not used is None."""
    used = [True] * len(graph.outvars) if used is None else used
    held = {variable: _widened(value) for variable, value in
            zip((*graph.constvars, *graph.invars), (*consts, *arguments), strict=True)}

    def read(variable):
        return _widened(variable.val) if hasattr(variable, "val") else held[variable]

    for equation, reading in _read(graph, used):
        with equation.ctx.manager:
            outputs = _replayed_equation(equation, [read(variable) for variable in equation.invars],
                                         reading)
        held.update(zip(equation.outvars, outputs, strict=True))
    return [read(variable) if kept else None for variable, kept in zip(graph.outvars, used, strict=True)]


def _wide(dtype):
    """`dtype`, a float widened to float64."""
    return jnp.float64 if _floating(dtype) else dtype


def _carried(body, consts: int, carries: int, used):
    """Which of a scan body's results its used results read: a used carry or output, and every carry
    a read result reads back in through the next step, until none is added."""
    reading = list(used)
    while True:
        read = {variable for equation, _ in _read(body, reading) for variable in equation.invars}
        read |= {variable for variable, kept in zip(body.outvars, reading, strict=True)
                 if kept and not hasattr(variable, "val")}
        added = [index for index in range(carries)
                 if not reading[index] and body.invars[consts + index] in read]
        if not added:
            return reading
        for index in added:
            reading[index] = True


def _replayed_equation(equation, values, used):
    name, params = equation.primitive.name, equation.params
    dtypes = [variable.aval.dtype for variable in (*equation.invars, *equation.outvars)
              if hasattr(variable.aval, "dtype")]
    if any(jnp.issubdtype(dtype, jnp.complexfloating) for dtype in dtypes):
        raise NotImplementedError(f"float64_replay has no float64 rule for complex {name}")
    if name == "jit":
        return _replayed(params["jaxpr"].jaxpr, params["jaxpr"].consts, values, used)
    if name in ("custom_jvp_call", "custom_vjp_call"):
        return _replayed(params["call_jaxpr"].jaxpr, params["call_jaxpr"].consts, values, used)
    if name in ("checkpoint", "remat2"):
        return _replayed(params["jaxpr"], (), values, used)
    if name == "reduce":
        # The combining body, replayed in float64 over the widened operands' pairs.
        body, count = params["jaxpr"], len(values) // 2

        def combine(left, right):
            return tuple(_replayed(body.jaxpr, body.consts, [*left, *right]))
        operands, initial = tuple(values[:count]), tuple(values[count:])
        return list(jax.lax.reduce(operands, initial, combine, params["dimensions"]))
    if name == "cond":
        # Replayed eagerly, the branch index is concrete: the branch the program takes, alone.
        branch = params["branches"][int(values[0])]
        return _replayed(branch.jaxpr, branch.consts, values[1:], used)
    if name in ("name", "optimization_barrier"):
        return list(values)
    if name == "scan":
        # The body, replayed in float64 for its operands' widened dtypes, as the scan's own: of its
        # results those the scan's used results read through every step (`_carried`), the rest zeros.
        body = params["jaxpr"]
        # The constants and carries the step reads first, by count or by their trees' leaves (`ft_in`).
        consts, carries = ((params["num_consts"], params["num_carry"]) if "num_carry" in params
                           else (len(params["ft_in"][0]), len(params["ft_in"][1])))
        reading = _carried(body.jaxpr, consts, carries, used)

        def step(*operands):
            outputs = _replayed(body.jaxpr, body.consts, list(operands), reading)
            return [jnp.zeros(variable.aval.shape, _wide(variable.aval.dtype)) if output is None else output
                    for output, variable in zip(outputs, body.jaxpr.outvars, strict=True)]
        widened = jax.make_jaxpr(step)(*(jax.ShapeDtypeStruct(variable.aval.shape, _wide(variable.aval.dtype))
                                         for variable in body.jaxpr.invars))
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
    widened = {key: jnp.float64 if key in ("dtype", "new_dtype", "preferred_element_type")
               and value is not None and _floating(value) else value for key, value in params.items()}
    outputs = equation.primitive.bind(*values, **equation.primitive.get_bind_params(widened))
    return outputs if equation.primitive.multiple_results else [outputs]
