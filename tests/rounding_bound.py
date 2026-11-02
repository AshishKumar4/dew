"""A program's first-order rounding bound, read from its jaxpr in float64 (`rounding_bound`).

Each rounding the program makes moves a value v by at most w(v) u, u the unit roundoff: a correctly
rounded operation (an add, a product, a quotient, a root, a narrowing conversion) by its result's
magnitude, a sum of m terms in any order by (m - 1) times the sum of the terms' magnitudes, an inner
product of m terms by m times it (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., 2002,
sections 2.2, 3.1 and 4.2), and an elementary function by its implementation's ulps times its result
(`ULPS`). To first order in u, each output element y then moves by at most u times
kappa(y) = sum over the roundings of |dy / dv| w(v), the sensitivities taken in float64 by
differentiating the program with a zero added at every rounded value (a tap). A rearrangement of the
program, XLA's fusions and sum orders, makes no rounding outside these bounds. The bound is first
order: it holds while u kappa stays below the output's own size, where the rounding is a perturbation.
"""

import jax
import jax.numpy as jnp
import numpy as np

EXACT = frozenset({
    "neg", "abs", "max", "min", "select_n", "reshape", "broadcast_in_dim", "transpose", "squeeze",
    "expand_dims", "slice", "dynamic_slice", "dynamic_update_slice", "concatenate", "pad", "gather",
    "copy", "copy_p", "rev", "iota", "sign", "floor", "ceil", "round", "reduce_max", "reduce_min",
    "argmax", "argmin", "eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not", "xor", "clamp",
    "is_finite", "stop_gradient", "optimization_barrier", "cummax", "cummin", "sort",
    "shift_left", "shift_right_logical", "shift_right_arithmetic", "population_count", "rem",
    "name", "pjit_name", "nextafter", "random_bits", "random_seed", "random_wrap", "random_unwrap",
    "threefry2x32", "split", "scatter", "stack", "unstack", "bitcast_convert_type", "device_put",
    "sharding_constraint", "mesh_cast", "pvary", "reshard", "tie_in", "reduce_and", "reduce_or",
    "select_and_gather_add_exact"})
"""Operations whose float results are their operands' values moved or selected, or exact: no rounding.
An integer result rounds nothing either."""

ONCE = frozenset({"add", "add_any", "sub", "mul", "div", "sqrt", "square", "convert_element_type",
                  "reduce_precision"})
"""IEEE 754's correctly rounded operations: one rounding, at most u times the result."""

ULPS = {"exp": 1.42, "log": 2.05, "log1p": 3.64, "expm1": 1.21, "logistic": 2.49, "tanh": 4.9,
        "erf": 6.71, "rsqrt": 2.8}
"""Each elementary function's error bound, in units of u times its true result: the largest of XLA
CPU's float32 implementations over every normal float32 input whose result is a normal float32, read
exhaustively against float64 references (2^32 inputs each, 11-02), rounded up at the second decimal.
XLA CPU flushes subnormal inputs to zero, so the bound holds where no elementary function reads one."""

SUMMED = frozenset({"dot_general", "reduce_sum", "cumsum", "scatter-add"})
"""Sums of terms, bounded by their terms' magnitudes."""

CALLS = ("pjit", "jit", "closed_call", "core_call", "remat", "remat2", "checkpoint", "custom_jvp_call",
         "custom_vjp_call", "custom_vjp_call_jaxpr", "custom_lin")
"""Calls whose bodies are evaluated inline, as the program's own operations."""


def rounding_bound(fn, *args, batch: int = 64):
    """`fn(*args)` in float64, and each output element's kappa, its first-order rounding sensitivity
    (module docstring): `fn`'s rounding in a unit roundoff u moves an element by at most u kappa, to
    first order. `args` are widened to float64; `fn`'s float32 conversions and constants are evaluated
    in float64, each a rounding of the program it is. Every output element's kappa is one reverse pass
    of the tapped program, `batch` of them at a time."""
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(_widened, args)
        closed = jax.make_jaxpr(fn)(*wide)
        flat = [jnp.asarray(leaf) for leaf in jax.tree.leaves(wide)]
        weights: list = []
        outputs = _run(closed.jaxpr, closed.consts, flat, None, weights)
        shapes = [np.shape(value) for value in outputs]
        sizes = [int(np.prod(shape)) for shape in shapes]

        def tapped(taps):
            tap = iter(taps)
            return jnp.concatenate([jnp.ravel(value) for value in
                                    _run(closed.jaxpr, closed.consts, flat, tap, None)])

        zeros = [jnp.zeros_like(weight) for weight in weights]
        magnitudes = [jnp.asarray(weight) for weight in weights]

        @jax.jit
        def sensitivities(basis):
            # The tapped program linearized at no tap and transposed, each output a reverse pass.
            _, tangent = jax.linearize(tapped, zeros)
            (cotangents,) = jax.vmap(jax.linear_transpose(tangent, zeros))(basis)
            # A rounding no output reads has no cotangent.
            return sum(jnp.sum(jnp.abs(cotangent) * magnitude, axis=tuple(range(1, cotangent.ndim)))
                       for cotangent, magnitude in zip(cotangents, magnitudes, strict=True)
                       if cotangent is not None)

        total = sum(sizes)
        kappa = np.zeros(total)
        for start in range(0, total, batch):
            rows = jnp.arange(start, min(start + batch, total))
            kappa[start:start + len(rows)] = np.asarray(sensitivities(jax.nn.one_hot(rows, total,
                                                                                      dtype=jnp.float64)))
    bounds, at = [], 0
    for shape, size in zip(shapes, sizes, strict=True):
        bounds.append(kappa[at:at + size].reshape(shape))
        at += size
    structure = jax.tree.structure(jax.eval_shape(fn, *args))
    return (structure.unflatten([np.asarray(value) for value in outputs]), structure.unflatten(bounds))


def _widened(leaf):
    value = np.asarray(leaf)
    return value.astype(np.float64) if np.issubdtype(value.dtype, np.floating) else value


def _wide(dtype):
    return jnp.float64 if dtype is not None and jnp.issubdtype(dtype, jnp.floating) else dtype


def _body(equation):
    """The jaxpr and constants a call runs (`CALLS`), or None."""
    if equation.primitive.name not in CALLS:
        return None
    for name in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        body = equation.params.get(name)
        if body is not None:
            return (body.jaxpr, body.consts) if hasattr(body, "consts") else (body, [])
    return None


def _run(jaxpr, consts, args, taps, weights):
    """`jaxpr`'s outputs from `args`, in float64: each rounded value plus the next of `taps` where taps
    are given, else each rounded value's bound w(v) appended to `weights` (`rounding_bound`)."""
    from jax.extend.core import Literal

    values = dict(zip(jaxpr.constvars + jaxpr.invars, [*consts, *args], strict=True))

    def read(var):
        if isinstance(var, Literal):
            value = np.asarray(var.val)
            return value.astype(np.float64) if np.issubdtype(value.dtype, np.floating) else value
        return values[var]

    for equation in jaxpr.eqns:
        inputs = [read(var) for var in equation.invars]
        outputs = _applied(equation, inputs, taps, weights)
        assert all(value is not None for value in outputs), (equation.primitive.name, sorted(equation.params))
        values.update(zip(equation.outvars, outputs, strict=True))
    return [read(var) for var in jaxpr.outvars]


def _applied(equation, inputs, taps, weights):
    """`equation`'s outputs, each rounded one tapped or weighed (`_run`)."""
    name, params = equation.primitive.name, dict(equation.params)
    body = _body(equation)
    if body is not None:
        return _run(*body, inputs, taps, weights)
    if name == "scan":
        return _scanned(params, inputs, taps, weights)
    if name in ("while", "cond", "shard_map"):
        raise ValueError(f"the rounding bound unrolls no {name}: it reads straight-line programs and scans")
    for key in ("new_dtype", "preferred_element_type", "dtype"):
        if key in params:
            params[key] = _wide(params[key])
    if name == "reduce_precision":
        # The float64 evaluation keeps its value, the rounding it is weighed.
        outputs = list(inputs)
    else:
        outputs = equation.primitive.bind(*inputs, **equation.primitive.get_bind_params(params))
        outputs = list(outputs) if equation.primitive.multiple_results else [outputs]
    floats = [jnp.issubdtype(var.aval.dtype, jnp.floating) for var in equation.outvars]
    if not any(floats) or name in EXACT:
        return outputs
    if name not in ONCE and name not in ULPS and name not in SUMMED and name not in ("integer_pow",
                                                                                     "one_minus_square"):
        raise ValueError(f"the rounding bound has no rule for {name}: add it to EXACT, ONCE, ULPS or SUMMED")
    if name == "convert_element_type" and not _narrows(equation, inputs):
        return outputs
    rounded = []
    for value, is_float in zip(outputs, floats, strict=True):
        if not is_float:
            rounded.append(value)
        elif taps is not None:
            rounded.append(value + next(taps))
        else:
            weights.append(_weight(equation, name, params, inputs, value))
            rounded.append(value)
    return rounded


def _narrows(equation, inputs) -> bool:
    """Whether a conversion rounds: to a float narrower than its operand. One from an integer is
    exact below 2^24, as float32 holds every integer there, which the programs' integers are (refused
    past it)."""
    source, target = equation.invars[0].aval.dtype, equation.outvars[0].aval.dtype
    if not jnp.issubdtype(target, jnp.floating):
        return False
    if not jnp.issubdtype(source, jnp.floating):
        value = inputs[0]
        concrete = not isinstance(value, jax.core.Tracer)
        if concrete and np.any(np.abs(np.asarray(value, np.float64)) >= 2 ** 24):
            raise ValueError("an integer of 2^24 or more converts to float32 with rounding the bound does "
                             "not weigh")
        return False
    return jnp.finfo(target).bits < jnp.finfo(source).bits


def _weight(equation, name, params, inputs, value):
    """The most one rounding at `equation` moves its result `value` (module docstring)."""
    magnitude = jnp.abs(value)
    if name in ONCE:
        return magnitude
    if name in ULPS:
        return ULPS[name] * magnitude
    if name == "integer_pow":
        exponent = abs(int(params["y"]))
        return max(exponent - 1, 1) * magnitude
    if name == "one_minus_square":
        # The square rounds by its magnitude, which the difference carries, and the difference by its own.
        return jnp.square(inputs[0]) + magnitude
    operands = [jnp.abs(operand) for operand in inputs]
    terms = equation.primitive.bind(*operands, **equation.primitive.get_bind_params(params))
    if name == "dot_general":
        (contracting, _), _ = params["dimension_numbers"]
        count = int(np.prod([equation.invars[0].aval.shape[axis] for axis in contracting]))
        return count * terms
    if name == "reduce_sum":
        count = int(np.prod([equation.invars[0].aval.shape[axis] for axis in params["axes"]]))
        return max(count - 1, 0) * terms
    if name == "scatter-add":
        # Each element adds its operand's value and at most every scattered update, one rounding each.
        numbers = params["dimension_numbers"]
        updates = equation.invars[2].aval.shape
        count = int(np.prod([size for axis, size in enumerate(updates)
                             if axis not in numbers.update_window_dims]))
        return count * terms
    count = equation.invars[0].aval.shape[params["axis"]]
    return max(count - 1, 0) * terms


def _scanned(params, inputs, taps, weights):
    """A scan's outputs, its body run step by step as the program's own operations (`scanned_steps`)."""
    from float64_terms import scanned_steps

    body = params["jaxpr"]

    def step(_, lanes):
        outputs = _run(body.jaxpr, body.consts, [value for (value,) in lanes], taps, weights)
        return [(value,) for value in outputs]

    return [value for (value,) in scanned_steps(params, [(value,) for value in inputs], step, lanes=1)]
