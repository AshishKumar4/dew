"""The derivative contracts' harness (ledger 2641): each custom-derivative seam's orders (`orders`), the
contract a seam is held to (`assert_contract`), the draws at float32's exponent edges (`EDGES`,
`normal_draw`) and the orders a seam may not take (`FORWARD_ORDERS`, `REVERSE_ONLY` and the rest). The
contracts themselves are tests/test_contracts_*.py, one a owning area, and tests/test_derivative_seams.py
holds every seam to one.

Float-range and nested-derivative contracts of Dew's custom-derivative seams (ledger row 2283, theme
G12): each seam's primal, its forward and reverse derivatives, a forward derivative of its gradient and a
second forward derivative, the last two again under a rematerialized call (`jax.checkpoint`), run at a
normal draw and at draws at float32's exponent edges. Each order runs (a custom derivative's tangent tree
and dtypes as JAX requires: rows 1861 and 1862, a boolean residual's tangent and a cotangent tree of the
wrong length), and each of its leaves is finite wherever the same order's float64 run is a normal float32
(row 1768, a contraction that overflowed before its division). Behaviour alone: no route, structure or bits
are read, and each seam is its public call."""


import jax
import jax.numpy as jnp
import numpy as np
from reference_error import widened

TINY, LARGEST = float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).max)


class LessFinite(AssertionError):
    """A seam's order not finite where its float64 run is a normal float32 (and the plain formula's is
    finite): what the contract refuses, apart from an order that does not run."""


def normal_float32(values) -> np.ndarray:
    """Whether each float64 value is a float32 a few roundings inside its normal range: from just above
    the least normal to just below the largest, or zero."""
    size = np.abs(np.asarray(values, np.float64))
    return (size == 0) | ((size >= TINY * (1 + 2.0**-16)) & (size <= LARGEST * (1 - 2.0**-16)))


def pulled(fn, primals, cotangent):
    """`fn`'s reverse derivative at `primals` pulling back `cotangent` in the output's own dtype."""
    out, pull = jax.vjp(fn, *primals)
    return pull(jax.tree.map(lambda given, made: jnp.asarray(given, made.dtype), cotangent, out))


FORWARD_ORDERS = ("jvp", "second jvp", "jvp of grad", "jvp of grad, remat", "second jvp, remat",
                  "jvp of grad, nested jit", "jvp of grad, scanned")
"""The orders a forward derivative takes part in: those a custom VJP, which has no forward mode, does not
take (a row names each such seam's gap)."""


GRADIENT_ORDERS = ("jvp of grad", "jvp of grad, remat", "jvp of grad, nested jit", "jvp of grad, scanned",
                   "grad of grad")
"""The orders that differentiate a gradient."""


REVERSE_ORDERS = ("vjp", *GRADIENT_ORDERS)
"""The orders a reverse derivative takes part in: those a forward mode alone, whose reverse mode is another
seam's, does not take."""


OWN_FORWARD_ORDERS = ("jvp", "second jvp", "second jvp, remat")
"""The orders that take a forward derivative of the seam itself, not of its gradient: those a custom VJP whose
backward JAX differentiates does not run."""


REVERSE_ONLY = (*FORWARD_ORDERS, "grad of grad")
"""The orders a reverse-only derivative over kernels with none of their own does not run (`jax.custom_vjp`):
every order through a forward derivative, and reverse over reverse, its backward kernels' own derivative
(a fused attention kernel's, ledger 2654, which `forward_mode_attention` serves forward; a recurrence
port's, ledger 2655)."""


def orders(fn, primals, direction, cotangent):
    """`fn` at each order the contract holds, by name: the primal, a forward and a reverse derivative, a
    forward derivative of the gradient of the output's sum against `cotangent` and a second forward
    derivative, the last two again with `fn` rematerialized; and the gradient's forward derivative traced
    as programs nest it (ledger 2641): under a jit within a jit, within a scan's body, and a reverse
    derivative of the gradient along `direction` (reverse over reverse)."""
    def scalar(f):
        return lambda *x: jnp.sum(f(*x) * cotangent)

    def forward(f):
        return lambda *x: jax.jvp(f, x, direction)[1]

    def gradient(f):
        return jax.grad(scalar(f), argnums=tuple(range(len(primals))))

    remat = jax.checkpoint(fn)
    return {
        "primal": lambda: fn(*primals),
        "jvp": lambda: forward(fn)(*primals),
        "vjp": lambda: pulled(fn, primals, cotangent),
        "jvp of grad": lambda: jax.jvp(gradient(fn), primals, direction)[1],
        "second jvp": lambda: jax.jvp(forward(fn), primals, direction)[1],
        "jvp of grad, remat": lambda: jax.jvp(gradient(remat), primals, direction)[1],
        "second jvp, remat": lambda: jax.jvp(forward(remat), primals, direction)[1],
        "jvp of grad, nested jit": lambda: jax.jit(
            lambda *x: jax.jvp(jax.jit(gradient(fn)), x, direction)[1])(*primals),
        "jvp of grad, scanned": lambda: jax.lax.scan(
            lambda carry, _: (carry, jax.jvp(gradient(fn), primals, direction)[1]), 0, None, length=1)[1],
        "grad of grad": lambda: jax.grad(
            lambda *x: sum(jnp.sum(jnp.asarray(along, part.dtype) * part)
                           for along, part in zip(direction, gradient(fn)(*x), strict=True)),
            argnums=tuple(range(len(primals))))(*primals),
    }


def ran(order, label: str, name: str):
    """`order()`'s leaves as arrays, a failure to run named by the seam and the order that raised it."""
    try:
        return jax.tree.map(np.asarray, order())
    except Exception as error:
        raise AssertionError(f"{label}: its {name} does not run: {type(error).__name__}: {error}") from error


def assert_contract(fn, primals, direction, cotangent, label: str, *, truth_fn=None, plain_fn=None,
                    held: tuple[str, ...] | None = None, without: tuple[str, ...] = ()) -> None:
    """Every order of `fn` runs at the draw, and each leaf is finite wherever the same order of its float64
    run is a normal float32 and, given `plain_fn` (the plain float32 formula, JAX's own derivatives of it),
    the plain formula's is finite: no less finite than the plain arithmetic (row 1768, a contraction Dew
    overflowed where the plain formula did not), where float32 itself overflows a partial sum the float64
    run cancels. The float64 run is `fn`'s own, or `truth_fn`'s, the plain float64 formula, where `fn`
    computes in a fixed precision whatever its inputs' (a declared dot algorithm). `held`, where given, names
    the orders a decision its caller cites holds the call to: each order it drops must break, so the
    exemption ends when the call holds it. `without` names orders the contract does not take yet, each
    joining with its row's fix (a row's head brings its own test)."""

    def orders_of(f, *drawn):
        return {name: order for name, order in orders(f, *drawn).items() if name not in without}

    narrow = orders_of(fn, primals, direction, cotangent)
    plain_orders = {} if plain_fn is None else orders_of(plain_fn, primals, direction, cotangent)
    plain = {name: ran(order, f"{label}, plainly", name) for name, order in plain_orders.items()}
    with jax.enable_x64(new_val=True):
        wide_primals, wide_direction, wide_cotangent = widened((primals, direction, cotangent))
        wide = orders_of(truth_fn or fn, wide_primals, wide_direction, wide_cotangent)
        truths = {name: ran(order, f"{label} in float64", name) for name, order in wide.items()}
    for name, order in narrow.items():
        values = jax.tree.leaves(ran(order, label, name))
        plains = jax.tree.leaves(plain[name]) if plain else [None] * len(values)
        broken = []
        for index, (value, truth, plainly) in enumerate(zip(values, jax.tree.leaves(truths[name]), plains,
                                                            strict=True)):
            if not np.issubdtype(value.dtype, np.inexact):
                continue
            kept = normal_float32(truth) & np.isfinite(truth)
            if plainly is not None:
                kept &= np.isfinite(plainly)
            lost = kept & ~np.isfinite(value)
            if lost.any():
                broken.append(f"{label}, {name}, leaf {index}: {int(lost.sum())} values not finite where "
                              f"float64's are normal float32s, e.g. {value[lost][:3]}")
        if held is not None and name not in held:
            if not broken:
                raise AssertionError(f"{label}, {name}: holds, so the exemption that drops it is over")
        elif broken:
            raise LessFinite(broken[0])


EDGES = {"normal": 1.0, "near the largest": 2.0**63, "near the least normal": 2.0**-63}
"""The draws' scales: unit values, and values whose products (a squared distance, a second derivative's
pairs) reach float32's largest or its least normal."""


def normal_draw(seed: int, *shapes, scale: float = 1.0) -> tuple:
    """Standard normal float32 arrays of `shapes` at `scale`, from `seed`."""
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal(shape) * scale, jnp.float32) for shape in shapes)


def joined(seam):
    """`seam`'s two outputs, the stream and its norm, side by side along the features: one output a contract
    pulls back."""
    return lambda *inputs: jnp.concatenate(seam(*inputs), axis=-1)


INTERPRETED = jax.default_backend() != "gpu"
"""The ported kernels interpret off a GPU."""


def flattened(fn):
    """`fn`'s outputs as one vector, which the orders through its gradient sum against one cotangent."""
    return lambda *operands: jnp.concatenate([out.ravel() for out in jax.tree.leaves(fn(*operands))])


def positive_draw(seed: int, shape: tuple, scale: float) -> jax.Array:
    """A float32 variance a row, positive, at `scale`: a draw's magnitude, at least a tenth of the scale."""
    rng = np.random.default_rng(seed)
    return jnp.asarray((np.abs(rng.standard_normal(shape)) + 0.1) * scale, jnp.float32)
