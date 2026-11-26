"""The GEMM template's derivation at its edges: each degenerate input (no, one or odd rows, rows, a depth or a
width no tile or step divides, a shallow depth, a narrow width, one column, no depth or width) a cell in each
form of call through it (a plain product, matrices side by side, a pair, a prologue and an epilogue on their
sides), dense and grouped, on each GPU family's derivation as `generation.assumed` routes it (interpreted),
its outputs and its forward's, input's and matrices' gradients bitwise XLA's (`grouped_matmul='xla'`) over
small integers, whose sums every route makes exactly."""

import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import runnable

from dew.nn.kernels import gemm, generation
from dew.nn.kernels.templates.hooks import Hook, Side

EDGES = {
    "no rows": (0, 64, 64),
    "one row": (1, 64, 64),
    "odd rows": (7, 64, 64),
    "rows no tile divides": (72, 64, 64),
    "depth no step divides": (64, 72, 64),
    "shallow depth": (64, 16, 64),
    "no depth": (64, 0, 64),
    "width no tile divides": (64, 64, 72),
    "narrow width": (64, 64, 8),
    "one column": (64, 64, 1),
    "no width": (64, 64, 0),
}
"""Each degenerate input of the derivation: its rows, depth and width."""

RUNS = tuple(name for name in ("sm80", "sm90", "sm100") if runnable(name))
"""The GPUs whose derivations a cell runs, each this process can (`generation_support.runnable`)."""

FORMS = ("plain", "beside", "paired", "hooked")
"""The forms of call a cell takes through its input: a product, matrices side by side (a second of half the
width, at least one column), a pair into two outputs, and a prologue on the depth with an epilogue on the
rows and the columns."""


def _scaled(x, scale):
    return x * scale[None, :]


def _biased(acc, bias, row_scale):
    return (acc + bias[None, :]) * row_scale[:, None]


def _groups(rows: int, dense: bool):
    """A grouped call's sizes over `rows`: three groups, the middle one empty and the first a third (every
    group empty where there are no rows); a dense call's none."""
    if dense:
        return None
    first = rows // 3
    return jnp.asarray([first, 0, rows - first], jnp.int32)


def _operands(rows: int, depth: int, width: int, groups: int, form: str, seed: int) -> dict:
    """Small integers in bf16 (rows, matrices, their sides and the outputs' cotangents), exact in every
    product and sum a route makes."""
    rng = np.random.default_rng(seed)

    def ints(*shape, low=-2, high=3):
        return jnp.asarray(rng.integers(low, high, size=shape), jnp.float32)

    widths = (width, max(width // 2, 1)) if form in ("beside", "paired") else (width,)
    if form == "paired":
        widths = (width, width)
    matrices = tuple(ints(groups, depth, each).astype(jnp.bfloat16) for each in widths)
    return {"rows": ints(rows, depth).astype(jnp.bfloat16), "matrices": matrices,
            "scale": ints(depth, low=1, high=3), "bias": ints(width), "row_scale": ints(rows, low=1, high=3),
            "cotangents": tuple(ints(rows, each) for each in widths)}


def _call(form: str, sizes, dense: bool):
    """The cell's call of `form`: its outputs from the rows, the matrices and the sides."""
    def call(rows, matrices, scale, bias, row_scale):
        held = tuple(matrix[0] for matrix in matrices) if dense else matrices
        options = {"out_dtype": jnp.float32, "interpret_on_cpu": True}
        if form == "plain":
            return (gemm.gemm(rows, held[0], sizes, **options),)
        if form == "beside":
            return tuple(gemm.gemm(rows, held[0], sizes, beside=held[1:], **options))
        if form == "paired":
            return tuple(gemm.gemm(rows, held[0], sizes, paired=held[1], out_dtypes=(jnp.float32,) * 2,
                                   **options))
        return (gemm.gemm(rows, held[0], sizes, prologue=Hook(_scaled, (Side(scale, ("depth",)),)),
                          epilogue=Hook(_biased, (Side(bias, ("columns",)), Side(row_scale, ("rows",)))),
                          **options),)

    return call


def _run(form: str, operands: dict, sizes, dense: bool, kernel: str | None, gpu: str):
    """The call's outputs and every operand's gradient under the outputs' cotangents, on `gpu` as `kernel`
    routes it (None: as derived)."""
    call = _call(form, sizes, dense)
    differentiated = (operands["rows"], operands["matrices"], operands["scale"], operands["bias"],
                      operands["row_scale"])
    chosen = {} if kernel is None else {"grouped_matmul": kernel}
    with generation.assumed(gpu), generation.chosen(**chosen):
        outs, pull = jax.vjp(call, *differentiated)
        return outs, pull(operands["cotangents"][:len(outs)])


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("edge", list(EDGES))
def test_a_degenerate_call_is_xlas(edge, form):
    """`edge`'s rows, depth and width in `form` (`FORMS`), dense and grouped (`_groups`), on an A100, an H100
    and a B200: the outputs and every gradient bitwise XLA's program's, the call taken by whichever route
    each GPU's derivation gives (a tile, or none and the next route)."""
    rows, depth, width = EDGES[edge]
    for dense in (True, False):
        sizes = _groups(rows, dense)
        label = f"{edge} {form} {'dense' if dense else 'grouped'}"
        operands = _operands(rows, depth, width, 1 if dense else 3, form, seed=zlib.crc32(label.encode()))
        reference = _run(form, operands, sizes, dense, "xla", "sm90")
        for gpu in RUNS:
            derived = _run(form, operands, sizes, dense, None, gpu)
            for got, expected in zip(jax.tree.leaves(derived), jax.tree.leaves(reference), strict=True):
                assert (got.shape, got.dtype) == (expected.shape, expected.dtype), f"{label} on {gpu}"
                np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(expected, np.float32),
                                              err_msg=f"{label} on {gpu}")
