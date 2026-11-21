"""Sparse latent attention's check-1 matrix (tests/kernel_matrix.py): `sparse_selection.SPARSE`'s template
kernel on par with the blocked XLA reference (`sparse_selection.blocked_context`, row 9023)."""

import functools

import generation_support
import jax
import jax.numpy as jnp
import pytest
from gpu_flow import cell
from kernel_matrix import AXES
from reference_error import assert_as_exact

from dew.nn import sparse_selection
from dew.nn.kernels import generation, selection

CHOICES = frozenset({"sparse_attention", "attention_mainloop"})
"""The `generation.CHOICES` entries the cells run (`tests/test_kernel_matrix.py`'s union, row 184)."""
KERNELS = {"sparse_attention": frozenset({"template"})}
"""The non-generic kernel of `sparse_attention` run under test (row 4513)."""

BATCH, LENGTH, HEADS, RANK, ROPE, TOP_K = 2, 4, 4, 32, 32, 4
SCALE = (RANK + ROPE) ** -0.5
SEEDS = 8


def _dtype(name: str):
    return jnp.bfloat16 if name == "bfloat16" else jnp.float16


def _case(combination: dict[str, str]) -> sparse_selection.SparseCase:
    dtype = _dtype(combination["dtype"])
    return sparse_selection.SparseCase(
        heads=HEADS, rank=RANK, rope=ROPE, top_k=TOP_K,
        dtypes=(dtype, dtype, dtype, dtype), rows=BATCH * LENGTH,
        pool_keys=BATCH * LENGTH, scale=SCALE,
    )


def supported(combination: dict[str, str]) -> bool:
    """Whether `sparse_selection.SPARSE` selects the template kernel for `combination` on its GPU."""
    with generation.assumed(combination["generation"]):
        return selection.select(sparse_selection.SPARSE, _case(combination)).kernel == "template"


CASES = (
    ("sm90", "bfloat16"),
    ("sm90", "float16"),
    ("sm80", "bfloat16"),
    ("sm80", "float16"),
)
"""The cells' cases in `AXES['sparse_attention']`'s order: every `(generation, dtype)` pair served."""


def _draw(dtype, seed: int):
    keys = jax.random.split(jax.random.key(seed), 5)
    absorbed = jax.random.normal(keys[0], (BATCH, LENGTH, HEADS, RANK), dtype)
    query_rot = jax.random.normal(keys[1], (BATCH, LENGTH, HEADS, ROPE), dtype)
    latent = jax.random.normal(keys[2], (BATCH, LENGTH, RANK), dtype)
    rot = jax.random.normal(keys[3], (BATCH, LENGTH, ROPE), dtype)
    indices = jnp.broadcast_to(jnp.arange(TOP_K, dtype=jnp.int32), (BATCH, LENGTH, TOP_K))
    return absorbed, query_rot, latent, rot, indices


@cell("sparse_attention", covers=lambda case: dict(zip(AXES["sparse_attention"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=generation_support.pinned(case[0]))
                                  for case in CASES], ids="-".join)
def a_sparse_attention_step_is_on_par(case: tuple[str, ...]) -> None:
    """`sparse_selection.templated_context` on `case`'s GPU on par with `blocked_context` (check 1)."""
    combination = dict(zip(AXES["sparse_attention"], case, strict=True))
    dtype = _dtype(combination["dtype"])
    ours_fn = functools.partial(sparse_selection.templated_context, scale=SCALE)
    ref_fn = functools.partial(sparse_selection.blocked_context, scale=SCALE, block=LENGTH)

    def run(*operands):
        with generation.chosen(sparse_attention="template"):
            return ours_fn(*operands)

    with generation.assumed(combination["generation"]):
        assert_as_exact(jax.jit(run), jax.jit(ref_fn), functools.partial(_draw, dtype),
                        f"sparse_attention {'-'.join(case)}", seeds=SEEDS)
