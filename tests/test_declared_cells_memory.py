"""A declared cell's route holds no more of a training step's memory than its receipt states (row 5972's P0:
the B200's and GH200's kernel-gradient cells sent the 4B's dW to XLA's dot, which XLA deferred to the
backward's end, every layer's cotangents held, a four-row step's temporaries 22.25 GB against 11.06 GB on
the template). Each generation's dense products in a stack of layers, forward and reverse, on a compile-only
device: their temporaries under the declared cells against the template's named in their place, past what
the cells' receipts state they hold (`grouped_matmul.HELD`)."""
import jax
import jax.numpy as jnp
import pytest
from kernel_compile import lowered_for

from dew.nn.kernels import gemm, generation, grouped_matmul

LAYERS, ROWS, WIDTH = 4, 4096, 2048
FAMILIES = {"sm80": "mma.sync", "sm90": "wgmma", "sm100": "tcgen05"}


def _step_temp(generation_name: str, chosen: dict) -> int:
    """The temporaries a `LAYERS`-deep stack of dense products takes, its forward and its reverse, on a
    compile-only device of `generation_name`, its products routed as `chosen` names (`generation.chosen`)."""
    def loss(x, weights):
        for w in weights:
            x = jnp.tanh(gemm.gemm(x, w, None, out_dtype=jnp.bfloat16))
        return jnp.sum(x.astype(jnp.float32))

    shapes = (jax.ShapeDtypeStruct((ROWS, WIDTH), jnp.bfloat16),
              tuple(jax.ShapeDtypeStruct((WIDTH, WIDTH), jnp.bfloat16) for _ in range(LAYERS)))
    with generation.chosen(**chosen):
        return lowered_for(jax.grad(loss, argnums=(0, 1)), shapes, generation_name).compile(
            ).memory_analysis().temp_size_in_bytes


@pytest.mark.parametrize("generation_name", list(FAMILIES))
def test_a_generations_declared_cells_hold_no_more_of_a_steps_memory_than_their_receipts_state(
        generation_name):
    """A stack of `LAYERS` dense bf16 products, `ROWS` x `WIDTH` by `WIDTH` x `WIDTH`, forward and reverse:
    its temporaries under its generation's declared cells (`grouped_matmul.DECLARED`) no larger than the
    template's named in their place (`generation.chosen(grouped_matmul='mosaic')`) by more than its cells'
    stated holds (`HELD`: outputs a layer each), past one output of slack."""
    family = FAMILIES[generation_name]
    held = sum(outputs for (cell_family, kind), (outputs, _) in grouped_matmul.HELD.items()
               if cell_family == family and kind.dense and kind.matrix is None)
    output = ROWS * WIDTH * 2
    cells = _step_temp(generation_name, {})
    template = _step_temp(generation_name, {"grouped_matmul": "mosaic"})
    assert cells <= template + (held * LAYERS + 1) * output, (cells, template, held)
