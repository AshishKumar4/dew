"""The NVFP4 product's check-1 matrix (tests/kernel_matrix.py): a product of an NVFP4 matrix (`formats.NvFp4`,
a quantized checkpoint's frozen weights) with its matrix decoded stage by stage in the template's mainloop
(`gemm.nvfp4_product`'s packed kernel, row 3621) on par with its matrix decoded once a call and multiplied.

Each cell runs the product's forward and its reverse mode in one program: the output and the rows' gradient
(the weights frozen; `reference_error.with_gradients`, a drawn cotangent), held to the product with the
selector's other kernel named, `xla`, the decoded matrix's product (`generation.chosen`, check 1), from the
plain product in float64 (the rows times the decoded values times the deferred factor).

The kernel under test is the packed kernel, named where the cells run it (no receipt yet qualifies it, so its
selector, `gemm.NVFP4`, takes it by name alone): the cells' `CHOICES` are `nvfp4`. The rows' gradient and the
`xla` arm are GEMMs whose route, `grouped_matmul`, runs at XLA, named in the cell (`XLA`), so the template's
launches leave this check to their own family's (tests/gemm_matrix.py).

What the route serves is read from the product's own selector: the product traced for its shapes as the
case's generation with the packed kernel named, the route its choice reports the packed kernel.
"""

import generation_support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import drawn_on_the_host
from gpu_flow import cell
from jax.sharding import Mesh
from kernel_matrix import AXES
from low_precision_support import nvfp4
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import formats, gemm, generation

ROWS = 192
"""The product's rows."""
SEEDS = 8
CHOICES = frozenset({"nvfp4"})
"""The `generation.CHOICES` entries the cells run: each cell asserts that the routes it records run `nvfp4`
`packed` and every choice outside `CHOICES` at XLA."""
KERNELS = {"nvfp4": frozenset({"packed"})}
"""The kernels under test of each choice the cells run: the template's in-stage decode."""
XLA = {"grouped_matmul": "xla"}
"""The product's other choice, named XLA in every cell."""
def _matrix(case: dict[str, str]) -> tuple[formats.NvFp4, np.ndarray]:
    """`case`'s NVFP4 matrix (`low_precision_support.nvfp4`, tensor scale 0.75) and its values in float64."""
    depth, width = int(case["depth"]), int(case["width"])
    # Formed on the host's CPU and held as NumPy's: a GPU tier's prepare computes on no other device (its
    # default a compile-only GPU's, rows 8020-8103).
    return drawn_on_the_host(lambda seed: nvfp4(seed, depth, width, scale=0.75), depth + width, held=True)


def supported(case: dict[str, str]) -> bool:
    """Whether the route serves `case` (tests/kernel_matrix.py): its product traced for its shapes as its
    generation within an abstract mesh of one of its devices, the packed kernel named and the product at
    XLA, reports the packed kernel for `nvfp4`."""
    matrix, _ = _matrix(case)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), case["generation"])
    x = jax.ShapeDtypeStruct((ROWS, int(case["depth"])), jnp.bfloat16)
    with (generation.assumed(case["generation"]), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(**XLA, nvfp4="packed"), generation.routes() as routes):
        jax.eval_shape(lambda rows: gemm.nvfp4_product(rows, matrix, out_dtype=case["out_dtype"]), x)
    return [route.kernel for route in routes if route.choice == "nvfp4"] == ["packed"]


CASES = (
    ('sm80', '128', '256', 'bfloat16'),
    ('sm90', '128', '256', 'float32'),
    ('sm90', '256', '512', 'bfloat16'),
    ('sm80', '256', '512', 'float32'),
    ('sm90', '128', '512', 'float32'),
    ('sm90', '256', '256', 'float32'),
)
"""The cells' cases, each its axis values in `AXES`' order: every pair of axis values the route serves
(`kernel_matrix.covering("nvfp4")`, regenerated where the axes or the route's support change)."""


def _step(matrix: formats.NvFp4, out_dtype: str, kernel: str):
    """The product's training step as `run(rows, cotangent)`, `nvfp4` named `kernel`."""
    def product(rows):
        with generation.chosen(**XLA, nvfp4=kernel):
            return gemm.nvfp4_product(rows, matrix, out_dtype=out_dtype)
    return with_gradients(product)


def _truth(values: np.ndarray):
    """The plain product's training step in float64: the rows times the matrix's values, formed within the
    truth's x64 scope (`assert_as_exact`), where float64 exists."""
    return with_gradients(lambda rows: rows @ jnp.asarray(values, jnp.float64))


@cell("nvfp4", covers=lambda case: dict(zip(AXES["nvfp4"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=generation_support.pinned(case[0]))
    for case in CASES], ids="-".join)
def an_nvfp4_products_training_step_is_on_par(case: tuple[str, ...]) -> None:
    """The NVFP4 product's training step of `case` on its GPU with its matrix decoded in the mainloop on par
    with it decoded once a call (check 1): the output and the rows' gradient, from the plain product in
    float64. The routes it records run `nvfp4` `packed` and every other choice at XLA."""
    combination = dict(zip(AXES["nvfp4"], case, strict=True))
    matrix, values = _matrix(combination)
    width, depth, out_dtype = int(combination["width"]), int(combination["depth"]), combination["out_dtype"]

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed))
        rows = jax.random.normal(keys[0], (ROWS, depth)).astype(jnp.bfloat16)
        return rows, jax.random.normal(keys[1], (ROWS, width)).astype(out_dtype)

    with generation.routes() as routes:
        assert_as_exact(jax.jit(_step(matrix, out_dtype, "packed")), jax.jit(_step(matrix, out_dtype, "xla")),
                        draw, "-".join(case), seeds=SEEDS, truth_fn=_truth(values))
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded if pair[0] not in CHOICES and pair[1] != "xla")
    assert ("nvfp4", "packed") in recorded and not unnamed, (sorted(recorded), unnamed)


SHARD_ROWS = (4, 8)
"""A per-shard cell's tokens, `[batch, length]`: its batch split over a two-device data axis."""
SHARD_EXPERTS, SHARD_HIDDEN, SHARD_INTERMEDIATE = 4, 256, 256
"""A per-shard cell's GPT-OSS experts: each bank's depth whole 256s and its width whole 64s, which the packed
kernel takes (`model_flows.GPT_OSS_PACKED`'s)."""


def per_shard(choice: str, kernel: str) -> dict:
    """`kernel`'s per-shard cells (`sharded.Shard`, tests/test_per_shard.py): GPT-OSS's experts with their
    routed matrices packed NVFP4 banks (`model_flows.packed_variables`, a packed load's), every token routed
    to every expert, as its packed model flow runs them, over tokens a two-device data axis splits, the banks
    and the biases whole, the products `kernel` (interpreted on the CPU's devices) and every other GEMM at XLA
    (`XLA`): the output and the gradients of the tokens, the routing weights and the biases on par with one
    device's, from the experts in float64 over each bank's exact matrix (`model_flows.exact_banks`)."""
    from jax.sharding import PartitionSpec
    from model_flows import exact_banks, packed_variables
    from sharded import Shard, layer_scope

    from dew.nn.gpt_oss import GptOssExperts
    from dew.quantization import QUANTIZATION

    del choice, kernel
    layer = GptOssExperts(SHARD_HIDDEN, SHARD_INTERMEDIATE, SHARD_EXPERTS, dtype=jnp.bfloat16)
    indices = jnp.broadcast_to(jnp.arange(SHARD_EXPERTS, dtype=jnp.int32), (*SHARD_ROWS, SHARD_EXPERTS))
    # Formed on the host's CPU and held as NumPy's (`_matrix`'s), the banks packed there.
    held = drawn_on_the_host(lambda seed: packed_variables(layer.init(
        jax.random.key(seed), jnp.zeros((*SHARD_ROWS, SHARD_HIDDEN), jnp.bfloat16),
        jnp.ones((*SHARD_ROWS, SHARD_EXPERTS), jnp.bfloat16), indices)), 5, held=True)
    banks = held[QUANTIZATION]

    def call(params, tokens, weights):
        return layer.apply({"params": params, QUANTIZATION: banks}, tokens, weights, indices)

    def exact(params, tokens, weights):
        # Each bank's exact matrix in float64, its tensor scale widened first (`exact_banks`).
        wide = jax.tree.map(
            lambda bank: formats.NvFp4(bank.words, bank.scales, jnp.asarray(bank.scale, jnp.float64)), banks,
            is_leaf=lambda leaf: isinstance(leaf, formats.NvFp4))
        matrices = exact_banks({QUANTIZATION: wide})[QUANTIZATION]
        return layer.clone(dtype=jnp.float64).apply({"params": params, QUANTIZATION: matrices}, tokens,
                                                    weights, indices)

    def truth(*operands):
        with generation.chosen(**XLA):
            return with_gradients(exact)(*operands)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 3)
        tokens, weights, cotangent = (jax.random.normal(key, (*SHARD_ROWS, width)).astype(jnp.bfloat16)
                                      for key, width in zip(keys, (SHARD_HIDDEN, SHARD_EXPERTS, SHARD_HIDDEN),
                                                            strict=True))
        return held["params"], tokens, weights, cotangent

    rows = PartitionSpec("data")
    return {"data": Shard(with_gradients(call), draw, lambda mesh: [PartitionSpec(), rows, rows, rows], truth,
                          SEEDS, "data", context=lambda: layer_scope(**XLA))}
