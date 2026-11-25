"""Every ordered schedule's reproducibility on the GPU (G14, the owner's rule: no unit tests on GPUs;
kernel-templates.md §8: every deterministic schedule is bitwise reproducible on each target), run in one
process (`gpu_flow`): each cell a reduction whose partials a GPU's arrival order could reorder and the
template sums in a fixed order, the same program on the same inputs run again, compiled afresh, giving the
same bits (`gpu_flow.repeats_bit_for_bit`) under the lane's deterministic ops (tests/lane_environment.py):

- a dense product's and a kernel gradient's contraction split in two (`Tiles.splits`), dense and grouped, an
  empty group among the groups, its fp32 partials summed in order;
- a decode step's keys cut into slices (`attention_decode.planned`), their states merged by their masses;
- a fused RMS norm's weight gradient, each program's partial row summed after it.

The attention backward's ordered launches and the selective scan's owned backward repeat in their own flows
(tests/test_gpu_attention.py's shared-bias steps; tests/test_selective_template.py) (row 6132).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest
from gpu_flow import Cell, repeats_bit_for_bit, run_cells
from test_norm_swiglu import fused_seam

from dew.nn.kernels import gemm, generation, grouped_matmul, tuning
from dew.nn.kernels.templates import attention as template, attention_decode

pytestmark = pytest.mark.gpu(reason="each ordered schedule's kernels: their partials' order on the GPU")

RUNS = 4
"""The runs each cell compares, the second compiled afresh (`repeats_bit_for_bit`)."""


def _normal(seed: int, shape: tuple[int, ...], dtype=jnp.bfloat16) -> jax.Array:
    """A draw of standard normals of `shape` in `dtype`."""
    return jax.random.normal(jax.random.key(seed), shape, jnp.float32).astype(dtype)


def a_split_products_partials_repeat_bit_for_bit(sizes: tuple[int, ...], *, gradient: bool) -> None:
    """A product whose contraction is split in two at the tile its call derives on this GPU (`Tiles.splits`:
    each run of steps its fp32 partial, the partials summed in order), a dense forward's (`MosaicGmm`) or a
    kernel gradient's (`MosaicTgmm`) over groups of `sizes` rows: the same bits each run, the split the
    launch took."""
    rows, depth, width = sum(sizes), 512, 256
    spec = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    count = jax.ShapeDtypeStruct((len(sizes),), jnp.int32)
    dense = len(sizes) == 1
    target = generation.gpu_target()
    if gradient:
        launch = grouped_matmul.MosaicTgmm()
        call = launch.bind(spec((rows, depth)), spec((rows, width)), count, interpret=False,
                           out_dtype="float32", dense=dense, exact_sums=True)
        operands = _normal(7, (rows, depth)), _normal(8, (rows, width))

        def step(x, dy):
            return gemm.gemm_t(x, dy, None if dense else jnp.asarray(sizes, jnp.int32), out_dtype=jnp.float32,
                               precision="highest")
    else:
        launch = grouped_matmul.MosaicGmm()
        call = launch.bind(spec((rows, depth)), spec((1, depth, width)), count, out_dtype="bfloat16",
                           trans_rhs=False, interpret=False, dense=True)
        operands = _normal(9, (rows, depth)), _normal(10, (depth, width))

        def step(x, w):
            return gemm.gemm(x, w, None, out_dtype=jnp.bfloat16)
    tile = dataclasses.replace(launch.derived(call, target), splits=2)
    if not launch.legal(call, tile, target):
        pytest.skip(f"this GPU's derived tile takes no two-way split: {tile}")
    with generation.chosen(grouped_matmul="mosaic"), tuning.pinned({type(launch): tile}), \
            tuning.seen() as launches:
        repeats_bit_for_bit(jax.jit(step), *operands, runs=RUNS)
    assert tile in {config for (op, _), (config, _) in zip(launches, launches.taken, strict=True)
                    if isinstance(op, type(launch))}, tile


def a_decode_steps_split_keys_merge_repeating_bit_for_bit() -> None:
    """A decode step of two rows over 4096 keys whose plan on this GPU cuts its keys into slices
    (`attention_decode.planned`), their states merged by their masses: the same bits each run."""
    keys = 4096
    spans = template.spans(1, keys, 2)
    query = _normal(11, (2, 1, 8, 64))
    key, value = _normal(12, (2, keys, 2, 64)), _normal(13, (2, keys, 2, 64))
    work, *_ = attention_decode.planned(query, key, value, spans, template.Hooks())
    if work is None:
        pytest.skip("this GPU's decode step takes no template plan")
    assert work["plain"].slices > 1, work
    repeats_bit_for_bit(jax.jit(lambda q, k, v: attention_decode.attend(q, k, v, spans)), query, key, value,
                        runs=RUNS)


def a_norms_weight_gradient_repeats_bit_for_bit() -> None:
    """A fused RMS norm's gradient over 4099 rows of 256 (`fused_seam`): its weight's, each program's partial
    row summed after it, and its input's, the same bits each run."""
    seam = fused_seam(1e-5, jnp.bfloat16, offset=False, after_cast=False)
    x, cotangent = _normal(14, (4099, 256)), _normal(15, (4099, 256))
    scale = 1 + 0.1 * _normal(16, (256,), jnp.float32)

    @jax.jit
    def step(x, scale, cotangent):
        _, pull = jax.vjp(lambda x, scale: seam(x, None, scale)[1], x, scale)
        return pull(cotangent)

    repeats_bit_for_bit(step, x, scale, cotangent, runs=RUNS,
                        names=["the input's gradient", "the weight's gradient"])


def test_every_ordered_schedule_repeats_bit_for_bit_on_this_gpu():
    """Every cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and all raised
    together."""
    run_cells([
        Cell("a dense product's split contraction repeats bit for bit",
             functools.partial(a_split_products_partials_repeat_bit_for_bit, (768,), gradient=False)),
        Cell("a dense kernel gradient's split contraction repeats bit for bit",
             functools.partial(a_split_products_partials_repeat_bit_for_bit, (768,), gradient=True)),
        Cell("a grouped kernel gradient's split contraction repeats bit for bit",
             functools.partial(a_split_products_partials_repeat_bit_for_bit, (200, 0, 568), gradient=True)),
        Cell("a decode step's split keys merge repeating bit for bit",
             a_decode_steps_split_keys_merge_repeating_bit_for_bit),
        Cell("a norm's weight gradient repeats bit for bit", a_norms_weight_gradient_repeats_bit_for_bit),
    ])
