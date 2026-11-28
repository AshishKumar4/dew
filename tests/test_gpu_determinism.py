"""Every ordered schedule's reproducibility on the GPU (G14, the owner's rule: no unit tests on GPUs;
kernel-templates.md §8: every deterministic schedule is bitwise reproducible on each target), run in one
process (`gpu_flow`): each cell a reduction of three or more partials, which a GPU's arrival order could sum
in any order and the template sums in a fixed one, the same program on the same inputs run again, retraced,
giving the same bits (`gpu_flow.repeats_bit_for_bit`) under the lane's deterministic ops
(tests/lane_environment.py):

- a dense product's contraction split in three (`Tiles.splits`), its partials chosen so their sums in two
  orders differ (`cancelling`), and a kernel gradient's, dense and grouped with an empty group among the
  groups, its fp32 partials summed in order;
- a decode step's merge of eight slices' states by their masses (`merged.merged_padded`);
- a fused RMS norm's weight gradient, each program's partial row summed after it.

Each cell's programs run on its GPU, its bits its property: no check-1 site, so its part's prepare passes it
by (`Cell.at_run`). The attention backward's ordered launches and the selective scan's owned backward repeat
in their own flows (tests/test_gpu_attention.py's shared-bias steps; tests/test_selective_template.py)
(row 6132).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpu_flow import Cell, repeats_bit_for_bit, run_cells
from test_merged import SPECS, WIDTH, _drawn
from test_norm_swiglu import fused_seam

from dew.nn.kernels import gemm, generation, grouped_matmul, tuning
from dew.nn.kernels.templates import merged

pytestmark = pytest.mark.gpu(reason="each ordered schedule's kernels: their partials' order on the GPU")

RUNS = 4
"""The runs each cell compares, the second retraced (`repeats_bit_for_bit`)."""

BITWISE = "a bitwise check of its programs' runs on its GPU, no check-1 site"
"""Why a cell compiles at its run (`Cell.at_run`): its programs' bits on its GPU are its property."""

SPLITS, CHUNK = 3, 256
"""The partials a split product's contraction is cut into, `CHUNK` deep each."""


def _normal(seed: int, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """A draw of standard normals of `shape` in `dtype`, NumPy's (a prepare computes on no device)."""
    return np.random.default_rng(seed).standard_normal(shape, np.float32).astype(dtype)


def cancelling(rows: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """A product's operands whose contraction's three `CHUNK`-deep partials are, at every output, 2^25, -2^25
    and 1: summed first two first, the 1 survives; the last two first, -2^25 + 1 rounds to -2^25 in fp32
    (its spacing there 4) and the 1 is lost, so a merge in another order gives other bits. Every partial is
    exact in bfloat16 and in an fp32 accumulator (256 products of 2^17 each)."""
    x = np.ones((rows, SPLITS * CHUNK), jnp.bfloat16)
    w = np.zeros((SPLITS * CHUNK, width), np.float32)
    w[:CHUNK], w[CHUNK:2 * CHUNK], w[2 * CHUNK] = 2.0 ** 17, -(2.0 ** 17), 1.0
    partials = np.float32(2.0 ** 25), np.float32(-(2.0 ** 25)), np.float32(1.0)
    assert (partials[0] + partials[1]) + partials[2] != partials[0] + (partials[1] + partials[2])
    return x, w.astype(jnp.bfloat16)


def a_split_products_partials_repeat_bit_for_bit(sizes: tuple[int, ...], *, gradient: bool) -> None:
    """A product whose contraction is split in three at the tile its call derives on this GPU (`Tiles.splits`:
    each run of steps its fp32 partial, the partials summed in order), a dense forward's (`MosaicGmm`) on
    `cancelling` operands or a kernel gradient's (`MosaicTgmm`) over groups of `sizes` rows: the same bits
    each run, the split the launch took."""
    rows, width = sum(sizes), 256
    depth = SPLITS * CHUNK
    spec = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    count = jax.ShapeDtypeStruct((len(sizes),), jnp.int32)
    dense = len(sizes) == 1
    target = generation.gpu_target()
    if gradient:
        launch = grouped_matmul.MosaicTgmm()
        call = launch.bind(spec((rows, depth)), spec((rows, width)), count, interpret=False,
                           out_dtype="float32", dense=dense, exact_sums=True)
        operands = _normal(7, (rows, depth), jnp.bfloat16), _normal(8, (rows, width), jnp.bfloat16)

        def step(x, dy):
            return gemm.gemm_t(x, dy, None if dense else jnp.asarray(sizes, jnp.int32), out_dtype=jnp.float32,
                               precision="highest")
    else:
        launch = grouped_matmul.MosaicGmm()
        call = launch.bind(spec((rows, depth)), spec((1, depth, width)), count, out_dtype="float32",
                           trans_rhs=False, interpret=False, dense=True)
        operands = cancelling(rows, width)

        def step(x, w):
            return gemm.gemm(x, w, None, out_dtype=jnp.float32)
    tile = dataclasses.replace(launch.derived(call, target), splits=SPLITS)
    if not launch.legal(call, tile, target):
        pytest.skip(f"this GPU's derived tile takes no {SPLITS}-way split: {tile}")
    with generation.chosen(grouped_matmul="mosaic"), tuning.pinned({type(launch): tile}), \
            tuning.seen() as launches:
        repeats_bit_for_bit(jax.jit(step), *operands, runs=RUNS)
    assert tile in {observed.config for observed in launches if isinstance(observed.op, type(launch))}, tile


def a_decode_merge_of_eight_slices_repeats_bit_for_bit() -> None:
    """A decode step's merge of eight slices' states (`merged.merged_padded`, the fold a split step selects),
    one of them of no mass: its sums over the slices the same bits each run."""
    padded, stacked = _drawn(0)[:2]
    merge = jax.jit(lambda padded, stacked: merged.merged_padded(SPECS["fold"], padded, stacked, queries=4,
                                                                 width=WIDTH))
    repeats_bit_for_bit(merge, padded, stacked, runs=RUNS)


def a_norms_weight_gradient_repeats_bit_for_bit() -> None:
    """A fused RMS norm's gradient over 4099 rows of 256 (`fused_seam`): its weight's, each program's partial
    row summed after it, and its input's, the same bits each run."""
    seam = fused_seam(1e-5, jnp.bfloat16, offset=False, after_cast=False)
    x, cotangent = _normal(14, (4099, 256), jnp.bfloat16), _normal(15, (4099, 256), jnp.bfloat16)
    scale = 1 + 0.1 * _normal(16, (256,))

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
        Cell("a dense product's three cancelling partials repeat bit for bit",
             functools.partial(a_split_products_partials_repeat_bit_for_bit, (768,), gradient=False),
             at_run=BITWISE),
        Cell("a dense kernel gradient's split contraction repeats bit for bit",
             functools.partial(a_split_products_partials_repeat_bit_for_bit, (768,), gradient=True),
             at_run=BITWISE),
        Cell("a grouped kernel gradient's split contraction repeats bit for bit",
             functools.partial(a_split_products_partials_repeat_bit_for_bit, (200, 0, 568), gradient=True),
             at_run=BITWISE),
        Cell("a decode merge of eight slices repeats bit for bit",
             a_decode_merge_of_eight_slices_repeats_bit_for_bit, at_run=BITWISE),
        Cell("a norm's weight gradient repeats bit for bit", a_norms_weight_gradient_repeats_bit_for_bit,
             at_run=BITWISE),
    ])
