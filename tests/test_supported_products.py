"""A product issued of its support's terms alone (`product.Support`, `Issuer.products`' `support`): the
doubling's merges (`recurrence_algebra.doubled`), whose supports hold every term the whole products sum,
issued on mma.sync of those terms alone, its instructions fewer and its outputs the whole products'."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, lowers
from kernel_compile import lowered_for
from test_recurrence_mosaic import resource_operands

from dew.nn.kernels import generation, recurrence_algebra as algebra
from dew.nn.kernels.templates import bundles, product
from dew.nn.kernels.templates.recurrence import mainloop, template

SIZE = 64


def _summed(left, right, held=None):
    """`left @ right` in float32, each output's terms summed in one order, those `held` (`[M, N, K]`) does not
    hold dropped where given."""
    terms = np.asarray(left, np.float32)[:, None, :] * np.asarray(right, np.float32).T[None, :, :]
    return (terms if held is None else np.where(held, terms, np.float32(0))).sum(axis=2, dtype=np.float32)


def _whole(name, left, right):
    del name
    return _summed(left, right)


def _of_support(name, left, right, support: product.Support):
    """The product of `support`'s terms alone, every output it does not keep NaN."""
    del name
    out = _summed(left, right, support.left[:, None, :] & support.right.T[None, :, :])
    return np.where(support.kept, out, np.float32(np.nan))


def _system(seed: int, scale: float) -> np.ndarray:
    normal = np.random.default_rng(seed).standard_normal
    rows, cols = np.indices((SIZE, SIZE))
    return np.where(rows > cols, scale * normal((SIZE, SIZE)), 0.0).astype(np.float32)


@pytest.mark.parametrize("scale", [0.05, 1.0, 4.0])
def test_the_doublings_merges_supports_hold_every_term_they_sum(scale):
    """`doubled` issued of its merges' supports alone (`_merges_support`), each product summing only the
    terms its support states and leaving every output it does not keep NaN, is `doubled` of the whole
    products bit for bit, each output's terms summed in one order, over systems whose inverses stay small
    and grow to thousands, finite (the contract's: a skipped zero times an infinity is the whole's NaN):
    every term the whole merges sum lies in its support and every output read is one it keeps."""
    for seed in range(4):
        a = _system(seed, scale)
        whole = algebra.doubled(a, _whole)
        assert np.isfinite(whole).all()
        np.testing.assert_array_equal(algebra.doubled(a, _whole, supported=_of_support), whole)


def test_a_support_missing_a_term_changes_the_doubling():
    """The witness above tells a term apart: a support whose inverse drops each block's last row's terms
    gives another inverse than the whole products'."""
    def short(name, left, right, support):
        clipped = support.left.copy()
        clipped[SIZE - 1] = False
        return _of_support(name, left, right, product.Support(clipped, support.right, support.kept))

    a = _system(0, 1.0)
    assert not np.array_equal(algebra.doubled(a, _whole, supported=short), algebra.doubled(a, _whole))


def _solve_call(target: str):
    """Qwen3.5-27B's gated delta net forward on `target` at its derived tiles: the tiles, and a function of
    its operands and their shapes."""
    facts = generation.gpu_target(target)
    q, k, v, g, beta, _ = resource_operands("qwen3.5-27b-gdn", jnp.bfloat16, 1, 256)
    bound = template.MOSAIC.bind(q, k, v, g, beta, None, None, bound=1.0)
    with generation.assumed(facts):
        tiles = template.MOSAIC.derived(bound, facts)
    forward = tiles.forward or tiles

    def run(*operands):
        return template.mosaic_rule(*operands, tiles=forward, bound=1.0)[0]

    return forward, run, (q, k, v, g, beta)


def _tf32_products(lowered, name: str) -> int:
    """The TF32 mma.sync instructions the Mosaic GPU kernel `name` begins in `lowered` issues."""
    import jax.experimental.mosaic.gpu as mgpu
    from jax.extend.mlir import ir

    from dew.nn.kernels.lowered import kernel_calls

    (kernel,) = [each for each in kernel_calls(lowered)
                 if each.kind == "mosaic_gpu" and each.name.startswith(name)]
    module = mgpu.deserialized(kernel.ir)
    found = []

    def counted(operation):
        if operation.name == "llvm.inline_asm":
            text = ir.StringAttr(operation.attributes["asm_string"]).value
            found.append("mma.sync" in text and ".tf32" in text)
        return ir.WalkResult.ADVANCE

    with module.context:
        module.operation.walk(counted)
    return sum(found)


@COMPILES_ON_CPU
@pytest.mark.parametrize("target", [pytest.param(name, marks=lowers(name)) for name in ("sm80", "sm120")])
def test_the_doublings_merges_issue_their_supports_instructions_alone(target):
    """Qwen3.5-27B's gated delta net solve lowered for an A100 and an RTX PRO 6000 issues as many TF32
    instructions as its merges' supports hold tiles of the issuer's (`Support.tiles` on `term_tile` and
    TF32's depth, a part pair each), a small share of its whole products' (each level's two merges the
    whole `[C, C]` tile: 544 an RTX PRO 6000's thread at main 01656004b)."""
    tiles, _, _ = _solve_call(target)
    family = bundles.of(target, jnp.bfloat16)
    rows, columns = family.term_tile
    parts = mainloop.merge_parts(family, jnp.bfloat16, held=1, ranged=False)
    pairs = len(product.terms(parts, parts))
    supports = [support for width in (2, 4, 8, 16, 32)
                for support in algebra._merges_support(tiles.chunk, width)]
    expected = pairs * sum(int(support.tiles(rows, columns, 8).sum()) for support in supports)
    whole = pairs * len(supports) * (tiles.chunk // rows) * (tiles.chunk // columns) * (tiles.chunk // 8)
    _, run, operands = _solve_call(target)
    found = _tf32_products(lowered_for(run, operands, target), "recurrence_alone_solved")
    assert found == expected < whole // 4, (found, expected, whole)


def _on_mma_sync() -> bool:
    return (jax.default_backend() == "gpu"
            and generation.family(generation.gpu_target().generation) == "mma.sync")


def _products_of(target: str, support: product.Support | None):
    """A kernel of `target`'s family taking `lhs @ rhs` `[SIZE, SIZE]` fp32 (TF32) in shared memory, of
    `support`'s terms alone where given (`Issuer.products`'), its accumulator back in device memory."""
    from jax.experimental.pallas import mosaic_gpu as plgpu

    from dew.nn.kernels.templates import mosaic

    family = bundles.of(target, jnp.float32)
    tile = (SIZE, SIZE)

    def body(lhs_gmem, rhs_gmem, out_gmem, lhs, rhs, out):
        lhs[...] = lhs_gmem[...]
        rhs[...] = rhs_gmem[...]
        mosaic.synced(interpret=False)
        acc = family.products(family.zeros(tile), lhs, rhs, support=support)
        out[...] = acc
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(out, out_gmem)
        plgpu.wait_smem_to_gmem(0)

    return jax.jit(plgpu.kernel(body, out_type=jax.ShapeDtypeStruct(tile, jnp.float32), num_threads=1,
                                thread_name="wg", scratch_types=[plgpu.SMEM(tile, jnp.float32)] * 3,
                                compiler_params=mosaic.compiler_params()))


@COMPILES_ON_CPU
@pytest.mark.parametrize("target", [pytest.param(name, marks=lowers(name)) for name in ("sm80", "sm120")])
def test_a_product_of_a_support_issues_its_tiles_instructions_alone(target):
    """Lowered for an A100 and an RTX PRO 6000, a TF32 product of a doubling merge's support
    (`Issuer.products`' `support`) issues one instruction a tile `Support.tiles` gives a term, the whole
    product one a tile of its every tile."""
    support = algebra._merges_support(SIZE, 8)[0]
    rows, columns = bundles.of(target, jnp.float32).term_tile
    shapes = (jax.ShapeDtypeStruct((SIZE, SIZE), jnp.float32),) * 2
    count = {key: _tf32_products(lowered_for(_products_of(target, given), shapes, target), "")
             for key, given in (("support", support), ("whole", None))}
    tiles = support.tiles(rows, columns, 8)
    assert count == {"support": int(tiles.sum()), "whole": tiles.size}, count


@pytest.mark.skipif(not _on_mma_sync(), reason="a product of a support issues on mma.sync's GPUs alone")
def test_a_product_of_its_support_is_the_whole_product_bit_for_bit_on_this_gpu():
    """On this GPU (an mma.sync part's), each of the doubling's merges (`_merges_support`) issued of its
    support's terms alone (`Issuer.products`' `support`) is the whole product of the same operands, drawn
    finite in its support and zero outside it, bit for bit at every output it keeps up to a zero's sign:
    each instruction skipped held only zero terms (the toolkit's review of d345808b0)."""
    target = generation.gpu_target().generation
    draw = np.random.default_rng(0).standard_normal
    whole = _products_of(target, None)
    for width in (2, 4, 8, 16, 32):
        for support in algebra._merges_support(SIZE, width):
            lhs, rhs = (jnp.asarray(np.where(held, draw((SIZE, SIZE)), 0.0), jnp.float32)
                        for held in (support.left, support.right))
            found, expected = (np.asarray(each(lhs, rhs)) for each in (_products_of(target, support), whole))
            assert np.isfinite(expected).all()
            np.testing.assert_array_equal(np.where(support.kept, found, 0.0),
                                          np.where(support.kept, expected, 0.0), err_msg=str(width))
