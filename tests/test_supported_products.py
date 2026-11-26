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


def _tf32_products(target: str) -> int:
    """The TF32 mma.sync instructions of the solve launch (`recurrence_alone_solved`) its module issues."""
    import jax.experimental.mosaic.gpu as mgpu
    from jax.extend.mlir import ir

    from dew.nn.kernels.lowered import kernel_calls

    _, run, operands = _solve_call(target)
    lowered = lowered_for(run, operands, target)
    (kernel,) = [each for each in kernel_calls(lowered)
                 if each.kind == "mosaic_gpu" and each.name.startswith("recurrence_alone_solved")]
    module = mgpu.deserialized(kernel.ir)
    pending, found = [module.operation], 0
    with module.context:
        while pending:
            for region in pending.pop().regions:
                for block in region.blocks:
                    for inner in block.operations:
                        if inner.operation.name == "llvm.inline_asm":
                            text = ir.StringAttr(inner.operation.attributes["asm_string"]).value
                            found += "mma.sync" in text and ".tf32" in text
                        pending.append(inner.operation)
    return found


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
    assert _tf32_products(target) == expected < whole // 4, (expected, whole)


def _on_mma_sync() -> bool:
    return (jax.default_backend() == "gpu"
            and generation.family(generation.gpu_target().generation) == "mma.sync")


@pytest.mark.skipif(not _on_mma_sync(), reason="a product of a support issues on mma.sync's GPUs alone")
def test_the_solve_of_its_supports_is_the_whole_solve_bit_for_bit_on_this_gpu(monkeypatch):
    """On this GPU (an mma.sync part's), Qwen3.5-27B's gated delta net forward with its solve's merges
    issued of their supports alone is the same forward with every merge whole, bit for bit up to a zero's
    sign, over finite draws: each instruction skipped held only zero terms."""
    from dew import _cache
    from dew.nn.kernels.templates import operands

    _, run, shapes = _solve_call(generation.gpu_target().generation)
    draw = np.random.default_rng(0).standard_normal
    given = [jnp.asarray(draw(x.shape), x.dtype) for x in shapes]
    given[3] = -jnp.abs(given[3]) * 0.1
    with _cache.uncached():
        supported = np.asarray(jax.jit(run)(*given))
    monkeypatch.setattr(operands.Issuer, "supported",
                        lambda self, name, lhs, rhs, support: self(name, lhs, rhs))
    with _cache.uncached():
        whole = np.asarray(jax.jit(run)(*given))
    # Equal values, NaN where NaN, a zero's sign aside (`product.Support`'s contract).
    np.testing.assert_array_equal(supported.astype(np.float32), whole.astype(np.float32))
