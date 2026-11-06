"""Scratch diagnostic (not for landing): which epilogues ran fused under the pass."""
import jax
import test_algebra_fusion as t
from dew import algebra
from dew.nn.kernels import generation


def test_probe():
    operands = t.draw()
    lines = []
    for name, named in t.ARMS:
        for layer in t.LAYERS:
            with generation.assumed(name), generation.chosen(**named), generation.routes() as found:
                jax.jit(algebra.rewritten(layer)).lower(*operands)
            ran = sorted({(r.choice, r.ran) for r in found if r.choice in ("gemm_epilogue", "grouped_matmul")})
            lines.append(f"{name} {named} {layer.__name__}: {ran}")
    raise AssertionError("\n".join(lines))
