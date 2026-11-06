import jax
from dew.algebra import fusion
from dew.nn.kernels import fused_gemm, generation
import test_algebra_fusion as t


def test_probe():
    x, w, bias = t.draw()
    closed = jax.make_jaxpr(t.layer)(x, w, bias)
    lines = [str(closed)]
    regions = fusion._regions(closed.jaxpr)
    lines.append(f"regions {regions}")
    with generation.assumed("sm90"):
        for region in regions:
            try:
                found = fused_gemm.analysis(fusion._replayed(closed.jaxpr, region, fused_gemm),
                                            *fusion._shapes(region.inputs))
                lines.append(f"analysis {found}")
            except Exception as error:  # noqa: BLE001
                lines.append(f"error {type(error).__name__}: {error}")
    raise AssertionError("\n".join(lines))
