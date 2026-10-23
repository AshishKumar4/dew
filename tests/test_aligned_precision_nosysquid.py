"""Fixed-draw selective-widening probes. Each item runs in an isolated process on armada."""
import pytest
from probe_aligned_dots_nosysquid import install
from test_aligned_structural_nosysquid import measure

CORE = "intra.p,scores.p,local.l,merge.l,merge.r,apply_u.l,apply_w.l,apply_w.r"
VARIANTS = {
    "two-primal": "intra.p,scores.p",
    "core": CORE,
    "core-more": CORE + ",local.e,memory.e,corrected.r,write.l,write.r",
    "core-vector": CORE + ",intra.d,scores.d",
    "no-merge": "intra.p,scores.p,local.l,apply_u.l,apply_w.l,apply_w.r",
    "transpose": ",".join(name + "." + side
                           for name in ("merge", "intra", "scores", "local", "apply_w", "apply_u",
                                        "corrected", "memory", "write") for side in ("l", "r", "e", "d")),
    "all": "all",
}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("width,size", [pytest.param(16, 16, id="small"),
                                       pytest.param(128, 64, id="production")])
@pytest.mark.parametrize("name,seeds", [pytest.param("scalar delta", 40, id="scalar"),
                                       pytest.param("vector delta", 40, id="vector"),
                                       pytest.param("gated", 320, id="gated")])
def test_precision(variant, width, size, name, seeds, monkeypatch):
    install(VARIANTS[variant])
    print("VARIANT", variant, flush=True)
    measure(width, size, name, seeds, monkeypatch)


@pytest.mark.parametrize("variant", ["chunk-adjoint", "chunk-adjoint-fast"])
@pytest.mark.parametrize("width,size", [pytest.param(16, 16, id="small"),
                                       pytest.param(128, 64, id="production")])
@pytest.mark.parametrize("name,seeds", [pytest.param("scalar delta", 40, id="scalar"),
                                       pytest.param("vector delta", 40, id="vector"),
                                       pytest.param("gated", 320, id="gated")])
def test_adjoint_ratios(variant, width, size, name, seeds, monkeypatch):
    from dew.nn.kernels import recurrence_chunk as c
    from probe_aligned_chunk_adjoint_nosysquid import chunk_adjoint, chunk_adjoint_wy_source
    monkeypatch.setattr(c, "chunk_delta_rule", {"chunk-adjoint": chunk_adjoint,
                                              "chunk-adjoint-fast": chunk_adjoint_wy_source}[variant])
    print("VARIANT", variant, flush=True)
    measure(width, size, name, seeds, monkeypatch)
