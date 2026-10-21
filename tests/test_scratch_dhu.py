import dataclasses

import fla_gdn_support as support
import pytest
from kernel_compile import compile_kernels

from dew.nn.kernels import fla_gdn


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_dhu_shared(stages):
    launch = fla_gdn._CHUNK_BWD_DHU
    tiles = fla_gdn.Tiles(0, 64, 4, stages)
    shape = support.COMPILED_SHAPES["wide keys"]
    import functools
    case = support.KernelCase(f"dhu/s{stages}", functools.partial(support._case_call, launch, tiles, shape,
                                                                 "bfloat16", reverse=False))
    found = compile_kernels([case], "sm80")
    raise AssertionError(f"DHU|{stages}|" + "; ".join(f"{r['name']} shared {r.get('shared')} limit {r.get('limit')} {r.get('error', '')[:200]}" for r in found))
