"""The code Dew keeps for upstream bugs, by the issue each waits on. It goes the
moment the pinned release carries the fix, so a jax pin that moves fails this
until every row is checked against the new release and deleted or kept."""

import jax
import jaxlib

PINNED = ("0.11.2", "0.11.2")
"""jax and jaxlib, as constraints.txt pins them."""

WORKAROUNDS = {
    "openxla/xla#49380, fixed by #49498": "dew.nn.scatter.DROPPED and tools/xla_scatter_drop_repro.py",
    "jax-ml/jax#40940": "constraints.txt's patched jax, and dew.training.runtime's pool cache refusal",
    "jax-ml/jax#40907": "the checks dew.nn.kv_cache and dew.inference.serving_kernel make outside shard_map",
    "openxla/xla#46500": "dew.nn.attention's refusal of cudnn under scan_layers",
    "openxla/xla#49299": "dew.diffusion.schedules.source_grids's whole-grid gather",
    "openxla/xla#50052": "dew.training.rungs's autotune regions",
    "Dew #1 (openxla/xla#49635, libtpu 0.0.50 pairs with the next jax)": "pyproject's libtpu 0.0.48 pin",
}


def test_the_jax_pin_is_the_one_the_workarounds_were_checked_against():
    pinned = (".".join(jax.__version__.split(".")[:3]), jaxlib.__version__)
    assert pinned == PINNED, (
        f"jax {pinned} is not the {PINNED} these were checked against; check each against the new "
        "release, delete what it fixes, then update PINNED:\n"
        + "\n".join(f"  {issue}: {code}" for issue, code in WORKAROUNDS.items()))
