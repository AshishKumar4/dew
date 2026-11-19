"""FlaxDiff's hybrid SSM-attention DiT, the model tools/flaxdiff_reference.py's `hybrid_dit` architecture
records, from FlaxDiff at that tool's HYBRID_COMMIT: a file of its own, as that commit is not the one its
simple model comes from, and tools/check_tool_types.py checks a file in one environment."""

from collections.abc import Callable

import flax.linen as nn

COHORT = "flaxdiff-hybrid"
"""The reference releases this tool runs against (tools/reference_releases.py), and the environment
tools/check_tool_types.py checks it in."""


def hybrid_dit() -> Callable[..., nn.Module]:
    """The hybrid DiT's class, imported once the run has put its FlaxDiff checkout on the path."""
    from flaxdiff.models.ssm_dit import HybridSSMAttentionDiT

    return HybridSSMAttentionDiT
