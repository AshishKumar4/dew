"""The attention template's backward choosing one launch or two by the cost model's price
(`attention_mosaic.MosaicBackward._cheaper`): on each GPU the one the receipts measured faster, at their
cells."""

import pytest
from test_attention_mosaic import _backward_call

from dew.nn.kernels import generation

pytestmark = pytest.mark.kernel

MEASURED = {
    # M-NA-13-a100 (9c25a2088): the two launches faster at every cell, its fused dQ adds throttled on the
    # global queue (M-NA-9-a100).
    ("sm80", "MHA d64"): ((2, 4096, 16, 16), 64, False),
    ("sm80", "Llama 3.2 1B"): ((2, 4096, 32, 8), 64, False),
    ("sm80", "Qwen3-8B"): ((2, 4096, 32, 8), 128, False),
    ("sm80", "Qwen3.5 d256"): ((4, 4096, 8, 2), 256, False),
    # ET-A05AB3-h100: the fused launch 5.20 against the two's 6.19 ms at Qwen3-8B causal, 1.19 against 1.53 at
    # width 64.
    ("sm90", "Qwen3-8B"): ((1, 4096, 32, 8), 128, True),
    ("sm90", "width 64"): ((1, 4096, 32, 8), 64, True),
    # An RTX 3090's mma.sync adds its fused dQ as an A100's do, at no rate it states
    # (`Architecture.global_reduction_bytes_per_clock`): the launch priced whole taken.
    ("sm86", "Llama 3.2 1B"): ((2, 4096, 32, 8), 64, False),
}
"""Each cell's call (rows, tokens, query heads, key heads), head width, and whether its measured faster
backward is the fused launch."""


@pytest.mark.parametrize("cell", list(MEASURED), ids=[" ".join(cell) for cell in MEASURED])
def test_the_fast_backward_takes_the_launches_its_receipts_measured_faster(cell, without_deterministic_ops):
    """The fast backward of a plain bf16 causal call at each receipt's cell (`MEASURED`) derives the fused
    launch where it measured faster and the two launches where they did, and on a GPU stating no rate for
    the fused launch's dQ adds the launches it prices whole: the derivation prices both
    (`MosaicBackward.price_config`), its dQ's global adds among the fused launch's work (row 12752)."""
    generation_name, _ = cell
    shape, width, fused = MEASURED[cell]
    with generation.assumed(generation_name):
        target = generation.gpu_target()
        op, ba = _backward_call(shape, width, accurate=False)
        derived = op.derived(ba, target)
    assert derived is not None and derived.fused == fused, derived
