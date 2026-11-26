"""The attention template's backward pricing its one launch and its two (`MosaicBackward.price_config`), which
its derivation takes the cheaper of (`MosaicBackward.derived`, row 12752)."""

import dataclasses

import pytest
from test_attention_mosaic import _backward_call

from dew.nn.kernels import generation

pytestmark = pytest.mark.kernel

CELLS = {
    # M-NA-13-a100 (9c25a2088): the two launches measured faster, the fused dQ adds throttled on the global
    # queue (M-NA-9-a100).
    ("sm80", "Llama 3.2 1B"): ((2, 4096, 32, 8), 64),
    ("sm80", "Qwen3.5 d256"): ((4, 4096, 8, 2), 256),
    # ET-A05AB3-h100: the fused launch measured faster.
    ("sm90", "Qwen3-8B"): ((1, 4096, 32, 8), 128),
    # The H100 training step a program certifies (tests/test_attention_mosaic_compile.py's), whose two
    # launches' dQ ptxas refused at 248 registers and 696 bytes of stack.
    ("sm90", "a certified step"): ((1, 1600, 4, 2), 128),
    # An RTX 3090, which states no rate of the fused launch's dQ adds
    # (`Architecture.global_reduction_bytes_per_clock`).
    ("sm86", "Llama 3.2 1B"): ((2, 4096, 32, 8), 64),
}
"""Each cell's call (rows, tokens, query heads, key heads) and head width."""


def _priced(cell, target: generation.GpuTarget | None = None, under: str | None = None):
    """The plain bf16 causal fast backward at `cell` on its GPU (or `target`): its candidates, each one's
    price asked with `under`'s GPU in effect (the cell's where None), and its derived config."""
    name, _ = cell
    shape, width = CELLS[cell]
    with generation.assumed(target or name):
        target = generation.gpu_target()
        op, ba = _backward_call(shape, width, accurate=False)
        candidates = op.candidates(ba, target)
        derived = op.derived(ba, target)
    with generation.assumed(under or target):
        prices = [op.price_config(ba, config, target) for config in candidates]
    return op, ba, target, candidates, prices, derived


@pytest.mark.parametrize("cell", list(CELLS), ids=[" ".join(cell) for cell in CELLS])
def test_a_backward_derives_a_legal_launch_from_prices_its_gpu_states(cell, without_deterministic_ops):
    """Each cell's backward derives a launch it takes, from both candidates' prices; a price is its target's
    whatever GPU is in effect where it is asked (an H100's launches priced with an A100 in effect, which has
    no `setmaxnreg` its WGMMA models read, and an A100's with an H100's)."""
    op, ba, target, candidates, prices, derived = _priced(cell)
    assert derived is not None and op.takes(ba, derived, target), derived
    assert len(candidates) == 2 and all(price is not None for price in prices), (candidates, prices)
    other = "sm80" if target.generation != "sm80" else "sm90"
    assert _priced(cell, under=other)[4] == prices


def test_a_launch_past_its_registers_or_at_a_rate_unstated_is_never_priced_whole(without_deterministic_ops):
    """A candidate whose work lies at a rate its GPU states none of, or whose registers its model holds past
    their cap, is priced unpriced, so a derivation never takes it beside one priced whole: the RTX 3090's
    fused launch (its dQ adds), the H100 certified step's two launches (their dQ refused for spilling), every
    launch on a GPU whose memory's bandwidth is unknown, each launch's device bytes then priced at none."""
    for cell, fused in ((("sm86", "Llama 3.2 1B"), True), (("sm90", "a certified step"), False)):
        _, _, _, candidates, prices, _ = _priced(cell)
        flagged = [unpriced for config, (unpriced, _) in zip(candidates, prices, strict=True)
                   if config.fused == fused]
        assert flagged == [True], (cell, candidates, prices)
    with generation.assumed("sm90"):
        target = generation.gpu_target()
    measured = {name: dataclasses.replace(part, bytes_per_second=None)
                for name, part in target.measured.items()}
    blind = dataclasses.replace(target, memory_bandwidth=None, measured=measured)
    assert blind.memory_rate is None
    _, _, _, candidates, prices, _ = _priced(("sm90", "Qwen3-8B"), target=blind)
    assert candidates and all(unpriced for unpriced, _ in prices), prices


def test_a_derivation_prices_a_gh200_at_its_own_measured_facts():
    """A derivation's priced facts on a GH200 (`attention_mosaic_derivation._priced`) are its part's, its
    device kind and the memory rate measured on it, not the canonical H100's, at its size: a build compiled
    ahead states its part (`generation.built_target`'s `device_kind`), so its prepare and its run price one
    part's facts (row 13252); a kind its generation names no part of prices the canonical one's."""
    from dew.nn.kernels.templates import attention_mosaic_derivation as derivation

    gh200, h100 = (generation.gpu_target(kind) for kind in ("NVIDIA GH200 480GB", "NVIDIA H100 80GB HBM3"))
    priced_gh200, priced_h100 = derivation._priced(gh200), derivation._priced(h100)
    assert priced_gh200.device_kind == gh200.device_kind
    assert priced_gh200.memory_rate == gh200.memory_rate != priced_h100.memory_rate
    unnamed = dataclasses.replace(h100, device_kind="an unnamed sm90 part")
    assert derivation._priced(unnamed).device_kind == generation.TARGETS["sm90"].device_kind

