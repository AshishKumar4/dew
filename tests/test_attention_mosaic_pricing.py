"""The attention template's backward pricing its one launch and its two (`MosaicBackward.price_config`), which
its derivation takes the cheaper of (`MosaicBackward.derived`, row 12752), and the bytes each launch states
its device memory's (row 14206)."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.state import types as state_types
from test_attention_mosaic import FED, FORM_WIDTHS, WGMMA, _backward_call, _pinned

from dew.nn import programs
from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_mosaic_derivation,
    cost,
)

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


@pytest.mark.parametrize("accurate, pin, launches, width", [
    (True, None, ("attention_dq", "attention_dkv"), 128),
    (False, attention_mosaic.BackwardTiles(64, 64, 2, 2, fused=True, family="wgmma.fed"),
     ("attention_dkv_fused",), 128),
    (False, attention_mosaic.BackwardTiles(128, 64, 2, 1, fused=True, family=WGMMA, **FED),
     ("attention_dkv_fused",), FORM_WIDTHS["fused-keys-first"]),
    (False, attention_mosaic.BackwardTiles(64, 64, 2, 2, fused=True, family=FED["dkv_family"],
                                           shared_keys=True), ("attention_dkv_shared",),
     FORM_WIDTHS["fused-shared-keys"]),
], ids=["dq-dkv", "fused", "fused-keys-first", "fused-shared-keys"])
def test_each_launch_of_a_training_attention_states_its_outputs_beside_its_operands(accurate, pin, launches,
                                                                                    width):
    """Every launch of a grouped causal attention's training forward and backward traced for an H100 (the
    accurate form's dQ and dK/dV launches, or a fused launch pinned: queries first, keys first, or keys
    shared by its consumers) states its outputs' bytes and the gradients' it adds into in place beside the
    operands its role reads, each once: a peak walk its query and keys (no values), the training walk its
    query, keys and values, a backward launch those and the output gradient. What it writes is compulsory
    traffic its device memory prices, every route alike (row 14206; Train 1513's review)."""
    shape = jax.ShapeDtypeStruct((1, 1024, 16, width), jnp.bfloat16)
    shared = jax.ShapeDtypeStruct((1, 1024, 4, width), jnp.bfloat16)
    spans = template.spans(1024, 1024, 4, causal=True)

    def trained(query, key, value):
        out, *residuals = attention_mosaic.forward(query, key, value, spans, natural=False,
                                                   residuals=template.Kept.SPLIT)
        with _pinned(backward=pin):
            return attention_mosaic.backward(query, key, value, spans, None, out, residuals=residuals,
                                             natural=False, accurate=accurate)

    with generation.assumed("sm90"):
        traced = jax.make_jaxpr(jax.jit(trained))(shape, shared, shared).jaxpr
        stating = [equation for equation in programs.equations(traced)
                   if (cost := equation.params.get("cost_estimate")) is not None and cost.bytes_accessed]
    # Each priced launch by its kernel's name (`name__l<its launch>`): the training forward's two and the
    # route's own, no other.
    names = [str(equation.params.get("name")).split("__")[0] for equation in stating]
    assert sorted(names) == sorted(("attention_peaks", "attention_training", *launches)), names

    def stored(variables) -> int:
        return sum(int(np.prod(var.aval.shape)) * var.aval.dtype.itemsize for var in variables)

    # The leading operands each launch's role reads.
    read = {"attention_peaks": 2, "attention_training": 3}
    for equation, name in zip(stating, names, strict=True):
        operands = read.get(name, 4)
        written = [var for var in equation.invars if isinstance(var.aval, state_types.AbstractRef)]
        stated = stored(equation.invars[:operands]) + stored(equation.outvars) + stored(written)
        assert equation.params["cost_estimate"].bytes_accessed == stated, name


def test_a_portable_choice_reads_no_parts_l2():
    """An RTX PRO 6000's priced attention choice reads its generation's canonical facts with no L2 read
    plateau, on the Server Edition measured at one (M-FS-113) and on the Workstation part measured at none
    alike: a part's L2 is its own (kernel-templates.md §6, Train 1508's F2)."""
    server, workstation = (generation.gpu_target(f"NVIDIA RTX PRO 6000 Blackwell {edition} Edition")
                           for edition in ("Server", "Workstation"))
    assert server.l2_rate is not None and workstation.l2_rate is None
    for target in (server, workstation):
        assert cost.rates(attention_mosaic_derivation._priced(target), device=True).l2 == math.inf
