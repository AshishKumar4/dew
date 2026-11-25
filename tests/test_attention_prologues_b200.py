"""The template's query and key prologues compiled for a B200
(`test_attention_prologues.compiled_spill_free`): a module of their own, so a process compiling them compiled
no other GPU's kernels before them, and no cell takes a process of its own again
(`kernel_compile.compile_mosaic_kernels`; row 12638: one took 127 s beside the H100's cells)."""

import pytest
from generation_support import COMPILES_ON_CPU
from test_attention_prologues import PROLOGUES, compiled_spill_free

from dew.nn.kernels.templates import attention as template


@COMPILES_ON_CPU
@pytest.mark.parametrize("prologue", PROLOGUES)
@pytest.mark.parametrize("kept", [template.Kept.SPLIT, template.Kept.NONE], ids=["training", "inference"])
def test_a_prologued_forward_compiles_spill_free_at_its_derived_config_on_a_b200(kept, prologue):
    """An attention forward under its query and key prologues over causal tokens at the config its derivation
    gives on a compile-only B200 (the 4B decision call's 16 heads over 4, 256 wide, 12,288 tokens): every
    launch compiled by ptxas spilling nothing (`compiled_spill_free`)."""
    compiled_spill_free("sm100", (12288, 16, 4, 256), kept, prologue)
