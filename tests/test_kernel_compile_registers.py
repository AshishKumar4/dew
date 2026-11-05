"""Pure-CPU regressions for ptxas 12.9's register allocation as the compile gate decides on it."""
import dataclasses

import kernel_compile as kc

from dew.nn.kernels.templates import resources


def test_a_spill_free_ignored_raise_does_not_hold():
    def assemble(cap):
        if cap < 128:
            return 213, 0, True
        return cap, 0 if cap >= 144 else 160, True
    assert kc.least_spill_free(assemble, raising=True) == 144

def test_spill_feasibility_is_not_assumed_monotone():
    def assemble(cap):
        return cap, 0 if cap in (192, 216, 224, 232, 240, 248) else 24, True
    assert kc.least_spill_free(assemble, raising=True) == 192

def test_a_smaller_spill_free_raise_does_not_admit_a_spilling_launch():
    def assemble(cap):
        return min(cap, 255), 16 if cap == 256 else 0, True
    model = resources.RegisterModel(registers=None, budget=256, consumers=1, most=1, step=8, producer=40)
    evidence = kc.cap_decisions(assemble, model, raising=True, honored=True)
    assert kc.decisions(evidence, model) == (False, 0)

def test_each_residency_is_assembled_at_its_own_cap():
    def assemble(cap):
        # 2 blocks get (512/2 - 40)/1 = 216 registers each.
        return cap, 16 if cap == 216 else 0, True
    model = resources.RegisterModel(registers=None, budget=256, consumers=1, most=2, step=8, producer=40)
    assert kc.decisions(kc.cap_decisions(assemble, model, raising=True, honored=True), model) == (True, 1)

def test_an_ignored_raise_cannot_increase_residency():
    def assemble(cap):
        return 232, 0, True
    model = resources.RegisterModel(registers=None, budget=232, consumers=2, most=2, step=8, producer=40)
    assert kc.decisions(kc.cap_decisions(assemble, model, raising=True, honored=True), model) == (True, 1)

def test_refusal_inside_one_family_step_is_tolerated():
    def assemble(cap):
        return cap, 0 if cap == 232 else 16, True
    model = resources.RegisterModel(registers=240, budget=232, consumers=2, most=1, step=8, producer=40)
    evidence = kc.cap_decisions(assemble, model, raising=True, honored=True)
    assert evidence["admitted"] and not evidence["refusal_holds"]
    assert kc.decisions(evidence, model) == (False, 0)

def test_refusal_of_a_tile_that_fits_one_step_below_fails():
    def assemble(cap):
        return cap, 0, True
    model = resources.RegisterModel(registers=240, budget=232, consumers=2, most=1, step=8, producer=40)
    evidence = kc.cap_decisions(assemble, model, raising=True, honored=True)
    assert kc.decisions(evidence, model) == (True, 1)

def test_refusal_band_uses_the_family_step():
    def assemble(cap):
        return cap, 0 if cap >= 216 else 16, True
    model = resources.RegisterModel(registers=240, budget=224, consumers=2, most=1, step=16, producer=40)
    evidence = kc.cap_decisions(assemble, model, raising=True, honored=True)
    assert kc.decisions(evidence, model) == (False, 0)
    # With an eight-register step this same refusal is outside the permitted band.
    model = dataclasses.replace(model, step=8)
    assert kc.decisions(kc.cap_decisions(assemble, model, raising=True, honored=True), model) == (True, 1)


def test_a_relocatable_compile_decides_a_raising_kernel_at_its_entry_cap():
    """Where the GPU compiles relocatably, ignoring a raise, the kernel is held to the cap its entry sets
    (two consumers and a producer, 168): a need at 176 refuses a launch the raise would admit."""
    def assemble(cap):
        return min(cap, 176), 0 if cap >= 176 else 24, True
    budget = resources.entry_registers(3)
    model = resources.RegisterModel(registers=None, budget=budget, consumers=2, most=1, step=8,
                                    producer=budget)
    assert budget == 168
    assert kc.decisions(kc.cap_decisions(assemble, model, raising=False, honored=False), model) == (False, 0)


SERIALIZING = ("ptxas info    : (C7512) Potential Performance Loss: wgmma.mma_async instructions are "
               "serialized due to insufficient register resources for the function "
               "'recurrence_output_mosaic_gpu_kernel'\n"
               "ptxas info    : Compiling entry function 'recurrence_output_mosaic_gpu_kernel' for 'sm_90a'\n"
               "ptxas info    : Used 152 registers, used 1 barriers, 0 bytes smem\n"
               "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads")
"""ptxas's report of the recurrence output kernel at a consumer cap of 152 (FS-LT-OUTPUT-CAPS1): spill-free,
its warpgroup MMAs serialized."""


def test_a_kernel_ptxas_serializes_fails_the_gate_as_a_spill_does():
    """A Mosaic GPU kernel whose warpgroup MMAs ptxas serializes (C7510 to C7515: each HGMMA waited, as a
    spill-free 152-register recurrence output was) fails the compile gate (`mosaic_failures`), its registers
    and spills clean: its MMAs run one at a time (row 3615; the kernel challenge's C7511 at 112 registers,
    C7512 at 96), which the registers and spills the gate read did not state."""
    read = kc.ptxas_read(SERIALIZING)
    assert (read.registers, read.spilled, read.losses) == (152, 0, ("C7512",))
    (failure,) = kc.mosaic_failures([_assembled_record("recurrence_output_mosaic_gpu_kernel", read)])
    assert "serialize" in failure


IGNORING = ("ptxas info    : (C7507) Potential Performance Loss: 'setmaxnreg' ignored to maintain minimum "
            "register requirements\n"
            "ptxas info    : Compiling entry function 'gemm_mosaic_gpu_kernel' for 'sm_90a'\n"
            "ptxas info    : Used 168 registers, used 1 barriers, 0 bytes smem\n"
            "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads")
"""ptxas's report of a warp-specialized kernel whose roles' paths rejoin after their setmaxnreg: the kernel
challenge's C7507 (notes/ptxas-re.md §4, 'setmaxnreg' ignored), its threads left at the entry's registers."""


def test_a_kernel_whose_register_raises_ptxas_ignores_fails_the_gate():
    """A Mosaic GPU kernel whose setmaxnreg ptxas ignores (C7507: code after the warp roles' split reachable
    from both budgets, so every raise and decrease is dropped and the threads keep the entry's registers)
    fails the compile gate (`mosaic_failures`), its registers and spills clean: its consumers never hold the
    registers its tile was sized for (row 4071)."""
    read = kc.ptxas_read(IGNORING)
    assert (read.registers, read.spilled, read.losses) == (168, 0, ("C7507",))
    (failure,) = kc.mosaic_failures([_assembled_record("gemm_mosaic_gpu_kernel", read)])
    assert "setmaxnreg" in failure


def _assembled_record(kernel: str, read) -> dict:
    """A compile record of one spill-free H100 kernel as ptxas `read` it."""
    return {"name": kernel, "generation": "sm90", "limit": 232448, "tmem_limit": 0,
            "kernels": [{"kernel": kernel, "shared": 1024, "tmem": 0, "registers": read.registers,
                         "spilled": read.spilled, "losses": read.losses, "unread": []}]}
