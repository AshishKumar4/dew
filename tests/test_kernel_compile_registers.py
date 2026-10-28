"""Pure-CPU regressions for ptxas 12.9's register allocation as the compile gate decides on it."""
import kernel_compile as kc

from dew.nn.kernels.templates import resources

PTX = ".maxntid 256, 1, 1\nsetmaxnreg.inc.sync.aligned.u32 256;"

def test_a_spill_free_ignored_raise_does_not_hold(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None, relocatable=False):
        if raised < 128:
            return 213, 0, True
        return raised, 0 if raised >= 144 else 160, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    assert kc.register_need(PTX, "sm90", honored=True) == 144

def test_spill_feasibility_is_not_assumed_monotone(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None, relocatable=False):
        return raised, 0 if raised in (192, 216, 224, 232, 240, 248) else 24, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    assert kc.register_need(PTX, "sm90", honored=True) == 192

def test_a_smaller_spill_free_raise_does_not_admit_a_spilling_launch(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None, relocatable=False):
        return min(raised, 255), 16 if raised == 256 else 0, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = {"budget": 256, "most": 1, "consumers": 1, "producer": 40}
    evidence = kc.register_decisions(PTX, "sm90", model, honored=True)
    assert kc.decisions(evidence, model) == (False, 0)

def test_each_residency_is_assembled_at_its_own_cap(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None, relocatable=False):
        # 2 blocks get (512/2 - 40)/1 = 216 registers each.
        return raised, 16 if raised == 216 else 0, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = {"budget": 256, "most": 2, "consumers": 1, "producer": 40}
    assert kc.decisions(kc.register_decisions(PTX, "sm90", model, honored=True), model) == (True, 1)

def test_an_ignored_raise_cannot_increase_residency(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None, relocatable=False):
        return 232, 0, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = {"budget": 232, "most": 2, "consumers": 2, "producer": 40}
    assert kc.decisions(kc.register_decisions(PTX, "sm90", model, honored=True), model) == (True, 1)

def test_refusal_inside_one_family_step_is_tolerated(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None, relocatable=False):
        return raised, 0 if raised == 232 else 16, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = {"registers": 240, "budget": 232, "most": 1, "consumers": 2, "producer": 40,
                 "register_step": 8}
    evidence = kc.register_decisions(PTX, "sm90", model, honored=True)
    assert evidence["admitted"] and not evidence["refusal_holds"]
    assert kc.decisions(evidence, model) == (False, 0)

def test_refusal_of_a_tile_that_fits_one_step_below_fails(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None, relocatable=False):
        return raised, 0, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = {"registers": 240, "budget": 232, "most": 1, "consumers": 2, "producer": 40,
                 "register_step": 8}
    evidence = kc.register_decisions(PTX, "sm90", model, honored=True)
    assert kc.decisions(evidence, model) == (True, 1)

def test_refusal_band_uses_the_family_step(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None, relocatable=False):
        return raised, 0 if raised >= 216 else 16, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = {"registers": 240, "budget": 224, "most": 1, "consumers": 2, "producer": 40,
                 "register_step": 16}
    evidence = kc.register_decisions(PTX, "sm90", model, honored=True)
    assert kc.decisions(evidence, model) == (False, 0)
    # With an eight-register step this same refusal is outside the permitted band.
    model["register_step"] = 8
    assert kc.decisions(kc.register_decisions(PTX, "sm90", model, honored=True), model) == (True, 1)


def test_a_relocatable_compile_decides_a_raising_kernel_at_its_entry_cap(monkeypatch):
    """Where the GPU compiles relocatably, ignoring a raise, the kernel is held to the cap its entry sets
    (two consumers and a producer, 168): a need at 176 refuses a launch the raise would admit."""
    seen = []

    def assemble(ptx, generation, *, cap, raised=None, relocatable=False):
        seen.append((cap, raised, relocatable))
        return min(cap, 176), 0 if cap >= 176 else 24, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    budget = resources.entry_registers(3)
    model = {"budget": budget, "most": 1, "consumers": 2, "producer": budget}
    assert budget == 168
    assert kc.decisions(kc.register_decisions(PTX, "sm90", model, honored=False), model) == (False, 0)
    assert all(raised is None and relocatable for _, raised, relocatable in seen)
