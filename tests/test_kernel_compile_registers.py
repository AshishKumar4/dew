"""Pure-CPU regressions for two ptxas 12.9 register-allocation behaviours."""
import sys
sys.path.insert(0, "tests")
import kernel_compile as kc

PTX = ".maxntid 256, 1, 1\nsetmaxnreg.inc.sync.aligned.u32 256;"

def test_a_spill_free_ignored_raise_does_not_hold(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None):
        if raised < 128:
            return 213, 0, True
        return raised, 0 if raised >= 144 else 160, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    assert kc.register_need(PTX, "sm90") == 144

def test_spill_feasibility_is_not_assumed_monotone(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None):
        return raised, 0 if raised in (192, 216, 224, 232, 240, 248) else 24, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    assert kc.register_need(PTX, "sm90") == 192

def test_a_smaller_spill_free_raise_does_not_admit_a_spilling_launch(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None):
        return min(raised, 255), 16 if raised == 256 else 0, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = dict(budget=256, most=1, consumers=1, producer=40)
    evidence = kc.register_decisions(PTX, "sm90", model)
    assert kc.decisions(evidence, model) == (False, 0)
    assert evidence["caps"][0] == dict(cap=256, registers=255, spilled=16,
                                      compiled=True, holds=False)

def test_each_residency_is_assembled_at_its_own_cap(monkeypatch):
    caps = []
    def assemble(ptx, generation, *, cap, raised=None):
        caps.append(raised)
        # 2 blocks get (512/2 - 40)/1 = 216 registers each.
        return raised, 16 if raised == 216 else 0, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = dict(budget=256, most=2, consumers=1, producer=40)
    assert kc.decisions(kc.register_decisions(PTX, "sm90", model), model) == (True, 1)
    assert caps == [256, 248, 216, 208]

def test_an_ignored_raise_cannot_increase_residency(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None):
        return 232, 0, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = dict(budget=232, most=2, consumers=2, producer=40)
    assert kc.decisions(kc.register_decisions(PTX, "sm90", model), model) == (True, 1)

def test_refusal_inside_one_family_step_is_tolerated(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None):
        return raised, 0 if raised == 232 else 16, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = dict(registers=240, budget=232, most=1, consumers=2, producer=40,
                 register_step=8)
    evidence = kc.register_decisions(PTX, "sm90", model)
    assert evidence["admitted"] and not evidence["refusal_holds"]
    assert kc.decisions(evidence, model) == (False, 0)

def test_refusal_of_a_tile_that_fits_one_step_below_fails(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None):
        return raised, 0, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = dict(registers=240, budget=232, most=1, consumers=2, producer=40,
                 register_step=8)
    evidence = kc.register_decisions(PTX, "sm90", model)
    assert kc.decisions(evidence, model) == (True, 1)

def test_refusal_band_uses_the_family_step(monkeypatch):
    def assemble(ptx, generation, *, cap, raised=None):
        return raised, 0 if raised >= 216 else 16, True
    monkeypatch.setattr(kc, "_ptxas", assemble)
    model = dict(registers=240, budget=224, most=1, consumers=2, producer=40,
                 register_step=16)
    evidence = kc.register_decisions(PTX, "sm90", model)
    assert [check["cap"] for check in evidence["caps"]] == [224, 208]
    assert kc.decisions(evidence, model) == (False, 0)
