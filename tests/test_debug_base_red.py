"""Scratch reproduction: print the derived launch and assembler's register decisions."""

import json

import pytest
from fla_gdn_support import KernelCase
from kernel_compile import compile_mosaic_kernels, mosaic_failures
from test_deepest_rings import CELLS, deepest

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention_mosaic, base, mosaic_sm90, resources


@pytest.mark.parametrize("cell,name", [("attention forward 128", "sm90"),
                                     ("attention backward 64", "sm80"),
                                     ("attention backward 64", "sm100")])
@pytest.mark.parametrize("mode", ["original", "prior_rule"])
def test_explain(cell, name, mode, monkeypatch):
    base._DERIVED.clear()
    if mode == "prior_rule":
        if cell.startswith("attention forward"):
            monkeypatch.setattr(mosaic_sm90.Wgmma, "derived_consumers", property(lambda self: 3))
        else:
            monkeypatch.setattr(attention_mosaic.MosaicBackward, "ranks_alone", lambda self, ba, target: False)
    with generation.assumed(name):
        op, ba, config = CELLS[cell][0]()
        at = deepest(op, ba, config, generation.gpu_target())
        found = compile_mosaic_kernels([KernelCase(f"{cell} {mode}", lambda: op.lowering(ba, at))], name)
        summary = {"cell": cell, "generation": name, "mode": mode, "derived": repr(config),
                   "deepest": repr(at), "registers": op.registers(ba, at),
                   "budget": op.register_budget(at),
                   "assembled": [{"models": record.get("models"),
                                  "kernels": [{key: kernel.get(key) for key in
                                               ("kernel", "shared", "registers", "spilled", "cap")}
                                              for kernel in record.get("kernels", [])]}
                                 for record in found]}
    print("EXPLAIN " + json.dumps(summary, default=str), flush=True)
    assert not (failed := mosaic_failures(found)), "\n".join(failed)


@pytest.mark.parametrize("mode", ["original", "prior_rule"])
def test_explain_matrix(mode, monkeypatch):
    import attention_matrix
    import kernel_matrix

    base._DERIVED.clear()
    attention_matrix._route.cache_clear()
    if mode == "prior_rule":
        monkeypatch.setattr(attention_mosaic.MosaicBackward, "ranks_alone", lambda self, ba, target: False)
    failed = []
    for label, case in kernel_matrix.covered("attention"):
        if case["generation"] != "sm90" or case["width"] != "128" or case["launch"] != "shared-keys":
            continue
        with generation.assumed(case["generation"]):
            op, ba, target = attention_matrix._backward_op(case)
            config = op.derived(ba, target)
            supported = attention_matrix.supported(case)
        print("MATRIX " + json.dumps({"mode": mode, "label": label, "config": repr(config),
                                      "supported": supported}), flush=True)
        if not supported:
            failed.append(label)
    assert not failed, failed


@pytest.mark.parametrize("cell,name", [("attention forward 128", "sm90"),
                                     ("attention backward 64", "sm80"),
                                     ("attention backward 64", "sm100")])
def test_certified_deepest(cell, name):
    base._DERIVED.clear()
    selected = []
    with generation.assumed(name):
        op, ba, config = CELLS[cell][0]()
        _, shapes = op.lowering(ba, config)

        def routed(*arrays):
            chosen = op.taken(ba)
            assert chosen is not None
            at = deepest(op, ba, chosen, generation.gpu_target())
            selected.append((repr(chosen), repr(at)))
            fn, _ = op.lowering(ba, at)
            return fn(*arrays)

        found = compile_mosaic_kernels([KernelCase(cell, lambda: (routed, shapes))], name)
    print("CERTIFIED " + json.dumps({"cell": cell, "generation": name, "chosen": selected,
                                     "failures": mosaic_failures(found)}, default=str), flush=True)
    assert not (failed := mosaic_failures(found)), "\n".join(failed)


def test_derived_decisions():
    for cell, name in (("attention forward 128", "sm90"), ("attention backward 64", "sm80"),
                       ("attention backward 64", "sm100")):
        base._DERIVED.clear()
        with generation.assumed(name):
            op, ba, config = CELLS[cell][0]()
            at = deepest(op, ba, config, generation.gpu_target())
            print("DECISION " + json.dumps({"cell": cell, "generation": name, "derived": repr(config),
                                            "deepest": repr(at), "registers": op.registers(ba, at),
                                            "budget": op.register_budget(at)}, default=str), flush=True)


def test_forward_raise(monkeypatch):
    original = resources.moves_registers
    monkeypatch.setattr(resources, "moves_registers",
                        lambda registers, warpgroups, consumers, programs=1:
                        (warpgroups == 3 and consumers == 2)
                        or original(registers, warpgroups, consumers, programs))
    base._DERIVED.clear()
    with generation.assumed("sm90"):
        op, ba, config = CELLS["attention forward 128"][0]()
        at = deepest(op, ba, config, generation.gpu_target())
        found = compile_mosaic_kernels([KernelCase("forward raised", lambda: op.lowering(ba, at))], "sm90")
    print("RAISED " + json.dumps({"config": repr(at), "kernels":
                                 [{key: kernel.get(key) for key in
                                   ("kernel", "registers", "spilled", "cap")}
                                  for record in found for kernel in record.get("kernels", [])]}), flush=True)
    assert not (failed := mosaic_failures(found)), "\n".join(failed)


@pytest.mark.parametrize("index", [0, 1])
def test_certified_matrix(index):
    import attention_matrix
    import kernel_matrix

    label, case = [(label, case) for label, case in kernel_matrix.covered("attention")
                   if case["generation"] == "sm90" and case["width"] == "128"
                   and case["launch"] == "shared-keys"][index]
    base._DERIVED.clear()
    chosen = []
    with generation.assumed("sm90"):
        op, ba, target = attention_matrix._backward_op(case)
        raw = op.derived(ba, target)
        _, shapes = op.lowering(ba, raw)

        def routed(*arrays):
            config = op.taken(ba)
            chosen.append(repr(config))
            assert config is not None
            fn, _ = op.lowering(ba, config)
            return fn(*arrays)

        found = compile_mosaic_kernels([KernelCase(label, lambda: (routed, shapes))], "sm90")
    print("CERTIFIED_MATRIX " + json.dumps({"label": label, "raw": repr(raw), "chosen": chosen,
                                            "failures": mosaic_failures(found)}), flush=True)
    assert not (failed := mosaic_failures(found)), "\n".join(failed)
