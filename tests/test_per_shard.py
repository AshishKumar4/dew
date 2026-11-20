"""Every Pallas or FFI kernel runs per shard (docs/design/testing.md, scenarios): each kernel a matrix family
tests (tests/kernel_matrix.py, `CELLS`' modules' `KERNELS`) that its layer's production path runs per shard,
named through that path on a two-device mesh splitting the axes it maps, computes what it computes on one
device by check 1 from float64, its split step moving none of its operands between devices but the moves its
layer intends (`sharded.assert_kernel_runs_per_shard`). Each family states its kernels' cells
(`per_shard(choice, kernel)`, through the layer objects its production path calls); which kernels must have
one is read from production: every kernel a model flow's programs launch split over two devices
(tests/split_flows.py, `split_routes`). A cell runs where its kernel runs: on the CPU's devices where they
interpret it as its generation, on a GPU's (an emulated mesh's among them) where its generation is a GPU's,
on a TPU's where a TPU's."""

import functools
import importlib
import re

import jax
import pytest
from generation_support import here, runnable
from kernel_matrix import CELLS
from model_flows import FLOWS, flow_flags, flow_named
from sharded import Shard, assert_kernel_runs_per_shard, interprets
from split_flows import SPLIT_ROUTES


def kernels() -> list[tuple[str, str, str]]:
    """Every kernel the matrix families test: each family's module, a choice its cells run and a kernel of
    that choice they test (`KERNELS`)."""
    modules = dict.fromkeys(module for module, _ in CELLS.values())
    return [(module, choice, kernel) for module in modules
            for choice, named in importlib.import_module(module).KERNELS.items() for kernel in sorted(named)]


@functools.cache
def stated(module: str, choice: str, kernel: str) -> dict[str, Shard]:
    """The family's per-shard cells of `kernel` (`per_shard`): none where the family states none, a family
    whose kernels no production path runs per shard. Built once a process: its layers' cells take 20 s for
    every family (row 6846), which collection's parameters and each flow's check both read."""
    per_shard = getattr(importlib.import_module(module), "per_shard", None)
    return {} if per_shard is None else per_shard(choice, kernel)


CHECK1_SHARDS = {
    "test_a_kernel_runs_per_shard[delta_decode-fused-gated_delta_net-data]": {"seeds": 8, "per_task": 2},
    "test_a_kernel_runs_per_shard[delta_decode-fused-kimi_delta_attention-data]": {"seeds": 8, "per_task": 2},
    "test_a_kernel_runs_per_shard[delta_decode-fused-mamba2-data]": {"seeds": 8, "per_task": 2},
    "test_a_kernel_runs_per_shard[delta_decode-step-gated_delta_net-data]": {"seeds": 8, "per_task": 2},
    "test_a_kernel_runs_per_shard[delta_decode-step-kimi_delta_attention-data]": {"seeds": 8, "per_task": 2},
    "test_a_kernel_runs_per_shard[delta_decode-step-mamba2-data]": {"seeds": 8, "per_task": 2},
    "test_a_kernel_runs_per_shard[gated_delta_rule-mosaic_gpu-data]": {"seeds": 8, "per_task": 1},
    "test_a_kernel_runs_per_shard[gated_delta_rule-mosaic_gpu-fsdp]": {"seeds": 8, "per_task": 1},
    "test_a_kernel_runs_per_shard[selective_scan-mosaic_gpu-data]": {"seeds": 8, "per_task": 1},
    "test_a_kernel_runs_per_shard[selective_scan-mosaic_gpu-fsdp]": {"seeds": 8, "per_task": 1},
    "test_a_kernel_runs_per_shard[ssd-mosaic_gpu-data]": {"seeds": 8, "per_task": 2},
    "test_a_kernel_runs_per_shard[ssd-mosaic_gpu-fsdp]": {"seeds": 8, "per_task": 2},
}
"""The cells whose check-1 seeds run as map tasks (`check1_shards`), by node id, a literal the CI plan reads
as it reads a mark's (tools/armada/ci.py): an interpreted Mosaic GPU cell's draw takes 15 to 90 s on
the CPU lane (a decode's 15 to 30, an SSD's 40, a selective scan's 50, a chunked delta rule's 85), so each
task takes what fits CI's group target."""


def cells(test: str) -> list:
    """Every per-shard cell the families state, as `test`'s parameters: each marked to run its seeds as map
    tasks where `CHECK1_SHARDS` names its node, its declared count the family's own."""
    def param(module: str, choice: str, kernel: str, label: str, seeds: int):
        # A node id of letters, digits, _ and -, as the plan reads ids (tools/armada/ci.py `sites`).
        name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{choice}-{kernel}-{label}")
        held = CHECK1_SHARDS.get(f"{test}[{name}]")
        assert held is None or held["seeds"] == seeds, (name, held, seeds)
        return pytest.param(module, choice, kernel, label, id=name,
                            marks=[pytest.mark.check1_shards(**held)] if held else [])

    return [param(module, choice, kernel, label, shard.seeds) for module, choice, kernel in kernels()
            for label, shard in stated(module, choice, kernel).items()]


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize(("module", "choice", "kernel", "label"), cells("test_a_kernel_runs_per_shard"))
def test_a_kernel_runs_per_shard(module, choice, kernel, label):
    """One kernel's per-shard cell where this process's devices run it (`sharded.interprets` on the CPU's;
    on a GPU's, a GPU generation's cell this GPU runs, `generation_support.runnable`; on a TPU's, its own
    generation's), skipped where they do not (spec alignment, 9002: on a GPU it took any GPU generation's
    cell for this one's). Its native run takes two devices of one generation, which a host is run on by hand:
    the GPU tier selects `gpu` flows on hosts of one GPU each, so no tier runs these cells natively yet."""
    shard = stated(module, choice, kernel)[label]
    platform = jax.default_backend()
    gpu_cell = shard.target.startswith("sm")
    held = (gpu_cell and runnable(shard.target) if platform == "gpu" else shard.target == here()
            if platform == "tpu" else interprets(choice, kernel, shard))
    if not held:
        pytest.skip(f"{choice} {kernel} runs as {shard.target} on its own devices, not on this {platform}")
    assert_kernel_runs_per_shard(choice, kernel, shard)


@pytest.mark.parametrize("flow", [flow.name for flow in FLOWS])
def test_a_kernel_every_split_flow_launches_has_a_cell(flow):
    """Each kernel off the reference path a model flow's programs launch split over two devices
    (`split_routes`, in a process of the flow's own XLA flags, `model_flows.flow_flags`, two devices its
    own) has a per-shard cell: production names the kernels that must run per shard, never a list."""
    from process_support import device_flags, fresh

    flags = device_flags(2, flow_flags(flow_named(flow)))
    launched = {tuple(pair) for pair in fresh(SPLIT_ROUTES, flow, flags=flags, platforms="cpu")}
    held = {(choice, kernel) for module, choice, kernel in kernels() if stated(module, choice, kernel)}
    assert launched <= held, sorted(launched - held)
