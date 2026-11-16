"""Every Pallas or FFI kernel runs per shard (docs/design/testing.md, scenarios): each kernel a matrix family
tests (tests/kernel_matrix.py, `CELLS`' modules' `KERNELS`) that its layer's production path runs per shard,
named through that path on a two-device mesh splitting the axes it maps, computes what it computes on one
device by check 1 from float64, its split step moving none of its operands between devices but the moves its
layer intends (`sharded.assert_kernel_runs_per_shard`). Each family states its kernels' cells
(`per_shard(choice, kernel)`, through the layer objects its production path calls); which kernels must have
one is read from production: every kernel a model flow's programs launch split over two devices
(tests/model_flows.py, `split_routes`). A cell runs where its kernel runs: on the CPU's devices where they
interpret it as its generation, on a GPU's (an emulated mesh's among them) where its generation is a GPU's,
on a TPU's where a TPU's."""

import importlib
import re

import jax
import numpy as np
import pytest
from kernel_matrix import CELLS
from model_flows import FLOWS, KINDS, flow_flags, flow_named, generic_kernels
from sharded import Shard, assert_kernel_runs_per_shard, interprets

GENERATIONS = ("sm80", "sm90")
"""The GPU generations the model flows run on (an A100's, a GH200's: tests/gpu_flow.py's tiers), whose
split programs name the kernels that must run per shard."""


def kernels() -> list[tuple[str, str, str]]:
    """Every kernel the matrix families test: each family's module, a choice its cells run and a kernel of
    that choice they test (`KERNELS`)."""
    modules = dict.fromkeys(module for module, _ in CELLS.values())
    return [(module, choice, kernel) for module in modules
            for choice, named in importlib.import_module(module).KERNELS.items() for kernel in sorted(named)]


def stated(module: str, choice: str, kernel: str) -> dict[str, Shard]:
    """The family's per-shard cells of `kernel` (`per_shard`): none where the family states none, a family
    whose kernels no production path runs per shard."""
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
    """One kernel's per-shard cell where this process's devices run it (`sharded.interprets` on the CPU's,
    its generation's own on a GPU's or a TPU's), skipped where they do not."""
    shard = stated(module, choice, kernel)[label]
    platform = jax.default_backend()
    held = (shard.target.startswith("sm") if platform == "gpu" else not shard.target.startswith("sm")
            if platform == "tpu" else interprets(choice, kernel, shard))
    if not held:
        pytest.skip(f"{choice} {kernel} runs as {shard.target} on its own devices, not on this {platform}")
    assert_kernel_runs_per_shard(choice, kernel, shard)


SPLIT_ROUTES = """import json, sys
import dew  # before jaxlib loads, so XLA_FLAGS' flags are proven held (`dew._devices.flags_hold`)
import test_per_shard
print(json.dumps(test_per_shard.split_routes(sys.argv[1])))
"""
"""A flow's split routes' process (`split_routes`, `test_a_kernel_every_split_flow_launches_has_a_cell`)."""


def split_routes(name: str) -> list[list[str]]:
    """The kernels off the reference path (`generic_kernels`) that flow `name`'s programs (`KINDS`: its
    training step, its decode loop) launch traced as each of `GENERATIONS`, their rows split over a run's
    two-device data axis under a run's layout (`placing`), at the flow's matmul precision, as a run traces
    them: each `[choice, kernel]`."""
    from dew.distributed.placement import Layout, MeshSpec
    from dew.nn.kernels import generation
    from dew.nn.sharding import placing

    flow = flow_named(name)
    generic = generic_kernels()
    mesh = MeshSpec().build(jax.devices()[:2])
    model = flow.model
    params = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=jax.NamedSharding(
            mesh, jax.sharding.PartitionSpec())),
        jax.eval_shape(lambda: model.init(jax.random.key(0), np.zeros((flow.rows, flow.length), np.int32)))[
            "params"])
    ids = jax.ShapeDtypeStruct((flow.rows, flow.length), np.int32,
                               sharding=jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data")))
    found = set()
    for target in GENERATIONS:
        for kind in KINDS:
            # At the flow's own matmul precision (`Flow.precision`), which a kernel's legality may read (an
            # fp32 SSD's Triton port takes the default precision alone), as its run multiplies.
            with (generation.assumed(target), jax.default_matmul_precision(flow.precision),
                  jax.set_mesh(mesh), placing(Layout()), generation.routes() as routes):
                jax.jit(getattr(flow, kind)(model)).trace(params, ids)
            found |= {(str(route.choice), route.kernel) for route in routes
                      if route.kernel not in generic[str(route.choice)]}
    return sorted(map(list, found))


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
