"""A training step lowered for a CUDA GPU on a host without one.

Dew's Pallas GPU kernels lower to Triton only for a GPU: on a CPU the suite
interprets them, so a kernel whose Triton lowering fails passes every CPU
test and fails the run's first compile on the GPU (c58 and c57, an A100 and
a G4 session, both ended there). `TrainerConfig.lower_for` names a CUDA
generation, and the run then builds as it would and lowers its first step as
on that GPU: every kernel choice and refusal reads the generation
(`dew.nn.kernels.generation.assumed`), the step traces on an abstract mesh of
that GPU's devices, the mesh JAX's Pallas lowering reads the GPU from, and
lowers for CUDA, which runs every Pallas-Triton lowering in the step. XLA's
own compile of the lowered step, and Triton's of each kernel, still need the
GPU.

A choice keyed on the platform reads `generation.backend()`, the assumed
generation's platform, and takes the GPU's path here. Two read
`generation.gpu_placed()`, this process's own GPU, which an assumed one never
is, so the dry run cannot see their GPU path: attention's cuDNN and
FlashAttention calls (`attention_kernels._hosted`), CUDA kernels and not Pallas ones,
whose trace asks the CUDA plugin for versions and libraries a CPU host lacks
and whose lowering asks for devices an abstract mesh lacks, even on a GPU
host, so attention traces its XLA path.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import jax

from dew.nn.kernels import generation
from dew.nn.kernels.generation import Route
from dew.nn.kernels.lowered import KernelCall, kernel_launches
from dew.nn.kernels.templates import report
from dew.telemetry.instrumentation import sizes

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class KernelReport:
    """Which kernels a step runs: each dispatch's decision over the step's trace (`routes`: its choice,
    kernel, the kernel it preferred and why that refused, its site; `generation.routes`) beside the
    launches its lowered program makes (`calls`, `kernel_launches`: how many of each kernel, its config and
    the FLOPs it states). The run log, the run record (`StepCompiled.kernels`, through `Planned`'s) and
    the lowering tool (`Lowered.kernels`, tools/kernel_picks.py) read this one report, and the FLOP count
    sizes each call as its `unsized` does (`instrumentation.sizes`)."""

    routes: tuple[Route, ...]
    calls: tuple[tuple[KernelCall, float], ...]
    """Each kernel call the lowered program makes, once, with the times it runs (`kernel_launches`)."""

    def record(self) -> dict:
        """Its JSON: each distinct route with its count, each kernel's launches, configs, stated FLOPs
        (None where any of its launches states none) and `padded`, the share of its launches' tile grids
        their tails pad, over its calls (`report.Launch.padded`; None where any states none), and
        `unsized`, the launches whose work the FLOP count cannot size (`instrumentation.sizes`); a count
        a loop of no static trip count leaves unknown is None, as the FLOP count's is."""
        routes = Counter(self.routes)
        launches: dict[tuple[str, str], dict] = {}
        for call, runs in self.calls:
            entry = launches.setdefault((call.kind or "", call.name), {
                "kind": call.kind, "name": call.name, "calls": 0.0, "configs": [], "flops": 0.0,
                "padded": 0.0})
            entry["calls"] += runs
            if call.config not in entry["configs"]:
                entry["configs"].append(call.config)
            stated, flops = entry["flops"], call.flops
            entry["flops"] = stated + flops * runs if stated is not None and flops is not None else None
            traced = None if call.launch is None else report.traced(call.launch)
            share = None if traced is None else traced.padded
            held = entry["padded"]
            entry["padded"] = None if held is None or share is None else held + share * runs
        for entry in launches.values():
            entry["calls"] = int(entry["calls"]) if math.isfinite(entry["calls"]) else None
            if entry["flops"] is not None and not math.isfinite(entry["flops"]):
                entry["flops"] = None
            # The padded share over the kernel's calls, each by its runs.
            padded, calls = entry["padded"], entry["calls"]
            entry["padded"] = padded / calls if padded is not None and calls else None

        def order(counted) -> tuple:
            route, _ = counted
            return route.choice, route.site, route.kernel

        return {"routes": [{**asdict(route), "count": count}
                           for route, count in sorted(routes.items(), key=order)],
                "launches": [launches[key] for key in sorted(launches)],
                "unsized": _known(sum(runs for call, runs in self.calls
                                      if not sizes(call.kind, call.flops)))}

    def summary(self) -> str:
        """One line for the run log: each kernel's launches, and the dispatches off their preferred kernel."""
        launches = ", ".join(f"{entry['name']} x{'?' if entry['calls'] is None else entry['calls']}"
                             for entry in self.record()["launches"])
        off = sorted({f"{route.choice}: {route.kernel} not {route.preferred}" for route in self.routes
                      if route.kernel != route.preferred})
        return f"kernels: {launches or 'none'}" + (f"; off their preferred: {'; '.join(off)}" if off else "")


def _known(count: float) -> int | None:
    """A count of launches as a record holds it: None where a loop of no static trip count leaves it
    unknown."""
    return int(count) if math.isfinite(count) else None


def kernel_report(routes, lowered: jax.stages.Lowered) -> KernelReport:
    """The `KernelReport` of a step whose trace reported `routes` and lowered to `lowered`, its calls without
    their kernels' IR, which nothing that reads a report reads: a run keeps the report after it releases
    the program (`planning.Planned.released`, the data lane)."""
    return KernelReport(tuple(routes),
                        tuple((replace(call, ir=b""), runs) for call, runs in kernel_launches(lowered)))


def routed[*Arguments](lower: Callable[[*Arguments], jax.stages.Lowered],
                        *arguments: *Arguments) -> tuple[jax.stages.Lowered, list[Route]]:
    """`lower(*arguments)`'s lowered program and the routes its dispatches reported as it traced
    (`generation.routes`)."""
    with generation.routes() as found:
        lowered = lower(*arguments)
    return lowered, found


class Routed:
    """A step's lowering, `lower()`, that keeps the routes and program of its last call (`routed`): a
    tuning run lowers again where a config moved (`tuning.tune`), and the last is the one compiled."""

    def __init__(self, lower: Callable[[], jax.stages.Lowered]):
        self._lower = lower
        self._last: tuple[jax.stages.Lowered, list[Route]] | None = None

    def __call__(self) -> jax.stages.Lowered:
        self._last = routed(self._lower)
        return self._last[0]

    def report(self) -> KernelReport:
        """The last lowering's `KernelReport`."""
        assert self._last is not None, "a report reads a lowering the step made"
        lowered, found = self._last
        return kernel_report(found, lowered)


class Lowered(SystemExit):
    """Raised by `Trainer.fit` under `lower_for` once the run's first step has lowered for its target.

    Lowering was the whole run, so uncaught it ends the program with status 0
    and no traceback, the step's kernels logged (`--trainer.lower-for`'s dry run). `kernels` is the step's
    kernel report (`KernelReport`: its dispatches' routes and its lowered launches), and `lowered` is
    the step itself. A step for devices this host is not compiles ahead (`dew.training.ahead`).
    """

    def __init__(self, target: str, lowered: jax.stages.Lowered, routes=()):
        super().__init__(0)
        self.generation = target
        self.lowered = lowered
        self.kernels = kernel_report(routes, lowered)
        _log.info("the step lowered for %s; its %s", target, self.kernels.summary())
