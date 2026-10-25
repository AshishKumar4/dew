"""The analytic features of one launch at one config on one target, which the choice layer logs beside
every timing and a ranker learns from (design §7).

A row has three parts, kept apart so a ranker reads only what a trace has:
- `legacy`: the inputs the analytic estimate reads (`Template.costed`), as they are, so the estimate is
  reproduced from a row bitwise (`cost.seconds`);
- `geometry` and `predictions`: the config's own fields, and quantities derived once from the legacy
  inputs and the target's facts, each in its unit, None where the target states no fact it needs;
- `facts`: the target's facts as the predictions read them. Its generation and device name are
  provenance (`timings.observation`), never a predictor.

A launch without a `costed` record has no features (`features` is None), and its timings are logged
all the same.
"""

import dataclasses
import math
from collections.abc import Mapping

from dew.nn.kernels import generation
from dew.nn.kernels.templates import base, cost

SCHEMA = 2
"""The features' format, raised when a field changes meaning."""


@dataclasses.dataclass(frozen=True)
class Features:
    """One launch's features at one config on one target (the module's docstring)."""
    legacy: cost.Cost
    geometry: Mapping[str, int | bool]
    predictions: Mapping[str, float | None]
    facts: Mapping[str, float | None]

    def row(self) -> dict:
        """The features as a timing row carries them: plain JSON values."""
        return {'schema': SCHEMA, 'legacy': dataclasses.asdict(self.legacy), 'geometry': dict(self.geometry),
                'predictions': dict(self.predictions), 'facts': dict(self.facts)}


def features(launch, ba, config, target: generation.GpuTarget) -> Features | None:
    """The features of `launch` running `ba` at `config` on `target`, or None where the launch states no
    `costed` record of its own (`Template.costed`): any other kernel op, or a template still costing by
    an override of `cost`."""
    if not isinstance(launch, base.Template) or type(launch).costed is base.Template.costed:
        return None
    costed = launch.costed(ba, config, target)
    facts = generation.effective(target)
    return Features(costed, _geometry(config), _predictions(costed, facts), _facts(costed, facts))


def _geometry(config) -> dict[str, int | bool]:
    """A config's own fields, where it is a dataclass of numbers (every template's is)."""
    if not dataclasses.is_dataclass(config):
        return {}
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)
            if isinstance(getattr(config, field.name), int | bool)}


def _facts(costed: cost.Cost, facts: generation.GpuTarget) -> dict[str, float | None]:
    rate = facts.tensor_rate(costed.mode) if costed.program_flops else None
    return {'multiprocessors': facts.multiprocessors, 'clock_hz': facts.clock_hz,
            'memory_bandwidth': facts.memory_bandwidth, 'tensor_rate_per_clock': rate,
            'shared_limit': facts.shared_limit, 'load_latency_clocks': facts.load_latency,
            'warpgroup_mma': facts.wgmma}


def _predictions(costed: cost.Cost, facts: generation.GpuTarget) -> dict[str, float | None]:
    """Quantities derived from the legacy inputs and the target's facts:
    - `flops`, the launch's tensor-core FLOPs; `intensity`, those per device byte;
    - `balance`, the target's tensor FLOPs per byte of memory bandwidth, and `ridge`, intensity over it
      (above one, the launch is the tensor cores' to bound, below, memory's);
    - `waves`, the programs over what the multiprocessors hold at once, and `tail`, the share of the
      last wave's slots no program fills;
    - `staged_per_step`, shared-memory bytes a program moves per pipeline step beside its matrix
      instructions, and `serial_per_step`, those it moves between them, which they wait for;
    - `hidden_steps`, the steps its stages in flight cover (`stages - 1`, at most its steps) and
      `waiting_share`, the waiting term's share beside the resident programs' service time
      (`cost.program_seconds`, `cost.waiting_seconds`, as `cost.seconds` reads them)."""
    flops = costed.programs * costed.program_flops
    multiprocessors, clock, bandwidth = facts.multiprocessors, facts.clock_hz, facts.memory_bandwidth
    rate = facts.tensor_rate(costed.mode) if costed.program_flops else None
    balance = (rate * multiprocessors * clock / bandwidth
               if rate is not None and multiprocessors is not None and clock is not None and bandwidth
               else None)
    intensity = flops / costed.device_bytes if costed.device_bytes else None
    slots = multiprocessors * costed.resident if multiprocessors is not None else None
    waves = costed.programs / slots if slots else None
    tail = math.ceil(waves) - waves if waves else None
    # A launch with no tensor work reads no rate, and a single-warp one its family's share of it, as
    # `cost.seconds` does.
    service = 1.0 if not costed.program_flops else rate
    if service is not None and costed.program_flops and costed.single_warp and facts.single_warp_issue:
        service *= facts.single_warp_issue
    resident = (min(costed.resident, max(math.ceil(costed.programs / multiprocessors), 1))
                if multiprocessors else costed.resident)
    program = (cost.program_seconds(costed, service, clock)
               if service is not None and clock is not None else None)
    waiting = (cost.waiting_seconds(costed, facts.load_latency, clock)
               if facts.load_latency is not None and clock is not None else None)
    return {
        'flops': flops, 'intensity': intensity, 'balance': balance,
        'ridge': intensity / balance if intensity is not None and balance else None,
        'waves': waves, 'tail': tail,
        'staged_per_step': costed.program_staged / costed.steps if costed.steps else None,
        'serial_per_step': costed.serial_staged / costed.steps if costed.steps else None,
        'hidden_steps': max(min(costed.stages - 1, math.ceil(costed.steps)), 1),
        'waiting_share': (waiting / (waiting + resident * program)
                          if waiting is not None and program is not None and waiting + program else None),
    }
