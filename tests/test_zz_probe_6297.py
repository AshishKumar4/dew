"""A probe (not for landing): why each diffusion objective's trace key is None."""

from test_trace_keys import built_objectives  # noqa: F401

from dew.objectives.base import UNKEYED, Objective, frozen_value


def test_probe(built_objectives):  # noqa: F811
    from dew.objectives.diffusion.objective import DiffusionObjective, _keyed_inputs

    for name in ("guidance_distillation", "ladd", "rcm", "run diffusion", "flow_grpo", "diffusion"):
        objective, _ = built_objectives[name]()
        print("##", name, type(objective).__name__, "key None:", objective.trace_key() is None)
        print("  ema", objective.ema is not None and objective.ema.stated)
        left = type(objective)._untraced()
        for held, value in sorted(vars(objective).items()):
            if held in left or held in ("variables", "ema"):
                continue
            if held == "_inputs" and value is not None:
                keyed = _keyed_inputs(value)
                print("  _inputs keyed:", frozen_value(keyed) is UNKEYED)
                for part in keyed[:2]:
                    print("    part", type(part).__name__, frozen_value(part) is UNKEYED)
                for entry in keyed[2]:
                    print("    cond", [(type(x).__name__, frozen_value(x) is UNKEYED) for x in entry])
                    if frozen_value(entry[2]) is UNKEYED:
                        print("      unconditional", repr(entry[2])[:200])
                continue
            if isinstance(value, Objective):
                print("  objective", held, "key None:", value.trace_key() is None,
                      "ema", value.ema is not None and value.ema.stated)
                continue
            if frozen_value(value) is UNKEYED:
                print("  UNKEYED", held, type(value).__name__)
        if isinstance(objective, DiffusionObjective):
            settings = {held: value for held, value in vars(objective).items() if held not in left}
            print("  traced_by None:", objective.traced_by(settings) is None)
