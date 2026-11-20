"""A probe (not for landing): why each diffusion objective's trace key is None."""

import functools

from test_trace_keys import built_objectives  # noqa: F401

from dew.objectives.base import HELD, UNKEYED, Objective, frozen_value


def test_probe(built_objectives):  # noqa: F811
    for name in ("guidance_distillation", "ladd", "rcm", "run diffusion"):
        objective, _ = built_objectives[name]()
        print("##", name, type(objective).__name__, "key None:", objective.trace_key() is None)
        print("  ema", objective.ema is not None and objective.ema.stated)
        left = set(type(objective)._untraced())
        cached = {held for klass in type(objective).__mro__ for held, member in vars(klass).items()
                  if isinstance(member, functools.cached_property)} - left
        run = {id(entry.module) for entry in objective.program_key()}
        state = {held: getattr(objective, held) for held in sorted(cached)}
        state.update((held, value) for held, value in vars(objective).items()
                     if held not in left and held not in HELD and id(value) not in run
                     and not objective._classes(held, value))
        for held, value in sorted(state.items()):
            if isinstance(value, Objective):
                print("  objective", held, "key None:", value.trace_key() is None)
                continue
            if held == "_inputs":
                continue
            if frozen_value(value) is UNKEYED:
                print("  UNKEYED", held, type(value).__name__, repr(value)[:160])
        own = {held: value for held, value in state.items() if not isinstance(value, Objective)}
        print("  traced_by None:", objective.traced_by(own) is None)
