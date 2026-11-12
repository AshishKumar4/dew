"""A process that keeps the executables of the programs it compiles, or runs them handed
(tests/test_compile_ahead.py): `python kept_programs_worker.py keep|take|verify|differences <file> <scale>
<offset>`. The programs' functions and their static argument's class are `kept_module`'s, a module the
test writes beside the file and puts on the path, so its source is a module's outside any installed
distribution. Prints, as JSON, each program's answer with what was kept and refused, the traces and
compiles the calls needed, the refusal a handing raised, or what the file's records hold otherwise than
this process."""

import json
import sys

import jax
import jax.numpy as jnp
import kept_module
import numpy as np

from dew.nn.kernels.options import owned_program
from dew.nn.sharding import measured_links
from dew.telemetry.compiles import Compiles
from dew.training.ahead import AheadPrograms, BuildTarget, NotTheKeptProgram, keeping, kept

mode, path, scale, offset = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
values = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
kept_module.OFFSET = offset
kept_module.Factor.factor = offset
scaled = owned_program(kept_module.scaled, static_argnums=(0,))
shifted = owned_program(kept_module.shifted)
helped = owned_program(kept_module.helped)
nested = owned_program(kept_module.nested)
applied = owned_program(kept_module.applied, static_argnums=(0,))
chosen = owned_program(kept_module.chosen, static_argnums=(0,))
# Two functions of one line and one code but for a constant.
one, two = owned_program(lambda held: held + 1), owned_program(lambda held: held + 2)


def answered() -> dict:
    """Each program's answer: over a static dataclass, a global its module holds, the same global read by
    a function it calls and by code nested in it, a class attribute a static argument's method reads, a
    static that is None and one that is an empty tuple, and the two functions of one line."""
    answers = {"scaled": scaled(kept_module.Scale(scale), values), "shifted": shifted(values),
               "helped": helped(values), "nested": nested(values),
               "applied": applied(kept_module.Factor(), values),
               "none": chosen(None, values), "empty": chosen((), values), "one": one(values),
               "two": two(values)}
    return {name: np.asarray(answer).tolist() for name, answer in answers.items()}


if mode == "differences":
    programs = AheadPrograms.load(path)
    assert programs.process is not None
    here = BuildTarget.of(jax.devices())
    found = {"target": [line.split()[0] for line in programs.target.differences(here)],
             "process": sorted(programs.process.differences())}
elif mode == "keep":
    with keeping() as programs:
        answers = answered()
        # A function that reads an array outside its arguments has no identity a file can state, and
        # neither has a call inside one of Dew's own contexts.
        def outside():
            return values * 2

        owned_program(outside)()
        with measured_links({}):
            scaled(kept_module.Scale(scale + 100), values)
    saved = programs.save(path)
    # The functions of the calls that have no identity, by the names a refusal states them by.
    unidentified = sorted(f"{function.__module__}.{function.__qualname__}"
                          for function in (outside, kept_module.scaled))
    found = {"answers": answers, "kept": len(saved.kept), "refused": sorted(programs.refused),
             "reasons": programs.refused, "cpu": saved.target.cpu, "unidentified": unidentified}
else:
    try:
        with kept(path, verify=mode == "verify"), Compiles(None) as compiles:
            answers = answered()
        ours = {"scaled", "shifted", "helped", "nested", "applied", "chosen", "<lambda>"}
        found = {"answers": answers,
                 "needed": sorted({program.name.removeprefix("jit(").removesuffix(")")
                                   for program in compiles.programs} & ours)}
    except NotTheKeptProgram as refusal:
        found = {"refused": type(refusal).__name__}
print(json.dumps(found))
