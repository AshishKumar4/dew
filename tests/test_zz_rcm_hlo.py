"""scratch: the rCM unet step's conditionals and the all-reduces at op 166 and 170."""

import itertools
import re

import jax
import optax
import test_capability_matrix as matrix

from dew.objectives.diffusion.consistency import ConsistencyDistillationObjective
from dew.training import Trainer


def test_print_the_rcm_unet_steps_conditionals():
    family = next(family for family in matrix.ALL if family.name == "unet")
    student = matrix.smooth(matrix.loaded(family).model)
    inputs = family.inputs()
    objective = ConsistencyDistillationObjective(
        student, matrix.presets.Flow()(), inputs, student_update_freq=1, max_simulation_steps=2,
        variables={matrix.TEACHER: matrix.teacher_of(family, student, inputs)}, ema_decay=None)
    batch = matrix.denoised_batch(inputs)
    trainer = Trainer(objective, optax.adam(1e-3), key=0)
    data = matrix.Loader(matrix.replayed(lambda partition: itertools.repeat(batch)), None, batch=matrix.ROWS,
                         records=None)
    trainer.fit(data, steps=1, log_every=1)
    text = trainer.executable.as_text()
    lines = text.splitlines()
    names = {}
    for line in lines:
        found = re.match(r"\s*(?:ROOT )?(%?[\w.\-]+) = ", line)
        if found:
            names[found.group(1).lstrip("%")] = line.strip()
    print("CONDITIONALS")
    for line in lines:
        if " conditional(" in line:
            print("  ", line.strip()[:600])
            predicate = re.search(r"conditional\(([^,)]+)", line)
            seen = 0
            target = predicate.group(1).lstrip("%").split(" ")[-1] if predicate else None
            while target and target in names and seen < 8:
                print("      <-", names[target][:400])
                operand = re.search(r"= [^(]*\(([^,)]+)", names[target])
                target = operand.group(1).lstrip("%").split(" ")[-1] if operand else None
                seen += 1
    print("ALL-REDUCES")
    computation = None
    for line in lines:
        head = re.match(r"\s*(?:ENTRY )?%?([\w.\-]+) .*\{\s*$", line)
        if head:
            computation = head.group(1)
        if "all-reduce(" in line and re.search(r"channel_id=(166|170)\b", line):
            print("  in", computation, ":", line.strip()[:500])
    print("CHANNELS", sorted({int(x) for x in re.findall(r"channel_id=(\d+)", text)})[:400])
