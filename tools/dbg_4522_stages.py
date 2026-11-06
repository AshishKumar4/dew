"""Compare model logits and categorical terms at the local window boundary."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import lane_environment

os.environ.update(lane_environment.configured(os.environ))
import jax
import jax.numpy as jnp
import numpy as np
import test_chunk_adaptation as t
from dew.nn.categorical import label_terms, partition
from reference_error import widened
from arm_arithmetic import summed_as_run, unrewritten


def stages(variables, hidden, targets, weights, ids):
    def one(variables, hidden, targets, weights):
        passed = t.MODEL.apply(variables, hidden[None], method="continuum_states")
        logits = t.MODEL.apply(variables, passed, method="head_logits")[0]
        peak, mass, shifted = partition(logits)
        losses = label_terms(logits, targets)[0]
        return passed, logits, peak, shifted, mass, losses, jnp.sum(weights * losses)
    return jax.vmap(one)(variables, hidden, targets, weights)


programs = {mode: jax.jit(stages, compiler_options=(
    {} if mode == "on" else {"xla_cpu_experimental_ynn_fusion_type": ""}))
    for mode in ("on", "off")}
names = ("states", "logits", "peak", "shifted", "mass", "losses", "weighed")
def fixed_terms(logits, targets, weights):
    peak, mass, shifted = partition(logits)
    losses = label_terms(logits, targets)[0]
    return peak, shifted, mass, losses, jnp.sum(weights * losses, axis=-1)


fixed = {mode: jax.jit(fixed_terms, compiler_options=(
    {} if mode == "on" else {"xla_cpu_experimental_ynn_fusion_type": ""}))
    for mode in ("on", "off")}
fixed_differences = dict.fromkeys(names[2:], 0)
records = {name: {"different": 0, "total": 0, "on_error": 0.0, "off_error": 0.0} for name in names}
for seed in range(64):
    inputs = t.window_segments("documents", seed)
    with summed_as_run():
        outputs = {mode: jax.device_get(program(*inputs)) for mode, program in programs.items()}
    with jax.enable_x64(True):
        truth = jax.device_get(unrewritten(jax.jit(stages))(*widened(inputs)))
    frozen = outputs["off"][1]
    on_terms, off_terms = (jax.device_get(fixed[mode](frozen, inputs[2], inputs[3]))
                          for mode in ("on", "off"))
    for name, on, off in zip(names[2:], on_terms, off_terms):
        fixed_differences[name] += int(np.sum(on != off))
    for name, on, off, exact in zip(names, outputs["on"], outputs["off"], truth):
        record = records[name]
        record["different"] += int(np.sum(on != off))
        record["total"] += int(on.size)
        record["on_error"] += float(np.sum((on - exact) ** 2))
        record["off_error"] += float(np.sum((off - exact) ** 2))
        if name == "mass" and seed < 3:
            different = np.argwhere(on != off)
            for index in different[:3]:
                index = tuple(index)
                print("MASS", seed, index, float(on[index]), float(off[index]), float(exact[index]),
                      flush=True)
for name, record in records.items():
    print(json.dumps({"stage": name, **record}), flush=True)
print("FIXED_LOGITS_DIFFERENCES", fixed_differences, flush=True)
