"""Scratch instrumentation for the packed-stream check's CPU programs."""

import collections
import dataclasses
import json
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_decision_context as context
from arm_arithmetic import summed_as_run, unrewritten, widened
from test_decision_streams import LETTERED, examples, in_streams


def test_diagnostic():
    packed, alone = in_streams(LETTERED, segments=3), in_streams(LETTERED, segments=1)
    dew, reference, draw, truth = context.packed_arms(packed, alone, examples(LETTERED))
    variables = draw(0)
    print("DRAWS", {seed: all(np.array_equal(a, b) for a, b in
          zip(jax.tree.leaves(variables), jax.tree.leaves(draw(seed)), strict=True))
          for seed in (1, 7, 39)}, flush=True)
    original = context.without_ynn
    outputs = {}
    for mode in os.environ.get("DBG_MODES", "off,on").split(","):
        context.COMPILED.clear()
        seen = set()

        def compile(fn, **options):
            objective = fn.__closure__[0].cell_contents
            arm_name = "packed" if objective is packed else "alone" if objective is alone else "truth"
            compiler_options = {"xla_cpu_experimental_ynn_fusion_type": ""} if mode == "off" else {}
            if mode.startswith("fusion:"):
                compiler_options = {"xla_cpu_experimental_ynn_fusion_type": mode.split(":", 1)[1]}
            compiled = jax.jit(fn, compiler_options=compiler_options, **options)

            def run(*args):
                identity = (arm_name, tuple((tuple(x.shape), str(x.dtype)) for x in jax.tree.leaves(args)))
                if identity not in seen:
                    seen.add(identity)
                    hlo = compiled.lower(*args).compile().as_text()
                    lines = [line for line in hlo.splitlines() if 'kind=kCustom' in line]
                    dots = collections.Counter(
                        re.sub(r"dot\(.*?\)", "dot(operands)", line.strip().split(", metadata=")[0]
                               .split(" = ")[1])
                        for line in hlo.splitlines() if re.search(r" = f\d+\[.*\] dot\(", line))
                    operations = collections.Counter(
                        re.search(r'op_name="([^"]+)"', line).group(1).split("/")[-1]
                        if 'op_name="' in line else "unnamed-dot" for line in lines)
                    if os.environ.get("DBG_HLO") and arm_name != "truth":
                        print("HLO", mode, arm_name, json.dumps({
                            "inputs": [str(x.shape) for x in jax.tree.leaves(args[1])],
                            "fusions": dict(operations), "dots": dict(dots)}), flush=True)
                return compiled(*args)
            return run

        context.without_ynn = compile
        with summed_as_run():
            d, r = dew(*variables), unrewritten(reference)(*variables)
        t = unrewritten(truth)(*variables)
        outputs[mode] = (d, r, t)
        total = [0.0, 0.0]
        for key in d:
            ed = np.asarray(d[key], np.float64) - np.asarray(t[key])
            er = np.asarray(r[key], np.float64) - np.asarray(t[key])
            total[0] += float(np.sum(ed ** 2))
            total[1] += float(np.sum(er ** 2))
            print("ROW", mode, repr(key.split("Which")[0]), json.dumps({
                "dew_rms": float(np.sqrt(np.mean(ed ** 2))),
                "ref_rms": float(np.sqrt(np.mean(er ** 2))),
                "ratio": float(np.sqrt(np.sum(ed ** 2) / np.sum(er ** 2)))}), flush=True)
        print("TOTAL", mode, total, "ratio", np.sqrt(total[0] / total[1]), flush=True)
    context.without_ynn = original


def test_fixed_float64():
    packed, alone = in_streams(LETTERED, segments=3), in_streams(LETTERED, segments=1)
    dew, reference, _, _ = context.packed_arms(packed, alone, examples(LETTERED))
    variables = packed.init(jax.random.key(0))
    for objective in (packed, alone):
        objective.model = dataclasses.replace(
            objective.model, backbone=objective.model.backbone.clone(dtype=jnp.float64),
            head=objective.model.head.clone(dtype=jnp.float64))
    context.COMPILED.clear()
    context.without_ynn = jax.jit
    with jax.enable_x64(new_val=True):
        d, r = unrewritten(dew)(widened(variables)), unrewritten(reference)(widened(variables))
    for key in d:
        print("FLOAT64", repr(key.split("Which")[0]),
              float(np.max(np.abs(d[key] - r[key]))), flush=True)
