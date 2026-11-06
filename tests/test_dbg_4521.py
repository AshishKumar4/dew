"""Scratch instrumentation for the packed-stream check's CPU programs."""

import collections
import json
import os
import re

import jax
import numpy as np
import pytest
import test_decision_context as context
from arm_arithmetic import summed_as_run, unrewritten
from test_decision_streams import LETTERED, examples, in_streams


def test_diagnostic():
    packed, alone = in_streams(LETTERED, segments=3), in_streams(LETTERED, segments=1)
    dew, reference, draw, truth = context.packed_arms(packed, alone, examples(LETTERED))
    variables = draw(0)
    original = context.without_ynn
    outputs = {}
    for mode in os.environ.get("DBG_MODES", "off,on").split(","):
        context.COMPILED.clear()
        seen = set()

        def compile(fn, **options):
            compiler_options = {"xla_cpu_experimental_ynn_fusion_type": ""} if mode == "off" else {}
            if mode.startswith("fusion:"):
                compiler_options = {"xla_cpu_experimental_ynn_fusion_type": mode.split(":", 1)[1]}
            compiled = jax.jit(fn, compiler_options=compiler_options, **options)

            def run(*args):
                identity = tuple((tuple(x.shape), str(x.dtype)) for x in jax.tree.leaves(args))
                if identity not in seen:
                    seen.add(identity)
                    hlo = compiled.lower(*args).compile().as_text()
                    lines = [line for line in hlo.splitlines() if 'kind=kCustom' in line]
                    short = [re.sub(r', metadata=.*', '', line.strip()) for line in lines]
                    if os.environ.get("DBG_HLO"):
                        print("HLO", mode, [x.shape for x in jax.tree.leaves(args[1])],
                              json.dumps(short)[:6000], flush=True)
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
