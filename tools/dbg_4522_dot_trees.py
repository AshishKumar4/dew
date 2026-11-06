"""Identify fp32 reduction orders on frozen local-window dot operands."""
import hashlib
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
from arm_arithmetic import summed_as_run
from dew.nn.precision import FP32_SUMS

for seed in (0, 1):
    variables = t.drawn_variables(seed)
    print("DRAW", seed, {name: hashlib.sha256(np.asarray(leaf).tobytes()).hexdigest()
                        for name, leaf in (
                            ("down", variables["params"]["continuum"]["down"]["kernel"]),
                            ("up", variables["params"]["continuum"]["up"]["kernel"]),
                            ("head", variables["params"]["embed_tokens"]["embedding"]))}, flush=True)


def patterns(lhs, rhs):
    a, b = lhs.astype(np.float64), rhs.astype(np.float64)
    terms = a[..., :, None] * b[:, None, :, :]
    rounded = terms.astype(np.float32)
    result = {}
    for chains in (1, 2, 4, 8):
        if chains > terms.shape[-2]:
            continue
        for fused in (True, False):
            accumulators = np.zeros((*terms.shape[:-2], chains, terms.shape[-1]), np.float32)
            for at in range(terms.shape[-2]):
                term = terms[..., at, :] if fused else rounded[..., at, :].astype(np.float64)
                accumulators[..., at % chains, :] = (
                    accumulators[..., at % chains, :].astype(np.float64) + term).astype(np.float32)
            while accumulators.shape[-2] > 1:
                accumulators = (accumulators[..., 0::2, :] +
                                accumulators[..., 1::2, :]).astype(np.float32)
            result[f"{chains}-chain-{'fma' if fused else 'mul-add'}"] = accumulators[..., 0, :]
    return result


variables, hidden, _, _, _ = t.window_segments("documents", 423)
level = variables["params"]["continuum"]
fed, passed = None, None
for name, lhs, rhs in (
        ("down", hidden, level["down"]["kernel"]),
        ("up", None, level["up"]["kernel"]),
        ("head", None, variables["params"]["embed_tokens"]["embedding"].swapaxes(-1, -2))):
    lhs = hidden if name == "down" else fed if name == "up" else passed
    precision = jax.lax.Precision.HIGHEST if name == "head" else FP32_SUMS

    def batch(a, b):
        return jnp.einsum("btd,bdf->btf", a, b, precision=precision)

    def token(a, b):
        return jax.lax.map(lambda at: jnp.einsum("bd,bdf->bf", at, b, precision=precision),
                           a.swapaxes(0, 1)).swapaxes(0, 1)

    outputs = {}
    with summed_as_run():
        for geometry, fn in (("batch", batch), ("token", token)):
            for mode in ("on", "off"):
                outputs[geometry, mode] = np.asarray(jax.jit(fn, compiler_options=(
                    {} if mode == "on" else {"xla_cpu_experimental_ynn_fusion_type": ""}))(lhs, rhs))
    exact = np.einsum("btd,bdf->btf", lhs.astype(np.float64), rhs.astype(np.float64))
    alternatives = patterns(lhs, rhs)
    for (geometry, mode), output in outputs.items():
        print(json.dumps({"dot": name, "geometry": geometry, "mode": mode, "elements": output.size,
                          "squared_error": float(np.sum((output - exact) ** 2)),
                          "different_from_batch_off": int(np.sum(output != outputs["batch", "off"])),
                          "matches": {label: int(np.sum(value == output))
                                      for label, value in alternatives.items()}}), flush=True)
    if name == "down":
        fed = np.asarray(jax.nn.silu(jnp.asarray(outputs["batch", "off"])))
    elif name == "up":
        passed = hidden + outputs["batch", "off"]
