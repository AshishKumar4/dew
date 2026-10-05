#!/usr/bin/env python3
"""Compare local NVFP4 input QDQ with its decoded weight-only path on the RTX 4080.

The source is tools/nvfp4_qdq_reference.py's tiny two-layer checkpoint: no
32B model fits decoded on this 16GB device. Both paths share identical
decoded variables, compute dtype and Dense precision. Report compile time
separately from 30 warmed SGD steps and ten public generate calls of 16
decode tokens per row. No native FP4 matmul is used on this Ada GPU.

    PYTHONPATH=src ~/.cache/dew/dew-gpu-run --short \
        /home/mrwhite0racle/Desktop/dew/.venv/bin/python tools/benchmark_nvfp4_storage.py
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.interop import Pretrained
from dew.nn.backbones import CausalTransformer
from dew.registry import from_record
from dew.sampling.text import Sampling, generate
from dew.training.quantization import NVFP4Input, nvfp4_input_qdq

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "codecs" / "nvfp4_qdq"


def measured(model, variables, ids) -> dict:
    def loss(parameters):
        logits = model.apply({**variables, "params": parameters}, ids[:, :-1], train=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:]).mean()

    @jax.jit
    def step(parameters):
        value, grads = jax.value_and_grad(loss)(parameters)
        return jax.tree.map(lambda p, g: p - 1e-3 * g, parameters, grads), value

    start = time.perf_counter()
    compiled = step.lower(variables["params"]).compile()
    compile_step = time.perf_counter() - start
    for _ in range(3):
        jax.block_until_ready(compiled(variables["params"]))
    start = time.perf_counter()
    params = variables["params"]
    for _ in range(30):
        params, value = compiled(params)
    jax.block_until_ready((params, value))
    step_seconds = (time.perf_counter() - start) / 30
    sampling = Sampling(temperature=0)
    start = time.perf_counter()
    output = generate(model, variables, ids[:, :8], 16, key=0, sampling=sampling)
    jax.block_until_ready(output)
    compile_serving = time.perf_counter() - start
    for _ in range(3):
        jax.block_until_ready(generate(model, variables, ids[:, :8], 16, key=0, sampling=sampling))
    start = time.perf_counter()
    for _ in range(10):
        jax.block_until_ready(generate(model, variables, ids[:, :8], 16, key=0, sampling=sampling))
    seconds = (time.perf_counter() - start) / 10
    return {"train_compile_s": compile_step, "step_ms": 1000 * step_seconds,
            "serving_compile_and_first_s": compile_serving,
            "serving_ms": 1000 * seconds, "decode_tokens_per_s": ids.shape[0] * 16 / seconds,
            "final_loss": float(value)}


def main() -> None:
    # PyTorch's CPU fp32 reference uses full fp32 products. Both benchmark
    # paths keep this same precision so TF32 cannot move QDQ boundaries.
    jax.config.update("jax_default_matmul_precision", "highest")
    report = {"device": str(jax.devices()[0]), "jax": jax.__version__, "dtype": "float32", "cases": {}}
    for kind in ("unrounded", "e4m3"):
        directory = FIXTURE / kind
        loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
        with np.load(directory / "reference.npz") as reference:
            ids = jnp.asarray(reference["ids"], jnp.int32)
            checks = []
            bf16_mismatches = 0
            for key in reference.files:
                if not key.endswith("/inputs"):
                    continue
                stem = key.removesuffix("/inputs")
                spec = NVFP4Input(float(reference[stem + "/global"].reshape(())), kind == "e4m3")
                qdq = jax.jit(lambda x, spec=spec: nvfp4_input_qdq(x, spec))(jnp.asarray(reference[key]))
                checks.append(float(np.max(np.abs(np.asarray(qdq) - reference[stem + "/qdq"]))))
                bf16 = jax.jit(lambda x, spec=spec: nvfp4_input_qdq(x, spec))(
                    jnp.asarray(reference[key], jnp.bfloat16))
                bf16_mismatches += int(np.count_nonzero(
                    np.asarray(bf16, np.float32).view(np.uint32)
                    != reference[stem + "/qdq_bf16"].view(np.uint32)))
                mask = np.asarray(bf16, np.float32) != reference[stem + "/qdq_bf16"]
                if np.any(mask):
                    index = tuple(np.argwhere(mask)[0])
                    print(kind, stem, index, "input", reference[key][index], "gpu",
                          np.asarray(bf16, np.float32)[index], "cpu", reference[stem + "/qdq_bf16"][index],
                          "count", int(mask.sum()), flush=True)
            logits = np.asarray(jax.jit(loaded.model.apply)(loaded.variables, ids))
            error = float(np.sqrt(np.mean((logits - reference["logits_f64"]) ** 2)))
            reference_error = float(np.sqrt(np.mean((reference["logits"] - reference["logits_f64"]) ** 2)))
            if error > 2 * reference_error or bf16_mismatches:
                raise AssertionError((kind, error / reference_error, bf16_mismatches))
        dense = from_record(CausalTransformer, dict(loaded.model_config))
        report["cases"][kind] = {
            "max_qdq_difference": max(checks), "logits_RMS_ratio": error / reference_error,
            "bf16_qdq_bit_mismatches": bf16_mismatches,
            "qdq": measured(loaded.model, loaded.variables, ids),
            "weight_only": measured(dense, loaded.variables, ids),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
