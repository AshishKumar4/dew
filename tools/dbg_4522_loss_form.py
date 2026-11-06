"""Separate the local window's loss rule from its batched head."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import jax
import jax.numpy as jnp
import numpy as np

import test_chunk_adaptation as t
from dew.nn.categorical import label_terms
from reference_error import with_gradients, widened


def make(kind, variant):
    def one(variables, hidden, targets, weights, ids):
        def loss(variables, hidden):
            def single(state, target):
                logits = t.MODEL.apply(
                    variables, t.MODEL.apply(variables, state[None], method="continuum_states"),
                    method="head_logits")[0]
                if variant == "literal":
                    return -jax.nn.log_softmax(logits)[target]
                if variant == "truth":
                    shifted = jnp.where(jnp.arange(logits.shape[-1]) == target, 0,
                                        logits - logits[target])
                    return jax.nn.logsumexp(shifted)
                return label_terms(logits, target)[0]
            if variant in ("literal", "truth", "token-categorical"):
                losses = jax.lax.map(lambda pair: single(*pair), (hidden, targets))
            else:
                logits = t.MODEL.apply(
                    variables, t.MODEL.apply(variables, hidden[None], method="continuum_states"),
                    method="head_logits")[0]
                losses = (label_terms(logits, targets)[0] if variant == "batch-categorical" else
                          -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[:, None],
                                              axis=-1)[:, 0])
            return jnp.sum(weights * losses)
        return with_gradients(loss)(variables, hidden, jnp.ones((), hidden.dtype))
    return jax.vmap(one)


kind = sys.argv[1] if len(sys.argv) > 1 else "documents"
variants = ("literal", "token-categorical", "batch-categorical", "batch-logsoftmax")
programs = {(variant, mode): jax.jit(make(kind, variant), compiler_options=(
    {} if mode == "on" else {"xla_cpu_experimental_ynn_fusion_type": ""}))
    for variant in variants for mode in ("on", "off")}
truth_program = jax.jit(make(kind, "truth"))
totals = {}
for seed in range(16):
    inputs = t.window_segments(kind, seed)
    with jax.enable_x64(True):
        truth = jax.device_get(truth_program(*widened(inputs)))
    outputs = {name: jax.device_get(program(*inputs)) for name, program in programs.items()}
    paths, _ = jax.tree.flatten_with_path(truth)
    for name, output in outputs.items():
        for (path, exact), actual in zip(paths, jax.tree.leaves(output)):
            key = ("/".join(str(getattr(part, "key", getattr(part, "idx", part))) for part in path),
                   *name)
            exact = np.asarray(exact)
            actual = np.asarray(actual)
            if exact.ndim < 2:
                continue
            width = exact.shape[-1]
            exact = exact.reshape(len(exact), -1, width)
            error = (actual.reshape(exact.shape) - exact)
            size = np.sqrt(np.mean(exact ** 2, axis=-1))
            relative = np.sqrt(np.mean(error ** 2, axis=-1)) / np.where(size > 0, size, np.inf)
            worst = np.max(relative, axis=-1)
            totals.setdefault(key, []).extend(worst.tolist())
    if seed == 0:
        for name, program in programs.items():
            text = program.lower(*inputs).compile().as_text()
            print("HLO", name, "ynn", text.lower().count("ynn"),
                  "bf16", text.count("bf16"), flush=True)
            Path("/tmp/4522-" + "-".join(name) + ".hlo").write_text(text)
for key, errors in totals.items():
    path, variant, mode = key
    base = np.asarray(totals[path, "literal", mode])
    errors = np.asarray(errors)
    print(json.dumps({"leaf": path, "variant": variant, "mode": mode,
                      "above4": int(np.sum(errors > 4 * base)),
                      "below4": int(np.sum(base > 4 * errors)),
                      "rms_ratio": float(np.sqrt(np.mean(errors ** 2) / np.mean(base ** 2))),
                      "max": float(np.max(errors))}), flush=True)
