"""Separate the local window's loss rule from its batched head."""
import json
import os
import sys
import functools
import contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import lane_environment

os.environ.update(lane_environment.configured(os.environ))

import jax
import jax.numpy as jnp
import numpy as np

import test_chunk_adaptation as t
from dew.nn.categorical import label_terms
from dew.nn.precision import compensated_sum, float32_einsum, float64_sum
from reference_error import with_gradients, widened
from arm_arithmetic import summed_as_run, unrewritten

from dew.objectives.lm.adaptation import _Chain
if os.environ.get("POLICY_HIGHEST"):
    import arm_arithmetic
    import dew.nn.precision
    arm_arithmetic.FP32_SUMS = "highest"
    dew.nn.precision.FP32_SUMS = "highest"
if os.environ.get("POLICY_USER_DEFAULT"):
    os.environ["DEW_WIDE_CPU_ARITHMETIC"] = "0"
    jax.config.update("jax_default_matmul_precision", None)
    summed_as_run = contextlib.nullcontext

_original_scored = _Chain.scored


def token_scored(self, versions, chosen, states, targets, weights):
    def score(slot):
        at, state, target, weight = slot
        return _original_scored(self, versions, [which[:, None] for which in at], state[:, None],
                                target[:, None], weight[:, None])
    slots = (tuple(which.swapaxes(0, 1) for which in chosen), states.swapaxes(0, 1),
             targets.swapaxes(0, 1), weights.swapaxes(0, 1))
    result = jax.lax.map(score, slots)
    return jax.tree.map(lambda value: jnp.squeeze(value, axis=2).swapaxes(0, 1), result)


def direct_loss(logits, targets, variant):
    if variant == "batch-compensated":
        peak = jax.lax.stop_gradient(jnp.max(logits, axis=-1))
        mass = compensated_sum(jnp.exp(logits - peak[:, None]), (logits.ndim - 1,))
        own = jnp.take_along_axis(logits, targets[:, None], axis=-1)[:, 0]
        return jnp.log(mass) - (own - peak)
    if variant == "batch-wide":
        with jax.enable_x64(True):
            return label_terms(logits.astype(jnp.float64), targets)[0].astype(logits.dtype)
    if variant == "batch-log1p":
        peak = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
        label = jax.nn.one_hot(targets, logits.shape[-1], dtype=bool)
        rest = jnp.sum(jnp.where(label, 0, jnp.exp(logits - peak)), axis=-1)
        own = jnp.take_along_axis(logits, targets[:, None], axis=-1)[:, 0]
        return jnp.where(own == peak[:, 0], jnp.log1p(rest), label_terms(logits, targets)[0])
    return label_terms(logits, targets)[0]


def make(kind, variant):
    def one(variables, hidden, targets, weights, ids):
        def loss(variables, hidden):
            if variant in ("actual", "equation", "actual-token"):
                fields = {} if kind == "documents" else {"segment_ids": ids[None]}
                adaptation = t.ChunkAdaptation(window=3, rate=0.5)
                _Chain.scored = token_scored if variant == "actual-token" else _original_scored
                scores = (t.adapted_scores(t.MODEL, variables, hidden[None], targets[None],
                                          weights[None], fields, adaptation, predict=False,
                                          logits=False).losses if variant != "equation" else
                          t.equation_71(t.MODEL, variables, hidden[None], targets[None], weights[None],
                                        fields, adaptation, truth=False))
                _Chain.scored = _original_scored
                return jnp.sum(weights * scores[0])
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
                if variant == "batch-high-silu" or variant.startswith("manual"):
                    def dot(spec, lhs, rhs, site):
                        if variant == "manual-wide-" + site:
                            return float64_sum(functools.partial(jnp.einsum, spec), lhs, rhs,
                                               dtype=lhs.dtype)
                        if variant == "manual-pad16" and site != "head":
                            columns = rhs.shape[-1]
                            rhs = jnp.pad(rhs, ((0, 0), (0, max(16 - columns, 0))))
                            return float32_einsum(spec, lhs, rhs)[..., :columns]
                        result = float32_einsum(spec, lhs, rhs)
                        return (jax.lax.optimization_barrier(result)
                                if variant == "manual-barrier-" + site else result)
                    level = variables["params"]["continuum"]
                    fed = dot("td,df->tf", hidden, level["down"]["kernel"], "down")
                    fed = (fed * jax.lax.logistic(fed, accuracy=jax.lax.AccuracyMode.HIGHEST)
                           if variant == "batch-high-silu" else jax.nn.silu(fed))
                    if variant == "manual-barrier-silu":
                        fed = jax.lax.optimization_barrier(fed)
                    passed = hidden + dot("tf,fd->td", fed, level["up"]["kernel"], "up")
                    logits = dot("td,vd->tv", passed, variables["params"]["embed_tokens"]["embedding"],
                                 "head")
                else:
                    logits = t.MODEL.apply(
                        variables, t.MODEL.apply(variables, hidden[None], method="continuum_states"),
                        method="head_logits")[0]
                losses = (direct_loss(logits, targets, variant) if variant != "batch-logsoftmax" else
                          -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[:, None],
                                              axis=-1)[:, 0])
            return jnp.sum(weights * losses)
        return with_gradients(loss)(variables, hidden, jnp.ones((), hidden.dtype))
    if os.environ.get("EXACT_LAYOUT"):
        return lambda *drawn: jax.lax.map(lambda example: jax.vmap(one)(*example), drawn)
    return jax.vmap(one)


kind = sys.argv[1] if len(sys.argv) > 1 else "documents"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 256
variants = ("literal", "token-categorical", "batch-categorical", "batch-logsoftmax", "batch-wide",
            "batch-log1p", "batch-compensated", "batch-high-silu", "actual", "equation")
if os.environ.get("DOT_SITES"):
    variants = ("literal", "batch-categorical", "batch-high-silu", "manual", "manual-wide-down",
                "manual-wide-up", "manual-wide-head")
if os.environ.get("BARRIERS"):
    variants = ("literal", "batch-categorical", "manual", "manual-barrier-down",
                "manual-barrier-up", "manual-barrier-head", "manual-barrier-silu")
if os.environ.get("TOKEN_PROOF"):
    variants = ("literal", "actual", "actual-token", "token-categorical", "equation")
if os.environ.get("PAD_PROOF"):
    variants = ("literal", "actual", "manual", "manual-pad16")
programs = {(variant, mode): jax.jit(make(kind, variant), compiler_options=(
    {} if mode == "on" else {"xla_cpu_experimental_ynn_fusion_type": ""}))
    for variant in variants for mode in ("on", "off")}
programs = {name: unrewritten(program) if name[0] in ("literal", "equation") else program
            for name, program in programs.items()}
truth_program = unrewritten(jax.jit(make(kind, "truth")))
totals = {}
for seed in range(count):
    inputs = (t.stacked_windows(kind, seed) if os.environ.get("EXACT_LAYOUT") else
              t.window_segments(kind, seed))
    with jax.enable_x64(True):
        truth = jax.device_get(truth_program(*widened(inputs)))
    with summed_as_run():
        outputs = {name: jax.device_get(program(*inputs)) for name, program in programs.items()}
    paths, _ = jax.tree.flatten_with_path(truth)
    for name, output in outputs.items():
        for (path, exact), actual in zip(paths, jax.tree.leaves(output)):
            key = ("/".join(str(getattr(part, "key", getattr(part, "idx", part))) for part in path),
                   *name)
            exact = np.asarray(exact)
            actual = np.asarray(actual)
            if exact.ndim < 2:
                if key[0] == "0":
                    errors = np.sqrt(np.mean(np.square(actual.astype(np.float64) - exact)))
                    totals.setdefault(key, []).append(float(errors))
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
            with summed_as_run():
                text = program.lower(*inputs).compile().as_text()
            print("HLO", name, "ynn", text.lower().count("ynn"),
                  "bf16", text.count("bf16"), "f64", text.count("f64"), flush=True)
            Path("/tmp/4522-" + "-".join(name) + ".hlo").write_text(text)
for key, errors in totals.items():
    path, variant, mode = key
    if path != "0":
        continue
    base = np.asarray(totals[path, "literal", mode])
    errors = np.asarray(errors)
    if not np.any(base):
        continue
    print(json.dumps({"leaf": path, "variant": variant, "mode": mode,
                      "above4": int(np.sum(errors > 4 * base)),
                      "below4": int(np.sum(base > 4 * errors)),
                      "rms_ratio": float(np.sqrt(np.mean(errors ** 2) / np.mean(base ** 2))),
                      "max": float(np.max(errors))}), flush=True)
    if path == "0" and variant != "literal" and mode == "on":
        off = np.asarray(totals[path, variant, "off"])
        off_base = np.asarray(totals[path, "literal", "off"])
        red = np.flatnonzero((errors > 4 * base) & (off <= 4 * off_base))
        print("RED_SEEDS", variant, [(int(seed), float(errors[seed]), float(base[seed]),
                                     float(off[seed]), float(off_base[seed])) for seed in red[:10]],
              flush=True)
