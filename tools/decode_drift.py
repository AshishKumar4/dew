"""Dew's long greedy decode held to float64 truth within what its own reference path errs and wavers by,
after vLLM's Kimi K3 work: error a per-kernel test with synthetic inputs cannot see builds up over
thousands of decode steps.

    PYTHONPATH=src python tools/decode_drift.py --model DIR [--steps 1536] [--prompt 64] --out drift.json

1. Dew's server, as it serves (`dew.pipeline`, bf16 weights), decodes one prompt greedily for `--steps`
   tokens at batch 1: the sequence T and each token's log-probability. It decodes the same prompt again
   in a batch of 4 beside three others; the record says where the two greedy paths part.
2. The reference path (attention_impl="reference", and `dew.nn.kernels.generation.chosen` naming XLA for
   the RMS norm, SwiGLU and the gated delta rule) scores T teacher-forced through its cached decode
   (`dew.sampling.text.prefill_state`, `decode_ops`): at batch 1; at batch 4, T in row 0; and with every
   embedding entry one ulp up (`jnp.nextafter`). The two variants measured against batch 1 are its noise:
   what the reference moves by under a change no answer should depend on.
3. The truth: T scored by one float64 forward of the model over the prompt and T. The reference's and the
   server's bf16 prefills of the same are recorded beside it: decode against prefill.

The gate: the server's per-step distance from the truth, in RMS over the decode and over every 64-step
block, within the reference's own distance plus its noise over the same steps (`dew.accuracy.within`: a
NaN or infinite anything fails). `passes` in the record, and the exit code. Qwen3.5's gated delta rule
decodes through its fused step whatever `chosen` names (dew.nn.linear), so for it the reference shares the
server's delta step until the decode honours the choice; the record says so (`reference_delta_rule`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import dew
from dew.accuracy import rms, within
from dew.inference.serving import Server
from dew.interop import Pretrained
from dew.nn.inputs import ModelInputs
from dew.nn.kernels.generation import chosen
from dew.nn.kv_cache import KVCache
from dew.objectives.base import token_log_probs
from dew.sampling import Sampling
from dew.sampling.text import decode_ops, prefill_state

REFERENCE_KERNELS = {"rms_norm": "xla", "swiglu": "xla", "gated_delta_rule": "xla"}
BATCH = 4
DELTA_RULE_FAMILIES = ("qwen3_next", "qwen3_5_text", "qwen3_5_moe_text")
"""The families whose layers decode a gated delta rule (dew.interop.families.qwen)."""
BLOCK = 64
"""Steps a block of the gate covers, so error gathered in one stretch of the decode is not averaged away."""


def greedy(task):
    """`task` decoding greedily with no end: no EOS, no source logits transform or stopping."""
    return dataclasses.replace(task, sampling=Sampling(temperature=0, eos_token_ids=None), logits=None,
                               stopping=(), strategy=None, n=1)


def capacity_for(prompt: int, steps: int) -> int:
    return -(-(prompt + steps) // 64) * 64


def served(task, rows: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Row 0's greedy tokens and their log-probabilities, `rows` served together by Dew's server."""
    server = Server.from_task(task, slots=len(rows), capacity=capacity_for(rows.shape[1], steps),
                                  admission=len(rows), kv_cache=KVCache(page_size=None), chunk=None,
                                  prefix_cache=False, decode_steps=1)
    tickets = [server.submit(row, max_new_tokens=steps, key=0) for row in rows]
    server.run()
    first = tickets[0].result().host()
    assert int(first.lengths[0]) == steps, (first.lengths, steps)
    tokens = first.tokens[0, first.prompt_width:first.prompt_width + steps]
    return np.asarray(tokens), np.asarray(first.raw_log_probs[0, :steps], np.float64)


def replayed(model, variables, pad: int, rows: np.ndarray, forced: np.ndarray) -> np.ndarray:
    """Row 0's log-probability of each of `forced`'s tokens, every row fed `forced` through the cached
    decode one token at a time (teacher forcing)."""
    model = model.with_cache_capacity(capacity_for(rows.shape[1], len(forced)))
    ops = decode_ops(model, variables, pad, depths=None)
    state, active = prefill_state(model, variables, ModelInputs(jnp.asarray(rows, jnp.int32)), ops)

    @jax.jit
    def step(state, token):
        score = token_log_probs(state.logits[:1], jnp.full((1,), token, jnp.int32))[0]
        return score, ops.advance(state, jnp.full((len(rows),), token, jnp.int32), active)

    scores = []
    for token in forced:
        score, state = step(state, int(token))
        scores.append(score)
    return np.asarray(jax.device_get(jnp.stack(scores)), np.float64)


def prefilled(model, variables, prompt: np.ndarray, forced: np.ndarray) -> np.ndarray:
    """Each of `forced`'s tokens scored by one forward over the prompt and `forced`, in the model's dtype."""
    sequence = jnp.asarray(np.concatenate([prompt, forced])[None, :-1], jnp.int32)
    logits = model.apply(variables, sequence, train=False, decode=False)[0, len(prompt) - 1:]
    return np.asarray(token_log_probs(logits, jnp.asarray(forced, jnp.int32)), np.float64)


def truth(model_dir: str, prompt: np.ndarray, forced: np.ndarray) -> np.ndarray:
    """`forced`'s tokens scored by the model in float64: its float32 load, every module and weight widened
    (as tests/reference_error.py's float64_forward widens one), and the log partition in float64."""
    loaded = Pretrained.load(model_dir, dtype=jnp.float32, param_dtype=jnp.float32,
                             attention_impl="reference")
    model = loaded.model
    with jax.enable_x64(new_val=True):
        inner = {"language_model": model.language_model.clone(dtype=jnp.float64)} \
            if hasattr(model, "language_model") else {}
        wide = model.clone(dtype=jnp.float64, **inner)
        variables = jax.tree.map(lambda leaf: np.asarray(
            leaf, np.float64 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf.dtype), loaded.variables)
        sequence = jnp.asarray(np.concatenate([prompt, forced])[None, :-1], jnp.int32)
        logits = wide.apply(variables, sequence, train=False, decode=False)[0, len(prompt) - 1:]
        scores = jax.nn.log_softmax(logits.astype(jnp.float64), axis=-1)
        chosen_scores = jnp.take_along_axis(scores, jnp.asarray(forced)[:, None], axis=-1)[:, 0]
        return np.asarray(chosen_scores, np.float64)


def perturbed(variables):
    """`variables` with every entry of the token embedding one ulp up, in its own dtype."""
    def moved(path, leaf):
        named = [getattr(key, "key", None) for key in path]
        if named[-2:] == ["embed_tokens", "embedding"]:
            return jnp.nextafter(leaf, jnp.full_like(leaf, jnp.inf))
        return leaf
    return jax.tree_util.tree_map_with_path(moved, variables)


def drift(first: np.ndarray, second: np.ndarray) -> dict:
    gap = np.abs(first - second)
    return {"rms": rms(gap), "max": float(gap.max()), "worst_step": int(gap.argmax())}


def gate(server: np.ndarray, reference: np.ndarray, noise: np.ndarray, true: np.ndarray) -> dict:
    """The server's distance from the truth against the reference's plus its noise, over the whole decode
    and each `BLOCK` steps of it (the module's docstring)."""
    blocks = [slice(start, start + BLOCK) for start in range(0, len(true), BLOCK)]
    def measured(steps: slice) -> dict:
        mine, theirs = rms(server[steps] - true[steps]), rms(reference[steps] - true[steps])
        return {"server": mine, "reference": theirs, "noise": rms(noise[steps]),
                "passes": within(mine, theirs + rms(noise[steps]))}
    whole, parts = measured(slice(None)), [measured(steps) for steps in blocks]
    failed = [index for index, part in enumerate(parts) if not part["passes"]]
    return {"whole": whole, "blocks_failed": failed, "passes": whole["passes"] and not failed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=1536)
    parser.add_argument("--prompt", type=int, default=64)
    parser.add_argument("--vocab-limit", type=int, default=30_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = np.random.default_rng(0).integers(0, args.vocab_limit, (BATCH, args.prompt), dtype=np.int32)

    task = greedy(dew.pipeline(args.model, dtype="bfloat16", param_dtype="bfloat16"))
    tokens, optimized = served(task, rows[:1], args.steps)
    together, _ = served(task, rows, args.steps)
    parted = np.flatnonzero(together != tokens)

    loaded = Pretrained.load(args.model, dtype="bfloat16", param_dtype="bfloat16",
                             attention_impl="reference")
    reference = greedy(loaded.text_generation())
    pad = task.sampling.pad
    with chosen(**REFERENCE_KERNELS):
        alone = replayed(reference.model, reference.variables, pad, rows[:1], tokens)
        batched = replayed(reference.model, reference.variables, pad, rows, tokens)
        moved = replayed(reference.model, perturbed(reference.variables), pad, rows[:1], tokens)
        reference_prefill = prefilled(reference.model, reference.variables, rows[0], tokens)
    served_prefill = prefilled(task.model, task.variables, rows[0], tokens)
    true = truth(args.model, rows[0], tokens)
    noise = np.maximum(np.abs(alone - batched), np.abs(alone - moved))
    verdict = gate(optimized, alone, noise, true)
    config = json.loads((Path(args.model) / "config.json").read_text())
    delta_rule = config.get("text_config", config)["model_type"] in DELTA_RULE_FAMILIES
    record = {
        "model": args.model, "steps": args.steps, "prompt": args.prompt,
        "device": jax.devices()[0].device_kind, "gate": verdict, "passes": verdict["passes"],
        "greedy_paths_part_at": int(parted[0]) if len(parted) else None,
        "reference_batch_4": drift(alone, batched), "reference_embedding_ulp": drift(alone, moved),
        "server_from_reference": drift(optimized, alone),
        "reference_decode_from_prefill": drift(alone, reference_prefill),
        "server_decode_from_its_prefill": drift(optimized, served_prefill),
        "reference_delta_rule": "the fused decode step (it ignores chosen)" if delta_rule else None,
        "steps_scored": {"tokens": tokens.tolist(), "server": optimized.tolist(), "reference": alone.tolist(),
                         "reference_batch_4": batched.tolist(), "reference_embedding_ulp": moved.tolist(),
                         "truth": true.tolist()}}
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({key: record[key] for key in ("gate", "greedy_paths_part_at", "passes")}))
    return 0 if record["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
