"""Dew's decode held to its own noise floor over a long greedy decode, as vLLM's Kimi K3 work gates its
kernels: error a per-kernel test with synthetic inputs cannot see builds up over thousands of steps.

    PYTHONPATH=src python tools/decode_drift.py --model DIR [--steps 1536] [--prompt 64] --out drift.json

1. Dew's server, as it serves (`dew.pipeline`, bf16 weights), decodes one prompt greedily for `--steps`
   tokens at batch 1: the sequence T and each token's log-probability. It decodes the same prompt again
   in a batch of 4 beside three others; the record says where the two greedy paths part.
2. The reference path (attention_impl="reference", and `dew.nn.kernels.generation.chosen` naming XLA for
   the RMS norm, SwiGLU and the gated delta rule) scores T teacher-forced through its cached decode
   (`dew.sampling.text.prefill_state`, `decode_ops`): at batch 1; at batch 4, T in row 0; and with every
   embedding entry one ulp up (`jnp.nextafter`). The first two measured beside batch 1 are the noise
   floor: what the reference itself moves by under a change no answer should depend on.
3. One forward over the prompt and T scores T again, as a prefill: decode against prefill.

Each comparison is the per-step |Δ log p| of T's tokens, its RMS and its largest. The gate holds the
server's distance from the reference (1 against 2's batch 1) within the floor, by RMS and by largest:
`passes` in the record, and the exit code. Qwen3.5's gated delta rule decodes through its fused step
whatever `chosen` names (dew.nn.linear), so its reference is reference attention with a fused delta
step until the decode honours the choice.
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
from dew.inference.serving import Server
from dew.interop import Pretrained
from dew.nn.inputs import ModelInputs
from dew.nn.kernels.generation import chosen
from dew.nn.kv_cache import KVCache
from dew.sampling import Sampling
from dew.sampling.text import decode_ops, prefill_state

REFERENCE_KERNELS = {"rms_norm": "xla", "swiglu": "xla", "gated_delta_rule": "xla"}
BATCH = 4


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
        scores = jax.nn.log_softmax(state.logits.astype(jnp.float32), axis=-1)[0, token]
        return scores, ops.advance(state, jnp.full((len(rows),), token, jnp.int32), active)

    scores = []
    for token in forced:
        score, state = step(state, int(token))
        scores.append(score)
    return np.asarray(jax.device_get(jnp.stack(scores)), np.float64)


def prefilled(model, variables, prompt: np.ndarray, forced: np.ndarray) -> np.ndarray:
    """Each of `forced`'s tokens scored by one forward over the prompt and `forced`."""
    sequence = jnp.asarray(np.concatenate([prompt, forced])[None, :-1], jnp.int32)
    logits = model.apply(variables, sequence, train=False, decode=False)
    scores = jax.nn.log_softmax(logits[0, len(prompt) - 1:].astype(jnp.float32), axis=-1)
    return np.asarray(jnp.take_along_axis(scores, jnp.asarray(forced)[:, None], axis=-1)[:, 0], np.float64)


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
    return {"rms": float(np.sqrt(np.mean(gap ** 2))), "max": float(gap.max()),
            "worst_step": int(gap.argmax())}


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
    with chosen(**REFERENCE_KERNELS):
        alone = replayed(reference.model, reference.variables, task.sampling.pad, rows[:1], tokens)
        batched = replayed(reference.model, reference.variables, task.sampling.pad, rows, tokens)
        moved = replayed(reference.model, perturbed(reference.variables), task.sampling.pad, rows[:1], tokens)
        prefill = prefilled(reference.model, reference.variables, rows[0], tokens)
    served_prefill = prefilled(task.model, task.variables, rows[0], tokens)

    floor = {key: max(drift(alone, batched)[key], drift(alone, moved)[key]) for key in ("rms", "max")}
    measured = drift(optimized, alone)
    record = {
        "model": args.model, "steps": args.steps, "prompt": args.prompt,
        "device": jax.devices()[0].device_kind,
        "greedy_paths_part_at": int(parted[0]) if len(parted) else None,
        "reference_batch_4": drift(alone, batched), "reference_embedding_ulp": drift(alone, moved),
        "floor": floor, "server_from_reference": measured,
        "reference_decode_from_prefill": drift(alone, prefill),
        "server_decode_from_its_prefill": drift(optimized, served_prefill),
        "passes": measured["rms"] <= floor["rms"] and measured["max"] <= floor["max"]}
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({key: record[key] for key in ("floor", "server_from_reference", "passes")}))
    return 0 if record["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
