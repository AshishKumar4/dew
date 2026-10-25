"""Dew's long greedy decode held to float64 truth as the gate holds a port (check 1,
tests/reference_error.py), after vLLM's Kimi K3 work: error a per-kernel test with synthetic inputs
cannot see builds up over thousands of decode steps.

    PYTHONPATH=src python tools/decode_drift.py --model DIR [--steps 1536] [--prompt 64] [--seeds 8] \\
        --out drift.json

Each seed draws a prompt (`prompts`). Dew's server, as it serves (`dew.pipeline`, bf16 weights), decodes
it greedily for `--steps` tokens at batch 1: the sequence T and each token's log-probability. The
reference path (attention_impl="reference", and `dew.nn.kernels.generation.chosen` naming XLA for the RMS
norm, SwiGLU and the gated delta rule) scores T teacher-forced through its cached decode
(`dew.sampling.text.prefill_state`, `decode_ops`). The truth: T scored by one float64 forward of the model
over the prompt and T. Check 1 (`assert_as_exact`) holds the server's log-probabilities to the reference's
from that truth over the declared seeds, each `BLOCK` steps a row, so error gathered in one stretch of the
decode is held by the worst row, not averaged away. `--seeds` is the model's declared count, sized by
`seeds_for`; the exit code is the verdict's.

Beside the verdict, the record reports the first seed's prompt decoded again in a batch of `BATCH` beside
others (where the greedy paths part), the reference rescored at that batch and with its embedding one ulp
up (what it wavers by under a change no answer should depend on), and both paths' prefills of the same
tokens: decode against prefill. Qwen3.5's gated delta rule decodes through its fused step whatever
`chosen` names (dew.nn.linear), so for it the reference shares the server's delta step until the decode
honours the choice; the record says so (`reference_delta_rule`).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import dew
from dew.accuracy import rms
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
"""Steps a row of check 1 covers: `--steps` is whole blocks."""


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


def replayer(model, variables, pad: int, batch: int, prompt: int, steps: int):
    """A function of `(rows, forced)` returning row 0's log-probability of each of `forced`'s tokens, the
    `batch` rows of `prompt` ids fed `forced` through the cached decode one token at a time (teacher
    forcing): its single-token step compiled once for every call."""
    model = model.with_cache_capacity(capacity_for(prompt, steps))
    ops = decode_ops(model, variables, pad, depths=None)

    @jax.jit
    def step(state, active, token):
        score = token_log_probs(state.logits[:1], jnp.full((1,), token, jnp.int32))[0]
        return score, ops.advance(state, jnp.full((batch,), token, jnp.int32), active)

    def replay(rows: np.ndarray, forced: np.ndarray) -> np.ndarray:
        state, active = prefill_state(model, variables, ModelInputs(jnp.asarray(rows, jnp.int32)), ops)
        scores = []
        for token in forced:
            score, state = step(state, active, int(token))
            scores.append(score)
        return np.asarray(jax.device_get(jnp.stack(scores)), np.float64)
    return replay


def prefilled(model, variables, prompt: np.ndarray, forced: np.ndarray) -> np.ndarray:
    """Each of `forced`'s tokens scored by one forward over the prompt and `forced`, in the model's dtype."""
    sequence = jnp.asarray(np.concatenate([prompt, forced])[None, :-1], jnp.int32)
    logits = model.apply(variables, sequence, train=False, decode=False)[0, len(prompt) - 1:]
    return np.asarray(token_log_probs(logits, jnp.asarray(forced, jnp.int32)), np.float64)


def truth(model_dir: str):
    """A function scoring a prompt's `forced` tokens by the model in float64: its float32 load, every
    module and weight widened (as tests/reference_error.py's float64_forward widens one), and the log
    partition in float64."""
    loaded = Pretrained.load(model_dir, dtype=jnp.float32, param_dtype=jnp.float32,
                             attention_impl="reference")
    model = loaded.model
    with jax.enable_x64(new_val=True):
        inner = {"language_model": model.language_model.clone(dtype=jnp.float64)} \
            if hasattr(model, "language_model") else {}
        wide = model.clone(dtype=jnp.float64, **inner)
        variables = jax.tree.map(lambda leaf: np.asarray(
            leaf, np.float64 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf.dtype), loaded.variables)

    def scored(prompt: np.ndarray, forced: np.ndarray) -> np.ndarray:
        with jax.enable_x64(new_val=True):
            sequence = jnp.asarray(np.concatenate([prompt, forced])[None, :-1], jnp.int32)
            logits = wide.apply(variables, sequence, train=False, decode=False)[0, len(prompt) - 1:]
            scores = jax.nn.log_softmax(logits.astype(jnp.float64), axis=-1)
            chosen_scores = jnp.take_along_axis(scores, jnp.asarray(forced)[:, None], axis=-1)[:, 0]
            return np.asarray(chosen_scores, np.float64)
    return scored


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


def prompts(vocab_limit: int, length: int, seed: int) -> np.ndarray:
    """A seed's `BATCH` prompts of `length` token ids below `vocab_limit`: its first is the one decoded
    alone, the rest the batch beside it."""
    return np.random.default_rng([0xD0DE, seed]).integers(0, vocab_limit, (BATCH, length), dtype=np.int32)


def main() -> int:
    from reference_error import assert_as_exact

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=1536)
    parser.add_argument("--prompt", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=8, help="the model's declared draws (`seeds_for`)")
    parser.add_argument("--vocab-limit", type=int, default=30_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps % BLOCK:
        parser.error(f"--steps is whole blocks of {BLOCK}")

    task = greedy(dew.pipeline(args.model, dtype="bfloat16", param_dtype="bfloat16"))
    loaded = Pretrained.load(args.model, dtype="bfloat16", param_dtype="bfloat16",
                             attention_impl="reference")
    reference = greedy(loaded.text_generation())
    pad = task.sampling.pad
    true = truth(args.model)
    with chosen(**REFERENCE_KERNELS):
        alone_of = replayer(reference.model, reference.variables, pad, 1, args.prompt, args.steps)
        batched_of = replayer(reference.model, reference.variables, pad, BATCH, args.prompt, args.steps)
        moved_of = replayer(reference.model, perturbed(reference.variables), pad, 1, args.prompt, args.steps)

    @functools.cache
    def draw(seed):
        """The seed's prompts and Dew's server's greedy decode of the first alone: its tokens and their
        log-probabilities."""
        rows = prompts(args.vocab_limit, args.prompt, seed)
        return (rows, *served(task, rows[:1], args.steps))

    def blocks(scores) -> np.ndarray:
        return np.asarray(scores).reshape(-1, BLOCK)

    def replay(rows, tokens, scores):
        with chosen(**REFERENCE_KERNELS):
            return blocks(alone_of(rows[:1], tokens))

    finding = None
    try:
        assert_as_exact(lambda rows, tokens, scores: blocks(scores), replay, draw,
                        f"{args.model}'s decode", seeds=args.seeds,
                        truth_fn=lambda rows, tokens, scores: blocks(true(rows[0], tokens)))
    except AssertionError as failed:
        finding = str(failed)

    rows, tokens, optimized = draw(0)
    together, _ = served(task, rows, args.steps)
    parted = np.flatnonzero(together != tokens)
    with chosen(**REFERENCE_KERNELS):
        alone, moved = alone_of(rows[:1], tokens), moved_of(rows[:1], tokens)
        batched = batched_of(rows, tokens)
        reference_prefill = prefilled(reference.model, reference.variables, rows[0], tokens)
    served_prefill = prefilled(task.model, task.variables, rows[0], tokens)
    config = json.loads((Path(args.model) / "config.json").read_text())
    delta_rule = config.get("text_config", config)["model_type"] in DELTA_RULE_FAMILIES
    record = {
        "model": args.model, "steps": args.steps, "prompt": args.prompt, "seeds": args.seeds,
        "device": jax.devices()[0].device_kind, "passes": finding is None, "finding": finding,
        "greedy_paths_part_at": int(parted[0]) if len(parted) else None,
        "reference_batch_4": drift(alone, batched), "reference_embedding_ulp": drift(alone, moved),
        "server_from_reference": drift(optimized, alone),
        "reference_decode_from_prefill": drift(alone, reference_prefill),
        "server_decode_from_its_prefill": drift(optimized, served_prefill),
        "reference_delta_rule": "the fused decode step (it ignores chosen)" if delta_rule else None,
        "steps_scored": {"tokens": tokens.tolist(), "server": optimized.tolist(), "reference": alone.tolist(),
                         "reference_batch_4": batched.tolist(), "reference_embedding_ulp": moved.tolist(),
                         "truth": true(rows[0], tokens).tolist()}}
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({key: record[key] for key in ("passes", "finding", "greedy_paths_part_at")}))
    return 0 if record["passes"] else 1


if __name__ == "__main__":
    # The tests' reference_error, as the test suite imports it.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    sys.exit(main())
