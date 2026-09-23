"""Routing and support replay against what vLLM actually returned.

`tools/vllm_replay_reference.py` served `tests/fixtures/hf/qwen3-moe-vllm`
in bf16 on vLLM 0.30.0 (an A100) with top-k 6, top-p 0.9 and temperature
0.8, asking for processed log-probabilities, the sampling mask and the routed experts, and
wrote every completion to `fixtures/rl/vllm_replay.json`. Dew loads the same
checkpoint, packs the completions as `Call` records and scores them with the
packed GRPO objective.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.interop.pretrained import load_pretrained
from dew.objectives.rl.grpo import GRPOObjective
from dew.objectives.rl.sessions import ROUTED_EXPERTS_KEY, ROUTED_KEY, Call, Session, Status, pack

FIXTURES = Path(__file__).parent / "fixtures"
RECORD = json.loads((FIXTURES / "rl/vllm_replay.json").read_text())
WIDTH = 24


def engine_batch():
    rollouts = []
    for number, call in enumerate(RECORD["calls"]):
        record = Call(tuple(call["prompt_ids"]), tuple(call["sampled_ids"]), tuple(call["behavior_log_probs"]),
                      "length", 0, routed_experts=np.asarray(call["routed_experts"]),
                      support=tuple(tuple(kept) for kept in call["support"]))
        rollouts.append(Session("t", "g", number, 0, (record,), Status.COMPLETED, float(number)))
    return rollouts, pack(rollouts, WIDTH, rows=len(rollouts), support_capacity=6 * WIDTH)


@pytest.fixture(scope="module")
def objective():
    pretrained = load_pretrained(str(FIXTURES / "hf/qwen3-moe-vllm"), dtype="float32", attention_impl="reference")
    grpo = GRPOObjective(pretrained.model, WIDTH - 1, sampling_temperature=RECORD["sampling"]["temperature"])
    return grpo, pretrained.variables


def test_packed_scoring_computes_the_engines_filtered_likelihoods(objective):
    """With the engine's routing replayed and each id renormalized over the
    support vLLM kept at its temperature, Dew's fp32 likelihoods are as
    close to transformers' float64 ones as transformers' own fp32 run
    (`reference_error`), and vLLM's reported ones as transformers' bf16 run
    on the record's routing: both tie the scored likelihood to the engine's. The tempered full-vocabulary likelihood misses by far more."""
    grpo, variables = objective
    _, batch = engine_batch()
    sampled = batch["response_mask"] != 0
    # Each session is one call, whose sampled ids sit in row-major order.
    order = np.argsort(batch["session_index"][sampled], kind="stable")
    reference, truth = (np.concatenate([call["reference"][name] for call in RECORD["calls"]])
                        for name in ("float32", "float64"))
    scored = np.asarray(grpo.packed_log_probs(variables, batch))[sampled][order]
    assert_as_exact_as_the_reference(scored, reference, truth, "filtered log-probs")
    # The engine leg: vLLM's bf16 behavior likelihoods are as close to the
    # truth as transformers' bf16 run on the record's routing.
    engine = np.asarray(batch["behavior_log_probs"])[sampled][order]
    bf16 = np.concatenate([call["reference"]["bfloat16"] for call in RECORD["calls"]])
    assert_as_exact_as_the_reference(engine, bf16, truth, "vLLM's behavior log-probs")
    unfiltered = {key: value for key, value in batch.items() if key not in ("support_ids", "support_columns")}
    raw = np.asarray(grpo.packed_log_probs(variables, unfiltered))[sampled][order]
    assert np.abs(raw - truth).max() > 0.5


def test_the_trainer_routes_every_token_as_the_record_says(objective):
    """The routing Dew's forward sows follows the record it replays. On
    vLLM's record Dew's own top-k already agrees, so the check also replays
    a record altered to send every covered id to the two experts Dew ranks
    last on each sparse layer: the sown routing follows it, and the
    likelihoods move."""
    grpo, variables = objective
    _, batch = engine_batch()
    ids = batch["input_ids"]
    packing = {"segment_ids": batch["text_segment_ids"], "positions": batch["text_positions"]}
    covered = batch[ROUTED_KEY][:, :-1]

    def sown(record):
        routes = None if record is None else (record, batch[ROUTED_KEY])
        return grpo.token_scores(variables, ids, routing=True, routes=routes, **packing).routing

    native = sown(None)
    altered = np.array(batch[ROUTED_EXPERTS_KEY])
    for layer in (1, 2):
        scores = np.asarray(native[f"layers_{layer}"]["mlp"]["gate"]["scores"][0])
        last = np.argsort(scores, axis=-1)[..., :2].astype(altered.dtype)
        altered[:, :-1, layer] = np.where(covered[..., None], last, altered[:, :-1, layer])
    for record in (batch[ROUTED_EXPERTS_KEY], altered):
        routing = sown(record)
        for layer in (1, 2):
            chosen = np.sort(np.asarray(routing[f"layers_{layer}"]["mlp"]["gate"]["indices"][0]), -1)
            np.testing.assert_array_equal(chosen[covered], np.sort(record[:, :-1, layer], -1)[covered])
    assert not np.array_equal(altered, batch[ROUTED_EXPERTS_KEY])
    moved = np.asarray(grpo.packed_log_probs(variables, {**batch, ROUTED_EXPERTS_KEY: altered}))
    kept = np.asarray(grpo.packed_log_probs(variables, batch))
    sampled = batch["response_mask"] != 0
    assert np.abs(moved - kept)[sampled].max() > 1e-2
