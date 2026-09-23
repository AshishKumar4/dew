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


def objective():
    pretrained = load_pretrained(str(FIXTURES / "hf/qwen3-moe-vllm"), dtype="float32", attention_impl="reference")
    grpo = GRPOObjective(pretrained.model, WIDTH - 1, sampling_temperature=RECORD["sampling"]["temperature"])
    return grpo, pretrained.variables


def test_the_record_decodes_to_the_models_layers():
    """vLLM indexes every decoder layer; qwen3-moe-vllm's layer 0 is dense,
    so its rows are the capture buffer's zeros."""
    for call in RECORD["calls"]:
        routed = np.asarray(call["routed_experts"])
        assert routed.shape == (len(call["prompt_ids"]) + len(call["sampled_ids"]) - 1, 3, 2)
        assert not routed[:, 0].any() and routed[:, 1:].any()


def test_packed_scoring_reproduces_the_engines_filtered_likelihoods():
    """With the engine's routing replayed and each id renormalized over the
    support vLLM kept, Dew's fp32 likelihoods match vLLM's bf16 processed
    ones up to vLLM's rounding: unbiased, 0.015 mean and 0.054 worst on the
    recorded fixture. The tempered full-vocabulary likelihood misses by
    about 1."""
    grpo, variables = objective()
    _, batch = engine_batch()
    scored = np.asarray(grpo.packed_log_probs(variables, batch))
    sampled = batch["response_mask"] != 0
    behavior = batch["behavior_log_probs"][sampled]
    error = scored[sampled] - behavior
    assert np.abs(error).mean() < 0.03 and np.abs(error).max() < 0.1 and abs(error.mean()) < 0.005
    unfiltered = {key: value for key, value in batch.items() if key not in ("support_ids", "support_columns")}
    raw = np.asarray(grpo.packed_log_probs(variables, unfiltered))[sampled]
    assert np.abs(raw - behavior).max() > 0.5


def test_the_trainer_routes_every_token_as_the_engine_did():
    """Dew's routing of the engine's ids, sown by the same forward the loss
    runs, equals vLLM's record on both sparse layers wherever the engine
    forwarded an id."""
    grpo, variables = objective()
    _, batch = engine_batch()
    ids = batch["input_ids"]
    packing = {"segment_ids": batch["text_segment_ids"], "positions": batch["text_positions"]}
    covered = batch[ROUTED_KEY][:, :-1]

    def routing(routes, layer):
        sown = grpo.token_scores(variables, ids, routing=True, routes=routes, **packing).routing
        return np.sort(np.asarray(sown[f"layers_{layer}"]["mlp"]["gate"]["indices"][0]), -1)

    for layer in (1, 2):
        engine = np.sort(batch[ROUTED_EXPERTS_KEY][:, :-1, layer], -1)
        replayed = routing((batch[ROUTED_EXPERTS_KEY], batch[ROUTED_KEY]), layer)
        np.testing.assert_array_equal(replayed[covered], engine[covered])
