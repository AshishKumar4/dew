"""Routing and support replay against what vLLM actually returned.

`tools/vllm_replay_reference.py` serves `tests/fixtures/hf/qwen3-moe-vllm` in bf16 on vLLM 0.30.0 with
top-k 6, top-p 0.9 and temperature 0.8, asking for processed log-probabilities, the sampling mask and the
routed experts, over drawn prompts (the `vllm-replay` recording, `FIXTURE`). Dew loads the same
checkpoint, packs the completions as `Call` records and scores them with the packed GRPO objective.
"""

import functools
from pathlib import Path

import numpy as np
import pytest
from reference_error import indexed_recording, recording
from reference_runs import assert_arms_as_recorded
from tool_support import load

from dew.batch import coordinates
from dew.interop.pretrained import Pretrained
from dew.objectives.rl.grpo import GRPOObjective
from dew.objectives.rl.sessions import ROUTED_EXPERTS_KEY, ROUTED_KEY, Call, Session, Status, pack

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLING = load("vllm_replay_reference").SAMPLING
WIDTH = 24


def engine_batch(inputs: dict[str, np.ndarray]):
    """A draw's completions as vLLM returned them (`dew`'s inputs) packed as `Call` records for the GRPO
    objective."""
    rollouts = []
    for number, prompt in enumerate(inputs["prompt_ids"]):
        record = Call(
            tuple(prompt.tolist()),
            tuple(inputs["sampled_ids"][number].tolist()),
            tuple(inputs["behavior_log_probs"][number].tolist()),
            "length",
            0,
            routed_experts=inputs["routed_experts"][number],
            support=tuple(tuple(int(index) for index in kept if index >= 0)
                          for kept in inputs["support"][number]),
        )
        rollouts.append(Session("t", "g", number, 0, (record,), Status.COMPLETED, float(number)))
    return rollouts, pack(rollouts, WIDTH, rows=len(rollouts), support_capacity=SAMPLING["top_k"] * WIDTH)


@pytest.fixture(scope="module")
def objective():
    return scorer()


FIXTURE = FIXTURES / "rl" / "vllm_replay_recording.npz"
"""tools/venv_reference.py's `vllm-replay` recording (tools/vllm_replay_worker.py): each draw's completions
as Dew replays them, and transformers' float32 and float64 scores of them."""
REPLAY_SEEDS = 16
"""The site's declared seeds, sized by `seeds_for` paired as its recording confirmed them."""
OUTPUTS = ("log_probs",)
PROMPTS, LENGTH = 4, 8


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's prompts, PROMPTS rows of LENGTH ids over the checkpoint's vocabulary, and vLLM's sampling
    seed."""
    rng = np.random.default_rng([4736, seed])
    return {"prompt_ids": rng.integers(0, 128, (PROMPTS, LENGTH)).astype(np.int32),
            "seed": np.asarray(seed, np.int64)}


@functools.cache
def scorer():
    """The packed GRPO objective over the checkpoint in float32, and its variables."""
    pretrained = Pretrained.load(str(FIXTURES / "hf/qwen3-moe-vllm"), dtype="float32",
                                 attention_impl="reference")
    grpo = GRPOObjective(pretrained.model, WIDTH - 1, sampling_temperature=SAMPLING["temperature"])
    return grpo, pretrained.variables


def dew(_, inputs) -> dict:
    """Dew's float32 log-probabilities of a draw's sampled ids, the engine's routing and kept support
    replayed (`packed_log_probs`), in the recording's layout."""
    grpo, variables = scorer()
    _, batch = engine_batch(inputs)
    sampled = batch["response_mask"] != 0
    order = np.argsort(batch["session_index"][sampled], kind="stable")
    scores = np.asarray(grpo.packed_log_probs(variables, batch))[sampled][order]
    return {"log_probs": scores.reshape(inputs["sampled_ids"].shape)}


def recorded_draw():
    """The recording's first draw, its completions and transformers' float64 scores of them."""
    drawn, _, truth = indexed_recording(recording(FIXTURE), "vllm-replay", REPLAY_SEEDS, inputs_of=draw)
    index, inputs = drawn(0)
    return inputs, np.asarray(truth(index, inputs)["log_probs"])


def test_packed_scoring_computes_the_engines_filtered_likelihoods():
    """With the engine's routing replayed and each id renormalized over the support vLLM kept at its
    temperature, Dew's fp32 likelihoods are on par with transformers' fp32 run on the same replay over drawn
    prompts and sampling seeds, both measured from transformers' float64 (check 1, row 4736)."""
    assert_arms_as_recorded((FIXTURE, draw, dew, OUTPUTS), "vllm-replay", REPLAY_SEEDS,
                            "vLLM replay log-probs", host=True)


def test_the_full_vocabulary_likelihood_misses_the_engines_filtered_one(objective):
    """The tempered full-vocabulary likelihood, the engine's kept support dropped, misses the filtered
    float64 truth by far more than rounding: the support is what the replay scores."""
    grpo, variables = objective
    inputs, truth = recorded_draw()
    _, batch = engine_batch(inputs)
    sampled = batch["response_mask"] != 0
    order = np.argsort(batch["session_index"][sampled], kind="stable")
    unfiltered = {key: value for key, value in batch.items() if key not in ("support_ids", "support_columns")}
    raw = np.asarray(grpo.packed_log_probs(variables, unfiltered))[sampled][order]
    assert np.abs(raw - truth.reshape(-1)).max() > 0.5


def test_the_trainer_routes_every_token_as_the_record_says(objective):
    """The routing Dew's forward sows follows the record it replays. On
    vLLM's record Dew's own top-k already agrees, so the check also replays
    a record altered to send every covered id to the two experts Dew ranks
    last on each sparse layer: the sown routing follows it, and the
    likelihoods move."""
    grpo, variables = objective
    _, batch = engine_batch(recorded_draw()[0])
    ids = batch["input_ids"]
    packing = {"coordinates": coordinates(segments=batch["text_segment_ids"],
                                          positions=batch["text_positions"])}
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
