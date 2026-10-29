"""A masked walk a block at a time (`DiscreteProcess.generate`'s `block`) and LLaDA2's reveal (`Confident`),
against LLaDA2's release (inclusionAI/LLaDA2.0-mini@dad945c, its own `generate`, recorded by
tools/llada_moe_reference.py --generate; ledger 1061)."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.diffusion.discrete import MDLM, Confident
from dew.interop import Pretrained
from dew.nn.inputs import ModelInputs

FIXTURES = Path(__file__).parent / "fixtures" / "hf"


def test_a_blockwise_walk_generates_what_llada2s_release_generates():
    """LLaDA2's tiny decoder walked in blocks of four places over four steps each, revealing as its release
    does (`Confident`, its best token alone), generates each recorded prompt's tokens as the release's own
    `generate` does: prompts ending inside a block, across one and on a block's end, so the walk takes a
    prompt's last block with its prompt slots held, sees each block's clean predecessors and none after it
    (`attention_key_runs`), and walks its last block whole past the eight tokens kept. The release's head
    holds its mask-token logit out (the recording's hook), as Dew's walk draws no mask token. The prompts
    walked together, left-padded to one width, generate the same, as does each prompt right-padded, its
    response past its padding: a row reads no padding and no other row. Each row reports its useful
    refinements: at `top_k` 1 every block is whole after its first step (each draw certain, past the
    threshold), so one a block its response reaches, through its end token's where it draws one."""
    recorded = json.loads((FIXTURES / "llada2-tiny" / "generate.json").read_text())
    settings = recorded["settings"]
    source = Pretrained.load(FIXTURES / "llada2-tiny", dtype="float32", attention_impl="reference")
    process = MDLM(mask_id=recorded["mask_id"])()
    solver = Confident(threshold=settings["threshold"], top_k=settings["top_k"])
    budget, block, steps = settings["gen_length"], settings["block_length"], settings["steps"]

    def generated(inputs):
        return process.generate(source.model, source.variables, inputs, budget, key=0, steps=steps,
                                solver=solver, eos_token_ids=(settings["eos_id"],), block=block)

    width = max(len(row["prompt"]) for row in recorded["rows"])
    padded = np.zeros((len(recorded["rows"]), width), np.int32)
    real = np.zeros(padded.shape, bool)
    for index, row in enumerate(recorded["rows"]):
        padded[index, width - len(row["prompt"]):] = row["prompt"]
        real[index, width - len(row["prompt"]):] = True
    together = generated(ModelInputs(jnp.asarray(padded), {"attention_mask": jnp.asarray(real)}))
    for index, row in enumerate(recorded["rows"]):
        alone = generated(jnp.asarray([row["prompt"]], jnp.int32))
        # The prompt right-padded by two: its response follows the padding.
        right = generated(ModelInputs(jnp.asarray([row["prompt"] + [0, 0]], jnp.int32), {
            "attention_mask": jnp.asarray([[True] * len(row["prompt"]) + [False, False]])}))
        # The blocks the response reaches, counted from the prompt's first token: through its end token's,
        # where it draws one, else its last's.
        last = (row["generated"].index(settings["eos_id"]) if settings["eos_id"] in row["generated"]
                else budget - 1)
        reached = (len(row["prompt"]) + last) // block - len(row["prompt"]) // block + 1
        for generation, at, start in ((alone, 0, len(row["prompt"])), (together, index, width),
                                      (right, 0, len(row["prompt"]) + 2)):
            length = int(generation.lengths[at])
            response = np.asarray(generation.tokens)[at, start:start + length]
            assert response.tolist() == row["generated"], (row["prompt"], at, response.tolist())
            assert int(generation.decoder_steps[at]) == reached, (row["prompt"], at, generation.decoder_steps)


def _release_reveal(confidence: np.ndarray, quota: int, threshold: float) -> np.ndarray:
    """LLaDA2's release's transfer of one row's step (modeling_llada2_moe.py, `generate`): every position
    above `threshold` where they number at least `quota`, else the `quota` (at most the masked positions)
    most confident, a tie at the quota's edge to the earlier slot, Dew's choice where `torch.topk` leaves
    it unspecified (`Confident`)."""
    high = confidence > threshold
    if high.sum() >= quota:
        return high
    chosen = np.zeros_like(high)
    chosen[np.argsort(-confidence, kind="stable")[:min(quota, int(np.isfinite(confidence).sum()))]] = True
    return chosen


def test_confident_spreads_a_fixed_quota_where_no_block_is_walked():
    """Without a block, `Confident`'s quota spreads the slots masked at the walk's first step over its
    steps, fixed: eight masked slots over three steps reveal 3, 3 and 2 at a threshold no draw passes,
    every slot revealed by the last step."""
    source = Pretrained.load(FIXTURES / "llada2-tiny", dtype="float32", attention_impl="reference")
    mask, steps = 120, 3
    process = MDLM(mask_id=mask)()
    denoise = process.denoiser(source.model, source.variables)
    solver = Confident(threshold=1.0, temperature=0.0)
    x = jnp.full((1, 8), mask, jnp.int32)
    state = solver.init(x, jnp.linspace(1.0, 0.0, steps + 1), process, key=jax.random.key(0))
    rng = np.random.default_rng(1061)
    revealed = []
    for _ in range(steps):
        logits = rng.normal(size=(1, 8, 128))
        logits[..., mask] = -np.inf
        log_probs = jax.nn.log_softmax(jnp.asarray(logits, jnp.float32), axis=-1)
        before = np.asarray(x)
        x, state = solver.step_rows(x, None, None, None, log_probs, state,
                                    jax.random.split(jax.random.key(1), 1), process, denoise)
        revealed.append(int(np.sum(np.asarray(x) != before)))
    assert revealed == [3, 3, 2]


def test_confident_refuses_a_temperature_that_is_no_finite_number():
    """A temperature of nan or infinity divides the scores into no distribution: refused."""
    for temperature in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            Confident(temperature=temperature)


@pytest.mark.parametrize("threshold", [0.95, 0.3])
def test_confident_reveals_as_llada2s_release_over_a_block(threshold):
    """`Confident` at temperature 0 reveals, at each of a block's steps, the positions LLaDA2's release's
    rule reveals from the same confidences (`_release_reveal`): those above the threshold where they meet
    the step's quota, else the quota's most confident, the quota spreading the block's eight slots (two of
    them its prompt's) over three steps as 3, 3, 2; a low threshold reveals past the quota. With no block
    walked, the quota spreads the slots masked at the first step, fixed across the steps."""
    source = Pretrained.load(FIXTURES / "llada2-tiny", dtype="float32", attention_impl="reference")
    mask, steps, width = 120, 3, 8
    process = MDLM(mask_id=mask)()
    block = jnp.ones((2, width), bool)
    mutable = block.at[:, :2].set(False)
    denoise = replace(process.denoiser(source.model, source.variables, mutable_mask=mutable), block=block)
    solver = Confident(threshold=threshold, temperature=0.0)
    x = jnp.where(mutable, mask, 7).astype(jnp.int32)
    state = solver.init(x, jnp.linspace(1.0, 0.0, steps + 1), process, key=jax.random.key(0))
    rng = np.random.default_rng(1061)
    for _ in range(steps):
        logits = rng.normal(size=(2, width, 128)) * rng.uniform(0.5, 8.0, size=(2, width, 1))
        logits[..., mask] = -np.inf
        log_probs = jax.nn.log_softmax(jnp.asarray(logits, jnp.float32), axis=-1)
        before = np.asarray(x)
        x, state = solver.step_rows(x, None, None, None, log_probs, state,
                                    jax.random.split(jax.random.key(1), 2), process, denoise)
        index = int(state[0]) - 1
        quota = width // steps + (index < width % steps)
        assert np.asarray(state[2]).tolist() == [width] * 2
        probability = np.exp(np.max(np.asarray(log_probs), axis=-1))
        for row in range(2):
            masked = before[row] == mask
            expected = _release_reveal(np.where(masked, probability[row], -np.inf), quota, threshold)
            assert np.array_equal(np.asarray(x)[row] != before[row], expected & masked), (row, index)
