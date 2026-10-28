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

FIXTURES = Path(__file__).parent / "fixtures" / "hf"


def test_a_blockwise_walk_generates_what_llada2s_release_generates():
    """LLaDA2's tiny decoder walked in blocks of four places over four steps each, revealing as its release
    does (`Confident`, its best token alone), generates each recorded prompt's tokens as the release's own
    `generate` does: prompts ending inside a block, across one and on a block's end, so the walk takes a
    prompt's last block with its prompt slots held, sees each block's clean predecessors and none after it
    (`attention_key_runs`), and walks its last block whole past the eight tokens kept. The release's head
    holds its mask-token logit out (the recording's hook), as Dew's walk draws no mask token."""
    recorded = json.loads((FIXTURES / "llada2-tiny" / "generate.json").read_text())
    settings = recorded["settings"]
    source = Pretrained.load(FIXTURES / "llada2-tiny", dtype="float32", attention_impl="reference")
    process = MDLM(mask_id=recorded["mask_id"])()
    solver = Confident(threshold=settings["threshold"], top_k=settings["top_k"])
    for row in recorded["rows"]:
        prompt = jnp.asarray([row["prompt"]], jnp.int32)
        generation = process.generate(source.model, source.variables, prompt, settings["gen_length"], key=0,
                                      steps=settings["steps"], solver=solver,
                                      eos_token_ids=(settings["eos_id"],), block=settings["block_length"])
        length = int(generation.lengths[0])
        response = np.asarray(generation.tokens)[0, len(row["prompt"]):len(row["prompt"]) + length]
        assert response.tolist() == row["generated"], (row["prompt"], response.tolist())


def _release_reveal(confidence: np.ndarray, quota: int, threshold: float) -> np.ndarray:
    """LLaDA2's release's transfer of one row's step (modeling_llada2_moe.py, `generate`), ported as written:
    every position above `threshold` where they number at least `quota`, else the `quota` (at most the
    masked positions) most confident."""
    high = confidence > threshold
    if high.sum() >= quota:
        return high
    chosen = np.zeros_like(high)
    chosen[np.argsort(-confidence, kind="stable")[:min(quota, int(np.isfinite(confidence).sum()))]] = True
    return chosen


@pytest.mark.parametrize("threshold", [0.95, 0.3])
def test_confident_reveals_as_llada2s_release_over_a_block(threshold):
    """`Confident` at temperature 0 reveals, at each of a block's steps, the positions LLaDA2's release's
    rule reveals from the same confidences (`_release_reveal`): those above the threshold where they meet
    the step's quota, else the quota's most confident, the quota spreading the block's eight slots (two of
    them its prompt's) over three steps as 3, 3, 2; a low threshold reveals past the quota."""
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
        probability = np.exp(np.max(np.asarray(log_probs), axis=-1))
        for row in range(2):
            masked = before[row] == mask
            expected = _release_reveal(np.where(masked, probability[row], -np.inf), quota, threshold)
            assert np.array_equal(np.asarray(x)[row] != before[row], expected & masked), (row, index)
