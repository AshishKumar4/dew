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

from dew.diffusion.discrete import MDLM, Confident, Unmask
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


def test_llada2s_task_walks_as_its_release_and_generates_the_recorded_prompts():
    """LLaDA2's task (`text_generation`, what `dew.pipeline` and a run return) walks as its release's
    `generate` defaults do: blocks of 32 over 32 steps each, revealing past 0.95 (`Confident`); and the
    task, at the recording's settings, generates each recorded prompt's tokens through its own call, the
    public path a user's request takes (ledger 2174). The settings a saved run records take the same
    walk."""
    from dew.inference.tasks import MaskedGeneration, MaskedSettings

    recorded = json.loads((FIXTURES / "llada2-tiny" / "generate.json").read_text())
    settings = recorded["settings"]
    source = Pretrained.load(FIXTURES / "llada2-tiny", dtype="float32", attention_impl="reference")
    task = source.text_generation()
    assert isinstance(task, MaskedGeneration) and isinstance(task.solver, Confident)
    assert (task.solver.threshold, task.solver.temperature, task.block, task.steps) == (0.95, 0.0, 32, 32)
    walked = replace(task, solver=Confident(threshold=settings["threshold"], top_k=settings["top_k"]),
                     steps=settings["steps"], block=settings["block_length"],
                     eos_token_ids=(settings["eos_id"],))
    for row in recorded["rows"]:
        generation = walked(jnp.asarray([row["prompt"]], jnp.int32), settings["gen_length"], key=0).host()
        start = len(row["prompt"])
        response = np.asarray(generation.tokens)[0, start:start + int(generation.lengths[0])]
        assert response.tolist() == row["generated"], (row["prompt"], response.tolist())
    saved = MaskedSettings(seq_len=16, max_new_tokens=8, tokenizer=None, processor=None,
                           process=task.process, solver=walked.solver,
                           sampling_steps=walked.steps, block=walked.block,
                           eos_token_ids=walked.eos_token_ids)
    rebuilt = MaskedGeneration.from_settings(task.model, task.variables, saved, task.processor)
    assert (rebuilt.solver, rebuilt.block, rebuilt.steps) == (walked.solver, walked.block, walked.steps)


def test_llada2s_task_continues_stops_and_takes_a_zero_budget():
    """LLaDA2's blockwise task over padded rows: its greedy reveal draws every continuation of a prompt
    alike (`n`); an end token it draws stops a row there, the rest padding; and a zero budget returns
    the prompts with no response and no refinement."""
    source = Pretrained.load(FIXTURES / "llada2-tiny", dtype="float32", attention_impl="reference")
    task = replace(source.text_generation(), block=4, steps=4)
    tokens = jnp.asarray([[0, 3, 5, 6], [1, 2, 3, 4], [0, 0, 7, 8]])
    inputs = ModelInputs(tokens, {"attention_mask": jnp.asarray([[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]],
                                                                bool)})
    single = task(inputs, 8, key=7).host()
    many = task(inputs, 8, key=7, n=3).host()
    np.testing.assert_array_equal(many.tokens.reshape(3, 3, 12), np.repeat(single.tokens[:, None], 3, 1))
    eos = int(single.tokens[0, 5])
    stopped = replace(task, eos_token_ids=(eos,))(inputs, 8, key=7).host()
    for row, response in enumerate(single.tokens[:, 4:]):
        hits = np.flatnonzero(response == eos)
        length = int(hits[0]) + 1 if hits.size else 8
        assert (int(stopped.lengths[row]), bool(stopped.terminated[row])) == (length, bool(hits.size))
        np.testing.assert_array_equal(stopped.tokens[row, 4:4 + length], response[:length])
        np.testing.assert_array_equal(stopped.tokens[row, 4 + length:], task.pad_token_id)
    empty = task(inputs, 0, key=7, n=2).host()
    np.testing.assert_array_equal(empty.tokens, np.repeat(np.asarray(tokens), 2, axis=0))
    np.testing.assert_array_equal(empty.lengths, 0)
    np.testing.assert_array_equal(empty.decoder_steps, 0)


def test_a_run_over_llada2_saves_and_serves_its_walk(tmp_path):
    """An objective over a loaded LLaDA2.0 source takes its walk by default (`masked_recipe`); a run of
    it saves the walk in its record, and the run's task (`MaskedGeneration.from_run`, `dew.pipeline` of
    the run) holds the same solver, block and steps and generates the live task's tokens, as
    `dew.pipeline` of the source itself walks (ledger 2174)."""
    import grain.python as grain
    import optax

    from dew.checkpoints import Checkpoints
    from dew.data import ByteTokenizer
    from dew.data.dataset import Loader
    from dew.diffusion.discrete import MDLM
    from dew.inference import RunProcessor, pipeline
    from dew.inference.tasks import MaskedGeneration
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective
    from dew.training import Trainer

    source = Pretrained.load(FIXTURES / "llada2-tiny", dtype="float32", attention_impl="xla")
    solver, steps, block = source.masked_recipe
    # LLaDA2.2 generates with an editing stage Dew does not run: its source, and a config Dew's export
    # writes for it (its block routing, no DELETE or SPLIT ids), take MDLM.
    from dew.interop.hf_decoders import family_entry

    assert Pretrained.load(FIXTURES / "llada22-tiny").masked_recipe is None
    exported = {**json.loads((FIXTURES / "llada22-tiny" / "config.json").read_text())}
    for field in ("delete_token_id", "split_token_id"):
        exported.pop(field)
    assert family_entry("llada2_moe").masked_walk(exported) is None
    assert isinstance(pipeline(str(FIXTURES / "llada2-tiny")).solver, Confident)
    objective = MaskedDiffusionObjective(source, MDLM(mask_id=source.model.mask_token_id)(), 8,
                                         ema_decay=None, processor=RunProcessor(ByteTokenizer()))
    assert (objective.solver, objective.steps, objective.block) == (solver, steps, block)
    # A solver the caller names walks whole rows unless it names a block too (ledger 2231).
    named = MaskedDiffusionObjective(source, MDLM(mask_id=source.model.mask_token_id)(), 8, ema_decay=None,
                                     processor=RunProcessor(ByteTokenizer()), solver=Unmask())
    assert named.block is None
    rows = jax.device_count()
    stream = grain.MapDataset.source([{"text": np.full((rows, 8), 7, np.int32)}]).repeat().to_iter_dataset()
    data = Loader(train=lambda partition: iter(stream), val=None, records=rows, batch=rows)
    state = Trainer(objective, optax.sgd(0.05), key=jax.random.key(19),
                    checkpoints=Checkpoints(str(tmp_path / "run"))).fit(data, steps=1, checkpoint_every=1,
                                                                         log_every=1)
    live = objective.pipeline(state)
    prompt = jnp.asarray([[3, 5, 6]], jnp.int32)
    expected = live(prompt, 8, key=7).host().tokens
    for restored in (MaskedGeneration.from_run(str(tmp_path / "run")), pipeline(str(tmp_path / "run"))):
        assert (restored.solver, restored.steps, restored.block) == (solver, steps, block)
        np.testing.assert_array_equal(restored(prompt, 8, key=7).host().tokens, expected)


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
