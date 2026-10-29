"""Masked diffusion against MDLM's own loss, reverse step and sampler.

tests/fixtures/mdlm/loss.npz holds kuleshov-group/mdlm's corruption (SUBS,
continuous time, antithetic times floored at 1e-3, as configs/config.yaml
trains) and `_ddpm_update`, executed as published by tools/mdlm_reference.py
around a backbone whose logits are `hidden @ head`, with the two uniforms
`MaskedDiffusionObjective` draws from `jax.random.key(0)` replayed into the
loss's `torch.rand` calls; recorded_nelbo.npz its `_loss` and gradients so
over check 1's seeds. tests/fixtures/mdlm/continuation.npz holds the
responses MDLM's `_sample` draws after a prompt
(tools/mdlm_sampling_reference.py).
"""

from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording

from dew.diffusion.discrete import MDLM, DiscreteDenoiser, Unmask
from dew.nn.protocols import OutputTable
from dew.objectives.base import Step
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.sampling import sample

REFERENCE = dict(np.load(Path(__file__).parent / "fixtures" / "mdlm" / "loss.npz"))
MASK = REFERENCE["head"].shape[1] - 1


class Fixed(nn.Module):
    """A backbone whose hidden states and head are its two weights."""

    causal: bool = False
    mask_token_id: None = None

    def setup(self):
        self.hidden = self.param("hidden", lambda _: jnp.asarray(REFERENCE["hidden"]))
        self.head = self.param("head", lambda _: jnp.asarray(REFERENCE["head"]))

    def __call__(self, tokens, **_):
        return self.hidden_states(tokens) @ self.head

    def hidden_states(self, tokens, train=False, positions=None, segment_ids=None):
        return self.hidden

    def output_table(self):
        return OutputTable(self.head, vocab_major=False)


def objective() -> MaskedDiffusionObjective:
    return MaskedDiffusionObjective(Fixed(), MDLM(mask_id=MASK)(), REFERENCE["tokens"].shape[1])


def test_the_corruption_is_mdlms_on_the_same_draws():
    """MDLM's antithetic time and masking on the uniforms the objective draws."""
    process = MDLM(mask_id=MASK)()
    time_key, mask_key, _ = jax.random.split(jax.random.key(0), 3)
    t = process.sample_t(time_key, REFERENCE["tokens"].shape[0])
    corrupted, _ = process.corrupt(mask_key, jnp.asarray(REFERENCE["tokens"]), t)
    np.testing.assert_array_equal(corrupted, REFERENCE["corrupted"])


RECORDING = Path(__file__).parent / "fixtures" / "mdlm" / "recorded_nelbo.npz"


def taped(tokens, seed) -> Step:
    """The objective's step reading the corruption its key draws from `seed`: its times and masks
    (`Step.tape`)."""
    process = MDLM(mask_id=MASK)()
    time_key, mask_key, _ = jax.random.split(jax.random.key(seed), 3)
    t = process.sample_t(time_key, tokens.shape[0])
    _, is_masked = process.corrupt(mask_key, tokens, t)
    return Step(jnp.asarray(0), jax.random.key(0), None, tape={"times": t, "mask": is_masked})


def nelbo(inputs) -> dict:
    """Dew's per-token NELBO, its mean and the mean's gradient in both weights over a draw
    (tools/mdlm_reference.py `drawn_inputs`), which the recording's paired sizing reads too."""
    model = objective()
    variables = {"params": {"hidden": inputs["hidden"], "head": inputs["head"]}}
    batch = {"text": jnp.asarray(inputs["tokens"], jnp.int32)}
    step = taped(batch["text"], inputs["seed"])
    def loss(variables):
        return model.scalar_loss(variables, batch, step)[0]

    value, gradient = jax.value_and_grad(loss)(variables)
    return {"nlls": model.evaluate(variables, batch, step).losses, "loss": value,
            "grad_hidden": gradient["params"]["hidden"], "grad_head": gradient["params"]["head"]}


def test_the_nelbo_and_its_gradient_are_mdlms():
    """The per-token NELBO terms (zero where a token stayed visible), their mean and its gradient in
    the backbone's weights are on par with MDLM's float32 run from its float64 run (check 1), over
    the recording's seeds of rows, weights and the objective's draws (tools/mdlm_reference.py
    `--record`, `drawn_inputs`), MDLM's loss reading the objective's corruption of each, which is MDLM's
    own on the fixture's draw (`test_the_corruption_is_mdlms_on_the_same_draws`); and the reported
    masked accuracy and fraction on the fixture's draw are the counts' own float32 divisions."""
    from tools.mdlm_reference import drawn_inputs

    seeds = declared_seeds(RECORDING)["nelbo"]
    draw, reference, truth = indexed_recording(recording(RECORDING), "nelbo", seeds, inputs_of=drawn_inputs)
    assert_as_exact(lambda index, inputs: nelbo(inputs), reference, draw, "MDLM's NELBO", seeds=seeds,
                    truth_fn=truth)

    variables = {"params": {"hidden": jnp.asarray(REFERENCE["hidden"]),
                            "head": jnp.asarray(REFERENCE["head"])}}
    batch = {"text": jnp.asarray(REFERENCE["tokens"], jnp.int32)}
    model = objective()
    _, aux = model.scalar_loss(variables, batch, taped(batch["text"], 0))
    tokens, corrupted = REFERENCE["tokens"], REFERENCE["corrupted"]
    logits = REFERENCE["hidden"].astype(np.float64) @ REFERENCE["head"].astype(np.float64)
    logits[..., MASK] = -np.inf
    masked = corrupted == MASK
    assert 0 < masked.sum() < masked.size
    right = logits.argmax(-1)[masked] == tokens[masked]
    # Each report is a count over a count, one float32 division, so it is that division's own value.
    np.testing.assert_array_equal(aux.metrics["masked_accuracy"],
                                  np.float32(right.sum()) / np.float32(right.size))
    np.testing.assert_array_equal(aux.metrics["masked_fraction"],
                                  np.float32(masked.sum()) / np.float32(masked.size))


def test_the_reverse_step_draws_from_mdlms_categorical():
    """MDLM's `_ddpm_update` from t = 0.6 to s = 0.35 draws each masked
    position from one categorical: a token with the model's probability
    times the share revealed, the mask with the share that stays. `Unmask`
    draws the reveal and the token separately; over 20000 steps, each masked
    position's outcomes follow MDLM's categorical, by Pearson's chi-square
    over every masked position's categories pooled (32 positions, 11 free
    categories each), at a one-in-a-billion false alarm. Visible positions
    keep their tokens."""
    process = MDLM(mask_id=MASK)()
    row = jnp.asarray(REFERENCE["reverse_row"], jnp.int32)
    t = jnp.full((row.shape[0],), float(REFERENCE["reverse_t"]), jnp.float32)
    s = jnp.full((row.shape[0],), float(REFERENCE["reverse_s"]), jnp.float32)
    variables = {"params": {"hidden": jnp.asarray(REFERENCE["hidden"]),
                            "head": jnp.asarray(REFERENCE["head"])}}
    denoise = DiscreteDenoiser(process, Fixed(), variables)
    denoised, log_probs = denoise(row, t)
    draws = 20000

    def step(key):
        return Unmask().step(row, t, s, denoised, log_probs, (), key, process, denoise)[0]

    outcomes = jax.vmap(step)(jax.random.split(jax.random.key(3), draws))
    masked = np.asarray(row == MASK)
    counts = np.asarray(jax.nn.one_hot(outcomes, MASK + 1).sum(axis=0))[masked]
    expected = draws * REFERENCE["reverse_categorical"][masked]
    visible = np.asarray(outcomes)[:, ~masked]
    np.testing.assert_array_equal(visible, np.broadcast_to(np.asarray(row)[~masked], visible.shape))
    statistic = float(np.sum(np.square(counts - expected) / expected))
    assert counts.size - counts.shape[0] == 352
    # scipy.stats.chi2.isf(1e-9, 352): the statistic a correct sampler exceeds once in a billion runs.
    assert statistic < 535.1, statistic


CONTINUATION = dict(np.load(Path(__file__).parent / "fixtures" / "mdlm" / "continuation.npz"))


class Reader(nn.Module):
    """Logits from the mean token embedding of the row plus each position's own."""

    @nn.compact
    def __call__(self, tokens):
        embed, position, head = (self.param(name, lambda _, name=name: jnp.asarray(CONTINUATION[name]))
                                 for name in ("embed", "position", "head"))
        return (jnp.mean(embed[tokens], axis=1, keepdims=True) + position) @ head


def test_a_continuation_draws_the_responses_mdlms_sampler_draws():
    """MDLM's `_sample` from the prompt and a masked response, for the
    fixture's steps, and `sample` over the masked process for as many steps
    with only the response mutable, draw the same distribution of responses:
    the reveal grid, each step's reveal and token draws and the closing
    argmax that removes the noise left. The stand-in reads the whole row, so
    the order positions are revealed in shapes the outcome. The two samples'
    counts over every response agree by the two-sample chi-square, at a
    one-in-a-billion false alarm, and the prompt is kept as given."""
    process = MDLM(mask_id=int(CONTINUATION["embed"].shape[0]) - 1)()
    prompt, response, rows = CONTINUATION["prompt"], int(CONTINUATION["response"]), int(CONTINUATION["rows"])
    row = jnp.concatenate([jnp.asarray(prompt, jnp.int32), jnp.full((response,), process.mask_id, jnp.int32)])
    variables = Reader().init(jax.random.key(0), row[None])
    mutable = (jnp.arange(row.shape[0]) >= len(prompt))[None]
    denoise = DiscreteDenoiser(process, Reader(), variables, mutable_mask=mutable)
    drawn = np.asarray(sample(denoise, jnp.tile(row, (rows, 1)), int(CONTINUATION["steps"]), solver=Unmask(),
                              key=jax.random.key(5)))
    np.testing.assert_array_equal(drawn[:, :len(prompt)], np.broadcast_to(prompt, (rows, len(prompt))))
    index = drawn[:, len(prompt):] @ (process.mask_id ** np.arange(response - 1, -1, -1))
    ours, theirs = np.bincount(index, minlength=process.mask_id ** response), CONTINUATION["counts"]
    assert ours.size == theirs.size == 64 and np.all(ours + theirs > 0)
    statistic = float(np.sum(np.square(ours - theirs) / (ours + theirs)))
    # scipy.stats.chi2.isf(1e-9, 63): the statistic two samples of one distribution exceed once in a billion.
    assert statistic < 155.07, statistic
