"""Masked diffusion against MDLM's own loss, reverse step and sampler.

tests/fixtures/mdlm/loss.npz holds kuleshov-group/mdlm's `_loss` (SUBS,
continuous time, antithetic times floored at 1e-3, as configs/config.yaml
trains) and `_ddpm_update`, executed as published by tools/mdlm_reference.py
around a backbone whose logits are `hidden @ head`, with the two uniforms
`MaskedDiffusionObjective` draws from `jax.random.key(0)` replayed into the
loss's `torch.rand` calls. tests/fixtures/mdlm/continuation.npz holds the
responses MDLM's `_sample` draws after a prompt
(tools/mdlm_sampling_reference.py).
"""

from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact_as_the_reference

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


def test_the_nelbo_and_its_gradient_are_mdlms():
    """The per-token NELBO terms (zero where a token stayed visible), their
    mean and its gradient in the backbone's weights, against MDLM's float32
    run, measured from its float64 run."""
    model = objective()
    variables = {"params": {"hidden": jnp.asarray(REFERENCE["hidden"]),
                            "head": jnp.asarray(REFERENCE["head"])}}
    batch = {"text": jnp.asarray(REFERENCE["tokens"], jnp.int32)}
    step = Step(jnp.asarray(0), jax.random.key(0), None)

    def mean(variables):
        return model.scalar_loss(variables, batch, step)[0]

    value, gradient = jax.value_and_grad(mean)(variables)
    terms = model.evaluate(variables, batch, step).losses
    assert_as_exact_as_the_reference(terms, REFERENCE["nlls"], REFERENCE["nlls_f64"], "per-token NELBO")
    assert_as_exact_as_the_reference(gradient["params"]["hidden"], REFERENCE["grad/hidden"],
                                     REFERENCE["grad/hidden_f64"], "gradient in the hidden states")
    assert_as_exact_as_the_reference(gradient["params"]["head"], REFERENCE["grad/head"],
                                     REFERENCE["grad/head_f64"], "gradient in the head")
    np.testing.assert_allclose(value, REFERENCE["loss_f64"], rtol=1e-6)


def test_the_reported_masked_accuracy_and_fraction_are_over_mdlms_masking():
    """Over MDLM's own corruption of the same draws: the share of masked
    positions whose float64 argmax, the mask token left out, is the clean
    token, and the share of positions masked."""
    tokens, corrupted = REFERENCE["tokens"], REFERENCE["corrupted"]
    logits = (REFERENCE["hidden"].astype(np.float64) @ REFERENCE["head"].astype(np.float64))
    logits[..., MASK] = -np.inf
    masked = corrupted == MASK
    variables = {"params": {name: jnp.asarray(REFERENCE[name]) for name in ("hidden", "head")}}
    _, aux = objective().loss(variables, {"text": jnp.asarray(tokens, jnp.int32)},
                              Step(jnp.asarray(0), jax.random.key(0), None))
    assert 0 < masked.sum() < masked.size
    right = logits.argmax(-1)[masked] == tokens[masked]
    np.testing.assert_allclose(aux.metrics["masked_accuracy"], right.mean())
    np.testing.assert_allclose(aux.metrics["masked_fraction"], masked.mean())


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
