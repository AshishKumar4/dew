"""Masked diffusion against MDLM's own loss and reverse step.

tests/fixtures/mdlm/loss.npz holds kuleshov-group/mdlm's `_loss` (SUBS,
continuous time, antithetic times floored at 1e-3, as configs/config.yaml
trains) and `_ddpm_update`, executed as published by tools/mdlm_reference.py
around a backbone whose logits are `hidden @ head`, with the two uniforms
`MaskedDiffusionObjective` draws from `jax.random.key(0)` replayed into the
loss's `torch.rand` calls.
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

REFERENCE = dict(np.load(Path(__file__).parent / "fixtures" / "mdlm" / "loss.npz"))
MASK = REFERENCE["head"].shape[1] - 1


class Fixed(nn.Module):
    """A backbone whose hidden states and head are its two weights."""

    causal: bool = False

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
