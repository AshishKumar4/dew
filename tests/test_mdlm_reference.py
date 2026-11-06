"""Masked diffusion against MDLM's own loss, reverse step and sampler.

kuleshov-group/mdlm's `_loss` (SUBS, continuous time, antithetic times floored
at 1e-3, as configs/config.yaml trains) and `_ddpm_update`, executed as published
by tools/mdlm_reference.py around a backbone whose logits are `hidden @ head`:
tests/fixtures/mdlm/loss.npz holds one draw, with the two uniforms
`MaskedDiffusionObjective` draws from `jax.random.key(0)` replayed into the
loss's `torch.rand` calls, and the reverse step's categorical;
tests/fixtures/mdlm/seeds.npz the loss over seeds, each seed's inputs drawn
again here. tests/fixtures/mdlm/continuation.npz holds the responses MDLM's
`_sample` draws after a prompt (tools/mdlm_sampling_reference.py).
"""

import functools
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from reference_error import declared_seeds
from reference_runs import assert_as_exact_as_recorded
from sampling_gate import assert_draws_as_declared
from tool_support import run_from

from dew.diffusion.discrete import MDLM, DiscreteDenoiser, Unmask
from dew.nn.protocols import OutputTable
from dew.objectives.base import Step
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.sampling import sample

REFERENCE = dict(np.load(Path(__file__).parent / "fixtures" / "mdlm" / "loss.npz"))
SEEDS = Path(__file__).parent / "fixtures" / "mdlm" / "seeds.npz"
MASK = REFERENCE["head"].shape[1] - 1


@functools.cache
def generator():
    """tools/mdlm_reference.py, whose seeded draw the recorded inputs are drawn again by, loaded as a
    script from its directory."""
    with run_from("mdlm_reference", "tools") as module:
        return module


class Fixed(nn.Module):
    """A backbone whose hidden states and head are its two weights, the fixture's draw's shapes unless
    `rows` is a seed's (`nelbo`)."""

    causal: bool = False
    mask_token_id: None = None
    rows: tuple[int, ...] = REFERENCE["hidden"].shape

    def setup(self):
        self.hidden = self.param("hidden", nn.initializers.zeros, self.rows)
        self.head = self.param("head", nn.initializers.zeros, REFERENCE["head"].shape)

    def __call__(self, tokens, **_):
        return self.hidden_states(tokens) @ self.head

    def hidden_states(self, tokens, train=False, positions=None, segment_ids=None):
        return self.hidden

    def output_table(self):
        return OutputTable(self.head, vocab_major=False)


def objective(rows: tuple[int, ...] = REFERENCE["hidden"].shape) -> MaskedDiffusionObjective:
    return MaskedDiffusionObjective(Fixed(rows=rows), MDLM(mask_id=MASK)(), rows[1])


def test_the_corruption_is_mdlms_on_the_same_draws():
    """MDLM's antithetic time and masking on the uniforms the objective draws."""
    process = MDLM(mask_id=MASK)()
    time_key, mask_key, _ = jax.random.split(jax.random.key(0), 3)
    t = process.sample_t(time_key, REFERENCE["tokens"].shape[0])
    corrupted, _ = process.corrupt(mask_key, jnp.asarray(REFERENCE["tokens"]), t)
    np.testing.assert_array_equal(corrupted, REFERENCE["corrupted"])


def nelbo(dtype, inputs) -> dict[str, jax.Array]:
    """The per-token NELBO terms (zero where a token stayed visible), their mean and its gradient in the
    backbone's weights at `dtype` on a draw, the objective reading the corruption the draw's uniforms
    give (`MDLM.stratified`, `MDLM.masked`) by name (`Step.tape`)."""
    tokens = jnp.asarray(inputs["tokens"], jnp.int32)
    model, process = objective(np.shape(inputs["hidden"])), MDLM(mask_id=MASK)()
    rows = tokens.shape[0]
    t = process.stratified(jnp.asarray(inputs["time_uniform"], dtype), jnp.arange(rows), rows)
    _, is_masked = process.masked(jnp.asarray(inputs["mask_uniform"], dtype), tokens, t)
    step = Step(jnp.asarray(0), jax.random.key(0), None, tape={"times": t, "mask": is_masked})
    variables = {"params": {name: jnp.asarray(inputs[name], dtype) for name in ("hidden", "head")}}
    batch = {"text": tokens}
    (value, _), gradient = jax.value_and_grad(lambda tree: model.scalar_loss(tree, batch, step),
                                              has_aux=True)(variables)
    return {"nlls": model.evaluate(variables, batch, step).losses, "loss": jnp.reshape(value, 1),
            "grad_hidden": gradient["params"]["hidden"], "grad_head": gradient["params"]["head"]}


def test_the_nelbo_and_its_gradient_are_mdlms():
    """The per-token NELBO terms, their mean and its gradient in the backbone's weights, as exact as
    MDLM's float32 run against its float64 run over drawn rows, weights and uniforms (check 1)."""
    assert_as_exact_as_recorded(SEEDS, "nelbo", declared_seeds(SEEDS)["nelbo"], generator().draw, nelbo,
                                "MDLM's NELBO")


def test_the_masked_metrics_count_the_masked_tokens_and_the_right_ones():
    """The step's masked accuracy and masked fraction, each a count over a count, on the fixture's draw:
    the masked tokens the reference's corruption masked, and those its float64 logits rank first."""
    model = objective()
    variables = {"params": {"hidden": jnp.asarray(REFERENCE["hidden"]),
                            "head": jnp.asarray(REFERENCE["head"])}}
    batch = {"text": jnp.asarray(REFERENCE["tokens"], jnp.int32)}
    process = MDLM(mask_id=MASK)()
    time_key, mask_key, _ = jax.random.split(jax.random.key(0), 3)
    t = process.sample_t(time_key, REFERENCE["tokens"].shape[0])
    _, is_masked = process.corrupt(mask_key, jnp.asarray(REFERENCE["tokens"]), t)
    step = Step(jnp.asarray(0), jax.random.key(0), None, tape={"times": t, "mask": is_masked})
    _, aux = model.scalar_loss(variables, batch, step)
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


REVERSE_SEEDS = 20
"""The sampling gate's seeds for MDLM's reverse step, 1000 steps each."""


def test_the_reverse_step_draws_from_mdlms_categorical():
    """MDLM's `_ddpm_update` from t = 0.6 to s = 0.35 draws each masked
    position from one categorical: a token with the model's probability
    times the share revealed, the mask with the share that stays. `Unmask`
    draws the reveal and the token separately; over 20000 steps, each masked
    position's outcomes follow MDLM's categorical, held by the sampling gate
    with a row a masked position (32 positions, 12 categories each). Visible
    positions keep their tokens."""
    process = MDLM(mask_id=MASK)()
    row = jnp.asarray(REFERENCE["reverse_row"], jnp.int32)
    t = jnp.full((row.shape[0],), float(REFERENCE["reverse_t"]), jnp.float32)
    s = jnp.full((row.shape[0],), float(REFERENCE["reverse_s"]), jnp.float32)
    variables = {"params": {"hidden": jnp.asarray(REFERENCE["hidden"]),
                            "head": jnp.asarray(REFERENCE["head"])}}
    denoise = DiscreteDenoiser(process, Fixed(), variables)
    denoised, log_probs = denoise(row, t)
    masked = np.asarray(row == MASK)

    @jax.jit
    def steps(key):
        return jax.vmap(lambda key: Unmask().step(row, t, s, denoised, log_probs, (), key, process,
                                                  denoise)[0])(jax.random.split(key, 1000))

    def counts(seed):
        outcomes = np.asarray(steps(jax.random.key(seed)))
        visible = outcomes[:, ~masked]
        np.testing.assert_array_equal(visible, np.broadcast_to(np.asarray(row)[~masked], visible.shape))
        return np.asarray(jax.nn.one_hot(outcomes, MASK + 1).sum(axis=0), np.int64)[masked]

    assert_draws_as_declared(counts, REFERENCE["reverse_categorical"][masked], "MDLM's reverse step",
                             seeds=REVERSE_SEEDS)


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
