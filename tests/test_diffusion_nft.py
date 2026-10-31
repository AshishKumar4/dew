"""DiffusionNFT against its released trainer and arXiv 2509.16117: NVlabs/DiffusionNFT at
cbb14f84b8312b620390dfcbe2fab69c1383b104, scripts/train_nft_sd3.py and flow_grpo/stat_tracking.py.

The loss's arithmetic is the release's, transcribed here in NumPy apart from Dew's code and held in
float64 within float64 rounding of it (`assert_computes_the_oracle`), its gradient included, which
reads the detached weights; the rollout's advantages and the old policy's update are the release's
arithmetic on the values the run produced, bit for bit where they are one computation.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_computes_the_oracle, chain_roundings, widened

from dew.diffusion import FlowMatchingScheduler, FlowMatchPredictionTransform, Process
from dew.inputs import CharTable, Condition, Field, InputSpec
from dew.objectives.rl.nft import DiffusionNFTObjective, NFTRollout, released_old_decay, released_policy_loss
from dew.training import Trainer

ROWS, SHAPE = 16, (4, 4, 2)


def transcribed_policy_loss(samples, live, old, r, beta, scale):
    """train_nft_sd3.py:934-960 in NumPy float64: the implicit policies' clean estimates, each one's
    mean squared error over its detached mean absolute error floored at 1e-5, mixed by r, over beta,
    times the advantage clip; and its gradient in `live`, the weights held."""
    axes = tuple(range(1, samples.ndim))
    positive = beta * live + (1 - beta) * old
    negative = (1 + beta) * old - beta * live
    count = np.prod(samples.shape[1:])
    terms, gradient = [], np.zeros_like(live)
    for estimate, sign, share in ((positive, beta, r), (negative, -beta, 1 - r)):
        error = estimate - samples
        weight = np.maximum(np.mean(np.abs(error), axis=axes), 1e-5)
        terms.append(share * np.mean(np.square(error), axis=axes) / weight)
        gradient += (share / weight).reshape(-1, *(1,) * len(axes)) * 2 * error * sign / count
    loss = (terms[0] + terms[1]) / beta * scale
    return loss, gradient / beta * scale


@pytest.mark.parametrize("beta", [1.0, 0.1])
def test_the_policy_loss_is_the_released_trainers(beta):
    """Each sample's policy loss at one time, and its gradient in the live estimate, which reads each
    implicit policy's detached weight as a constant, are the release's arithmetic within float64
    rounding: at beta 1 (the single-reward default) and 0.1 (OCR's and the multi-reward run's)."""
    rng = np.random.default_rng(5)
    samples, live, old = (rng.normal(size=(ROWS, *SHAPE)) for _ in range(3))
    r = rng.uniform(size=ROWS)

    def summed(live):
        return jnp.sum(released_policy_loss(samples, live, old, r, beta=beta, scale=5.0))

    with jax.enable_x64(new_val=True):
        found = np.asarray(released_policy_loss(samples, live, old, r, beta=beta, scale=5.0))
        gradient = np.asarray(jax.grad(summed)(live))
        roundings = chain_roundings(jax.make_jaxpr(jax.value_and_grad(summed))(live))
    loss, expected = transcribed_policy_loss(samples, live, old, r, beta, 5.0)
    assert_computes_the_oracle(found, loss, f"the policy loss at beta {beta}", roundings=roundings)
    assert_computes_the_oracle(gradient, expected, f"its gradient at beta {beta}", roundings=roundings)


def test_optimality_is_the_clipped_advantage_mapped_onto_the_unit_interval():
    """r is the advantage clipped to the clip and mapped linearly onto [0, 1], a zero advantage at
    one half (train_nft_sd3.py:910-929), within float64 rounding of the release's arithmetic."""
    advantages = np.linspace(-9.0, 9.0, 37)
    with jax.enable_x64(new_val=True):
        found = np.asarray(objective().optimality(jnp.asarray(advantages)))
    expected = np.clip(np.clip(advantages, -5, 5) / 5 / 2 + 0.5, 0, 1)
    assert_computes_the_oracle(found, expected, "optimality", roundings=4)


def objective(dtype=jnp.float32, **options):
    from dew.nn.backbones.dit import SimpleDiT

    process = Process(FlowMatchingScheduler(shift=2), FlowMatchPredictionTransform())
    inputs = InputSpec(Field("image", (4, 4, 1)), {
        "textcontext": Condition(CharTable.from_pretrained(tokens=3, features=4))})
    model = SimpleDiT(output_channels=1, patch_size=2, emb_features=8, num_layers=1, num_heads=2,
                      mlp_ratio=2, dtype=dtype)
    return DiffusionNFTObjective(model, process, inputs, steps=5, **{"kl": 0.0, **options})


def nudged(variables, key):
    """`variables` with every weight moved off its start, so the DiT's zeroed output moves with them."""
    leaves, tree = jax.tree.flatten(variables["params"])
    keys = jax.random.split(key, len(leaves))
    return {**variables, "params": jax.tree.unflatten(tree, [
        leaf + 0.1 * jax.random.normal(key, leaf.shape, leaf.dtype)
        for leaf, key in zip(leaves, keys, strict=True)])}


def fitted(nft, reward, prompts, **rollout):
    """A rollout over `nft` and the state before and after one update a fit takes from its batch."""
    from dew.data.dataset import Loader

    sampled = NFTRollout(nft, reward, **rollout)
    trainer = Trainer(nft, optax.sgd(1e-2), key=jax.random.key(21), rollout=sampled)
    initial = trainer.place()[0]
    count = len(next(iter(prompts.values())))
    data = Loader(train=lambda partition: itertools.repeat(prompts), val=None, records=count, batch=count)
    return sampled, initial, trainer.fit(data, steps=1, log_every=1)


PROMPTS = ["red", "blue", "gold", "teal", "pink", "gray", "jade", "plum"]
"""As many prompts as the data axis's eight devices take."""


def reward(images, context):
    """Each sample's closeness of its mean pixel to 0.2."""
    return -np.square(images.mean(axis=(1, 2, 3)) - 0.2)


def test_a_rollout_samples_the_old_policy_and_normalizes_by_the_batchs_spread():
    """The rollout samples with the old policy, the EMA slot, not the live weights: after an update
    that leaves the old policy most of itself, its samples are those the old weights draw, bit for
    bit. Each sample's advantage is its reward less its group's mean over the standard deviation of
    every reward in the batch plus 1e-4 (stat_tracking.py:14-30 under global_std), and its times are
    the sampler grid's first `train_steps`."""
    start = objective().init(jax.random.key(20))
    nft = objective(variables=nudged(start, jax.random.key(22)), old_decay=optax.constant_schedule(0.9))
    prompts = nft.inputs.tokenize(PROMPTS)
    rollout, initial, final = fitted(nft, reward, prompts, groups=3, steps=5, train_steps=3)
    batch = rollout(initial, prompts, jax.random.key(23))
    rewards = np.asarray(batch["rewards"], np.float64)
    grouped = rewards.reshape(8, 3)
    expected = (grouped - grouped.mean(axis=1, keepdims=True)) / (rewards.std() + 1e-4)
    np.testing.assert_array_equal(batch["advantages"], expected.reshape(-1).astype(np.float32))
    np.testing.assert_array_equal(batch["timesteps"], np.broadcast_to(nft.process.times(5)[:3], (24, 3)))

    assert int(final.updates) == 1
    assert not np.array_equal(*(np.asarray(jax.tree.leaves(state.variables["params"])[0])
                                for state in (initial, final)))
    old_weights = final.replace(variables=final.averaged)
    np.testing.assert_array_equal(rollout(final, prompts, jax.random.key(23))["latents"],
                                  rollout(old_weights, prompts, jax.random.key(23))["latents"])


def test_the_released_decay_is_its_trainers_schedule():
    """The old policy's decay at the update after s completed ones is min(0.001 (s + 1), 0.5), the
    released trainer's decay_type 1 at the gradient steps taken (train_nft_sd3.py:169-189 and
    1033-1038), within float64 rounding of that schedule over the first 2000 updates."""
    updates = np.arange(2000, dtype=np.float64)
    with jax.enable_x64(new_val=True):
        found = np.asarray(released_old_decay(updates))
    expected = np.minimum(0.001 * (updates + 1), 0.5)
    assert_computes_the_oracle(found, expected, "the old policy's decay", roundings=3)


def test_the_old_policy_follows_the_trained_one_at_its_decay():
    """At an update the old policy, the EMA slot, keeps its decay's share of itself and takes the rest
    from the trained weights, bit for bit at a decay of one half, whose arithmetic rounds once however
    it fuses."""
    from dew.averaging import ema_update

    nft = objective(variables=nudged(objective().init(jax.random.key(20)), jax.random.key(22)),
                    old_decay=optax.constant_schedule(0.5))
    _, initial, final = fitted(nft, lambda images, context: images.mean(axis=(1, 2, 3)),
                               nft.inputs.tokenize(PROMPTS), groups=2, steps=5, train_steps=2)
    jax.tree.map(np.testing.assert_array_equal, final.ema, ema_update(initial.ema, final.variables, 0.5))


@pytest.mark.parametrize("asked", [{"beta": 0.0}, {"adv_clip_max": 0.0}, {"kl": -1.0}, {"kl": 1e-4}],
                         ids=["beta", "clip", "kl", "kl without a base"])
def test_a_degenerate_strength_clip_or_regularizer_is_refused(asked):
    """A beta or a clip that is not positive, a negative kl weight, or a positive one with no starting
    tree to read the base model from, is refused as the objective is made."""
    with pytest.raises(ValueError):
        objective(**asked)


def test_the_loss_corrupts_by_its_process_own_noise_levels():
    """A DiT of zero weights predicts zero velocity, so its clean estimate is the corrupted sample
    itself and the live, old and implicit policies coincide: each sample's loss at time t is then the
    clip over beta times the mean squared over the mean absolute error of alpha(t) x + sigma(t) eps
    against x, alpha and sigma the process's own rates at t (a noise level of 2t / (1 + t) under its
    shift of 2): within float64 rounding of that closed form over the noise the step draws."""
    from dew.objectives.base import Step

    nft = objective(jnp.float64, beta=0.5)
    with jax.enable_x64(new_val=True):
        # Every weight zero, so the DiT's velocity is exactly zero.
        variables = jax.tree.map(jnp.zeros_like, nft.init(jax.random.key(30)))
    rng = np.random.default_rng(31)
    samples = rng.normal(size=(8, 4, 4, 1))
    times = np.broadcast_to(np.asarray([1.0, 0.75, 0.5]), (8, 3))
    batch = {"latents": samples, "timesteps": times, "advantages": rng.normal(size=8),
             **nft.inputs.tokenize(["red"] * 8)}
    step = Step(jnp.asarray(0), jax.random.key(32), {"params": variables["params"]})
    with jax.enable_x64(new_val=True):
        wide = widened(variables)
        stats, _ = nft.loss(wide, batch, step)
        found = float(stats.total / stats.mass)
        noise = np.asarray(step.sample("noise", jax.random.normal, (3, 4, 4, 1), dtype=jnp.float32, rows=8),
                           np.float64)
        roundings = chain_roundings(jax.make_jaxpr(lambda v: nft.loss(v, batch, step)[0].total)(wide))
        # The process's own rates at each time, a noise level of 2t / (1 + t) under a shift of 2, as it
        # computes them.
        alpha, sigma = (np.stack([np.asarray(rate).reshape(-1) for rate in rates], 1) for rates in zip(
            *(nft.process.rates(jnp.asarray(column), like=samples) for column in times.T), strict=True))
    assert not np.array_equal(sigma, times)
    shape = (8, 3, 1, 1, 1)
    error = alpha.reshape(shape) * samples[:, None] + sigma.reshape(shape) * noise - samples[:, None]
    per = np.mean(np.square(error), axis=(2, 3, 4)) / np.mean(np.abs(error), axis=(2, 3, 4))
    expected = np.mean(nft.adv_clip_max / nft.beta * per)
    assert_computes_the_oracle(np.asarray([found]), np.asarray([expected]), "the NFT loss",
                               roundings=roundings)
