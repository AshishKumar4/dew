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
from reference_error import assert_computes_the_oracle, chain_roundings

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


def objective(**options):
    from dew.nn.backbones.dit import SimpleDiT

    process = Process(FlowMatchingScheduler(shift=2), FlowMatchPredictionTransform())
    inputs = InputSpec(Field("image", (4, 4, 1)), {
        "textcontext": Condition(CharTable.from_pretrained(tokens=3, features=4))})
    model = SimpleDiT(output_channels=1, patch_size=2, emb_features=8, num_layers=1, num_heads=2,
                      mlp_ratio=2)
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


PROMPTS = ["red", "blue"]


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
    grouped = rewards.reshape(2, 3)
    expected = (grouped - grouped.mean(axis=1, keepdims=True)) / (rewards.std() + 1e-4)
    np.testing.assert_array_equal(batch["advantages"], expected.reshape(-1).astype(np.float32))
    np.testing.assert_array_equal(batch["timesteps"], np.broadcast_to(nft.process.times(5)[:3], (6, 3)))

    assert int(final.updates) == 1
    assert not np.array_equal(*(np.asarray(jax.tree.leaves(state.variables["params"])[0])
                                for state in (initial, final)))
    old_weights = final.replace(variables=final.averaged)
    np.testing.assert_array_equal(rollout(final, prompts, jax.random.key(23))["latents"],
                                  rollout(old_weights, prompts, jax.random.key(23))["latents"])


def test_the_old_policy_follows_the_trained_one_at_the_released_decay():
    """At an update the old policy keeps min(0.001 s, 0.5) of itself, s the gradient steps taken once
    it is made, and takes the rest from the trained weights (train_nft_sd3.py:169-189 and 1033-1038,
    decay_type 1): the EMA slot's update at the released schedule, bit for bit, the base model's
    velocity weighed in at the release's kl."""
    from dew.averaging import ema_update

    nft = objective(variables=nudged(objective().init(jax.random.key(20)), jax.random.key(22)), kl=1e-4)
    _, initial, final = fitted(nft, lambda images, context: images.mean(axis=(1, 2, 3)),
                               nft.inputs.tokenize(PROMPTS), groups=2, steps=5, train_steps=2)
    expected = ema_update(initial.ema, final.variables, released_old_decay(initial.updates))
    jax.tree.map(np.testing.assert_array_equal, final.ema, expected)


@pytest.mark.parametrize("asked", [{"beta": 0.0}, {"adv_clip_max": 0.0}, {"kl": -1.0}, {"kl": 1e-4}],
                         ids=["beta", "clip", "kl", "kl without a base"])
def test_a_degenerate_strength_clip_or_regularizer_is_refused(asked):
    """A beta or a clip that is not positive, a negative kl weight, or a positive one with no starting
    tree to read the base model from, is refused as the objective is made."""
    with pytest.raises(ValueError):
        objective(**asked)
