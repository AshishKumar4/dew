"""Dreamer 4's world model at small scale (`dew.objectives.world_model`): shortcut forcing's loss against
Eqs. 4, 6-8 of arXiv 2509.24527 transcribed in NumPy float64, its step sampling's grid, and the
dynamics model's block-causal reading of time."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.objectives.world_model import Dynamics, sampled_steps, shortcut_forcing_loss


def test_shortcut_forcing_is_the_papers_bootstrap():
    """Each frame's loss is its x-prediction's squared error where d is the smallest step, and else
    (1 - tau)^2 times the squared error of its velocity against the mean of the two half-step
    velocities the predictor takes itself, every frame times 0.9 tau + 0.1, within float64 rounding
    of Eq. 7 run in NumPy over a predictor with a closed form."""
    rng = np.random.default_rng(0)
    clean, noise = rng.normal(size=(3, 5, 4, 2)), rng.normal(size=(3, 5, 4, 2))
    d = 1 / 2.0 ** rng.integers(0, 4, (3, 5))
    tau = np.floor(rng.uniform(size=(3, 5)) / d) * d
    weights = rng.normal(size=(4, 2))

    def predictor(xp):
        def f(noisy, tau, d):
            shape = (*tau.shape, 1, 1)
            return xp.tanh(noisy * weights + tau.reshape(shape) - 2 * d.reshape(shape))
        return f

    with jax.enable_x64(new_val=True):
        found = np.asarray(shortcut_forcing_loss(predictor(jnp), clean, noise, tau, d, 1 / 8))
        roundings = chain_roundings(jax.make_jaxpr(
            lambda *a: shortcut_forcing_loss(predictor(jnp), *a, 1 / 8))(clean, noise, tau, d))
    f = predictor(np)
    expected = np.zeros(tau.shape)
    for b in range(3):
        for t in range(5):
            z1, z0, level, step = clean[b, t], noise[b, t], tau[b, t], d[b, t]
            noisy = (1 - level) * z0 + level * z1
            one = np.asarray([[level]]), np.asarray([[step]])
            predicted = f(noisy[None, None], *one)[0, 0]
            if step <= 1 / 8:
                loss = np.sum((predicted - z1) ** 2)
            else:
                half = np.asarray([[step / 2]])
                first = (f(noisy[None, None], one[0], half)[0, 0] - noisy) / (1 - level)
                between = noisy + first * step / 2
                later = f(between[None, None], one[0] + step / 2, half)[0, 0]
                second = (later - between) / (1 - level - step / 2)
                velocity = (predicted - noisy) / (1 - level)
                loss = (1 - level) ** 2 * np.sum((velocity - (first + second) / 2) ** 2)
            expected[b, t] = (0.9 * level + 0.1) * loss
    assert_computes_the_oracle(found, expected, "shortcut forcing", roundings=roundings)


def test_step_sizes_are_powers_of_two_and_levels_lie_on_their_grid():
    """Eq. 4: every step size is the reciprocal of a power of two up to K_max, every one is drawn, and
    each signal level is a multiple of its frame's step size below 1."""
    tau, d = (np.asarray(x) for x in sampled_steps(jax.random.key(1), (4096,), 64))
    exponents = -np.log2(d)
    np.testing.assert_array_equal(exponents, np.round(exponents))
    assert set(exponents.astype(int)) == set(range(7))
    np.testing.assert_array_equal(tau / d, np.round(tau / d))
    assert (tau >= 0).all() and (tau <= 1 - d).all()


def test_the_dynamics_reads_no_frame_after_its_own():
    """A frame's prediction is unchanged, bit for bit, by any change to a later frame's latents, level
    or step: the transformer attends block-causally over time."""
    model = Dynamics(latents=8, bottleneck=4, spatial=4, width=32, depth=2, heads=2, levels=4, frames=6)
    rng = np.random.default_rng(2)
    noisy = jnp.asarray(rng.normal(size=(2, 6, 8, 4)), jnp.float32)
    tau, d = jnp.full((2, 6), 0.5), jnp.full((2, 6), 0.25)
    params = model.init(jax.random.key(3), noisy, tau, d)
    changed = noisy.at[:, 4:].set(jnp.asarray(rng.normal(size=(2, 2, 8, 4)), jnp.float32))
    apply = jax.jit(model.apply)
    later = apply(params, changed, tau.at[:, 4:].set(0.125), d.at[:, 4:].set(0.5))
    np.testing.assert_array_equal(apply(params, noisy, tau, d)[:, :4], later[:, :4])


def test_a_labeled_model_trains_its_actions_into_its_predictions():
    """A dynamics model of two actions a step draws its action projection with the rest of its tree, and
    its loss over labeled video reaches that projection's weights: the actions move the predictions."""
    from dew.objectives.base import Step
    from dew.objectives.world_model import CausalTokenizer, WorldModelObjective

    tokenizer = CausalTokenizer((8, 8, 1), patch=4, width=16, depth=1, heads=2, latents=4, bottleneck=4)
    tokens = tokenizer.init(jax.random.key(4), jnp.zeros((1, 8, 8, 1)))
    dynamics = Dynamics(latents=4, bottleneck=4, spatial=2, action_size=2, width=16, depth=1, heads=2,
                        levels=3, frames=4)
    objective = WorldModelObjective(dynamics, tokenizer, tokens)
    variables = objective.fresh_variables(jax.random.key(5), None)
    rng = np.random.default_rng(6)
    batch = {"video": jnp.asarray(rng.uniform(-1, 1, (2, 4, 8, 8, 1)), jnp.float32),
             "actions": jnp.asarray(rng.normal(size=(2, 4, 2)), jnp.float32)}

    def loss(params):
        step = Step(jnp.asarray(0), jax.random.key(7), None)
        stats, _ = objective.loss({**variables, "params": params}, batch, step)
        return stats.total / stats.mass

    gradient = jax.grad(loss)(variables["params"])
    assert np.any(np.asarray(gradient["action_projection"]["kernel"]))


def test_a_clip_of_uint8_pixels_trains_as_the_frames_in_minus_one_to_one_it_encodes():
    """The data workers write clips as uint8 pixels (`presets.clips`), which the objectives read as the
    frames they encode, (pixel - 127.5) / 127.5 in [-1, 1], and floats as they are given: every pixel
    value is that frame, and a tokenizer's loss over a clip of pixels is its loss over those frames."""
    from dew.objectives.base import Step
    from dew.objectives.world_model import CausalTokenizer, TokenizerObjective, unit_frames

    every = np.arange(256, dtype=np.uint8)
    encoded = ((every.astype(np.float64) - 127.5) / 127.5).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(unit_frames(every)), encoded)
    np.testing.assert_array_equal(np.asarray(unit_frames(encoded)), encoded)

    pixels = np.random.default_rng(8).integers(0, 256, (2, 4, 8, 8, 1), dtype=np.uint8)
    frames = ((pixels.astype(np.float64) - 127.5) / 127.5).astype(np.float32)
    objective = TokenizerObjective(CausalTokenizer((8, 8, 1), patch=4, width=16, depth=1, heads=2,
                                                   latents=4, bottleneck=4))
    variables = objective.fresh_variables(jax.random.key(9), None)
    step = Step(jnp.asarray(0), jax.random.key(10), None)
    stats = [objective.loss(variables, {"video": video}, step)[0] for video in (pixels, frames)]
    np.testing.assert_array_equal(np.asarray(stats[0].total), np.asarray(stats[1].total))


def _tiny_models():
    """A tokenizer and a dynamics model at the tests' scale, the tokenizer's tree drawn."""
    from dew.objectives.world_model import CausalTokenizer

    tokenizer = CausalTokenizer((8, 8, 1), patch=4, width=16, depth=1, heads=2, latents=4, bottleneck=4)
    dynamics = Dynamics(latents=4, bottleneck=4, spatial=2, width=16, depth=1, heads=2, levels=3, frames=4)
    return tokenizer, tokenizer.init(jax.random.key(4), jnp.zeros((1, 8, 8, 1))), dynamics


def _objectives():
    """The tokenizer's objective and the world model's over it, each with its variables drawn."""
    from dew.objectives.world_model import TokenizerObjective, WorldModelObjective

    tokenizer, tokens, dynamics = _tiny_models()
    objectives = (TokenizerObjective(tokenizer), WorldModelObjective(dynamics, tokenizer, tokens))
    return [(objective, objective.fresh_variables(jax.random.key(5), None)) for objective in objectives]


def test_a_padded_batch_scores_its_real_clips_alone():
    """An evaluation's last batch, three clips padded to four (`VALID_ROWS`, as `FilledBatch` pads it),
    counts its three real clips alone in each objective's loss (`Objective.row_mean`): the padding's
    content changes nothing, its mass is the real clips', and a batch of padding alone holds no mass and
    moves no weight (row 4146: both losses counted the padding, and a batch of it kept a positive mass and a
    gradient)."""
    from dew.batch import VALID_ROWS
    from dew.objectives.base import Step

    video = jnp.asarray(np.random.default_rng(11).uniform(-1, 1, (4, 4, 8, 8, 1)), jnp.float32)
    real = jnp.asarray([True, True, True, False])
    padded = [{"video": clips, VALID_ROWS: real} for clips in (video, video.at[3].set(video[0]))]
    repeats = {"video": video, VALID_ROWS: jnp.zeros((4,), bool)}
    step = Step(jnp.asarray(0), jax.random.key(12), None, training=False)
    for objective, variables in _objectives():
        (other, _), (repeat, _) = (objective.loss(variables, batch, step) for batch in padded)
        three, _ = objective.loss(variables, {"video": video[:3]}, step)
        assert (float(other.total), float(other.mass)) == (float(repeat.total), float(repeat.mass))
        assert float(repeat.mass) == float(three.mass)

        def total(params, objective=objective, variables=variables):
            return objective.loss({**variables, "params": params}, repeats, step)[0].total

        empty, _ = objective.loss(variables, repeats, step)
        assert (float(empty.total), float(empty.mass)) == (0.0, 0.0)
        gradient = jax.grad(total)(variables["params"])
        assert not any(np.any(np.asarray(leaf)) for leaf in jax.tree.leaves(gradient))


def test_an_updates_corruption_is_each_clips_own_however_the_update_is_split():
    """Each clip's corruption (the tokenizer's mask rate and patch mask, the world model's step size,
    signal level and noise) is drawn from the update's example key at the clip's place (`Step.sample`),
    what it derives kept at its boundaries, so an update of four clips read whole or as two microbatches of
    two, each attempt's key its own, records the same draws and boundaries for every clip (row 4147: they
    were drawn batch-wide from each attempt's key)."""
    from dew.objectives.base import Recorder, Step

    video = jnp.asarray(np.random.default_rng(13).uniform(-1, 1, (4, 4, 8, 8, 1)), jnp.float32)
    update = jax.random.key(14)

    def recorded(objective, variables, parts):
        recorder = Recorder()
        for rows, attempt in parts:
            step = Step(jnp.asarray(0), jax.random.key(attempt), None, example_key=update,
                        example_index=jnp.asarray(rows, jnp.int32), example_count=4, recorder=recorder)
            objective.loss(variables, {"video": video[np.asarray(rows)]}, step)
        return recorder.tape(4)

    for objective, variables in _objectives():
        whole = recorded(objective, variables, [([0, 1, 2, 3], 15)])
        split = recorded(objective, variables, [([0, 1], 16), ([2, 3], 17)])
        assert whole and set(whole) == set(split)
        for name, tape in whole.items():
            np.testing.assert_array_equal(tape, split[name], err_msg=name)


def test_a_tokenizers_patches_are_each_frames_row_major_pixels_whatever_its_leading_axes():
    """The tokenizer patches each frame of a clip batch `[B, T, H, W, C]` as `nn.scan_orders.patchify` does
    one image (row 4150: it kept its own reshape): pixel (r, c, k) at patch (r // p) * (W / p) + c // p,
    place ((r % p) * p + c % p) * C + k, and unpatching returns the frames."""
    from dew.objectives.world_model import CausalTokenizer

    tokenizer = CausalTokenizer((8, 12, 2), patch=4)
    frames = jnp.arange(2 * 3 * 8 * 12 * 2, dtype=jnp.float32).reshape(2, 3, 8, 12, 2)
    patches = np.asarray(tokenizer._patchify(frames))
    assert patches.shape == (2, 3, 6, 32)
    r, c, k = np.meshgrid(np.arange(8), np.arange(12), np.arange(2), indexing="ij")
    placed = np.zeros_like(patches)
    placed[:, :, (r // 4) * 3 + c // 4, ((r % 4) * 4 + c % 4) * 2 + k] = np.asarray(frames)
    np.testing.assert_array_equal(patches, placed)
    np.testing.assert_array_equal(np.asarray(tokenizer._unpatchify(jnp.asarray(patches))), np.asarray(frames))


def test_a_world_models_tokenizer_rides_in_its_state_and_its_loss_reads_it_there():
    """The world model's frozen tokenizer is state (row 4148: it lived on the objective, a compiled constant
    outside the tree placement and checkpoints share): the drawn tree holds the given weights beside the
    dynamics' under its own collection, which the optimizer never sees, and the loss encodes with the
    weights the tree holds, other weights giving another loss."""
    from dew.objectives.base import Step
    from dew.objectives.world_model import TOKENIZER, WorldModelObjective

    tokenizer, tokens, dynamics = _tiny_models()
    objective = WorldModelObjective(dynamics, tokenizer, tokens)
    variables = objective.init(jax.random.key(5))
    assert jax.tree.structure(variables[TOKENIZER]) == jax.tree.structure(tokens)
    for held, given in zip(jax.tree.leaves(variables[TOKENIZER]), jax.tree.leaves(tokens), strict=True):
        np.testing.assert_array_equal(np.asarray(held), np.asarray(given))
    assert TOKENIZER not in variables["params"]
    video = jnp.asarray(np.random.default_rng(18).uniform(-1, 1, (2, 4, 8, 8, 1)), jnp.float32)
    step = Step(jnp.asarray(0), jax.random.key(19), None)
    other = tokenizer.init(jax.random.key(20), jnp.zeros((1, 8, 8, 1)))
    held, _ = objective.loss(variables, {"video": video}, step)
    swapped, _ = objective.loss({**variables, TOKENIZER: other}, {"video": video}, step)
    assert float(held.total) != float(swapped.total)
