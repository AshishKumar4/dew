"""I-JEPA and V-JEPA: masking, shapes, the objective, collapse telemetry, and
the objective through the general trainer."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import dataset
from recording import RecordingTracker
from reference_error import assert_as_exact_as_the_reference
from steady_state import steady_state

from dew.artifacts import Representations
from dew.inputs import Field
from dew.nn.backbones.jepa import JepaPredictor
from dew.objectives.base import Step, merge, part
from dew.objectives.jepa import (
    JepaEncoder,
    JepaObjective,
    JepaVideoEncoder,
    KnnProbe,
    LinearProbe,
    MultiBlockMask,
    linear_probe_accuracy,
    normalize_targets,
    representation_health,
)
from dew.training import Layout, MeshSpec, Trainer

RES = 32
PATCH = 4
GRID = (RES // PATCH, RES // PATCH)
FRAMES = 3


@pytest.fixture
def mask():
    return MultiBlockMask.for_grid(GRID, num_targets=4, scale=(0.15, 0.2))


def make_encoder(**kwargs):
    return JepaEncoder(patch_size=PATCH, emb_features=32, num_layers=2, num_heads=2,
                       mlp_ratio=2, **kwargs)


def make_predictor(**kwargs):
    return JepaPredictor(grid=GRID, emb_features=32, predictor_features=16,
                         num_layers=1, num_heads=2, mlp_ratio=2, **kwargs)


def make_objective(mask, **kwargs):
    return JepaObjective(make_encoder(), make_predictor(), mask,
                         sample=Field("image", (RES, RES, 3)), **kwargs)


def step_with(params, key=7, index=0):
    """What the trainer hands the objective: the EMA copy is the params themselves."""
    return Step(step=jnp.asarray(index), key=jax.random.PRNGKey(key), ema=params)


def images(seed=0, batch=4):
    return jnp.asarray(np.random.RandomState(seed).uniform(0, 255, (batch, RES, RES, 3)))


def videos(seed=0, batch=2):
    return jnp.asarray(
        np.random.RandomState(seed).uniform(0, 255, (batch, FRAMES, RES, RES, 3)))


# --- masking ---------------------------------------------------------------

def test_blocks_and_context_follow_the_documented_distribution(mask):
    """The static-shape multi-block draw: each block's shape uniform over
    `block_shapes`, its corner uniform over the corners where that shape
    fits, and the context a uniform subset of the targets' complement. Over
    4096 rows, Pearson's chi-square of the shapes and of each shape's
    corners, and of the context's inclusion rate per complement size, at a
    one-in-a-million false alarm."""
    from scipy import stats

    rows = 4096
    context, targets = (np.asarray(part) for part in mask.sample(jax.random.PRNGKey(5), rows))
    blocks = targets.reshape(-1, mask.block_area)
    top, left = np.divmod(blocks.min(axis=1), GRID[1])
    height = blocks.max(axis=1) // GRID[1] - top + 1
    shapes = [mask.block_shapes.index((int(h), mask.block_area // int(h))) for h in height]

    def assert_uniform(counts, what):
        p = stats.chisquare(counts).pvalue
        assert p > 1e-6, f"{what} is not uniform (p = {p:.1e})"

    assert_uniform(np.bincount(shapes, minlength=len(mask.block_shapes)), "the block shape")
    for index, (h, w) in enumerate(mask.block_shapes):
        chosen = np.asarray(shapes) == index
        corners = top[chosen] * (GRID[1] - w + 1) + left[chosen]
        assert_uniform(np.bincount(corners, minlength=(GRID[0] - h + 1) * (GRID[1] - w + 1)),
                       f"the {h}x{w} block's corner")
    covered = np.zeros((rows, mask.num_patches), bool)
    covered[np.arange(rows)[:, None], targets.reshape(rows, -1)] = True
    kept = np.zeros_like(covered)
    kept[np.arange(rows)[:, None], context] = True
    assert not (kept & covered).any()
    # Among rows leaving the same number of tokens free, every free position
    # is kept at the rate num_context / free, wherever the blocks fell.
    free = (~covered).sum(axis=1)
    size = np.bincount(free).argmax()
    same = free == size
    chance = mask.num_context / size
    offered, taken = (~covered[same]).sum(axis=0), kept[same].sum(axis=0)
    seen = offered > 0
    statistic = np.sum((taken - offered * chance)[seen] ** 2 / (offered * chance * (1 - chance))[seen])
    assert stats.chi2.sf(statistic, seen.sum() - 1) > 1e-6, "the context favours some free positions"


@pytest.mark.parametrize("scan_order", ["raster", "hilbert", "zigzag"])
def test_every_target_is_a_rectangle_of_the_grid_in_the_encoders_scan_order(scan_order):
    """The encoder sequences its patches in `scan_order`: a target block's
    tokens, sent back through that order, cover a whole h x w rectangle of
    the grid, and the context, sorted in the sequence, sees none of them."""
    from dew.nn.dit import scan_indices

    mask = MultiBlockMask.for_grid(GRID, num_targets=4, scale=(0.15, 0.2), scan_order=scan_order)
    order = scan_indices(scan_order, *GRID)
    grid_of = np.arange(mask.num_patches) if order is None else order
    context, targets = (np.asarray(part) for part in mask.sample(jax.random.PRNGKey(3), 8))
    for row, blocks in enumerate(targets):
        for block in blocks:
            assert np.all(np.diff(block) > 0), "a target block is not sorted in the sequence"
            rows, columns = np.divmod(grid_of[block], GRID[1])
            extent = (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
            assert extent == mask.block_area, "a target block is not a rectangle of the grid"
        assert np.all(np.diff(context[row]) > 0)
        assert not set(context[row]) & set(blocks.reshape(-1))


# --- the encoders and the predictor ----------------------------------------

def test_the_token_index_selects_tokens_with_their_positions(rng):
    """`token_idx` is a set of grid positions, and the encoder attends over
    whatever set it is given: asking for every token in a shuffled order gives
    the full encoding shuffled the same way, which holds only if each gathered
    token carries the sincos signal of its own grid position and no rotation
    by sequence index is applied."""
    encoder = make_encoder()
    x = images()
    every = jnp.arange(GRID[0] * GRID[1], dtype=jnp.int32)
    shuffled = jnp.stack([jax.random.permutation(jax.random.fold_in(rng, i), every)
                          for i in range(x.shape[0])])
    params = jax.tree.map(lambda p: p + 0.05, encoder.init(rng, x, shuffled))

    full = encoder.apply(params, x)
    subset = encoder.apply(params, x, shuffled)
    assert float(jnp.max(jnp.abs(full))) > 0.1
    # 1.5e-6 observed: the softmax sums in another order.
    assert jnp.allclose(subset, jnp.take_along_axis(full, shuffled[..., None], axis=1), atol=1e-5)


def test_the_video_token_index_selects_the_same_tubelet_in_every_frame(rng):
    """A video encoder's `token_idx` names patch positions shared by every
    frame, so the shuffled full encoding is the full encoding shuffled within
    each frame."""
    encoder = JepaVideoEncoder(patch_size=PATCH, emb_features=32, num_layers=1,
                               num_heads=2, mlp_ratio=2)
    x = videos()
    every = jnp.arange(GRID[0] * GRID[1], dtype=jnp.int32)
    shuffled = jnp.stack([jax.random.permutation(jax.random.fold_in(rng, i), every)
                          for i in range(x.shape[0])])
    params = jax.tree.map(lambda p: p + 0.05, encoder.init(rng, x, shuffled))

    full = encoder.apply(params, x)
    subset = encoder.apply(params, x, shuffled)
    expected = jnp.take_along_axis(full, shuffled[:, None, :, None], axis=2)
    assert jnp.allclose(subset, expected, atol=1e-5)


@pytest.mark.parametrize("factorized", [False, True], ids=["image", "video"])
def test_the_predictor_predicts_each_target_by_its_position(mask, rng, factorized):
    """The targets are mask tokens told their grid position, so the order they
    are asked in cannot matter: a shuffled target index gives the predictions
    shuffled the same way."""
    predictor = make_predictor(factorized=factorized)
    context_idx, target_idx = mask.sample(rng, 2)
    frames = (FRAMES,) if factorized else ()
    context = jax.random.normal(rng, (2, *frames, mask.num_context, 32))
    block = target_idx[:, 0]
    perm = jnp.stack([jax.random.permutation(jax.random.fold_in(rng, i), mask.block_area)
                      for i in range(2)])
    shuffled = jnp.take_along_axis(block, perm, axis=1)
    params = jax.tree.map(lambda p: p + 0.05, predictor.init(rng, context, context_idx, block))

    ordered = predictor.apply(params, context, context_idx, block)
    permuted = predictor.apply(params, context, context_idx, shuffled)
    assert float(jnp.max(jnp.abs(ordered))) > 0.1
    gather = perm[:, None, :, None] if factorized else perm[..., None]
    assert jnp.allclose(permuted, jnp.take_along_axis(ordered, gather, axis=-2), atol=1e-5)


# --- the objective ---------------------------------------------------------

def test_compiled_embeddings_follow_the_substituted_encoder(mask):
    objective = make_objective(mask)
    variables = objective.init(jax.random.key(0))
    batch = {"image": images(batch=2), "label": jnp.array([0, 1])}
    step = step_with(variables)
    before = objective.evaluate(variables, batch, step).features
    compiled = objective._embed
    replacement = objective.encoder.clone(norm_epsilon=1.)
    objective.substitute([replacement, objective.predictor])
    after = objective.evaluate(variables, batch, step).features
    fresh = make_objective(mask)
    fresh.substitute([replacement, objective.predictor])
    expected = fresh.evaluate(variables, batch, step).features
    assert not np.allclose(before, expected)
    np.testing.assert_array_equal(after, expected)
    assert objective._embed is not compiled
    assert objective._embed is objective._embed


def test_video_objective_trains(mask, rng):
    objective = JepaObjective(
        JepaVideoEncoder(patch_size=PATCH, emb_features=32, num_layers=1,
                         num_heads=2, mlp_ratio=2),
        make_predictor(factorized=True), mask,
        sample=Field("video", (FRAMES, RES, RES, 3)))
    params = objective.init(rng)
    batch = {"video": videos()}

    def loss_of(p):
        return objective.scalar_loss(p, batch, step_with(params))[0]

    initial = float(loss_of(params))
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(params)
    for _ in range(15):
        grads = jax.grad(loss_of)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    assert float(loss_of(params)) < initial * 0.9


def solid_colour_images(rs, n):
    """One random colour per image, so a target block is fully determined by
    anything else in the same image and by nothing in any other image."""
    colours = rs.uniform(20, 235, (n, 1, 1, 3))
    return jnp.asarray(np.clip(colours + rs.uniform(-15, 15, (n, RES, RES, 3)), 0, 255))


def context_ablation(objective, params, ema, data, mask, rng):
    """Prediction error from the true context vs. from another image's context."""
    n, blocks = data.shape[0], mask.num_targets
    context_idx, target_idx = mask.sample(rng, n)
    full = normalize_targets(objective.encode(part(ema, "context_encoder"), data))
    targets = jnp.take_along_axis(
        full[:, None], target_idx.reshape(n, blocks, -1, 1), axis=-2)
    context = objective.encode(part(params, "context_encoder"), data, context_idx)

    def error(ctx):
        predictions = objective.predictor.apply(
            part(params, "predictor"),
            jnp.repeat(ctx, blocks, axis=0),
            jnp.repeat(context_idx, blocks, axis=0),
            target_idx.reshape(n * blocks, -1),
        ).reshape(targets.shape)
        return float(jnp.mean((predictions - targets) ** 2))

    return error(context), error(jnp.roll(context, 1, axis=0))


def test_training_makes_the_prediction_depend_on_the_context(mask):
    """The loss can be driven down by predicting the average target while
    ignoring the context entirely, which would teach the encoder nothing.
    Swapping in another image's context must cost real accuracy."""
    rs = np.random.RandomState(0)
    train_x, test_x = solid_colour_images(rs, 128), solid_colour_images(rs, 32)
    normalized_test = (test_x - 127.5) / 127.5

    trainer = make_jepa_trainer(mask, learning_rate=3e-3,
                                momentum=(0.9, 0.99), momentum_steps=150)
    objective = trainer.objective
    initial = trainer.initial_state()
    before = context_ablation(objective, initial.variables, initial.ema,
                              normalized_test, mask, jax.random.PRNGKey(9))
    assert before[1] / before[0] < 1.5, "a fresh predictor should not favour any context"

    def batches():
        while True:
            yield {"image": np.asarray(train_x[rs.randint(0, len(train_x), 16)], np.uint8)}

    params = initial.variables
    batch = {"image": images()}
    step = step_with(initial.averaged)
    same = float(objective.scalar_loss(params, batch, step)[0])
    assert same > 0.1
    moved = {"params": {"context_encoder": jax.tree.map(
        lambda value: value + 0.1, initial.averaged["params"]["context_encoder"])}}
    other = float(objective.scalar_loss(params, batch, step_with(merge(initial.averaged, moved)))[0])
    # Far above fp32 rounding: the loss reads the averaged context encoder.
    assert abs(other - same) > 1e-4 * same

    state = trainer.fit(dataset(batches, batch=16), steps=150, log_every=50)

    after = context_ablation(objective, state.variables, state.ema,
                             normalized_test, mask, jax.random.PRNGKey(9))
    assert after[0] < before[0] / 2, "held-out prediction error did not improve"
    assert after[1] / after[0] > 5.0, "the predictor still ignores its context"



# --- collapse telemetry ----------------------------------------------------

def health_in(z, dtype):
    """`representation_health`'s two statistics in numpy at `dtype`: the
    mean per-dimension population std, and the RMS off-diagonal entry of the
    unbiased covariance."""
    z = np.asarray(z, dtype)
    centered = z - z.mean(axis=0)
    cov = centered.T @ centered / (len(z) - 1)
    off = cov[~np.eye(z.shape[1], dtype=bool)]
    return np.asarray([z.std(axis=0).mean(), np.sqrt(np.mean(off ** 2))])


def test_representation_health_is_the_float64_statistics():
    """Both statistics against numpy's in float64, as exact as numpy's float32,
    on embeddings whose dimensions are half redundant; a collapsed batch has
    neither, to float32's rounding of its mean."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(32, 8))
    z = np.concatenate([base, base + 0.1 * rng.normal(size=(32, 8))], axis=1).astype(np.float32)
    health = representation_health(jnp.asarray(z))
    dew = np.asarray([health["repr_std"], health["repr_cov_offdiag"]])
    assert_as_exact_as_the_reference(dew, health_in(z, np.float32), health_in(z, np.float64),
                                     "representation health")
    collapsed = representation_health(jnp.tile(jnp.asarray(z[:1]), (32, 1)))
    assert float(collapsed["repr_std"]) < 1e-6 and float(collapsed["repr_cov_offdiag"]) < 1e-6


# --- EMA target encoder ----------------------------------------------------

def make_jepa_trainer(mask, learning_rate=1e-3, fsdp=1, **kwargs):
    return Trainer(
        JepaObjective(make_encoder(), make_predictor(), mask,
                      sample=Field("image", (RES, RES, 3)), **kwargs),
        optax.adam(learning_rate),
        key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp),
        # This encoder's parameters are far below the production shard
        # threshold, so lower it or "FSDP on" would mean "all replicated"
        layout=Layout(min_shard=256),
    )


def image_batches(batch=8):
    while True:
        yield {"image": np.asarray(images(batch=batch), np.uint8),
               "label": np.arange(batch) % 4}


def test_target_encoder_tracks_the_context_encoder(mask):
    trainer = make_jepa_trainer(mask, momentum=(0.5, 0.5), momentum_steps=1)
    initial = jax.tree.map(np.asarray, trainer.initial_state().ema)

    state = trainer.fit(dataset(image_batches, batch=8), steps=3, log_every=1)

    # The EMA holds the context encoder alone: the predictor is not averaged.
    assert set(state.ema["params"]) == {"context_encoder"}
    context_moved = any(
        not np.allclose(a, b) for a, b in zip(
            jax.tree.leaves(state.ema["params"]["context_encoder"]),
            jax.tree.leaves(initial["params"]["context_encoder"]), strict=True))
    assert context_moved, "the target encoder never followed the context encoder"

    # and it followed without jumping: still between where it started and now
    ema = jax.tree.leaves(state.ema["params"]["context_encoder"])
    live = jax.tree.leaves(state.variables["params"]["context_encoder"])
    assert any(not np.allclose(a, b) for a, b in zip(ema, live, strict=True)), "EMA is not lagging"


@pytest.mark.mesh(devices=2)
def test_jepa_trains_under_fsdp(mask):
    """Where the two halves of the trainer meet: an objective that owns a
    multi-encoder parameter tree, run through the sharded, donating train step.

    Compiling is not the claim. The parameters have to be genuinely split
    across the fsdp axis, the EMA target encoder has to follow their layout,
    and the loss has to come back finite with its collapse telemetry intact.
    """
    trainer = make_jepa_trainer(mask, fsdp=2)
    tracker = trainer.tracker = RecordingTracker()
    state = trainer.fit(dataset(image_batches, batch=jax.device_count()), steps=2, log_every=1)

    sharded = [p for p in jax.tree.leaves(state.variables) if 'fsdp' in str(p.sharding.spec)]
    assert sharded, "no JEPA parameter was sharded over the fsdp axis"
    for param in sharded:
        assert param.addressable_shards[0].data.size == param.size // 2

    # The target encoder is a second copy of the same subtree, so it must land
    # on the mesh the same way, not gathered onto every device
    encoder_specs = [p.sharding.spec for p in
                     jax.tree.leaves(state.variables["params"]["context_encoder"])]
    assert encoder_specs == [p.sharding.spec for p in jax.tree.leaves(state.ema)]
    assert int(state.step) == 2
    ticks = [entry for _, entry in tracker.scalars if "train/loss" in entry]
    assert len(ticks) == 2
    assert all(np.isfinite(entry["train/loss"]) for entry in ticks)
    assert all(entry["train/repr_std"] > 0 for entry in ticks), "collapse telemetry was lost"


def test_an_adapted_pretrained_encoder_trains_its_factors_over_one_frozen_base(mask):
    """A pretrained encoder adapted by `LoRA.apply` starts JEPA from its own
    weights: its factors and the predictor train, its base comes back
    bitwise, and the target encoder is the average of the factors over that
    same frozen base, which evaluation reads."""
    from dew.inputs import unit_range
    from dew.lora import LoRA
    from dew.objectives.base import FROZEN

    encoder = make_encoder()
    pretrained = encoder.init(jax.random.key(5), jnp.ones((1, RES, RES, 3)),
                              jnp.arange(mask.num_context, dtype=jnp.int32)[None])
    adapter = LoRA(rank=2, modules=("up_proj",)).apply(encoder, pretrained, key=0)
    objective = JepaObjective(adapter.model, make_predictor(), mask, sample=Field("image", (RES, RES, 3)),
                              encoder_variables=adapter.variables, momentum=(0.5, 0.5), momentum_steps=1)
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0))
    initial = jax.tree.map(np.asarray, trainer.initial_state().variables)

    state = trainer.fit(dataset(image_batches, batch=8), steps=3, log_every=3)

    factors = state.variables["params"]["context_encoder"]
    assert factors and all(path[-1].key in ("lora_A", "lora_B")
                           for path, _ in jax.tree_util.tree_leaves_with_path(factors))
    for before, after in zip(jax.tree.leaves(pretrained["params"]),
                             jax.tree.leaves(state.variables[FROZEN]["context_encoder"]), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    assert any(not np.array_equal(before, after) for before, after in zip(
        jax.tree.leaves(initial["params"]["predictor"]),
        jax.tree.leaves(state.variables["params"]["predictor"]), strict=True))
    assert set(state.ema["params"]) == {"context_encoder"}
    lagging = state.ema["params"]["context_encoder"]
    assert any(not np.allclose(a, b) for a, b in zip(jax.tree.leaves(lagging), jax.tree.leaves(factors),
                                                       strict=True))

    pixels = images()
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=state.averaged)
    features = objective.evaluate(state.variables, {"image": pixels, "label": jnp.zeros((4,), jnp.int32)},
                                  step).features
    target = {"params": lagging, FROZEN: state.variables[FROZEN]["context_encoder"]}
    expected = jnp.mean(objective.encode(target, unit_range(pixels)), axis=1)
    np.testing.assert_allclose(np.asarray(features), np.asarray(expected), atol=1e-6, rtol=0)


# --- probes ----------------------------------------------------------------

def separable_embeddings(num_classes=4, per_class=8, dim=6, noise=0.05):
    rng = np.random.RandomState(0)
    centers = rng.normal(size=(num_classes, dim)) * 3
    labels = np.repeat(np.arange(num_classes), per_class)
    x = centers[labels] + rng.normal(size=(len(labels), dim)) * noise
    order = rng.permutation(len(labels))
    return jnp.asarray(x[order], dtype=jnp.float32), jnp.asarray(labels[order])


def test_probe_metrics_score_representations_and_average_over_the_pass():
    """Each batch's linear probe is fit by the program the first batch of its
    shape compiled, whatever its rates."""
    x, y = separable_embeddings()
    representations = Representations(features=x, labels=y)
    linear, knn = LinearProbe(4, steps=200), KnnProbe(4, k=3)
    assert linear.reads is Representations and linear.name == "batch_linear_probe_accuracy"
    assert linear.finalize(linear.merge(linear(representations, None), (0.0, 1))) == pytest.approx(
        float(linear_probe_accuracy(x, y, 4, steps=200)) / 2)
    assert knn.finalize(knn(representations, None)) > 0.9
    assert linear(representations, {})[0] > 0.9
    flipped = Representations(features=-x, labels=y)
    # The rates go in as operands and the accuracy comes back; nothing compiles.
    with steady_state(allow=("host_to_device", "device_to_host")):
        LinearProbe(4, steps=200, learning_rate=3e-2)(flipped, None)


@pytest.mark.parametrize("weights", ["init", "trainer"])
def test_the_loss_is_the_mean_squared_error_to_normalized_targets_of_the_same_blocks(mask, weights):
    """The documented objective, written in numpy over the modules' own
    outputs: the target encoder's whole view, layer-normalized without an
    affine, gathered at each block's indices, against the predictor's
    output for that block from the shared context, squared and averaged.
    Held to it in float64, as exact as the same composition in float32, on
    the objective's own initial weights and on a trainer's, whose target
    encoder reads the averaged copy."""
    if weights == "init":
        objective = make_objective(mask)
        params = averaged = objective.init(jax.random.PRNGKey(0))
    else:
        trainer = make_jepa_trainer(mask, learning_rate=3e-3, momentum=(0.9, 0.99), momentum_steps=150)
        objective, initial = trainer.objective, trainer.initial_state()
        params, averaged = initial.variables, initial.averaged
    batch, step = {"image": images()}, step_with(averaged)
    samples = (np.asarray(batch["image"], np.float32) - 127.5) / 127.5
    context_idx, target_idx = objective.mask.sample(jax.random.split(step.key)[0], len(samples))
    full = np.asarray(objective.encode(part(averaged, "context_encoder"), jnp.asarray(samples)))
    context = objective.encode(part(params, "context_encoder"), jnp.asarray(samples), context_idx)
    blocks = mask.num_targets
    repeated = (jnp.repeat(context, blocks, axis=0), jnp.repeat(context_idx, blocks, axis=0))
    predictions = np.asarray(objective.predictor.apply(
        part(params, "predictor"), *repeated, target_idx.reshape(len(samples) * blocks, -1)))

    def loss_in(dtype):
        z = full.astype(dtype)
        normed = (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + 1e-6)
        targets = np.stack([normed[row][np.asarray(target_idx)[row, block]]
                            for row in range(len(samples)) for block in range(blocks)])
        return np.mean(np.square(predictions.astype(dtype) - targets))

    dew = float(objective.scalar_loss(params, batch, step)[0])
    assert_as_exact_as_the_reference(np.asarray([dew]), np.asarray([loss_in(np.float32)]),
                                     np.asarray([loss_in(np.float64)]), "JEPA loss")
