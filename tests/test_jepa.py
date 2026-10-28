"""I-JEPA and V-JEPA: masking, shapes, the objective, collapse telemetry, and
the objective through the general trainer."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import dataset
from learning_gate import assert_learns
from recording import RecordingTracker
from reference_error import as_given, assert_as_exact, widened
from steady_state import steady_state

from dew.artifacts import Representations
from dew.batch import merge, part
from dew.inputs import Field
from dew.nn.backbones.jepa import JepaPredictor
from dew.objectives.base import Step
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


SEEDS = {'image token index': 8, 'video token index': 8, 'image predictor': 16, 'video predictor': 8,
         'adapted features': 8, 'JEPA loss init': 256, 'JEPA loss trainer': 256}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""


def shuffled_images(make, rng):
    """Check 1's draw of `make(seed)`'s batch and, for each of its rows, a shuffle of every grid
    position."""
    every = jnp.arange(GRID[0] * GRID[1], dtype=jnp.int32)

    def draw(seed):
        x = make(seed)
        keys = jax.random.split(jax.random.fold_in(rng, seed), x.shape[0])
        return x, jnp.stack([jax.random.permutation(key, every) for key in keys])
    return draw


def gathered(module, params, index, axis, dtype=None):
    """`module`'s full encoding of a batch with its tokens taken in `order` along `axis`
    (`index(full, order)`, the take's index): a shuffled subset's reference, and computing at `dtype`
    on its parameters widened, its truth."""
    apply = jax.jit((module if dtype is None else module.clone(dtype=dtype)).apply)

    def run(x, order):
        full = apply(params if dtype is None else widened(params), x)
        return jnp.take_along_axis(full, index(full, order), axis=axis)
    return run


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
    # Each offered position's kept and dropped counts against the rate's: Pearson's statistic over both
    # cells is the binomial one, sum (taken - offered chance)^2 / (offered chance (1 - chance)), on
    # one degree of freedom less than the positions.
    observed = np.concatenate([taken[seen], (offered - taken)[seen]])
    expected = np.concatenate([offered[seen] * chance, offered[seen] * (1 - chance)])
    p = stats.chisquare(observed, expected, ddof=int(seen.sum())).pvalue
    assert p > 1e-6, f"the context favours some free positions (p = {p:.1e})"


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
    draw = shuffled_images(images, rng)
    params = jax.tree.map(lambda p: p + 0.05, encoder.init(rng, *draw(0)))
    # Every token's encoding its own, so a shuffle shows.
    full = np.asarray(encoder.apply(params, images()))
    assert not (full[:, 1:] == full[:, :1]).all(axis=-1).any()
    assert_as_exact(jax.jit(lambda x, order: encoder.apply(params, x, order)),
                    gathered(encoder, params, lambda full, order: order[..., None], axis=1), draw,
                    "image token index", seeds=SEEDS['image token index'],
                    truth_fn=gathered(encoder, params, lambda full, order: order[..., None], axis=1,
                                      dtype=jnp.float64))


def test_the_video_token_index_selects_the_same_tubelet_in_every_frame(rng):
    """A video encoder's `token_idx` names patch positions shared by every
    frame, so the shuffled full encoding is the full encoding shuffled within
    each frame."""
    encoder = JepaVideoEncoder(patch_size=PATCH, emb_features=32, num_layers=1,
                               num_heads=2, mlp_ratio=2)
    draw = shuffled_images(videos, rng)
    params = jax.tree.map(lambda p: p + 0.05, encoder.init(rng, *draw(0)))

    def within_frames(full, order):
        return order[:, None, :, None]

    assert_as_exact(jax.jit(lambda x, order: encoder.apply(params, x, order)),
                    gathered(encoder, params, within_frames, axis=2), draw, "video token index",
                    seeds=SEEDS['video token index'],
                    truth_fn=gathered(encoder, params, within_frames, axis=2, dtype=jnp.float64))


@pytest.mark.parametrize("factorized", [False, True], ids=["image", "video"])
def test_the_predictor_predicts_each_target_by_its_position(mask, rng, factorized):
    """The targets are mask tokens told their grid position, so the order they
    are asked in cannot matter: a shuffled target index gives the predictions
    shuffled the same way."""
    predictor = make_predictor(factorized=factorized)
    frames = (FRAMES,) if factorized else ()

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(rng, seed), 3)
        context_idx, target_idx = mask.sample(keys[0], 2)
        context = jax.random.normal(keys[1], (2, *frames, mask.num_context, 32))
        perm = jnp.stack([jax.random.permutation(key, mask.block_area) for key in jax.random.split(keys[2])])
        return context, context_idx, target_idx[:, 0], perm

    params = jax.tree.map(lambda p: p + 0.05, predictor.init(rng, *draw(0)[:3]))

    def shuffled(context, context_idx, block, perm):
        return predictor.apply(params, context, context_idx, jnp.take_along_axis(block, perm, axis=1))

    def ordered(dtype=None):
        apply = jax.jit((predictor if dtype is None else predictor.clone(dtype=dtype)).apply)

        def run(context, context_idx, block, perm):
            predictions = apply(params if dtype is None else widened(params), context, context_idx, block)
            gather = perm[:, None, :, None] if factorized else perm[..., None]
            return jnp.take_along_axis(predictions, gather, axis=-2)
        return run

    # Every target's prediction its own, so a shuffle shows.
    predictions = np.asarray(ordered()(*draw(0)))
    assert not (predictions[..., 1:, :] == predictions[..., :1, :]).all(axis=-1).any()
    label = f"{'video' if factorized else 'image'} predictor"
    assert_as_exact(jax.jit(shuffled), ordered(), draw, label, seeds=SEEDS[label],
                    truth_fn=ordered(jnp.float64))


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
    assert not np.array_equal(before, expected)
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


def context_errors(objective, params, ema, data, mask, rng, draws=8):
    """Each pair of consecutive images' mean prediction error from their own contexts and from each
    other's, over `draws` mask draws: two arrays of a value a pair, pairs being independent draws."""
    n, blocks = data.shape[0], mask.num_targets
    full = normalize_targets(objective.encode(part(ema, "context_encoder"), data))

    def error(ctx, context_idx, target_idx):
        targets = jnp.take_along_axis(full[:, None], target_idx.reshape(n, blocks, -1, 1), axis=-2)
        predictions = objective.predictor.apply(
            part(params, "predictor"),
            jnp.repeat(ctx, blocks, axis=0),
            jnp.repeat(context_idx, blocks, axis=0),
            target_idx.reshape(n * blocks, -1),
        ).reshape(targets.shape)
        return np.asarray(jnp.mean((predictions - targets) ** 2, axis=(1, 2, 3)), np.float64)

    own, other = [], []
    for key in jax.random.split(rng, draws):
        context_idx, target_idx = mask.sample(key, n)
        context = objective.encode(part(params, "context_encoder"), data, context_idx)
        swapped = context.reshape(n // 2, 2, *context.shape[1:])[:, ::-1].reshape(context.shape)
        own.append(error(context, context_idx, target_idx))
        other.append(error(swapped, context_idx, target_idx))
    return tuple(np.mean(errors, axis=0).reshape(n // 2, 2).mean(axis=1) for errors in (own, other))


def test_a_video_objective_steps_on_an_explicit_data_axis(mask):
    """Each row's masks are its example's own draws, placed as the row is: under an Explicit data axis
    (`MeshSpec.explicit`) the video encoder repeats each row's token indices over its frames where
    the row lives, and the step trains to a finite loss."""
    objective = JepaObjective(
        JepaVideoEncoder(patch_size=PATCH, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=2),
        make_predictor(factorized=True), mask, sample=Field("video", (FRAMES, RES, RES, 3)))
    rows = jax.device_count()

    def batches():
        while True:
            yield {"video": np.asarray(videos(batch=rows), np.float32)}

    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec(explicit=("data",)))
    tracker = trainer.tracker = RecordingTracker()
    state = trainer.fit(dataset(batches, batch=rows), steps=1, log_every=1)
    assert int(state.step) == 1
    losses = [entry["train/loss"] for _, entry in tracker.scalars if "train/loss" in entry]
    assert losses and all(np.isfinite(loss) for loss in losses)


def test_training_makes_the_prediction_depend_on_the_context(mask):
    """The loss can be driven down by predicting the average target while
    ignoring the context entirely, which would teach the encoder nothing.
    Over 64 held-out pairs of images, each pair's errors averaged over 8 mask
    draws, a unit each of one verdict (the learning gate's paired contrasts):
    a fresh predictor's error from the other image's context stays under 1.5
    times its own, and training at least halves its own and makes the
    other's over five times it."""
    rs = np.random.RandomState(0)
    train_x, test_x = solid_colour_images(rs, 128), solid_colour_images(rs, 128)
    normalized_test = (test_x - 127.5) / 127.5

    trainer = make_jepa_trainer(mask, learning_rate=3e-3,
                                momentum=(0.9, 0.99), momentum_steps=150)
    objective = trainer.objective
    initial = trainer.initial_state()
    before = context_errors(objective, initial.variables, initial.ema,
                              normalized_test, mask, jax.random.PRNGKey(9))

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
    # The loss reads the averaged context encoder: moving it moves the loss.
    assert np.isfinite(other) and other != same

    state = trainer.fit(dataset(batches, batch=16), steps=150, log_every=50)

    after = context_errors(objective, state.variables, state.ema,
                             normalized_test, mask, jax.random.PRNGKey(9))
    contrasts = {"a fresh predictor favours no context": 1.5 * before[0] - before[1],
                 "held-out error halves": before[0] - 2 * after[0],
                 "the predictor reads its context": after[1] - 5 * after[0]}
    assert_learns(lambda pairs: {claim: values[pairs] for claim, values in contrasts.items()},
                  "context dependence", units=before[0].size)



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


HEALTH_SEEDS = 512
"""Check 1's declared seeds for the health statistics, sized by `seeds_for` (on armada)."""


def redundant_embeddings(seed: int) -> tuple[np.ndarray]:
    """32 embeddings of 16 dimensions, half of them a small perturbation of the other half."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(32, 8))
    return (np.concatenate([base, base + 0.1 * rng.normal(size=(32, 8))], axis=1).astype(np.float32),)


def health_of(z):
    health = representation_health(jnp.asarray(z))
    return jnp.stack([health["repr_std"], health["repr_cov_offdiag"]])


def test_representation_health_is_the_float64_statistics():
    """Both statistics on par with numpy's (check 1) on embeddings whose dimensions are half
    redundant; a collapsed batch has neither, to float32's rounding of its mean."""
    assert_as_exact(jax.jit(health_of), as_given(lambda z: health_in(z, z.dtype)), redundant_embeddings,
                    "representation health", seeds=HEALTH_SEEDS)
    (z,) = redundant_embeddings(0)
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
        not np.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(state.ema["params"]["context_encoder"]),
            jax.tree.leaves(initial["params"]["context_encoder"]), strict=True))
    assert context_moved, "the target encoder never followed the context encoder"

    # and it followed without jumping: still between where it started and now
    ema = jax.tree.leaves(state.ema["params"]["context_encoder"])
    live = jax.tree.leaves(state.variables["params"]["context_encoder"])
    assert any(not np.array_equal(a, b) for a, b in zip(ema, live, strict=True)), "EMA is not lagging"


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
    from dew.batch import FROZEN
    from dew.inputs import unit_range
    from dew.lora import LoRA

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
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(lagging), jax.tree.leaves(factors),
                                                       strict=True))

    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=state.averaged)
    target = {"params": lagging, FROZEN: state.variables[FROZEN]["context_encoder"]}

    def evaluated(pixels):
        return objective.evaluate(state.variables, {"image": pixels, "label": jnp.zeros((4,), jnp.int32)},
                                  step).features

    def pooled(dtype=None):
        """The target encoder's tokens of each image, averaged, computing at `dtype` on its weights
        widened to it."""
        apply = jax.jit((objective.encoder if dtype is None else objective.encoder.clone(dtype=dtype)).apply)
        def run(pixels):
            return jnp.mean(apply(target if dtype is None else widened(target), unit_range(pixels)), axis=1)
        return run

    assert_as_exact(evaluated, pooled(), lambda seed: (images(seed),), "adapted features",
                    seeds=SEEDS['adapted features'], truth_fn=pooled(jnp.float64))


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
    # An accuracy is a count over 32 and merging a batch of none halves it: exact.
    assert linear.finalize(linear.merge(linear(representations, None), (0.0, 1))) == (
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
    The loss on par with that composition in float32 over the modules'
    float32 outputs, from the float64 evaluation of the modules and the
    composition both, over drawn images and masks (check 1), on the
    objective's own initial weights and on a trainer's, whose target
    encoder reads the averaged copy."""
    if weights == "init":
        objective = make_objective(mask)
        params = averaged = objective.init(jax.random.PRNGKey(0))
    else:
        trainer = make_jepa_trainer(mask, learning_rate=3e-3, momentum=(0.9, 0.99), momentum_steps=150)
        objective, initial = trainer.objective, trainer.initial_state()
        params, averaged = initial.variables, initial.averaged
    blocks = mask.num_targets
    # The modules computing at float64, for the truth.
    wide = JepaObjective(objective.encoder.clone(dtype=jnp.float64),
                         objective.predictor.clone(dtype=jnp.float64), mask, sample=objective.sample)

    def draw(seed):
        pixels = images(seed)
        return (pixels, *objective.mask.sample(jax.random.fold_in(jax.random.PRNGKey(7), seed), len(pixels)))

    @jax.jit
    def scored(pixels, context_idx, target_idx):
        # The objective reads these masks (`Step.tape`).
        step = step_with(averaged).replace(tape={"context": context_idx, "targets": target_idx})
        return objective.scalar_loss(params, {"image": pixels}, step)[0]

    def loss_in(dtype):
        modules = objective if dtype == np.float32 else wide

        @jax.jit
        def outputs(live, target, samples, context_idx, target_idx):
            context = modules.encode(part(live, "context_encoder"), samples, context_idx)
            repeated = (jnp.repeat(context, blocks, axis=0), jnp.repeat(context_idx, blocks, axis=0))
            return modules.encode(part(target, "context_encoder"), samples), modules.predictor.apply(
                part(live, "predictor"), *repeated, target_idx.reshape(len(samples) * blocks, -1))

        @as_given
        def run(pixels, context_idx, target_idx):
            live, target = (params, averaged) if dtype == np.float32 else (widened(params), widened(averaged))
            samples = (jnp.asarray(pixels, dtype) - 127.5) / 127.5
            z, predictions = (np.asarray(value, dtype)
                              for value in outputs(live, target, samples, context_idx, target_idx))
            # I-JEPA's `F.layer_norm(h, (h.size(-1),))`, at PyTorch's default eps.
            normed = (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + 1e-5)
            targets = np.stack([normed[row][np.asarray(target_idx)[row, block]]
                                for row in range(len(samples)) for block in range(blocks)])
            return np.mean(np.square(predictions - targets))
        return run

    label = f"JEPA loss {weights}"
    assert_as_exact(scored, loss_in(np.float32), draw, label, seeds=SEEDS[label],
                    truth_fn=loss_in(np.float64))


def test_an_explicit_momentum_count_keeps_its_law_and_none_reads_the_runs_progress(mask):
    """A momentum ramp over `momentum_steps` updates reads the update count, an explicit 0 holding the first
    momentum at every update (optax's empty ramp); with none it ramps over the run's progress, 0 to 1,
    to the last momentum at the run's end."""
    held = make_objective(mask, momentum=(0.9, 0.99), momentum_steps=0).ema
    assert not held.progressed
    assert [float(held.decay(update)) for update in (0, 1, 100)] == pytest.approx([0.9] * 3)
    counted = make_objective(mask, momentum=(0.9, 0.99), momentum_steps=10).ema
    assert not counted.progressed and float(counted.decay(10)) == pytest.approx(0.99)
    assert float(counted.decay(5)) == pytest.approx(0.945)
    progressed = make_objective(mask, momentum=(0.9, 0.99)).ema
    assert progressed.progressed
    assert [float(progressed.decay(at)) for at in (0.0, 0.5, 1.0)] == pytest.approx([0.9, 0.945, 0.99])
