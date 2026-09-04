"""I-JEPA and V-JEPA: masking, shapes, the objective, collapse telemetry, and
the objective through the general trainer."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh

from dew.artifacts import Representations
from dew.inputs import Field
from dew.objectives.base import Step
from dew.objectives.jepa import (
    JepaEncoder, JepaVideoEncoder, JepaObjective, multi_block_mask,
    representation_health, normalize_targets, linear_probe_accuracy, knn_probe_accuracy,
)
from dew.nn.backbones.jepa import JepaPredictor
from dew.registry import metrics, models
from dew.training import Layout, MeshSpec, Trainer

RES = 32
PATCH = 4
GRID = (RES // PATCH, RES // PATCH)
FRAMES = 3


@pytest.fixture
def mask():
    return multi_block_mask(GRID, num_targets=4, scale=(0.15, 0.2))


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


class Data:
    def __init__(self, train, val=None, batch=4):
        self._train, self.val, self.batch, self.records = train, val, batch, None

    def train(self):
        return self._train()

    steps_per_epoch = None


def images(seed=0, batch=4):
    return jnp.asarray(np.random.RandomState(seed).uniform(0, 255, (batch, RES, RES, 3)))


def videos(seed=0, batch=2):
    return jnp.asarray(
        np.random.RandomState(seed).uniform(0, 255, (batch, FRAMES, RES, RES, 3)))


# --- masking ---------------------------------------------------------------

def test_context_and_targets_are_disjoint(mask):
    context_idx, target_idx = mask.sample(jax.random.PRNGKey(3), 8)
    for b in range(8):
        context = set(np.array(context_idx[b]).tolist())
        targets = set(np.array(target_idx[b]).reshape(-1).tolist())
        assert not (context & targets), "context encoder can see a target patch"
        assert max(targets) < mask.num_patches and min(context) >= 0


def test_target_coverage_sits_inside_the_configured_scale():
    scale = (0.15, 0.2)
    mask = multi_block_mask(GRID, num_targets=4, scale=scale)
    coverage = mask.block_area / mask.num_patches
    assert scale[0] <= coverage <= scale[1]
    for h, w in mask.block_shapes:
        assert h * w == mask.block_area
        assert 0.75 <= h / w <= 1.5


def test_masks_are_reproducible_from_a_seed(mask):
    a = mask.sample(jax.random.PRNGKey(11), 4)
    b = mask.sample(jax.random.PRNGKey(11), 4)
    c = mask.sample(jax.random.PRNGKey(12), 4)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b))
    assert not all(jnp.array_equal(x, y) for x, y in zip(a, c))


def test_block_shapes_and_positions_actually_vary(mask):
    blocks = np.array(mask.sample(jax.random.PRNGKey(5), 64)[1]).reshape(-1, mask.block_area)
    corners = {int(block.min()) for block in blocks}
    widths = {len(np.unique(block % GRID[1])) for block in blocks}
    assert len(corners) > 1, "every block landed in the same place"
    assert len(widths) > 1, "every block came out the same shape"


def test_geometry_that_cannot_exist_is_rejected():
    with pytest.raises(ValueError, match="aspect ratio"):
        multi_block_mask((4, 4), num_targets=2, scale=(0.18, 0.19))
    with pytest.raises(ValueError, match="no context"):
        multi_block_mask(GRID, num_targets=6, scale=(0.15, 0.2))


# --- forward shapes --------------------------------------------------------

def test_image_encoder_shapes(mask, rng):
    encoder = make_encoder()
    context_idx, _ = mask.sample(rng, 4)
    x = images()
    params = encoder.init(rng, x, context_idx)

    assert encoder.apply(params, x, context_idx).shape == (4, mask.num_context, 32)
    assert encoder.apply(params, x).shape == (4, mask.num_patches, 32)


def test_image_predictor_shapes(mask, rng):
    encoder, predictor = make_encoder(), make_predictor()
    context_idx, target_idx = mask.sample(rng, 4)
    x = images()
    encoder_params = encoder.init(rng, x, context_idx)
    context = encoder.apply(encoder_params, x, context_idx)

    block = target_idx[:, 0]
    params = predictor.init(rng, context, context_idx, block)
    assert predictor.apply(params, context, context_idx, block).shape == (4, mask.block_area, 32)


def test_video_encoder_and_predictor_shapes(mask, rng):
    encoder = JepaVideoEncoder(patch_size=PATCH, emb_features=32, num_layers=1,
                               num_heads=2, mlp_ratio=2)
    predictor = make_predictor(factorized=True)
    context_idx, target_idx = mask.sample(rng, 2)
    x = videos()

    encoder_params = encoder.init(rng, x, context_idx)
    context = encoder.apply(encoder_params, x, context_idx)
    assert context.shape == (2, FRAMES, mask.num_context, 32)
    assert encoder.apply(encoder_params, x).shape == (2, FRAMES, mask.num_patches, 32)

    block = target_idx[:, 0]
    params = predictor.init(rng, context, context_idx, block)
    out = predictor.apply(params, context, context_idx, block)
    assert out.shape == (2, FRAMES, mask.block_area, 32)


def test_video_encoder_carries_a_temporal_signal(mask, rng):
    """Tubelet masking keeps the spatial layout, so time is the only axis the
    encoder can mix along: perturbing frame 0 must move frame 2's embedding."""
    encoder = JepaVideoEncoder(patch_size=PATCH, emb_features=32, num_layers=1,
                               num_heads=2, mlp_ratio=2)
    context_idx, _ = mask.sample(rng, 2)
    x = videos()
    params = jax.tree.map(lambda p: p + 0.02, encoder.init(rng, x, context_idx))

    before = encoder.apply(params, x, context_idx)
    after = encoder.apply(params, x.at[:, 0].add(1.0), context_idx)
    assert not jnp.allclose(before[:, 2], after[:, 2]), "no information flow across frames"


def test_encoder_runs_on_the_ssm_mixer(mask, rng):
    """The hybrid SSM encoder is the point of the shared block: same interface."""
    encoder = make_encoder(ssm_attention_ratio="3:1", ssm_state_dim=8)
    context_idx, _ = mask.sample(rng, 4)
    x = images()
    params = encoder.init(rng, x, context_idx)
    out = encoder.apply(params, x, context_idx)
    assert out.shape == (4, mask.num_context, 32)
    assert jnp.all(jnp.isfinite(out))


# --- the objective ---------------------------------------------------------

def test_fresh_loss_is_non_trivial_and_training_reduces_it(mask, rng):
    objective = make_objective(mask)
    params = objective.init(rng)
    batch = {"image": images()}

    def loss_of(p):
        return objective.loss(p, batch, step_with(params))[0]

    initial = float(loss_of(params))
    assert initial > 0.1, "a fresh model already predicts the targets"

    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(params)
    for _ in range(20):
        grads = jax.grad(loss_of)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    assert float(loss_of(params)) < initial * 0.9, "the objective does not train"


def test_video_objective_trains(mask, rng):
    objective = JepaObjective(
        JepaVideoEncoder(patch_size=PATCH, emb_features=32, num_layers=1,
                         num_heads=2, mlp_ratio=2),
        make_predictor(factorized=True), mask,
        sample=Field("video", (FRAMES, RES, RES, 3)))
    params = objective.init(rng)
    batch = {"video": videos()}

    def loss_of(p):
        return objective.loss(p, batch, step_with(params))[0]

    initial = float(loss_of(params))
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(params)
    for _ in range(15):
        grads = jax.grad(loss_of)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    assert float(loss_of(params)) < initial * 0.9


def test_bf16_models_keep_the_loss_in_fp32(mask, rng):
    """The MSE and its gradient are the numbers the optimizer sees, so they
    must not be quantized to the models' bf16 compute dtype."""
    objective = JepaObjective(make_encoder(dtype=jnp.bfloat16),
                              make_predictor(dtype=jnp.bfloat16), mask,
                              sample=Field("image", (RES, RES, 3)))
    params = objective.init(rng)
    loss, aux = objective.loss(params, {"image": images()}, step_with(params))
    assert loss.dtype == jnp.float32
    assert all(a.dtype == jnp.float32 for a in aux.metrics.values())


def solid_colour_images(rs, n):
    """One random colour per image, so a target block is fully determined by
    anything else in the same image and by nothing in any other image."""
    colours = rs.uniform(20, 235, (n, 1, 1, 3))
    return jnp.asarray(np.clip(colours + rs.uniform(-15, 15, (n, RES, RES, 3)), 0, 255))


def context_ablation(objective, params, ema, data, mask, rng):
    """Prediction error from the true context vs. from another image's context."""
    n, blocks = data.shape[0], mask.num_targets
    context_idx, target_idx = mask.sample(rng, n)
    full = normalize_targets(objective.encode(ema["params"]["context_encoder"], data))
    targets = jnp.take_along_axis(
        full[:, None], target_idx.reshape(n, blocks, -1, 1), axis=-2)
    context = objective.encode(params["params"]["context_encoder"], data, context_idx)

    def error(ctx):
        predictions = objective.predictor.apply(
            {"params": params["params"]["predictor"]},
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
    before = context_ablation(objective, initial.params, initial.ema,
                              normalized_test, mask, jax.random.PRNGKey(9))
    assert before[1] / before[0] < 1.5, "a fresh predictor should not favour any context"

    def batches():
        while True:
            yield {"image": np.asarray(train_x[rs.randint(0, len(train_x), 16)], np.uint8)}

    state = trainer.fit(Data(batches, batch=16), steps=150, log_every=50)

    after = context_ablation(objective, state.params, state.ema,
                             normalized_test, mask, jax.random.PRNGKey(9))
    assert after[0] < before[0] / 2, "held-out prediction error did not improve"
    assert after[1] / after[0] > 5.0, "the predictor still ignores its context"


def test_no_gradient_reaches_the_target_branch(mask, rng):
    """stop_gradient on the target encoder is what stops the trivial solution."""
    objective = make_objective(mask)
    params = objective.init(rng)
    batch = {"image": images()}

    grads = jax.grad(
        lambda ema: objective.loss(params, batch, step_with(ema))[0]
    )(params)
    assert all(float(jnp.max(jnp.abs(g))) == 0.0 for g in jax.tree.leaves(grads))


# --- collapse telemetry ----------------------------------------------------

def test_representation_std_detects_collapse():
    healthy = jax.random.normal(jax.random.PRNGKey(0), (32, 16))
    collapsed = jnp.tile(healthy[:1], (32, 1))

    assert float(representation_health(healthy)["repr_std"]) > 0.5
    assert float(representation_health(collapsed)["repr_std"]) < 1e-6


def test_collapse_telemetry_flows_through_a_degenerate_encoder(mask, rng):
    """A constant encoder is the failure mode; the aux dict must show it."""
    class ConstantEncoder(JepaEncoder):
        def __call__(self, x, token_idx=None, train: bool = False):
            out = super().__call__(x, token_idx, train)
            return jnp.zeros_like(out) + jnp.arange(out.shape[-1], dtype=out.dtype)

    batch = {"image": images()}
    collapsed = JepaObjective(
        ConstantEncoder(patch_size=PATCH, emb_features=32, num_layers=2, num_heads=2,
                        mlp_ratio=2),
        make_predictor(), mask, sample=Field("image", (RES, RES, 3)))
    collapsed_params = collapsed.init(rng)
    _, collapsed_aux = collapsed.loss(collapsed_params, batch, step_with(collapsed_params))

    healthy = make_objective(mask)
    healthy_params = healthy.init(rng)
    _, healthy_aux = healthy.loss(healthy_params, batch, step_with(healthy_params))

    assert float(collapsed_aux.metrics["repr_std"]) < 1e-5
    assert float(healthy_aux.metrics["repr_std"]) > 1e-3


def test_offdiagonal_covariance_rises_with_redundant_dimensions():
    base = jax.random.normal(jax.random.PRNGKey(0), (64, 1))
    redundant = jnp.tile(base, (1, 8))                       # every dim identical
    independent = jax.random.normal(jax.random.PRNGKey(1), (64, 8))
    assert (float(representation_health(redundant)["repr_cov_offdiag"])
            > 5 * float(representation_health(independent)["repr_cov_offdiag"]))


# --- EMA target encoder ----------------------------------------------------

def test_momentum_schedule_endpoints(mask):
    objective = make_objective(mask, momentum=(0.996, 1.0), momentum_steps=1000)
    assert float(objective.ema.decay(0)) == pytest.approx(0.996)
    assert float(objective.ema.decay(1000)) == pytest.approx(1.0)
    assert objective.ema.select(("params", "context_encoder", "anything"))
    assert not objective.ema.select(("params", "predictor", "anything"))


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

    state = trainer.fit(Data(image_batches, batch=8), steps=3, log_every=1)

    # The EMA holds the context encoder alone: the predictor is not averaged.
    assert set(state.ema["params"]) == {"context_encoder"}
    context_moved = any(
        not np.allclose(a, b) for a, b in zip(
            jax.tree.leaves(state.ema["params"]["context_encoder"]),
            jax.tree.leaves(initial["params"]["context_encoder"])))
    assert context_moved, "the target encoder never followed the context encoder"

    # and it followed rather than jumped: still between where it started and now
    ema = jax.tree.leaves(state.ema["params"]["context_encoder"])
    live = jax.tree.leaves(state.params["params"]["context_encoder"])
    assert any(not np.allclose(a, b) for a, b in zip(ema, live)), "EMA is not lagging"


def test_jepa_trains_under_fsdp(mask):
    """Where the two halves of the trainer meet: an objective that owns a
    multi-encoder parameter tree, run through the sharded, donating train step.

    Compiling is not the claim. The parameters have to be genuinely split
    across the fsdp axis, the EMA target encoder has to follow their layout,
    and the loss has to come back finite with its collapse telemetry intact.
    """
    logged = []

    class Tracker:
        def log(self, scalars, step):
            logged.append(dict(scalars))

        def artifact(self, value, step):
            pass

    trainer = make_jepa_trainer(mask, fsdp=2)
    trainer.tracker = Tracker()
    state = trainer.fit(Data(image_batches, batch=jax.device_count()), steps=2, log_every=1)

    sharded = [p for p in jax.tree.leaves(state.params) if 'fsdp' in str(p.sharding.spec)]
    assert sharded, "no JEPA parameter was sharded over the fsdp axis"
    for param in sharded:
        assert param.addressable_shards[0].data.size == param.size // 2

    # The target encoder is a second copy of the same subtree, so it must land
    # on the mesh the same way rather than being gathered onto every device
    encoder_specs = [p.sharding.spec for p in
                     jax.tree.leaves(state.params["params"]["context_encoder"])]
    assert encoder_specs == [p.sharding.spec for p in jax.tree.leaves(state.ema)]
    assert int(state.step) == 2
    assert all(np.isfinite(entry["train/loss"]) for entry in logged)
    assert all(entry["train/repr_std"] > 0 for entry in logged), "collapse telemetry was lost"


def test_evaluation_returns_pooled_embeddings_with_the_labels(mask):
    objective = make_objective(mask)
    params = objective.init(jax.random.PRNGKey(0))
    batch = {"image": images(), "label": jnp.asarray([0, 1, 2, 3])}

    out = objective.evaluate(params, batch, step_with(params))

    assert isinstance(out, Representations)
    assert out.features.shape == (4, 32)
    np.testing.assert_array_equal(out.labels, [0, 1, 2, 3])


def test_evaluation_reads_the_ema_encoder(mask):
    objective = make_objective(mask)
    params = objective.init(jax.random.PRNGKey(0))
    ema = {"params": {"context_encoder": jax.tree.map(
        lambda p: p + 0.1, params["params"]["context_encoder"])}}
    batch = {"image": images(), "label": jnp.zeros((4,), jnp.int32)}
    from dew.objectives.base import merge

    live = objective.evaluate(params, batch, step_with(params))
    averaged = objective.evaluate(params, batch, step_with(merge(params, ema)))
    assert not np.allclose(live.features, averaged.features)


# --- probes ----------------------------------------------------------------

def separable_embeddings(num_classes=4, per_class=8, dim=6, noise=0.05):
    rng = np.random.RandomState(0)
    centers = rng.normal(size=(num_classes, dim)) * 3
    labels = np.repeat(np.arange(num_classes), per_class)
    x = centers[labels] + rng.normal(size=(len(labels), dim)) * noise
    order = rng.permutation(len(labels))
    return jnp.asarray(x[order], dtype=jnp.float32), jnp.asarray(labels[order])


def test_probes_separate_clustered_embeddings():
    x, y = separable_embeddings()
    assert float(linear_probe_accuracy(x, y, num_classes=4, steps=200)) > 0.9
    assert float(knn_probe_accuracy(x, y, num_classes=4, k=3)) > 0.9


def test_probe_metrics_score_representations_and_average_over_the_pass():
    x, y = separable_embeddings()
    representations = Representations(features=x, labels=y)
    linear, knn = metrics.linear_probe(4, steps=200), metrics.knn_probe(4, k=3)
    assert linear.reads is Representations and linear.name == "linear_probe_accuracy"
    assert linear.reduce([linear(representations, None), 0.0]) == pytest.approx(
        float(linear_probe_accuracy(x, y, 4, steps=200)) / 2)
    assert knn.reduce([knn(representations, None)]) > 0.9


def test_probes_are_at_chance_on_noise():
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 6))
    y = jnp.asarray(np.random.RandomState(1).randint(0, 4, 64))
    assert float(knn_probe_accuracy(x, y, num_classes=4, k=5)) < 0.6


# --- registry ---------------------------------------------------------------

@pytest.mark.parametrize("architecture", ["jepa_encoder", "jepa_video_encoder"])
def test_registry_builds_the_jepa_models(architecture):
    model = models.build(architecture, patch_size=PATCH, emb_features=32, num_layers=1,
                         scan_order="hilbert")
    assert model.emb_features == 32 and model.scan_order == 'hilbert'
