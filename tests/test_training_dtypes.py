"""Training and checkpoint precision against native Linen/Optax updates."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import raw_leaf
from flax import linen as nn, struct

from dew.checkpoints import Checkpoints
from dew.nn.blocks import TokenEmbedding
from dew.objectives import Aux, EMASpec, Objective, Ratio
from dew.training import Trainer


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("vocab", [1, 5])
def test_token_lookup_preserves_linen_forward_and_parameter_layout(dtype, vocab):
    model = TokenEmbedding(vocab, 3, dtype=dtype)
    reference = nn.Embed(vocab, 3, dtype=dtype)
    ids = jnp.asarray([[0, 0, vocab - 1]], jnp.int32)
    variables = reference.init(jax.random.key(4), ids)
    native = model.init(jax.random.key(4), ids)
    assert jax.tree.structure(native) == jax.tree.structure(variables)
    for actual, expected in zip(jax.tree.leaves(native), jax.tree.leaves(variables), strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(model.apply(variables, ids), reference.apply(variables, ids))
    query = jnp.asarray([[0.3, -0.2, 0.1]], dtype)
    np.testing.assert_array_equal(model.apply(variables, query, method=model.attend),
                                  reference.apply(variables, query, method=reference.attend))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_repeated_token_gradients_accumulate_in_master_precision(dtype):
    model = TokenEmbedding(3, 2, dtype=dtype)
    ids = jnp.asarray([0, 0, 0, 0, 1], jnp.int32)
    small = 2.0 ** (-8 if dtype == jnp.bfloat16 else -11)
    cotangent = jnp.asarray([[1, -1], [small, -small], [small, -small], [-1, 1], [2, 3]], dtype)
    variables = model.init(jax.random.key(5), ids)

    def loss(parameters):
        outputs = model.apply({"params": parameters}, ids)
        return jnp.sum(outputs.astype(jnp.float32) * cotangent.astype(jnp.float32))

    gradient = jax.jit(jax.grad(loss))(variables["params"])["embedding"]
    expected = np.zeros((3, 2), np.float32)
    np.add.at(expected, np.asarray(ids), np.asarray(cotangent, np.float32))
    assert gradient.dtype == jnp.float32
    np.testing.assert_array_equal(gradient, expected)


@pytest.mark.parametrize("vocab, padding_idx", [(5, 0), (5, 4), (5, -1), (5, -5), (1, 0), (1, -1)])
def test_a_padding_rows_lookups_train_it_no_more_than_torchs(vocab, padding_idx):
    """`padding_idx` leaves the row its id names untrained by every lookup of it, as torch's
    `nn.Embedding(padding_idx=...)` does, a negative id counted from the table's end and a table of
    one row alike: the gradient under a drawn cotangent of small integers is torch's to the bit, each
    row the sum of its lookups' cotangents, exact in float32 in any order. An id outside the table is
    refused."""
    import torch

    ids = np.asarray([[0, vocab - 1, 0], [vocab - 1, 0, vocab - 1]])
    cotangent = np.random.default_rng(7).integers(-8, 8, (*ids.shape, 3)).astype(np.float32)
    model = TokenEmbedding(vocab, 3, padding_idx=padding_idx)
    variables = model.init(jax.random.key(4), jnp.asarray(ids, jnp.int32))

    def loss(parameters):
        return jnp.sum(model.apply({"params": parameters}, jnp.asarray(ids, jnp.int32)) * cotangent)

    ours = jax.grad(loss)(variables["params"])["embedding"]
    reference = torch.nn.Embedding(vocab, 3, padding_idx=padding_idx)
    with torch.no_grad():
        reference.weight.copy_(torch.from_numpy(np.asarray(variables["params"]["embedding"])))
    (reference(torch.from_numpy(ids)) * torch.from_numpy(cotangent)).sum().backward()
    np.testing.assert_array_equal(np.asarray(ours), reference.weight.grad.numpy())
    assert not np.any(np.asarray(ours)[padding_idx])
    with pytest.raises(ValueError):
        TokenEmbedding(vocab, 3, padding_idx=vocab).init(jax.random.key(4), jnp.asarray(ids, jnp.int32))


@pytest.mark.parametrize("family", ["clip-token", "clip-position", "clip-vision", "siglip",
                                    "gemma3n", "decision", "joint"])
def test_family_embedding_gradients_sum_repeated_ids_in_master_precision(family):
    """Each family's own configured table receives a cancelling sequence of repeated-ID
    cotangents. Its fp32 master retains contributions lost by accumulation in bf16."""
    from dew.decision.head import DecisionHead, JointSchemaHead
    from dew.nn.text_encoders import CLIPTextTransformer, CLIPVisionTransformer
    from dew.nn.vision.gemma3n import Gemma3nProjectorModule
    from dew.nn.vision.siglip import SiglipVision, SiglipVisionTransformer

    dtype = jnp.bfloat16
    clip = CLIPTextTransformer(vocab_size=3, hidden_size=2, num_layers=0, num_heads=1,
                               max_position_embeddings=3, dtype=dtype)
    choices = {
        "clip-token": (clip, lambda module, ids: module.token_embedding(ids), "token_embedding"),
        "clip-position": (clip, lambda module, ids: module.position_embedding(ids), "position_embedding"),
        "clip-vision": (CLIPVisionTransformer(hidden_size=2, num_layers=0, num_heads=1,
                                             image_size=2, patch_size=1, dtype=dtype),
                        lambda module, ids: module.position_embedding(ids), "position_embedding"),
        "siglip": (SiglipVisionTransformer(SiglipVision(hidden_size=2, num_layers=0, num_heads=1,
                                                        image_size=2, patch_size=1), dtype=dtype),
                   lambda module, ids: module.position_embedding(ids), "position_embedding"),
        "gemma3n": (Gemma3nProjectorModule(2, 2, vocab_size=3, dtype=dtype),
                    lambda module, ids: module.embedding(ids), "embedding"),
        "decision": (DecisionHead(2, layers=0, dtype=dtype),
                     lambda module, ids: module.type_embedding(ids), "type_embedding"),
        "joint": (JointSchemaHead(2, width=2, heads=1, layers=0, routing_layers=0, dtype=dtype),
                  lambda module, ids: module.type_embedding(ids), "type_embedding"),
    }
    model, lookup, name = choices[family]
    ids = jnp.asarray([0, 0, 0, 0, 1], jnp.int32)
    cotangent = jnp.asarray([[1, -1], [2**-8, -2**-8], [2**-8, -2**-8], [-1, 1], [2, 3]], dtype)
    variables = model.init(jax.random.key(5), ids, method=lookup)

    def loss(params):
        outputs = model.apply({"params": params}, ids, method=lookup)
        return jnp.sum(outputs.astype(jnp.float32) * cotangent.astype(jnp.float32))

    gradient = jax.jit(jax.grad(loss))(variables["params"])[name]["embedding"]
    expected = np.zeros(gradient.shape, np.float32)
    np.add.at(expected, np.asarray(ids), np.asarray(cotangent, np.float32))
    assert gradient.dtype == jnp.float32
    np.testing.assert_array_equal(gradient, expected)


@struct.dataclass
class Moments:
    errors: Ratio
    predictions: jax.Array
    rows: jax.Array


class MixedDense(nn.Module):
    output_dtype: jax.typing.DTypeLike

    @nn.compact
    def __call__(self, x):
        hidden = nn.Dense(3, param_dtype=jnp.bfloat16)(x)
        return nn.Dense(1, param_dtype=self.output_dtype)(jnp.tanh(hidden))


class DenseObjective(Objective):
    def __init__(self, parameter_kind, loss_kind):
        self.loss_kind = loss_kind
        self.compute_dtype = jnp.float64 if "64" in parameter_kind else jnp.float32
        self.model = (MixedDense(self.compute_dtype) if parameter_kind.startswith("mixed") else
                      nn.Dense(1, param_dtype=jnp.float64 if parameter_kind == "float64" else jnp.bfloat16))

    def fresh_variables(self, key, held):
        return self.model.init(key, jnp.ones((1, 2), self.compute_dtype))

    def reference_loss(self, params, batch):
        prediction = self.model.apply({"params": params}, batch["x"])
        value = jnp.mean((prediction - batch["y"]) ** 2)
        return value + .125 * jnp.mean(prediction) ** 2 if self.loss_kind == "composite" else value

    def loss(self, variables, batch, step):
        prediction = self.model.apply(variables, batch["x"])
        errors = (prediction - batch["y"]) ** 2
        if self.loss_kind == "scalar":
            return jnp.mean(errors), Aux({})
        result = Ratio(jnp.sum(errors), jnp.asarray(errors.size, jnp.int32))
        if self.loss_kind == "mean":
            return result, Aux({})
        return Moments(result, jnp.sum(prediction), jnp.asarray(prediction.size, jnp.int32)), Aux({})

    def reduce_loss(self, stats):
        if isinstance(stats, Moments):
            value, active = stats.errors.mean()
            return value + .125 * (stats.predictions / stats.rows) ** 2, active
        return super().reduce_loss(stats)


def compare(actual, expected, *, exact=False, tolerance=2e-6):
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        got, want = raw_leaf(got), raw_leaf(want)
        assert got.dtype == want.dtype
        if exact or got.dtype == jnp.bfloat16:
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance * .1)


# XLA's TPU backend rewrites a program's 64-bit types into pairs of 32-bit
# ones: its float64 is no IEEE double, and the rewrite has no case for
# ragged_dot, which a routed mixture runs.
ON_TPU = jax.default_backend() == "tpu"


def exercise_updates_and_resume(
    tmp_path, parameter_kind, loss_kind, k, *, dynamic_scale=False, ema_decay=None
):
    objective = DenseObjective(parameter_kind, loss_kind)
    if ON_TPU and "64" in parameter_kind:
        with pytest.raises(ValueError):
            Trainer(objective, optax.adam(.01), key=jax.random.PRNGKey(17)).place()
        return
    if ema_decay is not None:
        objective.ema = EMASpec(optax.constant_schedule(ema_decay))
    optimizer = optax.adam(.01)
    checkpoints = Checkpoints(str(tmp_path / "run"))
    def trainer():
        return Trainer(objective, optimizer, key=jax.random.PRNGKey(17),
                       accumulation=k, dynamic_scale=dynamic_scale, checkpoints=checkpoints)
    train = trainer()
    state, _, _ = train.place()
    expected_params, expected_opt, expected_ema = state.variables["params"], state.opt_state, state.ema
    dtype = objective.compute_dtype
    x = jnp.array([[1.1234567890123, 2.2345678901234], [2.3456789012345, -1.4567890123456]], dtype)
    y = jnp.array([[.01234567890123], [1.2345678901234]], dtype)
    batch = {"x": jnp.tile(x, (jax.device_count(), 1)), "y": jnp.tile(y, (jax.device_count(), 1))}
    step = train.compile(state, batch)
    tolerance = 2e-13 if parameter_kind == "float64" else 2e-6
    # Compile the independent Linen loss and native Optax update together, as a
    # real training loop does. Eager Adam rounds every bf16 intermediate; GPU
    # fusion need not, so eager parameter bits are not the compiled contract.
    @jax.jit
    def reference_update(params, opt_state, batch):
        loss, grads = jax.value_and_grad(objective.reference_loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), opt_state

    for window in range(2):
        expected_loss, expected_params, expected_opt = reference_update(expected_params, expected_opt, batch)
        if expected_ema is not None:
            def average(old, new):
                work = np.float64 if old.dtype == jnp.float64 or new.dtype == jnp.float64 else np.float32
                weight = np.asarray(ema_decay, work)
                value = weight * np.asarray(old, work) + (np.asarray(1, work) - weight) * np.asarray(
                    new, work
                )
                return value.astype(old.dtype)
            expected_ema = jax.tree.map(average, expected_ema, {"params": expected_params})
        for micro in range(k):
            state, loss, _, finite, accepted, _ = step(state, batch)
            assert bool(finite) and bool(accepted)
            if window == 0 and micro == 0:
                checkpoints.save(1, state, None, {"loss": float(loss)})
            if micro < k - 1 and state.accumulation.gradient is not None:
                for gradient, parameter in zip(jax.tree.leaves(state.accumulation.gradient),
                                                jax.tree.leaves(state.variables["params"]), strict=True):
                    assert gradient.dtype == jnp.promote_types(parameter.dtype, jnp.float32)
        np.testing.assert_allclose(loss, expected_loss, rtol=tolerance, atol=tolerance * .1)
        compare(state.variables["params"], expected_params, tolerance=tolerance)
        compare(state.opt_state, expected_opt, tolerance=tolerance)
        compare(state.ema, expected_ema, tolerance=tolerance)
    assert int(state.updates) == 2
    checkpoints.wait()
    resumed_trainer = trainer()
    restored, _, _ = resumed_trainer.place()
    step = resumed_trainer.compile(restored, batch)
    for _ in range(2 * k - 1):
        restored, *_ = step(restored, batch)
    compare(restored, state, exact=True)


@pytest.mark.parametrize("parameter_kind,loss_kind,k", [
    ("bfloat16", "scalar", 1),
    ("bfloat16", "scalar", 2),
    ("bfloat16", "mean", 1),
    ("bfloat16", "mean", 2),
    ("mixed", "mean", 2),
    ("mixed", "composite", 2),
])
def test_adam_native_parameter_dtypes_survive_updates_and_checkpoints(tmp_path, parameter_kind, loss_kind, k):
    exercise_updates_and_resume(tmp_path, parameter_kind, loss_kind, k)


@pytest.mark.parametrize("parameter_kind,loss_kind,k", [
    ("float64", "scalar", 1),
    ("float64", "mean", 2),
    ("float64", "composite", 2),
    ("mixed64", "composite", 2),
])
def test_float64_statistics_and_updates_preserve_requested_precision(tmp_path, parameter_kind, loss_kind, k):
    with jax.enable_x64():
        total, mass = 1 + 2. ** -40, 3 + 2. ** -35
        reduced, _ = Ratio(jnp.array(total, jnp.float64), jnp.array(mass, jnp.float64)).mean()
        np.testing.assert_allclose(reduced, total / mass, rtol=0, atol=1e-15)
        exercise_updates_and_resume(tmp_path, parameter_kind, loss_kind, k)


@pytest.mark.parametrize("parameter_kind", ["bfloat16", "float64"])
def test_scaled_native_gradients_preserve_update_and_restart(tmp_path, parameter_kind):
    with jax.enable_x64():
        exercise_updates_and_resume(tmp_path, parameter_kind, "composite", 2, dynamic_scale=True)


@pytest.mark.parametrize("seq_aux", [False, True])
def test_router_reductions_preserve_float64_scores(seq_aux):
    from dew.nn.moe import global_router_loss, sequence_router_losses
    from dew.objectives.routing import router_moments

    def balance_loss(scores, indices):
        if seq_aux:
            return jnp.mean(sequence_router_losses(scores, indices, .2))
        return global_router_loss(router_moments(scores, indices, {}), .2)

    with jax.enable_x64():
        values = np.array([[[.8 + 2.**-35, .2 - 2.**-35], [.65, .35], [.6, .4]],
                           [[.9, .1], [.15, .85], [.2, .8]]], np.float64)
        indices = np.argmax(values, axis=-1)[..., None]
        counts = np.eye(2)[indices[..., 0]].sum(axis=1)
        if seq_aux:
            coefficients = .2 * 2 * counts[:, None, :] / (2 * 3**2)
        else:
            coefficients = .2 * 2 * counts.sum(axis=0) / 6**2
        expected_gradient = np.broadcast_to(coefficients, values.shape)
        expected = np.sum(values * expected_gradient)
        loss, gradient = jax.value_and_grad(
            lambda scores: balance_loss(scores, jnp.asarray(indices)))(jnp.asarray(values))
        np.testing.assert_allclose(loss, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=1e-15)


def test_optimizer_dtype_overflow_backs_off_and_drops_its_window():
    import dataclasses

    class Cancellation(Objective):
        def fresh_variables(self, key, held):
            return {"params": {"w": jnp.array(0., jnp.float16)}}
        def loss(self, variables, batch, step):
            value = variables["params"]["w"].astype(jnp.float32)
            first = batch["first"]
            left_mass = jnp.where(first, 100., 1.)
            right_mass = jnp.where(first, 1., 100.)
            left = Ratio(value * jnp.where(first, 60000., -50000.) * left_mass, left_mass)
            right = Ratio(value * jnp.where(first, -50000., 60000.) * right_mass, right_mass)
            return (left, right), Aux({})
        def reduce_loss(self, stats):
            left, active_left = stats[0].mean()
            right, active_right = stats[1].mean()
            return left + right, active_left | active_right

    trainer = Trainer(Cancellation(), optax.sgd(.01), key=jax.random.PRNGKey(2),
                      accumulation=2, dynamic_scale=True)
    initial = trainer.initial_state()
    initial = dataclasses.replace(initial, scale=dataclasses.replace(initial.scale, scale=jnp.array(1.)))
    batch = {"first": jnp.ones((), dtype=bool)}
    step = trainer.compile(initial, batch)
    prefix, _, _, _, accepted, _ = step(initial, batch)
    assert bool(accepted)
    # The step consumes the state, so the prefix is read before the rejected step.
    prefix_params = jax.tree.map(np.asarray, prefix.variables)
    # Each finalized contribution fits fp16; their fp32 sum does not. The rejection drops the window,
    # its accepted first microbatch with it (`dropped_window`).
    rejected, loss, _, finite, accepted, _ = step(prefix, {"first": jnp.zeros((), dtype=bool)})
    assert bool(finite) and float(loss) == 0 and not bool(accepted)
    assert int(rejected.step) == 2 and int(rejected.microstep) == 0 and int(rejected.updates) == 0
    assert float(rejected.scale.scale) == .5
    assert all(not np.any(np.asarray(leaf)) for leaf in rejected.accumulation.statistics)
    compare(rejected.variables, prefix_params, exact=True)


@pytest.mark.parametrize("decay", [0.5, 0.999, 0.9999])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_an_ema_leaf_the_step_left_alone_keeps_its_average_bit_for_bit(dtype, decay):
    """`ema_update` of a leaf whose live value is its average returns that average bit for bit, at any
    decay, however its program fuses the update: a parameter no gradient moved keeps its EMA as it was."""
    from dew.training.transaction import ema_update

    average = jnp.asarray(np.random.default_rng(0).standard_normal(4096), dtype)
    kept = jax.jit(ema_update)({"w": average}, {"w": average}, decay)["w"]
    assert kept.dtype == average.dtype and np.array_equal(np.asarray(kept), np.asarray(average))


def test_an_ema_of_finite_leaves_of_opposite_signs_stays_their_finite_average():
    """`ema_update` between finite leaves returns their weighted average wherever it is finite, at the ends
    of fp32's range too: an average of 2**127 and a live value of -(2**127) at decay 0.5 average to exactly 0,
    where their difference alone overflows."""
    from dew.training.transaction import ema_update

    average = jnp.asarray([2.0 ** 127, -(2.0 ** 127), 3e38], jnp.float32)
    live = jnp.asarray([-(2.0 ** 127), 2.0 ** 127, -3e38], jnp.float32)
    updated = jax.jit(ema_update)({"w": average}, {"w": live}, 0.5)["w"]
    assert np.array_equal(np.asarray(updated), np.zeros(3, np.float32))


def test_an_ema_of_a_leaf_the_step_left_alone_is_nan_at_a_nan_decay():
    """`ema_update` at a NaN decay returns NaN, a leaf the step left as it was among them, as the weighted
    sum does: the resulting state's finiteness check sees the bad decay, rather than a select that keeps the
    leaf hiding it."""
    from dew.training.transaction import ema_update

    leaf = jnp.asarray([1.0, -2.0], jnp.float32)
    updated = jax.jit(ema_update)({"w": leaf}, {"w": leaf}, jnp.float32(np.nan))["w"]
    assert np.isnan(np.asarray(updated)).all()


@pytest.mark.parametrize("parameter_kind", ["bfloat16", "mixed", "float64", "mixed64"])
def test_ema_storage_dtype_and_precision_survive_real_updates_and_restart(tmp_path, parameter_kind):
    with jax.enable_x64():
        exercise_updates_and_resume(tmp_path, parameter_kind, "mean", 2,
                                    ema_decay=jnp.array(.5, jnp.float32))


@pytest.mark.skipif(ON_TPU, reason="XLA's TPU rewrite of 64-bit types has no case for ragged_dot")
@pytest.mark.parametrize("bias_dtype", [jnp.bfloat16, jnp.float32, jnp.float64])
def test_x64_router_bias_storage_survives_updates_and_restart(tmp_path, bias_dtype):
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives import Step
    from dew.objectives.lm import LMObjective
    from dew.objectives.routing import ExpertBalance
    with jax.enable_x64():
        model = CausalTransformer(vocab_size=8, emb_features=8, num_layers=1,
                                  num_heads=2, mlp_features=16, max_seq_len=8,
                                  mixture={"experts": 2, "top_k": 1, "bias": True})
        initial = model.init(jax.random.PRNGKey(1), jnp.ones((1, 4), jnp.int32))
        initial["moe"] = jax.tree.map(lambda x: x.astype(bias_dtype), initial["moe"])
        objective = LMObjective(model, 4, variables=initial, head_chunks=1,
                                balance=ExpertBalance(balance_rate=.02, aux_loss_alpha=.1, seq_aux=False))
        checkpoints = Checkpoints(str(tmp_path / "router"))
        def build():
            return Trainer(objective, optax.sgd(.01), key=jax.random.PRNGKey(2),
                           accumulation=2, checkpoints=checkpoints)
        trainer = build()
        state, _, _ = trainer.place()
        batch = {"text": jnp.ones((jax.device_count(), 5), jnp.int32)}
        _, aux = objective.loss(state.variables, batch, Step(state.microstep, state.key, state.variables))
        counts = jax.tree.map(np.asarray, aux.effects)
        assert all(x.dtype == np.int64 for x in jax.tree.leaves(counts))
        def expected_bias(bias, count):
            total = sum(int(x) for x in count)
            direction = np.array([np.sign(total - count.size * int(x)) for x in count])
            return (np.asarray(bias, np.float64) + .02 * direction).astype(bias.dtype)
        expected = jax.tree.map(expected_bias, state.variables["moe"], counts)
        step = trainer.compile(state, batch)
        prefix, *_ = step(state, batch)
        checkpoints.save(1, prefix, None, {"loss": 0.})
        actual, *_ = step(prefix, batch)
        compare(actual.variables["moe"], expected, exact=True)
        for _ in range(2):
            actual, *_ = step(actual, batch)
        checkpoints.wait()
        resumed_trainer = build()
        restored, _, _ = resumed_trainer.place()
        step = resumed_trainer.compile(restored, batch)
        for _ in range(3):
            restored, *_ = step(restored, batch)
        compare(restored, actual, exact=True)
