"""EDM2's magnitude-preserving U-Net and forced weight normalization.

The fixtures are NVlabs/edm2's own `UNet` on a tiny configuration
(`tools/edm2_reference.py`), with every zero-initialized gain drawn so that
no branch is switched off: its output (unet.npz), and a few steps of its
training in train mode, where each forward writes the normalized weights
back (train.npz).
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.nn.backbones.edm2 import EDM2UNet
from dew.nn.dit import TextContext
from dew.nn.mp import MPConv, forced_weight_normalization, normalize

EDM2 = np.load(Path(__file__).resolve().parent / "fixtures" / "edm2" / "unet.npz")
TRAIN = np.load(Path(__file__).resolve().parent / "fixtures" / "edm2" / "train.npz")


def converted(arrays, prefix: str = "weights/") -> dict:
    """The reference's state dict as this model's variables: dotted names to
    nested modules, `weight` to `mp_kernel` in channels-last order."""
    variables: dict = {"params": {}, "constants": {}}
    for key in arrays.files:
        if not key.startswith(prefix):
            continue
        *path, leaf = key.removeprefix(prefix).split(".")
        value = arrays[key]
        if path and path[0] in ("enc", "dec"):
            path = [f"{path[0]}_{path[1]}", *path[2:]]
        if leaf in ("freqs", "phases"):
            collection, leaf = "constants", {"freqs": "frequencies", "phases": "phases"}[leaf]
        else:
            collection = "params"
            if leaf == "weight":
                leaf, value = "mp_kernel", np.moveaxis(value, (0, 1), (-1, -2))
        node = variables[collection]
        for name in path:
            node = node.setdefault(name, {})
        node[leaf] = jnp.asarray(value)
    return variables


def test_the_unet_is_edm2s_own():
    config = json.loads(str(EDM2["config"]))
    model = edm2_unet(config)
    x = jnp.asarray(np.moveaxis(EDM2["x"], 1, -1), jnp.float32)
    text = TextContext(hidden=jnp.asarray(EDM2["text"], jnp.float32)[:, None, :],
                       mask=jnp.ones((2, 1), jnp.int32))
    output = model.apply(converted(EDM2), x, jnp.asarray(EDM2["noise_labels"], jnp.float32), text)
    assert_as_exact_as_the_reference(np.moveaxis(np.asarray(output), -1, 1), EDM2["output32"],
                                     EDM2["output"], "edm2 unet")


def edm2_unet(config: dict) -> EDM2UNet:
    return EDM2UNet(output_channels=config["img_channels"], model_channels=config["model_channels"],
                    channel_mult=config["channel_mult"], num_blocks=config["num_blocks"],
                    attn_resolutions=config["attn_resolutions"],
                    channels_per_head=config["channels_per_head"])


def test_forced_weight_normalization_trains_as_edm2_does():
    """EDM2 normalizes a weight in place at each training forward and uses
    that weight normalized again (`MPConv.forward`, whose `w` aliases the
    parameter it overwrites); Dew stores the normalized weight after each
    update (`forced_weight_normalization`) and normalizes it at use. Both
    use normalize(normalize(w + update)) and store normalize(w + update), at
    a step's boundary apart, so from the reference's initial weights
    normalized, under the same SGD, they train alike: each step's output,
    and every stored weight and gain after it, as exact as the reference's
    float32 trajectory by tests/reference_error.py's rule."""
    config = json.loads(str(TRAIN["config"]))
    sgd = json.loads(str(TRAIN["sgd"]))
    model = edm2_unet(config)
    variables = converted(TRAIN)
    variables["params"] = jax.tree_util.tree_map_with_path(
        lambda path, value: normalize(value) if path[-1].key == "mp_kernel" else value, variables["params"])
    x = jnp.asarray(np.moveaxis(TRAIN["x"], 1, -1))
    noise = jnp.asarray(TRAIN["noise_labels"])
    text = TextContext(hidden=jnp.asarray(TRAIN["text"])[:, None, :], mask=jnp.ones((2, 1), jnp.int32))
    probe = jnp.asarray(np.moveaxis(TRAIN["probe"], 1, -1))
    optimizer = optax.chain(optax.sgd(sgd["lr"]), forced_weight_normalization())
    state = optimizer.init(variables["params"])

    def loss(params):
        output = model.apply({**variables, "params": params}, x, noise, text)
        return jnp.sum(output * probe), output

    step = jax.jit(jax.value_and_grad(loss, has_aux=True))
    params = variables["params"]
    for index in range(int(TRAIN["steps"])):
        (_, output), grads = step(params)
        assert_as_exact_as_the_reference(np.moveaxis(np.asarray(output), -1, 1),
                                         TRAIN[f"fp32/{index}/output"], TRAIN[f"fp64/{index}/output"],
                                         f"step {index} output")
        if index:
            stored = [converted(TRAIN, f"{precision}/{index}/weights/")["params"]
                      for precision in ("fp32", "fp64")]
            leaves = [np.concatenate([np.ravel(leaf) for leaf in jax.tree_util.tree_leaves(tree)])
                      for tree in (params, *stored)]
            assert_as_exact_as_the_reference(*leaves, f"step {index} stored weights and gains")
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)


def test_a_run_config_builds_the_unet_and_scores_a_batch():
    """The run's precision settings reach the model as every registered
    model takes them, attention kernel included."""
    from dew.config import ModelConfig
    from dew.data import TFDSImages
    from dew.diffusion.presets import EDM
    from dew.objectives import Step
    from dew.objectives.diffusion import Denoising, DiffusionRunConfig, TextCondition
    from dew.sampling import Euler

    config = DiffusionRunConfig(
        model=ModelConfig("edm2_unet", {"model_channels": 8, "channel_mult": [1, 2], "num_blocks": 1,
                                        "attn_resolutions": [2], "channels_per_head": 8,
                                        "dtype": "float32", "attention_impl": "xla"}),
        data=TFDSImages(image_size=4), preset=EDM(regime="pixel"),
        solver=Euler(), guidance=None, sampling_steps=2, ema_decay=None,
        val_metrics=(), text=TextCondition(encoder="char_table", checkpoint="char_table"),
        mode=Denoising(uncertainty=8))
    objective = config.build()
    params = objective.init(jax.random.PRNGKey(0))
    batch = {"image": np.full((2, 4, 4, 3), 200, np.uint8), **objective.inputs.tokenize(["a", "b"])}
    loss, _ = objective.loss(params, batch, Step(jnp.asarray(0), jax.random.PRNGKey(1), None))
    assert np.isfinite(float(loss.total / loss.mass))


def test_forced_weight_normalization_keeps_every_kernel_at_unit_magnitude():
    """Adam's steps grow an unconstrained weight's norm; the forced one stays
    at unit root-mean-square magnitude per output channel after every
    update, and a parameter that is not an MPConv weight takes Adam's own
    update."""
    layer = MPConv(6, (3, 3))
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 5, 5, 3))
    params = {"layer": layer.init(jax.random.PRNGKey(1), x)["params"], "bias": jnp.ones((6,))}

    def loss(variables):
        return jnp.sum((layer.apply({"params": variables["layer"]}, x) + variables["bias"]) ** 2)

    free, forced = optax.adam(0.3), optax.chain(optax.adam(0.3), forced_weight_normalization())
    first, _ = free.update(jax.grad(loss)(params), free.init(params), params)
    projected, _ = forced.update(jax.grad(loss)(params), forced.init(params), params)
    np.testing.assert_array_equal(projected["bias"], first["bias"])

    free_params, forced_params = params, params
    free_state, forced_state = free.init(params), forced.init(params)
    for _ in range(5):
        updates, free_state = free.update(jax.grad(loss)(free_params), free_state, free_params)
        free_params = optax.apply_updates(free_params, updates)
        updates, forced_state = forced.update(jax.grad(loss)(forced_params), forced_state,
                                              forced_params)
        forced_params = optax.apply_updates(forced_params, updates)

    def magnitude(kernel):
        return np.sqrt(np.mean(np.square(np.asarray(kernel)).reshape(-1, 6), axis=0))

    np.testing.assert_allclose(magnitude(forced_params["layer"]["mp_kernel"]), 1.0, rtol=2e-4)
    assert np.all(magnitude(free_params["layer"]["mp_kernel"]) > 1.01)


def test_the_uncertainty_head_learns_each_levels_weighted_error():
    """EDM2's u(sigma) minimizes E[w ||D - y||^2 / e^u + u]: at the optimum
    e^u is the weighted error the level leaves (Karras et al. 2024, Eq. 21),
    so the weighted error over e^u settles at one half under Dew's halved
    L2. The model here is fixed and only the head trains; the error each
    level leaves is then measured directly on fresh noise."""
    from flax import linen as nn

    from dew.diffusion import broadcast_rates, expand, presets
    from dew.inputs import Field, InputSpec
    from dew.objectives import Step
    from dew.objectives.diffusion import DiffusionObjective
    from dew.objectives.diffusion.objective import UNCERTAINTY
    from dew.sampling import Euler

    class Linear(nn.Module):
        @nn.compact
        def __call__(self, x, time, train=False):
            return nn.Dense(x.shape[-1])(x)

    process = presets.EDM(regime="pixel")()
    objective = DiffusionObjective(Linear(), process, InputSpec(Field("image", (4, 4, 3))),
                                   uncertainty=32, ema_decay=None, guidance=None,
                                   solver=Euler(), steps=2)
    params = objective.init(jax.random.PRNGKey(0))
    batch = {"image": np.asarray(jax.random.randint(jax.random.PRNGKey(1), (64, 4, 4, 3), 0, 256),
                                 np.uint8)}
    head = params["params"][UNCERTAINTY]

    def loss(head, key):
        tree = {**params, "params": {**params["params"], UNCERTAINTY: head}}
        total, _ = objective.loss(tree, batch, Step(jnp.asarray(0), key, None))
        return total.total / total.mass

    optimizer = optax.adam(3e-2)
    state = optimizer.init(head)
    step = jax.jit(lambda head, state, key: optimizer.update(
        jax.grad(loss)(head, key), state, head))
    for index in range(600):
        updates, state = step(head, state, jax.random.PRNGKey(100 + index))
        head = optax.apply_updates(head, updates)

    schedule = process.schedule
    samples = (jnp.asarray(batch["image"], jnp.float32) - 127.5) / 127.5
    for t in (-1.0, 0.0, 1.0):
        times = jnp.full((64,), t)
        rates = broadcast_rates(schedule, times, samples)
        errors = []
        for draw in range(16):
            noise = jax.random.normal(jax.random.PRNGKey(draw), samples.shape)
            noisy, c_in, target = process.prediction.forward_diffusion(samples, noise, rates)
            output = objective.model.apply(objective.model_variables(params), noisy * c_in,
                                           schedule.model_time(times))
            prediction = process.prediction.pred_transform(noisy, output, rates, times)
            errors.append(jnp.mean(optax.l2_loss(prediction, target)
                                   * expand(process.weight(times), prediction)))
        u = objective.uncertainty.apply(
            {"params": head, "constants": params["constants"][UNCERTAINTY]},
            schedule.model_time(times[:1]))[0]
        assert float(jnp.mean(jnp.asarray(errors)) * jnp.exp(-u)) == pytest.approx(0.5, rel=0.15)
