"""EDM2's magnitude-preserving U-Net and forced weight normalization.

The recording is NVlabs/edm2's own `UNet` on a tiny configuration
(`tools/edm2_reference.py`) over drawn weights, every zero-initialized gain
drawn so that no branch is switched off, and inputs: its output (`unet`), and
a few steps of its training in train mode, where each forward writes the
normalized weights back (`train`), each seed's inputs drawn again here.
"""

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from reference_error import assert_fp32_reduction_bound, declared_seeds
from reference_runs import assert_as_exact_as_recorded
from tool_support import run_from

from dew.nn.backbones.edm2 import EDM2UNet
from dew.nn.dit import TextContext
from dew.nn.mp import MP_KERNEL, MPConv, forced_weight_normalization, normalize

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "edm2" / "seeds.npz"


@functools.cache
def generator():
    """tools/edm2_reference.py, whose seeded draw the recorded inputs are drawn again by, loaded as a
    script from its directory."""
    with run_from("edm2_reference", "tools") as module:
        return module


def converted(arrays, prefix: str = "weights/") -> dict:
    """The reference's state dict as this model's variables: dotted names to
    nested modules, `weight` to `mp_kernel` in channels-last order."""
    variables: dict = {"params": {}, "constants": {}}
    for key in arrays:
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


def inputs_at(dtype, inputs) -> tuple:
    """A draw's weights as this model's variables, and its input, noise labels, text and probe, at
    `dtype`, channels last."""
    cast = {name: jnp.asarray(value, dtype) for name, value in inputs.items()}
    text = TextContext(hidden=cast["text"][:, None, :], mask=jnp.ones((2, 1), jnp.int32))
    return (converted(cast), jnp.moveaxis(cast["x"], 1, -1), cast["noise_labels"], text,
            jnp.moveaxis(cast["probe"], 1, -1))


def unet_output(dtype, inputs) -> dict:
    """The U-Net's output at `dtype` on a draw (`unet`'s leaf), channels first as the reference's."""
    variables, x, noise, text, _ = inputs_at(dtype, inputs)
    return {"output": jnp.moveaxis(edm2_unet(generator().CONFIG).apply(variables, x, noise, text), -1, 1)}


def test_the_unet_is_edm2s_own():
    """The U-Net's output, as exact as NVlabs/edm2's float32 run against its float64 run over drawn
    weights and inputs (check 1)."""
    assert_as_exact_as_recorded(FIXTURE, "unet", declared_seeds(FIXTURE)["unet"], generator().draw,
                                unet_output, "edm2 unet")


def edm2_unet(config: dict) -> EDM2UNet:
    return EDM2UNet(output_channels=config["img_channels"], model_channels=config["model_channels"],
                    channel_mult=config["channel_mult"], num_blocks=config["num_blocks"],
                    attn_resolutions=config["attn_resolutions"],
                    channels_per_head=config["channels_per_head"])


def trained(dtype, inputs) -> dict:
    """The `train` site at `dtype` on a draw: from the draw's weights normalized, STEPS steps of the
    reference's SGD under forced weight normalization, each step's output, and after the first every
    stored weight and gain in one vector."""
    reference = generator()
    model = edm2_unet(reference.CONFIG)
    variables, x, noise, text, probe = inputs_at(dtype, inputs)
    params = jax.tree_util.tree_map_with_path(
        lambda path, value: normalize(value, compensated=True) if path[-1].key == "mp_kernel" else value,
        variables["params"])
    optimizer = optax.chain(optax.sgd(reference.SGD["lr"]), forced_weight_normalization())
    state = optimizer.init(params)

    def loss(params):
        output = model.apply({**variables, "params": params}, x, noise, text)
        return jnp.sum(output * probe), output

    found = {}
    for index in range(reference.STEPS):
        (_, output), grads = jax.value_and_grad(loss, has_aux=True)(params)
        found[f"output_{index}"] = jnp.moveaxis(output, -1, 1)
        if index:
            found[f"stored_{index}"] = jnp.concatenate([jnp.ravel(leaf)
                                                        for leaf in jax.tree_util.tree_leaves(params)])
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    return found


def test_forced_weight_normalization_trains_as_edm2_does():
    """EDM2 normalizes a weight in place at each training forward and uses
    that weight normalized again (`MPConv.forward`, whose `w` aliases the
    parameter it overwrites); Dew stores the normalized weight after each
    update (`forced_weight_normalization`) and normalizes it at use. Both
    use normalize(normalize(w + update)) and store normalize(w + update), at
    a step's boundary apart, so from the reference's initial weights
    normalized, under the same SGD, they train alike: each step's output,
    and every stored weight and gain after it, as exact as the reference's
    float32 trajectory against its float64 one over drawn weights and inputs
    (check 1)."""
    assert_as_exact_as_recorded(FIXTURE, "train", declared_seeds(FIXTURE)["train"], generator().draw,
                                trained, "edm2 forced weight normalization")


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


def test_edm2s_output_head_sums_in_the_runs_output_dtype():
    """EDM2's final projection multiplies its bf16 compute's operands and sums them in the run's output
    dtype, float32 by default, returning it (`MPConv`'s `out_dtype`, its convolution and its dense branch
    alike), where sums in bf16 cast afterwards would round each one: within a float32 reduction's bound of
    the float64 sums of the same operands (`assert_fp32_reduction_bound`), which a bf16 result exceeds. A
    model of bf16 compute returns float32, its time embedding in bf16 as every block it conditions reads
    it; one of no dtype computing in float64 returns float64."""
    def product(kernel_size: tuple[int, ...], x, weight):
        if kernel_size:
            return jax.lax.conv_general_dilated(x, weight, (1, 1), "SAME",
                                                dimension_numbers=("NHWC", "HWIO", "NHWC"))
        return x @ weight

    x = jnp.asarray(np.random.default_rng(3).normal(size=(2, 8, 8, 64)), jnp.bfloat16)
    for kernel_size, operands in (((3, 3), x), ((), x.reshape(-1, 64))):
        conv = MPConv(8, kernel_size)
        params = conv.init(jax.random.key(1), operands)
        wide = conv.apply(params, operands, 1.0, out_dtype=jnp.float32)
        assert wide.dtype == jnp.float32 and conv.apply(params, operands, 1.0).dtype == jnp.bfloat16
        kernel = params["params"][MP_KERNEL]
        weight = normalize(kernel.astype(jnp.float32), compensated=True) / np.sqrt(np.prod(kernel.shape[:-1]))
        given = np.asarray(operands, np.float64), np.asarray(weight.astype(operands.dtype), np.float64)
        with jax.enable_x64(new_val=True):
            truth = np.asarray(product(kernel_size, *given))
            magnitudes = np.asarray(product(kernel_size, *map(np.abs, given)))
        assert_fp32_reduction_bound(wide, truth, magnitudes, int(np.prod(kernel.shape[:-1])))

    model = EDM2UNet(output_channels=4, model_channels=16, channel_mult=(1,), num_blocks=1,
                     attn_resolutions=(), channels_per_head=16, dtype=jnp.bfloat16)
    images, time = jnp.ones((1, 8, 8, 4), jnp.float32), jnp.ones((1,), jnp.float32)
    params = model.init(jax.random.key(0), images, time)
    params = {**params, "params": {**params["params"], "out_gain": jnp.ones(())}}
    out, held = model.apply(params, images, time, capture_intermediates=True, mutable=["intermediates"])
    assert out.dtype == jnp.float32 and bool(jnp.any(out != 0))
    assert held["intermediates"]["emb_noise"]["__call__"][0].dtype == jnp.bfloat16
    with jax.enable_x64(new_val=True):
        widened = jax.tree.map(lambda leaf: leaf.astype(jnp.float64), params)
        out = model.clone(dtype=None).apply(widened, images.astype(jnp.float64), time.astype(jnp.float64))
        assert out.dtype == jnp.float64
