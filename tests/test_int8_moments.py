"""8-bit optimizer state (`dew.optim.int8_moments`, `OptimConfig(state_dtype='int8')`): Adam's moments
stored blockwise in 8 bits, against fp32 Adam."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from learning_gate import assert_learns

from dew.config import OptimConfig
from dew.optim import OPTIMIZER_MAP, QUANTIZED_SIZE, QuantizedMoments, dequantized, dynamic_map, quantized

LR = 1e-2
BLOCK = 256


def solver(**kwargs):
    return OptimConfig(optimizer='adamw', learning_rate=LR, weight_decay=0.0, state_dtype='int8',
                       **kwargs).build(100)


def test_the_dynamic_maps_hold_256_ascending_values_from_their_range():
    """Each map holds 256 ascending values with 0 and 1: the unsigned one's are all at least
    0, and the signed one's negative values mirror its positive ones but 1 (Dettmers' dynamic tree map,
    bitsandbytes' and torchao's `create_dynamic_map`)."""
    for signed in (True, False):
        values = dynamic_map(signed=signed)
        assert values.shape == (256,) and np.all(np.diff(values) >= 0)
        assert values[-1] == 1.0 and 0.0 in values
    signed = dynamic_map(signed=True)
    np.testing.assert_array_equal(np.sort(-signed[signed < 0]), signed[(signed > 0) & (signed < 1)])
    assert np.all(dynamic_map(signed=False) >= 0)


def test_a_moment_is_coded_by_its_blocks_nearest_value():
    """Each value is coded by the map value nearest its fraction of its block's largest magnitude, and
    decoded as that value times the block's scale: a block's positive peak, the map's 1, comes back
    exactly, and a negative one as the signed map's least value, which stops short of -1 (the map
    bitsandbytes' and torchao's share)."""
    moment = jnp.asarray(np.random.default_rng(0).standard_normal((4, 2 * BLOCK)), jnp.float32)
    values = jnp.asarray(dynamic_map(signed=True))
    codes, scales = quantized(moment, BLOCK, signed=True)
    assert codes.dtype == jnp.uint8 and scales.shape == (4, 2)
    blocked = np.asarray(moment).reshape(4, 2, BLOCK)
    fractions = (blocked / np.asarray(scales)[..., None]).reshape(4, 2 * BLOCK)
    nearest = np.argmin(np.abs(np.asarray(values)[None, None, :] - fractions[..., None]), axis=-1)
    np.testing.assert_array_equal(np.asarray(codes), nearest)
    back = np.asarray(dequantized(codes, scales, values, BLOCK)).reshape(4, 2, BLOCK)
    peaks = np.abs(blocked).argmax(-1)[..., None]
    peak, decoded = np.take_along_axis(blocked, peaks, -1), np.take_along_axis(back, peaks, -1)
    least = np.asarray(values)[0] * np.asarray(scales)[..., None]
    np.testing.assert_array_equal(decoded, np.where(peak > 0, peak, least))


def searched(values: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """The code a search of the whole sorted map gives each of `fractions`, in NumPy: the value below it,
    or the one above where it is halfway there or past, torchao's `quantize_8bit_with_qmap` rule."""
    below = np.clip(np.searchsorted(values, fractions, side="right") - 1, 0, len(values) - 1)
    above = np.minimum(below + 1, len(values) - 1)
    halfway = (values[above] - values[below]) * np.float32(0.5)
    return np.where(fractions - values[below] >= halfway, above, below)


@pytest.mark.parametrize("signed", [True, False], ids=["signed", "unsigned"])
def test_a_values_code_is_the_one_a_search_of_the_whole_map_gives(signed):
    """`quantized` looks for each value's code among the few around its estimate (`_estimated_codes`), and
    finds the one a search of the whole map does, on every value where the search turns: each map value,
    the halfway points between neighbours, the float32 numbers either side of both, and 0, together with
    a quarter million values spread evenly over each decade's logarithm from 1e-9 to 1, of either sign
    where the map is signed. Each block's peak is 1, so a value is its own fraction."""
    values = dynamic_map(signed=signed)
    halves = values[:-1] + (values[1:] - values[:-1]) * np.float32(0.5)
    turns = np.concatenate([values, halves])
    rng = np.random.default_rng(0)
    spread = np.float32(10) ** rng.uniform(-9, 0, 1 << 18).astype(np.float32)
    if signed:
        spread *= rng.choice(np.asarray([-1, 1], np.float32), spread.shape)
    either = [np.nextafter(turns, np.float32(side)) for side in (-2, 2)]
    fractions = np.concatenate([turns, *either, np.zeros(1, np.float32), spread])
    fractions = np.clip(fractions, -1 if signed else 0, 1).astype(np.float32)
    rows = -(-fractions.size // (BLOCK - 1))
    laid = np.zeros((rows, BLOCK - 1), np.float32).reshape(-1)
    laid[:fractions.size] = fractions
    moment = np.concatenate([np.ones((rows, 1), np.float32), laid.reshape(rows, BLOCK - 1)], axis=1)
    codes, scales = quantized(jnp.asarray(moment), BLOCK, signed=signed)
    np.testing.assert_array_equal(np.asarray(scales), np.ones((rows, 1), np.float32))
    np.testing.assert_array_equal(np.asarray(codes), searched(values, moment).astype(np.uint8))


def test_int8_state_takes_optax_adamw_steps_from_the_fp32_moments():
    """The update is computed from the moments dequantized to fp32, so the first, from zero moments, which
    8 bits hold exactly, is fp32 AdamW's (`OPTIMIZER_MAP`), bit for bit. A leaf of `QUANTIZED_SIZE`
    values or more whose last axis the block divides keeps 8-bit codes and float32 block scales, a smaller
    one float32 moments; the state is a quarter of float32's and its blocks' scales."""
    params = {"big": jnp.zeros((16, 2 * BLOCK), jnp.float32), "small": jnp.zeros((8, 8), jnp.float32)}
    grads = jax.tree.map(lambda leaf: jax.random.normal(jax.random.key(1), leaf.shape) * 1e-2, params)
    reference, int8 = OPTIMIZER_MAP['adamw'](LR, weight_decay=0.0), solver()
    expected, _ = reference.update(grads, reference.init(params), params)
    update, state = int8.update(grads, int8.init(params), params)
    jax.tree.map(np.testing.assert_array_equal, update, expected)
    moments = state.inner_state[0]
    assert params["big"].size >= QUANTIZED_SIZE > params["small"].size
    for moment in (moments.mu, moments.nu):
        assert isinstance(moment, QuantizedMoments) and moment.codes["big"].dtype == jnp.uint8
        assert moment.scales["big"].shape == (16, 2)
        assert moment.codes["small"].dtype == jnp.float32 and moment.scales["small"] is None
        stored = moment.codes["big"].nbytes + moment.scales["big"].nbytes
        assert stored * 4 <= params["big"].nbytes * (1 + 4 / BLOCK)


UNITS = 8
STEPS = 200


def test_int8_state_trains_as_fp32_adam_does():
    """Over independent regressions (their own weights, data and targets), each trained 200 steps by
    AdamW with fp32 and with 8-bit moments from one start, the 8-bit run lowers the loss by at least half
    what the fp32 run does (the learning gate, `assert_learns`): its improvement twice over less the fp32
    run's, in units of the fp32 run's improvement, which each unit's fp32 run makes. Each layer holds 4096
    values or more, so its moments are 8-bit."""
    int8, reference = solver(), OPTIMIZER_MAP['adamw'](LR, weight_decay=0.0)

    def loss(params, x, y):
        hidden = jnp.tanh(x @ params["w1"])
        return jnp.mean((hidden @ params["w2"] - y) ** 2)

    def trained(optimizer, params, x, y):
        def step(carry, _):
            params, state = carry
            gradient = jax.grad(loss)(params, x, y)
            update, state = optimizer.update(gradient, state, params)
            return (optax.apply_updates(params, update), state), None
        (params, _), _ = jax.lax.scan(step, (params, optimizer.init(params)), None, length=STEPS)
        return loss(params, x, y)

    def unit(key):
        keys = jax.random.split(key, 5)
        params = {"w1": jax.random.normal(keys[0], (16, BLOCK)) / 4,
                  "w2": jax.random.normal(keys[1], (BLOCK, BLOCK)) / 16}
        x = jax.random.normal(keys[2], (64, 16))
        teacher = jnp.tanh(x @ jax.random.normal(keys[3], (16, BLOCK)) / 4)
        y = teacher @ jax.random.normal(keys[4], (BLOCK, BLOCK)) / 16
        return loss(params, x, y), trained(int8, params, x, y), trained(reference, params, x, y)

    before, int8_after, fp32_after = (np.asarray(value, np.float64) for value in
                                      jax.jit(jax.vmap(unit))(jax.random.split(jax.random.key(0), UNITS)))

    improvement, reference_improvement = before - int8_after, before - fp32_after
    assert np.all(reference_improvement > 0)

    def contrasts(units):
        return {"8-bit AdamW lowers the loss by half fp32's at least":
                ((2 * improvement - reference_improvement) / reference_improvement)[units]}

    assert_learns(contrasts, "8-bit moments", units=UNITS)


def split_dims(spec, ndim: int, axis: str) -> list[int]:
    """The dimensions of a leaf of `ndim` dimensions its sharding `spec` splits over the mesh's `axis`."""
    names = tuple(spec) + (None,) * (ndim - len(spec))
    return [dim for dim, name in enumerate(names) if axis in jax.tree.leaves(name)]


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis", ["tensor", "expert"])
def test_int8_moments_are_placed_as_their_parameters_are(axis):
    """A moment's codes and block scales are trees of the parameters' own paths, so a layout places each as
    its parameter's declared axes say: over a tensor axis of two devices a projection's codes are split as
    its kernel is, over an expert axis an expert's are, and the scales too, along each dimension of the
    parameter's split that their blocks divide by the axis."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Layout, MeshSpec, Trainer

    model = CausalTransformer(vocab_size=256, emb_features=512, num_layers=1, num_heads=2, mlp_features=1024,
                              max_seq_len=16, dtype=jnp.float32,
                              mixture={"experts": 4, "top_k": 2} if axis == "expert" else None)
    # Every leaf splits where it divides, the scales' few bytes too (`min_shard`).
    trainer = Trainer(LMObjective(model, 8), solver(), key=jax.random.key(0), mesh=MeshSpec(**{axis: 2}),
                      layout=Layout(min_shard=1, tolerance=1.0))
    state, _, _ = trainer.place()
    params = state.variables["params"]
    moments = [node for node in jax.tree.leaves(state.opt_state, is_leaf=lambda node: isinstance(
        node, QuantizedMoments)) if isinstance(node, QuantizedMoments)]
    assert len(moments) == 2
    split = 0
    for moment in moments:
        assert jax.tree.structure(moment.codes) == jax.tree.structure(params)
        scales = jax.tree.leaves(moment.scales, is_leaf=lambda leaf: leaf is None)
        for param, codes, scale in zip(jax.tree.leaves(params), jax.tree.leaves(moment.codes), scales,
                                       strict=True):
            assert codes.sharding.spec == param.sharding.spec
            dims = split_dims(param.sharding.spec, param.ndim, axis)
            split += bool(dims)
            if scale is not None:
                assert split_dims(scale.sharding.spec, scale.ndim, axis) == [
                    dim for dim in dims if scale.shape[dim] % 2 == 0]
    assert split > 0
