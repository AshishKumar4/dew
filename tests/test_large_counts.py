"""Count arithmetic without allocating the batches those counts describe."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.moe import load_balance_update


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint32, jnp.int64, jnp.uint64])
def test_router_direction_keeps_one_count_differences_when_total_overflows(dtype):
    with jax.enable_x64(np.dtype(dtype).itemsize == 8):
        maximum = np.iinfo(dtype).max
        cases = [[2**30, 2**30 + 1, 2**30 - 1],
                 [maximum, maximum - 1, maximum - 2],
                 [maximum, maximum], [maximum - 1, maximum], [0, 0, 0]]
        update = jax.jit(lambda counts: load_balance_update(counts, .125))
        for values in cases:
            total = sum(values)
            expected = [.125 * ((total > len(values) * x) - (total < len(values) * x))
                        for x in values]
            np.testing.assert_array_equal(update(jnp.asarray(values, dtype)), expected)


@pytest.mark.parametrize("dtype,experts", [(jnp.int8, 129), (jnp.uint8, 256),
                                         (jnp.int16, 32769), (jnp.uint16, 65536)])
@pytest.mark.parametrize("compiled", [False, True])
def test_router_direction_does_not_narrow_the_expert_divisor(dtype, experts, compiled):
    counts = jnp.ones(experts, dtype).at[0].set(0)
    update = jax.jit(load_balance_update) if compiled else load_balance_update
    expected = np.full(experts, -.125)
    expected[0] = .125
    np.testing.assert_array_equal(update(counts, .125), expected)


@pytest.mark.parametrize("sequence", [False, True])
def test_router_loss_is_invariant_to_large_window_replication(sequence):
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives import Step
    from dew.objectives.lm import LMObjective

    model = CausalTransformer(vocab_size=8, emb_features=8, num_layers=1,
                              num_heads=2, mlp_features=16, max_seq_len=8,
                              mixture={"experts": 2, "top_k": 1})
    objective = LMObjective(model, 4, head_chunks=1, aux_loss_alpha=.2, seq_aux=sequence)
    key = jax.random.key(0)
    variables = objective.init(key)
    batch = {"text": jnp.ones((2, 5), jnp.int32)}
    stats, _ = objective.loss(variables, batch, Step(jnp.array(0), key, None))
    expected, _ = objective.reduce_loss(stats)
    # Doubling scalar statistics represents 2**31 rows without any token buffer.
    repeated = jax.lax.fori_loop(0, 30, lambda _, s: jax.tree.map(lambda x: x + x, s), stats)
    actual, active = objective.reduce_loss(repeated)
    assert bool(active)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_diffusion_support_accepts_large_abstract_image_batches():
    from flax import linen as nn
    from dew.diffusion import presets
    from dew.inputs import Field, InputSpec
    from dew.objectives import Step
    from dew.objectives.diffusion import DiffusionObjective

    class Denoiser(nn.Module):
        def __call__(self, x, t, train=False):
            return x

    objective = DiffusionObjective(Denoiser(), presets.EDM()(),
                                   InputSpec(Field("image", (65536, 1, 1))))
    batch = {"image": jax.ShapeDtypeStruct((65536, 65536, 1, 1), jnp.float32)}
    stats, _ = jax.eval_shape(objective.loss, {"encoders": {}}, batch,
                              Step(jnp.array(0), jax.random.key(0), None))
    # A floating support cannot wrap to zero at 2**32, unlike an integer sum.
    assert jnp.issubdtype(stats.mass.dtype, jnp.floating)


@pytest.mark.parametrize("kind", ["dpo", "grpo", "masked", "jepa"])
def test_other_objectives_keep_large_abstract_support_floating(kind):
    from dew.diffusion.discrete import MDLM
    from dew.inputs import Field
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.jepa import JepaPredictor
    from dew.objectives import Step
    from dew.objectives.diffusion import MaskedDiffusionObjective
    from dew.objectives.jepa import JepaEncoder, JepaObjective
    from dew.objectives.jepa.masking import MultiBlockMask
    from dew.objectives.rl.grpo import GRPOObjective
    from dew.objectives.rl.preference import DPOObjective

    if kind == "jepa":
        objective = JepaObjective(
            JepaEncoder(patch_size=2, emb_features=8, num_layers=1, num_heads=2),
            JepaPredictor(grid=(2, 2), emb_features=8, predictor_features=8,
                          num_layers=1, num_heads=2),
            MultiBlockMask((2, 2), 2, ((1, 1),), 2), Field("image", (4, 4, 1)))
        batch = {"image": jax.ShapeDtypeStruct((2**28, 4, 4, 1), jnp.float32)}
    else:
        model = CausalTransformer(vocab_size=8, emb_features=8, num_layers=1,
                                  num_heads=2, mlp_features=16, max_seq_len=8,
                                  causal=kind != "masked")
        if kind == "dpo":
            objective = DPOObjective(model, 4, head_chunks=1)
            batch = {name: jax.ShapeDtypeStruct((2**31, 2, 5), jnp.int32)
                     for name in ("input_ids", "completion_mask")}
        elif kind == "masked":
            objective = MaskedDiffusionObjective(model, MDLM(mask_id=7)(), 4, head_chunks=1)
            batch = {"text": jax.ShapeDtypeStruct((2**30, 4), jnp.int32)}
        else:
            objective = GRPOObjective(model, 4, head_chunks=1)
            batch = {name: jax.ShapeDtypeStruct((2**30, 2), jnp.float32)
                     for name in ("old_log_probs", "advantages")}
            batch["response_mask"] = jax.ShapeDtypeStruct((2**30, 2), jnp.int32)
            batch["input_ids"] = jax.ShapeDtypeStruct((2**30, 5), jnp.int32)
    key = jax.random.key(0)
    variables = objective.init(key)
    stats, _ = jax.eval_shape(objective.loss, variables, batch, Step(jnp.array(0), key, variables))
    assert jnp.issubdtype(stats.mass.dtype, jnp.floating)
