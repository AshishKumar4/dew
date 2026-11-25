"""Native Check 1 for named gated MLP routes at a published model width."""

import jax.numpy as jnp
import pytest
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import generation

SEEDS = 8


@pytest.mark.gpu(reason="the model-width wide and paired products must hold on their native launches")
@pytest.mark.generations("sm80", "sm90")
@pytest.mark.parametrize(("target", "form"), [("sm80", "wide"), ("sm90", "wide"), ("sm90", "fused")])
def test_a_model_width_gated_mlp_holds_native_check_1(target, form):
    import generation_support
    from contract_support import normal_draw
    from tools.timing_cases import DENSE, TOKENS

    from dew.nn.mlp import GatedMLP

    if generation_support.here() != target:
        pytest.skip(f"this launch is qualified on {target}")
    hidden, width = DENSE["qwen3.5-4b down"]
    layer = GatedMLP(hidden_features=hidden, out_features=width, dtype=jnp.bfloat16)

    def parameters(gate, up, down):
        return {"params": {"gate_proj": {"kernel": gate}, "up_proj": {"kernel": up},
                           "down_proj": {"kernel": down}}}

    def draw(seed):
        gate, up = normal_draw([1958, seed], (width, hidden), (width, hidden), scale=width ** -0.5,
                               dtype=jnp.bfloat16)
        down, = normal_draw([1959, seed], (hidden, width), scale=hidden ** -0.5, dtype=jnp.bfloat16)
        x, dy = normal_draw([1960, seed], (TOKENS, width), (TOKENS, width), dtype=jnp.bfloat16)
        return parameters(gate, up, down), x, dy

    def under(route, call):
        def arm(*args):
            with (generation.chosen(grouped_matmul="mosaic", gated_mlp=route), generation.routes() as routes):
                result = call(*args)
            assert {entry.kernel for entry in routes if entry.choice == "gated_mlp"} == {route}
            return result
        return arm

    run = with_gradients(layer.apply)
    truth = with_gradients(layer.clone(dtype=jnp.float64).apply)
    assert_as_exact(under(form, run), under("apart", run), draw, f"{target} {form} Qwen3.5-4B MLP",
                    seeds=SEEDS, truth_fn=truth)
