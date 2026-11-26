"""Native Check 1 for named gated MLP routes at a published model width."""

import generation_support
import jax.numpy as jnp
import pytest
from contract_support import normal_draw, normal_drawn
from reference_error import assert_as_exact, with_gradients
from test_algebra_fusion import gated_projection

from dew import algebra
from dew.nn.kernels import generation
from dew.nn.mlp import GatedMLP
from tools.timing_cases import DENSE, TOKENS

SEEDS = 8


@pytest.mark.gpu(reason="the model-width wide and paired products must hold on their native launches")
@pytest.mark.generations("sm80", "sm90")
@pytest.mark.parametrize(("target", "form"), [("sm80", "wide"), ("sm90", "wide"), ("sm90", "fused")])
def test_a_model_width_gated_mlp_holds_native_check_1(target, form):
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
            choices = {} if route == "apart" else {"grouped_matmul": "mosaic"}
            with generation.chosen(gated_mlp=route, **choices), generation.routes() as routes:
                result = call(*args)
            assert {entry.kernel for entry in routes if entry.choice == "gated_mlp"} == {route}
            return result
        return arm

    run = with_gradients(layer.apply)
    truth = with_gradients(layer.clone(dtype=jnp.float64).apply)
    assert_as_exact(under(form, run), under("apart", run), draw, f"{target} {form} Qwen3.5-4B MLP",
                    seeds=SEEDS, truth_fn=truth)


@pytest.mark.gpu(reason="the fused gated producer and its gradients must hold on the launch's own GPU")
@pytest.mark.generations("sm90")
def test_a_native_gated_projection_prologue_holds_check_1():
    if generation_support.here() != "sm90":
        pytest.skip("the gated projection's side-tile prologue is qualified on sm90")
    hidden, width = DENSE["qwen3.5-4b down"]
    dims = ((256, hidden), (256, hidden), (hidden, width), (256, width))

    def run(*args):
        with generation.chosen(grouped_matmul="mosaic", gemm_prologue="fused"), generation.routes() as routes:
            out = with_gradients(algebra.rewritten(gated_projection, kernels=True))(*args)
        assert any(r.choice == "gemm_prologue" and r.kernel == "fused" for r in routes)
        return out

    def truth(*args):
        return gated_projection(*(arg.astype(jnp.float64) for arg in args))

    assert_as_exact(run, with_gradients(gated_projection), normal_drawn(3422, *dims, dtype=jnp.bfloat16),
                    "native gated projection prologue", seeds=SEEDS, truth_fn=with_gradients(truth))
