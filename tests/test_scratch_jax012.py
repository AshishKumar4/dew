import jax
import jax.numpy as jnp
import pytest
import test_kernel_configs as kc
from jax._src.interpreters import partial_eval as pe
from reference_error import equations

from dew.nn.kernels import chunked_launch, fla_gdn, fla_kda, recurrence


@pytest.mark.parametrize("module", [fla_gdn, fla_kda])
def test_sets(module):
    found = {type(ba.op).__name__ for ba in kc.rule_launches(module)}
    shared = {recurrence.ChunkLocalCumsum, *((fla_gdn.ChunkFwdH, fla_gdn.ChunkBwdDhu) if module is fla_kda else ())}
    want = {c.__name__ for c in shared | {type(op) for op in vars(module).values() if isinstance(op, chunked_launch._Launch)}}
    raise AssertionError(f"SETS|{module.__name__}|jax {jax.__version__}|missing {sorted(want - found)}|extra {sorted(found - want)}")


def test_remat():
    from dew.nn.backbones.decoder_block import remat_policy
    from dew.nn.linear import GatedDeltaNet
    from dew.nn.kernels import generation
    with generation.chosen(gated_delta_rule="fla"):
        layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=64)
        x = jax.random.normal(jax.random.key(71), (2, 64, 64))
        params = jax.jit(layer.init)(jax.random.key(73), x)

        def loss(params, x):
            return jnp.sum(layer.apply(params, x) ** 2)

        rung = remat_policy({"save": ("context", "delta_decay", "delta_inverse")})
        remat = jax.checkpoint(loss, policy=rung.checkpoint_policy())
        graph = jax.make_jaxpr(jax.grad(remat))(params, x)
        raw = [str(e.params["name"]) for e in equations(graph) if e.primitive.name == "pallas_call"]
        dced, _ = pe.dce_jaxpr(graph.jaxpr, [True] * len(graph.jaxpr.outvars))
        pruned = [str(e.params["name"]) for e in equations(dced) if e.primitive.name == "pallas_call"]
    raise AssertionError(f"REMAT|jax {jax.__version__}|raw {raw}|dce {pruned}")
