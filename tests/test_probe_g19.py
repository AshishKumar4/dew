import jax
import jax.numpy as jnp
import numpy as np
import selection_margins

from dew.nn.moe import Router


def test_probe():
    router = Router(num_experts=8, in_features=16, top_k=8, score_function="sigmoid", normalize_weights=True,
                    expert_groups=1, groups_per_token=1)
    hidden = jnp.asarray(np.random.default_rng(0x4394).standard_normal((6, 16)), jnp.float32)
    variables = router.init(jax.random.key(0), hidden)
    captured = selection_margins.capture(router)(variables, hidden)
    print("\nPROBE", jax.tree_util.tree_map(lambda x: x.shape, captured))
    print("PROBE margin", selection_margins.margin({"mixture": {"top_k": 8}}, captured, "x"))
