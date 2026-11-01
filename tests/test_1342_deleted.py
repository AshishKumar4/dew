"""CPU-sized reproduction of the row-1342 H100 banked warm-up failure."""

import jax
import jax.numpy as jnp
import pytest

from dew.inference import TextGeneration
from dew.inference.banks import HeldBanks
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.sampling import Sampling


def smoke_model():
    return CausalTransformer(
        vocab_size=512, emb_features=64, num_layers=2, num_heads=4,
        num_kv_heads=2, head_dim=16, mlp_features=256, max_seq_len=32,
        qk_norm=True, dtype=jnp.bfloat16, scan_layers=True,
        mixture={"experts": 8, "top_k": 2, "expert_features": 32,
                 "layers": (0, 1)})


def smoke_weights(model):
    shapes = jax.eval_shape(model.init, jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    keys = jax.tree.unflatten(
        jax.tree.structure(shapes),
        list(jax.random.split(jax.random.key(0), len(jax.tree.leaves(shapes)))))
    return jax.block_until_ready(jax.jit(lambda: jax.tree.map(
        lambda leaf, key: (jax.random.normal(key, leaf.shape, jnp.bfloat16) * 0.02).astype(
            jnp.bfloat16), shapes, keys))())


@pytest.mark.parametrize("release_source", [False, True], ids=["keep-source", "delete-source"])
def test_banked_smoke_serve(release_source):
    model = smoke_model()
    source = smoke_weights(model)
    banked = jax.block_until_ready(HeldBanks(source).place(model))
    embedding = source["params"]["embed_tokens"]["embedding"]
    placed = banked["params"]["embed_tokens"]["embedding"]
    print("embedding before deletion:", {
        "same_object": embedding is placed,
        "source_sharding": str(embedding.sharding),
        "placed_sharding": str(placed.sharding),
        "source_deleted": embedding.is_deleted(),
        "placed_deleted": placed.is_deleted(),
    })
    if release_source:
        for leaf in jax.tree.leaves(source):
            leaf.delete()
    print("embedding after deletion:", {
        "source_deleted": embedding.is_deleted(), "placed_deleted": placed.is_deleted(),
    })
    task = TextGeneration(model, banked, None, sampling=Sampling(temperature=0))
    server = Server.from_task(task, slots=2, capacity=32, admission=1)
    results = [result.host() for result in server([[100, 101, 102, 103]], 2, key=0)]
    assert results
