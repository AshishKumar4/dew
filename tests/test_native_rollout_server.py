"""Dew's own continuous-batching server as a rollout server.

A push lands between steps and later submissions report its version; the
served tree keeps the server's precision; a refused request raises at
submission and leaves the rest running.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inference import NativeRolloutServer, TextGeneration
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.rl import GRPOObjective
from dew.sampling import Sampling

VOCAB = 13
EOS = 12
BUDGET = 4
SAMPLING = Sampling(temperature=1.0, eos_id=EOS)


def objective():
    model = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=64, dtype="float32")
    return GRPOObjective(model, seq_len=9)


def test_the_native_server_serves_loaded_weights_at_its_own_precision():
    target = objective()
    params = target.init(jax.random.key(0))
    served = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), params)
    backend = Server.from_task(TextGeneration(target.model, served, None, sampling=SAMPLING), slots=2, capacity=64)
    server = NativeRolloutServer(backend)
    try:
        before = server.submit([1, 2, 3], BUDGET, seed=5).result()
        trained = jax.tree.map(lambda leaf: leaf * 1.5, params)
        server.load(trained, 7)
        after = server.submit([1, 2, 3], BUDGET, seed=5).result()
    finally:
        server.close()
    assert (before.version, after.version) == (0, 7)
    assert before.behavior_log_probs != after.behavior_log_probs
    for leaf, expected in zip(jax.tree.leaves(backend.variables), jax.tree.leaves(trained), strict=True):
        assert leaf.dtype == jnp.bfloat16
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(expected.astype(jnp.bfloat16)))


def test_a_refused_request_raises_at_submit_and_the_native_server_keeps_serving():
    target = objective()
    params = target.init(jax.random.key(0))
    server = NativeRolloutServer(Server.from_task(TextGeneration(target.model, params, None, sampling=SAMPLING),
                                                  slots=2, capacity=64))
    try:
        running = server.submit([1, 2, 3], 40, seed=1)
        with pytest.raises(ValueError, match="max_seq_len"):
            server.submit(list(range(1, 11)) * 6, 10, seed=2)
        assert len(running.result(timeout=120).tokens) >= 1
        assert server.submit([4, 5], 3, seed=3).result(timeout=120).prompt == (4, 5)
    finally:
        server.close()
