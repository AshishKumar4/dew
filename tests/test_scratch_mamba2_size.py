import functools
import time

import jax
import jax.numpy as jnp
import pytest
import test_mixers as tm
from reference_error import seeds_for
from residual_orders import orders, permuted, residual_width


@pytest.mark.parametrize("heads, width, state, groups", [(4, 16, 16, 2), (8, 16, 32, 2), (8, 32, 32, 4)])
def test_size(heads, width, state, groups):
    fields = {"num_heads": heads, "head_dim": width, "state_size": state, "n_groups": groups, "chunk_size": 64}
    documents = tm.DOCUMENTS
    total = sum(documents)
    model = tm.tiny(mixer=tm.mixers.build("mamba2", **fields), max_seq_len=max(128, total + tm.PADDING))
    tokens = jax.random.randint(jax.random.key(3), (1, total), 0, tm.VOCAB)
    variables = model.init(jax.random.key(4), tokens)
    initialize = jax.jit(model.init)
    span = residual_width(variables)

    def draw(seed):
        tokens = jax.random.randint(jax.random.key(3 + 3 * seed), (1, total), 0, tm.VOCAB)
        variables = initialize(jax.random.key(4 + 3 * seed), tokens)
        cotangent = jax.random.normal(jax.random.key(5 + 3 * seed), (1, total, tm.VOCAB))
        return permuted(variables, orders(span, 2, seed)[1]), tokens, cotangent

    def evaluated(model, variables, tokens, cotangent):
        with jax.default_matmul_precision("highest"):
            logits, grads = tm.pulled(tm.documents_alone(model, tokens, documents), variables, cotangent)
        return {"logits": logits, "grads": grads}

    reference = jax.jit(functools.partial(evaluated, model))
    truth = jax.jit(functools.partial(evaluated, model.clone(dtype=jnp.float64)))
    jax.block_until_ready(reference(*draw(0)))
    started = time.perf_counter()
    for seed in range(1, 5):
        jax.block_until_ready(reference(*draw(seed)))
    per_seed = (time.perf_counter() - started) / 4
    try:
        count = seeds_for(reference, draw, "mamba2", truth_fn=truth)
    except AssertionError as error:
        count = str(error)[-200:]
    raise AssertionError(f"SIZE|{heads},{width},{state},{groups}|{count}|ref per seed {per_seed:.3f}s")
