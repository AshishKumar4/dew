"""Float-range and nested-derivative contracts of attention's seams: the fused attention call in forward
mode, its kernels' reverse rules, the paged decode kernel (ledger 2641; the harness is
tests/contract_support.py)."""

import itertools

import jax
import jax.numpy as jnp
import pytest
from contract_support import EDGES, REVERSE_ONLY, assert_contract, normal_draw

from dew.nn import attention, kv_cache


def plain_attention(query, key, value):
    """Softmax attention as its formula reads, `[B, S, H, D]` each: every query's logits over every key at
    the width's inverse root, their softmax's weights on the values (none of Dew's attention code)."""
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) * query.shape[-1] ** -0.5
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, axis=-1), value)


def attention_draw(seed: int, shape: tuple, edge: str) -> tuple:
    """A query, key and value of `shape` at `edge`'s scale, a direction for each and an output cotangent."""
    primals = normal_draw(seed, shape, shape, shape, scale=EDGES[edge])
    direction = normal_draw(seed + 1, shape, shape, shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(seed + 2, shape)
    return primals, direction, cotangent


@pytest.mark.parametrize("edge", EDGES)
def test_a_fused_attention_call_in_forward_mode_holds_every_order(edge):
    """`attention.forward_differentiable` (a fused call under `forward_mode_attention`: the template's value,
    the reference path's tangent, a custom JVP): every order runs, finite wherever float64 is a normal
    float32 and the plain attention formula is, at queries, keys and values whose products near float32's
    ends."""
    primals, direction, cotangent = attention_draw(2654, (1, 16, 2, 16), edge)

    def fused(query, key, value):
        with attention.forward_mode_attention():
            return attention.scaled_dot_product_attention(query, key, value, implementation="template")

    assert_contract(fused, primals, direction, cotangent, f"a fused call in forward mode {edge}",
                    truth_fn=plain_attention, plain_fn=plain_attention)


@pytest.mark.parametrize(("implementation", "edge"), list(itertools.product(("template", "tpu"), EDGES)))
def test_a_fused_attention_kernel_holds_its_reverse_orders(implementation, edge):
    """The template's route (`attention_route._attend`) and splash (`splash._attention`, interpreted off a
    TPU), each a reverse-only custom derivative: its primal and its reverse derivative run, finite wherever
    float64 is a normal float32 and the plain attention formula is, at queries, keys and values whose
    products near float32's ends. Its other orders do not run (`REVERSE_ONLY`, ledger 2654)."""
    primals, direction, cotangent = attention_draw(2657, (1, 128, 2, 64), edge)

    def fused(query, key, value):
        return attention.scaled_dot_product_attention(query, key, value, implementation=implementation)

    assert_contract(fused, primals, direction, cotangent, f"{implementation} attention {edge}",
                    truth_fn=plain_attention, plain_fn=plain_attention, without=REVERSE_ONLY)


@pytest.mark.parametrize("edge", EDGES)
def test_the_paged_decode_kernel_holds_every_order(edge):
    """`kv_cache.gpu_paged` (the Pallas paged decode kernel's value, the gathered attention's derivatives
    through `generation.linear_tangent`), interpreted off a GPU: every order runs in the query, the key
    and value pages and the sinks, finite wherever float64 is a normal float32 and the plain formula, one
    query a row over its row's first `lengths` keys gathered from its pages with the sinks in the
    softmax, is."""
    table = jnp.asarray([[2, 0], [0, 3]], jnp.int32)
    lengths = jnp.asarray([21, 32], jnp.int32)
    shapes = ((2, 4, 64), (2, 4, 16, 64), (2, 4, 16, 64), (4,))
    primals = normal_draw(2660, *shapes, scale=EDGES[edge])
    direction = normal_draw(2661, *shapes, scale=EDGES[edge])
    (cotangent,) = normal_draw(2662, shapes[0])

    def paged(query, key, value, sinks):
        return kv_cache.gpu_paged(None)(query, key, value, table, lengths, None, sinks)

    def plain(query, key, value, sinks):
        # Each row's pages in its table's order, `[rows, keys, key heads, D]`, each key head read by two
        # query heads.
        keys, values = (jnp.repeat(jnp.moveaxis(pool[:, table], 0, 3).reshape(2, 32, 2, 64), 2, axis=2)
                        for pool in (key, value))
        logits = jnp.einsum("rhd,rkhd->rhk", query, keys) * 64 ** -0.5
        logits = jnp.where(jnp.arange(keys.shape[1]) < lengths[:, None, None], logits, -jnp.inf)
        sunk = jnp.broadcast_to(sinks[None, :, None], (2, 4, 1))
        weights = jax.nn.softmax(jnp.concatenate([logits, sunk], axis=-1), axis=-1)[..., :-1]
        return jnp.einsum("rhk,rkhd->rhd", weights, values)

    assert_contract(paged, primals, direction, cotangent, f"the paged decode kernel {edge}",
                    truth_fn=plain, plain_fn=plain)
