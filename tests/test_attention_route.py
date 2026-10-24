"""dew.nn.attention's `implementation='template'` (`attention_route.attend`) on the CPU's plain-JAX mainloop.

Its output and its gradients of the query, key, value and sinks are on par with Dew's reference path from
float64 (check 1), for each call form the route takes. Every case has grouped key heads: four query heads
over two. The selector refuses the route a mask and a bias by their inherent limits, and no target's
preference picks it, so only a call that names it runs it. On a mesh of two rows by four heads, with fewer
key heads than tensor shards, a named call runs per shard under `sharded.assert_runs_per_shard`.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec
from reference_error import assert_as_exact, with_gradients
from sharded import assert_runs_per_shard

from dew.nn import attention
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.attention_kernels import ATTENTION, kernel_refusals
from dew.nn.kernels import generation
from dew.nn.kernels.generation import INHERENT
from dew.nn.kernels.selection import KernelRefused
from dew.nn.kernels.templates import attention as template, attention_route

ROWS, HEADS, KV_HEADS, WIDTH = 2, 4, 2, 32
QUERIES, KEYS = 100, 96
"""A call's query and key counts, neither a multiple of a tile. A call of its own documents has as many keys
as queries (`geometry`)."""
LENGTHS = jnp.asarray([96, 61], jnp.int32)
"""Each row's keys: the second row ends short of the keys' end."""
STARTS = jnp.asarray([0, 17], jnp.int32)
"""Each row's first key: the second row starts past the first key."""
SOFTCAP = 0.5
"""Gemma 2's logit cap, as the template's own tests set it."""
SEEDS = 8
"""The reference arithmetic's declared draws (check 1), sized by `seeds_for`."""
DTYPES = [jnp.float32, jnp.bfloat16]
MESH_TENSOR, MESH_HEADS = 4, 8
"""The tensor shards of the per-shard test's mesh, beside its `ROWS` data shards, and its query heads: two on
each tensor shard, over one or two key heads, which is fewer than the shards."""

CASES = {
    "full-grouped": {},
    "causal": {"causal": True},
    "causal-window": {"causal": True, "sliding_window": 24},
    "packed-documents": {"segment_ids": "own"},
    "cross-documents-causal": {"causal": True, "segment_ids": "cross"},
    "key-lengths": {"lengths": True},
    "sinks-softcap": {"causal": True, "sinks": True, "softcap": SOFTCAP},
}


def own_documents() -> jax.Array:
    """A call's own packed documents `[2, 100]`: three and two documents, then padding (0)."""
    return jnp.asarray([[1] * 30 + [2] * 40 + [3] * 20 + [0] * 10, [4] * 55 + [5] * 45], jnp.int32)


def cross_documents() -> tuple[jax.Array, jax.Array]:
    """Another sequence's documents: the queries' ids `[2, 100]` in any order, the keys' `[2, 96]`
    ascending with padding after them, and the second row's second document holding no query."""
    queries = jnp.asarray([[2] * 20 + [1] * 50 + [0] * 30, [7] * 100], jnp.int32)
    keys = jnp.asarray([[1] * 60 + [2] * 30 + [0] * 6, [7] * 70 + [8] * 26], jnp.int32)
    return queries, keys


def geometry(case: dict) -> tuple[int, int]:
    """`case`'s query and key counts: a call of its own documents has as many keys as queries."""
    return (QUERIES, QUERIES) if case.get("segment_ids") == "own" else (QUERIES, KEYS)


def keywords(case: dict) -> dict:
    """`case`'s keywords of `scaled_dot_product_attention`, its ids, lengths and starts materialized. The
    sinks are an argument of their own."""
    keywords = {name: case[name] for name in ("causal", "sliding_window", "softcap") if name in case}
    if case.get("segment_ids") == "own":
        keywords["segment_ids"] = own_documents()
    if case.get("segment_ids") == "cross":
        keywords["segment_ids"], keywords["key_segment_ids"] = cross_documents()
    if case.get("lengths"):
        keywords["key_value_seq_lengths"] = LENGTHS
    if case.get("starts"):
        keywords["key_value_seq_starts"] = STARTS
    return keywords


def drawing(queries: int, keys: int, dtype, *, sinks: bool, heads: int = HEADS, kv_heads: int = KV_HEADS):
    """A draw `(query, key, value, sinks, dout)` in `dtype`: queries and the output's cotangent `[2, queries,
    heads, 32]`, keys and values `[2, keys, kv_heads, 32]`, and sinks `[heads]` where `sinks`."""

    def draw(seed: int):
        q_key, k_key, v_key, d_key, s_key = jax.random.split(jax.random.key(seed), 5)
        query = jax.random.normal(q_key, (ROWS, queries, heads, WIDTH), dtype)
        key = jax.random.normal(k_key, (ROWS, keys, kv_heads, WIDTH), dtype)
        value = jax.random.normal(v_key, (ROWS, keys, kv_heads, WIDTH), dtype)
        dout = jax.random.normal(d_key, (ROWS, queries, heads, WIDTH), dtype)
        logits = jax.random.normal(s_key, (heads,), dtype) if sinks else None
        return query, key, value, logits, dout

    return draw


def pulled(implementation: str, case: dict):
    """`case`'s call on `implementation`: its output, then its gradients of the query, key, value and sinks
    under the cotangent `dout`, the sinks' gradient None where the case has none."""

    def run(query, key, value, sinks, dout):
        def call(query, key, value, sinks):
            return scaled_dot_product_attention(query, key, value, implementation=implementation,
                                                sinks=sinks, **keywords(case))

        out, pull = jax.vjp(call, query, key, value, sinks)
        return (out, *pull(dout))

    return run


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
def test_a_named_template_route_is_on_par_with_the_reference_path_forward_and_backward(case, dtype):
    """The route's output and its gradients of the query, key, value and sinks, on par with the reference
    path (`implementation='reference'`) measured from float64 over the site's draws (check 1)."""
    queries, keys = geometry(case)
    assert_as_exact(jax.jit(pulled("template", case)), jax.jit(pulled("reference", case)),
                    drawing(queries, keys, dtype, sinks=bool(case.get("sinks"))),
                    f"template route {case} {jnp.dtype(dtype).name}", seeds=SEEDS)


MESH_CASES = {
    "own-documents": {"causal": True, "segment_ids": "own", "softcap": SOFTCAP},
    "cross-documents": {"causal": True, "segment_ids": "cross"},
    "lengths-and-starts": {"lengths": True, "starts": True},
}
"""The calls the per-shard test takes, each with sinks: its ids, lengths and starts split with their rows."""


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("key_heads", [1, 2], ids=["one-key-head", "two-key-heads"])
@pytest.mark.parametrize("case", MESH_CASES.values(), ids=list(MESH_CASES))
def test_a_named_template_call_keeps_the_kernel_contract_on_a_data_and_head_mesh(case, key_heads):
    """A named template call on a mesh of two rows by four heads, with one or two key heads, fewer than the
    four tensor shards, runs per shard (`sharded.assert_runs_per_shard`): its output and its gradients of the
    query, key, value and sinks are on par with one device's over the site's draws (check 1), measured from
    float64 on the reference path, and the compiled step moves no operand between devices. Each shard holds
    the key heads its query heads read, which `_per_shard` repeats until the tensor axis splits them."""
    queries, keys = geometry(case)
    mesh = Mesh(np.asarray(jax.devices()[:ROWS * MESH_TENSOR]).reshape(ROWS, MESH_TENSOR),
                ("data", "tensor"))
    rows_and_heads = PartitionSpec("data", None, "tensor")
    # Fewer key heads than tensor shards: no shard splits them, so each holds them whole, split by rows alone.
    rows = PartitionSpec("data")

    def routed(query, key, value, sinks):
        return scaled_dot_product_attention(query, key, value, implementation="template", sinks=sinks,
                                            **keywords(case))

    def reference(query, key, value, sinks):
        return scaled_dot_product_attention(query, key, value, implementation="reference", sinks=sinks,
                                            **keywords(case))

    draw = drawing(queries, keys, jnp.float32, sinks=True, heads=MESH_HEADS, kv_heads=key_heads)
    specs = [rows_and_heads, rows, rows, PartitionSpec("tensor"), rows_and_heads]
    assert_runs_per_shard(with_gradients(routed), draw, specs, mesh,
                          f"template split over rows and heads, {key_heads} key heads", seeds=SEEDS,
                          truth_fn=with_gradients(reference))


def test_a_route_call_under_forward_mode_keeps_the_template_value_and_the_reference_derivative():
    """Under `forward_mode_attention` a fused call keeps its kernel's value and takes its tangent from the
    reference path's JVP, so the route's primal and tangent are on par with the reference path's (check 1)."""
    case = CASES["causal-window"]

    def pushed(implementation: str):
        def run(query, key, value, _sinks, dout):
            def call(query, key, value):
                return scaled_dot_product_attention(query, key, value, implementation=implementation,
                                                    **keywords(case))

            with attention.forward_mode_attention():
                return jax.jvp(call, (query, key, value), (dout, value, key))

        return run

    assert_as_exact(jax.jit(pushed("template")), jax.jit(pushed("reference")),
                    drawing(QUERIES, KEYS, jnp.float32, sinks=False), "template route forward mode",
                    seeds=SEEDS)


def test_the_selector_refuses_a_mask_and_a_bias_by_their_inherent_limits():
    """A call with neither takes the route. The same call with a mask, or with a bias, is refused for that one
    thing as an inherent limit, and a call that names the route raises for it: the template reads validity
    as spans and takes no dense bias."""
    query = jnp.zeros((ROWS, QUERIES, HEADS, WIDTH), jnp.bfloat16)
    key = jnp.zeros((ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    plain = attention.attention_case(query, key, named="template")
    assert kernel_refusals(ATTENTION, "template", plain) == []
    dense = {
        "mask": jnp.ones((ROWS, 1, QUERIES, KEYS), bool),
        "bias": jnp.zeros((ROWS, HEADS, QUERIES, KEYS), jnp.bfloat16),
    }
    for name, operand in dense.items():
        case = attention.attention_case(query, key, named="template", **{name: operand})
        refusals = kernel_refusals(ATTENTION, "template", case)
        assert [(refusal.code, refusal.row) for refusal in refusals] == [(name, INHERENT)]
        with pytest.raises(KernelRefused):
            scaled_dot_product_attention(query, key, key, implementation="template", **{name: operand})


def test_no_target_prefers_the_template_so_only_a_named_call_takes_it():
    """On every target's row, 'auto' resolves each call to a kernel other than the template. The template runs
    where a call names it, and where `generation.chosen` names it for the attention choice."""
    query = jnp.zeros((ROWS, QUERIES, HEADS, WIDTH), jnp.bfloat16)
    key = jnp.zeros((ROWS, KEYS, KV_HEADS, WIDTH), jnp.bfloat16)
    sinks = jnp.zeros((HEADS,), jnp.bfloat16)
    calls = [
        {},
        {"causal": True, "sliding_window": 24},
        {"sinks": sinks, "softcap": SOFTCAP},
        {"causal": True, "key_value_seq_lengths": LENGTHS},
    ]
    for target in generation.TARGETS:
        with generation.assumed(target):
            for call in calls:
                auto = attention.resolve_implementation("auto", query, key, **call)
                assert auto != "template", (target, call)
                named = attention.resolve_implementation("template", query, key, **call)
                assert named == "template", (target, call)
    with generation.chosen(attention="template"):
        for call in calls:
            assert attention.resolve_implementation("auto", query, key, **call) == "template", call


def test_the_route_takes_score_hooks_and_the_initial_state_and_refuses_an_output_hook():
    """`attend` runs a score hook and the initial state, and refuses an output hook of its own: its backward
    takes the output before one, so the caller applies it."""
    query, key, value, sinks, _ = drawing(QUERIES, KEYS, jnp.float32, sinks=True)(0)
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    hooks = template.Hooks(score_mod=functools.partial(template.capped, SOFTCAP),
                           initial=template.sinks(sinks, ROWS, QUERIES, WIDTH))
    assert attention_route.attend(query, key, value, spans, hooks).shape == (ROWS, QUERIES, HEADS, WIDTH)
    with pytest.raises(ValueError):
        attention_route.attend(query, key, value, spans, template.Hooks(output_mod=jnp.tanh))
