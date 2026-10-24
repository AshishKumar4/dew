"""dew.nn.attention's `implementation='template'` (`attention_route.attend`) on the CPU's plain-JAX mainloop.

Its output and its gradients of the query, key, value and sinks are on par with Dew's reference path from
float64 (check 1), for each call form the route takes. Every case has grouped key heads: four query heads
over two. The selector refuses the route a mask and a bias by their inherent limits, and no target's
preference picks it, so only a call that names it runs it. On meshes, a named call runs per shard under
`sharded.assert_runs_per_shard`: with fewer key heads than the tensor shards, with key heads split partly, or
with key heads split in the other order from the query heads.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec
from reference_error import assert_as_exact, with_gradients
from sharded import assert_runs_per_shard
from test_attention_template import (
    DTYPES,
    KEYS,
    LENGTHS,
    QUERIES,
    ROWS,
    SEEDS,
    SOFTCAP,
    STARTS,
    cross_documents,
    geometry,
    own_documents,
)

from dew.nn import attention
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.attention_kernels import ATTENTION, kernel_refusals
from dew.nn.kernels import generation
from dew.nn.kernels.generation import INHERENT
from dew.nn.kernels.selection import KernelRefused
from dew.nn.kernels.templates import attention as template, attention_route
from dew.nn.sharding import DEFAULT_RULES

HEADS, KV_HEADS, WIDTH = 4, 2, 32
MESH_TENSOR, MESH_HEADS = 4, 8
"""The tensor shards of the per-shard test's mesh, beside its `ROWS` data shards, and its query heads: two on
each tensor shard, over one or two key heads, which is fewer than the shards."""
MESH_SEEDS = 512
"""The per-shard test's draws, sized by `seeds_for` with one device compared with
itself: the fewest, doubling from `FEWEST_SEEDS`, that resolve the factor for each
case, which is 512 for lengths and starts over two key heads."""

CASES = {
    "full-grouped": {},
    "causal": {"causal": True},
    "causal-window": {"causal": True, "sliding_window": 24},
    "packed-documents": {"segment_ids": "own"},
    "cross-documents-causal": {"causal": True, "segment_ids": "cross"},
    "key-lengths": {"lengths": True},
    "sinks-softcap": {"causal": True, "sinks": True, "softcap": SOFTCAP},
}


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


def template_call(case: dict, implementation: str = "template"):
    """`case`'s call of `scaled_dot_product_attention` on `implementation`, as a function of
    `(query, key, value, sinks)`."""

    def call(query, key, value, sinks):
        return scaled_dot_product_attention(query, key, value, implementation=implementation, sinks=sinks,
                                            **keywords(case))

    return call


def pulled(implementation: str, case: dict):
    """`case`'s call on `implementation`: its output, then its gradients of the query, key, value and sinks
    under the cotangent `dout`, the sinks' gradient None where the case has none."""

    def run(query, key, value, sinks, dout):
        out, pull = jax.vjp(template_call(case, implementation), query, key, value, sinks)
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
WIDE_HEADS = 24
"""The query heads of the partly split mesh tests: eight ways over the tensor and expert axes, three on each
shard."""
WIDE_SHARDS = 8
"""The shards those query heads split over: a tensor axis of two and an expert axis of four."""
QUERY_HEADS_EIGHT_WAYS = (("activation_heads", ("tensor", "expert")),)
"""The query heads' rule when they split over the tensor and expert axes together, as a trainer's layout may
split them."""
PARTLY_SPLIT_KEY_HEADS = {
    6: (*QUERY_HEADS_EIGHT_WAYS, ("activation_kv", ("tensor", "expert")), ("activation_kv", "tensor"),
        *DEFAULT_RULES),
    8: (*QUERY_HEADS_EIGHT_WAYS, ("activation_kv", "tensor"), *DEFAULT_RULES),
}
"""Rules that split the key heads over the tensor axis alone: six, which the full split does not divide, and
eight, which the tensor axis splits only in part of the query heads' eight ways."""
KEY_HEADS_LEADING_THE_OTHER_WAY = (*QUERY_HEADS_EIGHT_WAYS, ("activation_kv", ("expert", "tensor")),
                                   *DEFAULT_RULES)
"""Rules that split eight key heads over the expert and tensor axes, in the other order from the query
heads."""


def shares_on_one_device(routed, heads: int, kv_heads: int, shards: int):
    """`routed`, the route itself (`template_call`), on one device with its query heads in `shards` shares,
    each run as its own call over the key heads it reads: query head `h` reads key head `h // group`,
    gathered per share. A share has the shape its shard runs at, and a call's rounding depends on its
    shape (the packed documents' sinks gradient differs by about 1.5 between three query heads and
    twenty-four, with no mesh). A head split's shards compare with these shares, and the key and value
    gradients add across the shares that read a key head."""
    share, group = heads // shards, heads // kv_heads

    def run(query, key, value, sinks, dout):
        parts = []
        for shard in range(shards):
            first = shard * share
            read = np.arange(first, first + share) // group
            out, dq, dk, dv, dsinks = with_gradients(routed)(
                query[:, :, first:first + share], key[:, :, read], value[:, :, read],
                sinks[first:first + share], dout[:, :, first:first + share])
            parts.append((read, out, dq, dk, dv, dsinks))
        dk, dv = jnp.zeros(key.shape, key.dtype), jnp.zeros(value.shape, value.dtype)
        for read, _, _, part_dk, part_dv, _ in parts:
            dk, dv = dk.at[:, :, read].add(part_dk), dv.at[:, :, read].add(part_dv)
        return (jnp.concatenate([part[1] for part in parts], axis=2),
                jnp.concatenate([part[2] for part in parts], axis=2), dk, dv,
                jnp.concatenate([part[5] for part in parts], axis=0))

    return run


def per_shard_contract(case: dict, mesh: Mesh, specs: list, *, heads: int, kv_heads: int, rules, label: str,
                       moves=(), shards: int | None = None) -> None:
    """`case`'s named template call on `mesh`, its arguments placed by `specs` and the logical axis `rules` in
    context, runs per shard (`sharded.assert_runs_per_shard`), and its compiled step moves no operand between
    devices beyond `moves`. Its output and gradients are on par with one device's whole call over the site's
    draws (check 1), measured from float64 on the reference path. With `shards`, they are on par with the
    shares of the same route run on one device (`shares_on_one_device`) instead, and the split is also
    asserted against the fp32 reference path: its own error within 1.5 times that path's on every leaf."""
    queries, keys = geometry(case)
    draw = drawing(queries, keys, jnp.float32, sinks=True, heads=heads, kv_heads=kv_heads)

    routed, truth = template_call(case), with_gradients(template_call(case, "reference"))
    arm = None if shards is None else jax.jit(shares_on_one_device(routed, heads, kv_heads, shards))

    with nn.logical_axis_rules(rules):
        assert_runs_per_shard(with_gradients(routed), draw, specs, mesh, label, seeds=MESH_SEEDS,
                              truth_fn=truth, moves=moves, arm=arm)
        if shards is not None:
            assert_runs_per_shard(with_gradients(routed), draw, specs, mesh, f"{label}, the reference path",
                                  seeds=MESH_SEEDS, truth_fn=truth, moves=moves, arm=jax.jit(truth))


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("key_heads", [1, 2], ids=["one-key-head", "two-key-heads"])
@pytest.mark.parametrize("case", MESH_CASES.values(), ids=list(MESH_CASES))
def test_a_named_template_call_keeps_the_kernel_contract_on_a_data_and_head_mesh(case, key_heads):
    """A named template call on a mesh of two rows by four heads, with one or two key heads, fewer than the
    four tensor shards. It runs per shard (`sharded.assert_runs_per_shard`): its output and its gradients of
    the query, key, value and sinks are on par with one device's over the site's draws (check 1), measured
    from float64 on the reference path, and the compiled step moves no operand between devices."""
    mesh = Mesh(np.asarray(jax.devices()[:ROWS * MESH_TENSOR]).reshape(ROWS, MESH_TENSOR), ("data", "tensor"))
    rows_and_heads = PartitionSpec("data", None, "tensor")
    # Fewer key heads than tensor shards: no shard splits them, so each holds them whole, split by rows alone.
    rows = PartitionSpec("data")
    specs = [rows_and_heads, rows, rows, PartitionSpec("tensor"), rows_and_heads]
    per_shard_contract(case, mesh, specs, heads=MESH_HEADS, kv_heads=key_heads, rules=DEFAULT_RULES,
                       label=f"template split over rows and heads, {key_heads} key heads")


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kv_heads", [6, 8], ids=["six-key-heads", "eight-key-heads"])
def test_partly_split_key_heads_keep_the_kernel_contract_on_a_tensor_and_expert_mesh(kv_heads):
    """A named template call with 24 query heads split eight ways over a tensor axis of two and an expert axis
    of four, and key heads split over the tensor axis alone. Each shard holds part of the key heads its query
    heads read, and reads them at their offset in that part. It runs per shard, on par with its shares run on
    one device (`shares_on_one_device`), its own error within 1.5 times the fp32 reference path's on every
    leaf (check 1, from float64), and no operand moves between devices."""
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 4), ("tensor", "expert"))
    heads = PartitionSpec(None, None, ("tensor", "expert"))
    keys = PartitionSpec(None, None, "tensor")
    specs = [heads, keys, keys, PartitionSpec(("tensor", "expert")), heads]
    case = MESH_CASES["own-documents"]
    per_shard_contract(case, mesh, specs, heads=WIDE_HEADS, kv_heads=kv_heads,
                       rules=PARTLY_SPLIT_KEY_HEADS[kv_heads],
                       label=f"template split eight ways, {kv_heads} key heads on the tensor axis",
                       shards=WIDE_SHARDS)


@pytest.mark.mesh(devices=8)
def test_key_heads_the_query_heads_do_not_lead_are_gathered_on_a_tensor_and_expert_mesh():
    """Eight key heads split over the expert and tensor axes, in the other order from the query heads' split.
    No shard holds the key heads its query heads read, so every shard holds them whole, and the key and value
    heads are gathered once before the map. It runs per shard, on par with its shares run on one device, and
    its own error is within 1.5 times the fp32 reference path's on every leaf (check 1, from float64)."""
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 4), ("tensor", "expert"))
    heads = PartitionSpec(None, None, ("tensor", "expert"))
    keys = PartitionSpec(None, None, ("expert", "tensor"))
    specs = [heads, keys, keys, PartitionSpec(("tensor", "expert")), heads]
    case = MESH_CASES["own-documents"]
    per_shard_contract(case, mesh, specs, heads=WIDE_HEADS, kv_heads=8,
                       rules=KEY_HEADS_LEADING_THE_OTHER_WAY, moves=("all-gather",),
                       label="template split eight ways, key heads the other way", shards=WIDE_SHARDS)


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
    mask = jnp.ones((ROWS, 1, QUERIES, KEYS), bool)
    bias = jnp.zeros((ROWS, HEADS, QUERIES, KEYS), jnp.bfloat16)
    masked = attention.attention_case(query, key, named="template", mask=mask)
    biased = attention.attention_case(query, key, named="template", bias=bias)
    for case, name in ((masked, "mask"), (biased, "bias")):
        refusals = kernel_refusals(ATTENTION, "template", case)
        assert [(refusal.code, refusal.row) for refusal in refusals] == [(name, INHERENT)]
    with pytest.raises(KernelRefused):
        scaled_dot_product_attention(query, key, key, implementation="template", mask=mask)
    with pytest.raises(KernelRefused):
        scaled_dot_product_attention(query, key, key, implementation="template", bias=bias)


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
    assert sinks is not None
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    hooks = template.Hooks(score_mod=functools.partial(template.capped, SOFTCAP),
                           initial=template.sinks(sinks, ROWS, QUERIES, WIDTH))
    assert attention_route.attend(query, key, value, spans, hooks).shape == (ROWS, QUERIES, HEADS, WIDTH)
    with pytest.raises(ValueError):
        attention_route.attend(query, key, value, spans, template.Hooks(output_mod=jnp.tanh))
