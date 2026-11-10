"""dew.nn.attention's `implementation='template'` (`attention_route.attend`) on the CPU's plain-JAX mainloop.

Its output and its gradients of the query, key, value and sinks are on par with Dew's reference path from
float64 (check 1), for each call form the route takes, block groups among them, a cache call's included.
Every case has grouped key heads: four query heads over two. The selector refuses the route a mask and a
bias by their inherent limits; on a family whose Mosaic template has a routed mainloop (wgmma's) it runs
every other call its declared cell's old route does not take (`attention_kernels.DECLARED`), and otherwise a
call that names it runs it. On meshes, a named call runs per
shard under `sharded.assert_runs_per_shard`: with fewer key heads than the tensor shards, with key heads split
partly, with key heads split in the other order from the query heads, or with heads that take an axis the
rows' arguments name.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import attention_terms, kept
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec
from reference_error import assert_as_exact, with_gradients
from sharded import MOVES, assert_runs_per_shard
from test_attention_template import (
    DTYPES,
    KEYS,
    LENGTHS,
    QUERIES,
    ROWS,
    SEEDS,
    SOFTCAP,
    STARTS,
    block_groups,
    case_keep,
    cross_documents,
    geometry,
    key_validity,
    own_documents,
)

from dew.mesh import DEFAULT_RULES
from dew.nn import attention
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.inputs import counted_places
from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_route
from dew.nn.precision import Numerics

HEADS, KV_HEADS, WIDTH = 4, 2, 32
MESH_TENSOR, MESH_HEADS = 4, 8
"""The tensor shards of the per-shard test's mesh, beside its `ROWS` data shards, and its query heads: two on
each tensor shard, over one or two key heads, which is fewer than the shards."""
MESH_SEEDS = 512
"""The per-shard test's draws, sized by `seeds_for` with one device compared with
itself: the fewest, doubling from `FEWEST_SEEDS`, that resolve the factor for each
case, which is 512 for lengths and starts over two key heads."""
WIDE_SEEDS = {6: 64, 8: 8}
"""The partly split and gathered key heads' sites' draws by their key heads, sized by `seeds_for` paired,
the split run against each arm it is compared with (the shares on one device and the reference path): 16
and 64 over six key heads, 8 and 8 over eight either way (ET-WIDESEEDS, row 2195: at 512 each site took
117 to 120 s of a CPU task)."""

CASES = {
    "full-grouped": {},
    "causal": {"causal": True},
    "causal-window": {"causal": True, "sliding_window": 24},
    "packed-documents": {"segment_ids": "own"},
    "cross-documents-causal": {"causal": True, "segment_ids": "cross"},
    "key-lengths": {"lengths": True},
    "sinks-softcap": {"causal": True, "sinks": True, "softcap": SOFTCAP},
    "block-causal-documents-window": {"causal": True, "sliding_window": 24, "segment_ids": "own",
                                      "block_groups": True},
    "cache-call-blocks-starts": {"causal": True, "segment_ids": "cross", "block_groups": True, "starts": True,
                                 "sinks": True},
    "valid-documents-window-places": {"causal": True, "sliding_window": 24, "segment_ids": "own",
                                      "valid": True, "positions": True},
    "cache-call-valid": {"causal": True, "segment_ids": "cross", "valid": True, "sinks": True},
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
    if case.get("block_groups"):
        keywords["block_groups"], keywords["image_groups"] = (jnp.asarray(groups)
                                                              for groups in block_groups(geometry(case)[1]))
    if case.get("valid"):
        keywords["key_valid"] = key_validity(geometry(case)[1])
    if case.get("positions"):
        keywords["places"] = counted_places(keywords["segment_ids"], keywords["key_valid"])
    return keywords


def drawing(queries: int, keys: int, dtype, *, sinks: bool, heads: int = HEADS, kv_heads: int = KV_HEADS,
            rows: int = ROWS):
    """A draw `(query, key, value, sinks, dout)` in `dtype`: queries and the output's cotangent `[rows,
    queries, heads, 32]`, keys and values `[rows, keys, kv_heads, 32]`, and sinks `[heads]` where `sinks`."""

    def draw(seed: int):
        q_key, k_key, v_key, d_key, s_key = jax.random.split(jax.random.key(seed), 5)
        query = jax.random.normal(q_key, (rows, queries, heads, WIDTH), dtype)
        key = jax.random.normal(k_key, (rows, keys, kv_heads, WIDTH), dtype)
        value = jax.random.normal(v_key, (rows, keys, kv_heads, WIDTH), dtype)
        dout = jax.random.normal(d_key, (rows, queries, heads, WIDTH), dtype)
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


SINK_SITES = {"sinks-softcap": {"float32": 16, "bfloat16": 32},
              "cache-call-blocks-starts": {"float32": 64, "bfloat16": 128},
              "cache-call-valid": {"float32": 16, "bfloat16": 32}}
"""The seeds of each case with sinks at `SINK_HEADS`, sized by `seeds_for` there (the reference against
itself): a head's sink gradient is one value a draw, so at the site's `HEADS` the cache calls' did not
resolve within 512 seeds (bf16) and needed 256 to 512 elsewhere; the other cases resolve at `SEEDS`."""
SINK_HEADS = (32, 8)
"""The query and key heads of a case with sinks: enough sink gradients a draw to resolve in few seeds."""


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("name", CASES)
def test_a_named_template_route_is_on_par_with_the_reference_path_forward_and_backward(name, dtype):
    """The route's output and its gradients of the query, key, value and sinks, on par with the reference
    path (`implementation='reference'`) measured from float64 over the site's draws (check 1), each entry
    sized by its float64 terms over the keys the case keeps (`_terms`): a query that keeps one key has a dQ
    of zero in float64, which the fast form's D, formed from the rounded output, leaves a rounding off (the
    default form claims exact zeros only of a row of no keys, docs/design/testing.md); a case with sinks at
    `SINK_HEADS` over its own seeds (`SINK_SITES`)."""
    case = CASES[name]
    queries, keys = geometry(case)
    heads, kv_heads = SINK_HEADS if name in SINK_SITES else (HEADS, KV_HEADS)
    seeds = SINK_SITES[name][jnp.dtype(dtype).name] if name in SINK_SITES else SEEDS
    draw = drawing(queries, keys, dtype, sinks=bool(case.get("sinks")), heads=heads, kv_heads=kv_heads)
    assert_as_exact(jax.jit(pulled("template", case)), jax.jit(pulled("reference", case)), draw,
                    f"template route {case} {jnp.dtype(dtype).name}", seeds=seeds, terms=_terms(case))


MESH_CASES = {
    "own-documents": {"causal": True, "segment_ids": "own", "softcap": SOFTCAP},
    "cross-documents": {"causal": True, "segment_ids": "cross"},
    "lengths-and-starts": {"lengths": True, "starts": True},
    "own-documents-blocks": {"causal": True, "segment_ids": "own", "block_groups": True},
    "own-documents-valid-places": {"causal": True, "sliding_window": 24, "segment_ids": "own", "valid": True,
                                   "positions": True},
}
"""The calls the per-shard test takes, each with sinks: its ids, lengths, starts, block groups, key validity
and places split with their rows."""
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
HEADS_BEFORE_ROWS = (*QUERY_HEADS_EIGHT_WAYS, ("activation_kv", ("tensor", "expert")),
                     ("activation_batch", "expert"), *DEFAULT_RULES)
"""Rules under which the heads take the expert axis before the rows can: a query, its keys and its values
keep every row on each shard, where a `[B, S]` argument alone would split its rows over the expert axis."""
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
                       moves=(), shards: int | None = None, seeds: int = MESH_SEEDS) -> None:
    """`case`'s named template call on `mesh`, its arguments placed by `specs` and the logical axis `rules` in
    context, runs per shard (`sharded.assert_runs_per_shard`), and its compiled step moves no operand between
    devices beyond `moves`. Its output and gradients are on par with one device's whole call over the site's
    `seeds` draws (check 1), measured from float64 on the reference path. With `shards`, they are on par
    with the shares of the same route run on one device (`shares_on_one_device`) instead, and the split is
    also asserted against the fp32 reference path: its own error within 1.5 times that path's on every
    leaf."""
    queries, keys = geometry(case)
    draw = drawing(queries, keys, jnp.float32, sinks=True, heads=heads, kv_heads=kv_heads)

    routed, truth = template_call(case), with_gradients(template_call(case, "reference"))
    arm = None if shards is None else jax.jit(shares_on_one_device(routed, heads, kv_heads, shards))

    with nn.logical_axis_rules(rules):
        # With `shards`, the one split run of each draw compared with both arms (row 2156).
        assert_runs_per_shard(with_gradients(routed), draw, specs, mesh, label, seeds=seeds,
                              truth_fn=truth, moves=moves, arm=arm,
                              also=None if shards is None else ("the reference path", jax.jit(truth)))


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
                       shards=WIDE_SHARDS, seeds=WIDE_SEEDS[kv_heads])


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
                       label="template split eight ways, key heads the other way", shards=WIDE_SHARDS,
                       seeds=WIDE_SEEDS[8])


@pytest.mark.mesh(devices=8)
def test_a_calls_row_arguments_split_as_its_query_rows_do_where_the_heads_contend_for_the_axis():
    """Eight query and key heads split eight ways over a tensor axis of two and an expert axis of four,
    which the heads take before the rows can (`HEADS_BEFORE_ROWS`), so every shard holds all four rows.
    A causal call with block groups and no sinks or documents: its groups, `[B, S]`, split as the query's
    rows do, not as their own name alone would (over the expert axis), so each shard reads every row's
    groups. Its output is on par with one device's (check 1, from float64 on the reference path), and no
    operand moves between devices."""
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(2, 4), ("tensor", "expert"))
    heads = PartitionSpec(None, None, ("tensor", "expert"))
    case = {"causal": True, "block_groups": True}
    queries, keys = geometry(case)
    blocks, images = (jnp.asarray(np.tile(groups, (2, 1))) for groups in block_groups(keys))

    def call(implementation: str):
        def run(query, key, value):
            return scaled_dot_product_attention(query, key, value, implementation=implementation, causal=True,
                                                block_groups=blocks, image_groups=images)
        return run

    draw = drawing(queries, keys, jnp.float32, sinks=False, heads=8, kv_heads=8, rows=4)
    with nn.logical_axis_rules(HEADS_BEFORE_ROWS):
        assert_runs_per_shard(call("template"), lambda seed: draw(seed)[:3], [heads] * 3, mesh,
                              "template rows under contending heads", seeds=MESH_SEEDS,
                              truth_fn=call("reference"))


@pytest.mark.parametrize("name", ["causal-window", "block-causal-documents-window",
                                  "valid-documents-window-places"])
def test_a_route_call_under_forward_mode_keeps_the_template_value_and_the_reference_derivative(name):
    """Under `forward_mode_attention` a fused call keeps its kernel's value and takes its tangent from the
    reference path's JVP over the same keys, block groups among them, so the route's primal and tangent are
    on par with the reference path's (check 1)."""
    case = CASES[name]

    def pushed(implementation: str):
        def run(query, key, value, _sinks, dout):
            def call(query, key, value):
                return scaled_dot_product_attention(query, key, value, implementation=implementation,
                                                    **keywords(case))

            with attention.forward_mode_attention():
                return jax.jvp(call, (query, key, value), (dout, value, key))

        return run

    assert_as_exact(jax.jit(pushed("template")), jax.jit(pushed("reference")),
                    drawing(*geometry(case), jnp.float32, sinks=False), f"template route forward mode {name}",
                    seeds=SEEDS)


LOWERED = {
    "fp32-64-full": ("sm90", jnp.float32, 64, False, False),
    "fp32-128-causal": ("sm90", jnp.float32, 128, True, False),
    "fp32-128-full": ("sm90", jnp.float32, 128, False, False),
    **{f"bias-{generation_name}-{'causal' if causal else 'full'}":
       (generation_name, jnp.bfloat16, 128, causal, True)
       for generation_name in ("sm90", "sm80") for causal in (True, False)},
}
"""Calls lowered for a GPU on the named template: by the GPU, the operands' dtype, the head width, causal or
full, and a bias that varies by query (`[1, H, Q, K]` fp32) or none."""


@pytest.mark.parametrize("form", ["fast", "accurate"])
@pytest.mark.parametrize("differentiated", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("name", LOWERED)
def test_a_call_named_onto_the_template_lowers_for_a_gpu(name, differentiated, form):
    """A call named onto the template (`implementation="template"`) lowered for an H100 or an A100, in either
    form, inferred or differentiated: fp32 operands at widths 64 and 128, and bf16 under a bias that varies by
    query, each pass a kernel of a GPU mainloop that takes it (`lowered.kernel_calls`: Pallas's). Its values
    are check 1's on the template's mainloops
    (test_a_named_template_route_is_on_par_with_the_reference_path_forward_and_backward,
    test_a_dense_bias_named_onto_the_template_is_on_par_with_the_reference_path)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels.generation import run_precision
    from dew.nn.kernels.lowered import kernel_calls
    from dew.nn.precision import Numerics
    from dew.telemetry.instrumentation import PALLAS
    generation_name, dtype, width, causal, biased = LOWERED[name]
    shape = jax.ShapeDtypeStruct((1, 512, 2, width), dtype)
    bias = (jax.ShapeDtypeStruct((1, 2, 512, 512), jnp.float32),) if biased else ()

    def attend(query, key, value, *bias):
        return scaled_dot_product_attention(query, key, value, bias=bias[0] if bias else None, causal=causal,
                                            implementation="template")

    def trained(query, key, value, *rest):
        *bias, dout = rest
        out, pull = jax.vjp(attend, query, key, value, *bias)
        return out, pull(dout)

    with run_precision(Numerics(forms=form)):
        operands = (shape, shape, shape, *bias, *((shape,) if differentiated else ()))
        lowered = lowered_for(trained if differentiated else attend, operands, generation_name)
    assert any(call.kind in PALLAS for call in kernel_calls(lowered))


def test_the_jax_nn_form_takes_values_of_the_querys_heads_beside_keys_of_fewer():
    """The reference path's jax.nn form (`scale_logits`) over keys of one head and values of the query's two
    (Z-Image's joint attention's) is, bit for bit, the same call over the keys repeated to the query's heads:
    its query heads group over the keys' head only where the values share it (`_grouped_as_jax_nn`)."""
    streams = jax.random.split(jax.random.key(4), 3)
    query = jax.random.normal(streams[0], (2, 16, 2, 8), jnp.bfloat16)
    key = jax.random.normal(streams[1], (2, 16, 1, 8), jnp.bfloat16)
    value = jax.random.normal(streams[2], (2, 16, 2, 8), jnp.bfloat16)

    def attend(key):
        return scaled_dot_product_attention(query, key, value, implementation="reference", causal=True,
                                            scale_logits=True)

    repeated = jnp.repeat(key, 2, axis=2)
    assert np.asarray(attend(key)).tobytes() == np.asarray(attend(repeated)).tobytes()


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize(("heads", "kv_heads"), [(4, 4), (4, 2), (8, 1)], ids=["mha", "gqa", "mqa"])
def test_the_reference_scaling_its_logits_after_the_product_is_jax_nns_xla_attention(heads, kv_heads, causal):
    """The reference path with the scale after the product (`scale_logits`) is jax.nn's xla attention
    (`jax.nn.dot_product_attention`) bit for bit over bf16 operands: fp32 logits of their product at the
    dot algorithm jax.nn names for them, its query heads grouped over their key head as jax.nn groups them,
    scaled in fp32, masked and normalized in fp32, which the compile point's recognizer matches
    written-out attention against (row 4783). fp32 operands' products this process sums in float64
    (`precision.accumulated`, the test lane's wide CPU arithmetic), where jax.nn keeps XLA's sums; outside
    it they agree too (et-scalebits3). A CPU's dot refuses fp16's algorithm, jax.nn's attention with it."""
    dtype = jnp.bfloat16
    streams = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(streams[0], (1, 16, heads, 32), dtype)
    key, value = (jax.random.normal(stream, (1, 16, kv_heads, 32), dtype) for stream in streams[1:])
    ours = jax.jit(lambda q, k, v: scaled_dot_product_attention(q, k, v, causal=causal, scale_logits=True,
                                                             implementation="reference"))(query, key, value)
    xla = jax.jit(lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=causal,
                                                             implementation="xla"))(query, key, value)
    assert np.asarray(ours).tobytes() == np.asarray(xla).tobytes()


JAX_NN_FILL = -0.7 * float(jnp.finfo(jnp.float32).max)
"""jax.nn's masked logit (`jax._src.nn.functions._get_large_negative`), its fp32 logits'."""


def _rows_mask(queries: int, keys: int) -> jax.Array:
    """A `[1, 1, queries, keys]` mask whose rows 3 and 7 see no key and every other row a key in three,
    beside its own (rows 1 and 5 their own alone)."""
    rows, columns = np.arange(queries)[:, None], np.arange(keys)[None]
    seen = (columns % 3 == rows % 3) | (columns == rows)
    seen[[1, 5]] = columns[0] == np.asarray([[1], [5]])
    seen[[3, 7]] = False
    return jnp.asarray(seen)[None, None]


def _written_out(query, key, value, mask):
    """Attention as a program writes it out: fp32 logits of the product scaled after it, every masked one
    -inf, a softmax in fp32 rounded to the operands' dtype, its product with the values."""
    heads = query.shape[-2]
    key, value = (jnp.repeat(x, heads // x.shape[-2], axis=-2) for x in (key, value))
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, preferred_element_type=jnp.float32)
    logits = jnp.where(mask, logits * query.shape[-1] ** -0.5, -jnp.inf)
    weights = jax.nn.softmax(logits, axis=-1).astype(query.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, value)


@pytest.mark.parametrize("masked", [False, True], ids=["causal", "masked"])
@pytest.mark.parametrize(("heads", "kv_heads"), [(4, 4), (4, 2)], ids=["mha", "gqa"])
def test_the_reference_filling_as_jax_nn_is_jax_nns_xla_attention(heads, kv_heads, masked):
    """The reference path with jax.nn's masked logit (`mask_fill`) and its scale after the product is
    jax.nn's xla attention bit for bit over bf16 operands, a causal call and one whose mask leaves rows no
    key, which read every value alike in both (row 4783)."""
    streams = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(streams[0], (1, 16, heads, 32), jnp.bfloat16)
    key, value = (jax.random.normal(stream, (1, 16, kv_heads, 32), jnp.bfloat16) for stream in streams[1:])
    mask = _rows_mask(16, 16) if masked else None
    ours = jax.jit(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, causal=not masked, mask=mask, scale_logits=True, mask_fill=JAX_NN_FILL,
        implementation="reference"))(query, key, value)
    xla = jax.jit(lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=not masked, mask=mask,
                                                             implementation="xla"))(query, key, value)
    assert np.asarray(ours).tobytes() == np.asarray(xla).tobytes()


@pytest.mark.parametrize(("heads", "kv_heads"), [(4, 4), (4, 2)], ids=["mha", "gqa"])
def test_the_reference_filling_with_minus_infinity_is_the_written_out_program(heads, kv_heads):
    """The reference path with -inf for a masked logit and its scale after the product is attention written
    out with `where(mask, logits, -inf)` bit for bit over bf16 operands, a row that sees no key NaN in both
    (row 4783)."""
    streams = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(streams[0], (1, 16, heads, 32), jnp.bfloat16)
    key, value = (jax.random.normal(stream, (1, 16, kv_heads, 32), jnp.bfloat16) for stream in streams[1:])
    mask = _rows_mask(16, 16)
    ours = jax.jit(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, mask=mask, scale_logits=True, mask_fill=-jnp.inf,
        implementation="reference"))(query, key, value)
    written = jax.jit(_written_out)(query, key, value, mask)
    # Widened, which is exact, so NumPy's comparison takes a NaN for a NaN (bfloat16's it does not).
    np.testing.assert_array_equal(np.asarray(ours, np.float32), np.asarray(written, np.float32))
    assert np.isnan(np.asarray(ours, np.float32))[0, [3, 7]].all()


def _documents(rows: int, queries: int) -> jax.Array:
    """`[rows, queries]` documents: two a row, of 40 and the rest, beside padding (id 0) at queries 3 and 7,
    which under causality see no key, every other query a key of its own document's before it."""
    ids = np.where(np.arange(queries) < 40, 1, 2)
    ids[[3, 7]] = 0
    return jnp.asarray(np.broadcast_to(ids, (rows, queries)))


@pytest.mark.parametrize("sinks", [False, True], ids=["plain", "sinks"])
@pytest.mark.parametrize("fill", [JAX_NN_FILL, -2.0], ids=["jax-nn", "within-reach"])
def test_a_template_call_filling_its_masked_logits_is_on_par_with_the_portable_path(fill, sinks):
    """A causal call on the template over packed documents with padding rows, which see no key, under a
    finite masked logit (`mask_fill`): jax.nn's, which outweighs no key a row sees, so the template weighs
    its masked keys nothing and a padding row reads every value alike (beside its sink, the sink's share);
    and one within the logits' reach, which weighs a masked key beside them, the call the portable path's.
    Its output and gradients on par with the portable path's (check 1)."""
    documents = _documents(ROWS, QUERIES)

    def call(implementation: str):
        def run(query, key, value, sink_logits, dout):
            def attend(query, key, value, sink_logits):
                return scaled_dot_product_attention(query, key, value, causal=True, segment_ids=documents,
                                                    mask_fill=fill, sinks=sink_logits if sinks else None,
                                                    implementation=implementation)

            out, pull = jax.vjp(attend, query, key, value, sink_logits)
            return (out, *pull(dout)[:4 if sinks else 3])

        return run

    assert_as_exact(jax.jit(call("template")), jax.jit(call("reference")),
                    drawing(QUERIES, QUERIES, jnp.bfloat16, sinks=True), f"template filled {fill:g}",
                    seeds=SEEDS)


def test_a_template_call_filling_with_minus_infinity_reads_nan_where_a_row_sees_no_key():
    """A causal call on the template over packed documents with -inf for a masked logit: its padding rows,
    3 and 7, which see no key, NaN, as the portable path's, and every value's gradient NaN with them (the
    portable path's NaN weights reach every key); every other row and the query's and key's gradients on
    par with it (check 1)."""
    documents = _documents(ROWS, QUERIES)
    seen = (documents != 0)[:, :, None, None]

    def call(implementation: str):
        def run(query, key, value, _sinks, dout):
            def attend(query, key, value):
                return scaled_dot_product_attention(query, key, value, causal=True, segment_ids=documents,
                                                    mask_fill=-jnp.inf, implementation=implementation)

            out, pull = jax.vjp(attend, query, key, value)
            return (out, *pull(dout))

        return run

    draw = drawing(QUERIES, QUERIES, jnp.bfloat16, sinks=False)
    ours, theirs = (jax.jit(call(implementation))(*draw(0)) for implementation in ("template", "reference"))
    for mine, reference in ((ours[0], theirs[0]), (ours[3], theirs[3])):
        np.testing.assert_array_equal(np.isnan(np.asarray(mine, np.float32)),
                                      np.isnan(np.asarray(reference, np.float32)))
    assert np.isnan(np.asarray(ours[0], np.float32))[:, [3, 7]].all()
    assert np.isnan(np.asarray(ours[3], np.float32)).all()

    def finite(implementation: str):
        def run(*draw):
            out, dq, dk, _ = call(implementation)(*draw)
            return jnp.where(seen, out, 0), dq, dk

        return run

    assert_as_exact(jax.jit(finite("template")), jax.jit(finite("reference")), draw, "template filled -inf",
                    seeds=SEEDS)


@pytest.mark.parametrize("implementation", ["reference", "template"])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
def test_a_call_scaling_its_logits_after_the_product_is_on_par_with_flaxs_form(dtype, implementation):
    """A causal call whose scale takes the fp32 logits after the product (`scale_logits`, jax.nn's form,
    the reference path's `float32_einsum(query, key) * D ** -0.5`), on the reference path and on the
    template, which scales its fp32 products so natively: its output and its gradients on par with the
    reference path in flax's form, the query scaled before the product (check 1)."""
    def call(implementation: str, *, scale_logits: bool):
        def run(query, key, value, _sinks, dout):
            def attend(query, key, value):
                return scaled_dot_product_attention(query, key, value, causal=True, scale_logits=scale_logits,
                                                    implementation=implementation)

            out, pull = jax.vjp(attend, query, key, value)
            return (out, *pull(dout))

        return run

    assert_as_exact(jax.jit(call(implementation, scale_logits=True)),
                    jax.jit(call("reference", scale_logits=False)),
                    drawing(QUERIES, KEYS, dtype, sinks=False), f"logits scaled after {implementation}",
                    seeds=SEEDS)


@pytest.mark.parametrize("name", ["block-causal-documents-window", "valid-documents-window-places"])
def test_a_dropped_out_call_draws_over_the_keys_its_block_groups_mean(name):
    """Under active dropout a call runs the reference path's one draw (`_dropout_attention`): with block
    groups, or key validity and counted places, documents and a window it draws over the mask its spans
    mean, so it is on par (check 1) with one given that mask (`case_keep`) under the same key, whose float64
    run is the truth: a call that drew over other keys, or dropped others, would sit far from it."""
    case = CASES[name]
    dropped = functools.partial(scaled_dot_product_attention, dropout_rate=0.25,
                                dropout_rng=jax.random.key(3), deterministic=False)
    spanned, mask = keywords(case), jnp.asarray(case_keep(case))[:, None]

    def blocked(query, key, value):
        return dropped(query, key, value, **spanned)

    def masked(query, key, value):
        return dropped(query, key, value, mask=mask)

    draw = drawing(*geometry(case), jnp.float32, sinks=False)
    assert_as_exact(jax.jit(blocked), jax.jit(masked), lambda seed: draw(seed)[:3],
                    f"dropped {name} over its spans' mask", seeds=SEEDS)


SIDE_OF_ONE_SEEDS = 16
"""Check 1's declared seeds for a bf16 attention with a side of one."""


@pytest.mark.parametrize("sides", [(1, 12), (5, 1)], ids=["one query", "one key"])
def test_a_bf16_attention_with_a_side_of_one_is_on_par_with_float64(sides):
    """A bf16 attention whose queries or keys are one, a grouped decode step's or a one-key read's, is
    on par (check 1, float64 truth) on the device it runs on, its gradients too: an A100 computes jax.nn's
    BF16_BF16_F32 query-key product with a side of one off round-to-nearest-even at its ties (row 1550),
    which a call carried into its output."""
    from dew.nn.attention import reference_attention

    queries, keys = sides

    def drawn(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.standard_normal(shape), jnp.bfloat16) for shape in (
            (2, queries, HEADS, WIDTH), (2, keys, KV_HEADS, WIDTH), (2, keys, KV_HEADS, WIDTH)))

    def attended(attend):
        return lambda query, key, value: with_gradients(attend)(query, key, value, jnp.ones_like(query))

    def named(query, key, value):
        return scaled_dot_product_attention(query, key, value, implementation="xla")

    assert_as_exact(jax.jit(attended(named)), jax.jit(attended(reference_attention)), drawn,
                    f"bf16 attention of {queries} queries over {keys} keys", seeds=SIDE_OF_ONE_SEEDS)


def test_a_bias_of_the_key_heads_is_refused_by_the_template_not_read_past():
    """A bias of the key heads `[B, Hk, Q, K]` (Hk neither 1 nor the query heads) broadcasts to no call's
    heads: the template refuses it by name, as the reference path's broadcast does, and reads no head past
    the bias (row 2578)."""
    from dew.nn.kernels import selection

    queries = 40
    rng = np.random.default_rng(0)
    query = jnp.asarray(rng.standard_normal((2, queries, HEADS, WIDTH)), jnp.bfloat16)
    key = jnp.asarray(rng.standard_normal((2, queries, KV_HEADS, WIDTH)), jnp.bfloat16)
    bias = jnp.zeros((2, KV_HEADS, queries, queries), jnp.float32)
    with pytest.raises(selection.KernelRefused):
        scaled_dot_product_attention(query, key, key, bias=bias, causal=True, implementation="template")


DENSE_BIAS_SEEDS = 8
"""Check 1's declared seeds for a dense bias named onto the template."""


@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "fast"])
@pytest.mark.parametrize("rows, heads", [(2, HEADS), (1, HEADS), (1, 1)],
                         ids=["a bias a row", "one bias every row", "one bias every row and head"])
def test_a_dense_bias_named_onto_the_template_is_on_par_with_the_reference_path(rows, heads, deterministic,
                                                                                request):
    """A causal bf16 call whose bias varies by query, `[B or 1, H or 1, Q, K]` (a learned relative
    position's, T5's one table for every row), named onto the template: its score hook reads the bias a
    tile at a time where it is shared, never broadcast (`template.Biased`, a `Kind.TILE` parameter at
    `template.shared_place`), on par (check 1) with the reference path's dense bias from float64, its
    output and the query's, key's, value's and bias's gradients, the bias's in its own shape (rows 2081,
    2523). In both of the Mosaic backward's schedules for a shared bias: deterministic mode's, each program
    its own place summed after, and the default's, the programs' adds into one place (the flag set or
    dropped whatever the lane's, `with_deterministic_ops`, `without_deterministic_ops`; row 2645)."""
    request.getfixturevalue("with_deterministic_ops" if deterministic else "without_deterministic_ops")
    queries = 40

    def drawn(seed):
        rng = np.random.default_rng(seed)
        query, key, value, cotangent = (
            jnp.asarray(rng.standard_normal(shape), jnp.bfloat16) for shape in (
                (2, queries, HEADS, WIDTH), (2, queries, KV_HEADS, WIDTH), (2, queries, KV_HEADS, WIDTH),
                (2, queries, HEADS, WIDTH)))
        bias = jnp.asarray(0.5 * rng.standard_normal((rows, heads, queries, queries)), jnp.float32)
        return query, key, value, bias, cotangent

    def attended(implementation):
        return with_gradients(lambda query, key, value, bias: scaled_dot_product_attention(
            query, key, value, bias=bias, causal=True, implementation=implementation))

    keep = kept(queries, queries, causal=True)

    def terms(query, key, value, bias, dout):
        # A query of one key has a dQ of zero in float64, which a D from the rounded output rounds to
        # its own size (`attention_terms`, its bias's terms the fifth).
        return tuple(attention_terms(query, key, value, dout, keep, bias=bias))

    assert_as_exact(jax.jit(attended("template")), jax.jit(attended("reference")), drawn,
                    f"a dense bias of {rows} rows and {heads} heads on the template", seeds=DENSE_BIAS_SEEDS,
                    terms=terms)


def test_a_biased_query_of_one_key_has_nonzero_gradient_terms_where_its_gradient_is_zero():
    """attention_terms with a bias every row shares: the first query of a causal call keeps one key, so its
    query gradient is zero in float64, and its terms are not (a D from the rounded output rounds it to their
    size); the bias's own gradient's terms have the bias's shape, summed over the rows it shares."""
    rng = np.random.default_rng(0)
    query, key, value, dout = (rng.standard_normal((3, 16, 2, 8)) for _ in range(4))
    bias = 0.5 * rng.standard_normal((1, 2, 16, 16))
    keep = kept(16, 16, causal=True)
    _, dq_terms, _, _, bias_terms = attention_terms(query, key, value, dout, keep, bias=bias)
    with jax.enable_x64(new_val=True):
        _, pull = jax.vjp(lambda q, k, v, b: scaled_dot_product_attention(
            q, k, v, bias=b, causal=True, implementation="reference"), query, key, value, bias)
        dq = np.asarray(pull(dout)[0])
    assert np.all(dq[:, 0] == 0) and np.all(dq_terms[:, 0] > 0)
    assert bias_terms.shape == bias.shape


@pytest.mark.skipif(generation.device_generation() != "sm90",
                    reason="composes the route's mainloops, on an H100")
@pytest.mark.parametrize("case", ["causal", "packed-documents", "sinks-softcap"])
def an_fp16_call_on_an_h100_is_on_par(case):
    """fp16 on an H100, its Mosaic backward in two launches (row 3497: its parts scaled, each product
    restored into its sums, `operands.Issuer.summed`): on par with the reference path (check 1). A call
    without sinks is judged a term at a time (`attention_terms`): a query that keeps one key has a dQ of
    zero in float64, which a D formed from the rounded output rounds to its own size (G14's H100 flow,
    gate-g14f-h100)."""
    queries, keys = geometry(CASES[case])
    draw = drawing(queries, keys, jnp.float16, sinks=bool(CASES[case].get("sinks")))
    template_pass = jax.jit(pulled("template", CASES[case]))
    assert_as_exact(template_pass, jax.jit(pulled("reference", CASES[case])), draw,
                    f"fp16 Mosaic forward and backward, {case}", seeds=SEEDS,
                    terms=False if CASES[case].get("sinks") else _terms(CASES[case]))


def _terms(case: dict):
    """The float64 terms of `case`'s output and gradients, its sinks' among them (`attention_terms`), over
    the keys it keeps (its spans as a mask, `case_keep`), under its sinks and softcap, a function of a draw
    `(query, key, value, sinks, dout)`."""
    keep = case_keep(case)[:, None]

    def terms(query, key, value, sinks, dout):
        found = attention_terms(query, key, value, dout, keep, sinks=sinks, softcap=case.get("softcap"))
        # The sinks' gradient, which a call without sinks holds as None.
        return (*found, None) if sinks is None else tuple(found)

    return terms


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("length", [4096, 8192])
def a_long_sequence_is_on_par_with_the_reference_path(length, dtype):
    """A causal call of `length` tokens, two query heads over one key head (`WIDTH` wide), through the
    template on whichever mainloop the GPU runs it, on par with the reference path (check 1): D, the
    refinement and the denominator are sums the mainloops run in sequence over its key tiles where the
    reference sums by tree, so a ratio that grew with the sequence would show here (math review P2.5).
    Every entry is sized by the terms attention's float64 gradients carry through their cancellation
    (`_causal_terms`): the first query keeps one key, so its gradient is float64 zero, which a mainloop
    forming D from its rounded output (an A100's) holds to that rounding."""
    case = {"causal": True}
    draw = drawing(length, length, dtype, sinks=False, heads=2, kv_heads=1, rows=1)
    assert_as_exact(jax.jit(pulled("template", case)), jax.jit(pulled("reference", case)), draw,
                    f"template route causal {length} {jnp.dtype(dtype).name}", seeds=SEEDS,
                    truth_fn=_causal_truth(length), terms=_causal_terms(length))


TRUTH_ROWS = 1024
"""The queries a long call's float64 truth attends a block at a time (`_causal_truth`)."""


def _causal_truth(length: int):
    """The float64 truth of a causal call of `length` tokens (check 1's `truth_fn`): the reference path's
    attention of `TRUTH_ROWS` queries at a time over every key, masked as causal, each block's output and
    query gradient its own and the keys' and values' gradients the blocks' sum. The whole call's truth held
    `length` squared logits a head in float64, 3 GiB at 8192, more than a GPU tier job's pool of an A100
    (2.69 GiB, MAINGPU-a9e29b08f); a block's are an eighth of them."""
    keys = jnp.arange(length)

    def truth(query, key, value, sinks, dout):
        del sinks

        def block(first):
            seen = (first + jnp.arange(TRUTH_ROWS))[:, None] >= keys[None, :]

            def call(query, key, value):
                return scaled_dot_product_attention(query, key, value, implementation="reference",
                                                    mask=seen[None, None])

            row_query, row_dout = (jax.lax.dynamic_slice_in_dim(array, first, TRUTH_ROWS, axis=1)
                                   for array in (query, dout))
            out, pull = jax.vjp(call, row_query, key, value)
            return (out, *pull(row_dout))

        out, d_query, d_key, d_value = jax.lax.map(block, jnp.arange(0, length, TRUTH_ROWS))

        def whole(blocks):
            return jnp.moveaxis(blocks, 0, 1).reshape(query.shape)

        return whole(out), whole(d_query), d_key.sum(0), d_value.sum(0), None

    return truth


def _causal_terms(length: int):
    """The float64 terms of a causal call of `length` tokens (`attention_terms` over the keys each query
    keeps), `TRUTH_ROWS` queries at a time as its truth is (`_causal_truth`): each block's output's and query
    gradient's terms its own, the keys' and values' the root of the blocks' summed squares; a function of a
    draw `(query, key, value, sinks, dout)`, the sinks' gradient none."""
    def terms(query, key, value, sinks, dout):
        del sinks
        blocks = []
        for first in range(0, length, TRUTH_ROWS):
            rows = slice(first, first + TRUTH_ROWS)
            keep = (np.arange(first, first + TRUTH_ROWS)[:, None] >= np.arange(length)[None, :])[None, None]
            blocks.append(attention_terms(query[:, rows], key, value, dout[:, rows], keep))
        out, d_query = (np.concatenate([block[leaf] for block in blocks], axis=1) for leaf in (0, 1))
        d_key, d_value = (np.sqrt(sum(np.square(block[leaf]) for block in blocks)) for leaf in (2, 3))
        return out, d_query, d_key, d_value, None

    return terms


def test_the_accurate_form_is_one_option_away():
    """A run's numerics in the accurate forms (`generation.run_precision`, `Numerics.forms`) run every
    template call in the accurate form (natural logits and the accurate products, `attention_route.FORMS`),
    its forward and gradients bitwise a call's that names that form, and the fast forms run the default, a
    call's that names none; a form the template does not have is refused. Bit for bit
    by construction: the forms resolve to the named form's arguments before the call traces, so the two
    calls trace one program."""
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    query, key, value, _, dout = drawing(QUERIES, KEYS, jnp.float32, sinks=False)(0)

    def routed(q, k, v, **form):
        out, pull = jax.vjp(lambda q, k, v: attention_route.attend(q, k, v, spans, **form), q, k, v)
        return out, pull(dout)

    found = {}
    for form, named in (("accurate", {"natural": True, "accurate": True}), ("fast", {})):
        with generation.run_precision(Numerics(forms=form)):
            found[form] = jax.tree.leaves(jax.jit(routed)(query, key, value))
        want = jax.tree.leaves(jax.jit(functools.partial(routed, **named))(query, key, value))
        for mine, theirs in zip(found[form], want, strict=True):
            np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs), err_msg=form)
    with pytest.raises(ValueError):
        Numerics(forms="exact")


def test_narrowed_operands_hand_back_their_gradients_unrounded():
    """A call whose key and value arrive wider than the mainloops compute them (`operand_dtype`, a sharding
    wrapper's, so its partial gradients sum wide and round once) is on par with the reference path (check
    1), its forward and its key's and value's gradients, which it hands back in their own fp32."""
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    draw = drawing(QUERIES, KEYS, jnp.bfloat16, sinks=False)

    def pulled_over(attend, query, key, value, dout):
        out, pull = jax.vjp(lambda k, v: attend(query, k, v), key, value)
        return (out, *pull(dout))

    def wide(query, key, value, dout):
        def attend(query, key, value):
            return attention_route.attend(query, key, value, spans, operand_dtype=jnp.bfloat16)
        return pulled_over(attend, query, key.astype(jnp.float32), value.astype(jnp.float32), dout)

    def routed(query, key, value, dout):
        def attend(query, key, value):
            return scaled_dot_product_attention(query, key, value, causal=True, implementation="reference")
        return pulled_over(attend, query, key, value, dout)

    def drawn(seed):
        query, key, value, _, dout = draw(seed)
        return query, key, value, dout

    assert all(leaf.dtype == jnp.float32 for leaf in jax.eval_shape(wide, *drawn(0))[1:])
    assert_as_exact(jax.jit(wide), jax.jit(routed), drawn, "wide operands narrowed in the route", seeds=SEEDS)


def test_the_template_reports_the_mainloop_each_pass_runs():
    """A template call reports the mainloop its forward and its backward run (`generation.route`,
    'attention_mainloop'), one route a pass, which the kernel report reads (bible §6); which mainloop runs
    is the accuracy gate's and the measurements' to judge, not this test's."""
    with jax.default_device(jax.devices("cpu")[0]):
        query, key, value, sinks, dout = drawing(QUERIES, KEYS, jnp.float32, sinks=True)(0)
        with generation.routes() as found:
            jax.jit(pulled("template", CASES["causal"])).trace(query, key, value, sinks, dout)
    assert len([route for route in found if route.choice == "attention_mainloop"]) == 2


@pytest.mark.parametrize("named", ["portable", "mosaic", "triton"])
def test_a_named_mainloop_runs_or_is_refused(named):
    """`generation.chosen(attention_mainloop=...)` is honoured on every template pass (bible §6): on a CPU
    the plain-JAX mainloop, which runs there, traces, and Mosaic's and Triton's, which run on a GPU alone, are
    refused by name (`selection.KernelRefused`), none silently replaced (train 217's K3)."""
    from dew.nn.kernels import selection

    with jax.default_device(jax.devices("cpu")[0]):
        query, key, value, sinks, dout = drawing(QUERIES, KEYS, jnp.float32, sinks=True)(0)
        traced = jax.jit(pulled("template", CASES["causal"]))
        with generation.chosen(attention_mainloop=named):
            if named != "portable":
                with pytest.raises(selection.KernelRefused) as refused:
                    traced.trace(query, key, value, sinks, dout)
                assert refused.value.kernel == named
                return
            traced.trace(query, key, value, sinks, dout)


def test_a_dropping_call_traces_for_an_assumed_gpu():
    """A call that drops attention weights traces for an H100 (`generation.assumed`): Mosaic declines it and
    the Pallas-Triton mainloop refuses its dropout (`attention_triton.refused`), which the route had read of
    its spans alone, running Triton, which raised (review)."""
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    query, key, value, _, _ = drawing(QUERIES, KEYS, jnp.bfloat16, sinks=False)(0)
    drop = template.Dropout(jnp.asarray([1, 2], jnp.uint32), jnp.arange(ROWS, dtype=jnp.int32),
                            jnp.arange(HEADS, dtype=jnp.int32), 0.25)
    with generation.assumed("sm90"):
        jax.jit(lambda q, k, v: attention_route.attend(q, k, v, spans, template.Hooks(drop=drop))).trace(
            query, key, value)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axes, whole, local", [({"data": 2}, 6, 3), ({"tensor": 2}, 14, 7)],
                         ids=["rows", "heads"])
def test_the_mosaic_mainloop_lowers_on_each_shards_own_rows_and_heads(axes, whole, local):
    """A causal bf16 call over six rows of fourteen heads that names the template and its Mosaic mainloop
    (`generation.chosen`), its forward and backward lowered for an H100 on a mesh splitting its rows or its
    heads in two (`generation.gpu_mesh`): no array of any Mosaic launch holds the call's whole count of rows
    or heads, and each launch's operands hold a shard's own, half the call's
    (`attention_parallel.on_attention_shards`; the counts no other axis takes), so no device runs the
    global batch; and the lowered program's own operations move no operand between devices
    (`sharded.MOVES`, as a `shard_map` body would issue them; what the partitioner inserts after lowering
    is the compiled program's, which this CPU does not build)
    (bible §7: a new accelerator route qualifies through its real target lowering on a multi-device
    mesh)."""
    from jax.sharding import AxisType, NamedSharding

    from dew.nn.kernels.lowered import custom_calls, operations

    devices = jax.make_mesh(tuple(axes.values()), tuple(axes), axis_types=(AxisType.Auto,) * len(axes),
                            devices=jax.devices()[:math.prod(axes.values())])
    mesh = generation.gpu_mesh(devices, "sm90")
    # Counts no other axis of a launch's arrays takes, so each array's rows or heads read apart.
    shape = jax.ShapeDtypeStruct((6, 128, 14, 64), jnp.bfloat16)
    spec = PartitionSpec("data") if "data" in axes else PartitionSpec(None, None, "tensor")
    placed = jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=NamedSharding(mesh, spec))

    def loss(query, key, value):
        out = scaled_dot_product_attention(query, key, value, causal=True, implementation="template")
        return out.astype(jnp.float32).sum()

    with (generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh),
          generation.chosen(attention_mainloop="mosaic")):
        lowered = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2))).trace(placed, placed, placed).lower(
            lowering_platforms=("cuda",))
    launches = [call for call in custom_calls(lowered) if call.target == "mosaic_gpu_v2"]
    assert launches, "the template lowers no Mosaic launch"
    for call in launches:
        arrays = (*call.operands, *call.results)
        assert all(whole not in array for array in arrays), call
        assert any(local in array for array in arrays if len(array) >= 3), call
    moved = [name for name in operations(lowered)
             if name.removeprefix("stablehlo.").replace("_", "-") in MOVES]
    assert not moved, moved


def test_the_route_takes_score_hooks_and_the_initial_state_and_refuses_an_output_hook():
    """`attend` runs a score hook and the initial state, and refuses an output hook of its own: its backward
    takes the output before one, so the caller applies it."""
    query, key, value, sinks, _ = drawing(QUERIES, KEYS, jnp.float32, sinks=True)(0)
    assert sinks is not None
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    hooks = template.Hooks(score_mod=template.Capped(SOFTCAP),
                           initial=template.sinks(sinks, ROWS, QUERIES, WIDTH))
    assert attention_route.attend(query, key, value, spans, hooks).shape == (ROWS, QUERIES, HEADS, WIDTH)
    with pytest.raises(ValueError):
        attention_route.attend(query, key, value, spans, template.Hooks(output_mod=jnp.tanh))


@pytest.mark.parametrize("generation_name", ["sm86", "sm120"])
def test_a_call_no_triton_config_fits_lowers_on_another_mainloop(generation_name):
    """An fp32 call of 256-wide heads on a GPU whose block holds 99 KB of shared memory, past the least any
    Pallas-Triton config stages (100 KB), forward and backward: the route takes another mainloop and the
    call lowers for that GPU, no launch that cannot run chosen (`attention_triton.unfitted`)."""
    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import attention as template, attention_route

    query = jax.ShapeDtypeStruct((1, 128, 1, 256), jnp.float32)
    spans = template.spans(128, 128, 1, causal=True)

    def loss(query, key, value):
        return attention_route.attend(query, key, value, spans).sum()

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation_name)
    with generation.assumed(generation_name), jax.sharding.use_abstract_mesh(mesh):
        traced = jax.jit(jax.grad(loss, argnums=(0, 1, 2))).trace(query, query, query)
        traced.lower(lowering_platforms=("cuda",))


def test_a_pass_the_mosaic_mainloop_declines_reports_its_reason_under_its_row():
    """A pass the Mosaic mainloop declines on a GPU is taken by a later mainloop, and its route reports
    Mosaic's own reason (the analysis's code and detail) under the row its requirement declares, so the
    kernel-route audit reads a gap and not an undeclared refusal (row 2545), whichever mainloop takes it; a
    Triton refusal is its inherent limit."""
    from dew.nn.kernels import selection

    declined = selection.Refusal("registers", "no tile holds the walk's registers")
    spans = selection.Refusal("dropout", "the Triton mainloop drops no probabilities")
    [requirement] = [requirement for requirement in attention_route.MAINLOOP.kernel("mosaic").supports
                     if requirement.code == "declined"]
    for triton in (None, spans):
        route = selection.select(attention_route.MAINLOOP, attention_route._Pass(
            on_gpu=True, on_mosaic=True, mosaic=declined, triton=triton))
        assert route.kernel != "mosaic"
        assert (route.refusal.code, route.refusal.detail, route.refusal.row) == (
            "registers", declined.detail, requirement.row)
    [refused] = attention_route.MAINLOOP.kernel("triton").failed(
        attention_route._Pass(on_gpu=True, on_mosaic=True, mosaic=None, triton=spans))
    assert (refused.code, refused.row) == ("dropout", generation.INHERENT)


def test_a_cpu_call_whose_xla_scores_pass_its_devices_memory_holds_temporaries_linear_in_its_tokens():
    """On a CPU 'auto' routes a call by time, jax.nn's xla attention where its training step's peak, as
    Dew's footprint model derives it, fits the device's share of the process's memory
    (`attention_kernels._dense_fits`, `generation.host_memory_share`): a causal fp32 training step of 8 heads
    64 wide, the first of 1024 tokens doubled whose xla attention does not fit, takes the template, its
    compiled temporaries below a byte a score, the template's plain-JAX mainloop's, which grow with the
    tokens, and within the process's memory (rows 4826, 5283)."""
    from dew.nn import attention_kernels
    from dew.nn.attention import attention_case

    heads, width, tokens = 8, 64, 1024
    while attention_kernels._dense_fits(attention_case(
            jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.float32),
            jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.float32), causal=True)):
        tokens *= 2
    shape = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.float32)

    def loss(query, key, value):
        return jnp.sum(scaled_dot_product_attention(query, key, value, causal=True))

    step = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    analysis = step.lower(shape, shape, shape).compile().memory_analysis()
    assert analysis is not None
    assert analysis.temp_size_in_bytes < min(heads * tokens * tokens, generation.host_memory_share())


@pytest.mark.parametrize(("groups", "limits", "least"), [
    ("0::/outer/inner\n", {"sys/fs/cgroup/memory.max": "max", "sys/fs/cgroup/outer/memory.max": "2147483648",
                           "sys/fs/cgroup/outer/inner/memory.max": "4294967296"}, 2147483648),
    ("4:cpu,cpuacct:/job\n5:memory:/outer/inner\n",
     {"sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712",
      "sys/fs/cgroup/memory/outer/inner/memory.limit_in_bytes": "1073741824"}, 1073741824),
    ("0::/system.slice/run.service\n", {"sys/fs/cgroup/memory.max": "3221225472"}, 3221225472),
], ids=["v2-limiting-parent", "v1-nested", "v2-container"])
def test_a_process_is_held_to_its_control_groups_least_memory_limit(tmp_path, groups, limits, least):
    """A CPU call's memory share reads the limits on this process's control groups (`/proc/self/cgroup`),
    its own group's and every ancestor's (`generation.cgroup_memory_limits`): a parent's tighter limit
    under cgroup v2, a nested group's under v1's memory controller, and a container's own group, its mount's
    root, where the group its process names is not mounted."""
    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/self/cgroup").write_text(groups)
    for path, limit in limits.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(limit + "\n")
    assert min(generation.cgroup_memory_limits(str(tmp_path))) == least


@pytest.mark.parametrize("differentiated", [False, True], ids=["inference", "training"])
def test_a_call_only_spans_express_holds_no_dense_mask(differentiated):
    """A causal call of block groups (Gemma 4's images, a decision row's blocks), which only spans express
    (`AttentionCase.spans_only`), runs on the template where its visibility stays spans, its forward and its
    training alike: its compiled program's temporaries grow with its tokens, not their square, twice the
    tokens taking under three times the bytes, where a dense `[S, S]` mask or the reference path's `B H S S`
    logits take four times (rows 1495, 1953)."""
    heads, width = 4, 64

    def temporaries(tokens: int) -> int:
        query, key, value = (jnp.ones((1, tokens, h, width), jnp.bfloat16) for h in (heads, 2, 2))
        groups = jnp.repeat(jnp.arange(tokens // 128, dtype=jnp.int32), 128)[None]

        def attended(query, key, value):
            return scaled_dot_product_attention(query, key, value, causal=True, block_groups=groups,
                                                implementation="template")

        step = jax.grad(lambda *operands: attended(*operands).astype(jnp.float32).sum(),
                        argnums=(0, 1, 2)) if differentiated else attended
        analysis = jax.jit(step).lower(query, key, value).compile().memory_analysis()
        assert analysis is not None
        return analysis.temp_size_in_bytes

    assert temporaries(4096) <= 3 * temporaries(2048)
