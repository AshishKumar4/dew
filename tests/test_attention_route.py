"""dew.nn.attention's `implementation='template'` (`attention_route.attend`) on the CPU's plain-JAX mainloop.

Its output and its gradients of the query, key, value and sinks are on par with Dew's reference path from
float64 (check 1), for each call form the route takes, block groups among them, a cache call's included.
Every case has grouped key heads: four query heads over two. The selector refuses the route a mask and a
bias by their inherent limits; a target's measured row prefers it for a call with block groups, which no
other fused kernel reads, and otherwise only a call that names it runs it. On meshes, a named call runs per
shard under `sharded.assert_runs_per_shard`: with fewer key heads than the tensor shards, with key heads split
partly, with key heads split in the other order from the query heads, or with heads that take an axis the
rows' arguments name.
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
    block_groups,
    cross_documents,
    dews_mask,
    geometry,
    key_validity,
    own_documents,
)

from dew.nn import attention
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.attention_kernels import ATTENTION, OLD_ROUTES, kernel_refusals
from dew.nn.inputs import counted_places
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.generation import INHERENT
from dew.nn.kernels.selection import KernelRefused
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_mosaic_program,
    attention_route,
)
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
    path (`implementation='reference'`) measured from float64 over the site's draws (check 1); a case with
    sinks at `SINK_HEADS` over its own seeds (`SINK_SITES`)."""
    case = CASES[name]
    queries, keys = geometry(case)
    heads, kv_heads = SINK_HEADS if name in SINK_SITES else (HEADS, KV_HEADS)
    seeds = SINK_SITES[name][jnp.dtype(dtype).name] if name in SINK_SITES else SEEDS
    draw = drawing(queries, keys, dtype, sinks=bool(case.get("sinks")), heads=heads, kv_heads=kv_heads)
    assert_as_exact(jax.jit(pulled("template", case)), jax.jit(pulled("reference", case)), draw,
                    f"template route {case} {jnp.dtype(dtype).name}", seeds=seeds)


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


@pytest.mark.parametrize("name", ["block-causal-documents-window", "valid-documents-window-places"])
def test_a_dropped_out_call_draws_over_the_keys_its_block_groups_mean(name):
    """Under active dropout a call runs the reference path's one draw (`_dropout_attention`): with block
    groups, or key validity and counted places, documents and a window it draws over the mask its spans
    mean, so it is the same call as one given that mask (`dews_mask`) under the same key, bitwise."""
    case = CASES[name]
    query, key, value, _, _ = drawing(*geometry(case), jnp.float32, sinks=False)(0)
    dropped = functools.partial(scaled_dot_product_attention, dropout_rate=0.25,
                                dropout_rng=jax.random.key(3), deterministic=False)
    blocked = dropped(query, key, value, **keywords(case))
    masked = dropped(query, key, value, mask=jnp.asarray(dews_mask(case))[:, None])
    np.testing.assert_array_equal(np.asarray(blocked), np.asarray(masked))


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


def test_a_templated_family_runs_the_template_where_its_declared_route_does_not_take_a_call():
    """On a family whose Mosaic template has a routed mainloop (`attention_mosaic_program.BUNDLES`), 'auto'
    resolves each call to its declared old route where that takes it (`OLD_ROUTES`), else the template; on
    every other family to its row's route, never the template; and a call naming the template, or
    `generation.chosen` naming it for the attention choice, takes it everywhere (bible §4, owner 10-25)."""
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
        family = generation.family(target)
        with generation.assumed(target):
            for call in calls:
                # The kernel selected, not what this host runs of it (`attention_kernels._hosted`).
                with generation.routes() as found:
                    attention.resolve_implementation("auto", query, key, **call)
                (auto,) = [route.kernel for route in found if route.choice == "attention"]
                if family in attention_mosaic_program.BUNDLES:
                    old = OLD_ROUTES[family][0]
                    takes = not kernel_refusals(ATTENTION, old, attention.attention_case(query, key, **call))
                    assert auto == (old if takes else "template"), (target, call, auto)
                else:
                    assert auto != "template", (target, call)
                named = attention.resolve_implementation("template", query, key, **call)
                assert named == "template", (target, call)
    assert generation.family("sm90") in attention_mosaic_program.BUNDLES
    with generation.chosen(attention="template"):
        for call in calls:
            assert attention.resolve_implementation("auto", query, key, **call) == "template", call


@pytest.mark.skipif(generation.device_generation() != "sm90", reason="runs the route's backward, on an H100")
@pytest.mark.gpu(reason="runs the route's backward, on an H100")
@pytest.mark.parametrize("case", ["causal", "packed-documents", "sinks-softcap"])
def test_on_an_h100_the_routes_backward_is_the_mosaic_template(case):
    """On an H100 a bf16 call's backward runs the Mosaic mainloop's Template implementation
    (`attention_mosaic.MosaicBackward`, traced among the launches, `tuning.seen`) from the Triton forward's
    residuals, a causal call's, a spans call's and a hooked one's alike (`attention_mosaic.backward_takes`),
    which the route's parity test holds on par with the reference path."""
    queries, keys = geometry(CASES[case])
    operands = drawing(queries, keys, jnp.bfloat16, sinks=bool(CASES[case].get("sinks")))(0)
    with tuning.seen() as launches:
        jax.jit(pulled("template", CASES[case])).trace(*operands)
    assert any(isinstance(op, attention_mosaic.MosaicBackward) for op, _ in launches)


@pytest.mark.parametrize("kind", ["block_groups", "image_groups", "key_valid"])
def test_a_gpu_prefers_the_template_for_a_call_with_block_groups(kind):
    """A causal call with block or image groups, or key validity: flash and tokamax's Triton kernel refuse
    it by their inherent limits, and cuDNN takes it only as its mask, by measurement. 'auto' resolves a
    call with groups to the template on a family whose Mosaic template has a routed mainloop (wgmma's, its
    default) and where the target's measured row names it (`attention_blocks`: sm80's, where the template
    was faster and smaller than xla's mask at every length), and a call of key validity alone to the
    template on a templated family and to the generic path on every other, no row measuring it. Without
    causality the groups add no key, so the call is an ordinary one, where validity still hides its keys."""
    query = jnp.zeros((ROWS, QUERIES, HEADS, WIDTH), jnp.bfloat16)
    key = jnp.zeros((ROWS, QUERIES, KV_HEADS, WIDTH), jnp.bfloat16)
    groups = jnp.asarray(block_groups()[0])
    blocks = groups if kind == "block_groups" else None
    images = groups if kind == "image_groups" else None
    valid = key_validity(QUERIES) if kind == "key_valid" else None
    case = attention.attention_case(query, key, causal=True, block_groups=blocks, image_groups=images,
                                    key_valid=valid, named="template")
    assert case.spans_only and kernel_refusals(ATTENTION, "template", case) == []
    for kernel in ("flash", "triton"):
        assert ("spans_only", INHERENT) in [(refusal.code, refusal.row)
                                            for refusal in kernel_refusals(ATTENTION, kernel, case)]
    assert ("spans_only", "measured") in [(requirement.code, requirement.kind) for requirement
                                          in ATTENTION.kernel("cudnn").failed(case)]
    plain = attention.attention_case(query, key, block_groups=blocks, image_groups=images, key_valid=valid)
    assert plain.spans_only == (kind == "key_valid")
    for target in generation.TARGETS:
        with generation.assumed(target):
            auto = attention.resolve_implementation("auto", query, key, causal=True, block_groups=blocks,
                                                    image_groups=images, key_valid=valid)
            measured = generation.TARGETS[target].kernels.get("attention_blocks") == "template"
            templated = generation.family(target) in attention_mosaic_program.BUNDLES
            assert (auto == "template") == (templated or (measured and kind != "key_valid")), (target, auto)
    assert generation.TARGETS["sm80"].kernels["attention_blocks"] == "template"
    assert "attention_blocks" not in generation.TARGETS["sm90"].kernels


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
