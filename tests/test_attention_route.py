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

from dew.mesh import DEFAULT_RULES
from dew.nn import attention
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.attention_kernels import ATTENTION, kernel_refusals
from dew.nn.inputs import counted_places
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.generation import INHERENT
from dew.nn.kernels.selection import KernelRefused
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_route,
    attention_triton,
)

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


@pytest.mark.skipif(generation.device_generation() != "sm90",
                    reason="composes the route's mainloops, on an H100")
@pytest.mark.gpu(reason="composes the route's Mosaic forward and Triton backward, on an H100")
@pytest.mark.parametrize("case", ["causal", "packed-documents", "sinks-softcap"])
def test_an_fp16_call_on_an_h100_runs_the_mosaic_forward_then_the_triton_backward_on_par(case):
    """fp16 on an H100: the Mosaic forward, which takes 16-bit operands, then the frozen Triton backward from
    its residuals, as the Mosaic backward takes bf16 alone (`attention_mosaic.refused`), composed in that
    order (traced among the launches, `tuning.seen`) and on par with the reference path (check 1)."""
    queries, keys = geometry(CASES[case])
    draw = drawing(queries, keys, jnp.float16, sinks=bool(CASES[case].get("sinks")))
    template_pass = jax.jit(pulled("template", CASES[case]))
    with tuning.seen() as launches:
        template_pass.trace(*draw(0))
    ops = [op for op, _ in launches]
    assert any(isinstance(op, attention_mosaic.MosaicForward) for op in ops)
    assert any(isinstance(op, attention_triton.TritonBackward) for op in ops)
    assert not any(isinstance(op, attention_mosaic.MosaicBackward) for op in ops)
    assert_as_exact(template_pass, jax.jit(pulled("reference", CASES[case])), draw,
                    f"fp16 Mosaic forward, Triton backward, {case}", seeds=SEEDS)


@pytest.mark.gpu(reason="walks a sequence of thousands of keys, on a GPU")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: jnp.dtype(dtype).name)
@pytest.mark.parametrize("length", [4096, 8192])
def test_a_long_sequence_is_on_par_with_the_reference_path(length, dtype):
    """A causal call of `length` tokens, two query heads over one key head (`WIDTH` wide), through the
    template on whichever mainloop the GPU runs it, on par with the reference path (check 1): D, the
    refinement and the denominator are sums the mainloops run in sequence over its key tiles where the
    reference sums by tree, so a ratio that grew with the sequence would show here (math review P2.5)."""
    case = {"causal": True}
    draw = drawing(length, length, dtype, sinks=False, heads=2, kv_heads=1, rows=1)
    assert_as_exact(jax.jit(pulled("template", case)), jax.jit(pulled("reference", case)), draw,
                    f"template route causal {length} {jnp.dtype(dtype).name}", seeds=SEEDS)


@pytest.mark.skipif(generation.device_generation() != "sm90", reason="runs the route's backward, on an H100")
@pytest.mark.gpu(reason="runs the route's backward, on an H100")
@pytest.mark.parametrize("case", ["causal", "packed-documents", "sinks-softcap"])
def test_on_an_h100_the_routes_backward_is_the_mosaic_template(case):
    """On an H100 a bf16 call's training forward and backward run the Mosaic template
    (`attention_mosaic.MosaicForward`, `MosaicBackward`, traced among the launches, `tuning.seen`), the
    frozen Triton mainloop not at all, a causal call's, a spans call's and a hooked one's alike
    (`attention_mosaic.forward_takes`, `backward_takes`), and so does an inference forward; the route's
    parity test holds them on par with the reference path."""
    queries, keys = geometry(CASES[case])
    operands = drawing(queries, keys, jnp.bfloat16, sinks=bool(CASES[case].get("sinks")))(0)
    with tuning.seen() as launches:
        jax.jit(pulled("template", CASES[case])).trace(*operands)
    ops = [op for op, _ in launches]
    assert any(isinstance(op, attention_mosaic.MosaicForward) for op in ops)
    assert any(isinstance(op, attention_mosaic.MosaicBackward) for op in ops)
    frozen = (attention_triton.TritonForward, attention_triton.TritonBackward)
    assert not any(isinstance(op, frozen) for op in ops)
    with tuning.seen() as launches:
        jax.jit(template_call(CASES[case])).trace(*operands[:4])
    ops = [op for op, _ in launches]
    assert any(isinstance(op, attention_mosaic.MosaicForward) for op in ops)
    assert not any(isinstance(op, attention_triton.TritonForward) for op in ops)


@pytest.mark.parametrize("kind", ["block_groups", "image_groups", "key_valid"])
def test_a_gpu_prefers_the_template_for_a_call_with_block_groups(kind):
    """A causal call with block or image groups, or key validity: flash and tokamax's Triton kernel refuse
    it by their inherent limits, and cuDNN takes it only as its mask, by measurement. 'auto' resolves a
    call with groups or key validity to the template on a family whose Mosaic template has a routed
    mainloop (wgmma's, its default), and to the generic path by legality on every other, no old kernel's
    spans expressing it. Without causality the groups add no key, so the call is an ordinary one, where
    validity still hides its keys."""
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
            assert (auto == "template") == (target == "sm90"), (target, auto)


def test_the_accurate_form_is_one_option_away():
    """`generation.chosen(attention_form='accurate')` runs every template call in the accurate form (natural
    logits and the accurate products, `attention_route.FORMS`), its forward and gradients bitwise a call's
    that names that form, and 'fast' runs the default, a call's that names none; the two forms differ; a
    form the template does not have is refused."""
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    query, key, value, _, dout = drawing(QUERIES, KEYS, jnp.float32, sinks=False)(0)

    def routed(q, k, v, **form):
        out, pull = jax.vjp(lambda q, k, v: attention_route.attend(q, k, v, spans, **form), q, k, v)
        return out, pull(dout)

    found = {}
    for form, named in (("accurate", {"natural": True, "accurate": True}), ("fast", {})):
        with generation.chosen(attention_form=form):
            found[form] = jax.tree.leaves(jax.jit(routed)(query, key, value))
        want = jax.tree.leaves(jax.jit(functools.partial(routed, **named))(query, key, value))
        for mine, theirs in zip(found[form], want, strict=True):
            np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs), err_msg=form)
    assert any(not np.array_equal(accurate, fast) for accurate, fast in zip(*found.values(), strict=True))
    with generation.chosen(attention_form="exact"), pytest.raises(ValueError):
        attention_route.attend(query, key, value, spans)


def test_narrowed_operands_hand_back_their_gradients_unrounded():
    """A call whose key and value arrive wider than the mainloops compute them (`operand_dtype`, a sharding
    wrapper's, so its partial gradients sum wide and round once) runs bitwise the narrow call's forward, and
    hands back the key's and value's gradients in their own fp32, which round to the narrow call's own."""
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    query, key, value, _, dout = drawing(QUERIES, KEYS, jnp.bfloat16, sinks=False)(0)

    def routed(k, v, **narrowed):
        return jax.vjp(lambda k, v: attention_route.attend(query, k, v, spans, **narrowed), k, v)

    narrow, pull = routed(key, value)
    wide, pull_wide = routed(key.astype(jnp.float32), value.astype(jnp.float32), operand_dtype=jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(wide), np.asarray(narrow))
    for mine, theirs in zip(pull_wide(dout), pull(dout), strict=True):
        assert mine.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(mine.astype(jnp.bfloat16)), np.asarray(theirs))


def test_the_template_reports_the_mainloop_each_pass_runs():
    """A template call reports the mainloop its forward and its backward run (`generation.route`,
    'attention_mainloop'), so the kernel-route audit sees a call on a GPU that leaves the Mosaic template: a
    call placed on a CPU the plain-JAX one's, which a CPU prefers."""
    with jax.default_device(jax.devices("cpu")[0]):
        query, key, value, sinks, dout = drawing(QUERIES, KEYS, jnp.float32, sinks=True)(0)
        with generation.routes() as found:
            jax.jit(pulled("template", CASES["causal"])).trace(query, key, value, sinks, dout)
    reported = [(route.kernel, route.preferred, route.refusal) for route in found
                if route.choice == "attention_mainloop"]
    assert reported == [("portable", "portable", None)] * 2


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
    the frozen Triton mainloop refuses its dropout (`attention_triton.refused`), which the route had read of
    its spans alone, running Triton, which raised (review)."""
    spans = template.spans(QUERIES, KEYS, ROWS, causal=True)
    query, key, value, _, _ = drawing(QUERIES, KEYS, jnp.bfloat16, sinks=False)(0)
    drop = template.Dropout(jnp.asarray([1, 2], jnp.uint32), jnp.arange(ROWS, dtype=jnp.int32),
                            jnp.arange(HEADS, dtype=jnp.int32), 0.25)
    with generation.assumed("sm90"):
        jax.jit(lambda q, k, v: attention_route.attend(q, k, v, spans, template.Hooks(drop=drop))).trace(
            query, key, value)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axes, dimension, local", [({"data": 2}, 0, 2), ({"tensor": 2}, 2, 2)],
                         ids=["rows", "heads"])
def test_the_mosaic_mainloop_lowers_on_each_shards_own_rows_and_heads(axes, dimension, local):
    """A causal bf16 call over four rows of four heads that names the template, its forward and backward
    lowered for an H100 on a mesh splitting its rows or its heads in two (`generation.gpu_mesh`): every
    Mosaic launch takes each shard's own rows or heads, half the call's (`attention._per_shard`), so no
    device runs the global batch, and no gather brings another shard's queries, keys or values to it (bible
    §7: a new accelerator route qualifies through its real target lowering on a multi-device mesh)."""
    import math
    import re

    from jax.sharding import AxisType, NamedSharding

    from dew.nn.attention import scaled_dot_product_attention

    devices = jax.make_mesh(tuple(axes.values()), tuple(axes), axis_types=(AxisType.Auto,) * len(axes),
                            devices=jax.devices()[:math.prod(axes.values())])
    mesh = generation.gpu_mesh(devices, "sm90")
    shape = jax.ShapeDtypeStruct((4, 128, 4, 64), jnp.bfloat16)
    split = PartitionSpec("data") if "data" in axes else PartitionSpec(None, None, "tensor")
    placed = jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=NamedSharding(mesh, split))

    def loss(query, key, value):
        out = scaled_dot_product_attention(query, key, value, causal=True, implementation="template")
        return out.astype(jnp.float32).sum()

    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2))).trace(placed, placed, placed).lower(
            lowering_platforms=("cuda",))
    text = lowered.as_text(debug_info=False)
    launches = re.findall(r'custom_call @mosaic_gpu_v2\(.*?\) .*?: \((tensor<[^>]*>)', text)
    assert launches, "the template lowers no Mosaic launch"
    for operand in launches:
        extents = [int(extent) for extent in operand.removeprefix("tensor<").split("x")[:-1]]
        if len(extents) == 4:
            assert extents[dimension] == local, operand
    assert "all_gather" not in text and "all_to_all" not in text


def test_the_mainloop_is_the_shared_selectors_choice():
    """The template's mainloop is an op of the shared selector (`attention_route.MAINLOOP`,
    `selection.select`), one chooser for its preference, its named choice and its route: on a GPU Mosaic
    preferred, and where it declines a pass the frozen Triton one runs, Mosaic's reason the refusal, its row
    the coverage the roadmap tracks; where Triton refuses the spans too, the plain-JAX one; off a GPU the
    plain-JAX one, no refusal (train 221's 1843)."""
    from dew.nn.kernels import selection

    declined = attention_route._Pass(on_gpu=True, mosaic="a reason", triton=None)
    with generation.routes():
        route = selection.select(attention_route.MAINLOOP, declined)
    assert (route.kernel, route.preferred, route.refusal.code, route.refusal.detail) == (
        "triton", "mosaic", "declined", "a reason")
    assert route.refusal.row
    with generation.routes():
        refused = attention_route._Pass(on_gpu=True, mosaic="a reason", triton="spans")
        assert selection.select(attention_route.MAINLOOP, refused).kernel == "portable"
        route = selection.select(attention_route.MAINLOOP, attention_route._Pass(on_gpu=False, mosaic=None,
                                                                                 triton=None))
    assert (route.kernel, route.preferred, route.refusal) == ("portable", "portable", None)


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
