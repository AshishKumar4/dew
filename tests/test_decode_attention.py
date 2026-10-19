"""The decode step's attention kernels against jax.nn's xla attention, from
float64: the folded form over a cache's own layout (`folded_attention`), the
dense decode kernel over each row's keys up to its length and the paged one
through each row's page table (`dew.nn.kernels.decode_attention`), and the
dense kernel's derivatives, which are the folded attention's. What runs on
CUDA skips elsewhere; the derivatives' lowering for a GPU runs on any host."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference, assert_fp32_reduction_bound

from dew.nn.attention import scaled_dot_product_attention, with_key_lengths
from dew.nn.decode_geometry import folded_attention

pytestmark = pytest.mark.kernel


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("heads", [1, 2])
def test_folded_attention_is_jax_nn_attention(dtype, heads):
    """Every query head against every (key, head) pair, the other heads'
    products thrown away, computes jax.nn's xla attention. In fp32 it only
    reorders three sums, the logits' over the width, the softmax's over the
    keys and the value product's over keys times heads (exact zeros between),
    so it sits within their summed chains of jax.nn's
    (`assert_fp32_reduction_bound`), the logits' rounding carried through the
    softmax at twice the largest logit magnitude. In bf16 it is within
    tests/reference_error.py's rule of jax.nn against float64. Rows read 1,
    17 and all 40 keys."""
    rng = np.random.default_rng(0)
    query = rng.normal(size=(3, 4, heads, 256))
    key, value = rng.normal(size=(2, 3, 40, heads, 256)) * np.array([0.3, 1.0])[:, None, None, None, None]
    lengths = jnp.asarray([1, 17, 40], jnp.int32)
    q, k, v = (jnp.asarray(x, dtype) for x in (query, key, value))
    folded = folded_attention(q, k, v, with_key_lengths(None, lengths, 40))
    plain = jax.nn.dot_product_attention(q, k, v, key_value_seq_lengths=lengths,
                                         implementation='xla')
    assert folded.shape == plain.shape and folded.dtype == plain.dtype

    q, k, v = (np.asarray(x, np.float64) for x in (q, k, v))
    read = np.arange(40)[None, None, None, :] < np.asarray(lengths)[:, None, None, None]
    logits = np.where(read, np.einsum('btnd,bsnd->bnts', q, k) / 16.0, -np.inf)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    truth = np.einsum('bnts,bsnd->btnd', probs, v)
    if dtype == jnp.float32:
        spread = np.where(read, np.einsum('btnd,bsnd->bnts', np.abs(q), np.abs(k)) / 16.0, 0).max(-1)
        carried = 1 + 2 * spread.transpose(0, 2, 1)[..., None]
        magnitudes = np.einsum('bnts,bsnd->btnd', probs, np.abs(v)) * carried
        assert_fp32_reduction_bound(folded, plain, magnitudes, terms=256 + 40 + 40 * heads)
        return
    assert_as_exact_as_the_reference(np.asarray(folded, np.float32), np.asarray(plain, np.float32), truth,
                                     "folded")


@pytest.mark.gpu(reason="the kernel runs on CUDA")
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_the_decode_kernel_reads_each_row_to_its_length(dtype):
    """`dew.nn.kernels.decode_attention` reads each row's keys up to its
    length, block by block, and is within tests/reference_error.py's rule of
    jax.nn's xla attention against float64: rows reading 1 key (an idle
    slot), a block's worth, one past it, and the whole capacity, at
    Qwen3.5-0.8B's widths (4 query positions over 2 key heads of 256). At the
    default precision its fp32 products are fp32 ones, held to jax.nn's at
    fp32 precision; Triton's default is one TF32 product."""
    from dew.nn.kernels import decode_attention

    rng = np.random.default_rng(0)
    capacity = 4 * decode_attention.BLOCK
    lengths = jnp.asarray([1, decode_attention.BLOCK, decode_attention.BLOCK + 1, 77, capacity], jnp.int32)
    rows = lengths.shape[0]
    query = jnp.asarray(rng.normal(size=(rows, 4, 2, 256)), dtype)
    key = jnp.asarray(rng.normal(size=(rows, capacity, 2, 256)) * 0.3, dtype)
    value = jnp.asarray(rng.normal(size=(rows, capacity, 2, 256)), dtype)
    assert decode_attention.fits(query, key)
    # The kernel at the default precision, which the GPU lane otherwise sets to the highest
    # (tests/lane_environment.py): its fp32 products must be fp32 ones whatever the default.
    with jax.default_matmul_precision("default"):
        out = jax.jit(decode_attention.attend)(query, key, value, lengths)
    with jax.default_matmul_precision("float32"):
        plain = jax.nn.dot_product_attention(query, key, value, key_value_seq_lengths=lengths,
                                             implementation='xla')
    assert out.shape == plain.shape and out.dtype == plain.dtype
    q, k, v = (np.asarray(x, np.float64) for x in (query, key, value))
    read = np.arange(capacity)[None, None, None, :] < np.asarray(lengths)[:, None, None, None]
    logits = np.where(read, np.einsum('btnd,bsnd->bnts', q, k) / 16.0, -np.inf)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    truth = np.einsum('bnts,bsnd->btnd', probs / probs.sum(-1, keepdims=True), v)
    assert_as_exact_as_the_reference(np.asarray(out, np.float32), np.asarray(plain, np.float32), truth,
                                     "decode kernel")


def decode_step(lengths):
    """fp32 attention of a decode step's grouped query (`decode_geometry.grouped_query`) over key
    lengths on 'xla', which the decode kernel takes on a GPU (`decode_attention.fits`), and the folded
    attention it stands in for over the prefix those lengths mean."""
    def attended(query, key, value):
        return scaled_dot_product_attention(query, key, value, implementation='xla',
                                            key_value_seq_lengths=lengths)

    def folded(query, key, value):
        return folded_attention(query, key, value, with_key_lengths(None, lengths, key.shape[-3]))

    return attended, folded


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_attention_the_decode_kernel_takes_lowers_for_the_gpu_differentiated_in_both_modes(generation):
    """Lowered for the GPU on this CPU, a decode step's attention runs the decode kernel for its value
    under reverse and forward mode alike, its Pallas call having no derivative of its own."""
    from dew.nn.kernels.generation import assumed, gpu_mesh
    from dew.training._lowering import triton_kernels

    attended, _ = decode_step(jnp.asarray([1, 20], jnp.int32))
    query = jax.ShapeDtypeStruct((2, 4, 2, 64), jnp.float32)
    key = jax.ShapeDtypeStruct((2, 64, 2, 64), jnp.float32)

    def loss(*arrays):
        return jnp.square(attended(*arrays)).sum()

    mesh = gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), generation)
    for differentiated in (jax.value_and_grad(loss, argnums=(0, 1, 2)),
                           lambda *arrays: jax.jvp(loss, arrays, arrays)):
        with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
            lowered = jax.jit(differentiated).trace(query, key, key).lower(lowering_platforms=("cuda",))
        assert triton_kernels(lowered) == {"decode_attention": 1}


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_grouped_query_in_its_own_layout_reaches_the_decode_kernel(generation):
    """A decode step's query of 4 heads over 2 key heads, called in the layer's own `[B, 1, N, D]`
    layout with key lengths: the core reads each key head's query heads as its query positions
    (`decode_geometry.grouped_query`), so the decode kernel takes it, lowered for the GPU on this CPU."""
    from dew.nn.kernels.generation import assumed, gpu_mesh
    from dew.training._lowering import triton_kernels

    query = jax.ShapeDtypeStruct((2, 1, 4, 64), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((2, 64, 2, 64), jnp.bfloat16)

    def attend(query, key, value):
        return scaled_dot_product_attention(query, key, value, implementation='xla',
                                            key_value_seq_lengths=jnp.asarray([1, 20], jnp.int32))

    mesh = gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(attend).trace(query, key, key).lower(lowering_platforms=("cuda",))
    assert triton_kernels(lowered) == {"decode_attention": 1}


@pytest.mark.parametrize("implementation", ["xla", "reference"])
def test_a_grouped_query_attends_as_its_heads_do_alone(implementation):
    """Grouped in the core, a lone query of 8 heads over 2 key heads with key lengths gives each head's
    attention over its key head's keys, within tests/reference_error.py's rule of the reference over
    the key heads repeated, against float64."""
    from dew.nn.attention import reference_attention, repeat_kv_heads

    rng = np.random.default_rng(3)
    lengths = jnp.asarray([1, 9, 32], jnp.int32)
    query, key, value = (jnp.asarray(rng.normal(size=shape), jnp.float32)
                         for shape in ((3, 1, 8, 32), (3, 32, 2, 32), (3, 32, 2, 32)))
    found = scaled_dot_product_attention(query, key, value, implementation=implementation,
                                         key_value_seq_lengths=lengths)

    def alone(query, key, value):
        mask = (jnp.arange(key.shape[1]) < lengths[:, None])[:, None, None, :]
        return reference_attention(query, repeat_kv_heads(key, 8), repeat_kv_heads(value, 8), mask=mask)

    with jax.enable_x64(new_val=True):
        truth = alone(*(jnp.asarray(x, jnp.float64) for x in (query, key, value)))
    assert_as_exact_as_the_reference(np.asarray(found), np.asarray(alone(query, key, value)),
                                     np.asarray(truth), f"grouped {implementation}")


def test_a_lone_query_whose_heads_do_not_group_is_refused_by_name():
    """A lone query of 5 heads over 3 key heads is no grouped query: the core leaves it as it is, and
    the kernel refuses its head counts, as it refuses them in any call."""
    query = jnp.zeros((1, 1, 5, 16), jnp.float32)
    key = jnp.zeros((1, 8, 3, 16), jnp.float32)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(query, key, key, implementation="reference",
                                     key_value_seq_lengths=jnp.asarray([4], jnp.int32))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("kv_heads", [2, 1])
def test_a_grouped_query_keeps_the_tensor_axis_split(kv_heads):
    """Over a tensor axis of two, a lone query of 8 heads over 2 key heads, or over 1 that the core
    repeats to the axis's 2 as each shard would hold it, is grouped per shard: each device's program
    does half the work of the call off the mesh, by XLA's own count on the reference path (no Pallas
    call, whose work XLA does not count), and attends as that call does."""
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.training import MeshSpec

    rng = np.random.default_rng(4)
    lengths = jnp.asarray([5, 32], jnp.int32)
    query, key, value = (jnp.asarray(rng.normal(size=shape), jnp.float32)
                         for shape in ((2, 1, 8, 32), (2, 32, kv_heads, 32), (2, 32, kv_heads, 32)))

    def attend(query, key, value):
        return scaled_dot_product_attention(query, key, value, implementation="reference",
                                            key_value_seq_lengths=lengths)

    def flops(compiled):
        analysis = compiled.cost_analysis()
        return (analysis[0] if isinstance(analysis, list) else analysis)["flops"]

    whole = jax.jit(attend).lower(query, key, value).compile()
    mesh = MeshSpec(tensor=2).build(jax.devices()[:2])
    with jax.set_mesh(mesh):
        heads = NamedSharding(mesh, PartitionSpec(None, None, "tensor"))
        placed = [jax.device_put(x, heads if x.shape[2] % 2 == 0 else NamedSharding(mesh, PartitionSpec()))
                  for x in (query, key, value)]
        compiled = jax.jit(attend).lower(*placed).compile()
        split = compiled(*placed)
    np.testing.assert_allclose(np.asarray(split), np.asarray(whole(query, key, value)), atol=1e-6, rtol=1e-6)
    assert flops(compiled) <= 0.6 * flops(whole), (flops(compiled), flops(whole))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_grouped_query_over_one_key_head_reaches_the_decode_kernel_on_each_tensor_shard(generation):
    """Lowered for the GPU over a tensor axis of two, a lone query of 8 heads over a single key head,
    which the core repeats to the axis's two, reaches the decode kernel, which reads each row's keys
    to its length alone, on each shard's group."""
    from dew.nn.kernels.generation import assumed, gpu_mesh
    from dew.training._lowering import triton_kernels

    query = jax.ShapeDtypeStruct((2, 1, 8, 64), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((2, 256, 1, 64), jnp.bfloat16)

    def attend(query, key, value):
        return scaled_dot_product_attention(query, key, value, implementation='xla',
                                            key_value_seq_lengths=jnp.asarray([1, 20], jnp.int32))

    mesh = gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:2]), ("tensor",)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(attend).trace(query, key, key).lower(lowering_platforms=("cuda",))
    assert triton_kernels(lowered) == {"decode_attention": 1}


@pytest.mark.gpu(reason="the kernel runs on CUDA")
def test_attention_the_decode_kernel_takes_differentiates_as_the_folded_attention():
    """A decode step's attention, its value the kernel's, differentiates as the folded attention it
    stands in for, in both modes: the value, query, key and value gradients, and the tangent, within
    tests/reference_error.py's rule of the folded attention's against float64 attention."""
    from dew.nn.kernels import decode_attention

    rng = np.random.default_rng(1)
    lengths = jnp.asarray([1, 20, 64], jnp.int32)
    attended, folded = decode_step(lengths)
    query, key, value, *tangents = (jnp.asarray(rng.normal(size=shape), jnp.float32) for shape in (
        (3, 4, 2, 64), (3, 64, 2, 64), (3, 64, 2, 64), (3, 4, 2, 64), (3, 64, 2, 64), (3, 64, 2, 64)))
    assert decode_attention.fits(query, key)
    cotangent = jnp.asarray(rng.normal(size=query.shape), jnp.float32)

    def derivatives(attend, *arrays):
        def pulled(*a):
            return jnp.sum(attend(*a) * cotangent.astype(a[0].dtype))

        out, tangent = jax.jvp(attend, arrays, tuple(t.astype(arrays[0].dtype) for t in tangents))
        return (out, *jax.grad(pulled, argnums=(0, 1, 2))(*arrays), tangent)

    found = jax.jit(lambda *a: derivatives(attended, *a))(query, key, value)
    reference = jax.jit(lambda *a: derivatives(folded, *a))(query, key, value)
    def exact(query, key, value):
        # Each row over its first `lengths` keys, every step in the operands' float64.
        logits = jnp.einsum('btnd,bsnd->bnts', query, key) / np.sqrt(query.shape[-1])
        read = jnp.arange(key.shape[1]) < lengths[:, None, None, None]
        weights = jax.nn.softmax(jnp.where(read, logits, -jnp.inf), axis=-1)
        return jnp.einsum('bnts,bsnd->btnd', weights, value)

    with jax.enable_x64(new_val=True):
        truth = derivatives(exact, *(jnp.asarray(x, jnp.float64) for x in (query, key, value)))
    names = ("value", "dq", "dk", "dv", "tangent")
    for name, got, want, exact in zip(names, found, reference, truth, strict=True):
        assert_as_exact_as_the_reference(np.asarray(got), np.asarray(want), np.asarray(exact),
                                         f"the decode kernel's {name}")


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axes, rows, heads", [({"data": 2}, 2, 4), ({"tensor": 2}, 4, 2)],
                         ids=["rows", "heads"])
def test_a_decode_step_launches_the_decode_kernel_on_each_shard_of_its_mesh(axes, rows, heads):
    """A decoder's single-token step over four rows of four heads, lowered for an A100 on a mesh that
    splits its rows or its heads in two, launches the decode kernel on each shard's own rows and
    heads (`manual_map`), the shapes its call names: a Pallas call has no partitioning rule, and
    launched on the whole batch it had XLA gather every row's cache to every device each token."""
    import math
    import re

    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kernels.generation import assumed, gpu_mesh

    devices = jax.make_mesh(tuple(axes.values()), tuple(axes), axis_types=(AxisType.Auto,) * len(axes),
                            devices=jax.devices()[:math.prod(axes.values())])
    mesh = gpu_mesh(devices, "sm80")
    model = CausalTransformer(vocab_size=32, emb_features=64, num_layers=1, num_heads=4, num_kv_heads=4,
                              head_dim=64, mlp_features=64, max_seq_len=64, dtype="bfloat16",
                              attention_impl="xla")
    tokens = jax.ShapeDtypeStruct((4, 1), jnp.int32)
    with assumed("sm80"):
        params = jax.eval_shape(model.init, jax.random.key(0), tokens)
        cache = jax.eval_shape(lambda params: model.apply(params, 4, method=CausalTransformer.init_cache,
                                                          mutable=["cache"])[1]["cache"], params)

    def placed(tree, *spec):
        sharding = NamedSharding(mesh, PartitionSpec(*spec))
        return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), tree)

    def step(params, cache, tokens):
        return model.apply({**params, "cache": cache}, tokens, decode=True, mutable=["cache"])

    with assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(step).trace(placed(params), placed(cache), placed(tokens, "data" if "data" in axes
                                                                            else None))
        text = lowered.lower(lowering_platforms=("cuda",)).as_text(debug_info=False)
    # Each launch's operand types: the row lengths, the padded queries, the cache's keys and values.
    called = r'custom_call @__gpu\$xla\.gpu\.triton\(.*?name = "decode_attention".*?: \((.*?)\) ->'
    launches = re.findall(called, text)
    assert launches
    for types in launches:
        lengths, query = types.split(", ")[:2]
        assert lengths == f"tensor<{rows}x1xi32>" and query.startswith(f"tensor<{rows}x16x{heads}x"), types


@pytest.mark.gpu(reason="the kernel runs on CUDA")
@pytest.mark.parametrize("page_size", [16, 32])
def test_the_paged_decode_kernel_reads_each_row_through_its_table(page_size):
    """`decode_attention.attend_paged` reads a row's pages through its table,
    shuffled and shared between rows, to the row's length, its query heads
    sharing key heads as Qwen3-0.6B's do (16 over 8 of 128), and is within
    tests/reference_error.py's rule of jax.nn's xla attention over the
    gathered rows against float64: rows reading 1 key, a page, one past a
    block, and every page."""
    from dew.nn.kernels import decode_attention

    rng = np.random.default_rng(0)
    per_row, kv_heads, heads, width = 6, 8, 16, 128
    capacity = per_row * page_size
    lengths = jnp.asarray([1, page_size, decode_attention.BLOCK + 1, capacity], jnp.int32)
    rows = lengths.shape[0]
    count = rows * per_row + 3
    table = jnp.asarray(rng.permutation(count)[:rows * per_row].reshape(rows, per_row), jnp.int32)
    table = table.at[1, 0].set(table[0, 0])
    query = jnp.asarray(rng.normal(size=(rows, heads, width)), jnp.bfloat16)
    key_pages = jnp.asarray(rng.normal(size=(kv_heads, count, page_size, width)) * 0.3, jnp.bfloat16)
    value_pages = jnp.asarray(rng.normal(size=(kv_heads, count, page_size, width)), jnp.bfloat16)
    assert decode_attention.fits_paged(query, key_pages)
    out = jax.jit(decode_attention.attend_paged)(query, key_pages, value_pages, table, lengths)
    key, value = (jnp.moveaxis(pages[:, table].reshape(kv_heads, rows, capacity, width), 0, 2)
                  for pages in (key_pages, value_pages))
    plain = jax.nn.dot_product_attention(query[:, None], key, value, key_value_seq_lengths=lengths,
                                         implementation='xla')[:, 0]
    assert out.shape == plain.shape and out.dtype == plain.dtype
    q = np.asarray(query, np.float64).reshape(rows, kv_heads, heads // kv_heads, width)
    k, v = np.asarray(key, np.float64), np.asarray(value, np.float64)
    read = np.arange(capacity)[None, None, None, :] < np.asarray(lengths)[:, None, None, None]
    logits = np.where(read, np.einsum('bngd,bsnd->bngs', q, k) / np.sqrt(width), -np.inf)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    truth = np.einsum('bngs,bsnd->bngd', probs / probs.sum(-1, keepdims=True), v).reshape(rows, heads, width)
    assert_as_exact_as_the_reference(np.asarray(out, np.float32), np.asarray(plain, np.float32), truth,
                                     "paged decode kernel")


@pytest.mark.parametrize("target", ["sm80", "sm90", "sm120"])
def test_the_decode_kernels_compile_on_tritons_compiler_within_each_gpus_block(target):
    """The dense kernel under a decode step's attention and its gradient, in bf16 and fp32 at
    Qwen3.5-0.8B's widths (4 query positions over 2 key heads of 256, a 384-slot cache), and the paged
    kernel at Qwen3-0.6B's (16 query heads over 8 of 128) with pages of 16 and 32 keys, lowered for
    each GPU on this CPU and compiled by Triton within its block's shared memory. A first filter: the
    A100 runs them; the H100 and the RTX PRO 6000 have this check alone."""
    pytest.importorskip("triton")
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_kernels, failures

    from dew.nn.kernels import decode_attention

    def dense(dtype):
        lengths = jax.ShapeDtypeStruct((8,), jnp.int32)
        query = jax.ShapeDtypeStruct((8, 4, 2, 256), dtype)
        cache = jax.ShapeDtypeStruct((8, 384, 2, 256), dtype)

        def step(query, key, value, lengths):
            def loss(query, key, value):
                return scaled_dot_product_attention(query, key, value, implementation='xla',
                                                    key_value_seq_lengths=lengths).astype(jnp.float32).sum()
            return jax.value_and_grad(loss, argnums=(0, 1, 2))(query, key, value)
        return step, (query, cache, cache, lengths)

    def paged(page_size):
        pages = jax.ShapeDtypeStruct((8, 8 * 24 + 3, page_size, 128), jnp.bfloat16)
        return decode_attention.attend_paged, (jax.ShapeDtypeStruct((8, 16, 128), jnp.bfloat16), pages, pages,
                                               jax.ShapeDtypeStruct((8, 24), jnp.int32),
                                               jax.ShapeDtypeStruct((8,), jnp.int32))

    cases = [KernelCase(f"dense/{jnp.dtype(dtype).name}", lambda dtype=dtype: dense(dtype))
             for dtype in (jnp.bfloat16, jnp.float32)]
    cases += [KernelCase(f"paged/{size}", lambda size=size: paged(size)) for size in (16, 32)]
    found = compile_kernels(cases, target)
    assert len(found) == len(cases)
    assert not failures(found), "\n".join(failures(found))
    # Each case lowered to a kernel Triton compiled, which takes shared memory.
    assert all(record["shared"] > 0 for record in found), found


def cache_call(window=None):
    """A cache call of 6 tokens over 16 slots: row 0 left-padded by 2 after an empty cache, row 1
    right-padded by 1 after 3 held tokens, row 2 with a hole after 7, and row 3 all padding; the
    tokens' slots (`open_kv_cache`'s positions, -1 for padding), the slots filled after the call, and
    the cache's mask over them (`causal_attention_mask`), a window's where given."""
    from dew.nn.attention import causal_attention_mask

    held = np.array([0, 3, 7, 5])
    valid = np.array([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0], [1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]], bool)
    positions = np.where(valid, held[:, None] + np.cumsum(valid, axis=1) - 1, -1)
    filled = np.arange(16)[None, :] < (held + valid.sum(axis=1))[:, None]
    return (jnp.asarray(positions, jnp.int32), jnp.asarray(filled),
            causal_attention_mask(positions, 16, window, key_valid=filled))


def cache_documents(positions, filled):
    """A cache call's tokens as one document a row whose last keys they are (`CrossDocuments`)."""
    from dew.nn.documents import CrossDocuments

    return CrossDocuments(jnp.asarray(positions >= 0, jnp.int32), jnp.asarray(filled, jnp.int32))


@pytest.mark.parametrize("window", [None, 4])
def test_a_cache_calls_documents_aligned_to_their_last_keys_are_its_mask(window):
    """A cache call's tokens, as one document a row whose last keys they are (`CrossDocuments`
    under causality and a window, `cross_document_mask`), see the keys the cache's own mask over its
    slots gives them, row by row, left, right and inner padding alike."""
    from dew.nn.documents import cross_document_mask

    positions, filled, mask = cache_call(window)
    aligned = cross_document_mask(cache_documents(positions, filled), causal=True, sliding_window=window)
    np.testing.assert_array_equal(np.asarray(aligned), np.asarray(mask))


@pytest.mark.parametrize("implementation", ["xla", "reference"])
@pytest.mark.parametrize("window", [None, 4])
def test_a_cache_call_read_as_documents_attends_as_its_mask(implementation, window):
    """Attention of a cache call over its documents (causal, windowed, `CrossDocuments`) is bitwise
    the attention over the cache's mask on the kernels that build it, a padding row's output zero."""
    positions, filled, mask = cache_call(window)
    rng = np.random.default_rng(7)
    query = jnp.asarray(rng.normal(size=(4, 6, 2, 32)), jnp.float32)
    key, value = (jnp.asarray(rng.normal(size=(4, 16, 2, 32)), jnp.float32) for _ in range(2))
    read = scaled_dot_product_attention(query, key, value, implementation=implementation, causal=True,
                                        sliding_window=window, segment_ids=cache_documents(positions, filled),
                                        max_segments=1)
    masked = scaled_dot_product_attention(query, key, value, implementation=implementation, mask=mask)
    np.testing.assert_array_equal(np.asarray(read), np.asarray(masked))
    np.testing.assert_array_equal(np.asarray(read[3]), 0)


@pytest.mark.gpu(reason="FlashAttention runs on CUDA")
@pytest.mark.parametrize("window", [None, 4])
def test_flash_reads_a_cache_calls_documents_as_exactly_as_jax_nn_reads_its_mask(without_deterministic_ops,
                                                                                 window):
    """FlashAttention over a cache call's documents (its packed sequences, the queries put in id
    order and back), as 'auto' routes it where the target row names flash for packed documents, is
    within tests/reference_error.py's rule of jax.nn's xla attention over the cache's mask, against
    float64, padding rows zero."""
    from dew.nn.attention import flash_refusal

    positions, filled, mask = cache_call(window)
    rng = np.random.default_rng(8)
    query = jnp.asarray(rng.normal(size=(4, 6, 2, 64)), jnp.bfloat16)
    key, value = (jnp.asarray(rng.normal(size=(4, 16, 2, 64)), jnp.bfloat16) for _ in range(2))
    documents = cache_documents(positions, filled)
    assert flash_refusal(query, key, True, window, segment_ids=documents, max_segments=1) is None
    found = scaled_dot_product_attention(query, key, value, implementation="flash", causal=True,
                                         sliding_window=window, segment_ids=documents, max_segments=1)
    plain = jax.nn.dot_product_attention(query, key, value, mask=mask, implementation="xla")
    q, k, v = (np.asarray(x, np.float64) for x in (query, key, value))
    logits = np.where(np.asarray(mask), np.einsum("btnd,bsnd->bnts", q, k) / 8.0, -np.inf)
    seen = np.isfinite(logits).any(-1, keepdims=True)
    probs = np.where(seen, np.exp(logits - np.where(seen, logits.max(-1, keepdims=True), 0)), 0)
    probs = probs / np.where(seen, probs.sum(-1, keepdims=True), 1)
    truth = np.einsum("bnts,bsnd->btnd", probs, v)
    plain = np.where(np.asarray(positions >= 0)[:, :, None, None], np.asarray(plain, np.float32), 0)
    assert_as_exact_as_the_reference(np.asarray(found, np.float32), plain, truth, f"window {window}")
