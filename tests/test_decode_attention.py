"""The decode step's attention kernels against jax.nn's xla attention, from
float64: the folded form over a cache's own layout (`folded_attention`), the
dense decode kernel over each row's keys up to its length and the paged one
through each row's page table (`dew.nn.kernels.decode_attention`), and the
dense kernel's derivatives, which are the folded attention's. What runs on
CUDA skips elsewhere; the derivatives' lowering for a GPU runs on any host."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import attention_terms, exact_attention, float64_weights
from reference_error import (
    FEWEST_SEEDS,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    assert_fp32_reduction_bound,
)

from dew.nn.attention import attention_case, scaled_dot_product_attention, with_key_lengths
from dew.nn.attention_kernels import DECODE, kernel_refusals
from dew.nn.decode_geometry import folded_attention
from dew.nn.kernels.generation import chosen, routes as generation_routes

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
    with jax.enable_x64(new_val=True):
        probs = np.asarray(float64_weights(jnp.asarray(q), jnp.asarray(k), read, None, None)[0])
        truth = np.asarray(exact_attention(*(jnp.asarray(x) for x in (q, k, v)), keep=read))
    if dtype == jnp.float32:
        spread = np.where(read, np.einsum('btnd,bsnd->bnts', np.abs(q), np.abs(k)) / 16.0, 0).max(-1)
        carried = 1 + 2 * spread.transpose(0, 2, 1)[..., None]
        magnitudes = np.einsum('bnts,bsnd->btnd', probs, np.abs(v)) * carried
        assert_fp32_reduction_bound(folded, plain, magnitudes, terms=256 + 40 + 40 * heads)
        return
    assert_as_exact_as_the_reference(np.asarray(folded, np.float32), np.asarray(plain, np.float32), truth,
                                     "folded")


ON_GPU = jax.default_backend() == "gpu"
"""Whether the kernels compile for this host's GPU; elsewhere they run interpreted, sized as on an
A100 (`on_the_gpu`)."""


def on_the_gpu():
    """The host's own GPU, or elsewhere an A100's resources, which size the kernels' launches."""
    from dew.nn.kernels.generation import assumed

    return contextlib.nullcontext() if ON_GPU else assumed("sm80")


def dense_case(dtype, windowed=False):
    """Rows reading 1 key (an idle slot), a block's worth, one past it, 77 keys and the whole capacity
    of 32 blocks, at Qwen3.5-0.8B's widths (4 query positions over 2 key heads of 256), and their
    float64 truth: 32 steps of the loop, a slice each at the most, so the slices of each config cut
    short rows to empty ones. `windowed` starts the rows within a block, at one, mid-row and a
    window of 300 keys before a full row's end, as a sliding window's decode reads them."""
    from dew.nn.kernels import decode_attention

    rng = np.random.default_rng(0)
    capacity = 32 * decode_attention.BLOCK
    lengths = jnp.asarray([1, decode_attention.BLOCK, decode_attention.BLOCK + 1, 77, capacity], jnp.int32)
    rows = lengths.shape[0]
    query = jnp.asarray(rng.normal(size=(rows, 4, 2, 256)), dtype)
    key = jnp.asarray(rng.normal(size=(rows, capacity, 2, 256)) * 0.3, dtype)
    value = jnp.asarray(rng.normal(size=(rows, capacity, 2, 256)), dtype)
    q, k, v = (np.asarray(x, np.float64) for x in (query, key, value))
    starts = jnp.asarray([0, 3, 1, 40, capacity - 300] if windowed else [0] * rows, jnp.int32)
    read = spanned(starts, lengths, capacity)
    with jax.enable_x64(new_val=True):
        truth = np.asarray(exact_attention(*(jnp.asarray(x) for x in (q, k, v)), keep=read))
    return (query, key, value, lengths, *((starts,) if windowed else ())), truth


def spanned(starts, lengths, capacity):
    """Each row's keys from its start to its length, `[rows, 1, 1, capacity]`."""
    keys = np.arange(capacity)[None, None, None, :]
    first, last = (np.asarray(x)[:, None, None, None] for x in (starts, lengths))
    return (keys >= first) & (keys < last)


@pytest.mark.parametrize("windowed", [False, True], ids=["prefix", "window"])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_the_decode_kernel_reads_each_row_to_its_length_at_every_config(dtype, windowed):
    """`dew.nn.kernels.decode_attention` reads each row's keys up to its length, from its start where
    a window gives one, block by block, at each slice count it may take (`SPLITS`, `tuning.pinned`:
    one slice a row, or more whose fp32 sums XLA combines), and is within tests/reference_error.py's
    rule of jax.nn's xla attention under the same keys' mask against float64. Compiled
    on a GPU at the default precision, which the GPU lane otherwise sets to the highest
    (tests/lane_environment.py), its fp32 products must be fp32 ones, held to jax.nn's at fp32
    precision; Triton's default is one TF32 product. Elsewhere it runs interpreted."""
    from dew.nn.kernels import decode_attention, tuning

    arrays, truth = dense_case(dtype, windowed)
    if ON_GPU:
        case = attention_case(arrays[0], arrays[1], key_value_seq_lengths=arrays[3],
                              key_value_seq_starts=arrays[4] if windowed else None)
        assert kernel_refusals(DECODE, "decode", case) == []

    def attend(*arrays):
        return decode_attention.attend(*arrays, interpret=not ON_GPU)

    starts = arrays[4] if windowed else jnp.zeros_like(arrays[3])
    with jax.default_matmul_precision("float32"):
        plain = jax.nn.dot_product_attention(*arrays[:3], mask=jnp.asarray(spanned(starts, arrays[3],
                                                                                   arrays[1].shape[1])),
                                             implementation='xla')
    for config in map(decode_attention.Split, decode_attention.SPLITS):
        pin = tuning.pinned({decode_attention.DecodeAttention: config})
        with on_the_gpu(), pin, jax.default_matmul_precision("default"):
            out = jax.jit(attend)(*arrays)
        assert out.shape == plain.shape and out.dtype == plain.dtype
        assert_as_exact_as_the_reference(np.asarray(out, np.float32), np.asarray(plain, np.float32), truth,
                                         f"decode kernel, {config}")


SINKS = (0.5, -1.0, 2.0, -3.0)
"""Four query heads' sink logits, around and past the rows' logits' scale."""
SOFTCAP = 0.5
"""A softcap near the scaled logits' spread, where `tanh` bends them."""


def grouped_case(dtype):
    """`dense_case`'s keys, lengths and capacity under 2 query positions of 4 query heads, a key head's
    pair reading it as gpt-oss's 64 read their 8."""
    (_, key, value, lengths), _ = dense_case(dtype)
    query = jnp.asarray(np.random.default_rng(1).normal(size=(key.shape[0], 2, 4, key.shape[-1])), dtype)
    return query, key, value, lengths


def grouped_draw(dtype):
    """`grouped_case`'s shapes drawn afresh by seed, check 1's draws: its queries, keys and values (the keys
    at its 0.3 scale) over its rows' lengths."""
    (query, key, _, _) = grouped_case(dtype)

    def draw(seed):
        rng = np.random.default_rng(1000 + seed)
        return (jnp.asarray(rng.normal(size=query.shape), dtype),
                jnp.asarray(rng.normal(size=key.shape) * 0.3, dtype),
                jnp.asarray(rng.normal(size=key.shape), dtype))

    return draw


@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
@pytest.mark.parametrize("hooks", [{"sinks": True}, {"softcap": True}, {"sinks": True, "softcap": True}],
                         ids=["sinks", "softcap", "both"])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_the_decode_kernel_applies_sinks_and_a_softcap_at_every_config(dtype, hooks, natural):
    """Sinks (one value-free logit a query head in each row's softmax) and a logit softcap
    (`cap · tanh(logit / cap)`) inside the decode kernel, over query heads grouped on their key heads:
    each row read to its length at each slice count it may take, the sink carried by the first slice alone
    so the combine counts it once, within tests/reference_error.py's rule of the reference path's
    attention over the same keys, against float64 (`exact_attention`). The idle row of one key keeps
    its sink's share. The dispatch takes such a call to the decode kernel where it runs (`decodes`). In
    both forms of the exponent, the fold and the accurate one (`natural`); each slice count held to
    check 1 over fresh draws of the case's shapes (`grouped_draw`)."""
    from dew.nn.attention import reference_attention
    from dew.nn.attention_kernels import decodes
    from dew.nn.kernels import decode_attention, tuning

    arrays = grouped_case(dtype)
    query, key, value, lengths = arrays
    sinks = jnp.asarray(SINKS, jnp.float32) if hooks.get("sinks") else None
    softcap = SOFTCAP if hooks.get("softcap") else None
    with on_the_gpu():
        case = attention_case(query, key, key_value_seq_lengths=lengths, sinks=sinks, softcap=softcap)
        assert decodes(case)
        assert kernel_refusals(DECODE, "decode", case) == [] or not ON_GPU
    read = spanned(jnp.zeros_like(lengths), lengths, key.shape[1])

    def reference(query, key, value):
        with jax.default_matmul_precision("float32"):
            return reference_attention(query, key, value, mask=jnp.asarray(read), sinks=sinks,
                                       softcap=softcap)

    def truth(query, key, value):
        wide = None if sinks is None else jnp.asarray(np.asarray(sinks, np.float64))
        return exact_attention(query, key, value, wide, softcap=softcap, mask=read)

    def attend(query, key, value, lengths):
        return decode_attention.attend(query, key, value, lengths, softcap=softcap, sinks=sinks,
                                       natural=natural, interpret=not ON_GPU)

    for config in map(decode_attention.Split, decode_attention.SPLITS):
        def pinned(query, key, value, config=config):
            pin = tuning.pinned({decode_attention.DecodeAttention: config})
            with pin, jax.default_matmul_precision("default"):
                return attend(query, key, value, lengths)

        with on_the_gpu():
            out = jax.jit(pinned)(query, key, value)
            assert out.shape == reference(query, key, value).shape and out.dtype == query.dtype
            assert_as_exact(jax.jit(pinned), jax.jit(reference), grouped_draw(dtype),
                            f"decode kernel with {sorted(hooks)}, natural {natural}, {config}",
                            seeds=FEWEST_SEEDS, truth_fn=truth)


@pytest.mark.parametrize("softcap", [None, SOFTCAP], ids=["plain", "softcap"])
def test_a_decode_steps_sinks_take_their_gradient_and_tangent_through_the_dispatch(softcap):
    """A grouped decode step with sinks through Dew's attention (`scaled_dot_product_attention`) routes
    to the decode kernel, the sinks an operand of its derivative: traced, the gradient and the tangent
    with respect to them take that route; on a GPU each is within tests/reference_error.py's rule of the
    reference path's against float64 (`exact_attention`)."""

    from dew.nn.kernels import generation

    query, key, value, lengths = grouped_case(jnp.float32)
    sinks = jnp.asarray(SINKS, jnp.float32)
    read = spanned(jnp.zeros_like(lengths), lengths, key.shape[1])

    def attend(implementation):
        return lambda sinks: scaled_dot_product_attention(
            query, key, value, sinks=sinks, softcap=softcap, key_value_seq_lengths=lengths,
            implementation=implementation)

    def grad(implementation):
        return jax.grad(lambda sinks: jnp.sum(attend(implementation)(sinks) ** 2))

    def tangent(implementation):
        return lambda sinks: jax.jvp(attend(implementation), (sinks,), (jnp.ones_like(sinks),))[1]

    with on_the_gpu(), generation.routes() as found:
        jax.make_jaxpr(grad("auto"))(sinks)
        jax.make_jaxpr(tangent("auto"))(sinks)
    assert {(route.choice, route.kernel) for route in found if route.choice == "decode"} == {
        ("decode", "decode")}
    if not ON_GPU:
        return
    with jax.enable_x64(new_val=True):
        wide = [jnp.asarray(np.asarray(x, np.float64)) for x in (query, key, value, sinks)]

        def exact(sinks):
            return exact_attention(*wide[:3], sinks, softcap=softcap, mask=read)

        exact_grad = jax.grad(lambda s: jnp.sum(exact(s) ** 2))(wide[3])
        exact_tangent = jax.jvp(exact, (wide[3],), (jnp.ones_like(wide[3]),))[1]
    with jax.default_matmul_precision("default"):
        got = jax.jit(grad("auto"))(sinks), jax.jit(tangent("auto"))(sinks)
    with jax.default_matmul_precision("float32"):
        want = jax.jit(grad("reference"))(sinks), jax.jit(tangent("reference"))(sinks)
    for label, mine, theirs, truth in zip(("gradient", "tangent"), got, want, (exact_grad, exact_tangent),
                                          strict=True):
        assert_as_exact_as_the_reference(np.asarray(mine), np.asarray(theirs), np.asarray(truth),
                                         f"the sinks' {label} through the decode kernel")


@pytest.mark.parametrize("target", ["sm80", "sm90"])
@pytest.mark.parametrize("mode, choice, kernel", [("decode", "decode", "decode"),
                                                  ("decode-paged", "paged_decode", "triton")])
def test_a_gpt_oss_decode_step_reads_its_sinks_in_the_decode_kernels(target, mode, choice, kernel):
    """gpt-oss's decode step (its committed tiny config at the released head width, 4 query heads over
    2 key heads, sinks on every layer), traced as the GPU takes it (`tools/kernel_routes.traced`): its
    attention runs Dew's decode kernel over a compact cache and its paged kernel over a paged one,
    which apply the sinks, where it had taken the reference path."""
    import json
    from pathlib import Path

    from tools import kernel_routes

    fixture = Path(__file__).parent / "fixtures" / "hf" / "gpt-oss-tiny" / "config.json"
    config, _ = kernel_routes.published(json.loads(fixture.read_text()))
    seen = kernel_routes.traced(config, target, mode)
    assert {route.kernel for route in seen if route.choice == choice} == {kernel}


def test_a_launch_reads_its_slices_at_its_rows_next_power_of_two():
    """33 rows of 2 key heads over 1024 keys, traced as on an A100 (108 multiprocessors) on any host:
    the launch reads its slices at 64 rows (`decode_attention._rows`), whose 128 programs fill most of
    the GPU, so it takes one slice, though its own 66 programs would have taken 8."""
    from tuned import taken

    from dew.nn.kernels import decode_attention
    from dew.nn.kernels.generation import assumed

    query = jax.ShapeDtypeStruct((33, 4, 2, 256), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct((33, 1024, 2, 256), jnp.bfloat16)
    lengths = jax.ShapeDtypeStruct((33,), jnp.int32)
    with assumed("sm80"):
        assert taken(decode_attention.DecodeAttention,
                     lambda *a: decode_attention.attend(*a, interpret=not ON_GPU), query, cache, cache,
                     lengths) == [decode_attention.Split(1)]
    assert decode_attention._heuristic(66, 32, 108) == decode_attention.Split(8)


@pytest.mark.parametrize("processors, programs, steps, splits", [
    (None, 2, 128, 1), (108, 128, 128, 1), (108, 2, 128, 32), (132, 16, 128, 32), (132, 32, 4096, 16),
    (132, 64, 4096, 8), (108, 2, 4, 4)])
def test_a_launch_without_a_measurement_fills_six_programs_a_multiprocessor(processors, programs, steps,
                                                                            splits):
    """The heuristic (`_heuristic`): no slices on a device of no known multiprocessors or where the
    programs fill most of them; else the most whose programs fill at most six a multiprocessor, at most
    one a step: two programs over 108 multiprocessors take 32, sixteen over 132 take 32, and 32 and 64
    take 16 and 8, the H100's fastest for Qwen3-0.6B's 4 and 8 rows over 131072 keys; four steps take
    at most 4."""
    from dew.nn.kernels import decode_attention

    assert decode_attention._heuristic(programs, steps, processors) == decode_attention.Split(splits)
    # The slices it may take hold the heuristic's, and none past one where the programs fill the GPU.
    space = decode_attention._space(programs, steps, processors)
    assert decode_attention.Split(splits) in space
    assert (space == {decode_attention.Split(1)}) == (not processors or programs >= 0.8 * processors)


def decode_step(lengths):
    """fp32 attention of a decode step's grouped query (`decode_attention.grouped_query`) over key
    lengths through the decode kernel, chosen for the decode selector (`attention_kernels.DECODE`),
    which takes it on a GPU (`decode_attention.SUPPORTS`), and the folded attention it stands in for
    over the prefix those lengths mean."""
    def attended(query, key, value):
        with chosen(decode="decode"):
            return scaled_dot_product_attention(query, key, value, key_value_seq_lengths=lengths)

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
    (`decode_attention.grouped_query`), so the decode kernel takes it, lowered for the GPU on this CPU."""
    from dew.nn.kernels.generation import assumed, gpu_mesh
    from dew.training._lowering import triton_kernels

    query = jax.ShapeDtypeStruct((2, 1, 4, 64), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((2, 64, 2, 64), jnp.bfloat16)

    def attend(query, key, value):
        with chosen(decode="decode"):
            return scaled_dot_product_attention(query, key, value,
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


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_windowed_layers_decode_step_reads_its_window_through_the_decode_kernel(generation):
    """A causal transformer whose layers slide a window of 16 keys over a 64-slot cache, its decode
    step lowered for the GPU on this CPU: each layer reads its window through the decode kernel, from
    each row's first key under it, where a mask had sent the step to jax.nn's xla attention."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kernels.generation import assumed, gpu_mesh
    from dew.training._lowering import triton_kernels

    model = CausalTransformer(vocab_size=64, num_layers=2, emb_features=64, num_heads=4, num_kv_heads=2,
                              max_seq_len=64, layer_types=("sliding_attention",) * 2,
                              kinds={"sliding_attention": {"window": 16}}, attention_impl="xla",
                              dtype=jnp.bfloat16)
    tokens = jnp.zeros((2, 1), jnp.int32)
    variables = jax.eval_shape(lambda: model.init(jax.random.key(0), tokens))
    cache = jax.eval_shape(lambda variables: model.apply(variables, 2, method=CausalTransformer.init_cache,
                                                         mutable=["cache"])[1]["cache"], variables)

    def step(variables, cache, tokens):
        return model.apply({**variables, "cache": cache}, tokens, decode=True, mutable=["cache"])

    mesh = gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(step).trace(variables, cache, tokens).lower(lowering_platforms=("cuda",))
    assert triton_kernels(lowered)["decode_attention"] == 2


def test_a_windowed_decode_step_attends_as_its_whole_pass():
    """A windowed layer's decode, its keys from each row's first under a window of 5 over a 64-slot
    cache at widths the decode kernel takes (heads 16 wide), gives what its whole pass gives at each
    token, within tests/reference_error.py's rule of the whole pass against the float64 whole pass:
    the starts the decode reads are the window's keys. On a GPU the same steps run the decode kernel
    in every layer."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.training._lowering import triton_kernels

    model = CausalTransformer(vocab_size=64, num_layers=2, emb_features=64, num_heads=4, num_kv_heads=2,
                              max_seq_len=64, layer_types=("sliding_attention",) * 2,
                              kinds={"sliding_attention": {"window": 5}}, attention_impl="xla")
    tokens = jax.random.randint(jax.random.key(3), (2, 12), 0, 64)
    variables = model.init(jax.random.key(0), tokens)
    whole = model.apply(variables, tokens)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
        truth = np.asarray(model.clone(dtype=jnp.float64).apply(wide, tokens), np.float64)
    cache = model.apply(variables, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
    _, state = model.apply({**variables, "cache": cache}, tokens[:, :4], decode=True, mutable=["cache"])

    @jax.jit
    def step(cache, token):
        return model.apply({**variables, **cache}, token, decode=True, mutable=["cache"])

    if ON_GPU:
        assert triton_kernels(step.trace(state, tokens[:, 4:5]).lower())["decode_attention"] == 2
    steps = []
    for index in range(4, 12):
        out, state = step(state, tokens[:, index:index + 1])
        steps.append(out)
    decoded = np.asarray(jnp.concatenate(steps, axis=1), np.float64)
    assert_as_exact_as_the_reference(decoded, np.asarray(whole[:, 4:], np.float64), truth[:, 4:], "decode")


@pytest.mark.parametrize("options", [{}, {"force_fp32_for_softmax": False},
                                     {"precision": jax.lax.Precision.HIGHEST}],
                         ids=["default", "softmax", "precision"])
def test_key_starts_attend_as_the_mask_they_mean(options):
    """A lone grouped query over keys from each row's start to its length (`key_value_seq_starts`)
    beside an explicit mask, under 'auto', in the arithmetic its options ask for (the reference
    path's where they ask for it), gives what the same keys as one explicit mask give, bitwise, a row
    whose mask and start leave no key exactly zero, and its gradients the same."""
    rng = np.random.default_rng(5)
    query, key, value = (jnp.asarray(rng.normal(size=shape), jnp.float32)
                         for shape in ((3, 1, 4, 16), (3, 64, 2, 16), (3, 64, 2, 16)))
    lengths, starts = jnp.asarray([2, 40, 64], jnp.int32), jnp.asarray([1, 8, 0], jnp.int32)
    # The first row's mask keeps key 0 alone, below its start: it reads nothing.
    mask = jnp.ones((3, 1, 1, 64), bool).at[0, ..., 1:].set(False)

    def spans(query, key, value):
        return scaled_dot_product_attention(query, key, value, mask=mask, key_value_seq_lengths=lengths,
                                            key_value_seq_starts=starts, **options)

    def masked(query, key, value):
        keys = jnp.arange(64)
        kept = mask & ((keys >= starts[:, None]) & (keys < lengths[:, None]))[:, None, None, :]
        return scaled_dot_product_attention(query, key, value, mask=kept, **options)

    got, want = spans(query, key, value), masked(query, key, value)
    np.testing.assert_array_equal(np.asarray(got[0]), 0)
    # A mask, the starts folded into it here, is one path: the two calls agree bitwise.
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    grads = [jax.grad(lambda *a, f=f: jnp.sum(f(*a) ** 2), argnums=(0, 1, 2))(query, key, value)
             for f in (spans, masked)]
    for mine, theirs in zip(*grads, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


@pytest.mark.parametrize("options", [{"force_fp32_for_softmax": False},
                                     {"precision": jax.lax.Precision.HIGHEST}], ids=["softmax", "precision"])
def test_key_starts_keep_the_arithmetic_auto_is_asked_for_where_the_decode_kernel_would_run(options):
    """A lone grouped query of decode-kernel widths over key starts and lengths alone, traced as on
    an A100, asking for arithmetic only the reference path performs: 'auto' takes the reference
    path, not the decode kernel (which refused the options by name), and is within
    tests/reference_error.py's rule of the one-mask form against float64; the grouped query sums
    in another order."""
    from dew.nn.kernels.generation import assumed

    rng = np.random.default_rng(6)
    query, key, value = (jnp.asarray(rng.normal(size=shape), jnp.float32)
                         for shape in ((3, 1, 4, 16), (3, 64, 2, 16), (3, 64, 2, 16)))
    lengths, starts = jnp.asarray([2, 40, 64], jnp.int32), jnp.asarray([1, 8, 0], jnp.int32)
    keys = jnp.arange(64)
    kept = ((keys >= starts[:, None]) & (keys < lengths[:, None]))[:, None, None, :]
    with assumed("sm80"):
        got = scaled_dot_product_attention(query, key, value, key_value_seq_lengths=lengths,
                                           key_value_seq_starts=starts, **options)
        want = scaled_dot_product_attention(query, key, value, mask=kept, **options)
    with jax.enable_x64(new_val=True):
        truth = scaled_dot_product_attention(*(jnp.asarray(x, jnp.float64) for x in (query, key, value)),
                                             mask=kept, implementation="reference")
    assert_as_exact_as_the_reference(np.asarray(got), np.asarray(want), np.asarray(truth),
                                     "starts under reference arithmetic")


@pytest.mark.parametrize("cross", [False, True], ids=["documents", "cross-documents"])
def test_a_row_whose_documents_lie_below_its_start_reads_nothing(cross):
    """Packed documents and a key start, no lengths or mask: the first query's only key of its
    document lies below its row's start, so it reads nothing and comes back exactly zero, its
    cotangent too, as the one-mask form gives; the other query reads its key."""
    query = jnp.zeros((1, 2 if not cross else 1, 1, 16), jnp.float32)
    key = jnp.zeros((1, 2, 1, 16), jnp.float32)
    value = jnp.ones((1, 2, 1, 16), jnp.float32)
    starts = jnp.asarray([1], jnp.int32)
    documents = ({"segment_ids": jnp.asarray([[1]]), "key_segment_ids": jnp.asarray([[1, 2]])} if cross
                 else {"segment_ids": jnp.asarray([[1, 2]])})

    def attend(value):
        return scaled_dot_product_attention(query, key, value, key_value_seq_starts=starts, **documents)

    out = attend(value)
    np.testing.assert_array_equal(np.asarray(out[:, 0]), 0)
    if not cross:
        np.testing.assert_array_equal(np.asarray(out[:, 1]), 1)
    cotangent = jax.grad(lambda value: jnp.sum(attend(value)[:, 0]))(value)
    np.testing.assert_array_equal(np.asarray(cotangent), 0)


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
        with chosen(decode="decode"):
            return scaled_dot_product_attention(query, key, value,
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
    rng = np.random.default_rng(1)
    lengths = jnp.asarray([1, 20, 64], jnp.int32)
    attended, folded = decode_step(lengths)
    query, key, value, *tangents = (jnp.asarray(rng.normal(size=shape), jnp.float32) for shape in (
        (3, 4, 2, 64), (3, 64, 2, 64), (3, 64, 2, 64), (3, 4, 2, 64), (3, 64, 2, 64), (3, 64, 2, 64)))
    assert kernel_refusals(DECODE, "decode", attention_case(query, key, key_value_seq_lengths=lengths)) == []
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
        return exact_attention(query, key, value,
                               keep=np.arange(key.shape[1]) < np.asarray(lengths)[:, None, None, None])

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

    from jax.sharding import AxisType
    from sharded import placed

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

    def step(params, cache, tokens):
        return model.apply({**params, "cache": cache}, tokens, decode=True, mutable=["cache"])

    # A model naming xla lowers its decode step through the decode kernel, jax.nn's lowering of it.
    with assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(step).trace(placed(params, mesh), placed(cache, mesh),
                                      placed(tokens, mesh, "data" if "data" in axes else None))
        text = lowered.lower(lowering_platforms=("cuda",)).as_text(debug_info=False)
    # Each launch's operand types: the rows' key spans, the padded queries, the cache's keys and values.
    called = r'custom_call @__gpu\$xla\.gpu\.triton\(.*?name = "decode_attention".*?: \((.*?)\) ->'
    launches = re.findall(called, text)
    assert launches
    for types in launches:
        spans, query = types.split(", ")[:2]
        assert spans == f"tensor<{rows}x2xi32>" and query.startswith(f"tensor<{rows}x16x{heads}x"), types


def paged_case(page_size, windowed=False):
    """Rows reading 1 key, a page, one past a block and every page of 32 blocks through tables
    shuffled and shared between rows, their query heads sharing key heads as Qwen3-0.6B's do (16 over
    8 of 128), and the gathered rows' keys and values and the float64 truth. `windowed` starts the
    rows within a page, past a block and a window of 300 keys before a full row's end."""
    from dew.nn.kernels import decode_attention

    rng = np.random.default_rng(0)
    per_row, kv_heads, heads, width = 32 * decode_attention.BLOCK // page_size, 8, 16, 128
    capacity = per_row * page_size
    lengths = jnp.asarray([1, page_size, decode_attention.BLOCK + 1, capacity], jnp.int32)
    rows = lengths.shape[0]
    count = rows * per_row + 3
    table = jnp.asarray(rng.permutation(count)[:rows * per_row].reshape(rows, per_row), jnp.int32)
    table = table.at[1, 0].set(table[0, 0])
    query = jnp.asarray(rng.normal(size=(rows, heads, width)), jnp.bfloat16)
    key_pages = jnp.asarray(rng.normal(size=(kv_heads, count, page_size, width)) * 0.3, jnp.bfloat16)
    value_pages = jnp.asarray(rng.normal(size=(kv_heads, count, page_size, width)), jnp.bfloat16)
    key, value = (jnp.moveaxis(pages[:, table].reshape(kv_heads, rows, capacity, width), 0, 2)
                  for pages in (key_pages, value_pages))
    q = np.asarray(query, np.float64)[:, None]
    k, v = np.asarray(key, np.float64), np.asarray(value, np.float64)
    starts = jnp.asarray([0, 1, decode_attention.BLOCK + 1 - 2, capacity - 300] if windowed else [0] * rows,
                         jnp.int32)
    read = spanned(starts, lengths, capacity)
    with jax.enable_x64(new_val=True):
        truth = np.asarray(exact_attention(*(jnp.asarray(x) for x in (q, k, v)), keep=read))[:, 0]
    arrays = (query, key_pages, value_pages, table, lengths, *((starts,) if windowed else ()))
    return arrays, (key, value, starts, truth)


@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
@pytest.mark.parametrize("windowed", [False, True], ids=["prefix", "window"])
@pytest.mark.parametrize("page_size", [16, 32])
def test_the_paged_decode_kernel_reads_each_row_through_its_table_at_every_config(page_size, windowed,
                                                                                  natural):
    """`decode_attention.attend_paged` reads a row's pages through its table to the row's length, from
    its start where a window gives one, at each slice count it may take (`SPLITS`), and is within
    tests/reference_error.py's rule of jax.nn's xla attention over the gathered rows under the same
    keys' mask against float64, compiled on a GPU and interpreted elsewhere, in both forms of the
    exponent (`natural`)."""
    from dew.nn.kernels import decode_attention, tuning
    from dew.nn.kv_cache import PAGED, PagedCase

    arrays, (key, value, starts, truth) = paged_case(page_size, windowed)
    query, lengths = arrays[0], arrays[4]
    if ON_GPU:
        case = PagedCase(heads=query.shape[1], kv_heads=arrays[1].shape[0], head_dim=query.shape[2],
                         value_dim=query.shape[2], dtype=query.dtype, pool=arrays[1].dtype,
                         page_size=page_size, quantized=False, groups=1, head_shards=1, softcap=None,
                         arithmetic=False, named="auto", windowed=windowed)
        assert PAGED.kernel("triton").failed(case) == []

    def attend(*arrays):
        return decode_attention.attend_paged(*arrays, natural=natural, interpret=not ON_GPU)

    plain = jax.nn.dot_product_attention(query[:, None], key, value,
                                         mask=jnp.asarray(spanned(starts, lengths, key.shape[1])),
                                         implementation='xla')[:, 0]
    for config in map(decode_attention.Split, decode_attention.SPLITS):
        with on_the_gpu(), tuning.pinned({decode_attention.PagedDecodeAttention: config}):
            out = jax.jit(attend)(*arrays)
        assert out.shape == plain.shape and out.dtype == plain.dtype
        assert_as_exact_as_the_reference(np.asarray(out, np.float32), np.asarray(plain, np.float32), truth,
                                         f"paged decode kernel, natural {natural}, {config}")


@pytest.mark.parametrize("target", ["sm80", "sm90", "sm120"])
def test_the_decode_kernels_compile_on_tritons_compiler_within_each_gpus_block(target):
    """The dense kernel under a decode step's attention and its gradient, in bf16 and fp32 at
    Qwen3.5-0.8B's widths (4 query positions over 2 key heads of 256, a 1024-slot cache), and the
    paged kernel at Qwen3-0.6B's (16 query heads over 8 of 128) with pages of 16 and 32 keys over 1024
    keys a row, each at every slice count (`tuning.pinned`), which their 8 rows' 32 steps take on
    every GPU here, lowered for each GPU on this CPU and compiled by Triton
    within its block's shared memory; and the dense kernel in the accurate form (`natural`), raw without
    sinks and scaled with them, which the interpreted suite lowers for no GPU. A first filter: the A100
    runs them; the H100 and the RTX PRO 6000 have this check alone."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_kernels, failures

    from dew.nn.kernels import decode_attention, tuning

    def pinned(call, op, splits):
        fn, args = call()

        def run(*args):
            with tuning.pinned({op: decode_attention.Split(splits)}):
                return fn(*args)
        return run, args

    def dense(dtype):
        lengths = jax.ShapeDtypeStruct((8,), jnp.int32)
        query = jax.ShapeDtypeStruct((8, 4, 2, 256), dtype)
        cache = jax.ShapeDtypeStruct((8, 1024, 2, 256), dtype)

        def step(query, key, value, lengths):
            def loss(query, key, value):
                with chosen(decode="decode"):
                    out = scaled_dot_product_attention(query, key, value, key_value_seq_lengths=lengths)
                return out.astype(jnp.float32).sum()
            return jax.value_and_grad(loss, argnums=(0, 1, 2))(query, key, value)
        return step, (query, cache, cache, lengths)

    def natural(sinks: bool):
        lengths = jax.ShapeDtypeStruct((8,), jnp.int32)
        query = jax.ShapeDtypeStruct((8, 4, 2, 256), jnp.bfloat16)
        cache = jax.ShapeDtypeStruct((8, 1024, 2, 256), jnp.bfloat16)
        held = jnp.asarray(SINKS[:2], jnp.float32) if sinks else None

        def attend(query, key, value, lengths):
            return decode_attention.attend(query, key, value, lengths, sinks=held, natural=True)
        return attend, (query, cache, cache, lengths)

    def paged(page_size):
        per_row = 1024 // page_size
        pages = jax.ShapeDtypeStruct((8, 8 * per_row + 3, page_size, 128), jnp.bfloat16)
        return decode_attention.attend_paged, (jax.ShapeDtypeStruct((8, 16, 128), jnp.bfloat16), pages, pages,
                                               jax.ShapeDtypeStruct((8, per_row), jnp.int32),
                                               jax.ShapeDtypeStruct((8,), jnp.int32))

    # Every slice count the kernels may take, each a program shape of its own.
    cases = [KernelCase(f"dense/{jnp.dtype(dtype).name}/{splits}",
                        lambda dtype=dtype, splits=splits: pinned(lambda: dense(dtype),
                                                                  decode_attention.DecodeAttention, splits))
             for dtype in (jnp.bfloat16, jnp.float32) for splits in decode_attention.SPLITS]
    cases += [KernelCase(f"natural/{'sinks' if sinks else 'raw'}/{splits}",
                         lambda sinks=sinks, splits=splits: pinned(lambda: natural(sinks),
                                                                   decode_attention.DecodeAttention, splits))
              for sinks in (False, True) for splits in decode_attention.SPLITS]
    paged_op = decode_attention.PagedDecodeAttention
    cases += [KernelCase(f"paged/{size}/{splits}",
                         lambda size=size, splits=splits: pinned(lambda: paged(size), paged_op, splits))
              for size in (16, 32) for splits in decode_attention.SPLITS]
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
    """FlashAttention chosen for a cache call (`generation.chosen`) reads its documents as its packed
    sequences, the queries put in id order and back: its output and its query, key and value
    gradients are within tests/reference_error.py's rule of jax.nn's xla attention over the cache's
    mask, against float64, padding rows zero."""
    from dew.nn.attention_kernels import ATTENTION

    positions, filled, mask = cache_call(window)
    rng = np.random.default_rng(8)
    query = jnp.asarray(rng.normal(size=(4, 6, 2, 64)), jnp.bfloat16)
    key, value = (jnp.asarray(rng.normal(size=(4, 16, 2, 64)), jnp.bfloat16) for _ in range(2))
    cotangent = jnp.asarray(rng.normal(size=(4, 6, 2, 64)) * np.asarray(positions >= 0)[:, :, None, None],
                            jnp.float32)
    documents = cache_documents(positions, filled)

    def pulled(attend, *operands):
        wide = jnp.promote_types(operands[0].dtype, jnp.float32)
        out, pull = jax.vjp(lambda *operands: attend(*operands).astype(wide), *operands)
        return out, *pull(cotangent.astype(out.dtype))

    def flash(query, key, value):
        return scaled_dot_product_attention(query, key, value, implementation="flash", causal=True,
                                            sliding_window=window, segment_ids=documents, max_segments=1)

    def plain(query, key, value):
        out = jax.nn.dot_product_attention(query, key, value, mask=mask, implementation="xla")
        return out * jnp.asarray(positions >= 0, out.dtype)[:, :, None, None]

    def truth(query, key, value):
        return exact_attention(query, key, value, keep=np.asarray(mask))

    with chosen(attention="flash", attention_packed="flash", attention_window="flash"):
        case = attention_case(query, key, causal=True, sliding_window=window, segment_ids=documents,
                              max_segments=1)
        assert kernel_refusals(ATTENTION, "flash", case) == []
        found = pulled(flash, query, key, value)
    reference = pulled(plain, query, key, value)
    with jax.enable_x64(new_val=True):
        exact = pulled(truth, *(jnp.asarray(np.asarray(x, np.float64)) for x in (query, key, value)))
    # Each row sized by the float64 terms of its sums as the suite's flash gradients are
    # (`attention_terms`): a query that keeps one key has a gradient that cancels to zero, which a
    # flash kernel forms from its rounded output.
    terms = attention_terms(query, key, value, cotangent, mask)
    for name, mine, theirs, true, summed in zip(("out", "query", "key", "value"), found, reference, exact,
                                                terms, strict=True):
        assert_as_exact_as_the_reference(np.asarray(mine, np.float64), np.asarray(theirs, np.float64),
                                         np.asarray(true), f"window {window} {name}",
                                         terms=np.asarray(summed))


@pytest.mark.parametrize("width", [128, 96])
@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_bf16_decode_step_routes_to_the_kernel_its_generation_measured_fastest(generation, width,
                                                                                without_deterministic_ops):
    """A bf16 decode step of 16 query heads over 8 key heads, traced through the public attention for a
    GPU: an A100 runs the decode kernel its row prefers (`generation.TARGETS`' sm80 'decode', from the DCU
    receipt), and hands a width it does not take (96) on to cuDNN; an H100 runs cuDNN, its default. Out
    of the cuda lane's deterministic ops, which refuse cuDNN (`without_deterministic_ops`)."""
    from dew.nn.kernels.generation import assumed

    query = jnp.zeros((2, 1, 16, width), jnp.bfloat16)
    key = jnp.zeros((2, 256, 8, width), jnp.bfloat16)
    lengths = jnp.asarray([3, 200], jnp.int32)
    with assumed(generation), generation_routes() as found:
        jax.make_jaxpr(lambda q, k, v: scaled_dot_product_attention(q, k, v, key_value_seq_lengths=lengths))(
            query, key, key)
    [route] = [route for route in found if route.choice == "decode"]
    preferred = "decode" if generation == "sm80" else "cudnn"
    assert (route.preferred, route.kernel) == (preferred, preferred if width == 128 else "cudnn")
    assert (route.refusal is None) == (route.kernel == preferred)
