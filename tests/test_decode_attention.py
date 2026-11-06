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
from arm_arithmetic import float64_held
from attention_oracle import attention_on_par, exact_attention, float64_weights
from contract_support import normal_drawn
from model_support import as_stepped
from reference_error import FEWEST_SEEDS, as_given, assert_as_exact, assert_fp32_reduction_bound

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.attention_masks import with_key_lengths
from dew.nn.decode_geometry import folded_attention
from dew.nn.kernels import decode_attention
from dew.nn.kernels.generation import chosen
from dew.nn.precision import FP32_SUMS

pytestmark = pytest.mark.kernel


FOLDED_SEEDS = 8
"""Folded attention's declared check-1 seeds in bf16, sized by `seeds_for` paired."""


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("heads", [1, 2])
def test_folded_attention_is_jax_nn_attention(dtype, heads):
    """Every query head against every (key, head) pair, the other heads'
    products thrown away, computes jax.nn's xla attention. In fp32 it only
    reorders three sums, the logits' over the width, the softmax's over the
    keys and the value product's over keys times heads (exact zeros between),
    so it sits within their summed chains of jax.nn's
    (`assert_fp32_reduction_bound`), the logits' rounding carried through the
    softmax at twice the largest logit magnitude. In bf16 it is on par with
    jax.nn over drawn inputs (check 1, the truth jax.nn's float64 run). Rows
    read 1, 17 and all 40 keys."""
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
    if dtype == jnp.float32:
        spread = np.where(read, np.einsum('btnd,bsnd->bnts', np.abs(q), np.abs(k)) / 16.0, 0).max(-1)
        carried = 1 + 2 * spread.transpose(0, 2, 1)[..., None]
        magnitudes = np.einsum('bnts,bsnd->btnd', probs, np.abs(v)) * carried
        assert_fp32_reduction_bound(folded, plain, magnitudes, terms=256 + 40 + 40 * heads)
        return

    def draw(seed):
        rng = np.random.default_rng([0xF01, seed])
        query = rng.normal(size=(3, 4, heads, 256))
        key, value = rng.normal(size=(2, 3, 40, heads, 256)) * np.array([0.3, 1.0])[:, None, None, None, None]
        return tuple(jnp.asarray(x, dtype) for x in (query, key, value))

    # The truth attention written out over each row's keys: raw jax.nn takes its softmax in fp32 whatever
    # its operands (src/dew/nn/attention.py).
    reads = np.arange(40)[None, None, None, :] < np.asarray(lengths)[:, None, None, None]
    assert_as_exact(lambda q, k, v: folded_attention(q, k, v, with_key_lengths(None, lengths, 40)),
                    lambda q, k, v: jax.nn.dot_product_attention(q, k, v, key_value_seq_lengths=lengths,
                                                                 implementation='xla'),
                    draw, "folded", seeds=FOLDED_SEEDS, terms=True,
                    truth_fn=lambda q, k, v: exact_attention(q, k, v, keep=reads))


ON_GPU = jax.default_backend() == "gpu"
"""Whether the kernels compile for this host's GPU; elsewhere they run interpreted, sized as on an
A100 (`on_the_gpu`)."""


def on_the_gpu():
    """The host's own GPU, or elsewhere an A100's resources, which size the kernels' launches."""
    from dew.nn.kernels.generation import assumed

    return contextlib.nullcontext() if ON_GPU else assumed("sm80")


def dense_operands(dtype, windowed=False, seed: int = 0) -> tuple:
    """Rows reading 1 key (an idle slot), a block's worth, one past it, 77 keys and the whole capacity
    of 32 blocks, at Qwen3.5-0.8B's widths (4 query positions over 2 key heads of 256): the queries, keys,
    values and lengths, and with `windowed` the rows' starts within a block, at one, mid-row and a window of
    300 keys before a full row's end, as a sliding window's decode reads them; their values drawn by
    `seed`, the rows' lengths and starts the same at every seed."""
    rng = np.random.default_rng(seed)
    capacity = 32 * decode_attention.BLOCK
    lengths = jnp.asarray([1, decode_attention.BLOCK, decode_attention.BLOCK + 1, 77, capacity], jnp.int32)
    rows = lengths.shape[0]
    query = jnp.asarray(rng.normal(size=(rows, 4, 2, 256)), dtype)
    key = jnp.asarray(rng.normal(size=(rows, capacity, 2, 256)) * 0.3, dtype)
    value = jnp.asarray(rng.normal(size=(rows, capacity, 2, 256)), dtype)
    starts = jnp.asarray([0, 3, 1, 40, capacity - 300] if windowed else [0] * rows, jnp.int32)
    return query, key, value, lengths, *((starts,) if windowed else ())


def spanned(starts, lengths, capacity):
    """Each row's keys from its start to its length, `[rows, 1, 1, capacity]`."""
    keys = np.arange(capacity)[None, None, None, :]
    first, last = (np.asarray(x)[:, None, None, None] for x in (starts, lengths))
    return (keys >= first) & (keys < last)


DECODE_KERNEL_SEEDS = 8
"""The decode kernel's declared check-1 seeds a config, sized by `seeds_for` paired."""


@pytest.mark.parametrize("windowed", [False, True], ids=["prefix", "window"])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_the_decode_kernel_reads_each_row_to_its_length_at_every_config(dtype, windowed):
    """`dew.nn.kernels.decode_attention` reads each row's keys up to its length, from its start where
    a window gives one, block by block, at each slice count it may take (`SPLITS`, `tuning.pinned`:
    one slice a row, or more whose fp32 sums XLA combines), on par with jax.nn's xla attention under the
    same keys' mask over drawn inputs (check 1, the truth its float64 run). Compiled
    on a GPU at the default precision, which the GPU lane otherwise sets to the highest
    (tests/lane_environment.py), its fp32 products must be fp32 ones, held to jax.nn's at fp32
    precision; Triton's default is one TF32 product. Elsewhere it runs interpreted."""
    from dew.nn.kernels import decode_attention, tuning

    def draw(seed):
        return dense_operands(dtype, windowed, seed)[:3]

    lengths = dense_operands(dtype, windowed)[3]
    starts = dense_operands(dtype, windowed)[4] if windowed else jnp.zeros_like(lengths)
    mask = jnp.asarray(spanned(starts, lengths, draw(0)[1].shape[1]))
    windowing = (starts,) if windowed else ()

    def plain(query, key, value):
        with jax.default_matmul_precision("float32"):
            return jax.nn.dot_product_attention(query, key, value, mask=mask, implementation='xla')

    for config in map(decode_attention.Split, decode_attention.SPLITS):
        def attend(query, key, value, config=config):
            pin = tuning.pinned({decode_attention.DecodeAttention: config})
            with on_the_gpu(), pin, jax.default_matmul_precision("default"):
                return decode_attention.attend(query, key, value, lengths, *windowing, interpret=not ON_GPU)

        out = jax.jit(attend)(*draw(0))
        assert out.shape == jax.eval_shape(plain, *draw(0)).shape and out.dtype == draw(0)[0].dtype
        assert_as_exact(attend, plain, draw, f"decode kernel, {config}", seeds=DECODE_KERNEL_SEEDS,
                        terms=True, truth_fn=lambda q, k, v: exact_attention(q, k, v, keep=np.asarray(mask)))


SINKS = (0.5, -1.0, 2.0, -3.0)
"""Four query heads' sink logits, around and past the rows' logits' scale."""
SOFTCAP = 0.5
"""A softcap near the scaled logits' spread, where `tanh` bends them."""


def grouped_case(dtype):
    """`dense_operands`' keys, lengths and capacity under 2 query positions of 4 query heads, a key head's
    pair reading it as gpt-oss's 64 read their 8."""
    _, key, value, lengths = dense_operands(dtype)
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


DECODE_HOOKS = {"sinks": {"sinks": True}, "softcap": {"softcap": True},
                "both": {"sinks": True, "softcap": True}}
"""The hooked decode cases' hooks by name: sinks, a softcap, both."""
SINK_CASES = [(dtype, hooks, natural, splits) for dtype in ("bfloat16", "float32") for hooks in DECODE_HOOKS
              for natural in ("fold", "natural") for splits in decode_attention.SPLITS]
"""The hooked decode cases, `(dtype, hooks, form, slice count)`, by name."""
LONG_SINK_CASES = [("float32", "both", "natural", 32), ("float32", "both", "fold", 32)]
"""The hooked cases past two minutes in one process (137 and 126 s on armada in a verdict on main 495ef4e0e),
their check 1's draws run as armada's map tasks and a judge (`check1_shards`), each within two minutes."""


@pytest.mark.parametrize(("dtype", "hooks", "natural", "splits"),
                         [case for case in SINK_CASES if case not in LONG_SINK_CASES],
                         ids=["-".join(map(str, case)) for case in SINK_CASES if case not in LONG_SINK_CASES])
def test_the_decode_kernel_applies_sinks_and_a_softcap_at_every_config(dtype, hooks, natural, splits):
    """Sinks (one value-free logit a query head in each row's softmax) and a logit softcap
    (`cap · tanh(logit / cap)`) inside the decode kernel, over query heads grouped on their key heads:
    each row read to its length at each slice count it may take, the sink carried by the first slice alone
    so the combine counts it once, within tests/reference_error.py's rule of the reference path's
    attention over the same keys, against float64 (`exact_attention`). The idle row of one key keeps
    its sink's share. The dispatch takes such a call to the decode kernel where it runs (`decodes`). In
    both forms of the exponent, the fold and the accurate one (`natural`); each slice count held to
    check 1 over fresh draws of the case's shapes (`grouped_draw`), a slice count a case (row 2195's CI: the
    six together took 153 s of a CPU task)."""
    _sinks_and_softcap_hold_check_1(dtype, hooks, natural, splits)


@pytest.mark.parametrize(("dtype", "hooks", "natural", "splits"), LONG_SINK_CASES,
                         ids=["-".join(map(str, case)) for case in LONG_SINK_CASES])
@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_the_decode_kernel_applies_sinks_and_a_softcap_at_its_long_configs(dtype, hooks, natural, splits):
    """`test_the_decode_kernel_applies_sinks_and_a_softcap_at_every_config` at its long cases
    (`LONG_SINK_CASES`), their draws in map tasks of 4."""
    _sinks_and_softcap_hold_check_1(dtype, hooks, natural, splits)


def _sinks_and_softcap_hold_check_1(dtype, hooks, natural, splits):
    """A hooked decode case's check 1
    (`test_the_decode_kernel_applies_sinks_and_a_softcap_at_every_config`)."""
    from dew.nn.attention import reference_attention
    from dew.nn.kernels import tuning

    dtype, hooks, natural = jnp.dtype(dtype), DECODE_HOOKS[hooks], natural == "natural"
    arrays = grouped_case(dtype)
    query, key, value, lengths = arrays
    sinks = jnp.asarray(SINKS, jnp.float32) if hooks.get("sinks") else None
    softcap = SOFTCAP if hooks.get("softcap") else None
    read = spanned(jnp.zeros_like(lengths), lengths, key.shape[1])

    def reference(query, key, value):
        # The decode kernel interprets on this CPU what a GPU computes: its reference sums as the target
        # does (fp32), not as the CPU's policy does.
        with jax.default_matmul_precision(FP32_SUMS):
            return reference_attention(query, key, value, mask=jnp.asarray(read), sinks=sinks,
                                       softcap=softcap)

    def truth(query, key, value):
        wide = None if sinks is None else jnp.asarray(np.asarray(sinks, np.float64))
        return exact_attention(query, key, value, wide, softcap=softcap, mask=read)

    def attend(query, key, value, lengths):
        return decode_attention.attend(query, key, value, lengths, softcap=softcap, sinks=sinks,
                                       natural=natural, interpret=not ON_GPU)

    config = decode_attention.Split(splits)

    def pinned(query, key, value):
        pin = tuning.pinned({decode_attention.DecodeAttention: config})
        with pin, jax.default_matmul_precision("default"):
            return attend(query, key, value, lengths)

    with on_the_gpu():
        out = jax.jit(pinned)(query, key, value)
        assert out.shape == reference(query, key, value).shape and out.dtype == query.dtype
        assert_as_exact(jax.jit(pinned), jax.jit(reference), grouped_draw(dtype),
                        f"decode kernel with {sorted(hooks)}, natural {natural}, {config}",
                        seeds=FEWEST_SEEDS, truth_fn=truth)


SINK_SEEDS = 256
"""The sinks' gradient and tangent site's declared draws: `seeds_for` of its reference over this draw gives
256 on an H100, either cap."""


def a_decode_steps_sinks_take_their_gradient_and_tangent_through_the_dispatch(softcap):
    """A grouped decode step with sinks through Dew's attention (`scaled_dot_product_attention`), the sinks
    an operand of its derivative: on a GPU its gradient and tangent with respect to them are on par with the
    reference path's (check 1, `assert_as_exact`), over draws of the step's queries, keys, values and sinks
    at its lengths, whatever kernel runs them."""
    query, key, value, lengths = grouped_case(jnp.float32)

    def derivatives(implementation, precision):
        def attend(query, key, value, sinks):
            return scaled_dot_product_attention(
                query, key, value, sinks=sinks, softcap=softcap, key_value_seq_lengths=lengths,
                implementation=implementation)

        def both(query, key, value, sinks):
            with jax.default_matmul_precision(precision):
                grad = jax.grad(lambda sinks: jnp.sum(attend(query, key, value, sinks) ** 2))(sinks)
                tangent = jax.jvp(lambda sinks: attend(query, key, value, sinks), (sinks,),
                                  (jnp.ones_like(sinks),))[1]
            return grad, tangent
        return both

    sinks = jnp.asarray(SINKS, jnp.float32)

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 4)
        return (jax.random.normal(streams[0], query.shape, jnp.float32),
                jax.random.normal(streams[1], key.shape, jnp.float32) * 0.3,
                jax.random.normal(streams[2], value.shape, jnp.float32),
                jax.random.normal(streams[3], sinks.shape, jnp.float32) * 2.0)

    assert_as_exact(jax.jit(derivatives("auto", "default")), jax.jit(derivatives("reference", "float32")),
                    draw, f"the sinks' derivatives through the decode kernel, softcap {softcap}",
                    seeds=SINK_SEEDS)


def decode_step(lengths):
    """fp32 attention of a decode step's grouped query (`decode_geometry.grouped_query`) over key
    lengths through the decode kernel, chosen for the decode selector (`attention_kernels.DECODE`),
    which takes it on a GPU (`decode_attention.SUPPORTS`), and the folded attention it stands in for
    over the prefix those lengths mean."""
    def attended(query, key, value):
        with chosen(decode="decode"):
            return scaled_dot_product_attention(query, key, value, key_value_seq_lengths=lengths)

    def folded(query, key, value):
        return folded_attention(query, key, value, with_key_lengths(None, lengths, key.shape[-3]))

    return attended, folded


def test_a_decode_step_traced_with_x64_on_runs_the_decode_kernel():
    """A float32 decode step traced with x64 on, as a float64 truth around a served model traces it,
    lowers for the GPU on this CPU through the decode kernel: its block and slice counts take the span's
    integer type, not the int64 a Python int becomes."""
    from dew.nn.kernels.generation import assumed, gpu_mesh

    with jax.enable_x64(new_val=True):
        attended, _ = decode_step(jnp.asarray([1, 20], jnp.int32))
        query = jax.ShapeDtypeStruct((2, 4, 2, 64), jnp.float32)
        key = jax.ShapeDtypeStruct((2, 64, 2, 64), jnp.float32)
        mesh = gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
        with assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
            jax.jit(attended).trace(query, key, key).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_attention_the_decode_kernel_takes_lowers_for_the_gpu_differentiated_in_both_modes(generation):
    """Lowered for the GPU on this CPU, a decode step's attention runs the decode kernel for its value
    under reverse and forward mode alike, its Pallas call having no derivative of its own."""
    from dew.nn.kernels.generation import assumed, gpu_mesh

    attended, _ = decode_step(jnp.asarray([1, 20], jnp.int32))
    query = jax.ShapeDtypeStruct((2, 4, 2, 64), jnp.float32)
    key = jax.ShapeDtypeStruct((2, 64, 2, 64), jnp.float32)

    def loss(*arrays):
        return jnp.square(attended(*arrays)).sum()

    mesh = gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), generation)
    for differentiated in (jax.value_and_grad(loss, argnums=(0, 1, 2)),
                           lambda *arrays: jax.jvp(loss, arrays, arrays)):
        with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
            jax.jit(differentiated).trace(query, key, key).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_grouped_query_in_its_own_layout_reaches_the_decode_kernel(generation):
    """A decode step's query of 4 heads over 2 key heads, called in the layer's own `[B, 1, N, D]`
    layout with key lengths: the core reads each key head's query heads as its query positions
    (`decode_geometry.grouped_query`), so the decode kernel takes it, lowered for the GPU on this CPU."""
    from dew.nn.kernels.generation import assumed, gpu_mesh

    query = jax.ShapeDtypeStruct((2, 1, 4, 64), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((2, 64, 2, 64), jnp.bfloat16)

    mesh = gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(decode_step(jnp.asarray([1, 20], jnp.int32))[0]).trace(query, key, key).lower(
            lowering_platforms=("cuda",))


GROUPED_SEEDS = 8
"""A grouped lone query's declared check-1 seeds, sized by `seeds_for` paired."""


@pytest.mark.parametrize("implementation", ["xla", "reference"])
def test_a_grouped_query_attends_as_its_heads_do_alone(implementation):
    """Grouped in the core, a lone query of 8 heads over 2 key heads with key lengths gives each head's
    attention over its key head's keys, on par with the reference over the key heads repeated over
    drawn inputs (check 1, the truth its float64 run)."""
    from dew.nn.attention import reference_attention
    from dew.nn.attention_masks import repeat_kv_heads

    lengths = jnp.asarray([1, 9, 32], jnp.int32)

    draw = normal_drawn(3, (3, 1, 8, 32), (3, 32, 2, 32), (3, 32, 2, 32))

    def alone(query, key, value):
        mask = (jnp.arange(key.shape[1]) < lengths[:, None])[:, None, None, :]
        return reference_attention(query, repeat_kv_heads(key, 8), repeat_kv_heads(value, 8), mask=mask)

    assert_as_exact(lambda q, k, v: scaled_dot_product_attention(q, k, v, implementation=implementation,
                                                                 key_value_seq_lengths=lengths),
                    alone, draw, f"grouped {implementation}", seeds=GROUPED_SEEDS)


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_windowed_layers_decode_step_reads_its_window_through_the_decode_kernel(generation):
    """A causal transformer whose layers slide a window of 16 keys over a 64-slot cache, its decode
    step lowered for the GPU on this CPU: each layer reads its window through the decode kernel, from
    each row's first key under it, where a mask had sent the step to jax.nn's xla attention."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kernels.generation import assumed, gpu_mesh

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
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh), chosen(decode="decode"):
        jax.jit(step).trace(variables, cache, tokens).lower(lowering_platforms=("cuda",))


WINDOWED_DECODE_SEEDS = 8
"""A windowed decode's declared check-1 seeds against its whole passes, sized by `seeds_for` paired."""


def test_a_windowed_decode_step_attends_as_its_whole_pass():
    """A windowed layer's decode, its keys from each row's first under a window of 5 over a 64-slot
    cache at widths the decode kernel takes (heads 16 wide), gives what its whole passes give at each
    token, each step's at the width it reads (`as_stepped`), on par with them over drawn tokens (check 1,
    the truth the float64 whole pass): the starts the decode reads are the window's keys. On a GPU
    the same steps run the decode kernel in every layer."""
    from dew.nn.backbones.causal_transformer import CausalTransformer

    model = CausalTransformer(vocab_size=64, num_layers=2, emb_features=64, num_heads=4, num_kv_heads=2,
                              max_seq_len=64, layer_types=("sliding_attention",) * 2,
                              kinds={"sliding_attention": {"window": 5}}, attention_impl="xla")
    variables = model.init(jax.random.key(0), jnp.zeros((2, 12), jnp.int32))
    forward = jax.jit(lambda held: model.apply(variables, held))
    cache = model.apply(variables, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]

    @jax.jit
    def step(cache, token):
        return model.apply({**variables, **cache}, token, decode=True, mutable=["cache"])

    @as_given
    def decoded(tokens):
        _, state = model.apply({**variables, "cache": cache}, tokens[:, :4], decode=True, mutable=["cache"])
        steps = []
        for index in range(4, 12):
            out, state = step(state, tokens[:, index:index + 1])
            steps.append(out)
        return np.asarray(jnp.concatenate(steps, axis=1), np.float64)

    def draw(seed):
        return (np.asarray(jax.random.randint(jax.random.key(seed), (2, 12), 0, 64), np.int32),)

    wide = model.clone(dtype=jnp.float64)
    stepped = as_given(lambda tokens: np.asarray(as_stepped(forward, tokens, 4)[:, 4:], np.float64))
    assert_as_exact(decoded, stepped, draw, "decode", seeds=WINDOWED_DECODE_SEEDS,
                    truth_fn=lambda tokens: wide.apply(float64_held(variables), tokens)[:, 4:])


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
    # The same one path differentiated: the gradients agree bitwise by the same construction.
    grads = [jax.grad(lambda *a, f=f: jnp.sum(f(*a) ** 2), argnums=(0, 1, 2))(query, key, value)
             for f in (spans, masked)]
    for mine, theirs in zip(*grads, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


STARTS_SEEDS = 8
"""Key starts under reference arithmetic: declared check-1 seeds, sized by `seeds_for` paired."""


@pytest.mark.parametrize("options", [{"force_fp32_for_softmax": False},
                                     {"precision": jax.lax.Precision.HIGHEST}], ids=["softmax", "precision"])
def test_key_starts_keep_the_arithmetic_auto_is_asked_for_where_the_decode_kernel_would_run(options):
    """A lone grouped query of decode-kernel widths over key starts and lengths alone, traced as on
    an A100, asking for arithmetic only the reference path performs: 'auto' takes the reference
    path, not the decode kernel (which refused the options by name), and is on par with the one-mask
    form over drawn inputs (check 1, the truth the reference path's float64 run); the grouped query
    sums in another order."""
    from dew.nn.kernels.generation import assumed

    lengths, starts = jnp.asarray([2, 40, 64], jnp.int32), jnp.asarray([1, 8, 0], jnp.int32)
    keys = jnp.arange(64)
    kept = ((keys >= starts[:, None]) & (keys < lengths[:, None]))[:, None, None, :]

    draw = normal_drawn(6, (3, 1, 4, 16), (3, 64, 2, 16), (3, 64, 2, 16))

    def by_starts(query, key, value):
        with assumed("sm80"):
            return scaled_dot_product_attention(query, key, value, key_value_seq_lengths=lengths,
                                                key_value_seq_starts=starts, **options)

    def by_mask(query, key, value):
        with assumed("sm80"):
            return scaled_dot_product_attention(query, key, value, mask=kept, **options)

    assert_as_exact(by_starts, by_mask, draw, "starts under reference arithmetic", seeds=STARTS_SEEDS,
                    truth_fn=lambda q, k, v: scaled_dot_product_attention(q, k, v, mask=kept,
                                                                          implementation="reference"))



@pytest.mark.parametrize(("dtype", "declared"), [
    ("float32", "default"), ("float32", "high"), ("float32", "BF16_BF16_F32"),
    ("bfloat16", "BF16_BF16_BF16")])
def test_a_decode_step_under_a_declared_precision_its_kernels_do_not_compute_is_on_par(dtype, declared):
    """A decode step whose run declares products (`jax_default_matmul_precision`, a run's
    `Numerics.matmul`) no fused decode kernel computes (fp32 operands under TF32's default, bf16_3x's
    `high` or one bf16 pass; 16-bit operands summed in bf16), traced as on an H100: it is served by a path
    that honors the declaration (`decode_attention.PRODUCTS`), on par with the reference path under the
    same declaration (check 1, the truth the reference path's float64 run)."""
    from dew.nn.kernels.generation import assumed

    lengths = jnp.asarray([3, 64], jnp.int32)
    draw = normal_drawn(9, (2, 1, 8, 64), (2, 64, 2, 64), (2, 64, 2, 64), dtype=jnp.dtype(dtype))

    def attended(implementation):
        def attend(query, key, value):
            with assumed("sm90"), jax.default_matmul_precision(declared):
                return scaled_dot_product_attention(query, key, value, key_value_seq_lengths=lengths,
                                                    implementation=implementation)
        return attend

    assert_as_exact(attended("auto"), attended("reference"), draw, f"{dtype} decode under {declared}",
                    seeds=STARTS_SEEDS)

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


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_decode_step_over_no_keys_returns_the_empty_result(generation):
    """A decode step of no cache capacity, its row's length 0 and a sink beside it, chosen for an A100 or an
    H100 (`generation.assumed`): the decode kernel's dense and paged layouts return the empty result, the
    reference's value product over no keys, zero, before any split config forms over no steps (audit K6:
    an IndexError on both GPUs)."""
    from dew.nn.kernels.generation import assumed

    query, key = jnp.ones((1, 1, 1, 64), jnp.bfloat16), jnp.zeros((1, 0, 1, 64), jnp.bfloat16)
    lengths, sinks = jnp.zeros((1,), jnp.int32), jnp.zeros((1,), jnp.bfloat16)
    want = scaled_dot_product_attention(query, key, key, sinks=sinks, implementation="reference")
    np.testing.assert_array_equal(np.asarray(want, np.float32), 0.0)
    pages = jnp.zeros((1, 1, 16, 64), jnp.bfloat16)
    with assumed(generation), chosen(decode="decode"):
        dense = scaled_dot_product_attention(query, key, key, key_value_seq_lengths=lengths, sinks=sinks)
        paged = decode_attention.attend_paged(query[:, 0], pages, pages, jnp.zeros((1, 0), jnp.int32),
                                              lengths, sinks=sinks)
    # Exact data: the result's own shape, every entry exactly zero.
    assert dense.shape == query.shape and paged.shape == query[:, 0].shape
    np.testing.assert_array_equal(np.asarray(dense, np.float32), 0.0)
    np.testing.assert_array_equal(np.asarray(paged, np.float32), 0.0)


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

    query = jax.ShapeDtypeStruct((2, 1, 8, 64), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((2, 256, 1, 64), jnp.bfloat16)

    mesh = gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:2]), ("tensor",)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(decode_step(jnp.asarray([1, 20], jnp.int32))[0]).trace(query, key, key).lower(
            lowering_platforms=("cuda",))


DECODE_DERIVATIVE_SEEDS = 8
"""The decode kernel's derivatives' declared check-1 seeds, sized by `seeds_for` paired on a GPU."""


def attention_the_decode_kernel_takes_differentiates_as_the_folded_attention():
    """A decode step's attention, its value the kernel's, differentiates as the folded attention it
    stands in for, in both modes: the value, query, key and value gradients, and the tangent, on par
    with the folded attention's over drawn inputs (check 1, the truth float64 attention's)."""
    lengths = jnp.asarray([1, 20, 64], jnp.int32)
    attended, folded = decode_step(lengths)

    draw = normal_drawn(1, (3, 4, 2, 64), (3, 64, 2, 64), (3, 64, 2, 64), (3, 4, 2, 64), (3, 64, 2, 64),
                        (3, 64, 2, 64), (3, 4, 2, 64))

    def derivatives(attend):
        def run(query, key, value, *rest):
            *tangents, cotangent = rest
            arrays = (query, key, value)
            out, tangent = jax.jvp(attend, arrays, tuple(tangents))
            gradients = jax.grad(lambda *a: jnp.sum(attend(*a) * cotangent), argnums=(0, 1, 2))(*arrays)
            return {"value": out, "gradients": gradients, "tangent": tangent}
        return run

    def exact(query, key, value):
        # Each row over its first `lengths` keys, every step in the operands' float64.
        return exact_attention(query, key, value,
                               keep=np.arange(key.shape[1]) < np.asarray(lengths)[:, None, None, None])

    assert_as_exact(derivatives(attended), derivatives(folded), draw, "the decode kernel's derivatives",
                    seeds=DECODE_DERIVATIVE_SEEDS, truth_fn=derivatives(exact))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axes, rows, heads", [({"data": 2}, 2, 4), ({"tensor": 2}, 4, 2)],
                         ids=["rows", "heads"])
def test_a_decode_step_launches_the_decode_kernel_on_each_shard_of_its_mesh(axes, rows, heads):
    """A decoder's single-token step over four rows of four heads, lowered for an A100 on a mesh that
    splits its rows or its heads in two, launches the decode kernel on each shard's own rows and
    heads (`manual_map`), the shapes its call names: a Pallas call has no partitioning rule, and
    launched on the whole batch it had XLA gather every row's cache to every device each token."""
    import math

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

    # The decode kernel named for the step (`generation.chosen`), whose launches' operands this reads.
    with assumed("sm80"), jax.sharding.use_abstract_mesh(mesh), chosen(decode="decode"):
        lowered = jax.jit(step).trace(placed(params, mesh), placed(cache, mesh),
                                      placed(tokens, mesh, "data" if "data" in axes else None)).lower(
            lowering_platforms=("cuda",))
    # Each decode launch's operands, read from the lowered program's custom calls: the rows' key spans and
    # the padded queries a shard's own (`custom_calls`, `triton_calls`, in program order).
    from dew.nn.kernels.lowered import custom_calls, triton_calls
    from dew.telemetry.instrumentation import custom_call_kind

    calls = [call for call in custom_calls(lowered) if custom_call_kind(call.target) == "triton"]
    launches = [call for call, kernel in zip(calls, triton_calls(lowered), strict=True)
                if kernel.name == "decode_attention"]
    assert launches
    for call in launches:
        spans, query = call.operands[:2]
        assert spans == (rows, 2) and call.operand_types[0] == "i32" and query[:3] == (rows, 16, heads), call


def paged_case(page_size, windowed=False):
    """Rows reading 1 key, a page, one past a block and every page of 32 blocks through tables
    shuffled and shared between rows, their query heads sharing key heads as Qwen3-0.6B's do (16 over
    8 of 128): the table, the lengths and the starts (`windowed` starts the rows within a page, past a
    block and a window of 300 keys before a full row's end), a seed's draw of the queries and the key
    and value pages, and the rows' keys and values gathered through the table."""
    rng = np.random.default_rng(0)
    per_row, kv_heads, heads, width = 32 * decode_attention.BLOCK // page_size, 8, 16, 128
    capacity = per_row * page_size
    lengths = jnp.asarray([1, page_size, decode_attention.BLOCK + 1, capacity], jnp.int32)
    rows = lengths.shape[0]
    count = rows * per_row + 3
    table = jnp.asarray(rng.permutation(count)[:rows * per_row].reshape(rows, per_row), jnp.int32)
    table = table.at[1, 0].set(table[0, 0])
    starts = jnp.asarray([0, 1, decode_attention.BLOCK + 1 - 2, capacity - 300] if windowed else [0] * rows,
                         jnp.int32)

    def draw(seed):
        streams = np.random.default_rng([0x9A6, seed])
        return (jnp.asarray(streams.normal(size=(rows, heads, width)), jnp.bfloat16),
                jnp.asarray(streams.normal(size=(kv_heads, count, page_size, width)) * 0.3, jnp.bfloat16),
                jnp.asarray(streams.normal(size=(kv_heads, count, page_size, width)), jnp.bfloat16))

    def gathered(pages):
        return jnp.moveaxis(pages[:, table].reshape(kv_heads, rows, capacity, width), 0, 2)

    return table, lengths, starts, draw, gathered


PAGED_SEEDS = 8
"""The paged decode kernel's declared check-1 seeds a config, sized by `seeds_for` paired."""


@pytest.mark.parametrize("splits", decode_attention.SPLITS)
@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
@pytest.mark.parametrize("windowed", [False, True], ids=["prefix", "window"])
@pytest.mark.parametrize("page_size", [16, 32])
def test_the_paged_decode_kernel_reads_each_row_through_its_table_at_every_config(page_size, windowed,
                                                                                  natural, splits):
    """`decode_attention.attend_paged` reads a row's pages through its table to the row's length, from
    its start where a window gives one, at each slice count it may take (`SPLITS`, a case each: the six
    together took up to 101 s of a CPU task), on par with jax.nn's xla attention over the gathered rows
    under the same keys' mask over drawn inputs (check 1, the truth its float64 run), compiled on a GPU
    and interpreted elsewhere, in both forms of the exponent (`natural`)."""
    from dew.nn.kernels import tuning
    from dew.nn.kv_cache import PAGED, PagedCase

    table, lengths, starts, draw, gathered = paged_case(page_size, windowed)
    query, key_pages, _ = draw(0)
    if ON_GPU:
        case = PagedCase(heads=query.shape[1], kv_heads=key_pages.shape[0], head_dim=query.shape[2],
                         value_dim=query.shape[2], dtype=query.dtype, pool=key_pages.dtype,
                         page_size=page_size, quantized=None, groups=1, head_shards=1, softcap=None,
                         arithmetic=False, named="auto", windowed=windowed)
        assert PAGED.kernel("triton").failed(case) == []
    config = decode_attention.Split(splits)

    def attend(query, key_pages, value_pages):
        with on_the_gpu(), tuning.pinned({decode_attention.PagedDecodeAttention: config}):
            return decode_attention.attend_paged(query, key_pages, value_pages, table, lengths,
                                                 *((starts,) if windowed else ()), natural=natural,
                                                 interpret=not ON_GPU)

    def plain(query, key_pages, value_pages):
        key, value = gathered(key_pages), gathered(value_pages)
        return jax.nn.dot_product_attention(query[:, None], key, value,
                                            mask=jnp.asarray(spanned(starts, lengths, key.shape[1])),
                                            implementation='xla')[:, 0]

    out = jax.jit(attend)(*draw(0))
    assert out.shape == jax.eval_shape(plain, *draw(0)).shape and out.dtype == query.dtype
    def exact(query, key_pages, value_pages):
        key, value = gathered(key_pages), gathered(value_pages)
        return exact_attention(query[:, None], key, value,
                               keep=np.asarray(spanned(starts, lengths, key.shape[1])))[:, 0]

    assert_as_exact(attend, plain, draw, f"paged decode kernel, natural {natural}, {config}",
                    seeds=PAGED_SEEDS, terms=True, truth_fn=exact)


COMPILED = [(kernel, variant, splits) for kernel, variants in
            (("dense", ("bfloat16", "float32")), ("natural", ("raw", "sinks")), ("paged", (16, 32)))
            for variant in variants for splits in decode_attention.SPLITS]
"""The decode compile gate's cases: each kernel's variants at every slice count (`decode_attention.SPLITS`),
a program shape each, so every case compiles one kernel (row 2123: a kernel's twelve together took 228 to
263 s of a CPU task)."""


@pytest.mark.parametrize(("kernel", "variant", "splits"), COMPILED)
@pytest.mark.parametrize("target", ["sm80", "sm90", "sm120"])
def test_the_decode_kernels_compile_on_tritons_compiler_within_each_gpus_block(target, kernel, variant,
                                                                              splits):
    """The dense kernel under a decode step's attention and its gradient, in bf16 and fp32 at
    Qwen3.5-0.8B's widths (4 query positions over 2 key heads of 256, a 1024-slot cache), and the
    paged kernel at Qwen3-0.6B's (16 query heads over 8 of 128) with pages of 16 and 32 keys over 1024
    keys a row, each at every slice count (`tuning.pinned`), which their 8 rows' 32 steps take on
    every GPU here, lowered for each GPU on this CPU and compiled by Triton
    within its block's shared memory; and the dense kernel in the accurate form (`natural`), raw without
    sinks and scaled with them, which the interpreted suite lowers for no GPU. A first filter: the A100
    runs them; the H100 and the RTX PRO 6000 have this check alone. One kernel at one slice count a case
    (`COMPILED`)."""
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

    built = {"dense": lambda: dense(jnp.dtype(variant)), "natural": lambda: natural(variant == "sinks"),
             "paged": lambda: paged(variant)}[kernel]
    op = decode_attention.PagedDecodeAttention if kernel == "paged" else decode_attention.DecodeAttention
    cases = [KernelCase(f"{kernel}/{variant}/{splits}", lambda: pinned(built, op, splits))]
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
    from dew.nn.attention_masks import causal_attention_mask

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
    the attention over the cache's mask on the kernels that build it, a padding row's output zero: by
    construction, those kernels building the documents' very mask (`cross_document_mask`, which
    `test_a_cache_calls_documents_aligned_to_their_last_keys_are_its_mask` holds equal to the cache's)
    before the call, so the two calls run one program over one mask."""
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


def flash_reads_a_cache_calls_documents_as_exactly_as_jax_nn_reads_its_mask(without_deterministic_ops,
                                                                                 window):
    """FlashAttention chosen for a cache call (`generation.chosen`) reads its documents as its packed
    sequences, the queries put in id order and back: its output and its query, key and value
    gradients on par with jax.nn's xla attention over the cache's mask over drawn inputs (check 1, the
    truth float64 attention over the mask), padding rows zero."""

    positions, filled, mask = cache_call(window)
    documents = cache_documents(positions, filled)
    real = jnp.asarray(positions >= 0)[:, :, None, None]

    def draw(seed):
        rng = np.random.default_rng([8, seed])
        q, k, v = (jnp.asarray(rng.normal(size=(4, length, 2, 64)), jnp.bfloat16) for length in (6, 16, 16))
        # Padding rows take no cotangent.
        return q, k, v, jnp.where(real, jnp.asarray(rng.normal(size=(4, 6, 2, 64)), jnp.bfloat16), 0)

    def flash(query, key, value):
        with chosen(attention="flash"):
            return scaled_dot_product_attention(query, key, value, implementation="flash", causal=True,
                                                sliding_window=window, segment_ids=documents, max_segments=1)

    def plain(query, key, value):
        attended = jax.nn.dot_product_attention(query, key, value, mask=mask, implementation="xla")
        return jnp.where(real, attended, 0)

    attention_on_par(flash, plain, draw, f"flash over a cache call's documents, window {window}",
                     keep=np.asarray(mask) & np.asarray(positions >= 0)[:, None, :, None])
