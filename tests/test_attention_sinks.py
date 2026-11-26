"""GPT OSS eager-attention parity from tools/gpt_oss_reference.py."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import attention_on_par, kept
from contract_support import normal_drawn
from reference_error import FEWEST_SEEDS, assert_computes_the_oracle, chain_roundings

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels.templates.attention import dropout_keep, dropout_seed

FIXTURE = Path(__file__).parent / "fixtures" / "gpt_oss" / "attention.npz"


def test_xla_sink_attention_keeps_causal_window_and_gradients():
    """GPT-OSS's sinks under a causal window of three keys on the xla kernel, at the shapes of GPT-OSS's
    recorded eager attention (tools/gpt_oss_reference.py's fixture): over fresh draws its output and every
    gradient, the sinks' among them, on par with the reference path's from float64 (check 1,
    `attention_on_par`); the reference path's formula is GPT-OSS's to float64's own rounding
    (`test_sinks_a_softcap_and_dropout_compose_on_one_reference_path`)."""
    with np.load(FIXTURE) as fixture:
        shapes = [fixture[name].shape for name in ("query", "key", "value", "sinks", "output")]

    def attend(implementation):
        def run(query, key, value, sinks):
            return scaled_dot_product_attention(query, key, value, sinks=sinks,
                                                implementation=implementation, causal=True, sliding_window=3)
        return run

    queries, keys = shapes[0][1], shapes[1][1]
    attention_on_par(attend("xla"), attend("reference"), normal_drawn(0x515, *shapes),
                     "xla sinks in a window", keep=kept(queries, keys, causal=True, sliding_window=3),
                     seeds=FEWEST_SEEDS)


@pytest.mark.parametrize("dropout_rate", [0.0, 0.25])
def test_sinks_a_softcap_and_dropout_compose_on_one_reference_path(dropout_rate):
    """Sinks, Gemma 2's softcap and probability dropout ride one score
    pipeline (`reference_attention`): run in float64, a grouped causal call
    with all three, training or not, computes the float64 oracle (the scaled
    logits capped, the causal mask, a softmax a sink joins, the weights the
    draw keeps over the keep rate, the value product) within float64
    rounding."""
    with jax.enable_x64(new_val=True):
        keys = jax.random.split(jax.random.key(0), 4)
        query = jax.random.normal(keys[0], (2, 6, 4, 8), jnp.float64)
        key, value = (jax.random.normal(drawn, (2, 6, 2, 8), jnp.float64) for drawn in keys[1:3])
        sinks = jax.random.normal(keys[3], (4,), jnp.float64)
        rng = jax.random.key(1)

        def attend(query, key, value, sinks):
            return scaled_dot_product_attention(
                query, key, value, sinks=sinks, softcap=2.0, causal=True, implementation="xla",
                dropout_rate=dropout_rate, dropout_rng=rng, deterministic=False)

        group = jnp.arange(4) // 2
        raw = jnp.einsum("bqhd,bkhd->bhqk", query, key[:, :, group]) / np.sqrt(8)
        logits = jnp.where(jnp.tril(jnp.ones((6, 6), bool)), 2.0 * jnp.tanh(raw / 2.0), -jnp.inf)
        sink = jnp.broadcast_to(sinks[None, :, None, None], (2, 4, 6, 1))
        weights = jax.nn.softmax(jnp.concatenate([logits, sink], -1), axis=-1)[..., :-1]
        if dropout_rate:
            keep = dropout_keep(dropout_seed(rng), dropout_rate, weights.shape)
            scale = jnp.asarray(1.0 / (1.0 - dropout_rate), weights.dtype)
            weights = weights * jnp.where(keep, scale, 0)
        truth = jnp.einsum("bhqk,bkhd->bqhd", weights, value[:, :, group])
        assert_computes_the_oracle(
            np.asarray(attend(query, key, value, sinks)), np.asarray(truth), "sinks, softcap and dropout",
            roundings=chain_roundings(jax.make_jaxpr(attend)(query, key, value, sinks)))


@pytest.mark.parametrize("implementation", ["cudnn", "tpu"])
def test_fused_attention_refuses_sinks_by_name(implementation):
    heads = jnp.ones((1, 2, 2, 8))
    with pytest.raises(ValueError):
        scaled_dot_product_attention(heads, heads, heads,
                                     implementation=implementation, sinks=jnp.zeros(2))


def _scores_whole(attend, *operands) -> bool:
    """Whether `attend`'s program forms a whole score array of its operands' every row, head, query and key
    (`[..., Q, K]` of `B H Q K` entries) in the memory its kernels' launches read and write, at any depth
    but a kernel's own, its launch's outputs among them (`reference_error.equations` pruning
    `programs.opaque`): the reference path's, which a streamed kernel never holds; a key mask, `[B, 1, Q,
    K]`, holds no head's."""
    from reference_error import equations

    from dew.nn import programs

    rows, queries, heads, _ = operands[0].shape
    keys = operands[1].shape[1]

    def whole(shape) -> bool:
        return tuple(shape[-2:]) == (queries, keys) and np.prod(shape) >= rows * heads * queries * keys

    return any(whole(getattr(var.aval, "shape", ())) for equation in equations(
        jax.make_jaxpr(attend)(*operands).jaxpr, prune=programs.opaque) for var in equation.outvars)


def test_the_whole_scores_witness_reads_a_kernels_launch_and_not_its_body():
    """`_scores_whole` finds a whole score array a kernel's launch writes, as a native launch's outputs are
    memory the program holds (`reference_error.equations` pruning `programs.opaque`), and none a kernel
    forms within it: skipping the launch had let one writing `[B, H, Q, K]` pass (train 1320's review)."""
    from jax.experimental import pallas as pl

    query = jnp.zeros((1, 64, 2, 16), jnp.float32)

    def writes(q, k, v):
        def kernel(o_ref):
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((1, 2, 64, 64), jnp.float32))()

    def within(q, k, v):
        def kernel(o_ref):
            scores = jnp.zeros((1, 2, 64, 64), jnp.float32)
            o_ref[...] = jnp.sum(scores, axis=-1)

        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((1, 2, 64), jnp.float32))()

    assert _scores_whole(writes, query, query, query)
    assert not _scores_whole(within, query, query, query)


def test_a_querys_own_sinks_stream_where_the_kernels_taking_a_heads_alone_refuse_them():
    """Each query's own sinks (`[B, S, H]`, projected) on a TPU's causal bf16 call and an H100's decode step:
    splash, which takes a head's (`[H]`), refuses them by name, its refusals naming the requirement, and the
    call left to its selectors streams them, forming no whole `[B, H, Q, K]` score array as the reference
    path does. The
    route was overridden after selection, so a published route named splash while the template ran, and
    then left to the reference path's whole scores (row 13108)."""
    from dew.nn.attention import scaled_dot_product_attention
    from dew.nn.kernels import generation
    from dew.nn.kernels.selection import KernelRefused

    def attend(sinks, **options):
        return lambda q, k, v: scaled_dot_product_attention(q, k, v, sinks=sinks, **options)

    query = jnp.zeros((1, 2048, 4, 128), jnp.bfloat16)
    sinks = jnp.zeros((1, 2048, 4), jnp.float32)
    with generation.assumed("v5e"):
        with pytest.raises(KernelRefused) as refused:
            attend(sinks, causal=True, implementation="tpu")(query, query, query)
        assert "query_sinks" in {refusal.code for refusal in refused.value.refusals}
        assert not _scores_whole(attend(sinks, causal=True), query, query, query)
        assert _scores_whole(attend(sinks, causal=True, implementation="reference"), query, query, query)
    step, cache = jnp.zeros((2, 1, 4, 128), jnp.bfloat16), jnp.zeros((2, 4096, 4, 128), jnp.bfloat16)
    lengths, own = jnp.asarray([3000, 2000], jnp.int32), jnp.zeros((2, 1, 4), jnp.float32)
    with generation.assumed("sm90"):
        assert not _scores_whole(attend(own, key_value_seq_lengths=lengths), step, cache, cache)
        assert _scores_whole(attend(own, key_value_seq_lengths=lengths, implementation="reference"),
                             step, cache, cache)


def test_a_masked_call_with_a_heads_sinks_picks_its_kernel():
    """A call that built its mask, naming that it has sinks by a flag (`mask_kernel`'s `sinks`, a sliding
    layer's of GPT-OSS), picks a kernel that takes them: the flag no query's own sinks, where reading it
    as an array raised AttributeError in every model with sinks (train 1313's batch-12)."""
    from dew.nn.attention import mask_kernel
    from dew.nn.kernels import generation

    query = jnp.zeros((1, 4, 8, 64), jnp.bfloat16)
    key = jnp.zeros((1, 64, 2, 64), jnp.bfloat16)
    with generation.assumed("sm90"):
        # cuDNN, which takes a mask as its bias, takes no sinks.
        assert mask_kernel("auto", query, key, as_bias=True, sinks=True) != "cudnn"


def test_a_decode_step_under_accurate_numerics_streams_on_the_template():
    """A bf16 decode step with a head's sinks on an H100 under a run's accurate numerics (`Numerics(forms=
    "accurate")`): the decode kernel and cuDNN, which compute the fast form alone, refuse it when named, and
    the call left to its selector forms no whole score array (`_scores_whole`), the template's decode
    streaming it ahead of the reference path; the decode kernel ran it in its fast form (the 2026-11-26
    audit's decode finding)."""
    from dew.nn.attention import scaled_dot_product_attention
    from dew.nn.kernels import generation
    from dew.nn.kernels.selection import KernelRefused
    from dew.nn.precision import Numerics

    def attend(**options):
        sinks = jnp.zeros((4,), jnp.float32)
        return lambda q, k, v: scaled_dot_product_attention(
            q, k, v, sinks=sinks, key_value_seq_lengths=lengths, **options)

    step, cache = jnp.zeros((2, 1, 4, 128), jnp.bfloat16), jnp.zeros((2, 4096, 4, 128), jnp.bfloat16)
    lengths = jnp.asarray([3000, 2000], jnp.int32)
    with generation.assumed("sm90"), generation.run_precision(Numerics(forms="accurate")):
        for named in ("decode", "cudnn"):
            with generation.chosen(decode=named), pytest.raises(KernelRefused) as refused:
                jax.eval_shape(attend(), step, cache, cache)
            assert "form" in {refusal.code for refusal in refused.value.refusals}
        assert not _scores_whole(attend(), step, cache, cache)


@pytest.mark.parametrize("heads, kv_heads", [(4, 1), (4, 2)], ids=["shared", "grouped"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_a_masked_key_weighs_nothing_beside_a_sink_however_low_its_logit(heads, kv_heads, dtype):
    """Beside a sink a masked key is out of the softmax: a row whose every key is masked attends the
    sink alone and reads nothing, even where the sink's logit sits at or below the dtype's most
    negative finite value, where a finite floor let every masked key share the row."""
    from dew.nn.attention import reference_attention

    query = jax.random.normal(jax.random.key(0), (1, 3, heads, 8), dtype)
    key, value = (jax.random.normal(jax.random.key(seed), (1, 5, kv_heads, 8), dtype) for seed in (1, 2))
    mask = jnp.zeros((1, 1, 3, 5), bool).at[..., :2, :].set(True)
    # At float32's most negative finite logit, or below float16's, where a finite floor ties the sink.
    sinks = jnp.full((heads,), float(jnp.finfo(dtype).min) * (2 if dtype == jnp.float16 else 1), jnp.float32)
    out = reference_attention(query, key, value, mask=mask, sinks=sinks)
    assert out.dtype == dtype
    np.testing.assert_array_equal(np.asarray(out[:, 2]), 0)
    assert bool(jnp.all(jnp.isfinite(out)))

