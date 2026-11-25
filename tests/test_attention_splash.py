"""The pallas splash kernel behind the attention kernel 'tpu': what it computes and
which calls it takes.

Splash is a TPU kernel, but pallas can run a Mosaic kernel under its own
interpreter, and the splash kernel works there forward and backward, so
`tpu_attention` asks for the interpreter off a TPU backend and every parity
assertion in this file runs on CPU. On a TPU the same file runs against
Mosaic instead, because the interpreter is exactly what `jax.default_backend()
!= 'tpu'` turns off, and the one `on_tpu` case adds a length the interpreter
is not worth running; `tools/qualify_splash.py` is that run plus a timing.

The reference is the 'reference' attention kernel, the einsum and softmax path,
run in fp32 at HIGHEST precision on the same inputs. Off a TPU fp32 holds to
a few parts in a million; bf16, and fp32 on a TPU, hold to two ulps of the
output scale, which is the distance the cudnn parity in tests/test_kernels.py
pins between two correct kernels.
"""

import functools

import attention_kernels_matrix as kernels_matrix
import decode_matrix
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import (
    exact_attention,
    exact_value_and_grads,
    float64_weights,
    kept,
    packed_ids,
    qkv,
    reference,
    value_and_grads,
)
from gpu_flow import cells_of, run_cells
from reference_error import (
    FEWEST_SEEDS,
    as_given,
    assert_as_exact,
    assert_computes_the_oracle,
    with_gradients,
)

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.documents import document_mask
from dew.nn.precision import FP32_SUMS

on_tpu = pytest.mark.tpu(reason="needs a TPU; the interpreter covers the rest here")

SPLASH_SEEDS = {"cross": 32, "sinks": 64, "sinks-window": 32}
"""The interpreted kernel's check-1 sites' declared seeds past the fewest (`FEWEST_SEEDS`), by oracle case:
every case sized by `seeds_for` on armada's CPUs (2026-11-14, size-splash at 314eef0fe73), the sinks'
cases with their sinks' gradient, Dew paired with the reference (2026-11-22, row 1569)."""


SHARDED_SEEDS = 8
"""The per-shard contract's declared draws (check 1, `sharded.assert_runs_per_shard`), sized by
`seeds_for`: the fewest for which one device compared with itself resolves the factor."""


def _on_par_on_a_tpu(q_shape, kv_shape, dtype, structure, label: str, *, sunk: bool = False, bias=None):
    """Splash compiled by Mosaic on a TPU, its output and every gradient (the sinks' included), held to check
    1 over fresh draws and cotangents against XLA's own attention of the same operands at the precision a TPU
    runs it (DEFAULT), the truth attention written out in float64 on the host's CPU (`float64_weights`, its
    derivative anchored; JAX_PLATFORMS=tpu,cpu), the rows a packed call's padding holds (`live`) read as
    zero by all three."""
    structure = dict(structure)
    live = structure.pop("live", None)
    softcap = structure.pop("softcap", None)
    keep = kept(q_shape[1], (kv_shape or q_shape)[1], **structure)
    group = np.arange(q_shape[2]) // (q_shape[2] // (kv_shape or q_shape)[2])

    def draw(seed):
        # Host arrays, which the gate widens to float64 on the host for the truth; the arms take them onto
        # the TPU as they run.
        query, _, _ = qkv(q_shape, dtype, seed=10 * seed)
        _, key, value = qkv(kv_shape or q_shape, dtype, seed=10 * seed + 1)
        sinks = 2.0 + jax.random.normal(jax.random.PRNGKey(10 * seed + 2), (q_shape[2],), jnp.float32)
        cotangent = jax.random.normal(jax.random.PRNGKey(10 * seed + 3), q_shape, dtype)
        return tuple(np.asarray(x) for x in (query, key, value, *((sinks,) if sunk else ()), cotangent))

    def read(out):
        return out if live is None else jnp.where(jnp.asarray(live)[:, :, None, None], out, 0)

    def attend(implementation, **options):
        def run(query, key, value, *sinks):
            return read(scaled_dot_product_attention(
                query, key, value, implementation=implementation, sinks=sinks[0] if sinks else None,
                softcap=softcap, **({} if bias is None else {"bias": bias}), **structure, **options))
        return with_gradients(run)

    @as_given
    def truth(*inputs):
        # On the host's CPU, which runs float64 a TPU does not: inputs, bias and kept rows brought there.
        host = jax.devices("cpu")[0]
        with jax.enable_x64(new_val=True), jax.default_device(host):
            wide_bias = None if bias is None else jnp.asarray(np.asarray(bias, np.float64))

            def exact(query, key, value, *sinks):
                weights, _ = float64_weights(query, key, keep, sinks[0] if sinks else None, softcap,
                                             anchored=True, bias=wide_bias)
                out = jnp.einsum("bhqk,bkhd->bqhd", weights, value[:, :, group])
                return out if live is None else jnp.where(np.asarray(live)[:, :, None, None], out, 0)

            wide = [jnp.asarray(np.asarray(x, np.float64)) for x in inputs]
            return [np.asarray(x) for x in jax.jit(with_gradients(exact))(*wide)]

    assert_as_exact(attend('tpu'), attend('reference', precision=jax.lax.Precision.DEFAULT), draw,
                    f"{label}, {jnp.dtype(dtype).name}", seeds=FEWEST_SEEDS, truth_fn=truth)


@on_tpu
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("call", ["causal", "softcap", "sinks", "packed_window"])
def test_the_mosaic_kernel_is_on_par_at_a_length_only_a_tpu_affords(dtype, call):
    """Splash at a decoder's training shape, 2048 keys and a 128-wide head over grouped keys, far past what
    the interpreter is worth running, for each argument the kernel takes (`_on_par_on_a_tpu`): Mosaic
    compiles its tiling, lane padding and block skipping for real."""
    structure: dict = {"causal": True}
    if call == "softcap":
        structure["softcap"] = 1.0
    elif call == "packed_window":
        segment_ids = packed_ids((700, 600, 500), (1000, 1000), length=2048)
        structure.update(sliding_window=256, segment_ids=segment_ids, live=segment_ids != 0)
    _on_par_on_a_tpu((2, 2048, 8, 128), (2, 2048, 2, 128), dtype, structure, f"splash at 2048 keys, {call}",
                     sunk=call == "sinks")


@on_tpu
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_the_flash_fallback_keeps_packed_documents_apart_by_their_segment_ids(dtype):
    """A bias keeps an explicit 'tpu' off splash, onto the flash kernel, which reads the segment ids rather
    than a [B, H, S, S] mask bias (`_on_par_on_a_tpu`)."""
    bias = 0.5 * jax.random.normal(jax.random.PRNGKey(5), (1, 4, 1024, 1024), jnp.float32)
    segment_ids = packed_ids((300, 500, 100), (1000,), length=1024)
    structure = {"causal": True, "segment_ids": segment_ids, "live": segment_ids != 0}
    _on_par_on_a_tpu((2, 1024, 4, 128), None, dtype, structure, "the flash fallback", bias=bias.astype(dtype))


ORACLE_CASES = {
    "full": ((2, 256, 4, 64), None, {}),
    "causal": ((2, 256, 4, 64), None, {"causal": True}),
    "window": ((2, 256, 4, 64), None, {"causal": True, "sliding_window": 64}),
    "bidirectional-window": ((2, 256, 4, 64), None, {"sliding_window": 64}),
    "grouped": ((2, 256, 8, 64), (2, 256, 2, 64), {"causal": True}),
    "cross": ((2, 256, 4, 64), (2, 512, 4, 64), {}),
    "softcap": ((2, 256, 8, 64), (2, 256, 2, 64), {"causal": True, "softcap": 1.0}),
    "sinks": ((2, 256, 8, 64), (2, 256, 2, 64), {"causal": True, "sinks": True}),
    "sinks-window": ((2, 256, 8, 64), (2, 256, 2, 64),
                     {"causal": True, "sliding_window": 128, "sinks": True}),
    "mask": ((2, 256, 4, 64), None, {"causal": True, "mask": True}),
    "packed-window": ((2, 256, 4, 64), None, {"causal": True, "sliding_window": 64, "segment_ids": True}),
}


def with_fp32_sums(function):
    """`function` run with its fp32 products summed in fp32 (`FP32_SUMS`), as an accelerator sums."""
    def run(*arguments, **keywords):
        with jax.default_matmul_precision(FP32_SUMS):
            return function(*arguments, **keywords)
    return run


def oracle_layout(case: str) -> tuple[tuple, tuple | None, dict, bool]:
    """`ORACLE_CASES[case]`'s query and key shapes, its structure materialized (a dense mask of packed
    documents; segment ids and the rows they keep live) and whether it takes sinks, which each site draws."""
    q_shape, kv_shape, structure = ORACLE_CASES[case]
    structure = dict(structure)
    sunk = structure.pop("sinks", False)
    if structure.pop("mask", False):
        structure["mask"] = np.asarray(document_mask(packed_ids((100, 84, 72))[:1]))[:, None]
    if structure.pop("segment_ids", False):
        structure["segment_ids"] = packed_ids((100, 84, 50), (30, 190, 10))
        structure["live"] = structure["segment_ids"] != 0
    return q_shape, kv_shape, structure, sunk


@pytest.mark.parametrize("case", ORACLE_CASES)
def test_splash_agrees_with_the_reference_the_oracle_holds(case):
    """Splash in fp32, forward and every gradient (the sinks' included), on
    par from float64 with the fp32 reference attention it replaces, summed as
    the kernel's target sums (check 1 over drawn operands, the truth attention
    computed in float64 apart from every Dew path, `exact_value_and_grads`),
    whose own arithmetic in float64 is first held to that truth. Each argument
    the kernel takes, grouped key heads among them. Interpreted on a CPU, a
    kernel's dots are XLA:CPU's, not the kernel's, so its numerics on its own
    hardware are held by `test_splash_is_on_par_on_a_tpu`. The sinks'
    gradient is a leaf of the same check over its drawn sinks (row 1569,
    where 52 head-unit orders of one draw held it).
    """
    q_shape, kv_shape, structure, sunk = oracle_layout(case)
    query, _, _ = qkv(q_shape, jnp.float32)
    _, key, value = qkv(kv_shape or q_shape, jnp.float32, seed=1)
    if sunk:
        structure["sinks"] = 2.0 + jax.random.normal(jax.random.PRNGKey(2), (q_shape[2],), jnp.float32)
    splash = value_and_grads('tpu', query, key, value, **structure)
    with jax.enable_x64(new_val=True):
        wide = {name: jnp.asarray(np.asarray(x, np.float64)) if name == "sinks" else x
                for name, x in structure.items()}
        widened = (jnp.asarray(np.asarray(x, np.float64)) for x in (query, key, value))
        twin = value_and_grads('reference', *widened, force_fp32_for_softmax=False, **wide)
    names = ("output", "query", "key", "value", "sinks")[:len(splash)]
    truth = exact_value_and_grads(query, key, value, **structure)
    # Splash interprets the TPU kernel on this CPU: its reference sums as the target does (fp32),
    # not as the CPU's policy does.
    with jax.default_matmul_precision(FP32_SUMS):
        expected = reference(query, key, value, **structure)
    for name, mine, theirs, wide_theirs, want in zip(names, splash, expected, twin, truth, strict=True):
        assert mine.shape == theirs.shape == want.shape, name
        # A key's logit sums the head width, its weight the keys.
        assert_computes_the_oracle(wide_theirs, want, f"{case} {name}",
                                   roundings=query.shape[-1] + key.shape[1])

    def draw(seed):
        query, _, _ = qkv(q_shape, jnp.float32, seed=2 * seed)
        _, key, value = qkv(kv_shape or q_shape, jnp.float32, seed=2 * seed + 1)
        if not sunk:
            return query, key, value
        return query, key, value, 2.0 + jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(2), seed),
                                                          (q_shape[2],), jnp.float32)

    def leaves(run):
        @as_given
        def judged_leaves(query, key, value, *sinks):
            given = {**{name: value for name, value in structure.items() if name != "sinks"},
                     **({"sinks": sinks[0]} if sinks else {})}
            return tuple(run(query, key, value, **given))
        return judged_leaves

    assert_as_exact(leaves(functools.partial(value_and_grads, 'tpu')), leaves(with_fp32_sums(reference)),
                    draw, f"interpreted splash, {case}", seeds=SPLASH_SEEDS.get(case, FEWEST_SEEDS),
                    truth_fn=leaves(exact_value_and_grads))


@on_tpu
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("case", ORACLE_CASES)
def test_splash_is_on_par_on_a_tpu(case, dtype):
    """Splash compiled by Mosaic on a TPU, in fp32 and in bf16, its production dtype, its output and every
    gradient (the sinks' included, whose D is the dQ kernel's own, not the rounded output's) held to check 1
    over fresh draws against XLA's own attention of the same operands at the precision a TPU runs it
    (DEFAULT, as Splash's fp32 matmuls run), truth in float64 on the host. Interpreted on a CPU, a
    kernel's dots are XLA:CPU's, not the kernel's: its dV is bit for bit XLA:CPU's `lax.dot` of the
    weights transposed by the output cotangent, which XLA:CPU sums 2.6 times less exactly than the
    reference's contraction of the same operands over their leading axes, so the CPU sites above hold
    the function and this one the numerics, over every one of their cases (`oracle_layout`). The float64
    truth takes the host's CPU backend beside the TPU (JAX_PLATFORMS=tpu,cpu, as the TPU tier and
    tools/qualify_splash.py start it)."""
    q_shape, kv_shape, structure, sunk = oracle_layout(case)
    _on_par_on_a_tpu(q_shape, kv_shape, dtype, structure, f"splash on a tpu, {case}", sunk=sunk)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis, key_heads, packed",
                         [("data", 2, True), ("tensor", 2, False), ("tensor", 1, False)],
                         ids=["packed-rows", "heads", "heads-past-the-keys"])
def test_splash_keeps_the_kernel_contract_on_a_mesh(axis, key_heads, packed):
    """Splash over packed rows split two ways, query heads split two ways with the key heads they read,
    or query heads split past their one key head, which each shard then holds a copy of, runs per shard
    (`sharded.assert_runs_per_shard`) under pallas's interpreter: the output and the three input
    gradients on par with one device's over the site's draws, and no operand moved between devices."""
    from jax.sharding import Mesh, PartitionSpec
    from sharded import assert_runs_per_shard

    def draw(seed):
        query, _, _ = qkv((2, 256, 4, 64), jnp.float32, seed=3 * seed)
        _, key, value = qkv((2, 256, key_heads, 64), jnp.float32, seed=3 * seed + 1)
        return query, key, value, qkv((2, 256, 4, 64), jnp.float32, seed=3 * seed + 2)[0]

    structure = {"causal": True, "segment_ids": packed_ids((100, 156), (30, 226)) if packed else None}
    rows, heads = PartitionSpec(axis), PartitionSpec(None, None, axis)
    queries = rows if axis == "data" else heads
    keys = rows if axis == "data" else heads if key_heads > 1 else PartitionSpec()
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    splash = functools.partial(scaled_dot_product_attention, implementation='tpu', **structure)
    assert_runs_per_shard(with_gradients(splash), draw, [queries, keys, keys, queries], mesh,
                          f"splash split over {axis}, {key_heads} key heads", seeds=SHARDED_SEEDS,
                          truth_fn=with_gradients(lambda q, k, v: exact_attention(q, k, v, anchored=True,
                                                                                    **structure)))


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("axes", [{"data": 2}, {"tensor": 2}, {"data": 2, "tensor": 2}],
                         ids=["rows", "heads", "rows-and-heads"])
@pytest.mark.parametrize("kernel", ["splash", "flash"])
def test_the_tpu_kernels_lower_for_a_tpu_on_a_mesh_that_splits_rows_or_heads(axes, kernel):
    """jax refuses to lower a Mosaic kernel the partitioner would have to split ("Mosaic kernels
    cannot be automatically partitioned"), so on a v6e mesh splitting rows, heads or both, each
    kernel of the attention kernel 'tpu', splash and the flash kernel a bias sends a call to, lowers
    forward and backward as each shard's rows and heads (`sharded.tpu_lowering`); sinks and a bias
    split with their heads."""
    from sharded import placed, tpu_lowering, tpu_mesh

    mesh = tpu_mesh(**axes)
    rows, heads = ("data" if "data" in axes else None), ("tensor" if "tensor" in axes else None)
    query = placed(jax.ShapeDtypeStruct((4, 256, 4, 128), jnp.bfloat16), mesh, rows, None, heads)
    key = placed(jax.ShapeDtypeStruct((4, 256, 2, 128), jnp.bfloat16), mesh, rows, None, heads)
    extra = (placed(jax.ShapeDtypeStruct((4,), jnp.float32), mesh, heads) if kernel == "splash"
             else placed(jax.ShapeDtypeStruct((1, 4, 256, 256), jnp.float32), mesh, None, heads))

    def loss(q, k, v, extra):
        call = {"sinks": extra} if kernel == "splash" else {"bias": extra}
        out = scaled_dot_product_attention(q, k, v, implementation='tpu', causal=True, **call)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    tpu_lowering(jax.grad(loss, argnums=(0, 1, 2, 3)), (query, key, key, extra), mesh)


@pytest.mark.parametrize("structure", [{"causal": True}, {"causal": True, "softcap": 1.0},
                                       {"causal": True, "sinks": True}], ids=["causal", "softcap", "sinks"])
def test_a_query_of_one_key_takes_an_exactly_zero_query_gradient(structure):
    """A causal first query reads one key, whose weight is exactly one, so the softmax's derivative and the
    reference's query gradient there are exactly zero; Splash's is too, its delta formed from the weights
    (`splash_block_sizes`), where the flash form, rowsum(o * do), left the output's rounding (1.2e-6 on a
    v5e). With a sink the key's weight is below one, and the row is not zero: the test holds the
    reference's zeros alone."""
    query, _, _ = qkv((2, 256, 4, 64), jnp.float32)
    _, key, value = qkv((2, 256, 4, 64), jnp.float32, seed=1)
    structure = dict(structure)
    if structure.pop("sinks", False):
        structure["sinks"] = 2.0 + jax.random.normal(jax.random.PRNGKey(2), (4,), jnp.float32)
    splash = value_and_grads('tpu', query, key, value, **structure)
    expected = reference(query, key, value, **structure)
    zero = expected[1] == 0
    np.testing.assert_array_equal(splash[1][zero], 0.0)
    if "sinks" not in structure:
        assert zero[:, 0].all()


def test_a_row_weighs_its_own_keys_however_far_below_every_finite_logit():
    """A causal call whose first query's one key has a logit of about -3e38, below any finite stand-in for a
    hidden key's: Splash weighs that key whole and the hidden keys nothing, so the row reads the key's own
    value, 0, beside hidden values of 1; every gradient finite."""
    width, tokens = 128, 128
    query = np.zeros((1, tokens, 1, width), np.float32)
    key = np.zeros((1, tokens, 1, width), np.float32)
    value = np.ones((1, tokens, 1, width), np.float32)
    query[0, 0, 0, 0], key[0, 0, 0, 0], value[0, 0] = -5.8e19, 5.8e19, 0.0
    out, *grads = value_and_grads('tpu', *(jnp.asarray(x) for x in (query, key, value)), causal=True)
    np.testing.assert_array_equal(out[0, 0], 0.0)
    assert all(np.isfinite(grad).all() for grad in grads)


@on_tpu
@pytest.mark.parametrize("dtype, width, softcap", [
    (jnp.bfloat16, 128, True), (jnp.float32, 128, True), (jnp.float32, 256, False),
    (jnp.bfloat16, 256, True)], ids=["bf16-softcap", "fp32-softcap", "fp32-256", "bf16-256-softcap"])
def test_splash_compiles_within_a_tpus_scoped_vmem(dtype, width, softcap):
    """Splash's forward and backward over 2048 tokens, compiled by Mosaic on this TPU and run, every output
    and gradient finite: within its scoped VMEM, where a v5e's 16 MiB refused the fused backward's fp32
    softcap at width 128 at 18.6 MiB and width 256 at 16.2 to 22.3 MiB until its blocks were sized to it
    (`splash.block_sizes`)."""
    query, key, value = qkv((1, 2048, 2, width), dtype)
    capped = {"softcap": 30.0} if softcap else {}
    found = value_and_grads('tpu', query, key, value, causal=True, **capped)
    assert all(np.isfinite(array).all() for array in found)


@on_tpu
def test_the_attention_kernels_matrix_holds_on_this_tpu():
    """The attention op's kernels' check-1 matrix (tests/kernel_matrix.py) on the TPU tier: splash's training
    step through the public call on par with the reference path at every pair of axis values it serves,
    each cell on its own TPU (`attention_kernels_matrix`)."""
    run_cells(cells_of(kernels_matrix.a_kernels_training_step_is_on_par))


@on_tpu
def test_the_decode_matrix_holds_on_this_tpu():
    """Decode's check-1 matrix (tests/kernel_matrix.py) on the TPU tier: a paged pool's decode step through
    jax's Pallas paged attention on par with the reference path at every pair of axis values it serves, each
    cell on its own TPU (`decode_matrix`)."""
    run_cells(cells_of(decode_matrix.a_decode_step_is_on_par))


@on_tpu
@pytest.mark.parametrize(("width", "precision"), [(64, None), (128, "highest")],
                         ids=["default-64", "highest-128"])
def test_a_paged_decode_step_on_this_tpu_falls_back_where_jaxs_kernel_refuses(width, precision):
    """A bf16 paged decoder step on this TPU (`CausalTransformer` -> `CausalSelfAttention`, rows 7385,
    10142, 10760, 10761, 10825) at a width (64) or precision (`'highest'` on v5e/v6e) where
    `kv_cache.TPU_PAGED_SUPPORTS` refuses jax's Pallas paged attention: on par with the compact cache's
    reference decode (check 1) through the layer's gathered fallback where refused, and through `tpu` where
    a newer TPU's MXU takes fp32-by-bf16."""
    from decode_matrix import assert_paged_decode_on_par

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kernels import generation
    def model(layout, dtype):
        return CausalTransformer(vocab_size=64, emb_features=width * 2, num_layers=1, num_heads=4,
                                 num_kv_heads=2, head_dim=width, mlp_features=width * 2, max_seq_len=32,
                                 dtype=dtype, kv_cache=layout)

    served = width % 128 == 0 and (precision is None or generation.gpu_target().fp32_by_bf16_precise)
    with jax.default_matmul_precision(precision), generation.routes() as routes:
        assert_paged_decode_on_par(model, f"a paged decode step {width} wide at {precision}",
                                   steps=6, seeds=8, ref_impl="reference")
    assert (("paged_decode", "tpu") in {(found.choice, found.kernel) for found in routes}) == served


def test_splash_span_mask_matches_key_spans_without_dense_numpy_mask():
    """Row 956, 12219: `splash_mask_descriptor(..., spans=...)` builds a Splash `MultiHeadMask` directly
    from static single-row `KeySpans` (causal, chunked, windowed, and multi-run spans) without allocating a
    dense `NumpyMask`, and runs interpreted Splash attention matching `template.dense(spans)`."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

    from dew.nn.attention_tpu import splash_attention, splash_mask_descriptor, tpu_attention
    from dew.nn.kernels.templates import attention as template

    def span_desc(spans: template.KeySpans, *, q_len: int = 128, kv_len: int = 128, heads: int = 2):
        return splash_mask_descriptor(
            q_len, kv_len, heads, causal=False, sliding_window=None, mask=None, spans=spans)

    chunk_spans = template.spans(128, 128, 1, causal=True, chunk=32)
    desc = span_desc(chunk_spans)
    assert desc is not None and not any(isinstance(m, splash_attention_mask.NumpyMask) for m in desc.masks)
    np.testing.assert_array_equal(desc.masks[0][:, :], np.asarray(template.dense(chunk_spans)[0]))
    np.testing.assert_array_equal(desc.masks[0][32:64, 0:64],
                                  np.asarray(template.dense(chunk_spans)[0, 32:64, 0:64]))
    assert bool(desc.masks[0][35, 32]) and not bool(desc.masks[0][35, 0])
    assert desc == span_desc(chunk_spans)

    # Multi-run spans: union of two disjoint intervals per query.
    q_pos = np.arange(128, dtype=np.int32)
    runs = np.stack([np.stack([np.zeros_like(q_pos), np.minimum(16, q_pos + 1)], axis=-1),
                     np.stack([np.maximum(q_pos - 15, 0), q_pos + 1], axis=-1)], axis=1)[None]
    run_spans = template.spans(128, 128, 1, runs=runs)
    run_desc = span_desc(run_spans)
    assert run_desc is not None
    np.testing.assert_array_equal(run_desc.masks[0][:, :], np.asarray(template.dense(run_spans)[0]))
    # Large static spans past SPLASH_DENSE_MASK_CELLS (4 Mi cells) succeed via `_SpanMask` without NumpyMask.
    wide_spans = template.spans(2048, 2048, 1, causal=True, chunk=256)
    wide_desc = span_desc(wide_spans, q_len=2048, kv_len=2048, heads=16)
    assert wide_desc is not None and not any(
        isinstance(m, splash_attention_mask.NumpyMask) for m in wide_desc.masks)

    # End-to-end interpreted Splash forward with chunked `_SpanMask` matches `NumpyMask` bit for bit.
    dense_chunk = np.asarray(template.dense(chunk_spans)[0])
    numpy_desc = splash_attention_mask.MultiHeadMask([splash_attention_mask.NumpyMask(dense_chunk)] * 2)
    query, key, value = qkv((1, 128, 2, 64), jnp.float32, seed=956)
    got = splash_attention(query, key, value, desc, softcap=None, sinks=None, segment_ids=None,
                           interpret=True)
    ref = splash_attention(query, key, value, numpy_desc, softcap=None, sinks=None, segment_ids=None,
                           interpret=True)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    # Public `scaled_dot_product_attention(..., implementation='tpu', key_runs=...)` routes static spans
    # directly through `_SpanMask` and matches interpreted Splash with `run_desc` bit for bit.
    public_got = scaled_dot_product_attention(query, key, value, implementation="tpu", key_runs=runs)
    direct_ref = splash_attention(query, key, value, run_desc, softcap=None, sinks=None, segment_ids=None,
                                  interpret=True)
    np.testing.assert_array_equal(np.asarray(public_got), np.asarray(direct_ref))
    # Closed-over concrete `KeySpans` (including genuine `per_span > 1` repeated spans) and static `key_runs`
    # stay on host under `jax.jit`, routing through `_SpanMask` without tracer conversion errors.
    block_starts = (np.arange(32, dtype=np.int32) * 4)[None, :]
    block_ends = np.minimum(block_starts + 32, 128)
    repeated_spans = template.KeySpans(jnp.asarray(block_starts), jnp.asarray(block_ends), None, None, 128,
                                       per_span=4)
    rep_desc = span_desc(repeated_spans)
    assert rep_desc is not None
    rep_ref = splash_attention(query, key, value, rep_desc, softcap=None, sinks=None, segment_ids=None,
                               interpret=True)
    jitted_rep = jax.jit(lambda q, k, v: tpu_attention(
        q, k, v, None, None, causal=False, sliding_window=None, softcap=None, sinks=None, segment_ids=None,
        interpret=True, descriptor=span_desc(repeated_spans)))(query, key, value)
    np.testing.assert_array_equal(np.asarray(jitted_rep), np.asarray(rep_ref))
    jitted_public = jax.jit(lambda q, k, v: scaled_dot_product_attention(
        q, k, v, implementation="tpu", key_runs=runs))(query, key, value)
    np.testing.assert_array_equal(np.asarray(jitted_public), np.asarray(direct_ref))
    # Broadcastable `[Q, 1]` and `[1, K]` concrete masks combined with static spans and `softcap` stay on
    # Splash without falling through to Flash, matching an independently materialized span-and-mask oracle.
    softcap = 30.0
    assert splash_mask_descriptor(4096, 4096, 1, causal=False, sliding_window=None,
                                  mask=np.ones((4096, 1), bool)) is None
    dense_runs = np.asarray(template.dense(run_spans)[0])
    q_mask = (np.arange(128) % 2 == 0)[:, None]
    k_mask = (np.arange(128) % 3 != 0)[None, :]
    for bcast_mask, row_keep in ((q_mask, q_mask), (k_mask, None)):
        combined_desc = splash_mask_descriptor(128, 128, 2, causal=False, sliding_window=None,
                                               mask=bcast_mask, spans=run_spans)
        assert combined_desc is not None
        oracle_dense = dense_runs & np.broadcast_to(bcast_mask, (128, 128))
        oracle_desc = splash_attention_mask.MultiHeadMask([splash_attention_mask.NumpyMask(oracle_dense)] * 2)
        oracle_ref = splash_attention(query, key, value, oracle_desc, softcap=softcap, sinks=None,
                                      segment_ids=None, interpret=True)
        if row_keep is not None:
            oracle_ref = jnp.where(jnp.asarray(row_keep)[None, :, None, :], oracle_ref, 0.0)
        masked_got = scaled_dot_product_attention(query, key, value, mask=bcast_mask, implementation="tpu",
                                                  key_runs=runs, softcap=softcap)
        jitted_masked = jax.jit(lambda q, k, v, m=bcast_mask: scaled_dot_product_attention(
            q, k, v, mask=m, implementation="tpu", key_runs=runs, softcap=softcap))(query, key, value)
        np.testing.assert_array_equal(np.asarray(masked_got), np.asarray(oracle_ref))
        np.testing.assert_array_equal(np.asarray(jitted_masked), np.asarray(oracle_ref))
        direct_prebuilt = tpu_attention(query, key, value, None, None, causal=False, sliding_window=None,
                                        softcap=softcap, sinks=None, segment_ids=None, interpret=True,
                                        descriptor=combined_desc)
        if row_keep is not None:
            direct_prebuilt = jnp.where(jnp.asarray(row_keep)[None, :, None, :], direct_prebuilt, 0.0)
        np.testing.assert_array_equal(np.asarray(direct_prebuilt), np.asarray(oracle_ref))
    # Confirm `[1, K]` genuinely alters visible rows' outputs relative to unmasked `key_runs`.
    unmasked_sc = scaled_dot_product_attention(query, key, value, implementation="tpu", key_runs=runs,
                                               softcap=softcap)
    k_masked_sc = scaled_dot_product_attention(query, key, value, mask=k_mask, implementation="tpu",
                                               key_runs=runs, softcap=softcap)
    assert not np.array_equal(np.asarray(k_masked_sc), np.asarray(unmasked_sc))


def test_bare_splash_supports_jvp_and_higher_order_with_sinks_softcap_and_segments():
    """Row 2654: bare `splash._attention` (via `tpu_attention` without `forward_mode_attention()`) supports
    `jax.jvp`, `jax.jvp(jax.grad(...))`, and `jax.grad(jax.grad(...))` with causal sparse `MaskInfo`,
    `softcap`, `sinks`, and `segment_ids`."""
    from dew.nn.attention_tpu import splash_mask_descriptor, tpu_attention

    b, s, h, d = 1, 128, 8, 64
    seg = jnp.concatenate([jnp.ones((b, 64), jnp.int32), jnp.full((b, 64), 2, jnp.int32)], axis=1)
    desc = splash_mask_descriptor(s, s, h, causal=True, sliding_window=None, mask=None)

    def call(qi, ki, vi, si):
        return tpu_attention(qi, ki, vi, None, None, causal=True, sliding_window=None, softcap=20.0,
                             sinks=si, segment_ids=seg, interpret=True, descriptor=desc)

    def ref(qi, ki, vi, si):
        return scaled_dot_product_attention(qi, ki, vi, causal=True, softcap=20.0, sinks=si,
                                            segment_ids=seg, implementation="reference")

    def draw(seed: int):
        rng = np.random.default_rng([2654, seed])
        return (
            jnp.asarray(rng.standard_normal((b, s, h, d)), jnp.float32),
            jnp.asarray(rng.standard_normal((b, s, h, d)), jnp.float32),
            jnp.asarray(rng.standard_normal((b, s, h, d)), jnp.float32),
            jnp.asarray(rng.standard_normal((h,)), jnp.float32),
            jnp.asarray(rng.standard_normal((b, s, h, d)), jnp.float32),
            jnp.asarray(rng.standard_normal((b, s, h, d)), jnp.float32),
            jnp.asarray(rng.standard_normal((b, s, h, d)), jnp.float32),
            jnp.asarray(rng.standard_normal((h,)), jnp.float32),
            jnp.asarray(rng.standard_normal((b, s, h, d)), jnp.float32),
        )

    def jvp_and_hvp(fn):
        def run(qi, ki, vi, si, dqi, dki, dvi, dsi, cot):
            out, dout = jax.jvp(fn, (qi, ki, vi, si), (dqi, dki, dvi, dsi))
            grads, hvp = jax.jvp(
                jax.grad(lambda qx, kx, vx, sx: jnp.sum(fn(qx, kx, vx, sx) * cot), argnums=(0, 1, 2, 3)),
                (qi, ki, vi, si), (dqi, dki, dvi, dsi))
            return out, dout, grads, hvp
        return run

    assert_as_exact(jax.jit(jvp_and_hvp(call)), jax.jit(jvp_and_hvp(ref)), draw,
                    "bare splash jvp and hvp with sinks softcap and segments", seeds=SPLASH_SEEDS["sinks"])
    q, k, v, sinks, _, _, _, _, cot = draw(0)
    d2q = jax.grad(lambda qi: jnp.sum(
        jax.grad(lambda qx: jnp.sum(call(qx, k, v, sinks) * cot))(qi) ** 2))(q)
    assert jnp.all(jnp.isfinite(d2q))


def test_native_bf16_splash_lowers_jvp_and_omitted_tiles_have_zero_jvp():
    """Row 2654: native BF16 Splash (`interpret=False` and explicit `precision='highest'`) lowers `jax.jvp`
    for TPU without reverse-only `at_precision`, and skipped `block_mask == 0` tiles contribute exact zero
    JVP even when `mask_value` is finite (`0.0`)."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

    from dew.nn import programs
    from dew.nn.attention_tpu import splash_mask_descriptor, tpu_attention
    from dew.nn.kernels import generation, splash

    # Native BF16 `tpu_attention(interpret=False)` and explicit `precision='highest'` lower `jax.jvp` on v5e.
    q_bf = jax.ShapeDtypeStruct((1, 128, 2, 64), jnp.bfloat16)
    desc_bf = splash_mask_descriptor(128, 128, 2, causal=True, sliding_window=None, mask=None)
    def jvp_tpu(qi, ki, vi, dqi):
        return jax.jvp(
            lambda x: tpu_attention(x, ki, vi, None, None, causal=True, sliding_window=None, softcap=None,
                                    sinks=None, segment_ids=None, interpret=False, descriptor=desc_bf),
            (qi,), (dqi,))

    def jvp_hi(qi, ki, vi, dqi):
        return jax.jvp(
            lambda x: scaled_dot_product_attention(
                x, ki, vi, causal=True, implementation="tpu", precision="highest"),
            (qi,), (dqi,))

    with generation.assumed("v5e"):
        for fn in (jvp_tpu, jvp_hi):
            traced = jax.make_jaxpr(fn)(q_bf, q_bf, q_bf, q_bf).jaxpr
            names = [str(eq.params.get("name", "")) for eq in programs.equations(traced)
                     if programs.opaque(eq)]
            assert any("splash_mha_fwd" in name for name in names), names
            jax.jit(fn).trace(q_bf, q_bf, q_bf, q_bf).lower(lowering_platforms=("tpu",))
        with jax.default_matmul_precision("highest"):
            jax.jit(jvp_tpu).trace(q_bf, q_bf, q_bf, q_bf).lower(lowering_platforms=("tpu",))

    # Multi-tile 256x256 mask: tile 0 visible, tile 1 omitted (`block_mask == 0`, shrunken grid) with finite
    # `mask_value=0.0`: omitted tile 1 yields exact zero JVP and native dV even with NaN poison in `v`/`dv`.
    dense = np.zeros((256, 256), dtype=bool)
    dense[:, :128] = True
    m_desc = splash_attention_mask.MultiHeadMask([splash_attention_mask.NumpyMask(dense)])
    blk = splash.BlockSizes(128, 128, 128, 128, 128, 128, 128, 128)
    kernel = splash.make_splash_mha(m_desc, block_sizes=blk, mask_value=0.0, interpret=True)
    assert kernel.fwd_mask_info.data_next is not None and kernel.fwd_mask_info.data_next.shape[-1] == 1
    z = jnp.zeros((1, 256, 64), jnp.float32)
    dv = jnp.zeros((1, 256, 64), jnp.float32).at[:, 128:, :].set(1.0)
    _, dout = jax.jvp(lambda vi: kernel(z, z, vi), (z,), (dv,))
    np.testing.assert_array_equal(np.asarray(dout), np.zeros((1, 256, 64), jnp.float32))
    v_nan = z.at[:, 128:, :].set(jnp.nan)
    dv_nan = z.at[:, 128:, :].set(jnp.nan)
    out_nan, dout_nan = jax.jvp(lambda vi: kernel(z, z, vi), (v_nan,), (dv_nan,))
    np.testing.assert_array_equal(np.asarray(out_nan), np.zeros((1, 256, 64), jnp.float32))
    np.testing.assert_array_equal(np.asarray(dout_nan), np.zeros((1, 256, 64), jnp.float32))
    dv_grad = jax.grad(lambda vi: jnp.sum(kernel(z, z, vi)))(v_nan)
    np.testing.assert_array_equal(np.asarray(dv_grad[:, 128:, :]), np.zeros((1, 128, 64), jnp.float32))
    # `configured_precision('default', exact=True)` overrides an outer `'highest'` for 16-bit
    # linear_tangent rules.
    rng = np.random.default_rng(26541)
    q16 = jnp.asarray(rng.standard_normal((1, 128, 64)), jnp.bfloat16)
    k16 = jnp.asarray(rng.standard_normal((1, 128, 64)), jnp.bfloat16)
    v16 = jnp.asarray(rng.standard_normal((1, 128, 64)), jnp.bfloat16)
    dq16 = jnp.asarray(rng.standard_normal((1, 128, 64)), jnp.bfloat16)
    d128 = splash_mask_descriptor(128, 128, 1, causal=True, sliding_window=None, mask=None)
    k128 = splash.make_splash_mha(d128, block_sizes=blk, interpret=True)
    op_def = splash._build_splash_op(blk, k128.call, "default")
    args16 = (k128.fwd_mask_info, k128.dq_mask_info, k128.dkv_mask_info, q16, k16, v16, None, None)
    with jax.default_matmul_precision("default"):
        o_def, do_def = jax.jvp(lambda qi: op_def(*args16[:3], qi, *args16[4:]), (q16,), (dq16,))
    with jax.default_matmul_precision("highest"):
        o_hi, do_hi = jax.jvp(lambda qi: op_def(*args16[:3], qi, *args16[4:]), (q16,), (dq16,))
    np.testing.assert_array_equal(np.asarray(o_hi), np.asarray(o_def))
    np.testing.assert_array_equal(np.asarray(do_hi), np.asarray(do_def))


def test_splash_tangents_take_the_kernels_products_and_starting_maximum():
    """Splash's forward-mode rule computes as its kernel (`splash._dot`, `_forward`): its score products at
    the kernel's precision whatever the caller's ambient one (bf16 on a TPU), so a key `2^-9` from the others
    keeps its own score's tangent (`dQ` along that component: `127 * 2^-23` on its value, where a
    bf16 product rounded every key to one score and gave 0); and with no sink, from the mask value as its
    walk's starting maximum, so logits of -200 beneath a mask value of 0 weigh nothing, the kernel's
    underflow, not their mean (the head review of 0320fa0a3f9, F2 and F4)."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

    from dew.nn.kernels import splash

    full = splash_attention_mask.MultiHeadMask([splash_attention_mask.NumpyMask(np.ones((128, 128), bool))])
    blocks = splash.BlockSizes(128, 128, 128, 128, 128, 128, 128, 128)
    kernel = splash.make_splash_mha(full, block_sizes=blocks, interpret=True)
    query = np.zeros((1, 128, 128), np.float32)
    key = np.zeros((1, 128, 128), np.float32)
    key[:, :, 0] = 1.0
    key[0, 5, 0] = 1.0 + 2.0 ** -9
    value = np.zeros((1, 128, 128), np.float32)
    value[0, 5, :] = 1.0
    direction = np.zeros_like(query)
    direction[..., 0] = 1.0
    with jax.default_matmul_precision("bfloat16"):
        _, tangent = jax.jvp(lambda q: kernel(q, jnp.asarray(key), jnp.asarray(value)),
                             (jnp.asarray(query),), (jnp.asarray(direction),))
    # Every weight 2^-7 and every term exact in fp32: the tangent is 127 * 2^-23 bit for bit.
    np.testing.assert_array_equal(np.asarray(tangent), np.float32(127 * 2.0 ** -23))

    floored = splash.make_splash_mha(full, block_sizes=blocks, mask_value=0.0, interpret=True)
    query[..., 0] = -200.0
    key[0, 5, 0] = 1.0
    out, tangent = jax.jvp(lambda v: floored(jnp.asarray(query), jnp.asarray(key), v),
                           (jnp.asarray(value),), (jnp.ones_like(jnp.asarray(value)),))
    np.testing.assert_array_equal(np.asarray(out), 0.0)
    np.testing.assert_array_equal(np.asarray(tangent), 0.0)


def test_splash_mixed_derivatives_read_no_omitted_key():
    """An omitted tile's keys (`block_mask == 0`) reach no derivative: 256 queries over 256 keys whose mask
    visits the first 128 alone, the rest NaN in keys and values, finite visible operands; dQ's pullback moved
    along its output cotangent (forward over reverse, the portable rule's) stays finite, where the omitted
    keys entered the scores' contraction and their transpose met a score cotangent of zero (`0 * NaN`)."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

    from dew.nn.kernels import splash

    visible = np.zeros((256, 256), bool)
    visible[:, :128] = True
    mask = splash_attention_mask.MultiHeadMask([splash_attention_mask.NumpyMask(visible)])
    blocks = splash.BlockSizes(128, 128, 128, 128, 128, 128, 128, 128)
    kernel = splash.make_splash_mha(mask, block_sizes=blocks, interpret=True)
    rng = np.random.default_rng(2654)
    query = jnp.asarray(rng.normal(size=(1, 256, 128)), jnp.float32)
    key = jnp.asarray(rng.normal(size=(1, 256, 128)), jnp.float32).at[:, 128:].set(jnp.nan)
    value = jnp.asarray(rng.normal(size=(1, 256, 128)), jnp.float32).at[:, 128:].set(jnp.nan)
    cotangent = jnp.ones((1, 256, 128), jnp.float32)

    def query_gradient(q, ct):
        _, pull = jax.vjp(lambda x: kernel(x, key, value), q)
        return pull(ct)[0]

    _, moved = jax.jvp(query_gradient, (query, cotangent), (jnp.zeros_like(query), cotangent))
    assert np.isfinite(np.asarray(moved)).all()
