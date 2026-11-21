"""Float-range and nested-derivative contracts of attention's seams: the fused attention call in forward
mode, its kernels' reverse rules, the paged decode kernel, a head's sinks (ledger 2641; the harness is
tests/contract_support.py)."""

import functools
import itertools

import jax
import jax.numpy as jnp
import pytest
from attention_oracle import plain_merged
from contract_support import EDGES, ORDER_SETS, REVERSE_ONLY, assert_drawn_contract, normal_draw
from reference_error import FEWEST_SEEDS

from dew.nn import attention, kv_cache
from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_algebra as algebra,
    attention_quantized,
    merged,
)
from dew.nn.sharding import manual_map


def plain_attention(query, key, value):
    """Softmax attention as its formula reads, `[B, S, H, D]` each: every query's logits over every key at
    the width's inverse root, their softmax's weights on the values (none of Dew's attention code)."""
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) * query.shape[-1] ** -0.5
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, axis=-1), value)


def attention_inputs(salt: int, shape: tuple):
    """One `assert_drawn_contract` builder of a query, key and value of `shape`, their directions at `scale`
    and an output cotangent, `salt` and the two after it keeping its streams apart."""
    def inputs(seed: int, scale: float) -> tuple:
        return (normal_draw([salt, seed], shape, shape, shape, scale=scale),
                normal_draw([salt + 1, seed], shape, shape, shape, scale=scale),
                normal_draw([salt + 2, seed], shape)[0])

    return inputs


@pytest.fixture(autouse=True)
def _released_as_each_test_ends():
    """Each test's compiled executables released as it ends (`conftest.release`), where the file's releases
    them at its end alone: a group of its cells holding every earlier cell's programs, each order's on a
    mesh and its check-1 seeds', was killed at 10 GiB on armada."""
    import conftest

    yield
    conftest.release()


@pytest.mark.parametrize("edge", EDGES)
def test_a_fused_attention_call_in_forward_mode_holds_every_order(edge):
    """`attention.forward_differentiable` (a fused call under `forward_mode_attention`: the template's value,
    the reference path's tangent, a custom JVP): every order runs, finite wherever float64 is a normal
    float32 and the plain attention formula is, at queries, keys and values whose products near float32's
    ends."""
    def fused(query, key, value):
        with attention.forward_mode_attention():
            return attention.scaled_dot_product_attention(query, key, value, implementation="template")

    assert_drawn_contract(fused, attention_inputs(2654, (1, 16, 2, 16)), EDGES[edge],
                          f"a fused call in forward mode {edge}", truth_fn=plain_attention,
                          plain_fn=plain_attention)


@pytest.mark.parametrize("edge", EDGES)
def test_a_heads_sinks_hold_every_order(edge):
    """`template.sinks`' spread of each head's sink over its batch rows and queries (reverse mode one
    compensated sum, row 4524): every order runs, finite wherever float64 is a normal float32 and the plain
    broadcast is."""
    def inputs(seed: int, scale: float) -> tuple:
        return (normal_draw([4524, seed], (3,), scale=scale),
                normal_draw([4525, seed], (3,), scale=scale),
                normal_draw([4526, seed], (2, 3, 5))[0])

    def plain(values):
        return jnp.broadcast_to(values[None, :, None], (2, 3, 5))

    assert_drawn_contract(lambda values: template.sinks(values, 2, 5, 4).m, inputs, EDGES[edge],
                          f"a head's sinks {edge}", plain_fn=plain)


@pytest.mark.parametrize(("implementation", "edge"), list(itertools.product(("template", "tpu"), EDGES)))
def test_a_fused_attention_kernel_holds_its_reverse_orders(implementation, edge):
    """The template's route (`attention_route._attend`) and splash (`splash._attention`, interpreted off a
    TPU), each a reverse-only custom derivative: its primal and its reverse derivative run, finite wherever
    float64 is a normal float32 and the plain attention formula is, at queries, keys and values whose
    products near float32's ends. Its other orders do not run (`REVERSE_ONLY`, ledger 2654)."""
    def fused(query, key, value):
        return attention.scaled_dot_product_attention(query, key, value, implementation=implementation)

    assert_drawn_contract(fused, attention_inputs(2657, (1, 128, 2, 64)), EDGES[edge],
                          f"{implementation} attention {edge}", truth_fn=plain_attention,
                          plain_fn=plain_attention, without=REVERSE_ONLY)


@pytest.mark.parametrize("edge", EDGES)
def test_a_quantized_attention_holds_its_reverse_orders(edge):
    """A quantized model's attention on the template (`attention_quantized.attend`, row 4858), a reverse-only
    custom derivative: its primal and its straight-through reverse derivative run, finite wherever float64
    is a normal float32 and the plain attention formula is, at queries, keys and values whose products near
    float32's ends, over a causal call. Its other orders do not run (`REVERSE_ONLY`)."""
    spans = template.spans(128, 128, 1, causal=True)

    def plain(query, key, value):
        causal = jnp.tril(jnp.ones((128, 128), bool))
        logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) * query.shape[-1] ** -0.5
        return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(jnp.where(causal, logits, -jnp.inf), axis=-1),
                          value)

    assert_drawn_contract(lambda query, key, value: attention_quantized.attend(query, key, value, spans,
                                                                               template.Hooks()),
                          attention_inputs(4858, (1, 128, 2, 64)), EDGES[edge],
                          f"quantized attention {edge}", truth_fn=plain, plain_fn=plain,
                          without=REVERSE_ONLY)


@pytest.mark.parametrize("edge", EDGES)
def test_the_paged_decode_kernel_holds_every_order(edge, kernel="gpu_paged"):
    """`kv_cache.gpu_paged` (the Pallas paged decode kernel's value), interpreted off a GPU, and
    `kv_cache.template_paged` (the attention template's decode's, at its pages of 64 keys, on an H100 in
    the GPU flow, tests/test_gpu_attention.py: Mosaic GPU's interpreter runs its kernel through ordered
    callbacks, which no rematerialization differentiates), the gathered attention's derivatives through
    `generation.linear_tangent`: every order runs in the query, the key and value pages and the sinks,
    finite wherever float64 is a normal float32 and the plain formula, one query a row over its row's first
    `lengths` keys gathered from its pages with the sinks in the softmax, is."""
    page = 16 if kernel == "gpu_paged" else 64
    table = jnp.asarray([[2, 0], [0, 3]], jnp.int32)
    lengths = jnp.asarray([page + 5, 2 * page], jnp.int32)
    shapes = ((2, 4, 64), (2, 4, page, 64), (2, 4, page, 64), (4,))

    def inputs(seed: int, scale: float) -> tuple:
        # The query, pages and sinks, their directions and the output's cotangent; the template's decode
        # takes 16-bit operands, a bf16 cache's step.
        primals = normal_draw([2660, seed], *shapes, scale=scale)
        direction = normal_draw([2661, seed], *shapes, scale=scale)
        (cotangent,) = normal_draw([2662, seed], shapes[0])
        if kernel == "template_paged":
            primals, direction = ((*(x.astype(jnp.bfloat16) for x in drawn[:3]), drawn[3])
                                  for drawn in (primals, direction))
            cotangent = cotangent.astype(jnp.bfloat16)
        return primals, direction, cotangent

    def paged(query, key, value, sinks):
        # Launched per shard as a decode step launches it (`kv_cache.decode_specs`): a kernel takes no axis
        # the partitioner holds.
        def local(query, key, value, lengths, table, sinks):
            return getattr(kv_cache, kernel)(None)(query, key, value, table, lengths, None, sinks)

        specs, out = kv_cache.decode_specs(query.shape, paged=True, sinks=True)
        return manual_map(local, specs, out)(query, key, value, lengths, table, sinks)

    def plain(query, key, value, sinks):
        # Each row's pages in its table's order, `[rows, keys, key heads, D]`, each key head read by two
        # query heads.
        keys, values = (jnp.repeat(jnp.moveaxis(pool[:, table], 0, 3).reshape(2, 2 * page, 2, 64), 2, axis=2)
                        for pool in (key, value))
        logits = jnp.einsum("rhd,rkhd->rhk", query, keys) * 64 ** -0.5
        logits = jnp.where(jnp.arange(keys.shape[1]) < lengths[:, None, None], logits, -jnp.inf)
        sunk = jnp.broadcast_to(sinks[None, :, None], (2, 4, 1))
        weights = jax.nn.softmax(jnp.concatenate([logits, sunk], axis=-1), axis=-1)[..., :-1]
        return jnp.einsum("rhk,rkhd->rhd", weights, values)

    # Declared seed count so prepared GPU-tier runs (`test_gpu_attention.py`'s `template_paged` cell) never
    # call `seeds_for` under `COMPILED` (row 9656).
    assert_drawn_contract(paged, inputs, EDGES[edge], f"the paged decode kernel {edge} {kernel}",
                          truth_fn=plain, plain_fn=plain, seeds=8)


@pytest.mark.parametrize("taken", list(ORDER_SETS))
@pytest.mark.parametrize("edge", EDGES)
def test_a_padded_merges_derivatives_are_the_laws(edge, taken):
    """`merged.merged_padded` (a split decode step's slices merged where its launch wrote them; its value one
    launch on a GPU, its tangent the law's path's, a custom JVP): every order runs, finite wherever float64
    is a normal float32 and the plain merge (`attention_oracle.plain_merged`, independent of the law) is, at
    padded outputs and maxima near float32's ends, two slices' totals drawn positive, the plain merge its
    truth; each order set a case of its own (`ORDER_SETS`: every order in one case, the law's path its plain
    formula and truth, held its group to 1,798 s)."""
    spec = algebra.finished_form(algebra.Forward("full", 3, ids=False, scale=1.0, natural=True))

    def inputs(seed: int, scale: float) -> tuple:
        # Two slices' padded outputs (three rows of which two are queries, 12 columns of which 8 are its
        # width) and maxima at `scale`, totals drawn positive, their directions and a cotangent.
        padded, maxima, totals = normal_draw([7124, seed], (1, 2, 3, 1, 12), (1, 1, 2, 3), (1, 1, 2, 3),
                                             scale=scale)
        direction = normal_draw([7125, seed], (1, 2, 3, 1, 12), (2, 1, 1, 2, 3), scale=scale)
        (cotangent,) = normal_draw([7126, seed], (1, 2, 1, 8))
        return (padded, jnp.stack([maxima, jnp.abs(totals)])), direction, cotangent

    def merge(padded, stacked):
        return merged.merged_padded(spec, padded, stacked, queries=2, width=8).acc

    plain = functools.partial(plain_merged, rate=1.0, queries=2, width=8)
    assert_drawn_contract(merge, inputs, EDGES[edge], f"a padded merge {edge} ({taken})", truth_fn=plain,
                          plain_fn=plain, seeds=FEWEST_SEEDS, taking=ORDER_SETS[taken])


@pytest.mark.parametrize("edge", EDGES)
def test_the_narrowing_settle_holds_every_order(edge):
    """`attention.narrowing_settled` (a surrogate whose derivative is its quotient's on every entry): every
    order finite wherever float64 is a normal float32 and the plain quotient is, the quotient's own, over
    drawn fp32 quotients of sums over a divisor of 3/2 as an fp16 output narrows them."""
    divisor = jnp.full((9,), 1.5, jnp.float32)

    def settled(out):
        return template.narrowing_settled(out, out * divisor, divisor, jnp.float16)

    def plain(out):
        return out

    def inputs(seed, scale):
        """A seed's quotients and their direction at `scale`, and its cotangent."""
        (out, along, cotangent) = normal_draw(seed, (9,), (9,), (9,), scale=scale)
        return (out,), (along,), cotangent

    assert_drawn_contract(settled, inputs, EDGES[edge], f"the narrowing settle {edge}", truth_fn=plain,
                          plain_fn=plain)


def test_a_nudged_quotient_narrows_once_and_keeps_its_tangent():
    """`attention.narrowing_settled` at an fp16 tie it nudges: the quotient 1 + 2^-11 formed of sums
    1 + 2^-11 - 2^-23 over a divisor 1 - 2^-24, whose exact value is below the tie, narrows to 1 as that
    value rounds once, and its tangent along the quotient is the quotient's own, 1 (the head review of
    fd1af38ac, F1: a nudged entry's tangent was none)."""
    tie = jnp.float32(1 + 2.0 ** -11)
    acc, divisor = jnp.float32(1 + 2.0 ** -11 - 2.0 ** -23), jnp.float32(1 - 2.0 ** -24)
    settled, tangent = jax.jvp(lambda out: template.narrowing_settled(out, acc, divisor, jnp.float16), (tie,),
                               (jnp.float32(1.0),))
    assert settled.astype(jnp.float16) == 1.0 and tangent == 1.0


@pytest.mark.parametrize("edge", EDGES)
def test_the_sparse_latent_kernel_holds_every_order(edge):
    """`sparse_selection.templated_context` (the attention template's decode over each query's selected
    latents, its value, interpreted off a GPU), the blocked XLA path's derivatives through
    `generation.linear_tangent`: every order runs in the absorbed queries, the rope queries, the latents and
    the rope heads, finite wherever float64 is a normal float32 and the plain formula, each query's softmax
    over its selected latents and rope heads, is; a slot naming no key reads nothing."""
    from dew.nn import sparse_selection

    shapes = ((1, 3, 2, 32), (1, 3, 2, 32), (1, 3, 32), (1, 3, 32))
    selected = jnp.asarray([[[0, -1], [1, 0], [2, 0]]], jnp.int32)
    scale = 64 ** -0.5

    def inputs(seed, size: float) -> tuple:
        # The operands, their directions and the context's cotangent from `seed`, bf16: the template's
        # decode reads 16-bit latents.
        primals, direction = (tuple(x.astype(jnp.bfloat16)
                                    for x in normal_draw([tag, seed], *shapes, scale=size))
                              for tag in (9023, 9024))
        (cotangent,) = normal_draw([9025, seed], (1, 3, 2, 32))
        return primals, direction, cotangent.astype(jnp.bfloat16)

    def sparse(absorbed, query_rot, latent, rot):
        return sparse_selection.templated_context(scale, None, 4)(absorbed, query_rot, latent, rot,
                                                                  selected)

    def plain(absorbed, query_rot, latent, rot):
        at = jnp.maximum(selected, 0)
        keys, rots = (jax.vmap(lambda rows, where: rows[where])(table, at) for table in (latent, rot))
        logits = (jnp.einsum("bqhr,bqkr->bqhk", absorbed, keys)
                  + jnp.einsum("bqhp,bqkp->bqhk", query_rot, rots)) * scale
        logits = jnp.where(selected[:, :, None, :] >= 0, logits, -jnp.inf)
        return jnp.einsum("bqhk,bqkr->bqhr", jax.nn.softmax(logits, axis=-1), keys)

    # Off a GPU, Mosaic GPU's interpreter uses OrderedIOEffect callbacks that JAX's checkpoint/remat refuses
    # to differentiate (same as `template_paged` above); on a GPU, every order including remat runs.
    without = () if generation.placed_platform() == "gpu" else ("jvp of grad, remat", "second jvp, remat")
    with generation.assumed("sm80"):
        assert_drawn_contract(sparse, inputs, EDGES[edge], f"the sparse latent kernel {edge}",
                              truth_fn=plain, plain_fn=plain, without=without, seeds=8)


@pytest.mark.parametrize("edge", EDGES)
def test_a_sink_softmax_holds_every_order(edge):
    """The sink softmax (`attention_sinks.sink_softmax`: its reverse mode returning the logits' cotangent
    only with the sinks' gradient formed): every order, forward and reverse, under a nested jit and a scan,
    in the logits and the sinks, finite wherever float64 is a normal float32 and wherever its plain softmax
    (`_weights`, its float64 run the truth) is, at logits near float32's ends."""
    from dew.nn.attention_sinks import _weights, sink_softmax

    def inputs(seed: int, scale: float) -> tuple:
        return (normal_draw([6175, seed], (2, 4, 8, 16), (4,), scale=scale),
                normal_draw([6176, seed], (2, 4, 8, 16), (4,), scale=scale),
                normal_draw([6177, seed], (2, 4, 8, 16))[0])

    assert_drawn_contract(sink_softmax, inputs, EDGES[edge], f"a sink softmax {edge}",
                          plain_fn=lambda logits, sinks: _weights(logits, sinks, None))

