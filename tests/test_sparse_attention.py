"""DeepSeek sparse attention executed over the selection, not a mask.

`sparse_latent_attention` attends each query's top-k keys in MLA's latent
space. The reference is the attention the V3.2 reference computes: expanded
keys and values under the selection folded into a `[B, S, T]` mask, which
`tests/test_mla.py` holds against transformers 5.16.1 (the V3.2 fixture's
seven tokens exceed its top-k of four, so that parity test runs this path).
The selection is on par with it from the reference in float64 over drawn
inputs (check 1, tests/reference_error.py), and a dropped-out adapter's call,
which keeps the masked kernel, with the masked kernel's call: two programs,
which XLA holds to no common bits. A quantized kv_b_proj's call is still held
to the masked kernel's bit for bit: its FP8 rounding of the latents parts
float32 from float64 at 7 seeds of 1024 (the input scale's, on armada), a
whole FP8 step at an element, so check 1's control did not resolve within
512 seeds; its draw needs a margin on those roundings, the frozen product's
cotangents' too, first.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import attending
from reference_error import as_given, assert_as_exact, dew_path, widened, with_gradients

from dew.lora import LoRA
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels import generation
from dew.nn.kernels.generation import device_generation
from dew.nn.mla import MultiHeadLatentAttention
from dew.nn.sparse_selection import selection_mask, sparse_latent_attention, top_k_selection
from dew.quantization import QUANTIZATION, InputQuantization

HEADS, RANK, NOPE, ROPE, VALUE = 4, 24, 16, 8, 12
SEEDS = {'selection': 8, 'templated selection': 8, 'packed selection': 8, 'sparse layer': 8,
         'dropped-out adapter': 16,
         'act-quantized kv_b_proj': 8, 'weight-quantized kv_b_proj': 8}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""


def pieces(length, batch=2, seed=0, rank=RANK, rope=ROPE):
    keys = jax.random.split(jax.random.key(seed), 7)
    q_nope = jax.random.normal(keys[0], (batch, length, HEADS, NOPE))
    q_rot = jax.random.normal(keys[1], (batch, length, HEADS, rope))
    latent = jax.random.normal(keys[2], (batch, length, rank))
    rot = jax.random.normal(keys[3], (batch, length, rope))
    key_weight = jax.random.normal(keys[4], (rank, HEADS, NOPE)) / np.sqrt(rank)
    value_weight = jax.random.normal(keys[5], (rank, HEADS, VALUE)) / np.sqrt(rank)
    scores = jax.random.normal(keys[6], (batch, length, length))
    return q_nope, q_rot, latent, rot, key_weight, value_weight, scores


def expanded_reference(q_nope, q_rot, latent, rot, key_weight, value_weight, mask):
    """The V3.2 reference's arithmetic: per-head keys and values, masked softmax."""
    nope = jnp.einsum('btr,rhn->bthn', latent, key_weight)
    key = jnp.concatenate([nope, jnp.broadcast_to(rot[:, :, None], (*nope.shape[:3], rot.shape[-1]))], -1)
    value = jnp.einsum('btr,rhv->bthv', latent, value_weight)
    query = jnp.concatenate([q_nope, q_rot], -1)
    return scaled_dot_product_attention(query, key, value, implementation="reference",
                                        mask=mask[:, None])


def causal_selection(scores, top_k, segments=None):
    length = scores.shape[-1]
    keep = jnp.tril(jnp.ones((length, length), bool))[None]
    if segments is not None:
        keep = keep & (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] != 0)
    return top_k_selection(scores, keep, top_k)


def selection_draw(length, top_k, segments=None, cotangent=False, dtype=jnp.float32, **widths):
    """Check 1's draw of a selection's inputs: a seed's pieces (`pieces`, at `widths`) in `dtype` and the
    selection their scores make (`causal_selection`, inside `segments` where given), with `cotangent` a
    readout of the attention drawn apart from them."""
    def draw(seed):
        *operands, scores = pieces(length, seed=seed, **widths)
        drawn = (*(operand.astype(dtype) for operand in operands), causal_selection(scores, top_k, segments))
        if not cotangent:
            return drawn
        # A host stream of the seed's own, apart from every seed's `pieces` keys.
        rng = np.random.default_rng([0xC07A, seed])
        return *drawn, rng.standard_normal((len(drawn[0]), length, HEADS, VALUE)).astype(dtype)
    return draw


def selection_arms(length, rope=ROPE, **options):
    """The selection's two arms on a draw's operands and indices (`selection_draw`):
    `sparse_latent_attention` over the indices, under `options`, and the expanded reference under
    their mask."""
    def sparse(*drawn):
        return sparse_latent_attention(*drawn[:-1], drawn[-1], scale=1 / np.sqrt(NOPE + rope), **options)

    def dense(*drawn):
        return expanded_reference(*drawn[:-1], selection_mask(drawn[-1], length))

    return sparse, dense


def _pulled(attend):
    """The operands' gradients under the drawn cotangent, through the gate's pullback."""
    def run(*drawn):
        *operands, indices, cotangent = drawn
        return with_gradients(lambda *pieces: attend(*pieces, indices))(*operands, cotangent)
    return jax.jit(run)


def test_the_selection_attends_as_the_masked_expanded_reference():
    """Forward and every input gradient at fp32. A query whose causal set is
    smaller than k carries unselected slots, which read nothing. The block
    size of 5 splits 23 queries unevenly, so padding rides the last block."""
    length, top_k = 23, 6
    sparse, dense = selection_arms(length, block=5)
    assert_as_exact(_pulled(sparse), _pulled(dense), selection_draw(length, top_k, cotangent=True),
                    'selection', seeds=SEEDS['selection'])


def test_the_template_reads_the_selection_as_the_masked_expanded_reference():
    """On an A100, the attention template's decode loads each query's selected latents into its key tiles
    (`SPARSE`'s 'template', interpreted here), two rows of two queries at a latent 32 wide and a rope head
    32, bf16: forward and every input gradient on par with the expanded reference under the selection's
    mask (check 1), exercising both batch page offsets and a query whose causal set is smaller than k
    reading its named slots alone (row 9023)."""
    length, top_k = 2, 2
    sparse, dense = selection_arms(length, rope=32)
    with generation.assumed("sm80"), generation.chosen(sparse_attention="template"):
        assert_as_exact(_pulled(sparse), _pulled(dense),
                        selection_draw(length, top_k, cotangent=True, dtype=jnp.bfloat16, batch=2, rank=32,
                                       rope=32),
                        'templated selection', seeds=SEEDS['templated selection'])


def test_mixed_dtype_sparse_selection_refuses_template_and_keeps_fp32_range():
    """On a GPU, mixed fp32 queries/rope beside fp16 latents must not select `'template'` (which would narrow
    `2**16` to fp16 inf); it falls back to `'xla'` and stays finite."""
    q_nope = jnp.zeros((1, 2, 2, 16), jnp.float32)
    q_rot = jnp.full((1, 2, 2, 32), 2.0 ** 16, jnp.float32)
    latent = jnp.ones((1, 2, 32), jnp.float16)
    rot = jnp.full((1, 2, 32), 2.0 ** -16, jnp.float32)
    kw = jnp.zeros((32, 2, 16), jnp.float32)
    vw = jnp.ones((32, 2, 12), jnp.float32)
    indices = jnp.asarray([[[0, 1], [0, 1]]], jnp.int32)
    with generation.assumed("sm80"):
        out = sparse_latent_attention(q_nope, q_rot, latent, rot, kw, vw, indices, scale=1.0)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_packed_documents_select_inside_their_own_document():
    """On par (check 1) on the real tokens, the padding's queries defined by neither."""
    length, top_k = 19, 4
    segments = jnp.asarray(np.repeat([[1, 2, 3, 0]], [6, 7, 4, 2], axis=1).repeat(2, 0))

    def on_real(attend):
        return jax.jit(lambda *drawn: jnp.where((segments != 0)[..., None, None], attend(*drawn), 0))

    sparse, dense = selection_arms(length)
    assert_as_exact(on_real(sparse), on_real(dense), selection_draw(length, top_k, segments),
                    'packed selection', seeds=SEEDS['packed selection'])


@attending("reference")
def test_the_layer_runs_the_selection_and_matches_its_masked_kernel():
    """The V3.2 layer past its top-k takes the sparse path; an open `qk`
    collection sends the same call through the masked dense kernel, on par
    (check 1), its truth that kernel in float64."""
    layer = MultiHeadLatentAttention(
        emb_features=32, num_heads=HEADS, max_seq_len=64, q_lora_rank=16, kv_lora_rank=RANK,
        qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, v_head_dim=VALUE,
        index_topk=5, index_n_heads=2, index_head_dim=16)
    variables = layer.init(jax.random.key(3), jnp.zeros((2, 21, 32)))

    def masked(variables, hidden):
        return layer.apply(variables, hidden, mutable=["qk"])[0]

    assert_as_exact(jax.jit(partial(layer.apply, variables)), jax.jit(partial(masked, variables)),
                    hidden_draw, 'sparse layer', seeds=SEEDS['sparse layer'],
                    truth_fn=partial(masked, widened(variables)))


def hidden_draw(seed):
    """Check 1's draw of a layer's hidden states."""
    return (jax.random.normal(jax.random.fold_in(jax.random.key(2), seed), (2, 21, 32)),)


@attending("reference")
def test_an_adapter_dropping_out_kv_b_proj_keeps_the_masked_kernel():
    """LoRA dropout on kv_b_proj draws a mask per token and latent dimension.
    The absorbed path applies kv_b_proj to the identity, where the same draw
    would be one mask for every token, so such a training call runs the
    masked kernel: on par (check 1) with the same call with `qk` open, rng for
    rng, its truth that call in float64 under the same mask. The rate is a
    float32, so the mask draws float32 uniforms under x64 too: a float rate
    would draw float64 ones from other bits, another mask."""
    layer = MultiHeadLatentAttention(
        emb_features=32, num_heads=HEADS, max_seq_len=64, q_lora_rank=16, kv_lora_rank=RANK,
        qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, v_head_dim=VALUE,
        index_topk=5, index_n_heads=2, index_head_dim=16)
    adapter = LoRA(rank=4, modules=("kv_b_proj",), alpha=8.0, dropout=np.float32(0.5)).apply(
        layer, layer.init(jax.random.key(3), jnp.zeros((2, 21, 32))), key=3)
    adapted, variables = adapter.model, adapter.variables
    factor = variables["params"]["kv_b_proj"]["lora_B"]
    variables["params"]["kv_b_proj"]["lora_B"] = jax.random.normal(jax.random.key(4), factor.shape)
    rngs = {"dropout": jax.random.key(5)}

    def masked(variables, hidden):
        return adapted.apply(variables, hidden, rngs=rngs, mutable=["qk"])[0]

    assert_as_exact(jax.jit(lambda hidden: adapted.apply(variables, hidden, rngs=rngs)),
                    jax.jit(partial(masked, variables)), hidden_draw, 'dropped-out adapter',
                    seeds=SEEDS['dropped-out adapter'], truth_fn=partial(masked, widened(variables)))


@attending("reference")
@pytest.mark.parametrize("quantized", ["input", "frozen"])
def test_a_quantized_kv_b_proj_keeps_the_masked_kernel(quantized):
    """A checkpoint's FP8 input scale at kv_b_proj, or a frozen FP8 product
    there, rounds the latents themselves. The absorbed path applies kv_b_proj
    to the identity, where the rounding would land on the identity instead,
    so such a call runs the masked kernel: its output and every gradient are
    the same call's with `qk` open."""
    layer = MultiHeadLatentAttention(
        emb_features=32, num_heads=HEADS, max_seq_len=64, q_lora_rank=16, kv_lora_rank=RANK,
        qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, v_head_dim=VALUE,
        index_topk=5, index_n_heads=2, index_head_dim=16)
    hidden = jax.random.normal(jax.random.key(2), (2, 21, 32))
    variables = layer.init(jax.random.key(3), hidden)
    if quantized == "input":
        record = InputQuantization(sites=(("kv_b_proj", "fp8"),))
        held = {"kv_b_proj": {"fp8_input_scale": jnp.asarray(0.01, jnp.float32)}}
    else:
        record = InputQuantization(frozen_fp8=("kv_b_proj",), measured_on=device_generation())
        held = record.packed(variables["params"])
    model = record.apply(layer)
    cotangent = jax.random.normal(jax.random.key(4), hidden.shape)

    def run(*, qk: bool):
        def loss(params, x):
            tree = {"params": params, QUANTIZATION: held}
            output = model.apply(tree, x, mutable=["qk"])[0] if qk else model.apply(tree, x)
            return jnp.sum(output * cotangent), output

        (_, output), gradients = jax.value_and_grad(loss, (0, 1), has_aux=True)(variables["params"], hidden)
        return output, gradients

    for sparse, masked in zip(jax.tree.leaves(run(qk=False)), jax.tree.leaves(run(qk=True)), strict=True):
        np.testing.assert_array_equal(np.asarray(sparse), np.asarray(masked))


def latent_layer() -> MultiHeadLatentAttention:
    """The V3.2 layer the quantized kv_b_proj checks run, its selection narrower than its keys."""
    return MultiHeadLatentAttention(
        emb_features=32, num_heads=HEADS, max_seq_len=64, q_lora_rank=16, kv_lora_rank=RANK,
        qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, v_head_dim=VALUE,
        index_topk=5, index_n_heads=2, index_head_dim=16)


def produced_shapes(jaxpr) -> set[tuple[int, ...]]:
    """Every shape an equation of `jaxpr`, or of a program nested in one, produces."""
    shapes = set()
    for equation in jaxpr.eqns:
        shapes |= {tuple(var.aval.shape) for var in equation.outvars if hasattr(var.aval, "shape")}
        for parameter in equation.params.values():
            for nested in parameter if isinstance(parameter, tuple | list) else (parameter,):
                program = getattr(nested, "jaxpr", nested)
                if hasattr(program, "eqns"):
                    shapes |= produced_shapes(program)
    return shapes


@attending("reference")
@pytest.mark.parametrize("weight_only", [False, True], ids=["act", "weight"])
def test_a_qwix_quantized_kv_b_proj_attends_as_its_latents_allow(weight_only):
    """Qwix's fake-quantized products at kv_b_proj (`dew.quantization.Quantization`). Quantizing their
    activations rounds kv_b_proj's input, the latents themselves; the absorbed path applies kv_b_proj
    to the identity, where the rounding would land on the identity instead, so such a call runs the
    masked kernel, whose logits are `[B, H, S, T]`. Weight-only, the product stays linear in the
    latents, and the absorbed path reads the quantized matrix itself, so the call stays on the
    selection and holds no such logits. Either is on par (check 1) with the same call with `qk` open,
    its truth that call in float64, each seed drawing its own weights."""
    pytest.importorskip("qwix")
    from dew.quantization import Quantization

    layer = latent_layer()
    model = Quantization(patterns=("kv_b_proj",), weight_only=weight_only).apply(layer)
    init = jax.jit(layer.init)

    def draw(seed):
        return (init(jax.random.key(seed), jnp.zeros((2, 21, 32))), *hidden_draw(seed))

    def masked(variables, hidden):
        return model.apply(variables, hidden, mutable=["qk"])[0]

    traced = jax.make_jaxpr(model.apply)(*draw(0))
    assert ((2, HEADS, 21, 21) in produced_shapes(traced.jaxpr)) == (not weight_only)
    label = 'weight-quantized kv_b_proj' if weight_only else 'act-quantized kv_b_proj'
    assert_as_exact(jax.jit(model.apply), dew_path(jax.jit(masked)), draw, label, seeds=SEEDS[label],
                    truth_fn=as_given(lambda variables, hidden: masked(widened(variables), hidden)))


@attending("reference")
def test_a_provider_dew_did_not_configure_keeps_kv_b_proj_on_the_masked_kernel():
    """A model quantized through `quantize_with` by a provider of its caller's own, under the caller's
    opaque cache key (here 7), computes under no Dew quantization whose rules a layer can read, so MLA
    takes kv_b_proj to be quantized however the provider may: it runs the masked kernel, computing the
    same call with `qk` open bit for bit."""
    qwix = pytest.importorskip("qwix")
    from dew.quantization import quantize_with

    layer = latent_layer()
    provider = qwix.QtProvider([qwix.QtRule(module_path="kv_b_proj", weight_qtype="int8", act_qtype="int8")])
    model = quantize_with(layer, provider, key=7)
    variables, hidden = layer.init(jax.random.key(3), jnp.zeros((2, 21, 32))), hidden_draw(0)[0]
    np.testing.assert_array_equal(np.asarray(model.apply(variables, hidden)),
                                  np.asarray(model.apply(variables, hidden, mutable=["qk"])[0]))


def test_memory_follows_the_selection_not_the_sequence():
    """At 2048 keys and a top-k of 64 the masked kernel holds `[B, H, S, T]`
    logits; the selection holds `[B, S, H, K]` of them a block of queries at a
    time, so all it holds is less than that one tensor. Both measured from the
    compiled executables."""
    length, top_k = 2048, 64
    q_nope, q_rot, latent, rot, key_weight, value_weight, scores = pieces(length, batch=1)
    indices = causal_selection(scores, top_k)
    args = (q_nope, q_rot, latent, rot, key_weight, value_weight)
    mask = selection_mask(indices, length)

    def held(fn, *inputs):
        stats = jax.jit(fn).lower(*inputs).compile().memory_analysis()
        assert stats is not None
        # A TPU executable reports its temporaries only inside the peak (its
        # temp_size_in_bytes is 0); the CPU backend's peak leaves them out.
        if jax.default_backend() == "tpu":
            return stats.peak_memory_in_bytes
        return stats.argument_size_in_bytes + stats.output_size_in_bytes + stats.temp_size_in_bytes

    dense = held(expanded_reference, *args, mask)
    selected = held(lambda *a: sparse_latent_attention(*a, scale=1.0), *args, indices)
    logits = HEADS * length * length * 4
    assert dense >= logits
    assert selected < logits


def test_a_selection_ranks_every_allowed_key_above_every_forbidden_one():
    """`top_k_selection` takes an allowed key at fp32's least or -inf ahead of every forbidden key, the stable
    top-k's lower index notwithstanding (row 13536): over scores [least, least, least, 5] with the first two
    forbidden, its top two are the last key and the third, not the last and an empty slot; and among the
    allowed by their scores, fp32's least ahead of -inf."""
    least = np.finfo(np.float32).min
    for low in (least, -np.inf):
        scores = jnp.asarray([[[least, least, low, 5.0]]], jnp.float32)
        keep = jnp.asarray([[[False, False, True, True]]])
        np.testing.assert_array_equal(np.asarray(top_k_selection(scores, keep, 2))[0, 0], [3, 2])
    # Among the allowed, their scores' order: fp32's least above -inf, whatever their indices.
    both = jnp.asarray([[[-np.inf, least]]], jnp.float32)
    np.testing.assert_array_equal(np.asarray(top_k_selection(both, jnp.ones((1, 1, 2), bool), 1))[0, 0], [1])


def test_a_querys_empty_selection_slots_leave_the_next_querys_keys_alone():
    """A selection's -1 slots (`top_k_selection` leaves them where fewer keys are allowed than the top
    k) mark no key in `selection_mask`, so packed documents keep their own keys: the second query, the
    first of its document, sees its own key alone and never the first query's key 0."""
    segments = jnp.asarray([[1, 2, 2, 2]])
    indices = causal_selection(jnp.zeros((1, 4, 4)), 2, segments)
    np.testing.assert_array_equal(np.asarray(indices)[0, :2], [[0, -1], [1, -1]])
    expected = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0]], bool)
    np.testing.assert_array_equal(np.asarray(jax.jit(selection_mask, static_argnums=1)(indices, 4))[0],
                                  expected)


def test_the_sparse_latent_template_shards_over_batch_and_heads():
    """`_template_forward` maps over the active mesh's batch, sequence and head axes (`_template_specs`,
    rows 10528, 10530): each shard builds its own local latent pool and local table offsets, launches
    `attention_decode.attend` on its local query rows and local heads instead of replicating the global
    batch, handles replicated operands on a fully Explicit mesh, and matches the unsharded call across
    multiple local batch rows with distinct latent values."""
    from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P

    from dew.mesh import DATA_AXIS, SEQUENCE_AXIS, TENSOR_AXIS
    from dew.nn.sparse_selection import _template_forward

    mesh = AbstractMesh((2, 2, 2), (DATA_AXIS, SEQUENCE_AXIS, TENSOR_AXIS),
                        axis_types=(AxisType.Auto, AxisType.Auto, AxisType.Auto))
    batch, length, heads, rank, rope, keys, top_k = 4, 8, 4, 16, 16, 16, 4
    absorbed = jax.ShapeDtypeStruct((batch, length, heads, rank), jnp.bfloat16)
    query_rot = jax.ShapeDtypeStruct((batch, length, heads, rope), jnp.bfloat16)
    latent = jax.ShapeDtypeStruct((batch, keys, rank), jnp.bfloat16)
    rot = jax.ShapeDtypeStruct((batch, keys, rope), jnp.bfloat16)
    indices = jax.ShapeDtypeStruct((batch, length, top_k), jnp.int32)
    with jax.sharding.use_abstract_mesh(mesh), generation.assumed("sm90"):
        traced = jax.make_jaxpr(lambda *args: _template_forward(*args, scale=1.0)[0])(
            absorbed, query_rot, latent, rot, indices)
    (mapped,) = [eq for eq in traced.jaxpr.eqns if eq.primitive.name == "shard_map"]
    assert mapped.params["in_specs"] == (
        P(DATA_AXIS, SEQUENCE_AXIS, TENSOR_AXIS), P(DATA_AXIS, SEQUENCE_AXIS, TENSOR_AXIS),
        P(DATA_AXIS), P(DATA_AXIS), P(DATA_AXIS, SEQUENCE_AXIS))
    assert mapped.params["out_specs"] == (P(DATA_AXIS, SEQUENCE_AXIS, TENSOR_AXIS),)
    inner_shapes = produced_shapes(mapped.params["jaxpr"])
    # Local batch is 2 (not 4), local length is 4 (not 8), local heads is 2 (not 4): local pool has 32 keys.
    assert (1, 2 * keys, 1, rank + rope) in inner_shapes
    assert (1, batch * keys, 1, rank + rope) not in inner_shapes
    assert (2 * (length // 2), 1, 2, rank + rope) in inner_shapes
    assert (batch * length, 1, heads, rank + rope) not in inner_shapes

    # Replicated operands on a fully Explicit mesh reshard cleanly into the map and restore their placement.
    explicit = AbstractMesh((2,), (DATA_AXIS,), axis_types=(AxisType.Explicit,))
    rep = NamedSharding(explicit, P())
    rep_args = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=rep)
                     for x in (absorbed, query_rot, latent, rot, indices))
    with jax.sharding.use_abstract_mesh(explicit), generation.assumed("sm90"):
        explicit_traced = jax.make_jaxpr(lambda *args: _template_forward(*args, scale=1.0)[0])(*rep_args)
    assert explicit_traced.out_avals[0].sharding.spec == P(None, None, None, None)

    # Behavioral check: distinct per-batch-row latents across 2 rows per shard must match the unsharded call.
    rng = jax.random.split(jax.random.key(10530), 4)
    scale_rows = jnp.arange(1, batch + 1, dtype=jnp.bfloat16)[:, None, None]
    a_val = jax.random.normal(rng[0], (batch, length, heads, rank), jnp.bfloat16)
    qr_val = jax.random.normal(rng[1], (batch, length, heads, rope), jnp.bfloat16)
    lat_val = jax.random.normal(rng[2], (batch, keys, rank), jnp.bfloat16) * scale_rows
    rot_val = jax.random.normal(rng[3], (batch, keys, rope), jnp.bfloat16)
    sel = jnp.broadcast_to(jnp.arange(top_k, dtype=jnp.int32), (batch, length, top_k))
    with generation.assumed("sm90"):
        full = _template_forward(a_val, qr_val, lat_val, rot_val, sel, scale=1.0)[0]
        halves = jnp.concatenate([
            _template_forward(a_val[s], qr_val[s], lat_val[s], rot_val[s], sel[s], scale=1.0)[0]
            for s in (slice(0, 2), slice(2, 4))
        ], axis=0)
    np.testing.assert_array_equal(np.asarray(full), np.asarray(halves))
