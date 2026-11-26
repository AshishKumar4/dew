"""The Mosaic mainloop's launches as Dew's FLOP count reads them: each states its hardware work, counted
from its kernel's emission, and declares the model's share of it (`instrumentation`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from test_attention_mosaic import WGMMA, _pinned

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_mosaic

pytestmark = pytest.mark.kernel


def _counted(fn, *arrays, generation_name: str = "sm90") -> tuple[float, float]:
    """The model's share of the Mosaic launches' work `fn` lowers for a GPU of `generation_name` (an H100's)
    each declares (`instrumentation.MODEL_FLOPS`) and the hardware's work each states (its cost estimate), as
    the lowered program's structured reader reads them (`lowered.kernel_launches`, each launch times the calls
    a call of the program makes)."""
    from dew.nn.kernels.lowered import kernel_launches

    # Lowered deviceless for the GPU whatever the host, so a GPU host of another generation lowers no TMA of
    # its own.
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation_name)
    with generation.assumed(generation_name), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(fn).trace(*arrays).lower(lowering_platforms=("cuda",))
    launches = [(call, times) for call, times in kernel_launches(lowered) if call.kind == "mosaic_gpu"]
    assert launches and all(call.flops is not None and call.model_flops is not None for call, _ in launches)
    return (sum((call.model_flops or 0.0) * times for call, times in launches),
            sum((call.flops or 0.0) * times for call, times in launches))


@pytest.mark.parametrize("causal, queries, keys, pairs", [
    (True, 256, 256, 256 * 257 // 2), (True, 128, 256, 128 * 129 // 2),
    (True, 256, 128, 128 * 129 // 2 + 128 * 128), (True, 64, 192, 64 * 65 // 2),
    (False, 256, 256, 256 * 256), (False, 64, 192, 64 * 192)],
    ids=["causal", "causal-fewer-queries", "causal-more-queries", "causal-padded", "full", "full-padded"])
def test_the_mosaic_launches_state_the_models_attention_flops(causal, queries, keys, pairs):
    """A bf16 call's plain forward, training forward and backward through the Mosaic mainloop, each lowered
    for an H100: every launch states its hardware work, counted from its kernel's emission
    (`emission.costed`), which Dew's FLOP count reads to size it (`instrumentation._stated_flops`: no launch
    unsized), and declares the model's share of it; together the model's are the library's fused
    attention's own count, a multiply-add a
    reached pair, head and product column (a causal call's `pairs` each query's keys up to its own place,
    written out: the first key aligned to the first query, as the mainloop masks them): the forward's two
    products, the training forward's two and the backward's four (`instrumentation._fused_attention_flops`).
    The hardware's work is every tile its walk issues, whole: each 64 by 64 tile holding a reached pair over
    the padded rows (a padded consumer's and a causal diagonal tile's masked half computed with the rest, as
    an H100's tensor-pipe counts read them, row 3948's receipt), and one recomputed `Q K^T` more in the
    training forward and in the backward; a rerun (the fold's natural walk) adds none. The launches' own
    statements are read."""
    from dew.telemetry.instrumentation import hlo_flops

    rows, heads, kv_heads, width = 1, 4, 2, 64
    query = jax.ShapeDtypeStruct((rows, queries, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, keys, kv_heads, width), jnp.bfloat16)
    spans = template.spans(queries, keys, rows, causal=causal)
    forward = 2 * rows * heads * pairs * (width + width)
    # The 64 by 64 tiles the pinned launches walk that hold a pair a causal call reaches (a key at or before
    # its query), each computed whole, over the queries padded to a program's two consumers' rows (a padded
    # consumer walks its keys as a real one does) and the call's keys: the keys pad to two tiles too, a tile
    # wholly past them walked by none.
    block, consumers = 64, 2
    padded = -(-queries // (consumers * block)) * consumers * block
    walked = sum(block * block for first_q in range(0, padded, block) for first_k in range(0, keys, block)
                 if not causal or first_k < first_q + block)
    issued = 2 * rows * heads * walked * (width + width)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")

    def trained(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, residuals=template.Kept.SPLIT)

    with generation.assumed("sm90"):
        residuals = jax.eval_shape(trained, query, key, key)[1:]

    def backward(query, key, value, dout, residuals):
        return attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals)

    # The training forward recomputes its walk's `Q K^T`; the fast backward its `S` beside its four.
    product = issued // 2
    pins = (attention_mosaic.Tiles(block, block, 2, consumers, family=WGMMA),
            attention_mosaic.BackwardTiles(block, block, 2, consumers, fused=True, family=WGMMA))
    with _pinned(*pins):
        plain = _counted(lambda q, k, v: attention_mosaic.forward(q, k, v, spans), query, key, key)
    assert plain == (forward, issued)
    # A forward whose launches walk its value columns a block apiece: each recomputes `Q K^T`, which the
    # model owns once. The blocks' launches are one kernel called once a block, as the step's FLOP count
    # weighs it (`hlo_flops`, against the whole walk's count).
    def weighed(fn):
        with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
            found = hlo_flops(jax.jit(fn).trace(query, key, key).lower(lowering_platforms=("cuda",))
                              .as_text(dialect="hlo"))
        assert found is not None
        return found.model, found.hardware

    with _pinned(*pins):
        whole = weighed(lambda q, k, v: attention_mosaic.forward(q, k, v, spans))
    with _pinned(attention_mosaic.Tiles(block, block, 2, consumers, family=WGMMA, columns=width // 2)):
        blocked = weighed(lambda q, k, v: attention_mosaic.forward(q, k, v, spans))
    assert blocked == (whole[0], whole[1] + product)
    with _pinned(*pins):
        assert _counted(trained, query, key, key) == (forward, issued + product)
        assert _counted(backward, query, key, key, query, residuals) == (2 * forward, 2 * issued + product)


def test_a_common_walk_states_every_tile_each_holder_walks():
    """The fast backward's one launch fed by its consumers (`wgmma.fed`, a program two key blocks, each
    consumer walking every query tile either block is read by, `Family.walk_own`) over a causal call of 128
    queries and 192 keys, its keys padded to 256, lowered for an H100: the launch states the tiles each
    consumer walks (`attention_mosaic_program.walks`), one its own keys are not seen by too. The first
    program's consumers walk both 64-row query tiles each, the second keys' tile read by the second alone;
    the second program's none, its first keys past every query's and its second past the keys."""
    rows, heads, width, block = 1, 2, 64, 64
    query = jax.ShapeDtypeStruct((rows, 128, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, 192, heads, width), jnp.bfloat16)
    spans = template.spans(128, 192, rows, causal=True)
    with generation.assumed("sm90"):
        residuals = jax.eval_shape(lambda q, k, v: attention_mosaic.forward(
            q, k, v, spans, residuals=template.Kept.SPLIT), query, key, key)[1:]

    def backward(query, key, value, dout, residuals):
        return attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals)

    # The backward's five products (`S` recomputed, dP's, dV's, dK's and dQ's) over four tiles where each
    # consumer walked its own keys' readers alone: three.
    walked, pairs = 4 * block * block, 128 * 129 // 2
    with _pinned(backward=attention_mosaic.BackwardTiles(block, block, 2, 2, fused=True, family="wgmma.fed")):
        assert _counted(backward, query, key, key, query, residuals) == (
            2 * rows * heads * pairs * 4 * width, 2 * rows * heads * walked * 5 * width)
