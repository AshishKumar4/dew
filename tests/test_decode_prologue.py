"""The decode step's fused attention prologue (`dew.nn.kernels.decode_prologue`)."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.attention import rms_normalized
from dew.nn.kernels import decode_prologue
from dew.nn.kv_cache import write_cache
from dew.nn.rope import apply_rotary, rotary_freqs

ROWS, SLOTS, HEADS, KV, DIM = 8, 32, 4, 2, 32


def unfused(packed, q_weight, k_weight, cos, sin, slots, key_cache, value_cache, scale_after_cast):
    """The XLA steps the kernel replaces, as the attention's unfused decode runs them."""
    def norm(x, weight):
        return rms_normalized(x, weight, 1e-6, jnp.bfloat16, False, scale_after_cast, True)  # noqa: FBT003

    query = norm(packed[:, None, :HEADS], q_weight)
    key = norm(packed[:, None, HEADS:HEADS + KV], k_weight)
    query, key = (apply_rotary(x, cos[:, None], sin[:, None]) for x in (query, key))
    key_cache = write_cache(key_cache, key, slots[:, None])
    value_cache = write_cache(value_cache, packed[:, None, HEADS + KV:], slots[:, None])
    return key_cache, value_cache, query[:, 0].reshape(ROWS, KV, HEADS // KV, DIM).transpose(0, 2, 1, 3)


CUDA = pytest.mark.skipif(jax.default_backend() != "gpu", reason="the kernel is CUDA's")


@CUDA
@pytest.mark.parametrize("scale_after_cast", [True, False])
@pytest.mark.parametrize("weight_dtype", [jnp.bfloat16, jnp.float32])
def test_the_kernel_writes_and_folds_what_the_unfused_steps_do(scale_after_cast, weight_dtype):
    """Norms, rotation, both cache writes and the GQA fold, bitwise the XLA
    steps', for either side of the norm's cast and either weight dtype; a
    row at slot -1 stores nothing."""
    rng = np.random.default_rng(0)
    spread = rng.lognormal(0, 1, (ROWS, 1, 1))
    packed = jnp.asarray(rng.normal(size=(ROWS, HEADS + 2 * KV, DIM)) * spread, jnp.bfloat16)
    q_weight, k_weight = (jnp.asarray(rng.uniform(0.3, 2.0, DIM), weight_dtype) for _ in range(2))
    slots = jnp.asarray([3, -1, 0, 31, 7, 7, 12, -1], jnp.int32)
    cos, sin = rotary_freqs(jnp.maximum(slots, 0)[:, None], DIM, 1e6, dtype=jnp.float32)
    cos, sin = cos[:, 0], sin[:, 0]
    caches = [jnp.asarray(rng.normal(size=(ROWS, SLOTS, KV, DIM)), jnp.bfloat16) for _ in range(2)]

    want = jax.jit(unfused, static_argnums=8)(
        packed, q_weight, k_weight, cos, sin, slots, *caches, scale_after_cast)
    got = jax.jit(lambda *args: decode_prologue.decode_prologue(
        *args, heads=HEADS, epsilon=1e-6, scale_after_cast=scale_after_cast))(
        packed, q_weight, k_weight, cos, sin, slots, *caches)
    for expected, actual in zip(want, got, strict=True):
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint16),
                                      np.asarray(expected).view(np.uint16))


def test_the_kernel_takes_power_of_two_blocks_only():
    """Triton's blocks are powers of two: Qwen3-0.6B's 16 + 2 x 8 heads fit, a
    48-head packing does not and keeps the unfused steps."""
    assert decode_prologue.fits(16, 8, 128)
    assert not decode_prologue.fits(32, 8, 128)
    assert not decode_prologue.fits(16, 8, 96)


@CUDA
def test_a_served_decoder_takes_the_fused_step_with_the_same_draws(monkeypatch):
    """A plain Qwen3-style decoder serving on CUDA runs its decode steps
    through the kernel, one call a layer, and draws the same tokens and
    log-probabilities as with the unfused steps."""
    from test_serving import task

    from dew.inference import TextGeneration
    from dew.inference.serving import Server, _inference_projections
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kv_cache import KVCache
    from dew.sampling import Sampling

    model = CausalTransformer(vocab_size=128, emb_features=64, num_layers=2, num_heads=4, num_kv_heads=2,
                              head_dim=32, mlp="swiglu", mlp_features=128, max_seq_len=64, qk_norm=True,
                              tie_embeddings=True, dtype=jnp.bfloat16)
    variables = model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    # Placement packs a decoder's q, k and v into the one projection the kernel reads.
    variables = _inference_projections(model, jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), variables))

    def served(*, fused: bool):
        with monkeypatch.context() as patched:
            if not fused:
                patched.setattr(decode_prologue, "fits", lambda *_: False)
            served_task = TextGeneration(model, variables, task().processor,
                                         sampling=Sampling(temperature=1.0, eos_id=None))
            server = Server.from_task(served_task, slots=4, capacity=64, kv_cache=KVCache(page_size=None))
            tickets = [server.submit(np.arange(3 + index, 9 + 2 * index, dtype=np.int32), 6, key=index)
                       for index in range(6)]
            server.run()
            return [ticket.result().host() for ticket in tickets]

    monkeypatch.setattr(decode_prologue, "ADOPTED", True)
    calls = []
    original = decode_prologue.decode_prologue
    monkeypatch.setattr(decode_prologue, "decode_prologue",
                        lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))
    fused = served(fused=True)
    assert calls
    plain = served(fused=False)
    for a, b in zip(fused, plain, strict=True):
        for field in dataclasses.fields(a):
            left, right = getattr(a, field.name), getattr(b, field.name)
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
