"""The template's query and key prologues (`Hooks.query_mod`, `Hooks.key_mod`: kernel-templates.md's forward,
row 5743): QK-norm and RoPE as a Qwen3 block writes them, taken where each mainloop loads its operands or as
XLA's program before the launch, and their pullbacks past the kernels, on par with the prologue written out
in plain JAX before the reference (check 1)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS
from reference_error import assert_as_exact, with_gradients
from test_attention_mosaic import SEEDS, WGMMA, _device, _pinned

from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_mosaic, attention_route

ROWS, TOKENS, HEADS, KV_HEADS = 1, 128, 2, 1
"""A call's rows, tokens and heads, its draws each case's `SEEDS`: check 1's fewest (test_attention_mosaic's,
`reference_error.seeds_for`)."""
EPSILON = 1e-6


def normed(x, positions, weight):
    """QK-norm as a model writes it: the row's root mean square over its width in fp32 at least, times its
    weight, in the operand's dtype; the positions unread."""
    del positions
    wide = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    return (wide * jax.lax.rsqrt(jnp.mean(wide * wide, axis=-1, keepdims=True) + EPSILON)
            * weight).astype(x.dtype)


def rotated(x, positions, weight, frequencies):
    """QK-norm, then RoPE's rotation of the width's halves at each row's position, `frequencies` `[1, D]`
    the halves' frequencies twice over (Qwen3's order), in the operand's dtype."""
    wide = normed(x, positions, weight)
    wide = wide.astype(jnp.promote_types(wide.dtype, jnp.float32))
    half = x.shape[-1] // 2
    angles = positions.astype(wide.dtype) * frequencies[:, :half].astype(wide.dtype)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    first, second = wide[:, :half], wide[:, half:]
    return jnp.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1).astype(x.dtype)


def weighed(x, positions, weight, frequencies):
    """A light prologue: each entry times its width's weight and its row's position's term, an eighth added,
    in the operand's dtype (a tile's few fp32 values live at once, which a consumer holds where it loads the
    query at its own config; the eighth makes it no function a power of two passes through, so a launch's
    prescale taken before it differs); the frequencies unread."""
    del frequencies
    wide = jnp.promote_types(x.dtype, jnp.float32)
    taken = 1.0 + positions.astype(wide) / 1024.0
    return (x.astype(wide) * weight.astype(wide) * taken + 0.125).astype(x.dtype)


def rounded(x, positions, weight, frequencies):
    """A light prologue in its operands' own arithmetic: each entry times its width's weight twice, each
    product in the promotion of the operand's and the weight's dtypes (bf16 both, `ROUNDED`, so each product
    rounds), so a weight a launch widened to fp32 rounds once where the reference rounds twice; the positions
    and frequencies unread."""
    del positions, frequencies
    return (x * weight) * weight


PROLOGUES = {"weighed": weighed, "rotated": rotated, "rounded": rounded}
"""The prologues the Mosaic forward is checked under: light ones its consumers take where they load the
query, and Qwen3's QK-norm and RoPE, which its derivation places where its consumers hold it at their own
config, else before the launch."""

ROUNDED = {"rounded": jnp.bfloat16}
"""The prologues whose weights a call passes narrower than fp32, each its dtype."""


def frequencies(width: int) -> np.ndarray:
    """RoPE's frequencies at base 10,000 for a head `width` wide, `[1, D]` fp32, the halves' twice over: a
    constant of the call, as a model holds it."""
    half = (10_000.0 ** (-np.arange(width // 2) / (width // 2))).astype(np.float32)
    return np.concatenate([half, half])[None, :]


def written_out(operand: jax.Array, weight: jax.Array, frequency: jax.Array) -> jax.Array:
    """`rotated` over a whole `[B, T, H, D]` operand in plain JAX, no hook machinery: each token at its
    index along the sequence, each head reading `weight` `[1 or H, D]`."""
    wide = operand.astype(jnp.promote_types(operand.dtype, jnp.float32))
    scale = jax.lax.rsqrt(jnp.mean(wide * wide, axis=-1, keepdims=True) + EPSILON)
    normed_ = (wide * scale * weight[None, None]).astype(operand.dtype).astype(wide.dtype)
    half = operand.shape[-1] // 2
    angles = (jnp.arange(operand.shape[1], dtype=wide.dtype)[None, :, None, None]
              * frequency[0, :half].astype(wide.dtype))
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    first, second = normed_[..., :half], normed_[..., half:]
    return jnp.concatenate([first * cos - second * sin, second * cos + first * sin],
                           axis=-1).astype(operand.dtype)


def written_reference(spans, width: int):
    """The reference over the query and keys under QK-norm and RoPE written out in plain JAX (`written_out`),
    as `(query, key, value, query_weight, key_weight)` takes them."""
    def reference(query, key, value, query_weight, key_weight):
        frequency = frequencies(width)
        return template.reference(written_out(query, query_weight, frequency),
                                  written_out(key, key_weight, frequency), value, spans)

    return reference


def hooks(query_weight, key_weight, width: int, prologue=rotated) -> template.Hooks:
    """The query's and key's prologues (`rotated` by default), each its own weight `[1, D]` and the
    frequencies, width parameters (`template.Kind.WIDTH`)."""
    frequency = template.Param(frequencies(width), template.Kind.WIDTH)
    return template.Hooks(query_mod=prologue, key_mod=prologue,
                          query_params=(template.Param(query_weight, template.Kind.WIDTH), frequency),
                          key_params=(template.Param(key_weight, template.Kind.WIDTH), frequency))


def drawing(dtype, width: int):
    """A draw `(query, key, value, query_weight, key_weight, dout)` by seed: unit Gaussians in `dtype`, the
    norms' weights about one in fp32."""
    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 6)
        shapes = ((ROWS, TOKENS, HEADS, width), (ROWS, TOKENS, KV_HEADS, width),
                  (ROWS, TOKENS, KV_HEADS, width))
        operands = [jax.random.normal(stream, shape, jnp.float32).astype(dtype)
                    for stream, shape in zip(streams[:3], shapes, strict=True)]
        weights = [1.0 + 0.1 * jax.random.normal(stream, (1, width), jnp.float32) for stream in streams[3:5]]
        dout = jax.random.normal(streams[5], (ROWS, TOKENS, HEADS, width), jnp.float32).astype(dtype)
        return (*operands, *weights, dout)

    return draw


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=lambda dtype: jnp.dtype(dtype).name)
def test_a_prologued_route_call_is_on_par_with_its_prologues_written_out_forward_and_backward(dtype):
    """The route's call under a query and a key prologue (QK-norm, then RoPE), on the CPU's mainloop: its
    output and its gradients of the query, key, value and both norms' weights on par with the prologues
    written out in plain JAX before the reference (check 1)."""
    width = 32
    spans = template.spans(TOKENS, TOKENS, ROWS, causal=True)

    def ours(query, key, value, query_weight, key_weight):
        return attention_route.attend(query, key, value, spans, hooks(query_weight, key_weight, width))

    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(written_reference(spans, width))),
                    drawing(dtype, width), f"prologued route {jnp.dtype(dtype).name}", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("prologue", PROLOGUES)
@pytest.mark.parametrize("training", [template.Kept.NONE, template.Kept.SPLIT], ids=["plain", "training"])
@pytest.mark.parametrize(("family", "generation_name", "width"), [
    (WGMMA, "sm90", 64), ("tcgen05.chunks", "sm100", 64), ("tcgen05.chunks", "sm100", 128)])
def test_the_mosaic_forward_takes_its_query_prologue_where_it_loads_the_query_interpreted(
        family, generation_name, width, training, prologue):
    """The Mosaic forward under a query and a key prologue (`PROLOGUES`), its consumers taking the query's
    on each query tile where they load it where they hold it and the key's taken before the launch, under
    the interpreter on wgmma and on
    tcgen05's chunks: its output on par with the reference's, which
    takes them over the whole operands (check 1)."""
    spans = template.spans(TOKENS, TOKENS, ROWS, causal=True)
    tiles = attention_mosaic.Tiles(64, 64, 2, 2, family=family)
    target = generation.gpu_target(generation_name)

    def ours(query, key, value, query_weight, key_weight):
        with _device(interpret=True, target=target), _pinned(tiles):
            out = attention_mosaic.forward(query, key, value, spans, given(query_weight, key_weight),
                                           residuals=training, interpret=True)
        return out if training is template.Kept.NONE else out[0]

    def reference(query, key, value, query_weight, key_weight):
        return template.reference(query, key, value, spans, given(query_weight, key_weight))

    def given(query_weight, key_weight):
        dtype = ROUNDED.get(prologue, jnp.float32)
        return hooks(query_weight.astype(dtype), key_weight.astype(dtype), width, PROLOGUES[prologue])

    def draw(seed):
        return drawing(jnp.bfloat16, width)(seed)[:5]

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"mosaic {training.name.lower()} forward {prologue} {family} {width}", seeds=SEEDS)


@COMPILES_ON_CPU
@pytest.mark.parametrize("prologue", PROLOGUES)
@pytest.mark.parametrize("kept", [template.Kept.SPLIT, template.Kept.NONE], ids=["training", "inference"])
@pytest.mark.parametrize(("generation_name", "tokens", "heads", "kv_heads", "width"),
                         [("sm90", 4096, 32, 8, 128), ("sm100", 12288, 16, 4, 256)],
                         ids=["qwen3-8b-h100", "4b-b200"])
def test_a_prologued_forward_compiles_spill_free_at_its_derived_config(generation_name, tokens, heads,
                                                                       kv_heads, width, kept, prologue):
    """An attention forward under its query and key prologues (`PROLOGUES`) over causal tokens, at the config
    its derivation gives on a compile-only H100 (Qwen3-8B's 32 heads over 8, 128 wide, 4,096 tokens) and
    B200 (the 4B decision call's 16 over 4, 256 wide, 12,288 tokens): every launch compiled by ptxas spilling
    nothing, the query prologue taken in it where it holds it (`mosaic_failures`)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    spans = template.spans(tokens, tokens, 1, causal=True)
    query = jax.ShapeDtypeStruct((1, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, tokens, kv_heads, width), jnp.bfloat16)
    weight = jax.ShapeDtypeStruct((1, width), jnp.float32)

    def forward(query, key, value, query_weight, key_weight):
        return attention_mosaic.forward(query, key, value, spans,
                                        hooks(query_weight, key_weight, width, PROLOGUES[prologue]),
                                        residuals=kept)

    with generation.assumed(generation_name):
        case = KernelCase("prologued forward", lambda: (forward, (query, key, key, weight, weight)))
        found = compile_mosaic_kernels([case], generation_name)
    failed = mosaic_failures(found)
    assert not failed, "\n".join(failed)


def by_dtype(x, positions, weight, frequencies):
    """`rotated` on a bf16 operand and the operand times its weight on any other: a prologue whose program
    depends on its operand's dtype, so one differentiated at another dtype than it ran at is another
    function's derivative."""
    if x.dtype == jnp.bfloat16:
        return rotated(x, positions, weight, frequencies)
    return x * weight.astype(x.dtype)


def test_a_prologued_route_call_over_narrowed_operands_differentiates_the_prologue_it_ran():
    """The route's call over fp32 operands it narrows to bf16 (`attend`'s `operand_dtype`; bf16 values, so
    the narrowing loses none) under a key and query prologue whose program depends on its operand's dtype
    (`by_dtype`): its output and its gradients of the query, keys, values and both weights on par with QK-norm
    and RoPE written out over the bf16 operands before the reference (check 1): the prologues' pullbacks
    taken at the operands the forward took them at."""
    width = 32
    spans = template.spans(TOKENS, TOKENS, ROWS, causal=True)

    def ours(query, key, value, query_weight, key_weight):
        frequency = template.Param(frequencies(width), template.Kind.WIDTH)
        hooked = template.Hooks(
            query_mod=by_dtype, query_params=(template.Param(query_weight, template.Kind.WIDTH), frequency),
            key_mod=by_dtype, key_params=(template.Param(key_weight, template.Kind.WIDTH), frequency))
        wide = [operand.astype(jnp.float32) for operand in (query, key, value)]
        return attention_route.attend(*wide, spans, hooked, operand_dtype=jnp.bfloat16).astype(query.dtype)

    assert_as_exact(jax.jit(with_gradients(ours)), jax.jit(with_gradients(written_reference(spans, width))),
                    drawing(jnp.bfloat16, width), "prologued route narrowed", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize(("dtype", "side"), [(jnp.bfloat16, "key"), (jnp.bfloat16, "both"),
                                             (jnp.float32, "both")], ids=["key", "both", "fp32"])
def test_the_certify_form_takes_its_prologues_as_the_forward_does_interpreted(dtype, side):
    """The forward's certify form (`attention_mosaic.certified`) under a key prologue alone, and beside a
    query prologue, interpreted on a Hopper's tiles: its output on par with the reference's under the same
    prologues (check 1), its operands taken under them as the forward takes them (`key_prologue_taken`,
    `_forward_call`); an fp32 call, whose launch takes its operands as 16-bit parts and its query prologue
    before them, refused by the certify form, which walks 16-bit operands alone (`certified_walk`), not run
    over another call's parts."""
    width = 64
    spans = template.spans(TOKENS, TOKENS, ROWS, causal=True)

    def given(query_weight, key_weight):
        hooked = hooks(query_weight, key_weight, width, weighed)
        return hooked if side == "both" else dataclasses.replace(hooked, query_mod=None, query_params=())

    def ours(query, key, value, query_weight, key_weight):
        with generation.assumed("sm90"):
            _, (out, _, _) = attention_mosaic.certified(query, key, value, spans,
                                                        given(query_weight, key_weight), interpret=True)
        return out

    def reference(query, key, value, query_weight, key_weight):
        return template.reference(query, key, value, spans, given(query_weight, key_weight))

    def draw(seed):
        return drawing(dtype, width)(seed)[:5]

    if dtype == jnp.float32:
        with pytest.raises(NotImplementedError):
            ours(*draw(0))
        return
    assert_as_exact(ours, jax.jit(reference), draw, f"certified {side} prologue {jnp.dtype(dtype).name}",
                    seeds=SEEDS)


def test_a_prologues_parameters_without_its_prologue_are_refused():
    """A query or key prologue's parameters given with no prologue to read them (`Hooks.query_mod`,
    `key_mod` None) are refused where the hooks are built, as no prologue would differentiate along them."""
    weight = template.Param(jnp.ones((1, 32), jnp.float32), template.Kind.WIDTH)
    for side in ("query", "key"):
        with pytest.raises(ValueError):
            template.Hooks(**{f"{side}_params": (weight,)})


@INTERPRETS
@pytest.mark.parametrize("prologue", ["weighed", "rounded"])
def test_a_query_prologue_taken_in_the_launch_is_bitwise_the_one_taken_before_it_interpreted(prologue):
    """The Mosaic forward with its query prologue taken where its consumers load the query
    (`attention_mosaic_forward._prologued`) and with the same prologue taken over the whole query before the
    launch (`attention.prologued`), at the same pinned tiles on tcgen05's chunks: bitwise the same output,
    the prologue's values, its parameters' dtypes (`ROUNDED`'s bf16 weights, each product rounded) and its
    place before the launch's power of two alike."""
    width = 64
    spans = template.spans(TOKENS, TOKENS, ROWS, causal=True)
    tiles = attention_mosaic.Tiles(64, 64, 2, 2, family="tcgen05.chunks")
    target = generation.gpu_target("sm100")
    query, key, value, query_weight, key_weight = drawing(jnp.bfloat16, width)(0)[:5]
    dtype = ROUNDED.get(prologue, jnp.float32)
    hooked = hooks(query_weight.astype(dtype), key_weight.astype(dtype), width, PROLOGUES[prologue])
    taken = dataclasses.replace(hooked, query_mod=None, query_params=())
    with _device(interpret=True, target=target), _pinned(tiles):
        fused = attention_mosaic.forward(query, key, value, spans, hooked, interpret=True)
        before = attention_mosaic.forward(template.prologued(query, hooked.query_mod, hooked.query_params),
                                          key, value, spans, taken, interpret=True)
    np.testing.assert_array_equal(np.asarray(fused, np.float32), np.asarray(before, np.float32))
