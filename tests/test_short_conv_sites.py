"""Causal short convolutions at an attention layer's keys, values and output (`ShortConvSite`, Inkling 2's;
theme M71)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER, decode_steps, decode_with, model_draw
from reference_error import assert_as_exact

from dew.batch import coordinates
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.mixers import AttentionMixer
from dew.nn.short_conv_site import ShortConvs, ShortConvSite

FEATURES, HEADS, KERNEL, ROWS, LENGTH = 8, 2, 4, 2, 7
SEEDS = {"site": 8, "decode": 8}
"""Check 1's declared seeds for each site of the file."""


def _by_hand(dynamic: bool, dtype):
    """The site's formula in `dtype`: each token's window of its last `KERNEL` inputs (zeros before the
    row), oldest first, times the taps (`conv/weight`, the last the token's own) and, where `dynamic`, each
    token's own taps a head (`dynamic` of `side`), added back to the token and divided by `sqrt(KERNEL + 1)`,
    or `sqrt(2 KERNEL + 1)` with dynamic taps."""
    def run(variables, x, side):
        p = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), variables["params"])
        x = jnp.asarray(x, dtype)
        taps = p["conv"]["weight"][:, 0, :]
        padded = jnp.concatenate([jnp.zeros((ROWS, KERNEL - 1, FEATURES), dtype), x], axis=1)
        windows = jnp.stack([padded[:, j:j + LENGTH] for j in range(KERNEL)], axis=-1)
        weights = taps[None, None]
        if dynamic:
            own = (jnp.asarray(side, dtype) @ p["dynamic"]["kernel"]).reshape(ROWS, LENGTH, HEADS, 1, KERNEL)
            weights = (taps.reshape(HEADS, FEATURES // HEADS, KERNEL)[None, None] + own).reshape(
                ROWS, LENGTH, FEATURES, KERNEL)
        scale = jnp.sqrt(jnp.asarray(2 * KERNEL + 1 if dynamic else KERNEL + 1, dtype))
        return (jnp.sum(windows * weights, axis=-1) + x) / scale
    return jax.jit(run)


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_a_short_conv_site_is_its_formula(dynamic):
    """A site of static taps, or static and per-token dynamic taps a head, computes its formula
    (`_by_hand`, check 1 against it in float64)."""
    site = ShortConvSite(FEATURES, KERNEL, HEADS if dynamic else None)

    def draw(seed):
        rng = np.random.default_rng([0xC0F, seed])
        x = jnp.asarray(rng.normal(size=(ROWS, LENGTH, FEATURES)), jnp.float32)
        side = jnp.asarray(rng.normal(size=(ROWS, LENGTH, 5)), jnp.float32)
        variables = nn.unbox(site.init(jax.random.key(seed), x, dynamic_input=side if dynamic else None))
        variables = jax.tree.map(lambda leaf: jnp.asarray(rng.normal(size=leaf.shape), leaf.dtype), variables)
        return variables, x, side

    def run(variables, x, side):
        return site.apply(variables, x, dynamic_input=side if dynamic else None)
    label = f"a {'dynamic' if dynamic else 'static'} short convolution"
    assert_as_exact(jax.jit(run), _by_hand(dynamic, jnp.float32), draw, label, seeds=SEEDS["site"],
                    truth_fn=_by_hand(dynamic, jnp.float64))


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_a_sites_history_carries_its_rows_real_inputs_alone(dynamic):
    """A site's cached calls: a right-padded prompt, then each row's next token, convolve that token as one
    call over the row's real tokens does, bit for bit; the padding enters no row's history."""
    site = ShortConvSite(FEATURES, KERNEL, HEADS if dynamic else None)
    rng = np.random.default_rng(0xC10)
    x = jnp.asarray(rng.normal(size=(ROWS, 6, FEATURES)), jnp.float32)
    side = jnp.asarray(rng.normal(size=(ROWS, 6, 5)), jnp.float32)
    reals = (3, 5)
    valid = jnp.arange(5)[None] < jnp.asarray(reals)[:, None]
    variables = nn.unbox(site.init(jax.random.key(0), x[:, :5],
                                   dynamic_input=side[:, :5] if dynamic else None))
    params = {"params": variables["params"]}
    cache = {"cache": {"conv_state": jnp.zeros((ROWS, KERNEL - 1, FEATURES), jnp.float32)}}
    prompt = MixerCall(AttentionMetadata(coordinates=coordinates((ROWS, 5), valid=valid, right_padded=True)),
                       decode=True)
    _, held = site.apply({**params, **cache}, x[:, :5], prompt, side[:, :5] if dynamic else None,
                         mutable=["cache"])
    nexts = jnp.stack([x[row, reals[row]] for row in range(ROWS)])[:, None]
    sides = jnp.stack([side[row, reals[row]] for row in range(ROWS)])[:, None]
    stepped, _ = site.apply({**params, **held}, nexts, MixerCall(decode=True), sides if dynamic else None,
                            mutable=["cache"])
    for row, real in enumerate(reals):
        whole = site.apply(params, x[row:row + 1, :real + 1],
                           dynamic_input=side[row:row + 1, :real + 1] if dynamic else None)
        np.testing.assert_array_equal(np.asarray(stepped[row, 0]), np.asarray(whole[0, -1]))


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_a_packed_prompts_last_document_continues_through_the_sites_history(dynamic):
    """A cached prompt packing two documents (four tokens of segment 1, then one of segment 2), then the
    next tokens stated as the second document's (segment 2): they convolve as one call over the second
    document alone does, bit for bit, its history holding none of the first's inputs."""
    site = ShortConvSite(FEATURES, KERNEL, HEADS if dynamic else None)
    rng = np.random.default_rng(0xC11)
    x = jnp.asarray(rng.normal(size=(1, 8, FEATURES)), jnp.float32)
    side = jnp.asarray(rng.normal(size=(1, 8, 5)), jnp.float32)
    variables = nn.unbox(site.init(jax.random.key(0), x[:, :5],
                                   dynamic_input=side[:, :5] if dynamic else None))
    params = {"params": variables["params"]}
    cache = {"cache": {"conv_state": jnp.zeros((1, KERNEL - 1, FEATURES), jnp.float32)}}

    def call(segments):
        segments = jnp.asarray([segments], jnp.int32)
        return MixerCall(AttentionMetadata(coordinates=coordinates(segments.shape, segments=segments)),
                         decode=True)
    _, held = site.apply({**params, **cache}, x[:, :5], call([1, 1, 1, 1, 2]),
                         side[:, :5] if dynamic else None, mutable=["cache"])
    stepped, _ = site.apply({**params, **held}, x[:, 5:], call([2, 2, 2]), side[:, 5:] if dynamic else None,
                            mutable=["cache"])
    whole = site.apply(params, x[:, 4:], dynamic_input=side[:, 4:] if dynamic else None)
    np.testing.assert_array_equal(np.asarray(stepped), np.asarray(whole[:, -3:]))


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_a_gapped_row_convolves_its_real_tokens_alone(dynamic):
    """A row whose valid slots hold gaps (`right_padded` unstated, as native generation states a step's
    rows): each real token convolves the real tokens before it, as a call over them alone does, bit for
    bit."""
    site = ShortConvSite(FEATURES, KERNEL, HEADS if dynamic else None)
    rng = np.random.default_rng(0xC12)
    x = jnp.asarray(rng.normal(size=(1, 7, FEATURES)), jnp.float32)
    side = jnp.asarray(rng.normal(size=(1, 7, 5)), jnp.float32)
    params = nn.unbox(site.init(jax.random.key(0), x, dynamic_input=side if dynamic else None))
    valid = jnp.asarray([[True, False, True, True, False, False, True]])
    laid = MixerCall(AttentionMetadata(coordinates=coordinates(valid.shape, valid=valid)))
    gapped = site.apply(params, x, laid, side if dynamic else None)
    real = np.flatnonzero(np.asarray(valid[0]))
    alone = site.apply(params, x[:, real], dynamic_input=side[:, real] if dynamic else None)
    np.testing.assert_array_equal(np.asarray(gapped[:, real]), np.asarray(alone))


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_a_layer_with_short_convolutions_decodes_as_it_prefills(dynamic):
    """A decoder whose attention convolves its keys and values and whose blocks convolve their mixer's and
    feed-forward's outputs (`LayerKind.mixer_conv`, `feedforward_conv`), decoded a token at a time after a
    prompt of five, on par with its prefill over every token (check 1 against it in float64)."""
    model = CausalTransformer(**{**TINY_DECODER, "vocab_size": 64, "num_kv_heads": 2, "dtype": "float32",
                                 "mixer": AttentionMixer(short_convs=ShortConvs(KERNEL, dynamic=dynamic)),
                                 "layer_types": ("full_attention",) * TINY_DECODER["num_layers"],
                                 "kinds": {"full_attention": LayerKind(mixer_conv=KERNEL,
                                                                       feedforward_conv=KERNEL)}})
    steps = decode_steps(model)

    def decoded(variables, ids):
        return decode_with(steps, variables, ids[:, :5], ids[:, 5:], prefilled=True)
    assert_as_exact(decoded, jax.jit(model.apply), model_draw(model, (2, 9), 64),
                    "a short-convolving layer's decode", seeds=SEEDS["decode"],
                    truth_fn=jax.jit(model.clone(dtype=jnp.float64).apply))


def test_a_convolving_decoder_refuses_forks_and_mixed_steps_before_any_call():
    """A decoder whose blocks convolve (`LayerKind.mixer_conv`) refuses a forked call and a server's mixed
    step, and one whose attention convolves its keys and values refuses a forked call, as their sites read
    each row's own inputs and history: routing reads the refusals before it picks either path, where the
    same decoder without convolutions takes both."""
    def decoder(mixer, kind):
        types = ("full_attention",) * TINY_DECODER["num_layers"]
        return CausalTransformer(**{**TINY_DECODER, "vocab_size": 64, "num_kv_heads": 2, "dtype": "float32",
                                    "mixer": mixer, "layer_types": types, "kinds": {"full_attention": kind}})
    plain = decoder(AttentionMixer(), LayerKind())
    blocks = decoder(AttentionMixer(), LayerKind(mixer_conv=KERNEL))
    keys = decoder(AttentionMixer(short_convs=ShortConvs(KERNEL)), LayerKind())
    assert plain.fork_refusal() is None and plain.mixed_admission_refusal() is None
    assert blocks.fork_refusal() is not None and blocks.mixed_admission_refusal() is not None
    assert keys.fork_refusal() is not None
