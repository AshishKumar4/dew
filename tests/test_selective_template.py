"""Mamba's selective scan on the recurrence template (`templates.recurrence.selective`): its output and every
input's gradient on par from float64 with the portable scan's (`dew.nn.mixers.mamba.selective_scan`, check
1), one row and packed documents, interpreted off a GPU over two channel blocks (a block's view of a wider
row strided in global memory), and at Falcon Mamba 7B's widths on a GPU (`tests/test_gpu_recurrence.py`);
and a Mamba layer through its selector under the template's choice (`recurrence_chunk.SELECTIVE`) on par
with the layer's portable scan, at the mixer's own dtypes (its scan in fp32), with packed documents and
with a held state."""
import functools

import jax
import jax.numpy as jnp
import pytest
from reference_error import FEWEST_SEEDS, assert_as_exact, with_gradients

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.kernels import generation
from dew.nn.kernels.templates import bundles, mosaic_sm90
from dew.nn.kernels.templates.recurrence import selective
from dew.nn.mixers.mamba import Mamba, selective_scan as portable

pytestmark = pytest.mark.kernel

INTERPRETED = jax.default_backend() != "gpu"
"""The template interprets off a GPU, by the wgmma bundle, whose rings and barriers the interpreter models."""

SHAPES = {"interpreted": (1, 70, 256, 16), "falcon-mamba-7b": (2, 2048, 8192, 16)}
"""Each case's rows, tokens, channels and states: a ragged chunk over two channel blocks, and Falcon Mamba
7B's widths."""


def draw(dtype, seed: int, rows: int, length: int, channels: int, states: int, packed: bool):
    """A draw's inputs, their outputs' cotangents last: steps after a softplus, decays `-exp(N(0, 1/4))`,
    input and readout gates normal, `x`, `dt`, `B`, `C` and the cotangents in `dtype`, documents starting at a
    third and two thirds of each row where `packed`."""
    keys = jax.random.split(jax.random.key(seed), 7)
    x = jax.random.normal(keys[0], (rows, length, channels)).astype(dtype)
    dt = jax.nn.softplus(jax.random.normal(keys[1], (rows, length, channels)) - 3.0).astype(dtype)
    a = -jnp.exp(0.5 * jax.random.normal(keys[2], (channels, states)))
    b, c = (jax.random.normal(key, (rows, length, states)).astype(dtype) for key in keys[3:5])
    skip = jax.random.normal(keys[5], (channels,))
    starts = None
    if packed:
        starts = jnp.zeros((rows, length), bool).at[:, (0, length // 3, 2 * length // 3)].set(True)
    dy = jax.random.normal(keys[6], (rows, length, channels)).astype(dtype)
    return (x, dt, a, b, c, skip), starts, dy


def scanned(fn, starts):
    """`fn`'s output alone, its documents' starts `starts`."""
    return lambda *inputs: fn(*inputs, starts=starts)[0]


def the_template_is_as_exact_as_the_portable_scan(case: str, packed: bool, dtype=jnp.bfloat16):
    """`case`'s output and every input's gradient through the template on par from float64 with the portable
    scan's, over check 1's fewest draws."""
    shape = SHAPES[case]
    family = mosaic_sm90.WGMMA if INTERPRETED else bundles.of(generation.device_generation(), dtype)
    template = functools.partial(selective.selective_scan, family=family, interpret=INTERPRETED)

    def draws(seed):
        inputs, _, dy = draw(dtype, seed, *shape, packed)
        return (*inputs, dy)

    starts = draw(dtype, 0, *shape, packed)[1]
    assert_as_exact(jax.jit(with_gradients(scanned(template, starts))),
                    jax.jit(with_gradients(scanned(portable, starts))), draws,
                    f"the selective template, {case}{', packed' if packed else ''}", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(with_gradients(scanned(portable, starts))))


@pytest.mark.parametrize("packed", [False, True], ids=["one row", "packed documents"])
def test_the_template_is_as_exact_as_the_portable_scan_interpreted(packed):
    the_template_is_as_exact_as_the_portable_scan("interpreted", packed)


LAYER = {"emb_features": 64, "intermediate_size": 128, "state_size": 16, "time_step_rank": 8,
         "chunk_size": 32}
"""A Mamba layer one channel block wide (`recurrence_chunk.SELECTIVE_BLOCK`), Mamba's 16 states."""


def the_layer_is_as_exact_under_the_template(packed: bool, held: bool, *, length: int = 70,
                                             dtype=jnp.bfloat16):
    """A Mamba layer in `dtype` (its scan widened to fp32, as the mixer runs it) over two rows of `length`
    tokens, through its selector with the template chosen (`generation.chosen`), on par from float64 with
    the layer's portable scan, output and every input's and parameter's gradient, over check 1's fewest
    draws: documents starting a third and two thirds in where `packed` (`MixerCall`'s segments), and the
    scan from a held state where `held` (the decode cache a first call of `length // 2` tokens leaves, the
    second call's tokens alone compared)."""
    layer = Mamba(**LAYER, dtype=dtype)
    variables = layer.init(jax.random.key(1), jnp.zeros((2, length, LAYER["emb_features"]), dtype))
    metadata = None
    if packed:
        starts = jnp.zeros((2, length), jnp.int32).at[:, (length // 3, 2 * length // 3)].set(1)
        metadata = AttentionMetadata(coordinates=coordinates(segments=jnp.cumsum(starts, axis=1)))
    call = MixerCall(metadata, decode=held)

    def run(module, chosen):
        def mixed(variables, x):
            with chosen():
                if not held:
                    return module.apply(variables, x, call)
                first = length // 2
                allocated = module.apply(variables, x[:, :1], call, mutable=["cache"])[1]
                _, cache = module.apply({**variables, **allocated}, x[:, :first], call, mutable=["cache"])
                return module.apply({**variables, **cache}, x[:, first:], call, mutable=["cache"])[0]

        return mixed

    def template_chosen():
        return generation.chosen(selective_scan="mosaic_gpu")

    def portable_chosen():
        return generation.chosen(selective_scan="xla")

    def draws(seed):
        keys = jax.random.split(jax.random.key(seed))
        x = jax.random.normal(keys[0], (2, length, LAYER["emb_features"])).astype(dtype)
        out = (2, length - length // 2 if held else length, LAYER["emb_features"])
        return variables, x, jax.random.normal(keys[1], out).astype(dtype)

    truth = layer.clone(dtype=jnp.float64)
    assert_as_exact(jax.jit(with_gradients(run(layer, template_chosen))),
                    jax.jit(with_gradients(run(layer, portable_chosen))), draws,
                    f"a Mamba layer under the template{', packed' if packed else ''}"
                    f"{', held' if held else ''}",
                    seeds=FEWEST_SEEDS, truth_fn=jax.jit(with_gradients(run(truth, portable_chosen))))


@pytest.mark.parametrize("packed, held", [(False, False), (True, False), (False, True)],
                         ids=["one row", "packed documents", "a held state"])
def test_a_mamba_layer_is_as_exact_under_the_template_interpreted(packed, held):
    the_layer_is_as_exact_under_the_template(packed, held)


REGIMES = {
    "decays near one": {"steps": 1e-3},
    "decays near zero": {"steps": 1e3},
    "saturated steps": {"steps": None},
    "resets on chunk edges": {"resets": 32},
    "a long sequence": {"length": 8192},
}
"""Where a scan's recurrence is hardest to hold, each as the draw it changes: steps a thousandth or a thousand
times the draw's (every decay `exp(dt A)` near one, or near zero), steps saturated at 10 (None), documents
starting at every chunk's first token (`resets` tokens apart), and a sequence of 8192 tokens."""


def the_template_holds_a_regime(regime: str, shape: tuple, dtype=jnp.float32):
    """The template's output and every input's gradient on par from float64 with the portable scan's over
    check 1's fewest draws, in `regime` (`REGIMES`) at `shape`'s rows, tokens, channels and states."""
    change = REGIMES[regime]
    rows, length, channels, states = shape
    length = change.get("length", length)
    family = mosaic_sm90.WGMMA if INTERPRETED else bundles.of(generation.device_generation(), dtype)
    template = functools.partial(selective.selective_scan, family=family, interpret=INTERPRETED)
    starts = None
    if "resets" in change:
        starts = jnp.zeros((rows, length), bool).at[:, ::change["resets"]].set(True)

    def draws(seed):
        (x, dt, a, b, c, skip), _, dy = draw(dtype, seed, rows, length, channels, states, packed=False)
        if "steps" in change:
            scaled = jnp.full_like(dt, 10.0) if change["steps"] is None else dt * change["steps"]
            dt = scaled.astype(dtype)
        return x, dt, a, b, c, skip, dy

    assert_as_exact(jax.jit(with_gradients(scanned(template, starts))),
                    jax.jit(with_gradients(scanned(portable, starts))), draws,
                    f"the selective template, {regime}", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(with_gradients(scanned(portable, starts))))


@pytest.mark.parametrize("regime", ["decays near zero", "resets on chunk edges"])
def test_the_template_holds_its_hardest_regimes_interpreted(regime):
    """Two of the regimes interpreted, at the interpreted shape; every regime runs at Falcon Mamba 7B's widths
    on a GPU (`tests/test_gpu_recurrence.py`)."""
    the_template_holds_a_regime(regime, SHAPES["interpreted"])
