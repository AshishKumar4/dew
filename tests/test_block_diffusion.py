"""Block diffusion: schedule, acceptance, renoise and self-conditioning.

The sampler side of DiffusionGemma needs no model: uniform canvases, a
temperature annealed by the remaining step count, entropy-bound acceptance
and uniform renoising, all against hand-computed cases. The one model piece,
the self-conditioning MLP, meets its reference at fp32 on a tiny fixture
(max |difference| 1.1e-06, tolerance 1e-4); with its gate dead the same
fixture leaves by 3.3, so the branch is live.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.diffusion.block import BlockProcess, sample_canvas
from dew.nn.diffusion_gemma import SelfConditioning, soft_embeddings, translate_weights

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


def process() -> BlockProcess:
    return BlockProcess(canvas_length=6, vocab_size=8)


def test_entropy_bound_accepts_the_confident_positions():
    """Entropies near 0.09, 0.37 and 0.69 under a bound of 0.05: the first
    row always stays (its own excess is zero) while the other two break the
    bound, so the denoiser token lands at position zero alone."""
    current = np.array([[1, 1, 1]], np.int32)
    denoised = np.array([[2, 2, 2]], np.int32)
    logits = np.array([[[2.0, -2.0], [1.0, -1.0], [0.0, 0.0]]], np.float32)
    accepted, mask = BlockProcess(
        canvas_length=3, vocab_size=2, entropy_bound=0.05).accept(
            current, denoised, logits)
    assert mask.tolist() == [[True, False, False]]
    assert accepted.tolist() == [[2, 1, 1]]


def test_renoise_keeps_accepted_positions_and_resamples_the_rest():
    """Accepted ids pass through bit-identical; the rest come back uniform
    over the vocabulary."""
    rerun = process().renoise(jax.random.key(0), np.array([[4, 4, 4, 4]]),
                              np.array([[True, False, True, False]]))
    assert rerun[0, 0] == 4 and rerun[0, 2] == 4
    assert bool(((rerun[0, 1::2] >= 0) & (rerun[0, 1::2] < 8)).all())
    kept = process().renoise(jax.random.key(1), np.array([[4, 4]]),
                             np.array([[True, True]]))
    assert kept.tolist() == [[4, 4]]


@pytest.mark.parametrize("field,value", [
    ("canvas_length", 0), ("vocab_size", 0), ("entropy_bound", 0.0),
    ("max_steps", 0), ("t_min", 0.9),
])
def test_a_process_with_no_sensible_schedule_is_refused(field, value):
    with pytest.raises(ValueError, match=field):
        fields: dict = {"canvas_length": 6, "vocab_size": 8}
        fields[field] = value
        BlockProcess(**fields)


def test_a_certain_denoiser_writes_its_tokens():
    """A stub sure of token 3 everywhere ends all 3s: boundless acceptance
    keeps every draw, and the first step conditions on nothing while later
    steps condition on tempered logits."""
    seen = []

    class Stub:
        process = BlockProcess(canvas_length=6, vocab_size=8, entropy_bound=1e9)

        def __call__(self, canvas, prev):
            seen.append(None if prev is None else True)
            logits = np.full((2, 6, 8), -100.0, np.float32)
            logits[..., 3] = 100.0
            return logits

    out = sample_canvas(jax.random.key(0), Stub(), (2, 6), steps=4)
    assert out.shape == (2, 6) and out.dtype == np.int32
    assert bool((np.asarray(out) == 3).all())
    assert seen[0] is None and all(seen[1:])


def test_self_conditioning_matches_the_reference_implementation():
    """fp32 parity on the tiny self-conditioning MLP, and the gate is live."""
    directory = FIXTURES / "diffusion-gemma-sc-tiny"
    config = json.loads((directory / "config.json").read_text())
    module = SelfConditioning(
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        norm_eps=config["rms_norm_eps"])
    variables = {"params": translate_weights(
        load_file(str(directory / "model.safetensors")))}
    embeds = np.load(directory / "inputs.npy")
    signal = np.load(directory / "signal.npy")
    reference = np.load(directory / "ref.npy")
    assert np.max(np.abs(np.asarray(
        module.apply(variables, embeds, signal)) - reference)) < 1e-4
    dead = dict(variables["params"])
    dead["gate_proj"] = {"kernel": np.zeros_like(dead["gate_proj"]["kernel"])}
    assert np.max(np.abs(np.asarray(
        module.apply({"params": dead}, embeds, signal)) - reference)) > 1.0


def test_soft_embeddings_is_softmax_against_the_table():
    table = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    got = np.asarray(soft_embeddings(
        np.array([[[10.0, 0.0], [0.0, 0.0]]], np.float32), table, 2.0),
        dtype=np.float32)
    np.testing.assert_allclose(got, [[[2.0, 0.0], [1.0, 1.0]]], atol=1e-3)
