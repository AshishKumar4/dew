"""Each scenario's learning claim (docs/design/testing.md, §Scenarios: "loss decreases"), judged as a learning
claim is (`learning_gate.assert_learns`) over independent runs: training on a language whose every token
follows its predecessor by one cuts a held-out batch's loss, for the dense decoder, a mixture of experts, a
gated delta hybrid, a masked diffusion LM and a model loaded from a quantized GGUF file. The flow matching,
JEPA, supervised and world-model scenarios hold theirs in their own files."""

import itertools
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from learning_gate import FEWEST_UNITS, assert_learns
from model_support import attending
from scenario_contract import held_out_measures
from tiny_configs import MATRIX_DECODER

from dew.data.dataset import Loader, replayed
from dew.diffusion.discrete import MDLM
from dew.interop import Pretrained
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.lm import LMObjective
from dew.registry import models

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
ROWS, SEQ = 8, 16
LOW, HIGH = 3, 31
"""The language's ids: each row counts up from its first id, wrapping from HIGH back to LOW, ids every
fixture's vocabulary holds."""

RUNS = FEWEST_UNITS
STEPS = 40
"""Each scenario's runs: the fewest the learning gate judges, forty steps of Adam each."""


def counted(starts, width: int = SEQ + 1, high: int = HIGH) -> np.ndarray:
    """The language's rows from their first ids `starts` (`[rows, 1]`): each `width` ids counting up by
    one, wrapping from `high` back to LOW."""
    return (LOW + (np.asarray(starts) - LOW + np.arange(width)) % (high - LOW)).astype(np.int32)


def counting(rows: int, seed) -> np.ndarray:
    """`rows` sequences of SEQ + 1 ids, each counting up by one from a drawn first id, from LOW to HIGH."""
    return counted(np.random.default_rng(seed).integers(LOW, HIGH, size=(rows, 1)))


def text(rows: int, seed, width: int = SEQ + 1) -> dict:
    return {"text": counting(rows, seed)[:, :width]}


def loader(run: int, width: int = SEQ + 1) -> Loader:
    """Run `run`'s stream: a fresh batch of the language each step, from its own seeds."""
    return Loader(replayed(lambda partition: (text(ROWS, [run, step], width) for step in itertools.count())),
                  None, batch=ROWS, records=None)


def judged(objective, label: str, width: int = SEQ + 1) -> None:
    """The scenario's claim, testing.md's "loss decreases", over RUNS independent runs, each on its own
    stream and held-out batch: each run's contrast the share of its held-out loss training removed, positive
    where the loss after is below the loss before, at no margin past that."""
    def lowered(runs):
        before, after = held_out_measures(objective, optax.adam(1e-2), lambda run: loader(run, width),
                                          lambda run: text(64, [run, 10_000], width), runs=runs, steps=STEPS)
        return {"training lowers the held-out loss": 1 - after / before}

    assert_learns(lowered, label, units=RUNS)


def released(directory: str) -> LMObjective:
    """The next-token objective over a released tiny checkpoint's model, from its released weights."""
    source = Pretrained.load(FIXTURES / directory, dtype="float32")
    return LMObjective(source.model, SEQ, variables=jax.device_get(source.variables))


@attending("reference")
def test_a_dense_decoder_learns_its_language():
    """The dense LM scenario: a two-layer decoder's next-token loss."""
    judged(LMObjective(models.build("causal_transformer", **MATRIX_DECODER), SEQ), "a dense decoder")


RELEASED = {"a mixture of experts": "deepseek-v3-tiny", "a gated delta hybrid": "qwen35-tiny"}
"""The scenarios a released tiny checkpoint holds: mixture of experts (DeepSeek-V3's, latent attention over
routed experts) and hybrid and state-space (Qwen3.5's gated delta net layers beside attention)."""


@attending("reference")
@pytest.mark.parametrize("label", RELEASED)
def test_a_released_decoder_learns_its_language(label):
    """The mixture-of-experts and hybrid scenarios: a released decoder's next-token loss, from its released
    weights."""
    judged(released(RELEASED[label]), label)


@attending("reference")
def test_a_masked_diffusion_lm_learns_its_language():
    """The diffusion language model scenario: masked diffusion over a bidirectional decoder, its loss over the
    whole row, masked at the id past the language's (HIGH)."""
    model = models.build("causal_transformer", **MATRIX_DECODER, causal=False)
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=HIGH)(), SEQ, ema_decay=None)
    judged(objective, "a masked diffusion LM", width=SEQ)


@attending("reference")
def test_a_model_loaded_from_a_quantized_file_learns_its_language(tmp_path):
    """The quantized interchange scenario: a llama loaded from a GGUF file of Q8_0 matrices, trained on from
    its dequantized weights."""
    import gguf
    from test_gguf import hf_tensors, write_gguf

    rng = np.random.default_rng(0)
    write_gguf(tmp_path / "model.gguf", "llama", gguf.GGMLQuantizationType.Q8_0, hf_tensors("llama", rng),
               rng)
    source = Pretrained.load(tmp_path, dtype="float32", gguf_file="model.gguf")
    judged(LMObjective(source.model, SEQ, variables=jax.device_get(source.variables)),
           "a quantized file's llama")
