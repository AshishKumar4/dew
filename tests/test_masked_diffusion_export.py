"""Trained source-format exports of the masked-diffusion decoder families.

`load_pretrained` binds every tensor of a LLaDA or Dream checkpoint to the
leaf it loaded into, so a trained model writes back into the source's own
tensor names beside the config it came with rather than a config derived
from the built model. These cases run that path end to end on the committed
tiny checkpoints: one real `Trainer` step of plain SGD under
`MaskedDiffusionObjective` started from the loaded weights,
`Pretrained.save`, then the export read back by `load_pretrained` and by
transformers 5.16.1 on the same ids.

tools/masked_diffusion_export_reference.py owns the pipeline and prints the
numbers:

    JAX_PLATFORMS=cpu PYTHONPATH=src python tools/masked_diffusion_export_reference.py

Observed on CPU in fp32, every position's argmax equal and the tolerance
1e-4 on the trained export:

| family | source tensors | logit magnitude | max abs logit difference |
| ------ | -------------- | --------------- | ------------------------ |
| llada  |             21 |            0.89 |                  6.9e-07 |
| dream  |             27 |            4.86 |                  3.3e-06 |

transformers 5.16.1 carries no LLaDA or Dream class; both releases ship
remote code. It carries the block each of them is, so the reference is stock
`LlamaForCausalLM` for LLaDA and `Qwen2ForCausalLM` for Dream over the
export's own tensors, run with an all-visible 4-D mask for the bidirectional
reading both releases hard-code. LLaDA's tensors are renamed onto the llama
layout for that load; Dream's already are the qwen2 layout. The reference's
loading report has to name no missing, unexpected or mismatched tensor.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from dew.interop import load_pretrained
from dew.interop.hf_decoders import save_pretrained_decoder
from tests.test_masked_diffusion import flat
from tools import masked_diffusion_export_reference as tool

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "hf"
LOGITS = 1e-4
MOVEMENT = 1e-4

CASES = {case.name: case for case in tool.CASES}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("masked-diffusion-export")


@pytest.fixture(scope="module")
def trips(workspace):
    """Each family trained and exported once, shared by every case here."""
    cache = {}

    def trip(name: str):
        if name not in cache:
            cache[name] = tool.round_trip(CASES[name], workspace)
        return cache[name]

    return trip


@pytest.fixture(params=list(CASES), ids=list(CASES))
def trip(request, trips):
    return trips(request.param)


def test_every_source_tensor_is_bound_and_written_back_under_its_own_name(trip):
    """No tensor of the checkpoint is renamed or dropped on the way out. The
    llama spelling the shared leaf map reads LLaDA through is an internal
    detail of the load; what lands on disk is the release's own table."""
    bound = [layout.name for layout in trip.source.weight_layouts]

    assert len(bound) == len(set(bound)), "a source tensor is bound twice"
    assert set(bound) == set(trip.source_tensors), "a source tensor is unbound"
    assert not trip.source.retained_tensors
    assert set(trip.exported_tensors) == set(trip.source_tensors)
    for name, tensor in trip.source_tensors.items():
        assert trip.exported_tensors[name].shape == tensor.shape, name


def test_the_export_carries_the_trained_weights_not_the_loaded_ones(trip):
    """One SGD step of the masked-diffusion NELBO moves the embedding, the
    attention and MLP projections, the norms and the head, and it is the
    moved values the source layout writes back."""
    distances = tool.moved(trip)

    assert set(distances) >= {"embedding", "attention", "mlp", "norm", "head"}
    for kind, distance in distances.items():
        assert distance > MOVEMENT, f"{kind} moved {distance:.3e}"


def test_the_trained_export_reloads_leaf_for_leaf_and_recomputes_the_logits(trip):
    """`load_pretrained` reads the export back into the same model with the
    trained values bit for bit, so it computes the same logits."""
    assert trip.reloaded.model == trip.source.model, "the export rebuilds a different model"

    held, again = flat(trip.trained), flat(trip.reloaded.variables)
    assert held.keys() == again.keys()
    for name, leaf in again.items():
        assert np.array_equal(np.asarray(leaf), np.asarray(held[name])), name
    np.testing.assert_array_equal(
        tool.logits(trip.reloaded, trip.reloaded.variables, trip.ids), trip.ours)


def test_transformers_reads_the_trained_export(trip):
    """The export is a complete checkpoint for the reference implementation:
    nothing missing, unexpected or mismatched, same ids, same argmax, and
    the logits agree to 1e-4."""
    assert dict(trip.report) == {"missing_keys": [], "unexpected_keys": [],
                                 "mismatched_keys": [], "error_msgs": []}
    assert np.array_equal(np.argmax(trip.theirs, -1), np.argmax(trip.ours, -1))
    difference = float(np.max(np.abs(trip.theirs - trip.ours)))
    assert difference < LOGITS, f"max |logit difference| {difference:.3e}"


def test_the_export_keeps_the_sources_own_config(trip):
    """The source's config is retained, not rebuilt from the model: a derived
    config would drop the aliases and the training flags the release carries
    and that the loader reads past."""
    directory = FIXTURES / trip.case.fixture

    assert (json.loads((trip.export / "config.json").read_text())
            == json.loads((directory / "config.json").read_text()))


def test_the_bidirectional_reading_is_what_the_reference_agrees_with(trip):
    """The reference is the causal block with the mask lifted, which is the
    forward both releases hard-code. Reading the same export causally moves
    the logits well past the tolerance, so the agreement above is evidence
    about the attention and not only about the weights."""
    causal = trip.reloaded.model.clone(causal=True)
    theirs = np.asarray(causal.apply(trip.reloaded.variables, trip.ids), np.float32)

    difference = float(np.max(np.abs(theirs - trip.ours)))
    assert difference > LOGITS, f"a causal read moved the logits {difference:.3e}"


@pytest.mark.parametrize("fixture", ["llada-tiny", "dream-tiny"])
def test_the_derived_config_writer_round_trips_each_family(fixture, tmp_path):
    """The other writer: `save_pretrained_decoder` derives the config from a
    built model, for a masked-diffusion model that came from no checkpoint.
    It has to write names its own family reads back, so LLaDA's leaves go
    out under the release's OLMo-style spellings and Dream's config carries
    the split o_proj bias its reference builds."""
    source = load_pretrained(str(FIXTURES / fixture), dtype="float32",
                             attention_impl="reference")

    save_pretrained_decoder(source.model, source.variables, tmp_path)
    written = json.loads((tmp_path / "config.json").read_text())
    reloaded = load_pretrained(str(tmp_path), dtype="float32", attention_impl="reference")

    assert written["mask_token_id"] == source.model.mask_token_id
    assert reloaded.model == source.model, "the derived config rebuilds a different model"
    held, again = flat(source.variables), flat(reloaded.variables)
    assert held.keys() == again.keys()
    for name, leaf in again.items():
        assert np.array_equal(np.asarray(leaf), np.asarray(held[name])), name
