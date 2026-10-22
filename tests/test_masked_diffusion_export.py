"""Trained source-format exports of the masked-diffusion decoder families.

`Pretrained.load` binds every tensor of a LLaDA or Dream checkpoint to the
leaf it loaded into, so a trained model writes back into the source's own
tensor names beside the config it came with rather than a config derived
from the built model. These cases run that path end to end on the committed
tiny checkpoints: one real `Trainer` step of plain SGD under
`MaskedDiffusionObjective` started from the loaded weights,
`Pretrained.save`, then the export read back by `Pretrained.load` and by
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
from interop_support import assert_restores
from model_support import cached_round_trips, flat_tree

from dew.data import Memory
from dew.data.compile import compile
from dew.data.pipeline import Pipeline
from dew.interop import Pretrained, PretrainedDecoder
from tools import masked_diffusion_export_reference as tool

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "hf"
LOGITS = 1e-4
CASES = {case.name: case for case in tool.CASES}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("masked-diffusion-export")


@pytest.fixture(scope="module")
def trips(workspace):
    """Each family trained and exported once, shared by every case here."""
    return cached_round_trips(tool.round_trip, CASES, workspace)


@pytest.fixture(params=list(CASES), ids=list(CASES))
def trip(request, trips):
    return trips(request.param)


def test_the_trained_export_reloads_leaf_for_leaf_and_recomputes_the_logits(trip):
    """`Pretrained.load` reads the export back into the same model with the
    trained values bit for bit, so it computes the same logits."""
    assert_restores(trip.source.model, trip.trained, trip.reloaded, trip.ids)


def test_transformers_reads_the_trained_export(trip):
    """The export is a complete checkpoint for the reference implementation:
    nothing missing, unexpected or mismatched, same ids, same argmax, and
    the logits agree to 1e-4."""
    assert dict(trip.report) == {"missing_keys": [], "unexpected_keys": [],
                                 "mismatched_keys": [], "error_msgs": []}
    assert np.array_equal(np.argmax(trip.theirs, -1), np.argmax(trip.ours, -1))
    difference = float(np.max(np.abs(trip.theirs - trip.ours)))
    assert difference < LOGITS, f"max |logit difference| {difference:.3e}"


@pytest.mark.parametrize("fixture", ["llada-tiny", "dream-tiny"])
def test_the_derived_config_writer_round_trips_each_family(fixture, tmp_path):
    """The other writer: `PretrainedDecoder.from_model` derives the config from a
    built model, for a masked-diffusion model that came from no checkpoint.
    It has to write names its own family reads back, so LLaDA's leaves go
    out under the release's OLMo-style spellings and Dream's config carries
    the split o_proj bias its reference builds."""
    source = Pretrained.load(str(FIXTURES / fixture), dtype="float32",
                             attention_impl="reference")

    PretrainedDecoder.from_model(source.model, source.variables).save(tmp_path)
    written = json.loads((tmp_path / "config.json").read_text())
    reloaded = Pretrained.load(str(tmp_path), dtype="float32", attention_impl="reference")

    assert written["mask_token_id"] == source.model.mask_token_id
    assert reloaded.model == source.model, "the derived config rebuilds a different model"
    held, again = flat_tree(source.variables), flat_tree(reloaded.variables)
    assert held.keys() == again.keys()
    for name, leaf in again.items():
        assert np.array_equal(np.asarray(leaf), np.asarray(held[name])), name
    prompt = np.load(FIXTURES / fixture / 'input_ids.npy')[:1, :4]
    generated = reloaded.text_generation()(prompt, 2, key=7, steps=2).host().tokens
    np.testing.assert_array_equal(generated[:, :4], prompt)
    assert generated.shape == (1, 6) and np.all(generated[:, 4:] != source.model.mask_token_id)


@pytest.mark.parametrize("fixture", ["llada-tiny", "dream-tiny"])
def test_a_masked_diffusion_run_exports_and_reloads_as_a_masked_decoder(fixture, tmp_path):
    """A run of `MaskedDiffusionObjective` from a loaded checkpoint, read back
    by `Pretrained.from_run`, is a masked decoder by the run's objective; its
    export in the family its model derives is one by the export's own config,
    which `Pretrained.load` reads to the run's trained logits."""
    import math

    import grain.python as grain
    import jax
    import optax

    from dew.checkpoints import Checkpoints
    from dew.data import Loading
    from dew.diffusion.discrete import MDLM
    from dew.interop import PretrainedMaskedDecoder
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective
    from dew.training import Trainer

    source = Pretrained.load(str(FIXTURES / fixture), dtype="float32", attention_impl="reference")
    ids = np.load(FIXTURES / fixture / "input_ids.npy")
    objective = MaskedDiffusionObjective(source.model, MDLM(mask_id=source.model.mask_token_id)(),
                                         int(ids.shape[1]), ema_decay=None, variables=source.variables)
    count = math.lcm(int(ids.shape[0]), jax.device_count())
    rows = [{"text": row} for row in np.concatenate([ids] * (count // int(ids.shape[0])))]
    data = compile(Pipeline(Memory(grain.MapDataset.source(rows))).repeat().batch(count), read="indexed",
        loading=Loading(workers=0))
    checkpoints = Checkpoints(str(tmp_path / "run"))
    state = Trainer(objective, optax.sgd(1.0), key=jax.random.key(0), checkpoints=checkpoints).fit(
        data, steps=1, checkpoint_every=1)
    checkpoints.wait()

    exported = Pretrained.from_run(tmp_path / "run")
    exported.save(tmp_path / "export")
    reloaded = Pretrained.load(str(tmp_path / "export"), dtype="float32", attention_impl="reference")

    assert isinstance(exported, PretrainedMaskedDecoder) and isinstance(reloaded, PretrainedMaskedDecoder)
    assert reloaded.model.mask_token_id == source.model.mask_token_id
    np.testing.assert_array_equal(tool.logits(reloaded, reloaded.variables, ids),
                                  tool.logits(source, state.variables, ids))

    # The export generates with the run's response budget, through no KV cache.
    from dew.inference.tasks import MaskedGeneration

    task, saved = reloaded.text_generation(), MaskedGeneration.from_run(str(tmp_path / "run"))
    assert (task.max_new_tokens, task.pad_token_id) == (saved.max_new_tokens, saved.pad_token_id)
    assert task.max_new_tokens == int(ids.shape[1])
    assert task(ids[:1, :4], 4, key=0).tokens.shape[-1] == 8
