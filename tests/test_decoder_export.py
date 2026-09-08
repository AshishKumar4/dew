"""Trained source-format exports of the routed decoder families.

`load_pretrained` binds every tensor of a routed checkpoint to the leaf it
loaded into, so a trained model writes back into the source's own tensor
names beside the config and generation config it came with, rather than a
config derived from the built model. These cases run that path end to end
on the committed tiny checkpoints: one real `Trainer` step of plain SGD
under `LMObjective`, `Pretrained.save`, then the export read back by
`load_pretrained` and by transformers 5.16.1 on the same ids.

tools/decoder_export_reference.py owns the pipeline and prints the numbers:

    JAX_PLATFORMS=cpu PYTHONPATH=src python tools/decoder_export_reference.py

Observed on CPU in fp32, every position's argmax equal and the tolerance
1e-4 on logits of magnitude 6, over the trained export:

| family       | source tensors | per-expert | max abs logit difference |
| ------------ | -------------- | ---------- | ------------------------ |
| mixtral      |             41 |         24 |                  2.9e-06 |
| qwen3_moe    |             46 |         12 |                  2.0e-06 |
| glm4_moe     |            103 |         48 |                  2.2e-06 |
| deepseek_v2  |             48 |         24 |                  3.2e-06 |
| deepseek_v3  |             53 |         24 |                  5.1e-06 |
| deepseek_v32 |             63 |         24 |                  3.4e-06 |
| llama4_text  |             45 |          0 |                  4.2e-06 |

Llama 4 ships one fused `experts.gate_up_proj` per routed layer instead of
one tensor per expert, so its export runs the fused path and holds no
indexed binding. transformers' Glm4Moe has no MTP depth and ignores those
tensors of the GLM checkpoint, so the reference agrees on the trunk and the
depth's own weights are held to account through the dew reload.

The tiny checkpoints carry no tokenizer, so the tokenizer half of an export
is exercised on a copy of one with the committed byte-level BPE beside it.
"""

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from dew.interop import load_pretrained
from dew.interop.safetensors_io import save_hf_layout
from tools import decoder_export_reference as tool

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "hf"
TOKENIZER = REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "tiny-tools"
LOGITS = 1e-4
MOVEMENT = 1e-4
# The families whose checkpoint names one tensor per expert, which the load
# stacks and the export slices back apart. Llama 4 fuses its experts
# instead and is covered by every other case here.
INDEXED = ("mixtral", "qwen3_moe", "glm4_moe", "deepseek_v2", "deepseek_v3", "deepseek_v32")


CASES = {case.name: case for case in tool.CASES}
BALANCED = tuple(name for name, case in CASES.items() if case.balance_rate is not None)


def flat(tree):
    return {".".join(str(entry.key) for entry in path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("decoder-export")


@pytest.fixture(scope="module")
def trips(workspace):
    """Each family trained and exported once, shared by every case here.

    A second training step over the same checkpoint would buy nothing, and
    the family-specific claims read the same round trip the parameterized
    ones do.
    """
    cache = {}

    def trip(name: str):
        if name not in cache:
            cache[name] = tool.round_trip(CASES[name], workspace)
        return cache[name]

    return trip


@pytest.fixture(params=list(CASES), ids=list(CASES))
def trip(request, trips):
    return trips(request.param)


@pytest.fixture(params=INDEXED, ids=INDEXED)
def indexed(request, trips):
    """A family whose checkpoint holds one tensor per expert, with them."""
    found = trips(request.param)
    bindings = [layout for layout in found.source.weight_layouts
                if layout.expert_index is not None]
    assert bindings, f"{request.param} binds no per-expert tensor"
    return found, bindings


def test_every_source_tensor_is_bound_or_retained_and_written_back(trip):
    """No tensor of the checkpoint disappears on the way out: each one is
    either bound to a leaf or retained by name, and the export holds the
    same table the source did."""
    bound = [layout.name for layout in trip.source.weight_layouts]
    retained = set(trip.source.retained_tensors)

    assert len(bound) == len(set(bound)), "a source tensor is bound twice"
    assert not retained & set(bound), "a tensor is both bound and retained"
    assert set(bound) | retained == set(trip.source_tensors)
    assert set(trip.exported_tensors) == set(trip.source_tensors)
    for name, tensor in trip.source_tensors.items():
        assert trip.exported_tensors[name].shape == tensor.shape, name


def test_the_export_carries_the_trained_weights_not_the_loaded_ones(trip):
    """One SGD step moves the embedding, the routed experts and the router,
    and it is the moved values the source layout writes back."""
    distances = tool.moved(trip)

    assert set(distances) >= {"embedding", "expert", "router"}
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
    """The export is a checkpoint the reference implementation loads: same
    ids, same argmax, and the logits agree to 1e-4."""
    assert np.array_equal(np.argmax(trip.theirs, -1), np.argmax(trip.ours, -1))
    difference = float(np.max(np.abs(trip.theirs - trip.ours)))
    assert difference < LOGITS, f"max |logit difference| {difference:.3e}"


def test_the_export_keeps_the_sources_config_and_generation_config(trip):
    """The source's own config and generation config are retained, not
    rebuilt from the model: a derived config would drop the fields the
    reference reads and the loader ignores."""
    directory = FIXTURES / trip.case.fixture

    for name in ("config.json", "generation_config.json"):
        assert (json.loads((trip.export / name).read_text())
                == json.loads((directory / name).read_text())), name


def test_a_rotated_expert_index_writes_a_model_that_disagrees(indexed, tmp_path):
    """`expert_index` decides which slice of a stacked leaf a per-expert
    tensor holds. Rotating every index writes the same tensor table with
    each expert's weights under its neighbour's name, and the model that
    comes back computes something else."""
    trip, bindings = indexed
    mixture = trip.source.model.mixture
    assert mixture is not None
    rotated = tuple(
        dataclasses.replace(layout, expert_index=(layout.expert_index + 1) % mixture.experts)
        if layout.expert_index is not None else layout
        for layout in trip.source.weight_layouts)

    export = tmp_path / "rotated"
    dataclasses.replace(trip.source, weight_layouts=rotated).save(
        export, variables=trip.trained)
    again = load_pretrained(str(export), dtype="float32", attention_impl="reference")

    assert set(tool.source_tensors(export)) == set(trip.source_tensors), (
        "the rotation changed the tensor table, so the difference below is not numerical")
    theirs = tool.logits(again, again.variables, trip.ids)
    difference = float(np.max(np.abs(theirs - trip.ours)))
    assert difference > LOGITS, f"rotating {len(bindings)} experts moved the logits {difference:.3e}"


def test_an_expert_index_past_the_stack_is_refused(indexed):
    """A binding that asks for an expert the leaf does not hold names the
    tensor and the shape it found rather than writing a reshaped slice."""
    trip, bindings = indexed
    mixture = trip.source.model.mixture
    assert mixture is not None
    beyond = dataclasses.replace(bindings[0], expert_index=mixture.experts)

    with pytest.raises(ValueError, match=bindings[0].name.replace(".", r"\.")):
        beyond.export(trip.trained)


def test_the_mtp_copies_carry_the_trained_embedding_and_head(trips):
    """GLM's prediction depth ships copies of the trunk's embedding and
    head, which it shares. The export writes the trained weights into both
    names, so a reload still sees a depth that shares them."""
    trip = trips("glm4_moe")
    exported, source = trip.exported_tensors, trip.source_tensors
    depth = f"model.layers.{trip.source.model.num_layers}"

    for copy, trunk in ((f"{depth}.embed_tokens.weight", "model.embed_tokens.weight"),
                        (f"{depth}.shared_head.head.weight", "lm_head.weight")):
        assert np.array_equal(exported[copy], exported[trunk]), copy
        moved = float(np.max(np.abs(exported[copy].astype(np.float32)
                                    - source[copy].astype(np.float32))))
        assert moved > MOVEMENT, f"{copy} still holds the checkpoint's values"


@pytest.mark.parametrize("name", BALANCED, ids=BALANCED)
def test_the_balancing_bias_moves_by_its_rate_and_lands_in_the_export(name, trips):
    """The routers' balancing bias is state the step moves, not a
    parameter. It is a sign step of the balance rate, so every entry of
    the exported bias sits either where the checkpoint left it or one
    whole rate away, and the busiest expert's entry has moved."""
    trip = trips(name)
    rate = CASES[name].balance_rate
    assert rate is not None
    biases = [layout.name for layout in trip.source.weight_layouts
              if layout.name.endswith("e_score_correction_bias")]
    assert biases, f"{name} carries no balancing bias"

    for bias in biases:
        moved = np.abs(trip.exported_tensors[bias].astype(np.float32)
                       - trip.source_tensors[bias].astype(np.float32))
        assert float(np.max(moved)) == pytest.approx(rate, rel=1e-5), bias
        assert np.all((moved < rate * 1e-3) | (np.abs(moved - rate) < rate * 1e-5)), (
            f"{bias} moved by {sorted(set(moved.tolist()))}, not by whole rate steps")


def test_the_export_carries_the_sources_tokenizer(tmp_path):
    """A source with tokenizer files hands them to the export, so the
    directory `Pretrained.save` leaves behind reads back with a processor
    that decodes what the source's did."""
    from shutil import copyfile, copytree

    directory = tmp_path / "with-tokenizer"
    copytree(FIXTURES / "mixtral-tiny", directory)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        copyfile(TOKENIZER / name, directory / name)

    source = load_pretrained(str(directory), dtype="float32", attention_impl="reference")
    assert source.processor is not None, "the source's tokenizer files were not read"
    export = tmp_path / "exported"
    source.save(export)

    assert (export / "tokenizer.json").is_file()
    again = load_pretrained(str(export), dtype="float32", attention_impl="reference")
    assert again.processor is not None, "the export carries no tokenizer"
    rows = np.arange(8, 32, dtype=np.int32).reshape(2, 12)
    assert again.processor.decode(rows) == source.processor.decode(rows)


def test_a_quantized_source_loads_and_refuses_to_write_its_own_config(tmp_path):
    """A checkpoint that ships fp8 blocks arrives dequantized, so its
    weights no longer hold the format its config declares. Training on it
    works; writing it back under that config is refused."""
    directory = tmp_path / "fp8"
    source = FIXTURES / "deepseek-v3-tiny"
    tensors = tool.source_tensors(source)
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                     "weight_block_size": [128, 128]}
    scaled = "model.layers.0.self_attn.o_proj.weight"
    tensors[scaled + "_scale_inv"] = np.ones((1, 1), np.float32)
    save_hf_layout(tensors, config, directory)

    quantized = load_pretrained(str(directory), dtype="float32", attention_impl="reference")
    ids = np.load(source / "input_ids.npy")
    reference = np.load(source / "logits.npy")

    np.testing.assert_allclose(
        tool.logits(quantized, quantized.variables, ids), reference, atol=LOGITS, rtol=0)
    assert quantized.weight_layouts, "the quantized source bound no tensor to train"
    with pytest.raises(ValueError, match="quantization_config"):
        quantized.save(tmp_path / "refused")


def test_an_mtp_copy_that_differs_from_the_trunk_names_the_tensor(tmp_path):
    """The depth shares the trunk's head, so a checkpoint whose copy is a
    different matrix is refused naming the tensor rather than loaded as a
    model that computes something else."""
    source = FIXTURES / "glm4-moe-tiny"
    tensors = tool.source_tensors(source)
    copy = "model.layers.2.shared_head.head.weight"
    tensors[copy] = tensors[copy] + 1.0
    directory = tmp_path / "broken"
    save_hf_layout(tensors, json.loads((source / "config.json").read_text()), directory)

    with pytest.raises(ValueError, match=copy.replace(".", r"\.")):
        load_pretrained(str(directory), dtype="float32", attention_impl="reference")
