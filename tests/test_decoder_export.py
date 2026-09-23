"""Source-format decoder lifecycles on committed tiny fixtures.

One real Trainer step under LMObjective is exported through Pretrained.save,
reloaded by Dew, and read by the declared Transformers implementation. Tests
enforce source names, layouts, tokenizer assets, numerical bounds and gradient
coverage; tools/decoder_export_reference.py reports per-run measurements.

tools/decoder_export_reference.py owns the pipeline and prints the numbers:

    JAX_PLATFORMS=cpu PYTHONPATH=src python tools/decoder_export_reference.py

Observed on CPU in fp32, every position's argmax equal and the tolerance
1e-4 on logits of magnitude 6, over the trained export:

| family       | source tensors | per-expert | max abs logit difference |
| ------------ | -------------- | ---------- | ------------------------ |
| mixtral      |             41 |         24 |                  2.9e-06 |
| qwen3_moe    |             46 |         12 |                  2.0e-06 |
| glm4_moe     |            103 |         48 |                  2.2e-06 |
| glm_moe_dsa  |            298 |        144 |                  3.8e-06 |
| deepseek_v2  |             48 |         24 |                  3.2e-06 |
| deepseek_v3  |             53 |         24 |                  5.1e-06 |
| deepseek_v32 |             63 |         24 |                  3.4e-06 |
| kimi_k2      |             65 |         36 |                  2.6e-06 |
| kimi_k25     |            100 |         36 |                  2.9e-06 |
| llama4_text  |             45 |          0 |                  4.2e-06 |
| olmo3        |             47 |          0 |                  3.6e-06 |
| qwen3_next   |            195 |         96 |                  1.4e-04 |

Llama 4 ships one fused `experts.gate_up_proj` per routed layer instead of
one tensor per expert, so its export runs the fused path and holds no
indexed binding. transformers' Glm4Moe and GlmMoeDsa have no MTP depth and
ignore those tensors of the GLM checkpoints, so the reference agrees on the
trunk and the depth's own weights are held to account through the dew
reload. OLMo 3 is the dense case, and the one whose rotary differs between
its layer kinds: its fixture carries the released 7B YaRN on the
full-attention layers alone, so the export has to write that per-kind rope
back and not one table for the model. Qwen3-Next ships its prediction layer
as mtp.* tensors transformers ignores on load
(modeling_qwen3_next.py:877), so like GLM's depth it is held to account
through the dew reload and the MTP loss term of the training step; its
larger residue is the delta net layers' fp32 rounding, the same residue
tests/test_hf_decoders.py records for qwen35-tiny.

glm_moe_dsa and kimi_k25 are also held to the reference's gradients: dew's
gradient of the next-token cross entropy, written into the source layout
through the same bindings the export uses, against the reference's `.grad`
(the per-expert tensors read their slice of the fused parameters).
Observed max scaled error 1.5e-06 over 224 tensors for glm_moe_dsa and
4.5e-07 over 64 for kimi_k25, both at the 1e-4 bound. DeepSeek V4 compares
the actual LMObjective gradients including its executable prediction depth,
covering every source parameter except independently checked no-gradient
selector leaves. Its 361 tensors are bound, including 168 expert tensors;
the reference depth follows the official raw-stream MTPBlock composition.

transformers 5.16.1 registers no `kimi_k2` config, so the tool names the
class Kimi's release points its `auto_map` at, `DeepseekV3ForCausalLM`,
which is also the class transformers itself substitutes where it reads the
name (`Kimi_K25Config.__post_init__`). It registers that class's own
per-expert tensor conversion under the checkpoint's model_type, since
transformers keys the conversion by model_type; without it the release's
one tensor per expert reaches no converter and the loading report names 2
missing and 36 unexpected keys.

kimi_k25 is the Kimi K2.5 repo: the same decoder nested under
`language_model.model.*` inside a vision wrapper, so 35 of its 100 tensors
are the tower and the projector, which this has no counterpart for. They
are retained by name at load and written back byte for byte, and the
reference is `Kimi_K25ForConditionalGeneration` on input_ids alone, which
is that wrapper's text half. Media placeholder ids follow its token-zero
embedding rule; no vision or projector computation is claimed.

The tiny checkpoints carry no tokenizer, so the tokenizer half of an export
is exercised on a copy of one with the committed byte-level BPE beside it.

Fused and per-expert layouts exercise their respective inverse mappings.
Prediction tensors omitted by upstream trunk classes are checked separately,
not treated as evidence of an executed upstream prediction layer. Gradient
scope is stated by each gate. Released-scale and tied-selector behavior are
not established by the tiny untied-cutoff fixtures.
"""

import dataclasses
import json
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest

from dew.interop import load_pretrained
from dew.interop.safetensors_io import save_hf_layout
from tools import decoder_export_reference as tool

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "hf"
TOKENIZER = REPO_ROOT / "tests" / "fixtures" / "tokenizers" / "tiny-tools"
LOGITS = 1e-4
"""Cross-framework fp32 parity, relative to the logit scale: the two forwards
sum in different orders, and that noise grows with the logits. Absolute,
the same bound fails the untrained qwen3_next fixture (1.18e-4 at max
|logit| 6.4) with identical code on either side of an unrelated change."""
MOVEMENT = 1e-4
# The families whose checkpoint names one tensor per expert, which the load
# stacks and the export slices back apart. Llama 4 fuses its experts
# instead and is covered by every other case here.
INDEXED = ("mixtral", "qwen3_moe", "glm4_moe", "glm_moe_dsa", "deepseek_v2",
           "deepseek_v3", "deepseek_v32", "deepseek_v4", "kimi_k2", "kimi_k25",
           "qwen3_next", "glm5_next")


CASES = {case.name: case for case in tool.CASES}
BALANCED = tuple(name for name, case in CASES.items() if case.balance_rate is not None)
# GLM-4 and GLM DSA ship embedding/head copies; GLM-5.3-Flash and Qwen do not.
COPIED_MTP = ("glm4_moe", "glm_moe_dsa")
# The families whose training claim is held to the reference's gradients
# too: dew's gradient of the next-token cross entropy, written back into
# the source's tensor layout, against the reference's `.grad`.
GRADIENTS = ("glm_moe_dsa", "deepseek_v4", "kimi_k25", "qwen3_next", "glm5_next")
GRADIENT = 1e-4


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
        if name in retained or name.endswith('.tid2eid'):
            np.testing.assert_array_equal(trip.exported_tensors[name], tensor, err_msg=name)
            assert trip.exported_tensors[name].dtype == tensor.dtype, name


def test_the_export_carries_the_trained_weights_not_the_loaded_ones(trip):
    """One SGD step moves every tensor kind the checkpoint holds, and it is
    the moved values the source layout writes back. Every decoder holds an
    embedding and attention projections; a routed one holds its experts and
    router besides."""
    distances = tool.moved(trip)
    routed = trip.source.model.mixture is not None

    assert set(distances) >= {"embedding", "attention"} | (
        {"expert", "router"} if routed else {"feedforward"})
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
    ids, same argmax, and the logits agree to `LOGITS` of their scale."""
    assert np.array_equal(np.argmax(trip.theirs, -1), np.argmax(trip.ours, -1))
    difference = float(np.max(np.abs(trip.theirs - trip.ours)))
    scale = float(np.max(np.abs(trip.theirs)))
    assert difference < LOGITS * scale, f"max |logit difference| {difference:.3e} at scale {scale:.2f}"


def test_the_export_keeps_the_sources_config_and_generation_config(trip):
    """The source's own config and generation config are retained, not
    rebuilt from the model: a derived config would drop the fields the
    reference reads and the loader ignores."""
    directory = FIXTURES / trip.case.fixture

    assert json.loads((trip.export / 'config.json').read_text()) == json.loads(
        (directory / 'config.json').read_text())
    source_file = directory / 'generation_config.json'
    exported_file = trip.export / 'generation_config.json'
    expected = json.loads(source_file.read_text()) if source_file.exists() else {}
    actual = json.loads(exported_file.read_text()) if exported_file.exists() else {}
    assert actual == expected, 'generation_config.json'


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
    scale = float(np.max(np.abs(trip.ours)))
    assert difference > LOGITS * scale, f"rotating {len(bindings)} experts moved the logits {difference:.3e}"


def test_an_expert_index_past_the_stack_is_refused(indexed):
    """A binding that asks for an expert the leaf does not hold names the
    tensor and the shape it found rather than writing a reshaped slice."""
    trip, bindings = indexed
    mixture = trip.source.model.mixture
    assert mixture is not None
    beyond = dataclasses.replace(bindings[0], expert_index=mixture.experts)

    with pytest.raises(ValueError, match=bindings[0].name.replace(".", r"\.")):
        beyond.export(trip.trained)


@pytest.mark.parametrize("name", COPIED_MTP, ids=COPIED_MTP)
def test_the_mtp_copies_carry_the_trained_embedding_and_head(name, trips):
    """GLM's prediction depth ships copies of the trunk's embedding and
    head, which it shares. The export writes the trained weights into both
    names, so a reload still sees a depth that shares them."""
    trip = trips(name)
    exported, source = trip.exported_tensors, trip.source_tensors
    depth = f"model.layers.{trip.source.model.num_layers}"

    for copy, trunk in ((f"{depth}.embed_tokens.weight", "model.embed_tokens.weight"),
                        (f"{depth}.shared_head.head.weight", "lm_head.weight")):
        assert np.array_equal(exported[copy], exported[trunk]), copy
        moved = float(np.max(np.abs(exported[copy].astype(np.float32)
                                    - source[copy].astype(np.float32))))
        assert moved > MOVEMENT, f"{copy} still holds the checkpoint's values"


def test_glm5_next_trained_prediction_depth_reads_its_export(trips):
    trip = trips('glm5_next')
    model = trip.source.model
    states, _ = model.apply(trip.trained, trip.ids, method=model.states_and_logits)
    assert np.shape(states) == (*trip.ids.shape, model.emb_features)
    actual = np.asarray(model.apply(
        trip.trained, states, trip.ids, method=model.mtp_logits)[0])
    expected = tool.glm5_prediction_logits(trip.case, trip.export, trip.ids)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    difference = float(np.max(np.abs(actual - expected)))
    assert difference < LOGITS, f"max |prediction-logit difference| {difference:.3e}"
    projection = f'model.layers.{model.num_layers}.eh_proj.weight'
    assert np.any(trip.exported_tensors[projection] != trip.source_tensors[projection])


@pytest.mark.parametrize("name", GRADIENTS, ids=GRADIENTS)
def test_the_gradients_match_the_reference_implementation(name, trips):
    """Compare all trunk next-token CE gradients by source tensor name.

    This gate does not claim prediction-loss gradient parity for the
    ordinary reference classes, which do not execute those depths; V4 is
    the exception, whose raw-stream depth runs under the actual
    LMObjective loss and whose depth gradients are therefore compared too.
    Selector gradients must be absent in the reference and exactly zero in
    Dew. The helper starts from every non-None upstream gradient and
    enforces the source/layout bijection before reporting errors.
    """
    trip = trips(name)
    errors = tool.gradient_parity(trip)
    nonfinite = {tensor: error for tensor, error in errors.items() if not np.isfinite(error)}
    assert not nonfinite, nonfinite
    worst = max(errors, key=errors.__getitem__)
    assert errors[worst] < GRADIENT, f"{worst} gradient scaled error {errors[worst]:.3e}"


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
              if layout.paths[0][-1] == "e_score_correction_bias"]
    assert biases, f"{name} carries no balancing bias"

    for bias in biases:
        moved = np.abs(trip.exported_tensors[bias].astype(np.float32)
                       - trip.source_tensors[bias].astype(np.float32))
        assert float(np.max(moved)) == pytest.approx(rate, rel=1e-5), bias
        assert np.all((moved < rate * 1e-3) | (np.abs(moved - rate) < rate * 1e-5)), (
            f"{bias} moved by {sorted(set(moved.tolist()))}, not by whole rate steps")


def test_deepseek_v4_trained_mtp_export_matches_reference(trips):
    import torch

    from tools.deepseek_v4_reference import load_mtp_reference

    trip = trips('deepseek_v4')
    _, streams = trip.source.model.apply(trip.trained, trip.ids,
                                         method=trip.source.model.hidden_and_mtp_inputs)
    actual = trip.source.model.apply(trip.trained, streams, trip.ids,
                                     method=trip.source.model.mtp_logits)[0]
    reference, _ = tool.reference_model(trip.case, trip.export)
    reference.eval()
    reference.set_attn_implementation('eager')
    captured = []

    def capture(module, args, output):
        captured.append(output)

    handle = reference.model.layers[-1].register_forward_hook(capture)
    ids = torch.from_numpy(trip.ids.astype(np.int64))
    # The trained step puts indexer scores on exact ties, which torch's
    # top-k and jax's break differently; the reference runs under Dew's
    # tie rule so the comparison is of the composition, not of the split.
    try:
        with torch.no_grad(), tool.lower_index_ties(reference):
            reference(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    # The depth's own layer is sliding attention and selects nothing, so
    # the contract scopes the trunk run alone, which is what ties the
    # streams it reads.
    depth = load_mtp_reference(trip.export, reference.config)
    with torch.no_grad():
        expected = depth(reference, captured[0][:, :-1], ids[:, 1:]).numpy()
    np.testing.assert_allclose(actual, expected, atol=LOGITS, rtol=0)
    for projection in ('e_proj', 'h_proj'):
        name = f'mtp.0.{projection}.weight'
        assert np.max(np.abs(trip.exported_tensors[name] - trip.source_tensors[name])) > 0


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


def test_a_quantized_source_exports_trained_weights_in_its_original_format(tmp_path):
    """The source config describes real FP8 bytes after a training update."""
    import torch
    from safetensors.torch import load_file, save_file

    directory = tmp_path / "fp8"
    source = FIXTURES / "deepseek-v3-tiny"
    tensors = tool.source_tensors(source)
    config = json.loads((source / "config.json").read_text())
    scaled = "model.layers.0.self_attn.o_proj.weight"
    rounded = torch.from_numpy(np.array(tensors[scaled], copy=True)).to(
        torch.float8_e4m3fn
    )
    dense_tensors = {**tensors, scaled: rounded.float().numpy()}
    expected_directory = tmp_path / "dequantized"
    save_hf_layout(dense_tensors, config, expected_directory)
    config["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
    }
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config))
    packed = {
        name: torch.from_numpy(np.array(value, copy=True))
        for name, value in tensors.items()
    }
    packed[scaled] = rounded
    packed[scaled + "_scale_inv"] = torch.ones((1, 1), dtype=torch.float32)
    save_file(packed, str(directory / "model.safetensors"))
    quantized = load_pretrained(directory, dtype="float32", attention_impl="reference")
    expected = load_pretrained(
        expected_directory, dtype="float32", attention_impl="reference"
    )
    ids = np.load(source / "input_ids.npy")
    np.testing.assert_array_equal(
        tool.logits(quantized, quantized.variables, ids),
        tool.logits(expected, expected.variables, ids),
    )
    state = tool.train(CASES["deepseek_v3"], quantized, ids)
    destination = tmp_path / "exported"
    quantized.save(destination, variables=state.params)
    packed_export = load_file(str(destination / "model.safetensors"))
    assert packed_export[scaled].dtype == torch.float8_e4m3fn
    assert json.loads((destination / "config.json").read_text()) == config
    float_export = {
        name: value.float().numpy()
        for name, value in packed_export.items()
        if name != scaled + "_scale_inv"
    }
    float_export[scaled] = (
        packed_export[scaled].float() * packed_export[scaled + "_scale_inv"][0, 0]
    ).numpy()
    original_quantized = dense_tensors[scaled]
    assert not np.array_equal(float_export[scaled], original_quantized)
    plain_config = {name: value for name, value in config.items() if name != "quantization_config"}
    plain_directory = tmp_path / "decoded-export"
    save_hf_layout(float_export, plain_config, plain_directory)
    decoded = load_pretrained(plain_directory, dtype="float32", attention_impl="reference")
    reloaded = load_pretrained(destination, dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(tool.logits(reloaded, reloaded.variables, ids),
                                  tool.logits(decoded, decoded.variables, ids))
    for layout in quantized.weight_layouts:
        if layout.name != scaled:
            np.testing.assert_array_equal(float_export[layout.name], layout.export(state.params))
    without_provenance = dataclasses.replace(quantized, quantized_tensors=())
    with pytest.raises(ValueError, match="recorded no quantized tensors"):
        without_provenance.save(tmp_path / "refused", variables=state.params)


def test_mxfp4_source_reexports_the_trained_experts_and_preserves_float_tensors(tmp_path):
    """A GPT OSS source shipped MXFP4 writes its trained experts back as
    blocks and scales the released reader decodes, and every float tensor
    -- biases, router, sinks, embeddings -- as itself."""
    import torch
    from safetensors.numpy import load_file
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    from dew.interop.codecs import pack_mxfp4

    source = FIXTURES / "gpt-oss-tiny"
    tensors = load_file(str(source / "model.safetensors"))
    stems = tuple(name for name in tensors if name.endswith((".experts.gate_up_proj", ".experts.down_proj")))
    packed = pack_mxfp4(tensors, stems)
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "mxfp4"}
    directory = tmp_path / "source"
    save_hf_layout(packed, config, directory)
    loaded = load_pretrained(directory, dtype="float32", attention_impl="reference")
    ids = np.load(source / "input_ids.npy")
    state = tool.train(tool.Case("gpt_oss", "gpt-oss-tiny"), loaded, ids)
    destination = tmp_path / "export"
    loaded.save(destination, variables=state.params)
    emitted = load_file(str(destination / "model.safetensors"))
    assert set(emitted) == set(packed)
    assert json.loads((destination / "config.json").read_text()) == config
    decoded = {name: value for name, value in emitted.items()
               if not name.endswith(("_blocks", "_scales"))}
    for stem in stems:
        decoded[stem] = convert_moe_packed_tensors(
            torch.from_numpy(emitted[stem + "_blocks"]),
            torch.from_numpy(emitted[stem + "_scales"])).float().numpy()
    assert any(not np.array_equal(decoded[layout.name], layout.export(loaded.variables))
               for layout in loaded.weight_layouts if layout.name in stems)
    for layout in loaded.weight_layouts:
        if layout.name not in stems:
            np.testing.assert_array_equal(decoded[layout.name], layout.export(state.params))
    plain_directory = tmp_path / "decoded"
    plain_config = {name: value for name, value in config.items() if name != "quantization_config"}
    save_hf_layout(decoded, plain_config, plain_directory)
    plain = load_pretrained(plain_directory, dtype="float32", attention_impl="reference")
    reloaded = load_pretrained(destination, dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(tool.logits(plain, plain.variables, ids),
                                  tool.logits(reloaded, reloaded.variables, ids))
    missing = dataclasses.replace(loaded, quantized_tensors=())
    with pytest.raises(ValueError, match="recorded no quantized tensors"):
        missing.save(tmp_path / "missing-provenance", variables=state.params)


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


def test_a_shared_copy_cannot_hide_an_undeclared_prediction_depth(tmp_path):
    source = FIXTURES / "glm4-moe-tiny"
    config = json.loads((source / "config.json").read_text())
    tensors = tool.source_tensors(source)
    first_absent = config["num_hidden_layers"] + config["num_nextn_predict_layers"]
    name = f"model.layers.{first_absent}.embed_tokens.weight"
    tensors[name] = tensors["model.embed_tokens.weight"]
    directory = tmp_path / "undeclared-depth"
    save_hf_layout(tensors, config, directory)
    with pytest.raises(ValueError, match="undeclared prediction depth"):
        load_pretrained(directory, dtype="float32", attention_impl="reference")


@pytest.mark.parametrize("name", ["deepseek_v2", "deepseek_v3", "deepseek_v32"])
def test_the_latent_norms_keep_the_reference_epsilon(name, tmp_path):
    """DeepSeek's q_a_layernorm and kv_a_layernorm run RMSNorm's 1e-6
    whatever rms_norm_eps configures: a checkpoint whose trunk epsilon is
    1e-3 still parities the reference, so the mixer's epsilon is its own."""
    source = FIXTURES / CASES[name].fixture
    config = json.loads((source / "config.json").read_text())
    config["rms_norm_eps"] = 1e-3
    directory = tmp_path / name
    save_hf_layout(tool.source_tensors(source), config, directory)
    ids = np.load(source / "input_ids.npy")

    loaded = load_pretrained(str(directory), dtype="float32", attention_impl="reference")
    np.testing.assert_allclose(tool.logits(loaded, loaded.variables, ids),
                               tool.reference_logits(CASES[name], directory, ids),
                               atol=LOGITS, rtol=0)


@pytest.mark.parametrize("kind", ["fp8", "mxfp4"])
def test_codec_parameter_storage_follows_fp32_dequantization(kind):
    from dew.interop.codecs import dequantize_checkpoint, pack_fp8, pack_mxfp4, unpack_mxfp4

    weight = (np.arange(15, dtype=np.float32).reshape(3, 5) - 7) / 11 if kind == "fp8" else (
        np.arange(2 * 64 * 48, dtype=np.float32).reshape(2, 64, 48) % 13 - 6) / 7
    source = {"weight": weight, "state": np.asarray([.1234567], np.float32),
              "indices": np.asarray([0, 255], np.uint8)}
    if kind == "fp8":
        packed = pack_fp8(source, ("weight",), block=2, ue8m0=False)
        masters = dequantize_checkpoint(packed, block=2)
        native = dequantize_checkpoint(packed, block=2, param_dtype="bfloat16")
    else:
        packed = pack_mxfp4(source, ("weight",))
        masters = unpack_mxfp4(packed)
        native = unpack_mxfp4(packed, param_dtype="bfloat16")
    assert masters["weight"].dtype == np.float32
    assert native["weight"].dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(native["weight"], masters["weight"].astype(ml_dtypes.bfloat16))
    for name in ("state", "indices"):
        assert native[name].dtype == source[name].dtype
        np.testing.assert_array_equal(native[name], source[name])


@pytest.mark.parametrize("kind", ["fp8", "mxfp4"])
def test_codec_rejects_integer_parameter_storage(kind):
    from dew.interop.codecs import dequantize_checkpoint, pack_mxfp4, unpack_mxfp4

    if kind == "fp8":
        packed = {"weight": np.ones((1, 1), np.float32),
                  "weight_scale_inv": np.full((1, 1), 1.25, np.float32)}
        with pytest.raises(ValueError, match="int32"):
            dequantize_checkpoint(packed, 1, param_dtype="int32")
    else:
        packed = pack_mxfp4({"weight": np.full((1, 32, 2), 1.5, np.float32)}, ("weight",))
        with pytest.raises(ValueError, match="int32"):
            unpack_mxfp4(packed, param_dtype="int32")


@pytest.mark.parametrize("kind", ["fp8", "mxfp4"])
def test_public_quantized_load_obeys_parameter_storage(tmp_path, kind):
    from test_interop import assert_parameter_storage

    from dew.interop.codecs import pack_fp8, pack_mxfp4

    fixture = FIXTURES / ("deepseek-v3-tiny" if kind == "fp8" else "gpt-oss-tiny")
    tensors = tool.source_tensors(fixture)
    config = json.loads((fixture / "config.json").read_text())
    if kind == "fp8":
        packed = pack_fp8(tensors, ("model.layers.0.self_attn.o_proj.weight",), block=128, ue8m0=False)
        config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                         "weight_block_size": [128, 128]}
    else:
        stems = tuple(name for name in tensors if name.endswith(
            (".experts.gate_up_proj", ".experts.down_proj")))
        packed = pack_mxfp4(tensors, stems)
        config["quantization_config"] = {"quant_method": "mxfp4"}
    directory = tmp_path / "quantized"
    save_hf_layout(packed, config, directory)
    masters = load_pretrained(directory, dtype="bfloat16", attention_impl="xla")
    native = load_pretrained(directory, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
    assert masters.quantized_tensors == native.quantized_tensors
    assert_parameter_storage(masters.variables, native.variables, lambda path: path[0] == "params")


@pytest.mark.parametrize("same_values", [True, False], ids=["equal-before-rounding", "different-before-rounding"])
def test_public_quantized_alias_check_uses_original_fp32_values(tmp_path, same_values):
    from dew.interop.codecs import E4M3

    fixture = FIXTURES / "deepseek-v3-tiny"
    tensors = tool.source_tensors(fixture)
    config = json.loads((fixture / "config.json").read_text())
    config["tie_word_embeddings"] = True
    config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                     "weight_block_size": [128, 128]}
    shape = tensors["model.embed_tokens.weight"].shape
    decoded = np.float32(1. + 1. / 1024)
    unquantized = decoded if same_values else np.float32(1. + 2. / 1024)
    tensors["model.embed_tokens.weight"] = np.full(shape, unquantized, np.float32)
    tensors["lm_head.weight"] = np.ones(shape, dtype=E4M3)
    tensors["lm_head.weight_scale_inv"] = np.full(
        ((shape[0] + 127) // 128, (shape[1] + 127) // 128), decoded, np.float32)
    for name in tensors:
        if name.startswith("model.layers.") and name.endswith(".embed_tokens.weight"):
            tensors[name] = np.full_like(tensors[name], unquantized, dtype=np.float32)
        elif name.startswith("model.layers.") and name.endswith(".shared_head.head.weight"):
            tensors[name] = np.full_like(tensors[name], decoded, dtype=np.float32)
    directory = tmp_path / "mixed-aliases"
    save_hf_layout(tensors, config, directory)
    if not same_values:
        with pytest.raises(ValueError, match="tie_word_embeddings"):
            load_pretrained(directory, param_dtype="bfloat16")
    else:
        loaded = load_pretrained(directory, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
        values = loaded.variables["params"]["embed_tokens"]["embedding"]
        assert np.asarray(values).dtype == ml_dtypes.bfloat16
        np.testing.assert_array_equal(values, np.full(shape, decoded).astype(ml_dtypes.bfloat16))


def test_public_quantized_diffusion_gemma_rejects_rounded_shared_copies(tmp_path):
    from dew.interop.codecs import E4M3

    fixture = FIXTURES / "diffusion-gemma-workflow"
    tensors = tool.source_tensors(fixture)
    config = json.loads((fixture / "config.json").read_text())
    decoder = next(name for name in tensors if name.startswith("model.decoder.layers.")
                   and name.endswith(".self_attn.q_proj.weight"))
    encoder = decoder.replace("model.decoder.", "model.encoder.language_model.")
    shape = tensors[decoder].shape
    tensors[decoder] = np.ones(shape, dtype=E4M3)
    tensors[decoder + "_scale_inv"] = np.ones(
        ((shape[0] + 127) // 128, (shape[1] + 127) // 128), np.float32)
    tensors[encoder] = np.full(shape, 1. + 1. / 1024, np.float32)
    config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                     "weight_block_size": [128, 128]}
    directory = tmp_path / "shared-decoder"
    save_hf_layout(tensors, config, directory)
    with pytest.raises(ValueError, match="differs between the encoder and the decoder"):
        load_pretrained(directory, param_dtype="bfloat16")


def test_diffusion_gemma_source_export_resolves_omitted_embedding_tie_default(tmp_path):
    from shutil import copytree

    from dew.interop.diffusion_gemma import export_weights

    source = copytree(FIXTURES / "diffusion-gemma-workflow", tmp_path / "source")
    config = json.loads((source / "config.json").read_text())
    del config["text_config"]["tie_word_embeddings"]
    (source / "config.json").write_text(json.dumps(config))
    loaded = load_pretrained(source, dtype="float32", attention_impl="reference")
    destination = tmp_path / "export"
    loaded.save(destination)
    restored = load_pretrained(destination, dtype="float32", attention_impl="reference")
    for expected, actual in zip(jax.tree.leaves(loaded.variables),
                                jax.tree.leaves(restored.variables), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for stated in (False, None):
        # An explicit null is the reference's own False, not the family's
        # tied default (configuration_utils.py reads the key it is given).
        disagreeing = {**config, "text_config": {**config["text_config"],
                                                 "tie_word_embeddings": stated}}
        with pytest.raises(ValueError, match="tie_word_embeddings"):
            export_weights(loaded.model, loaded.variables, disagreeing)


@pytest.mark.parametrize("fixture, mode, dense", [
    ("gemma4-ple", "frozen", True),
    ("gemma4-ple", "frozen", False),
    ("gemma4-kvshare", "frozen", False),
    ("gemma4-e2b", "frozen", False),
    ("gemma4-moe-tiny", "frozen", False),
    ("gemma4-moe-tiny", "trainable", False),
])
def test_standalone_gemma4_export_preserves_computation(fixture, mode, dense, tmp_path):
    """Only model + variables reach the writer; HF receives no source template."""
    import jax.numpy as jnp
    import torch
    from flax.core import unfreeze
    from transformers import Gemma4ForCausalLM

    from dew.interop.hf_decoders import save_pretrained_decoder

    loaded = load_pretrained(FIXTURES / fixture, dtype="float32", attention_impl="reference")
    model = loaded.model.clone(layer_scalar=mode)
    ids = np.load(FIXTURES / fixture / "input_ids.npy")
    if dense:
        model = model.clone(per_layer_input_dim=None, per_layer_input_vocab=None)
        variables = unfreeze(model.init(jax.random.key(11), jnp.asarray(ids)))
    else:
        variables = unfreeze(loaded.variables)
    for index in range(model.num_layers):
        layer = f"layers_{index}"
        scalar = variables["constants"][layer].pop("layer_scalar")
        collection = "params" if mode == "trainable" else "constants"
        variables[collection][layer]["layer_scalar"] = jnp.full_like(scalar, 0.75 + index * 0.125)
    expected = np.asarray(jax.jit(model.apply)(variables, jnp.asarray(ids)))
    save_pretrained_decoder(model, variables, tmp_path)

    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    actual = np.asarray(jax.jit(restored.model.apply)(restored.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=LOGITS, rtol=0)
    reference, report = Gemma4ForCausalLM.from_pretrained(
        tmp_path, dtype=torch.float32, attn_implementation="eager", output_loading_info=True)
    assert not report["missing_keys"] and not report["unexpected_keys"]
    assert not report["mismatched_keys"] and not report["error_msgs"]
    with torch.no_grad():
        reference_logits = reference.eval()(torch.from_numpy(ids), use_cache=False).logits.numpy()
    np.testing.assert_allclose(reference_logits, expected, atol=LOGITS, rtol=0)
    # HF stores layer scalars as buffers, even when their exported values were trained.
    for index in range(model.num_layers):
        layer = f"layers_{index}"
        collection = "params" if mode == "trainable" else "constants"
        np.testing.assert_array_equal(restored.variables["constants"][layer]["layer_scalar"],
                                      variables[collection][layer]["layer_scalar"])


def glm5_native_export_case(variant):
    import jax.numpy as jnp
    from flax.core import unfreeze

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.dsa_kpool import KPoolSparseAttentionMixer
    from dew.nn.kda import KimiDeltaAttentionMixer

    source = load_pretrained(FIXTURES / "glm5-next-tiny", dtype="float32", attention_impl="reference")
    model, variables = source.model, unfreeze(dict(source.variables))
    assert isinstance(model, CausalTransformer)
    params = variables["params"]
    if variant == "native_geometry":
        assert model.kinds is not None
        kinds = dict(model.kinds)
        linear, sparse = kinds["linear_attention"].mixer, kinds["full_attention"].mixer
        assert isinstance(linear, KimiDeltaAttentionMixer) and isinstance(sparse, KPoolSparseAttentionMixer)
        kinds["linear_attention"] = dataclasses.replace(
            kinds["linear_attention"], mixer=dataclasses.replace(linear, linear_lower_bound=None))
        kinds["full_attention"] = dataclasses.replace(
            kinds["full_attention"], mixer=dataclasses.replace(
                sparse, index_kpool=3, index_topk=6, index_kpool_always_select_tail=False))
        assert model.mixture is not None
        model = model.clone(kinds=kinds, tie_embeddings=True, index_share_for_mtp_iteration=False,
                            mixture=dataclasses.replace(model.mixture, layers=(1, 3, 4), norm_topk_prob=False))
        del params["lm_head"]
        params["layers_1"]["mlp"] = jax.tree.map(lambda leaf: leaf, params["layers_4"]["mlp"])
        variables["moe"]["layers_1"] = jax.tree.map(lambda leaf: leaf, variables["moe"]["layers_4"])
        for block in (params["layers_3"], params["mtp_0"]["block"]):
            indexer = block["self_attn"]["indexer"]
            ape = indexer["index_kpool_compress_ape"]
            indexer["index_kpool_compress_ape"] = jnp.concatenate([ape, ape[:1]], axis=0)
    elif variant == "dense":
        model = model.clone(mixture=None, num_nextn_predict_layers=0, index_share_for_mtp_iteration=False)
        for index in (3, 4):
            params[f"layers_{index}"]["mlp"] = jax.tree.map(lambda leaf: leaf, params["layers_0"]["mlp"])
        del params["mtp_0"]
        del variables["moe"]
    params["norm"]["scale"] = params["norm"]["scale"] * jnp.float32(1.125)
    for index in range(model.num_layers):
        for site in ("attn_hc", "ffn_hc"):
            params[f"layers_{index}"][site]["scale"] = (
                params[f"layers_{index}"][site]["scale"] * jnp.float32(0.8) + jnp.float32(0.1))
    if model.num_nextn_predict_layers:
        params["mtp_0"]["final_norm"]["scale"] = params["mtp_0"]["final_norm"]["scale"] * jnp.float32(1.25)
    if "moe" in variables:
        variables["moe"] = jax.tree.map(
            lambda leaf: leaf + jnp.linspace(-0.01, 0.01, leaf.size, dtype=leaf.dtype).reshape(leaf.shape),
            variables["moe"])
    ids = np.load(FIXTURES / "glm5-next-tiny" / "input_ids.npy")
    return model, variables, ids


@pytest.mark.parametrize("variant", ["released", "native_geometry", "dense"])
def test_standalone_glm5_export_preserves_native_and_source_computation(variant, tmp_path):
    import jax.numpy as jnp

    from dew.interop.hf_decoders import save_pretrained_decoder
    from dew.sampling import Sampling, Speculative, generate

    model, variables, ids = glm5_native_export_case(variant)
    expected = np.asarray(jax.jit(model.apply)(variables, jnp.asarray(ids)))
    cache = model.apply(variables, ids.shape[0], method="init_cache", mutable=["cache"])[1]
    if model.num_nextn_predict_layers:
        cache = model.apply({**variables, **cache}, ids.shape[0], method="init_mtp_cache", mutable=["cache"])[1]
    save_pretrained_decoder(model, {**variables, **cache}, tmp_path)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    actual = np.asarray(jax.jit(restored.model.apply)(restored.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=LOGITS, rtol=0)
    np.testing.assert_allclose(tool.reference_logits(CASES["glm5_next"], tmp_path, ids),
                               expected, atol=LOGITS, rtol=0)
    written = json.loads((tmp_path / "config.json").read_text())
    assert written["index_share_for_mtp_iteration"] == model.index_share_for_mtp_iteration
    if model.num_nextn_predict_layers:
        states = model.apply(variables, jnp.asarray(ids), method="hidden_states")
        prediction = model.apply(variables, states, jnp.asarray(ids), method="mtp_logits")[0]
        reloaded_states = restored.model.apply(restored.variables, jnp.asarray(ids), method="hidden_states")
        reloaded_prediction = restored.model.apply(
            restored.variables, reloaded_states, jnp.asarray(ids), method="mtp_logits")[0]
        np.testing.assert_allclose(reloaded_prediction, prediction, atol=LOGITS, rtol=0)
        np.testing.assert_allclose(tool.glm5_prediction_logits(CASES["glm5_next"], tmp_path, ids),
                                   prediction, atol=LOGITS, rtol=0)
        ordinary = generate(model, variables, jnp.asarray(ids[:, :5]), 4, key=jax.random.key(0),
                            sampling=Sampling(temperature=0)).host()
        speculative = restored.text_generation(sampling=Sampling(temperature=0))(
            ids[:, :5], max_new_tokens=4, seed=0, strategy=Speculative(block=3)).host()
        np.testing.assert_array_equal(speculative.tokens, ordinary.tokens)
        np.testing.assert_array_equal(speculative.lengths, ordinary.lengths)
    for collection in ("params", "moe"):
        if collection in variables:
            actual_leaves = flat(restored.variables[collection])
            for name, value in flat(variables[collection]).items():
                np.testing.assert_array_equal(actual_leaves[name], value)
