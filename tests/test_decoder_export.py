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
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest
from interop_support import assert_restores, assert_same_stored_tensors, written_tensors
from model_support import cached_round_trips, flat_tree, streamed_export
from reference_error import (
    FACTOR,
    ORDERS,
    assert_as_exact_as_the_reference,
    assert_as_exact_over_orders,
    assert_computes_the_oracle,
    distance,
    float64_forward,
    within,
)
from residual_orders import orders, permuted, residual_width
from whole_suite import timed

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.safetensors_io import save_hf_layout
from dew.interop.sources import load_shards
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

PAST_TARGET = {"glm5": {"released": 173}}
"""The cases past a group's 120 s (`timed`): GLM-5's released standalone export, 173 s in train 279's CI
before the export checks compiled Dew's forward, 99 s on armada at c937fc46a after (ledgers 2182, 2278);
every train runs its native-geometry and dense exports. The gradients and every case reading GLM-5 Next's
round trip, once past it, run each train since the compile: 52-55 s and 41-51 s at c937fc46a."""


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
    return cached_round_trips(tool.round_trip, CASES, workspace)


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
    same table the source did. GPT-NeoX binds no layout: the decoder
    export's encoder writes its whole table, which the last half holds."""
    bound = [layout.name for layout in trip.source.weight_layouts]
    retained = set(trip.source.retained_tensors)

    if trip.case.name != "gpt_neox":
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
    routed = getattr(trip.source.model, "mixture", None) is not None
    recurrent = trip.case.name == "mamba2"

    assert set(distances) >= {"embedding", "mixer" if recurrent else "attention"} | (
        {"expert", "router"} if routed else set() if recurrent else {"feedforward"})
    for kind, moved in distances.items():
        assert moved > MOVEMENT, f"{kind} moved {moved:.3e}"


def test_the_trained_export_reloads_leaf_for_leaf_and_recomputes_the_logits(trip):
    """`Pretrained.load` reads the export back into the same model with the
    trained values bit for bit, so it computes the same logits."""
    assert_restores(trip.source.model, trip.trained, trip.reloaded, trip.ids)


def test_transformers_reads_the_trained_export(trip, tmp_path):
    """The export is a checkpoint the reference implementation loads with a
    clean report (`tool.reference_model`), and Dew's trained logits are as
    exact as transformers' own over the exported files: tests/reference_error.py's
    rule, against transformers in float64 over the same files. A case with
    `orders` is held over ORDERS residual orders of the trained weights, each
    exported and read by transformers, after checking in float64 that an
    order computes the same logits."""
    directory = FIXTURES / trip.case.fixture
    assert json.loads((trip.export / "config.json").read_text()) == json.loads(
        (directory / "config.json").read_text())
    source_file = directory / "generation_config.json"
    exported_file = trip.export / "generation_config.json"
    expected = json.loads(source_file.read_text()) if source_file.exists() else {}
    actual = json.loads(exported_file.read_text()) if exported_file.exists() else {}
    assert actual == expected
    label = f"{trip.case.name} logits"
    if not trip.case.orders:
        _, terms = float64_forward(trip.source.model, trip.trained, np.asarray(trip.ids, np.int32))
        assert_as_exact_as_the_reference(trip.ours, trip.theirs, trip.truth, label, terms=terms)
        return
    forward = jax.jit(trip.source.model.apply)  # one compile for every order
    mine, theirs = [], [distance(trip.theirs, trip.truth)]
    for k, order in enumerate(orders(residual_width(trip.trained), ORDERS, tool.SEED)):
        variables, export = permuted(trip.trained, order), tmp_path / str(k)
        mine.append(distance(np.asarray(forward(variables, trip.ids)), trip.truth))
        if k == 0:
            continue
        trip.source.save(export, variables=variables)
        theirs.append(distance(tool.reference_logits(trip.case, export, trip.ids), trip.truth))
        if k == 1:
            model = trip.source.model
            roundings = model.num_layers * (model.emb_features + trip.ids.shape[1]) + model.emb_features
            assert_computes_the_oracle(tool.reference_logits(trip.case, export, trip.ids, wide=True),
                                       trip.truth, f"{label} in order 1", roundings=roundings)
    assert_as_exact_over_orders(mine, theirs, label)


def test_a_bfloat16_source_exports_its_trained_float32_masters(tmp_path):
    """A bfloat16 checkpoint loads into float32 masters, and one training
    step leaves them between bfloat16 values. The export writes each weight
    at its leaf's float32 width, so the reload holds the trained masters bit
    for bit; only the int64 token-to-expert tables keep their stored width."""
    fixture = FIXTURES / "deepseek-v4-tiny"
    directory = tmp_path / "bfloat16"
    save_hf_layout({name: tensor.astype(ml_dtypes.bfloat16) if tensor.dtype == np.float32 else tensor
                    for name, tensor in load_shards(fixture).items()},
                   json.loads((fixture / "config.json").read_text()), directory)
    (directory / "input_ids.npy").write_bytes((fixture / "input_ids.npy").read_bytes())
    source, ids, trained, reloaded = tool.trained_export(
        directory, tmp_path / "export", functools.partial(tool.train, CASES["deepseek_v4"]))

    assert source.weight_layouts, "deepseek_v4 writes its export through the source layout"
    stored = load_shards(directory)
    for name, tensor in load_shards(tmp_path / "export").items():
        assert tensor.dtype == (np.float32 if stored[name].dtype == ml_dtypes.bfloat16
                                else stored[name].dtype), name
    assert_restores(source.model, trained, reloaded, ids)


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
    again = Pretrained.load(str(export), dtype="float32", attention_impl="reference")

    assert set(load_shards(export)) == set(trip.source_tensors), (
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

    with pytest.raises(ValueError):
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

    The gradients are held to the reference's in float64 by the float64
    rule: Dew's deviation from them, entry by entry relative to
    max(1, |truth|) and pooled over every compared tensor, within FACTOR of
    the reference's own. Pooled, one tensor's rounding draw does not decide
    it, and a wrong tensor still adds its whole error: Dew and the reference
    each sit about 6e-5 from float64 on qwen3_next's first conv taps on an
    H100, and 1.05e-4 apart, which a constant bound between the two read as
    a failure, Dew the nearer.
    """
    trip = trips(name)
    errors = tool.gradient_parity(trip)
    mine, theirs = (np.concatenate([pair[side] for pair in errors.values()]) for side in (0, 1))
    assert np.all(np.isfinite(mine)), sorted(tensor for tensor, (ours, _) in errors.items()
                                             if not np.all(np.isfinite(ours)))
    worst = max(errors, key=lambda tensor: float(np.max(np.abs(errors[tensor][0]))))
    assert within(distance(mine, 0), FACTOR * distance(theirs, 0)), (
        f"dew is {distance(mine, 0):.3e} from float64, the reference {distance(theirs, 0):.3e}; "
        f"dew's largest deviation is on {worst}")


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

    source = Pretrained.load(str(directory), dtype="float32", attention_impl="reference")
    assert source.processor is not None, "the source's tokenizer files were not read"
    export = tmp_path / "exported"
    source.save(export)

    assert (export / "tokenizer.json").is_file()
    again = Pretrained.load(str(export), dtype="float32", attention_impl="reference")
    assert again.processor is not None, "the export carries no tokenizer"
    rows = np.arange(8, 32, dtype=np.int32).reshape(2, 12)
    assert again.processor.decode(rows) == source.processor.decode(rows)


def test_the_export_assets_rewrite_a_tokenizers_nested_files_in_place(tmp_path):
    """A tokenizer's files under a directory of their own (a processor's chat_templates/) are
    published one file at a time like the rest, so a second export into the same directory replaces
    them, as a weight reload into a serving directory does."""
    from dew.interop.hf_decoders import save_export_assets

    class Templated:
        def __init__(self, template):
            self.template = template

        def save_pretrained(self, directory, /):
            (Path(directory) / "chat_templates").mkdir()
            (Path(directory) / "chat_templates" / "brief.jinja").write_text(self.template)
            (Path(directory) / "tokenizer.json").write_text("{}")

    save_export_assets(tmp_path, tokenizer=Templated("first"))
    save_export_assets(tmp_path, tokenizer=Templated("second"))
    assert (tmp_path / "chat_templates" / "brief.jinja").read_text() == "second"
    assert (tmp_path / "tokenizer.json").read_text() == "{}"


def test_an_export_past_its_shard_size_writes_shards_that_transformers_reads(tmp_path):
    """Past `max_shard_size` the weights go out as numbered shards and their
    index, which transformers and Dew both read back to the same logits; a
    later export that fits one file removes the shards, so no loader reads
    stale weights beside it."""
    import torch
    from transformers import AutoModelForCausalLM

    source = Pretrained.load(str(FIXTURES / "qwen3-tiny"), dtype="float32", attention_impl="reference")
    ids = np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy")
    expected = np.asarray(source.model.apply(source.variables, ids))
    export = tmp_path / "export"
    source.save(export, max_shard_size=64 * 1024)

    shards = sorted(path.name for path in export.glob("model-*.safetensors"))
    assert len(shards) > 1 and not (export / "model.safetensors").exists()
    index = json.loads((export / "model.safetensors.index.json").read_text())
    assert sorted(set(index["weight_map"].values())) == shards
    reference = AutoModelForCausalLM.from_pretrained(export, dtype=torch.float32, attn_implementation="eager")
    with torch.no_grad():
        theirs = reference(torch.from_numpy(ids.astype(np.int64))).logits.numpy()
    np.testing.assert_allclose(theirs, expected, atol=LOGITS, rtol=0)
    again = Pretrained.load(str(export), dtype="float32", attention_impl="reference")
    np.testing.assert_allclose(
        np.asarray(again.model.apply(again.variables, ids)), expected, atol=LOGITS, rtol=0
    )

    source.save(export)
    assert sorted(path.name for path in export.glob("model*")) == ["model.safetensors"]


SYNTHETIC_CHECKPOINT = """
import sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from dew.interop import Pretrained, PretrainedDecoder

source = Pretrained.load(sys.argv[1], dtype="float32", attention_impl="reference")
model = source.model.clone(num_layers=12, layer_types=("full_attention",) * 12, vocab_size=16000,
                           emb_features=1024, num_heads=16, num_kv_heads=8, head_dim=64, mlp_features=3072)
variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16),
                         model.init(jax.random.key(0), np.zeros((1, 4), np.int32)))
PretrainedDecoder.from_model(model, variables).save(sys.argv[2])
for asset in Path(sys.argv[1]).glob("tokenizer*"):
    (Path(sys.argv[2]) / asset.name).write_bytes(asset.read_bytes())
"""


def test_a_sharded_export_holds_a_few_shards_on_the_host(tmp_path):
    """A 320 MB bfloat16 decoder written by one process, then loaded onto a
    mesh and saved in 50 MB shards by a fresh one
    (tools/benchmark_streaming_export.py), so no heap the writer freed is
    there for the save to reuse. Each host tensor is built as its shard is
    written, so RSS over the save rises by the shard's tensors, their
    serialized bytes and one transient tensor, four shards at most
    (measured 0.12-0.13 GB); building the whole export first rises by the
    model (0.32 GB on this checkpoint, 0.57 GB of Qwen3-0.6B's 1.11)."""
    measured = streamed_export(SYNTHETIC_CHECKPOINT, FIXTURES / "qwen3-tiny", tmp_path, "50MB")
    assert measured["shards"] > 4
    assert measured["rise_gb"] < 4 * 50e6 / 2**30, measured


def test_a_failed_export_ends_with_its_own_error(tmp_path):
    """tools/benchmark_streaming_export.py samples the host's memory from a thread while it saves: a save
    that fails (its directory here a file) ends the process with the save's error, the sampler stopped with
    it, where the sampler, stopped only after a save that succeeded, kept the process from exiting (row
    1579). The deadline is the test's own, past the tiny checkpoint's seconds."""
    out = tmp_path / "out"
    out.write_text("a file where the export's directory goes\n")
    environment = {name: value for name, value in os.environ.items() if name != "XLA_FLAGS"}
    run = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "benchmark_streaming_export.py"),
                          str(FIXTURES / "qwen3-tiny"), "main", str(out)], capture_output=True, text=True,
                         env=environment | {"JAX_PLATFORMS": "cpu", "PYTHONPATH": str(REPO_ROOT / "src")},
                         timeout=300)
    assert run.returncode == 1, run.stderr[-2000:]


def test_a_decoder_stating_a_full_width_rotary_exports_and_transformers_reads_it(tmp_path):
    """A Qwen3 decoder built with `partial_rotary_factor=1.0` rotates its whole head, as with none
    stated: written from the model and its variables alone (`PretrainedDecoder.from_model`), it
    saves, reloads to its own logits, and transformers reads the export to logits Dew's are as
    exact as, both from transformers' in float64."""
    import jax.numpy as jnp
    import torch
    from transformers import AutoModelForCausalLM

    from dew.interop import PretrainedDecoder
    from tools.torch_precision import float64

    loaded = Pretrained.load(FIXTURES / "qwen3-tiny", dtype="float32", attention_impl="reference")
    model = loaded.model.clone(partial_rotary_factor=1.0)
    ids = np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy")
    expected = np.asarray(model.apply(loaded.variables, jnp.asarray(ids)))
    PretrainedDecoder.from_model(model, loaded.variables).save(tmp_path)
    restored = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(np.asarray(restored.model.apply(restored.variables, jnp.asarray(ids))),
                                  expected)

    def theirs(dtype):
        with float64() if dtype == torch.float64 else torch.no_grad():
            reference = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation="eager")
            with torch.no_grad():
                logits = reference.to(dtype).eval()(torch.from_numpy(ids).long(), use_cache=False).logits
        return logits.double().numpy()

    assert_as_exact_as_the_reference(expected.astype(np.float64), theirs(torch.float32),
                                     theirs(torch.float64), "full-width rotary export logits")


def test_a_policy_transformers_cannot_hold_is_refused_not_dropped():
    from dew.interop.generation_config import generation_config_of
    from dew.sampling import Sampling

    with pytest.raises(ValueError):
        generation_config_of(Sampling(presence_penalty=0.5))


def test_a_quantized_source_exports_trained_weights_in_its_original_format(tmp_path):
    """The source config describes real FP8 bytes after a training update."""
    import torch
    from safetensors.torch import load_file, save_file

    directory = tmp_path / "fp8"
    source = FIXTURES / "deepseek-v3-tiny"
    tensors = load_shards(source)
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
    quantized = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    expected = Pretrained.load(
        expected_directory, dtype="float32", attention_impl="reference"
    )
    ids = np.load(source / "input_ids.npy")
    np.testing.assert_array_equal(
        tool.logits(quantized, quantized.variables, ids),
        tool.logits(expected, expected.variables, ids),
    )
    state = tool.train(CASES["deepseek_v3"], quantized, ids)
    destination = tmp_path / "exported"
    quantized.save(destination, variables=state.variables)
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
    decoded = Pretrained.load(plain_directory, dtype="float32", attention_impl="reference")
    reloaded = Pretrained.load(destination, dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(tool.logits(reloaded, reloaded.variables, ids),
                                  tool.logits(decoded, decoded.variables, ids))
    for layout in quantized.weight_layouts:
        if layout.name != scaled:
            np.testing.assert_array_equal(float_export[layout.name], layout.export(state.variables))
    without_provenance = dataclasses.replace(quantized, quantized_tensors=())
    with pytest.raises(ValueError):
        without_provenance.save(tmp_path / "refused", variables=state.variables)


PREPARED = {"gpt2": ("gpt2-tiny", "transformer.h.0.attn.c_attn.weight", ""),
            "gpt2-bare": ("gpt2-tiny", "h.0.attn.c_attn.weight", "transformer."),
            "gptj": ("gptj-tiny", "transformer.h.0.attn.q_proj.weight", ""),
            "gpt-neox": ("gpt-neox-tiny", "gpt_neox.layers.0.attention.query_key_value.weight", ""),
            "modernbert": ("modernbert-tiny", "model.layers.0.attn.Wqkv.weight", "")}
"""Families whose `prepare` rewrites a tensor before it binds, the one each
checkpoint ships in FP8 here, and the prefix its names leave out: GPT-2's
Conv1D qkv, also as the bare GPT2Model stores it without `transformer.`,
GPT-J's query rows in rotary order, GPT-NeoX's and ModernBERT's fused qkv."""


@pytest.mark.parametrize("case", list(PREPARED), ids=list(PREPARED))
def test_a_quantized_source_of_a_preparing_family_exports_updated_weights_in_its_format(case, tmp_path):
    """The family's encoder undoes its preparation (GPT-J's reverse rotary
    order, GPT-2's transpose, fused qkv and nested names) and the format
    encodes the weights it shipped quantized again: the export holds the
    source's own tensor names, each the dense export of the same weights
    with that one tensor in FP8, under the source's config, and Dew reads
    back the weights its decoded export holds. A source placed on a mesh as
    it streams, and one loaded as shapes alone for a resume to fill, write
    the same export of the same weights."""
    from dew.interop.codecs import source_quantization
    from dew.training import MeshSpec

    fixture, name, bare = PREPARED[case]
    tensors = {stored.removeprefix(bare): tensor
               for stored, tensor in load_shards(FIXTURES / fixture).items()}
    config = json.loads((FIXTURES / fixture / "config.json").read_text())
    plain = dict(config)
    config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    quantization = source_quantization(config)
    assert quantization is not None
    source = quantization.requantize(tensors, (name,))
    save_hf_layout(source, config, tmp_path / "fp8")
    loaded = Pretrained.load(tmp_path / "fp8", dtype="float32", attention_impl="reference")
    assert loaded.quantized_tensors == (name,)
    updated = jax.tree.map(
        lambda leaf: leaf * 1.03 + 1e-3 if np.issubdtype(leaf.dtype, np.floating) else leaf, loaded.variables)

    loaded.save(tmp_path / "export", variables=updated)
    for placement, options in (("placed", {"mesh": MeshSpec(fsdp=1)}), ("shapes", {"weights": False})):
        again = Pretrained.load(tmp_path / "fp8", dtype="float32", attention_impl="reference", **options)
        assert jax.tree.structure(again.variables) == jax.tree.structure(loaded.variables)
        again.save(tmp_path / placement, variables=updated)
        assert_same_stored_tensors(load_shards(tmp_path / placement),
                                   load_shards(tmp_path / "export"))
    dense = {stored.removeprefix(bare): tensor for stored, tensor in
             dataclasses.replace(loaded, config=plain, quantized_tensors=()).export(updated).items()}
    expected = quantization.requantize(dense, (name,))
    exported = load_shards(tmp_path / "export")
    assert json.loads((tmp_path / "export" / "config.json").read_text()) == config
    assert set(exported) == set(expected) == set(source)
    for tensor, value in expected.items():
        assert exported[tensor].dtype == value.dtype, tensor
        np.testing.assert_array_equal(exported[tensor], value, err_msg=tensor)
    save_hf_layout({**dense, name: quantization.decode(exported, name)}, plain, tmp_path / "decoded")
    reloaded = Pretrained.load(tmp_path / "export", dtype="float32", attention_impl="reference")
    decoded = Pretrained.load(tmp_path / "decoded", dtype="float32", attention_impl="reference")
    assert_restores(decoded.model, decoded.variables, reloaded, np.load(FIXTURES / fixture / "input_ids.npy"))


def test_mxfp4_source_reexports_the_trained_experts_and_preserves_float_tensors(tmp_path):
    """A GPT OSS source shipped MXFP4 writes its trained experts back as
    blocks and scales the released reader decodes, and every float tensor
    -- biases, router, sinks, embeddings -- as itself."""
    import torch
    from safetensors.numpy import load_file
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    from dew.interop.codecs import MXFP4

    source = FIXTURES / "gpt-oss-tiny"
    tensors = load_file(str(source / "model.safetensors"))
    stems = tuple(name for name in tensors if name.endswith((".experts.gate_up_proj", ".experts.down_proj")))
    packed = MXFP4.requantize(tensors, stems)
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "mxfp4"}
    directory = tmp_path / "source"
    save_hf_layout(packed, config, directory)
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    ids = np.load(source / "input_ids.npy")
    state = tool.train(tool.Case("gpt_oss", "gpt-oss-tiny"), loaded, ids)
    destination = tmp_path / "export"
    loaded.save(destination, variables=state.variables)
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
            np.testing.assert_array_equal(decoded[layout.name], layout.export(state.variables))
    plain_directory = tmp_path / "decoded"
    plain_config = {name: value for name, value in config.items() if name != "quantization_config"}
    save_hf_layout(decoded, plain_config, plain_directory)
    plain = Pretrained.load(plain_directory, dtype="float32", attention_impl="reference")
    reloaded = Pretrained.load(destination, dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(tool.logits(plain, plain.variables, ids),
                                  tool.logits(reloaded, reloaded.variables, ids))
    missing = dataclasses.replace(loaded, quantized_tensors=())
    with pytest.raises(ValueError):
        missing.save(tmp_path / "missing-provenance", variables=state.variables)


def test_an_mtp_copy_that_differs_from_the_trunk_names_the_tensor(tmp_path):
    """The depth shares the trunk's head, so a checkpoint whose copy is a
    different matrix is refused naming the tensor rather than loaded as a
    model that computes something else."""
    source = FIXTURES / "glm4-moe-tiny"
    tensors = load_shards(source)
    copy = "model.layers.2.shared_head.head.weight"
    tensors[copy] = tensors[copy] + 1.0
    directory = tmp_path / "broken"
    save_hf_layout(tensors, json.loads((source / "config.json").read_text()), directory)

    with pytest.raises(ValueError):
        Pretrained.load(str(directory), dtype="float32", attention_impl="reference")


def test_a_shared_copy_cannot_hide_an_undeclared_prediction_depth(tmp_path):
    source = FIXTURES / "glm4-moe-tiny"
    config = json.loads((source / "config.json").read_text())
    tensors = load_shards(source)
    first_absent = config["num_hidden_layers"] + config["num_nextn_predict_layers"]
    name = f"model.layers.{first_absent}.embed_tokens.weight"
    tensors[name] = tensors["model.embed_tokens.weight"]
    directory = tmp_path / "undeclared-depth"
    save_hf_layout(tensors, config, directory)
    with pytest.raises(ValueError):
        Pretrained.load(directory, dtype="float32", attention_impl="reference")


@pytest.mark.parametrize("name", ["deepseek_v2", "deepseek_v3", "deepseek_v32"])
def test_the_latent_norms_keep_the_reference_epsilon(name, tmp_path):
    """DeepSeek's q_a_layernorm and kv_a_layernorm run RMSNorm's 1e-6
    whatever rms_norm_eps configures: a checkpoint whose trunk epsilon is
    1e-3 still parities the reference, so the mixer's epsilon is its own."""
    source = FIXTURES / CASES[name].fixture
    config = json.loads((source / "config.json").read_text())
    config["rms_norm_eps"] = 1e-3
    directory = tmp_path / name
    save_hf_layout(load_shards(source), config, directory)
    ids = np.load(source / "input_ids.npy")

    loaded = Pretrained.load(str(directory), dtype="float32", attention_impl="reference")
    np.testing.assert_allclose(tool.logits(loaded, loaded.variables, ids),
                               tool.reference_logits(CASES[name], directory, ids),
                               atol=LOGITS, rtol=0)


@pytest.mark.parametrize("kind", ["fp8", "mxfp4"])
def test_codec_rejects_integer_parameter_storage(kind):
    from dew.interop.codecs import MXFP4, fp8_blocks

    if kind == "fp8":
        packed = {"weight": np.ones((1, 1), np.float32),
                  "weight_scale_inv": np.full((1, 1), 1.25, np.float32)}
        with pytest.raises(ValueError):
            fp8_blocks(1).dequantize(packed, param_dtype="int32")
    else:
        packed = MXFP4.requantize({"weight": np.full((1, 32, 2), 1.5, np.float32)}, ("weight",))
        with pytest.raises(ValueError):
            MXFP4.dequantize(packed, param_dtype="int32")


@pytest.mark.parametrize("kind", ["fp8", "mxfp4"])
def test_public_quantized_load_obeys_parameter_storage(tmp_path, kind):
    from test_interop import assert_parameter_storage

    from dew.interop.codecs import MXFP4, fp8_blocks

    fixture = FIXTURES / ("deepseek-v3-tiny" if kind == "fp8" else "gpt-oss-tiny")
    tensors = load_shards(fixture)
    config = json.loads((fixture / "config.json").read_text())
    if kind == "fp8":
        packed = fp8_blocks(128, ue8m0=False).requantize(tensors, ("model.layers.0.self_attn.o_proj.weight",))
        config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                         "weight_block_size": [128, 128]}
    else:
        stems = tuple(name for name in tensors if name.endswith(
            (".experts.gate_up_proj", ".experts.down_proj")))
        packed = MXFP4.requantize(tensors, stems)
        config["quantization_config"] = {"quant_method": "mxfp4"}
    directory = tmp_path / "quantized"
    save_hf_layout(packed, config, directory)
    masters = Pretrained.load(directory, dtype="bfloat16", attention_impl="xla")
    native = Pretrained.load(directory, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
    assert masters.quantized_tensors == native.quantized_tensors
    assert_parameter_storage(masters.variables, native.variables, lambda path: path[0] == "params")


@pytest.mark.parametrize("fixture", ["deepseek-v3-tiny", "kimi-k25-tiny"])
@pytest.mark.parametrize(
    "same_values", [True, False], ids=["equal-before-rounding", "different-before-rounding"]
)
def test_public_quantized_alias_check_uses_original_fp32_values(tmp_path, fixture, same_values):
    """An FP8 tied head and its float32 embedding are one tensor only if
    they hold equal values before narrowing to bfloat16 storage, under the
    names the source's own family gives them. Kimi K2.5 computes what
    DeepSeek V3 does and names its head and embedding under
    `language_model.`."""
    from dew.interop.codecs import E4M3
    from dew.interop.hf_decoders import families

    tensors = load_shards(FIXTURES / fixture)
    config = json.loads((FIXTURES / fixture / "config.json").read_text())
    head, embedding = families()[config["model_type"]].tied_head_names
    for tied in (config, config.get("text_config", {})):
        tied["tie_word_embeddings"] = True
    config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                     "weight_block_size": [128, 128]}
    shape = tensors[embedding].shape
    decoded = np.float32(1. + 1. / 1024)
    unquantized = decoded if same_values else np.float32(1. + 2. / 1024)
    tensors[embedding] = np.full(shape, unquantized, np.float32)
    tensors[head] = np.ones(shape, dtype=E4M3)
    tensors[f"{head}_scale_inv"] = np.full(
        ((shape[0] + 127) // 128, (shape[1] + 127) // 128), decoded, np.float32)
    for name in tensors:
        if name.startswith("model.layers.") and name.endswith(".embed_tokens.weight"):
            tensors[name] = np.full_like(tensors[name], unquantized, dtype=np.float32)
        elif name.startswith("model.layers.") and name.endswith(".shared_head.head.weight"):
            tensors[name] = np.full_like(tensors[name], decoded, dtype=np.float32)
    directory = tmp_path / "mixed-aliases"
    save_hf_layout(tensors, config, directory)
    if not same_values:
        with pytest.raises(ValueError):
            Pretrained.load(directory, param_dtype="bfloat16")
    else:
        loaded = Pretrained.load(directory, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
        values = loaded.variables["params"]["embed_tokens"]["embedding"]
        assert np.asarray(values).dtype == ml_dtypes.bfloat16
        np.testing.assert_array_equal(values, np.full(shape, decoded).astype(ml_dtypes.bfloat16))


def test_a_tied_head_stored_with_its_depths_copies_and_no_trunk_embedding_loads_as_tied(tmp_path):
    """GLM's released layout stores each prediction depth's copies of the
    trunk embedding and head. A tied source that keeps those copies and the
    head but no trunk embedding still holds the embedding twice over; it
    loads leaf for leaf as the same source with the trunk embedding."""
    fixture = FIXTURES / "glm4-moe-tiny"
    tensors = load_shards(fixture)
    config = json.loads((fixture / "config.json").read_text())
    config["tie_word_embeddings"] = True
    embedding = tensors["model.embed_tokens.weight"]
    depth = f"model.layers.{config['num_hidden_layers']}"
    tensors.update({"lm_head.weight": embedding.copy(), f"{depth}.embed_tokens.weight": embedding.copy(),
                    f"{depth}.shared_head.head.weight": embedding.copy()})
    save_hf_layout(tensors, config, tmp_path / "trunk")
    save_hf_layout({name: tensor for name, tensor in tensors.items() if name != "model.embed_tokens.weight"},
                   config, tmp_path / "depths")
    trunk = Pretrained.load(tmp_path / "trunk", dtype="float32", attention_impl="xla")
    depths = Pretrained.load(tmp_path / "depths", dtype="float32", attention_impl="xla")
    assert jax.tree.structure(depths.variables) == jax.tree.structure(trunk.variables)
    for ours, theirs in zip(jax.tree.leaves(depths.variables), jax.tree.leaves(trunk.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))


COPIES = {"equal": (1. + 1. / 1024, "float32", True), "different": (1. + 2. / 1024, "float32", False),
          "equal-narrowed": (1. + 1. / 1024, "bfloat16", False)}
"""A dense copy of a gate whose FP8 packing decodes to 1 + 1/1024, the
storage it loads at, and whether it loads."""


@pytest.mark.parametrize("copy, storage, loads", list(COPIES.values()), ids=list(COPIES))
def test_a_quantized_packed_tensor_and_its_part_stored_apart_must_be_one_tensor(
        tmp_path, copy, storage, loads):
    """Phi-3's gate_up_proj stored in FP8 decodes to 1 + 1/1024, beside a
    dense float32 gate_proj. At float32 storage the decode is the dense
    copy's tensor: an equal copy loads as the packed tensor alone does, and
    1 + 2/1024 is refused. At bfloat16 storage the codec has rounded the
    packed one, so even an equal copy is a different tensor and is refused
    by name rather than loading whichever copy came first."""
    from dew.interop.codecs import E4M3

    fixture = FIXTURES / "phi3-tiny"
    tensors = load_shards(fixture)
    config = json.loads((fixture / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    packed, part = "model.layers.0.mlp.gate_up_proj.weight", "model.layers.0.mlp.gate_proj.weight"
    shape = tensors[packed].shape
    tensors[packed] = np.ones(shape, dtype=E4M3)
    tensors[packed + "_scale_inv"] = np.full(((shape[0] + 127) // 128, (shape[1] + 127) // 128),
                                             1. + 1. / 1024, np.float32)
    save_hf_layout(tensors, config, tmp_path / "alone")
    save_hf_layout({**tensors, part: np.full((shape[0] // 2, shape[1]), copy, np.float32)}, config,
                   tmp_path / "apart")
    if not loads:
        with pytest.raises(ValueError):
            Pretrained.load(tmp_path / "apart", dtype="float32", param_dtype=storage, attention_impl="xla")
        return
    apart = Pretrained.load(tmp_path / "apart", dtype="float32", param_dtype=storage, attention_impl="xla")
    alone = Pretrained.load(tmp_path / "alone", dtype="float32", param_dtype=storage, attention_impl="xla")
    assert jax.tree.structure(apart.variables) == jax.tree.structure(alone.variables)
    for ours, theirs in zip(jax.tree.leaves(apart.variables), jax.tree.leaves(alone.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    # Without its weights the copy is checked by its shape, and the load holds what a load with them does.
    shapes = Pretrained.load(tmp_path / "apart", dtype="float32", param_dtype=storage, attention_impl="xla",
                             weights=False)
    assert jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), shapes.variables) == jax.tree.map(
        lambda leaf: (leaf.shape, leaf.dtype), apart.variables)


def test_public_quantized_diffusion_gemma_rejects_rounded_shared_copies(tmp_path):
    from dew.interop.codecs import E4M3

    fixture = FIXTURES / "diffusion-gemma-workflow"
    tensors = load_shards(fixture)
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
    with pytest.raises(ValueError):
        Pretrained.load(directory, param_dtype="bfloat16")


def test_diffusion_gemma_source_export_resolves_omitted_embedding_tie_default(tmp_path):
    from shutil import copytree

    from dew.interop.diffusion_gemma import export_weights

    source = copytree(FIXTURES / "diffusion-gemma-workflow", tmp_path / "source")
    config = json.loads((source / "config.json").read_text())
    del config["text_config"]["tie_word_embeddings"]
    (source / "config.json").write_text(json.dumps(config))
    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    destination = tmp_path / "export"
    loaded.save(destination)
    restored = Pretrained.load(destination, dtype="float32", attention_impl="reference")
    for expected, actual in zip(jax.tree.leaves(loaded.variables),
                                jax.tree.leaves(restored.variables), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for stated in (False, None):
        # An explicit null is the reference's own False, not the family's
        # tied default (configuration_utils.py reads the key it is given).
        disagreeing = {**config, "text_config": {**config["text_config"],
                                                 "tie_word_embeddings": stated}}
        with pytest.raises(ValueError):
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

    from dew.interop import PretrainedDecoder

    loaded = Pretrained.load(FIXTURES / fixture, dtype="float32", attention_impl="reference")
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
    PretrainedDecoder.from_model(model, variables).save(tmp_path)

    restored = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
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


@pytest.mark.parametrize("fixture, trainable", [
    ("gemma4-moe-tiny", lambda model: model.clone(layer_scalar="trainable")),
    ("gemma4-native-tiny", lambda model: model.clone(
        language_model=model.language_model.clone(layer_scalar="trainable"))),
], ids=["decoder", "wrapper"])
def test_a_loaded_gemma4_writes_its_trained_layer_scalars_under_their_source_names(fixture, trainable,
                                                                                  tmp_path):
    """A loaded Gemma 4, alone or inside its media wrapper, saves through the
    source names it was read from. Trained, its layer scalars sit in
    `params`, and those values are what the export's buffers carry: the
    reload holds them under `constants` and computes the trained logits."""
    import jax.numpy as jnp
    from flax.core import unfreeze
    from flax.traverse_util import flatten_dict, unflatten_dict

    loaded = Pretrained.load(FIXTURES / fixture, dtype="float32", attention_impl="reference")
    model = trainable(loaded.model)
    variables = unfreeze(loaded.variables)
    constants, params = flatten_dict(variables["constants"]), flatten_dict(variables["params"])
    scalars = sorted(path for path in constants if path[-1] == "layer_scalar")
    assert scalars
    for index, path in enumerate(scalars):
        params[path] = jnp.full_like(constants.pop(path), 0.75 + index * 0.125)
    variables.update(constants=unflatten_dict(constants), params=unflatten_dict(params))
    ids = jnp.arange(2, 10, dtype=jnp.int32)[None, :]
    expected = np.asarray(jax.jit(model.apply)(variables, ids))

    dataclasses.replace(loaded, model=model).save(tmp_path, variables=variables)

    restored = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    held = flatten_dict(restored.variables["constants"])
    for path in scalars:
        np.testing.assert_array_equal(held[path], params[path])
    actual = np.asarray(jax.jit(restored.model.apply)(restored.variables, ids))
    np.testing.assert_array_equal(actual, expected)


def glm5_native_export_case(variant):
    import jax.numpy as jnp
    from flax.core import unfreeze

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.dsa_kpool import KPoolSparseAttentionMixer
    from dew.nn.kda import KimiDeltaAttentionMixer

    source = Pretrained.load(FIXTURES / "glm5-next-tiny", dtype="float32", attention_impl="reference")
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
        model = model.clone(
            kinds=kinds,
            tie_embeddings=True,
            index_share_for_mtp_iteration=False,
            mixture=dataclasses.replace(model.mixture, layers=(1, 3, 4), norm_topk_prob=False),
        )
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


@pytest.mark.parametrize(("fixture", "changed"), [
    ("gemma4-ple", {"v_norm": False}),
    ("gemma4-ple", {"qk_norm_scope": "full"}),
    ("gemma4-ple", {"embedding_scale": False}),
    ("gemma4-moe-tiny", {"mixture.capacity_factor": 1.25}),
    ("glm5-next-tiny", {"sandwich_norms": True}),
    ("glm5-next-tiny", {"mlp": "geglu"}),
    ("glm5-next-tiny", {"mixture.bias": False}),
    ("gemma4-ple", {"scale_after_cast": True}),
    ("glm5-next-tiny", {"layer_scalar": "frozen"}),
])
def test_a_model_its_family_config_cannot_carry_is_refused_at_export(fixture, changed):
    """A computation the family's config does not carry, a norm, an MLP or a router it fixes,
    reads back as another model, so its export is refused wherever the config is written; so
    is a way of running it the config does not record and the family fixes."""
    from dew.interop import PretrainedDecoder
    from dew.interop.hf_decoders import export_config

    loaded = Pretrained.load(FIXTURES / fixture, dtype="float32", attention_impl="reference")
    # Gemma 4 exports its layer scalars as frozen or trainable, which a loaded model leaves open.
    exportable = loaded.model.clone(layer_scalar="frozen") if fixture.startswith("gemma4") else loaded.model
    model = exportable
    for name, value in changed.items():
        if name.startswith("mixture."):
            routed = dataclasses.replace(model.mixture, **{name.removeprefix("mixture."): value})
            model = model.clone(mixture=routed)
        else:
            model = model.clone(**{name: value})
    export_config(exportable)
    with pytest.raises(ValueError):
        export_config(model)
    with pytest.raises(ValueError):
        PretrainedDecoder.from_model(model, loaded.variables)


@pytest.mark.parametrize("variant", timed(PAST_TARGET, "glm5", ["released", "native_geometry", "dense"]))
def test_standalone_glm5_export_preserves_native_and_source_computation(variant, tmp_path):
    import jax.numpy as jnp

    from dew.interop import PretrainedDecoder
    from dew.sampling import Sampling, Speculative, generate

    model, variables, ids = glm5_native_export_case(variant)
    expected = np.asarray(jax.jit(model.apply)(variables, jnp.asarray(ids)))
    cache = model.apply(variables, ids.shape[0], method="init_cache", mutable=["cache"])[1]
    if model.num_nextn_predict_layers:
        cache = model.apply({**variables, **cache}, ids.shape[0], method="init_mtp_cache", mutable=["cache"])[
            1
        ]
    PretrainedDecoder.from_model(model, {**variables, **cache}).save(tmp_path)
    restored = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    actual = np.asarray(jax.jit(restored.model.apply)(restored.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=LOGITS, rtol=0)
    np.testing.assert_allclose(tool.reference_logits(CASES["glm5_next"], tmp_path, ids),
                               expected, atol=LOGITS, rtol=0)
    written = json.loads((tmp_path / "config.json").read_text())
    assert written["index_share_for_mtp_iteration"] == model.index_share_for_mtp_iteration
    if model.num_nextn_predict_layers:
        # Each forward compiled once, as a user runs it, not op by op.
        def depths(model, variables):
            states = model.apply(variables, jnp.asarray(ids), method="hidden_states")
            return model.apply(variables, states, jnp.asarray(ids), method="mtp_logits")[0]
        prediction = jax.jit(functools.partial(depths, model))(variables)
        reloaded_prediction = jax.jit(functools.partial(depths, restored.model))(restored.variables)
        np.testing.assert_allclose(reloaded_prediction, prediction, atol=LOGITS, rtol=0)
        np.testing.assert_allclose(tool.glm5_prediction_logits(CASES["glm5_next"], tmp_path, ids),
                                   prediction, atol=LOGITS, rtol=0)
        ordinary = generate(model, variables, jnp.asarray(ids[:, :5]), 4, key=jax.random.key(0),
                            sampling=Sampling(temperature=0)).host()
        speculative = restored.text_generation(sampling=Sampling(temperature=0))(
            ids[:, :5], max_new_tokens=4, key=0, strategy=Speculative(block=3)).host()
        np.testing.assert_array_equal(speculative.tokens, ordinary.tokens)
        np.testing.assert_array_equal(speculative.lengths, ordinary.lengths)
    for collection in ("params", "moe"):
        if collection in variables:
            actual_leaves = flat_tree(restored.variables[collection])
            for name, value in flat_tree(variables[collection]).items():
                np.testing.assert_array_equal(actual_leaves[name], value)


def assert_same_tensors(written: dict, expected: dict) -> None:
    assert set(written) == set(expected)
    for tensor, value in expected.items():
        value = np.asarray(value)
        assert written[tensor].dtype == value.dtype and np.array_equal(written[tensor], value), tensor


@pytest.mark.parametrize("name", ["qwen3-moe-tiny", "deepseek-v3-tiny", "glm5-next-tiny"])
def test_a_moe_decoder_saves_every_tensor_its_source_holds(name, tmp_path):
    """A causal MoE decoder saved in its source's layout writes every tensor the source checkpoint
    holds, each expert's projections split from their stacked kernel and DeepSeek V3's balancing
    bias from the `moe` collection, bit for bit, and its config keeps `use_cache`. GLM-5 Next's is the
    source-layout route its trained round trip exports by, every train while that runs in the whole
    suite alone (`WHOLE_SUITE`)."""
    loaded = Pretrained.load(FIXTURES / name, dtype="float32", param_dtype="auto", attention_impl="xla")
    loaded.save(tmp_path)
    assert_same_tensors(written_tensors(tmp_path), written_tensors(FIXTURES / name))
    assert json.loads((tmp_path / "config.json").read_text())["use_cache"] is True


def test_a_moe_decoder_written_from_its_model_holds_its_sources_tensors(tmp_path):
    """A mixture-of-experts decoder written from its model and variables alone
    (`PretrainedDecoder.from_model`, the native export its family writes by, `decoder_tensors`) writes
    every tensor its source checkpoint holds, bit for bit: LLaDA2's experts' projections sliced from
    their stacked kernels by the plan the load stacked them by (`decoder_parts.expert_path`), its query,
    key and value joined into their unequal-width pack, each oriented as the load read it
    (`decoder_parts.stored_order`)."""
    loaded = Pretrained.load(FIXTURES / "llada2-tiny", dtype="float32", param_dtype="auto",
                             attention_impl="xla")
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    assert_same_tensors(written_tensors(tmp_path), written_tensors(FIXTURES / "llada2-tiny"))


def test_a_dense_decoder_written_from_its_model_holds_its_sources_tensors(tmp_path):
    """A causal decoder written from its model and variables alone (`PretrainedDecoder.from_model`,
    the encoder every family's derived export shares) writes every tensor its source checkpoint holds,
    bit for bit, and records `use_cache` as a causal decoder's config does."""
    loaded = Pretrained.load(FIXTURES / "qwen3-tiny", dtype="float32", param_dtype="auto",
                             attention_impl="xla")
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    assert_same_tensors(written_tensors(tmp_path), written_tensors(FIXTURES / "qwen3-tiny"))
    assert json.loads((tmp_path / "config.json").read_text())["use_cache"] is True
