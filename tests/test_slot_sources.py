"""Released slot decisions through the native composite loader and task export."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
from sharded import assert_sharded

from dew.decision import Choice, Decide
from dew.distributed.placement import Layout, MeshSpec
from dew.interop import Pretrained

BACKBONE = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"
REFERENCE = Path(__file__).parent / "fixtures" / "slot_readout" / "tiny"


def slot_release(tmp_path, readout, n_slots=32):
    import torch

    source = tmp_path / "release"
    shutil.copytree(BACKBONE, source)
    config = {"readout": "slot_lm" if readout == "slot_lm" else "head",
              "head_dim": 8, "n_slots": n_slots}
    (source / "readout.json").write_text(json.dumps(config))
    stored = {"slot": np.arange(32 * 32, dtype=np.float32).reshape(32, 32) / 1024,
              "q.weight": np.arange(8 * 32, dtype=np.float32).reshape(8, 32) / 256,
              "k.weight": np.arange(8 * 32, dtype=np.float32).reshape(8, 32) / 512}
    if readout == "head":
        torch.save({name: torch.from_numpy(value) for name, value in stored.items()},
                   source / "readout_head.pt")
    return source, stored


@pytest.mark.parametrize("readout", ["slot_lm", "head", "fresh_head"])
@pytest.mark.parametrize("n_slots", [16, 32, 512])
def test_a_slot_release_loads_its_readout_and_exports_the_same_task(tmp_path, readout, n_slots):
    """The stored head keeps all its rows and native q/k orientation. A fresh head draws only
    the available label rows; the byte tokenizer has all letters but no single-token pairs.
    All three become decision tasks and survive a native task export with their parameters."""
    source, stored = slot_release(tmp_path, readout, n_slots)
    bundle = Pretrained.load(source, dtype="float32", attention_impl="xla")
    task = Decide.of(bundle)
    [row] = task.layout.rows(task.tokenizer, task.specials, "Refund requested.",
                            {"team": Choice("Which team?", ["billing", "technical"])})
    task.model.head.validate_rows([row])
    assert row.questions[0].label_ids == tuple((task.tokenizer.encode(label)[0],) for label in "AB")
    if readout == "head":
        for name, native in (("slot", "slots"), ("q.weight", "q"), ("k.weight", "k")):
            np.testing.assert_array_equal(bundle.head_variables["params"][native], stored[name])
    elif readout == "fresh_head":
        labels = [task.tokenizer.encode(label)[0] for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_slots]]
        table = task.model.table(task.variables)
        np.testing.assert_array_equal(bundle.head_variables["params"]["slots"], np.asarray(table)[labels])
        assert np.count_nonzero(bundle.head_variables["params"]["q"]) > 0
        assert np.count_nonzero(bundle.head_variables["params"]["k"]) == 0
    else:
        assert not jax.tree.leaves(bundle.head_variables)
    shapes = Pretrained.load(source, dtype="float32", attention_impl="xla", weights=False)
    host = Pretrained.load(source, dtype="float32", attention_impl="xla", on_host=True)
    for abstract, held, original in zip(jax.tree.leaves((shapes.variables, shapes.head_variables)),
                                       jax.tree.leaves((host.variables, host.head_variables)),
                                       jax.tree.leaves((bundle.variables, bundle.head_variables)),
                                       strict=True):
        assert isinstance(abstract, jax.ShapeDtypeStruct)
        assert isinstance(held, np.ndarray)
        assert abstract.shape == held.shape == original.shape
        assert abstract.dtype == held.dtype == original.dtype
        np.testing.assert_array_equal(held, original)
    exported = tmp_path / "exported"
    bundle.save(exported)
    restored = Decide.from_pretrained(exported, dtype="float32", attention_impl="xla")
    overflow = Choice("Which one?", [f"option{index}" for index in range(min(n_slots, 26) + 1)])
    for reader in (task, restored):
        with pytest.raises(ValueError):
            reader.layout.rows(reader.tokenizer, reader.specials, "state", {"overflow": overflow})
    assert restored.layout.rows(restored.tokenizer, restored.specials, "Refund requested.",
                                {"team": Choice("Which team?", ["billing", "technical"])}) == [row]
    for actual, expected in zip(jax.tree.leaves(restored.variables), jax.tree.leaves(task.variables),
                               strict=True):
        np.testing.assert_array_equal(actual, expected)


_REOPENED = """
import sys
import numpy as np
from dew.interop import Pretrained
bundle = Pretrained.load("dew/tiny-slot", dtype="float32", attention_impl="xla")
np.savez(sys.argv[1], revision=np.asarray(bundle.revision),
         **{name: np.asarray(value) for name, value in bundle.head_variables["params"].items()})
"""


def test_a_cached_slot_release_reopens_with_the_hub_offline(tmp_path):
    """A Hub slot release whose readout file the cache holds at its commit reopens with the Hub
    offline: the pinned snapshot holding the file is its presence, so nothing asks the Hub."""
    import os
    import subprocess
    import sys

    from interop_support import hub_cache

    source, stored = slot_release(tmp_path, "head")
    commit = hub_cache(tmp_path / "hub", "dew/tiny-slot", source)
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HUB_OFFLINE": "1", "JAX_PLATFORMS": "cpu"}
    run = subprocess.run([sys.executable, "-c", _REOPENED, str(tmp_path / "head.npz")], capture_output=True,
                         text=True, env=env)
    assert run.returncode == 0, run.stderr[-2000:]
    with np.load(tmp_path / "head.npz") as head:
        assert str(head["revision"]) == commit
        for name, native in (("slot", "slots"), ("q.weight", "q"), ("k.weight", "k")):
            np.testing.assert_array_equal(head[native], stored[name])


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("readout", ["slot_lm", "head", "fresh_head"])
@pytest.mark.parametrize("n_slots", [16, 32, 512])
def test_a_loaded_readout_uses_its_own_layout_beside_the_backbone(tmp_path, readout, n_slots):
    source, _ = slot_release(tmp_path, readout, n_slots)
    bundle = Pretrained.load(source, dtype="float32", attention_impl="xla", on_host=True)
    mesh = MeshSpec(fsdp=2).build(devices=jax.devices()[:2])
    for minimum in (1, 2048):
        placed = Pretrained.load(source, dtype="float32", attention_impl="xla", mesh=mesh,
                                 layout=Layout(min_shard=minimum, tolerance=1.0))
        assert_sharded(placed.variables, mesh)
        if minimum == 1 and readout != "slot_lm":
            assert_sharded(placed.head_variables, mesh)
        if minimum == 2048:
            assert all(leaf.is_fully_replicated for leaf in jax.tree.leaves(placed.head_variables))
        for actual, expected in zip(jax.tree.leaves((placed.variables, placed.head_variables)),
                                   jax.tree.leaves((bundle.variables, bundle.head_variables)), strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("readout", ["head", "fresh_head"])
def test_a_bfloat16_readout_keeps_its_storage_through_the_released_format(tmp_path, readout):
    import jax.numpy as jnp

    source, _ = slot_release(tmp_path, readout, 16)
    bundle = Pretrained.load(source, dtype="bfloat16", param_dtype="bfloat16", attention_impl="xla")
    assert all(leaf.dtype == jnp.bfloat16 for leaf in jax.tree.leaves(bundle.head_variables))
    exported = tmp_path / "exported"
    bundle.save(exported)
    restored = Pretrained.load(exported, dtype="bfloat16", param_dtype="auto", attention_impl="xla")
    for actual, expected in zip(jax.tree.leaves((restored.variables, restored.head_variables)),
                               jax.tree.leaves((bundle.variables, bundle.head_variables)), strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_a_block_slot_source_trains_checkpoints_exports_and_serves_its_updated_readout(tmp_path):
    import optax
    from decision_support import decision_rows

    from dew.batch import part
    from dew.checkpoints import Checkpoints
    from dew.decision import DecisionObjective, Example
    from dew.inference import ScoringServer
    from dew.training import Trainer

    source, _ = slot_release(tmp_path, "head", 16)
    config = json.loads((source / "config.json").read_text())
    config["modeling_overrides"] = {
        "readout": "head", "head_dim": 8, "n_slots": 16,
        "attention": {"full_attention": "block_bidirectional", "linear_attention": "causal"}}
    (source / "config.json").write_text(json.dumps(config))
    (source / "readout.json").unlink()
    bundle = Pretrained.load(source, dtype="float32", attention_impl="xla")
    task = bundle.pipeline()
    task = replace(task, layout=replace(task.layout, max_len=512))
    request = {"state": "Refund requested.", "questions": {
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": None, "technical": None}}}}
    examples = [Example.of({**request, "answers": {"team": 0}})]
    objective = DecisionObjective(task, shuffle_options=False)
    data = decision_rows(examples, 1, objective=objective.data())
    mesh = MeshSpec().build(devices=jax.devices()[:1])
    with Checkpoints(str(tmp_path / "run"), keep=1) as checkpoints:
        trained = Trainer(objective, optax.sgd(1e-4), key=7, mesh=mesh, checkpoints=checkpoints).fit(
            data, steps=1, checkpoint_every=1, log_every=100)
        checkpoints.wait()
        loaded = Decide.from_run(str(tmp_path / "run"))
    for component in ("backbone", "head"):
        assert any(not np.array_equal(before, after) for before, after in zip(
            jax.tree.leaves(part(task.variables, component)),
            jax.tree.leaves(part(trained.variables, component)), strict=True))
    updated = replace(bundle, variables=part(trained.variables, "backbone"),
                      head_variables=part(trained.variables, "head"))
    updated.save(tmp_path / "updated")
    exported = Decide.from_pretrained(tmp_path / "updated", dtype="float32", attention_impl="xla")
    question = {"team": Choice("Which team?", ["billing", "technical"])}
    rows = loaded.layout.rows(loaded.tokenizer, loaded.specials, request["state"], question)
    np.testing.assert_array_equal(loaded.scored(rows), exported.scored(rows))
    server = ScoringServer.from_task(*exported.scoring())
    first, second = server.submit(request), server.submit(request)
    server.step()
    assert not server.busy
    assert first.result() == second.result()


class HostRows:
    """A host array with a transfer budget: a native gather may cross JAX's array boundary,
    the complete vocabulary may not. Both paths read the same real NumPy values."""

    def __init__(self, values):
        self.values = values
        self.dtype = values.dtype

    def __jax_array__(self, dtype=None):
        raise AssertionError("initializing a few slots must not transfer the entire host vocabulary")

    def take(self, indices, axis=None):
        return self.values.take(indices, axis=axis)


def test_a_slot_draw_transfers_only_selected_host_rows_and_matches_a_device_draw():
    import jax.numpy as jnp
    from test_decision_readout import ids, rows_of

    from dew.decision import DecisionInputs
    from dew.decision.head import SlotReadout

    table = np.arange(256 * 32, dtype=np.float32).reshape(256, 32) / 8192
    head = SlotReadout(32, tuple(row[0] for row in ids()), head_dim=8, slot_rows=8)
    inputs = DecisionInputs.collate(rows_of(0), 0)
    states = jnp.zeros((*inputs.tokens.shape, 32), jnp.float32)
    key = jax.random.key(3)
    host = head.init(key, states, inputs, table)
    limited = head.init(key, states, inputs, HostRows(table))
    device = head.init(key, states, inputs, jnp.asarray(table))
    for actual, constrained, expected in zip(jax.tree.leaves(host), jax.tree.leaves(limited),
                                            jax.tree.leaves(device), strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(constrained, actual)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(host["params"]["slots"][:3], table[[65, 66, 67]])


@pytest.mark.parametrize("mode", ["slot_lm", "head"])
@pytest.mark.parametrize("attention", ["causal", "block"])
def test_the_loaded_slot_runtime_matches_author_scores_and_readout_vjps(tmp_path, mode, attention):
    import jax.numpy as jnp
    from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording

    from dew.batch import part
    from dew.decision import DecisionInputs, Question

    source = tmp_path / "source"
    shutil.copytree(REFERENCE, source)
    (source / "readout.json").write_text(json.dumps({
        "readout": mode, "head_dim": 8, "n_slots": 16,
        "attention": "noncausal" if attention == "block" else "causal"}))
    bundle = Pretrained.load(source, dtype="float32", attention_impl="reference")
    task = Decide.of(bundle)
    path = REFERENCE / "reference.npz"
    site = f"{attention}_{mode}"
    requests = json.loads((REFERENCE / "requests.json").read_text())
    seeds = declared_seeds(path)[site]
    draw, reference, truth = indexed_recording(recording(path), site, seeds)
    head = task.model.head

    @jax.jit
    def forward(inputs, probe):
        states = task.model.states(task.variables, inputs)
        table = None if head.draws_from_rows else task.model.output_table(task.variables)

        def observed(params, hidden):
            ids = head.tied_labels()
            tied = None if ids is None else task.model.label_logits(task.variables, hidden, inputs, ids)
            scores = head.apply({"params": params}, hidden, inputs, table, tied=tied)[:, 0]
            return jnp.sum(scores * probe)

        gradient, hidden = jax.grad(observed, (0, 1))(
            part(task.variables, "head").get("params", {}), states)
        result = {"scores": task.model.logits(task.variables, inputs)[:, 0], "gradient.hidden": hidden}
        if mode == "head":
            result.update({f"gradient.{name}": gradient[native] for name, native in
                           (("slot", "slots"), ("q.weight", "q"), ("k.weight", "k"))})
        return result

    def native(index, held):
        rows = []
        for request in requests[int(index)]:
            questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
            rows.extend(task.layout.rows(task.tokenizer, task.specials, request["state"], questions))
        laid = DecisionInputs.collate(rows, task.specials.pad, options=held["options"].shape[1])
        mask = np.arange(held["tokens"].shape[1])[None] < held["lengths"][:, None]
        for row, encoded in enumerate(rows):
            np.testing.assert_array_equal(encoded.tokens, held["tokens"][row, :held["lengths"][row]])
            np.testing.assert_array_equal(encoded.questions[0].options,
                                          held["spans"][row, held["options"][row]])
            if attention == "block":
                np.testing.assert_array_equal(encoded.blocks, held["blocks"][row, :held["lengths"][row]])
        inputs = replace(laid, tokens=jnp.asarray(held["tokens"]), segments=jnp.asarray(mask, jnp.int32),
                         positions=jnp.broadcast_to(jnp.arange(mask.shape[1]), mask.shape),
                         blocks=jnp.asarray(held["blocks"]) if attention == "block" else None)
        probe = np.zeros(held["options"].shape, np.float32)
        probe[held["options"]] = held["probe"]
        result = forward(inputs, jnp.asarray(probe))
        assert np.isneginf(np.asarray(result["scores"])[~held["options"]]).all()
        return {**result, "scores": np.asarray(result["scores"])[held["options"]]}

    assert_as_exact(native, reference, draw, f"released slot runtime {site}", seeds=seeds, truth_fn=truth)
