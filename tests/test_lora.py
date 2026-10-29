"""LoRA adapters against PEFT 0.20.0 and Diffusers 0.34.0.

The fixtures under tests/fixtures/lora come from tools/lora_reference.py:
a PEFT adapter on the llama-tiny decoder with a rank and an alpha pattern,
and a Diffusers LoRA file on the tiny Stable Diffusion pipeline with a
UNet and a text-encoder component. Every forward here runs in fp32.

Observed against the references, all under the 1e-4 bound:

- llama-tiny unmerged logits 5.7e-06, merged logits 7.3e-06, merged weights
  3.0e-08, loss 1e-06, adapter gradients 1.4e-06, logits after one SGD step
  1.1e-05, stepped factors 7.1e-08, merged export reloaded 8.8e-06.
- sd-tiny text encoder hidden states 3.6e-07; UNet prediction 7.3e-06
  adapted and merged, fused weights 6.0e-08. The bundle is the Flax
  pipeline, whose UNet approximates the GELU and normalizes attention
  inputs with an epsilon of 1e-5; the torch reference computes the GELU
  exactly and uses 1e-6, so the UNet runs as the torch one here
  (`torch_unet`).
"""

import dataclasses
import json
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Replaying
from flax import linen as nn
from interop_support import extract_fixture, hub_cache
from model_support import flat_tree
from reference_error import as_given, assert_as_exact, assert_as_exact_as_the_reference, widened
from safetensors.numpy import load_file
from test_joined import row_joins

from dew import lora
from dew.batch import FROZEN, merge, thaw
from dew.data.dataset import Loader
from dew.diffusion.process import DenoisingCondition
from dew.interop.pretrained import Pretrained
from dew.interop.safetensors_io import read_file, write_file
from dew.lora import Adapter, AdapterBase, LoRA, Target
from dew.nn.sharding import ADAPTER_RANK, declarations, declared_axes, parameter_path
from dew.objectives.base import Step
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import LMObjective
from dew.optim import OPTIMIZER_MAP, muon_weight_dimension_numbers
from dew.training import Checkpoints, Layout, MeshSpec, Trainer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "lora"
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"
ADAPTER = FIXTURES / "llama-tiny" / "adapter"

IMAGES = ROOT / "tests" / "fixtures" / "tfds" / "dew_images" / "1.0.0"


@pytest.fixture(scope="module")
def decoder():
    return Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURES / "llama-tiny" / "reference.npz") as data:
        return {key: data[key] for key in data}


@pytest.fixture(scope="module")
def loaded(decoder):
    return LoRA.load(decoder.model, decoder.variables, ADAPTER, layouts=decoder.layouts)


def _rebound(adapter: Adapter, base: nn.Module, **changes) -> Adapter:
    """`adapter`'s targets and factors bound to `base` again with `changes`
    (rslora, dropout, targets) applied, since the adapted model computes
    the branch its adapter was bound with."""
    fields = {"targets": adapter.targets, "rslora": adapter.rslora, "dropout": adapter.dropout, **changes}
    return Adapter.bound(base, adapter.variables, fields["targets"], fields["rslora"], fields["dropout"],
                         adapter.layouts)


def _is_factor(path) -> bool:
    return path[-1].key in lora.FACTORS


def _factors(adapter, tree, directory) -> dict[str, np.ndarray]:
    """The adapter's leaves in `tree`, in PEFT's layout under PEFT's names."""
    adapter.save(tree, directory)
    return {
        key.removeprefix(lora.PEFT_PREFIX): value
        for key, value in load_file(directory / lora.PEFT_WEIGHTS).items()
    }


def _source_tensor(source, tree, name: str) -> np.ndarray:
    return next(layout for layout in source.weight_layouts if layout.name == name).export(tree)


def test_the_unmerged_forward_matches_peft(loaded, reference):
    """The adapted model reads the split tree, factors under `params` and the
    base under `frozen`, as it comes from the loader."""
    adapter = loaded
    assert sorted(adapter.variables) == [FROZEN, "params"]
    logits = adapter.model.apply(adapter.variables, jnp.asarray(reference["input_ids"]))
    np.testing.assert_allclose(np.asarray(logits), reference["adapted_logits"], atol=1e-4, rtol=0)
    assert np.max(np.abs(reference["adapted_logits"] - reference["base_logits"])) > 1


def test_eval_prediction_disables_adapter_dropout(loaded, reference):
    adapter = loaded
    ids = jnp.asarray(reference["input_ids"])
    objective = LMObjective(adapter.model, ids.shape[1] - 1, ema_decay=None)
    predictions = {}
    for train in (False, True):
        predictions[train] = [
            objective.predict(adapter.variables, {"text": ids},
                              Step(step=jnp.int32(0), key=jax.random.key(seed), ema=None),
                              train=train)[2].logits
            for seed in (0, 1)
        ]
    np.testing.assert_array_equal(*predictions[False])
    np.testing.assert_allclose(predictions[False][0], reference["adapted_logits"][:, :-1],
                               atol=1e-4, rtol=0)
    assert not np.array_equal(*predictions[True])


def test_the_merge_matches_peft_weights_and_logits(decoder, loaded, reference):
    """W + scale * B A into every kernel, the factors gone, the plain model
    over the merged tree agreeing with merge_and_unload."""
    adapter = loaded
    merged = adapter.merge(adapter.variables)
    assert sorted(merged) == ["params"]
    assert not any(_is_factor(path) for path, _ in jax.tree_util.tree_leaves_with_path(merged))
    for key in reference:
        if key.startswith("merged/"):
            np.testing.assert_allclose(_source_tensor(decoder, merged, key.removeprefix("merged/")),
                                       reference[key], atol=1e-6, rtol=0)
    logits = decoder.model.apply(merged, jnp.asarray(reference["input_ids"]))
    np.testing.assert_allclose(np.asarray(logits), reference["merged_logits"], atol=1e-4, rtol=0)


def test_adapter_gradients_of_the_token_loss_match_peft(decoder, loaded, reference, tmp_path):
    """The mean next-token cross entropy through the LM objective over the
    frozen split, differentiated with respect to the adapter alone."""
    # The reference stepped in eval mode; the objective's forward is a
    # training one, so the branch's dropout is turned off to compare.
    adapter = _rebound(loaded, decoder.model, dropout=0.0)
    tokens = jnp.asarray(reference["input_ids"])
    objective = LMObjective(adapter.model, tokens.shape[1] - 1, variables=adapter.variables, ema_decay=None)
    params = objective.init(jax.random.key(0))
    assert sorted(params) == [FROZEN, "params"]
    assert all(_is_factor(path) for path, _ in jax.tree_util.tree_leaves_with_path(params["params"]))

    def loss(moving):
        stats, _ = objective.loss({**params, "params": moving}, {"text": tokens},
                                  Step(step=jnp.int32(0), key=jax.random.key(1), ema=None))
        return objective.reduce_loss(stats)[0]

    value, gradient = jax.jit(jax.value_and_grad(loss))(params["params"])
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    exported = _factors(adapter, merge(adapter.variables, {"params": gradient}), tmp_path)
    for key in reference:
        if key.startswith("grad/"):
            np.testing.assert_allclose(exported[key.removeprefix("grad/")], reference[key], atol=1e-4, rtol=0)


def test_an_adapted_head_trains_on_its_exact_logits(decoder, reference):
    """No matrix alone is a head with factors on it, so the LM objective
    scores the model's exact logits whole: the loss is the cross entropy of
    the adapted forward's logits, and SGD through the Trainer moves every
    factor, lowers the loss and leaves the frozen base bitwise."""
    adapter = LoRA(rank=2, modules=("q_proj", "lm_head")).apply(decoder.model, decoder.variables, key=0,
                                                                  layouts=decoder.layouts)
    tokens = np.asarray(reference["input_ids"])
    rows = 2 * jax.device_count()
    objective = LMObjective(adapter.model, tokens.shape[1] - 1, variables=adapter.variables, ema_decay=None)
    batch, step = {"text": jnp.asarray(tokens)}, Step(jnp.int32(0), jax.random.key(1), None)
    trainer = Trainer(objective, optax.sgd(0.5), key=jax.random.key(3), mesh=MeshSpec(),
                      layout=Layout(min_shard=2**30))
    initial = trainer.initial_state()
    logits = adapter.model.apply(initial.variables, batch["text"][:, :-1])
    first = objective.scalar_loss(initial.variables, batch, step)[0]
    np.testing.assert_allclose(first, optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["text"][:, 1:]).mean(), rtol=1e-5)

    data = Loader(train=lambda partition: iter([{"text": tokens[np.arange(rows) % 2]}] * 3), val=None,
                   records=rows, batch=rows)
    state = trainer.fit(data, steps=3, log_every=3)
    assert objective.scalar_loss(state.variables, batch, step)[0] < first
    leaves = {name: [jax.tree.leaves(tree.variables[name]) for tree in (initial, state)]
              for name in (FROZEN, "params")}
    for before, after in zip(*leaves[FROZEN], strict=True):
        np.testing.assert_array_equal(before, after)
    assert all(np.any(before != after) for before, after in zip(*leaves["params"], strict=True))


def test_one_trainer_step_moves_the_adapter_and_nothing_else(decoder, loaded, reference, tmp_path):
    """A real SGD step through the Trainer: the frozen collection comes back
    bitwise, every factor moves, and the logits and factors agree with the
    reference's step on the adapter parameters. The run records the
    adapter's per-module ranks and alphas, so the adapter it rebuilds
    computes the stepped logits and writes the same PEFT files."""
    adapter = _rebound(loaded, decoder.model, dropout=0.0)
    meta = json.loads((FIXTURES / "llama-tiny" / "meta.json").read_text())
    tokens = np.asarray(reference["input_ids"])
    rows = 2 * jax.device_count()
    objective = LMObjective(adapter.model, tokens.shape[1] - 1, variables=adapter.variables, ema_decay=None)
    data = Loader(
        train=lambda partition: iter([{"text": tokens[np.arange(rows) % 2]}]),
        val=None,
        records=rows,
        batch=rows,
    )
    checkpoints = Checkpoints(str(tmp_path / "run"))
    trainer = Trainer(objective, optax.sgd(meta["learning_rate"]), key=jax.random.key(3),
                      mesh=MeshSpec(), layout=Layout(min_shard=2**30), checkpoints=checkpoints)
    initial = trainer.initial_state()

    state = trainer.fit(data, steps=1, log_every=1, checkpoint_every=1)
    checkpoints.wait()

    for before, after in zip(
        jax.tree.leaves(initial.variables[FROZEN]), jax.tree.leaves(state.variables[FROZEN]), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    assert all(
        bool(jnp.any(before != after))
        for before, after in zip(
            jax.tree.leaves(initial.variables["params"]),
            jax.tree.leaves(state.variables["params"]), strict=True
        )
    )
    logits = adapter.model.apply(state.variables, jnp.asarray(tokens))
    np.testing.assert_allclose(np.asarray(logits), reference["updated_logits"], atol=1e-4, rtol=0)
    exported = _factors(adapter, state.variables, tmp_path / "adapter")
    for key in reference:
        if key.startswith("updated/"):
            np.testing.assert_allclose(
                exported[key.removeprefix("updated/")], reference[key], atol=1e-4, rtol=0
            )
    rebuilt = Adapter.from_run(tmp_path / "run", ema=False)
    assert rebuilt.targets == adapter.targets
    np.testing.assert_array_equal(np.asarray(rebuilt.model.apply(rebuilt.variables, jnp.asarray(tokens))),
                                  np.asarray(logits))
    rebuilt.save(rebuilt.variables, tmp_path / "rebuilt")
    _same_files(tmp_path / "rebuilt", tmp_path / "adapter")
    # The merged full export reloads as a plain source at the stepped logits.
    decoder.save(tmp_path / "merged", variables=adapter.merge(state.variables))
    reloaded = Pretrained.load(tmp_path / "merged", dtype="float32", attention_impl="reference")
    np.testing.assert_allclose(np.asarray(reloaded.model.apply(reloaded.variables, jnp.asarray(tokens))),
                               reference["updated_logits"], atol=1e-4, rtol=0)


def test_a_source_fine_tunes_through_its_own_adapted_bundle(decoder, reference, tmp_path):
    """`source.adapt(LoRA(...))` is the source with the adapted model, the
    factors and the adapter beside them. An objective over it trains the
    factors alone; the adapter writes PEFT's files and `save` the merged
    weights straight from the trainer's tree; and an objective over a
    bundle decodes text through the bundle's processor."""
    from dew.data import ByteTokenizer
    from dew.inference import RunProcessor
    from dew.sampling import Sampling

    source = decoder
    tuned = source.adapt(LoRA(rank=2, modules=("q_proj", "v_proj")), key=0)
    assert source.adapter is None and tuned.adapter is not None
    tokens = np.asarray(reference["input_ids"])
    rows = 2 * jax.device_count()
    batch = {"text": tokens[np.arange(rows) % 2]}
    objective = LMObjective(tuned, tokens.shape[1] - 1)
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(3), mesh=MeshSpec(),
                      layout=Layout(min_shard=2**30))
    initial = trainer.initial_state()
    state = trainer.fit(
        Loader(train=lambda partition: iter([batch, batch]), val=None, records=rows, batch=rows),
        steps=2,
        log_every=2,
    )

    assert set(flat_tree(state.variables["params"])) == {
        f"{'.'.join(target[1:])}.{factor}" for target in tuned.adapter.targets for factor in lora.FACTORS}
    for before, after in zip(
        jax.tree.leaves(initial.variables[FROZEN]), jax.tree.leaves(state.variables[FROZEN]), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    trained = thaw(state.variables)
    tuned.adapter.save(state.variables, tmp_path / "adapter")
    read = LoRA.load(source.model, source.variables, tmp_path / "adapter", layouts=source.layouts)
    for name, leaf in flat_tree(thaw(read.variables)).items():
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(flat_tree(trained)[name]), err_msg=name)
    # The export is the merged weights in the source's layout, so the reload
    # computes exactly what the source model computes on them; how close the
    # merge is to the adapted forward is the PEFT parity test's to bound.
    tuned.save(tmp_path / "merged", variables=state.variables)
    reloaded = Pretrained.load(tmp_path / "merged", dtype="float32", attention_impl="reference")
    ids = jnp.asarray(tokens)
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)),
                                  np.asarray(source.model.apply(tuned.adapter.merge(state.variables), ids)))

    # A source that ships a tokenizer hands its processor to the objective.
    processor = RunProcessor(ByteTokenizer())
    tuned = dataclasses.replace(tuned, processor=processor)
    task = LMObjective(tuned, tokens.shape[1] - 1).pipeline(state)
    assert task.processor is processor
    greedy = Sampling(temperature=0)
    np.testing.assert_array_equal(task("hello", 3, key=1, sampling=greedy).host().tokens,
                                  task([list(b"hello")], 3, key=1, sampling=greedy).host().tokens)
    plain = dataclasses.replace(source, processor=processor)
    assert LMObjective(plain, tokens.shape[1] - 1).pipeline(initial).processor is processor
    other = RunProcessor(ByteTokenizer())
    assert objective.pipeline(state, processor=other).processor is other


def test_export_writes_the_peft_file_back(decoder, loaded, tmp_path):
    """The factors land bitwise where PEFT wrote them, under its names, and
    the config resolves every module to the same rank and alpha on reload."""
    adapter = loaded
    adapter.save(adapter.variables, tmp_path)
    ours = load_file(tmp_path / lora.PEFT_WEIGHTS)
    theirs = load_file(ADAPTER / lora.PEFT_WEIGHTS)
    assert ours.keys() == theirs.keys()
    for key in theirs:
        np.testing.assert_array_equal(ours[key], theirs[key])
    config = json.loads((tmp_path / lora.PEFT_CONFIG).read_text())
    assert (config["r"], config["lora_alpha"], config["use_rslora"], config["lora_dropout"]) == (
        4,
        8,
        False,
        0.1,
    )
    assert config["rank_pattern"] == {"model.layers.1.self_attn.v_proj": 2}
    assert config["alpha_pattern"] == {"model.layers.0.mlp.down_proj": 3, "model.layers.1.mlp.down_proj": 3}
    again = LoRA.load(decoder.model, decoder.variables, tmp_path, layouts=decoder.layouts)
    assert (again.targets, again.rslora, again.dropout) == (adapter.targets, adapter.rslora, adapter.dropout)
    for ours_leaf, theirs_leaf in zip(
        jax.tree.leaves(again.variables), jax.tree.leaves(adapter.variables), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(ours_leaf), np.asarray(theirs_leaf))


_HUB_ADAPTER = """
import json, sys
import jax, numpy as np
from dew.interop.pretrained import Pretrained
from dew.lora import LoRA

for name, directory in (("dew/llama-tiny", sys.argv[1]), (sys.argv[3], sys.argv[2])):
    source = Pretrained.load(name, dtype="float32", attention_impl="reference")
    adapted = source.adapt(LoRA(rank=3, modules=("q_proj", "v_proj")), key=0)
    variables = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jax.random.normal(jax.random.key(7), leaf.shape, leaf.dtype)
        if path[-1].key == "lora_B" else leaf, adapted.variables)
    adapted.adapter.save(variables, directory)
    tokens = np.load(sys.argv[3] + "/input_ids.npy")
    np.save(directory + "/logits.npy", np.asarray(adapted.model.apply(variables, tokens)))
    base = LoRA.load(source.model, source.variables, directory, layouts=source.layouts).bases.get("")
    print(json.dumps(None if base is None else [base.name, base.revision]))
    # The run's record of the adapter keeps its base through JSON.
    from dew.lora import AdapterRecord
    from dew.registry import from_record, to_record, wrapper_layers

    record = wrapper_layers(type(adapted.model))[1][0]._dew_wrapper()
    kept = from_record(AdapterRecord, json.loads(json.dumps(to_record(record, AdapterRecord))))
    print(json.dumps(kept.bases == record.bases))
"""


def test_an_adapter_names_the_hub_model_it_was_bound_to_and_peft_reloads_it_there(tmp_path):
    """An adapter bound to a model loaded from the Hub names it and the commit the Hub resolved it
    to in PEFT's config, and PEFT, reading only that config and the cache, loads the same base and
    computes the logits Dew computes; Dew's own reload keeps the name. One bound to a local
    directory names none, since no other machine resolves it. The process runs offline over a Hub
    cache holding the tiny Llama at one commit."""
    commit = hub_cache(tmp_path / "hub", "dew/llama-tiny", LLAMA)
    hub, local = tmp_path / "hub-adapter", tmp_path / "local-adapter"

    def run(script, *arguments):
        return offline(tmp_path / "hub", script, *arguments)

    reloaded = [json.loads(line) for line in run(_HUB_ADAPTER, hub, local, LLAMA) if line]
    assert reloaded == [["dew/llama-tiny", commit], True, None, True]
    config = json.loads((hub / "adapter_config.json").read_text())
    assert (config["base_model_name_or_path"], config["revision"]) == ("dew/llama-tiny", commit)
    written = json.loads((local / "adapter_config.json").read_text())
    assert not {"base_model_name_or_path", "revision"} & set(written)
    run(_PEFT_READS, hub, LLAMA / "input_ids.npy")
    np.testing.assert_allclose(np.load(hub / "peft.npy"), np.load(hub / "logits.npy"), atol=2e-5)


def offline(cache: Path, script: str, *arguments) -> list[str]:
    """`script`'s output lines, run in a fresh process that resolves the Hub from `cache` alone."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "HF_HUB_CACHE": str(cache), "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run([sys.executable, "-c", script, *map(str, arguments)], env=env, check=False,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout.split("\n")


_HUB_PIPELINE_ADAPTER = """
import json, sys
from dew.interop.pretrained import load_diffusion_source
from dew.interop.safetensors_io import read_file
from dew.lora import DIFFUSERS_METADATA, DIFFUSERS_WEIGHTS, LoRA

source = load_diffusion_source("dew/sd-tiny", dtype="float32", attention_impl="reference")
adapted = source.adapt(LoRA(rank=2, modules=("to_q", "to_v")), key=0)
adapted.adapter.save(adapted.variables, sys.argv[1])
_, metadata = read_file(sys.argv[1] + "/" + DIFFUSERS_WEIGHTS)
header = json.loads(metadata[DIFFUSERS_METADATA])
print(json.dumps({key: value for key, value in header.items()
                  if key.endswith((".base_model_name_or_path", ".revision"))}))
"""


def test_a_pipeline_a_run_reads_off_the_hub_names_it_as_its_adapters_base(tmp_path):
    """The loader a diffusion run reads its pipeline with (`load_diffusion_source`),
    offline over a Hub cache holding the tiny SD pipeline at one commit, binds an
    adapter whose Diffusers file names that pipeline and commit as the denoiser's
    base, as an adapter of `Pretrained.load`'s source does."""
    source = extract_fixture(ROOT / "tests/fixtures/tiny_diffusers.tar.xz", tmp_path / "fixtures") / "sd"
    commit = hub_cache(tmp_path / "hub", "dew/sd-tiny", source)
    [named] = [json.loads(line) for line in offline(tmp_path / "hub", _HUB_PIPELINE_ADAPTER,
                                                    tmp_path / "adapter") if line]
    assert named == {"unet.base_model_name_or_path": "dew/sd-tiny", "unet.revision": commit}


_PEFT_READS = """
import json, sys
import numpy as np, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

directory = sys.argv[1]
config = json.load(open(directory + "/adapter_config.json"))
base = AutoModelForCausalLM.from_pretrained(config["base_model_name_or_path"], revision=config["revision"])
with torch.no_grad():
    logits = PeftModel.from_pretrained(base, directory).eval()(torch.as_tensor(np.load(sys.argv[2]))).logits
np.save(directory + "/peft.npy", logits.numpy())
"""


@pytest.mark.mesh(devices=2)
def test_an_adapters_factors_take_their_kernels_axes_on_each_side_of_the_rank(loaded):
    """Each factor of an adapted projection takes its kernel's axes on its own side of the rank,
    `lora_A` `[in, r]` the input's and `lora_B` `[r, out]` the output's, and the rank its own axis,
    which no rule places: over an fsdp mesh `lora_A` splits its input as the kernel does and the
    rank stays whole, and Muon orthogonalizes each as a matrix from one side to the other. The
    factors took the module's two kernel axes as they stood, so `lora_B`'s rank was named the
    kernel's input and split over fsdp."""
    variables = loaded.variables
    declared = declarations(variables)
    factors = {"/".join(parameter_path(path)[-2:]): (path, leaf)
               for path, leaf in jax.tree_util.tree_leaves_with_path(variables)
               if parameter_path(path)[-1] in lora.FACTORS and "layers_0" in parameter_path(path)}
    assert {name: declared_axes(path, leaf.ndim, declared) for name, (path, leaf) in factors.items()} == {
        "q_proj/lora_A": ("embed", ADAPTER_RANK), "q_proj/lora_B": (ADAPTER_RANK, "heads"),
        "v_proj/lora_A": ("embed", ADAPTER_RANK), "v_proj/lora_B": (ADAPTER_RANK, "kv"),
        "down_proj/lora_A": ("mlp", ADAPTER_RANK), "down_proj/lora_B": (ADAPTER_RANK, "embed")}
    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    placed = Layout(min_shard=1).shardings(mesh, variables)
    specs = {"/".join(parameter_path(path)[-2:]): sharding.spec
             for path, sharding in jax.tree_util.tree_leaves_with_path(placed)
             if "/".join(parameter_path(path)[-2:]) in factors and "layers_0" in parameter_path(path)}
    assert specs["q_proj/lora_A"][0] == specs["v_proj/lora_A"][0] == "fsdp", specs
    assert all(not spec or spec[0] is None for name, spec in specs.items() if name.endswith("lora_B")), specs
    muon = muon_weight_dimension_numbers(variables["params"])
    for name, (path, _) in factors.items():
        numbers = muon
        for key in path[1:]:
            numbers = numbers[key.key]
        assert (numbers.reduction_axis, numbers.output_axis) == ((0,), (1,)), name


def test_muon_steps_an_adapted_head_whose_factors_it_splits_between_its_optimizers(decoder, reference):
    """An adapter on the vocabulary head as well as q_proj: Muon takes the head's `lora_A`
    ('embed', rank) as a matrix and AdamW its `lora_B` (rank, 'vocab'), a lookup, and optax hands
    each optimizer its own masked tree, so a factor's axes are read off it alone. They were read
    off both factors, so the masked tree raised on Muon's first update."""
    adapter = LoRA(rank=2, modules=("q_proj", "lm_head")).apply(
        decoder.model, decoder.variables, key=0, layouts=decoder.layouts)
    # A fresh adapter's B is zero, which leaves A's gradient zero; a drawn B moves both.
    factors = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jax.random.normal(jax.random.key(7), leaf.shape, leaf.dtype)
        if path[-1].key == "lora_B" else leaf, adapter.variables["params"])
    numbers = muon_weight_dimension_numbers(factors)
    assert numbers["lm_head"]["lora_A"] is not None and numbers["lm_head"]["lora_B"] is None
    optimizer = OPTIMIZER_MAP["muon"](1e-3)
    tokens = jnp.asarray(reference["input_ids"])

    def loss(moving):
        return jnp.mean(jnp.square(adapter.model.apply({**adapter.variables, "params": moving}, tokens)))

    updates, _ = optimizer.update(jax.grad(loss)(factors), optimizer.init(factors), factors)
    moved = optax.apply_updates(factors, updates)
    for path, before in jax.tree_util.tree_leaves_with_path(factors):
        after = moved
        for key in path:
            after = after[key.key]
        assert not np.array_equal(np.asarray(after), np.asarray(before)), jax.tree_util.keystr(path)


def test_a_fresh_adapter_is_the_identity_and_matches_by_suffix(decoder, reference):
    """PEFT's target_modules: a suffix names every projection under it; B
    starts at zero so the adapted forward is the base forward; A is drawn
    on +-1/sqrt(fan_in). An unset alpha is PEFT's own default, twice the rank."""
    adapter = LoRA(rank=3, modules=("q_proj", "layers.1.mlp.up_proj")).apply(
        decoder.model, decoder.variables, key=0, layouts=decoder.layouts)
    variables = adapter.variables
    assert set(adapter.targets) == {("params", "layers_0", "self_attn", "q_proj"),
                                    ("params", "layers_1", "self_attn", "q_proj"),
                                    ("params", "layers_1", "mlp", "up_proj")}
    assert {target.alpha for target in adapter.targets.values()} == {6.0}
    tokens = jnp.asarray(reference["input_ids"])
    np.testing.assert_array_equal(np.asarray(adapter.model.apply(variables, tokens)),
                                  np.asarray(decoder.model.apply(decoder.variables, tokens)))
    a = variables["params"]["layers_1"]["mlp"]["up_proj"]["lora_A"]
    assert a.shape == (64, 3) and 0 < float(jnp.abs(a).max()) <= 1 / 8
    assert not bool(jnp.any(variables["params"]["layers_1"]["mlp"]["up_proj"]["lora_B"]))
    with pytest.raises(ValueError):
        LoRA(rank=2, modules=("q_proj", "w_proj"), alpha=2.0).apply(
            decoder.model, decoder.variables, key=0, layouts=decoder.layouts)


@pytest.mark.parametrize("scan", [False, True])
def test_an_adapter_on_gate_and_up_keeps_its_branch_where_the_row_joins_them(decoder, reference, scan):
    """An adapted model's gate and up, with their projections apart or a join
    of them into one GEMM chosen (`dew.nn.joined`), keep the adapter's branch,
    in a scanned stack too: the branch moves the logits off the base model's
    either way."""
    adapter = LoRA(rank=3, modules=("gate_proj", "up_proj")).apply(
        decoder.model, decoder.variables, key=0, layouts=decoder.layouts)
    model = adapter.model.clone(scan_layers=scan)
    variables = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jax.random.normal(jax.random.key(7), leaf.shape, leaf.dtype)
        if path[-1].key == "lora_B" else leaf, adapter.variables)
    tokens = jnp.asarray(reference["input_ids"])
    base = np.asarray(decoder.model.apply(decoder.variables, tokens))
    for joins in (False, True):
        with row_joins("gate_up", joins=joins):
            adapted = np.asarray(model.apply(variables, tokens))
        assert np.max(np.abs(adapted - base)) > 1e-2, joins


class _BranchHost(nn.Module):
    """A target Dense carried as a submodule, the way models carry one."""

    proj: nn.Module
    keyword: bool = False

    def __call__(self, x):
        return self.proj(inputs=x) if self.keyword else self.proj(x)


def _adapted(model, tree, *, contracted=1, rank=2, alpha=4.0, seed=0):
    """A target on `proj`'s kernel with nonzero factors spliced in, and the
    merged base beside it. A fresh adapter's zero B proves nothing."""
    keys = iter(jax.random.split(jax.random.key(seed), 2))
    shape = tree["params"]["proj"]["kernel"].shape
    factors = {"lora_A": jnp.asarray(jax.random.normal(next(keys), (*shape[:contracted], rank))),
               "lora_B": jnp.asarray(jax.random.normal(next(keys), (rank, *shape[contracted:])))}
    adapted_tree = merge(tree, {"params": {"proj": factors}})
    adapter = Adapter.bound(model, adapted_tree, {("params", "proj"): Target(rank, alpha)}, rslora=False,
                            dropout=0.0, layouts={})
    return adapter, adapter.model, adapted_tree, adapter.merge(adapted_tree)


CLIPPED_SEEDS = 64
"""The clipped adapter site's reference-only sizing, including its input and factor gradients."""


def clipped_adapter_case():
    """A bounded Gemma linear with nonzero LoRA factors, against the clipped linear equation."""
    import torch
    import torch.nn.functional as functional

    from dew.nn.vision.gemma4 import Gemma4ClippableLinear

    model = _BranchHost(Gemma4ClippableLinear(12, use_clipped_linears=True))
    variables = model.init(jax.random.key(7), jnp.zeros((8, 16), jnp.float32))
    limits = variables["constants"]["proj"]
    limits.update(input_min=jnp.float32(-0.7), input_max=jnp.float32(0.7),
                  output_min=jnp.float32(-1), output_max=jnp.float32(1))
    rank, alpha = 2, 4.0
    adapter = LoRA(rank=rank, modules=("proj",), alpha=alpha).apply(model, variables, key=9)
    kernel, bias = (np.asarray(variables["params"]["proj"][name]) for name in ("kernel", "bias"))

    def draw(seed):
        rng = np.random.default_rng(seed)
        factors = {"proj": {"lora_A": rng.standard_normal((16, rank), np.float32) * np.float32(0.25),
                            "lora_B": rng.standard_normal((rank, 12), np.float32) * np.float32(0.1)}}
        return factors, rng.standard_normal((8, 16), np.float32), rng.standard_normal((8, 12), np.float32)

    @jax.jit
    def forward(factors, inputs, cotangent):
        output, pullback = jax.vjp(
            lambda params, values: adapter.model.apply({**adapter.variables, "params": params}, values),
            factors, inputs)
        parameters, values = pullback(cotangent)
        return {"output": output, "parameters": parameters, "inputs": values}

    def reference(dtype, factors, inputs, cotangent):
        x = torch.tensor(np.asarray(inputs), dtype=dtype, requires_grad=True)
        a, b = (torch.tensor(np.asarray(factors["proj"][name]), dtype=dtype, requires_grad=True)
                for name in lora.FACTORS)
        low, high = (torch.tensor(np.asarray(limits[name]), dtype=dtype)
                     for name in ("input_min", "input_max"))
        clipped = x.clamp(low, high)
        base = functional.linear(clipped, torch.tensor(kernel.T, dtype=dtype),
                                 torch.tensor(bias, dtype=dtype))
        delta = functional.linear(functional.linear(clipped, a.T), b.T)
        low, high = (torch.tensor(np.asarray(limits[name]), dtype=dtype)
                     for name in ("output_min", "output_max"))
        output = (base + (alpha / rank) * delta).clamp(low, high)
        dx, da, db = torch.autograd.grad(output, (x, a, b), torch.tensor(np.asarray(cotangent), dtype=dtype))
        return {"output": output.detach().numpy(), "inputs": dx.numpy(),
                "parameters": {"proj": {"lora_A": da.numpy(), "lora_B": db.numpy()}}}

    return model, variables, adapter, draw, forward, reference


def test_a_clipped_linear_adapts_once_inside_its_bounds_and_reloads(tmp_path):
    import torch

    model, variables, adapter, draw, forward, reference = clipped_adapter_case()
    assert_as_exact(forward, as_given(lambda *args: reference(torch.float32, *args)), draw,
                    "clipped linear adapter", seeds=CLIPPED_SEEDS,
                    truth_fn=lambda *args: reference(torch.float64, *args))
    factors, inputs, _ = draw(3)
    def loss(params):
        return jnp.mean(jnp.square(adapter.model.apply({**adapter.variables, "params": params}, inputs)))

    optimizer = optax.sgd(0.01)
    updates, _ = optimizer.update(jax.grad(loss)(factors), optimizer.init(factors), factors)
    moving = optax.apply_updates(factors, updates)
    assert any(not np.array_equal(before, after)
               for before, after in zip(jax.tree.leaves(factors), jax.tree.leaves(moving), strict=True))
    trained = {**adapter.variables, "params": moving}
    adapter.save(trained, tmp_path)
    restored = LoRA.load(model, variables, tmp_path)
    np.testing.assert_array_equal(restored.model.apply(restored.variables, inputs),
                                  adapter.model.apply(trained, inputs))
    merged = model.apply(adapter.merge(trained), inputs)
    assert np.all(np.isfinite(merged)) and np.all(np.abs(merged) <= 1)


def test_field_only_linear_subclasses_train_their_adapter():
    from dew.nn.dense import Dense

    class FieldDense(Dense):
        use_bias: bool = False

    class FieldFlaxDense(nn.Dense):
        use_bias: bool = False

    inputs = jnp.asarray(np.random.default_rng(9).standard_normal((4, 6)), jnp.float32)
    for projection in (FieldDense(5), FieldFlaxDense(5)):
        model = _BranchHost(projection)
        variables = model.init(jax.random.key(3), inputs)
        adapter = LoRA(rank=2, modules=("proj",)).apply(model, variables, key=4)
        gradient = jax.grad(lambda params, bound=adapter: jnp.sum(jnp.square(bound.model.apply(
            {**bound.variables, "params": params}, inputs))))(adapter.variables["params"])
        assert np.any(np.asarray(gradient["proj"]["lora_B"]))


@pytest.mark.parametrize("arithmetic", [False, True])
def test_an_opaque_linear_override_is_refused_instead_of_training_unused_factors(arithmetic):
    from dew.nn.dense import Dense

    class OpaqueCall(Dense):
        @nn.compact
        def __call__(self, inputs):
            return jnp.tanh(super().__call__(inputs))

    class OpaqueArithmetic(Dense):
        def _linear(self, inputs):
            return jnp.tanh(super()._linear(inputs))

    model = _BranchHost(OpaqueArithmetic(5) if arithmetic else OpaqueCall(5))
    inputs = jnp.ones((4, 6), jnp.float32)
    variables = model.init(jax.random.key(3), inputs)
    adapter = LoRA(rank=2, modules=("proj",)).apply(model, variables, key=4)
    with pytest.raises(TypeError):
        adapter.model.apply(adapter.variables, inputs)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("boundary", ["input", "output"])
def test_quantized_clipped_adapters_keep_both_clamp_boundaries(frozen, boundary):
    """Clamped inputs have zero derivatives while interior inputs still train. A saturated
    output is exactly its stored bound, including when a packed base replaces the linear call."""
    from dew.nn.kernels.generation import device_generation
    from dew.quantization import QUANTIZATION, InputQuantization

    model, variables, _, _, _, _ = clipped_adapter_case()
    kernel = variables["params"]["proj"]["kernel"]
    variables["params"]["proj"]["kernel"] = jnp.full_like(kernel, 1 / 256 if boundary == "input" else 1)
    checkpoint = InputQuantization(sites=(("proj", "nvfp4"),))
    quantized = checkpoint.apply(model)
    variables = {**variables, QUANTIZATION: jax.tree.map(
        lambda scale: jnp.full_like(scale, 0.1), checkpoint.declared())}
    packed = (InputQuantization(frozen_fp8=("proj",), measured_on=device_generation()) if frozen else None)
    adapter = LoRA(rank=2, modules=("proj",)).apply(quantized, variables, key=4, fp8_base=packed)
    inputs = (jnp.stack([jnp.full((16,), 2.0), jnp.full((16,), 0.1)]) if boundary == "input"
              else jnp.full((2, 16), 0.5)).astype(jnp.float32)
    output = adapter.model.apply(adapter.variables, inputs)
    gradient = jax.grad(lambda values: jnp.sum(adapter.model.apply(adapter.variables, values)))(inputs)
    if boundary == "input":
        assert np.all(np.asarray(output) > 0)
        np.testing.assert_array_equal(gradient[0], jnp.zeros_like(gradient[0]))
        assert np.all(np.asarray(gradient[1]) != 0)
    else:
        upper = variables["constants"]["proj"]["output_max"]
        np.testing.assert_array_equal(output, jnp.broadcast_to(upper, output.shape))
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))


def test_the_branch_reads_a_keyword_input():
    """Flax calls Dense's inputs by keyword as well as positionally; the
    branch has to read `inputs` from kwargs where the call passes it there."""
    model = _BranchHost(nn.Dense(4), keyword=True)
    x = jnp.asarray(np.random.RandomState(0).randn(2, 3), jnp.float32)
    adapter, adapted, adapted_tree, merged = _adapted(model, model.init(jax.random.key(0), x))
    node = adapted_tree["params"]["proj"]
    expected = (jnp.einsum("bi,ij->bj", x, node["kernel"]) + node["bias"]
                + adapter.scale(adapter.targets[("params", "proj")])
                * jnp.einsum("bi,ij,jk->bk", x, node["lora_A"], node["lora_B"]))

    np.testing.assert_allclose(
        np.asarray(adapted.apply(adapted_tree, x)), np.asarray(expected), atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        np.asarray(model.apply(merged, x)), np.asarray(expected), atol=1e-5, rtol=0)


@pytest.mark.parametrize("features", [5, 3], ids=["mismatched-widths", "square"])
def test_a_multi_axis_target_contracts_the_sorted_axes(features):
    """DenseGeneral sorts the contracted axes before pairing them with the
    kernel's leading dims; the branch has to do the same. Widths that differ
    crash the unsorted pairing outright, and equal widths answer a different
    contraction, so the oracle is the einsum the kernel means."""
    model = _BranchHost(nn.DenseGeneral(4, axis=(-1, -2)))
    x = jnp.asarray(np.random.RandomState(1).randn(2, 3, features), jnp.float32)
    adapter, adapted, adapted_tree, merged = _adapted(
        model, model.init(jax.random.key(0), x), contracted=2)
    node = adapted_tree["params"]["proj"]
    created = adapted.init(jax.random.key(9), x)["params"]["proj"]
    assert created["lora_A"].shape == (3, features, 2)
    assert created["lora_B"].shape == (2, 4)
    expected = jnp.einsum("bij,ijk->bk", x, node["kernel"]) + node["bias"] + (
        adapter.scale(adapter.targets[("params", "proj")])
        * jnp.einsum("bij,ijr,rk->bk", x, node["lora_A"], node["lora_B"]))

    np.testing.assert_allclose(
        np.asarray(adapted.apply(adapted_tree, x)), np.asarray(expected), atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        np.asarray(model.apply(merged, x)), np.asarray(expected), atol=1e-5, rtol=0)


# --------------------------------------------------------------------------
# The Diffusers file on the tiny Stable Diffusion pipeline
# --------------------------------------------------------------------------

SD = FIXTURES / "sd-tiny"
TEXT_ROOT = ("encoders", "conditioning", "text_encoder", "text_model")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    destination = tmp_path_factory.mktemp("sd")
    with tarfile.open(ROOT / "tests/fixtures/tiny_diffusers.tar.xz") as archive:
        archive.extractall(destination, members=[m for m in archive.getmembers() if m.name.startswith("sd/")],
                           filter="data")
    source = Pretrained.load(destination / "sd", dtype="float32", attention_impl="reference")
    return dataclasses.replace(source, model=torch_unet(source.model))


def torch_unet(model):
    """The Flax-declared tiny SD UNet as Diffusers' PyTorch UNet computes it:
    the exact GELU, and 1e-6 for the epsilon of the attention blocks'
    GroupNorm, where the Flax class uses 1e-5 (attention_flax.py:368,
    transformer_2d.py:176, Diffusers 0.34.0)."""
    return dataclasses.replace(model, approximate_gelu=False, attention_norm_epsilon=1e-6)


@pytest.fixture(scope="module")
def sd_reference():
    with np.load(SD / "reference.npz") as data:
        return {key: data[key] for key in data}


def _subtree(tree, path):
    for part in path:
        tree = tree[part]
    return tree


def test_the_text_encoder_component_merges_into_the_conditioning_tower(pipeline, sd_reference):
    """The file's text-encoder factors land under the conditioning tower's
    own paths at its rank and alpha: merged in, the tower computes the
    context Diffusers' adapted encoder does."""
    adapter = LoRA.load(pipeline.model, pipeline.variables, SD, layouts=pipeline.layouts)
    tower = pipeline.inputs.conditions["conditioning"].encoder.towers[0]
    ids = jnp.asarray(sd_reference["prompt_ids"])
    merged = adapter.merge(adapter.variables)
    features = tower.apply({"params": _subtree(merged, TEXT_ROOT)}, ids)
    np.testing.assert_allclose(np.asarray(features.last_hidden_state), sd_reference["context"], atol=1e-4,
                               rtol=0)
    plain = tower.apply({"params": _subtree(pipeline.variables, TEXT_ROOT)}, ids)
    assert np.max(np.abs(np.asarray(plain.last_hidden_state) - sd_reference["context"])) > 1e-3


def test_the_unet_component_matches_the_pipeline_adapted_and_fused(pipeline, sd_reference):
    """to_q's [in, heads, depth] and to_out.0's [heads, depth, features]
    DenseGeneral kernels take their factors in kernel layout and merge back
    to the fused torch weights."""
    adapter = LoRA.load(pipeline.model, pipeline.variables, SD, layouts=pipeline.layouts)
    variables = adapter.variables
    condition = DenoisingCondition(jnp.asarray(sd_reference["context"]), None, None)
    latent, time = jnp.asarray(sd_reference["latent"]), jnp.asarray(sd_reference["time"])
    predicted = adapter.model.apply(
        {"params": variables["params"], FROZEN: variables[FROZEN]}, latent, time,
        conditioning=condition)
    np.testing.assert_allclose(np.asarray(predicted), sd_reference["adapted"], atol=1e-4, rtol=0)
    assert np.max(np.abs(sd_reference["adapted"] - sd_reference["base"])) > 0.1
    merged = adapter.merge(variables)
    fused = pipeline.model.apply({"params": merged["params"]}, latent, time, conditioning=condition)
    np.testing.assert_allclose(np.asarray(fused), sd_reference["adapted"], atol=1e-4, rtol=0)
    layouts = {layout.name.replace("/", "."): layout for layout in pipeline.weight_layouts}
    for key in sd_reference:
        if key.startswith("merged/"):
            np.testing.assert_allclose(layouts[key.removeprefix("merged/")].export(merged), sd_reference[key],
                                       atol=1e-6, rtol=0)


def test_a_pipeline_files_components_keep_the_bases_each_names(pipeline, tmp_path):
    """Diffusers writes each component's PEFT config apart, and the tool that bound
    each names the model it bound it to, so a file's components may name different
    bases. The file loads, and each component's base goes back out as it came in:
    saved, reloaded, and through a run's record of the adapter."""
    from dew.lora import AdapterRecord
    from dew.registry import from_record, to_record, wrapper_layers

    tensors, metadata = read_file(SD / lora.DIFFUSERS_WEIGHTS)
    header = json.loads(metadata[lora.DIFFUSERS_METADATA])
    bases = {"unet": AdapterBase("org/sd", "a" * 40), "text_encoder": AdapterBase("org/clip", None)}
    named = {**header, "unet.revision": "a" * 40,
             **{f"{component}.base_model_name_or_path": base.name for component, base in bases.items()}}
    write_file(tensors, tmp_path / lora.DIFFUSERS_WEIGHTS,
               {**metadata, lora.DIFFUSERS_METADATA: json.dumps(named)})
    loaded = LoRA.load(pipeline.model, pipeline.variables, tmp_path, layouts=pipeline.layouts)
    assert loaded.bases == bases
    loaded.save(loaded.variables, tmp_path / "again")
    again = LoRA.load(pipeline.model, pipeline.variables, tmp_path / "again", layouts=pipeline.layouts)
    assert again.bases == bases
    # This file's components differ in rank, which a run's record of the adapter keeps beside the bases.
    record = wrapper_layers(type(again.model))[1][0]._dew_wrapper()
    kept = from_record(AdapterRecord, json.loads(json.dumps(to_record(record, AdapterRecord))))
    assert kept.bases == bases and kept.bound()[0] == again.targets


def test_export_writes_the_diffusers_file_back(pipeline, tmp_path):
    adapter = LoRA.load(pipeline.model, pipeline.variables, SD, layouts=pipeline.layouts)
    adapter.save(adapter.variables, tmp_path)
    ours, metadata = read_file(tmp_path / lora.DIFFUSERS_WEIGHTS)
    theirs, _ = read_file(SD / lora.DIFFUSERS_WEIGHTS)
    assert ours.keys() == theirs.keys()
    for key in theirs:
        np.testing.assert_array_equal(ours[key], theirs[key])
    header = json.loads(metadata[lora.DIFFUSERS_METADATA])
    assert (
        header["unet.r"],
        header["unet.lora_alpha"],
        header["text_encoder.r"],
        header["text_encoder.lora_alpha"],
    ) == (4, 6, 2, 5)
    again = LoRA.load(pipeline.model, pipeline.variables, tmp_path, layouts=pipeline.layouts)
    assert again.targets == adapter.targets


# --------------------------------------------------------------------------
# An adapter on a model the registry built, from a run config
# --------------------------------------------------------------------------


REGISTRY_FIELDS = {"vocab_size": 256, "emb_features": 16, "num_layers": 2, "num_heads": 2,
                   "head_dim": 8, "mlp_features": 32, "max_seq_len": 16}



def squared_output(model, variables, inputs):
    """The mean squared output of `model` over `inputs`, in float32, as a loss's ratio."""
    from dew.objectives.base import Ratio

    outputs = model.apply(variables, inputs)
    return Ratio(jnp.sum(outputs.astype(jnp.float32) ** 2), jnp.asarray(outputs.size, jnp.float32))

def _registry_decoder():
    """A tiny causal transformer and its variables, built from the registry:
    no published checkpoint, so no weight layouts to resolve names through."""
    from dew.registry import ModelConfig

    config = ModelConfig("causal_transformer", {**REGISTRY_FIELDS,
                                              "dtype": "float32", "attention_impl": "reference"})
    model = config.build()
    return config, model, model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))


REGISTRY_SEEDS = 16
"""The fresh adapter's check-1 seeds, sized paired by `seeds_for` on an A100, where XLA merges the
adapter's product with the base's."""


def test_a_registry_model_adapts_through_the_names_its_own_kernels_carry(tmp_path):
    """No loader in sight: the module paths under `params` are the names
    `target_modules` matches, the fresh adapter is the identity (its `B` exactly zero, so the adapted
    model computes the base model's function, held by check 1 against the base model from its
    float64 twin: the two are different programs), and the factors write and read back in PEFT's
    layout under those names."""
    _, model, variables = _registry_decoder()
    adapter = LoRA(rank=2, modules=("q_proj", "layers_1.mlp.up_proj"), alpha=4.0).apply(
        model, variables, key=1)
    adapted = adapter.variables
    assert set(adapter.targets) == {("params", "layers_0", "self_attn", "q_proj"),
                                    ("params", "layers_1", "self_attn", "q_proj"),
                                    ("params", "layers_1", "mlp", "up_proj")}
    factors = [leaf for path, leaf in flat_tree(adapted["params"]).items() if path.endswith("lora_B")]
    assert len(factors) == 3 and not any(np.any(np.asarray(factor)) for factor in factors)
    wide = model.clone(dtype=jnp.float64)

    def draw(seed):
        return (np.random.default_rng([0x10A, seed]).integers(0, REGISTRY_FIELDS["vocab_size"], (2, 8)),)

    def truth(tokens):
        with jax.enable_x64(new_val=True):
            return np.asarray(wide.apply(widened(variables), jnp.asarray(tokens)))

    assert_as_exact(lambda tokens: adapter.model.apply(adapted, tokens),
                    lambda tokens: model.apply(variables, tokens), draw,
                    "a fresh adapter against its base", seeds=REGISTRY_SEEDS, truth_fn=truth)

    adapter.save(adapted, tmp_path)
    written = load_file(tmp_path / lora.PEFT_WEIGHTS)
    assert set(written) == {f"{lora.PEFT_PREFIX}{name}.lora_{factor}.weight"
                            for name in ("layers_0.self_attn.q_proj", "layers_1.self_attn.q_proj",
                                         "layers_1.mlp.up_proj") for factor in "AB"}
    assert written[f"{lora.PEFT_PREFIX}layers_1.mlp.up_proj.lora_A.weight"].shape == (2, 16)
    again = LoRA.load(model, variables, tmp_path)
    assert again.targets == adapter.targets
    for ours, theirs in zip(jax.tree.leaves(again.variables), jax.tree.leaves(adapted), strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))

    with pytest.raises(ValueError):
        LoRA(rank=2, modules=("to_q",), alpha=2.0).apply(model, variables, key=0)

    # Targets bound with no names say so instead of writing a file under
    # names they never resolved.
    unnamed = Adapter.bound(model, adapted, adapter.targets, rslora=False, dropout=0.0, layouts={})
    with pytest.raises(ValueError):
        unnamed.save(adapted, tmp_path / "unnamed")


def test_trained_factors_over_a_bf16_base_resume_on_the_rebuilt_model_as_an_uninterrupted_run(tmp_path):
    """Factors trained over bfloat16 base weights: a run checkpointed after two steps resumes on the
    model rebuilt from its record (`ModelConfig.from_model`), whose own initializer draws the
    restore template, its factors stored in float32 as attached. The restored state is the first
    run's step-2 state bit for bit, and its next two steps reach every leaf and the outputs an
    uninterrupted four-step run reaches, bit for bit. A template whose factors took the base's
    bfloat16 would round the restored factors; a resume that drew afresh would start elsewhere."""
    from dew.inputs import Field, InputSpec
    from dew.objectives.base import Objective
    from dew.registry import ModelConfig

    model = nn.Sequential([nn.Dense(8, param_dtype=jnp.bfloat16), nn.Dense(4, param_dtype=jnp.bfloat16)])
    rows = 2 * jax.device_count()
    inputs = np.asarray(jax.random.normal(jax.random.key(1), (rows, 6)))
    adapter = LoRA(rank=2, modules=("layers_0", "layers_1")).apply(
        model, model.init(jax.random.key(0), inputs), key=0)
    rebuilt = ModelConfig.from_model(adapter.model).build()

    class Squared(Objective):
        """The mean squared output; its variables those given, or its model's own draw."""

        inputs = InputSpec(Field("x", (6,)))

        def __init__(self, model, variables=None):
            self.model, self.variables = model, variables

        def init(self, key, variables=None):
            if variables is not None:
                return variables
            return self.model.init(key, inputs) if self.variables is None else self.variables

        def loss(self, variables, batch, step):
            return squared_output(self.model, variables, batch["x"])

    def data():
        # A stream that keeps its position, so a resume continues where the checkpoint left it;
        # each step's batch is its own, so a resume that restarted the data would diverge.
        return Loader(train=lambda partition: Replaying(lambda step: {"x": inputs * (1 + step)}), val=None,
                      records=rows, batch=rows)

    def trainer(objective, directory):
        return Trainer(objective, optax.adam(1e-2), key=jax.random.key(3), mesh=MeshSpec(),
                       checkpoints=Checkpoints(str(directory)))

    def factors(tree):
        return {name: leaf for name, leaf in flat_tree(tree["params"]).items() if name.endswith(lora.FACTORS)}

    whole = trainer(Squared(adapter.model, adapter.variables), tmp_path / "whole").fit(data(), steps=4,
                                                                                      log_every=4)
    first = trainer(Squared(adapter.model, adapter.variables), tmp_path / "split")
    halfway = first.fit(data(), steps=2, log_every=2, checkpoint_every=2)
    first.checkpoints.wait()

    resumed = trainer(Squared(rebuilt), tmp_path / "split")
    template = factors(Squared(rebuilt).init(jax.random.key(3)))
    assert {leaf.dtype for leaf in template.values()} == {jnp.dtype(jnp.float32)}
    restored, _, _ = resumed.place()
    assert int(restored.step) == 2
    for name, leaf in factors(halfway.variables).items():
        ours = factors(restored.variables)[name]
        assert ours.dtype == leaf.dtype and np.array_equal(np.asarray(ours), np.asarray(leaf)), name
    continued = resumed.fit(data(), steps=4, log_every=2)
    assert int(continued.step) == 4
    for (path, ours), theirs in zip(jax.tree_util.tree_leaves_with_path(continued.variables),
                                    jax.tree.leaves(whole.variables), strict=True):
        assert ours.dtype == theirs.dtype and np.array_equal(np.asarray(ours), np.asarray(theirs)), \
            jax.tree_util.keystr(path)
    np.testing.assert_array_equal(np.asarray(rebuilt.apply(continued.variables, inputs)),
                                  np.asarray(adapter.model.apply(whole.variables, inputs)))


def test_the_branch_reaches_every_layer_of_a_scanned_run(decoder, reference):
    """A scanned stack runs its like layers as one module, `layers_0_1`; the
    adapter's targets are named per layer, so the branch resolves the run to
    its layers and every layer's factors carry gradient. Before this, a
    stack cloned to `scan_layers=True` after adapting silently trained only
    the layers that ran alone."""
    adapter = LoRA(rank=2, modules=("q_proj",), alpha=4.0).apply(decoder.model, decoder.variables, key=1,
                                                                 layouts=decoder.layouts)
    variables = adapter.variables
    scanned = adapter.model.clone(scan_layers=True)
    tokens = jnp.asarray(reference["input_ids"])
    plain = adapter.model.apply(variables, tokens)
    np.testing.assert_allclose(
        np.asarray(scanned.apply(variables, tokens)), np.asarray(plain), rtol=1e-5, atol=1e-5
    )

    def loss(params):
        return jnp.mean(scanned.apply({**variables, "params": params}, tokens) ** 2)

    grads = jax.grad(loss)(variables["params"])
    for layer in ("layers_0", "layers_1"):
        factors = grads[layer]["self_attn"]["q_proj"]
        assert float(jnp.abs(factors["lora_B"]).max()) > 0, layer
    # One scanned module runs both layers, so their targets must agree.
    uneven = {**adapter.targets, ("params", "layers_1", "self_attn", "q_proj"): Target(2, 8.0)}
    with pytest.raises(ValueError):
        _rebound(adapter, decoder.model.clone(scan_layers=True), targets=uneven).model.apply(
            variables, tokens)


# --------------------------------------------------------------------------
# A published pipeline's own adapter, on its denoiser
# --------------------------------------------------------------------------


DENOISER_MODULES = ("to_q", "to_k", "to_v", "to_out.0")
PROMPTS = ["a red bird", "two cats"]


@pytest.fixture(scope="module")
def pipelines(tmp_path_factory):
    """The tiny FLUX and SD3 pipelines and the tiny SD one, loaded at float32."""
    root = tmp_path_factory.mktemp("pipelines")
    for family in ("flux", "sd3"):
        with tarfile.open(ROOT / f"tests/fixtures/{family}_source.tar.xz") as archive:
            archive.extractall(root / family, filter="data")
    with tarfile.open(ROOT / "tests/fixtures/tiny_diffusers.tar.xz") as archive:
        archive.extractall(root, members=[m for m in archive.getmembers() if m.name.startswith("sd/")],
                           filter="data")
    directories = {"flux": root / "flux" / "pipeline", "sd3": root / "sd3" / "pipeline", "sd": root / "sd"}
    return {family: Pretrained.load(directory, dtype="float32", attention_impl="xla")
            for family, directory in directories.items()}


def _sampled(task) -> np.ndarray:
    return np.asarray(task(PROMPTS, steps=2, key=1).host().images)


def _moved_b(tuned, seed: int):
    """The tuned variables with every B drawn away from zero, as a run leaves them."""
    keys = iter(jax.random.split(jax.random.key(seed), len(tuned.adapter.targets)))
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: (0.2 * jax.random.normal(next(keys), leaf.shape, leaf.dtype)
                            if path[-1].key == "lora_B" else leaf), tuned.variables)


@pytest.mark.parametrize("family", ["flux", "sd3", "sd"])
def test_a_pipeline_adapts_its_denoiser_alone_and_starts_as_the_source(family, pipelines):
    """`source.adapt(LoRA(...))` binds the denoiser's projections the modules name,
    never a text tower's: `q_proj` names only CLIP and T5 projections here,
    so it matches nothing. B starts at zero, so the adapted denoiser computes
    the source's own arithmetic plus exact zeros.

    Called op by op, every base op sees the source's inputs, so one call
    is bitwise the source's. Compiled whole, XLA fuses the zero branch's add
    into the reductions that read a projection (the RMSNorms after `to_q`
    and `to_k`), and how such a fusion vectorizes depends on the CPU: on
    CI's AMD runner the two sampled programs part by a few float32 ulps. So
    the sample is held to rounding, the adapted run as exact as the source,
    both measured from the source in float64 (`reference_error`); a branch
    that moved the output at all would sit far outside it."""
    source = pipelines[family]
    tuned = source.adapt(LoRA(rank=2, modules=DENOISER_MODULES), key=0)
    assert source.adapter is None and tuned.adapter is not None
    assert tuned.adapter.targets and all(path[0] == "params" for path in tuned.adapter.targets)
    task = source.text_to_image()
    prepared = task.prepare(PROMPTS, key=1, steps=2)
    time = jnp.full((prepared.noise.shape[0],), 0.7, jnp.float32)
    called = [np.asarray(task.process.denoiser(bundle.model, bundle.variables, prepared.conditions)
                         .raw(prepared.noise, time)) for bundle in (source, tuned)]
    np.testing.assert_array_equal(*called)

    def sampled(bundle, inputs) -> np.ndarray:
        return np.asarray(bundle.text_to_image()(inputs, steps=2, key=1, decode=False).host().latents)

    with jax.enable_x64():
        wide = dataclasses.replace(source, model=source.model.clone(dtype=jnp.float64),
                                   variables=widened(source.variables))
        truth = sampled(wide, widened(prepared))
    assert_as_exact_as_the_reference(sampled(tuned, prepared), sampled(source, prepared), truth, family)
    with pytest.raises(ValueError):
        source.adapt(LoRA(rank=2, modules=("q_proj",)), key=0)
    with pytest.raises(ValueError):
        tuned.adapt(LoRA(rank=2, modules=("to_q",)), key=1)


@pytest.mark.parametrize("family", ["flux", "sd3"])
def test_a_pipeline_lora_run_moves_its_factors_and_nothing_else(family, pipelines):
    """The adapted pipeline's objective trains the factors: after three steps
    the base transformer, the text towers and the VAE are bitwise where they
    started, every B has moved, and the trained state publishes a task."""
    tuned = pipelines[family].adapt(LoRA(rank=2, modules=DENOISER_MODULES), key=0)
    objective = DiffusionObjective(tuned, ema_decay=None, unconditional_prob=0.0)
    assert type(objective) is DiffusionObjective
    rows, (height, width, channels) = jax.device_count(), objective.inputs.sample.shape
    batch = {"image": np.tile(np.arange(height * width * channels, dtype=np.uint8).reshape(
                 1, height, width, channels), (rows, 1, 1, 1)),
             **objective.inputs.tokenize([PROMPTS[row % 2] for row in range(rows)])}
    trainer = Trainer(objective, optax.sgd(1e-2), key=jax.random.key(3))
    initial = trainer.initial_state()
    data = Loader(train=lambda partition: iter([batch] * 3), val=None, records=rows, batch=rows)
    state = trainer.fit(data, steps=3, log_every=3)

    assert set(flat_tree(state.variables["params"])) == {
        f"{'.'.join(path[1:])}.{factor}" for path in tuned.adapter.targets for factor in lora.FACTORS}
    for collection in (FROZEN, "encoders", "autoencoder"):
        for before, after in zip(jax.tree.leaves(initial.variables[collection]),
                                 jax.tree.leaves(state.variables[collection]), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=collection)
    assert all(bool(jnp.any(leaf)) for name, leaf in flat_tree(state.variables["params"]).items()
               if name.endswith("lora_B"))
    assert np.isfinite(_sampled(objective.pipeline(state))).all()


@pytest.mark.parametrize("family", ["flux", "sd3"])
def test_a_tuned_pipeline_saves_its_adapter_and_its_merged_weights(family, pipelines, tmp_path):
    """`adapter.save` writes the Diffusers file under the denoiser's component,
    which `LoRA.load` reads back to the same adapter, and `save` writes the
    pipeline with the factors merged into its kernels: reloaded, it samples
    exactly what the source does over the merged weights."""
    source = pipelines[family]
    tuned = source.adapt(LoRA(rank=2, modules=DENOISER_MODULES), key=0)
    variables = _moved_b(tuned, 7)
    tuned.adapter.save(variables, tmp_path / "adapter")
    tensors, metadata = read_file(tmp_path / "adapter" / lora.DIFFUSERS_WEIGHTS)
    assert tensors and all(name.startswith("transformer.") for name in tensors)
    assert json.loads(metadata[lora.DIFFUSERS_METADATA])["transformer.r"] == 2
    loaded = LoRA.load(source.model, source.variables, tmp_path / "adapter", layouts=source.layouts)
    assert (loaded.targets, loaded.layouts) == (tuned.adapter.targets, tuned.adapter.layouts)
    for name, leaf in flat_tree(thaw(loaded.variables)).items():
        if "lora_" in name:
            np.testing.assert_array_equal(np.asarray(leaf), np.asarray(flat_tree(thaw(variables))[name]),
                                          err_msg=name)

    tuned.save(tmp_path / "merged", variables=variables)
    reloaded = Pretrained.load(tmp_path / "merged", dtype="float32", attention_impl="xla")
    merged = dataclasses.replace(source, variables=tuned.adapter.merge(variables))
    np.testing.assert_array_equal(_sampled(reloaded.text_to_image()), _sampled(merged.text_to_image()))
    assert np.abs(_sampled(merged.text_to_image()) - _sampled(source.text_to_image())).max() > 0


EXPORTS = ROOT / "tests" / "fixtures" / "lora_exports"


def _recorded(name: str) -> dict[str, np.ndarray]:
    with np.load(EXPORTS / f"{name}.npz") as data:
        return {key: data[key] for key in data if key != "meta"}


def _layout(directory: Path) -> dict:
    """Each file of an adapter as its reader parses it, values aside: a safetensors file's metadata
    and every tensor's name, dtype and shape, a JSON file's contents."""
    from safetensors import safe_open

    layout = {}
    for path in sorted(directory.iterdir()):
        if path.suffix == ".safetensors":
            with safe_open(str(path), "numpy") as file:
                layout[path.name] = (file.metadata(), {name: (str(file.get_tensor(name).dtype),
                                                              file.get_tensor(name).shape)
                                                       for name in sorted(file.keys())})
        else:
            layout[path.name] = json.loads(path.read_text())
    return layout


def _written_as_read(name: str, source, variables, directory: Path) -> Adapter:
    """The adapter `export_case` wrote into `directory` on this device reads back as the variables
    it wrote, in the layout of the files PEFT or Diffusers read (tests/fixtures/lora_exports/<name>,
    which tools/lora_export_reference.py's `export` wrote); returns the adapter those files hold,
    which the reader's outputs were recorded from."""
    assert _layout(directory) == _layout(EXPORTS / name), (
        "Dew's adapter files changed: rerun both halves of tools/lora_export_reference.py, "
        "`consume` in its PEFT environment, so PEFT and Diffusers read the new files")
    again = LoRA.load(source.model, source.variables, directory, layouts=source.layouts)
    for written, read in zip(jax.tree.leaves(variables["params"]), jax.tree.leaves(again.variables["params"]),
                             strict=True):
        np.testing.assert_array_equal(np.asarray(read), np.asarray(written))
    return LoRA.load(source.model, source.variables, EXPORTS / name, layouts=source.layouts)


@pytest.mark.parametrize("name", ["llama", "llama-rslora", "llama-patterns"])
def test_peft_reads_the_adapter_dew_writes_at_dews_logits(name, decoder, tmp_path):
    """tools/lora_export_reference.py records PEFT 0.20.0 reading the
    adapter a case writes (a fresh one at a rank and alpha, one with
    rsLoRA, and the committed one whose rank and alpha patterns Dew writes
    back, each with its factors moved) onto transformers' llama-tiny: every
    tensor of the file lands, and the adapted and `merge_and_unload` logits
    are recorded in float32 and float64. The adapter written here, on this
    device, reads back as written, in the layout of the files PEFT read, and
    Dew's adapted and merged logits from those files hold the float64 rule
    against PEFT's."""
    from tools.lora_export_reference import CASES, export_case

    _, written = export_case(CASES[name], decoder, tmp_path)
    adapter = _written_as_read(name, decoder, written, tmp_path)
    variables, arrays = adapter.variables, _recorded(name)
    ids = jnp.asarray(arrays["input_ids"])
    adapted = adapter.model.apply(variables, ids)
    merged = decoder.model.apply(adapter.merge(variables), ids)
    assert np.abs(arrays["fp32.adapted"] - arrays["fp32.base"]).max() > 1
    assert_as_exact_as_the_reference(np.asarray(adapted), arrays["fp32.adapted"], arrays["fp64.adapted"],
                                     f"{name} adapted")
    assert_as_exact_as_the_reference(np.asarray(merged), arrays["fp32.merged"], arrays["fp64.merged"],
                                     f"{name} merged")


@pytest.mark.parametrize("family", ["sd", "flux", "sd3"])
def test_diffusers_reads_the_adapter_dew_writes_at_dews_prediction(family, pipelines, tmp_path):
    """The pipelines' counterpart: Diffusers 0.34.0's `load_lora_weights`
    reads the file a case writes (rsLoRA on FLUX), every tensor lands in the
    denoiser, and its adapted and `fuse_lora` predictions are recorded in
    float32 and float64. Dew's adapted and merged predictions hold the
    float64 rule against them. The tiny SD source declares Flax classes,
    whose pipeline loads no adapter; its PyTorch UNet reads the file, so
    Dew's runs as that UNet (`torch_unet`)."""
    from tools.lora_export_reference import CASES, export_case

    source = pipelines[family]
    _, written = export_case(CASES[family], source, tmp_path)
    adapter = _written_as_read(family, source, written, tmp_path)
    variables, arrays = adapter.variables, _recorded(family)
    model = torch_unet(source.model) if family == "sd" else source.model
    if family == "sd":
        adapter = _rebound(adapter, model)
    own = {name: tree for name, tree in variables.items() if name not in ("encoders", "autoencoder")}
    guidance = jnp.asarray(arrays["guidance"]) if arrays.get("guidance", np.zeros(0)).size else None
    condition = DenoisingCondition(jnp.asarray(arrays["context"]),
                                   jnp.asarray(arrays["pooled"]) if "pooled" in arrays else None,
                                   guidance=guidance)
    latent, times = jnp.asarray(arrays["latent"]), jnp.asarray(arrays["times"])
    # The UNet takes its condition by keyword alone.
    adapted = adapter.model.apply(own, latent, times, conditioning=condition)
    merged = model.apply(adapter.merge(own), latent, times, conditioning=condition)
    assert np.abs(arrays["fp32.adapted"] - arrays["fp32.base"]).max() > 0.1
    assert_as_exact_as_the_reference(np.asarray(adapted), arrays["fp32.adapted"], arrays["fp64.adapted"],
                                     f"{family} adapted")
    assert_as_exact_as_the_reference(np.asarray(merged), arrays["fp32.merged"], arrays["fp64.merged"],
                                     f"{family} merged")
def _recipe_run(tmp_path, image_size, *args):
    """A diffusion recipe run from argv as a user types it: two steps on the
    TFDS image fixture, with `args` ending in the run's LoRA flags
    (`--lora.rank`, `--lora.modules`) and no subcommand. Returns the parsed
    config and the trained state."""
    from dew.config.diffusion import DiffusionRunConfig
    from dew.data import images as image_steps, presets
    from dew.data.sources.tfds import TFDS

    # Each image captioned from its row (`images.caption`), the condition the run's text tower reads.
    images = TFDS("dew_images", path=str(IMAGES), split="all").shuffle(seed=0).repeat().pipe(
        presets.images, image_size, batch=jax.device_count(), caption=image_steps.caption())
    config = DiffusionRunConfig.cli([
        "--model.dtype", "float32", "--model.attention-impl", "xla", "--objective.solver",
        '{"class": "euler"}', "--objective.guidance", "None", "--objective.steps", "2",
        "--objective.ema-decay", "None", "--val-metrics", "[]",
        "--trainer.checkpoint-dir", str(tmp_path),
        "--trainer.steps", "2", "--trainer.eval-every", "None", "--trainer.checkpoint-every", "2",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False", "--trainer.name", "run",
        *args], default=DiffusionRunConfig(data=images))
    return config, config.run()


def _same_files(ours: Path, theirs: Path) -> None:
    """Every file one adapter directory holds, byte for byte in the other."""
    names = sorted(path.name for path in theirs.iterdir())
    assert sorted(path.name for path in ours.iterdir()) == names and names
    for name in names:
        assert (ours / name).read_bytes() == (theirs / name).read_bytes(), name


def test_a_pretrained_pipeline_run_from_the_command_line_trains_its_lora(pipelines, tmp_path):
    """`--pretrained <pipeline> --lora.rank 2 --lora.modules to_q ...` through
    the diffusion recipe's own argv: the flags alone turn the adapter on, the
    run binds it to the loaded denoiser, two steps move every factor and
    leave the base, the text towers and the VAE bitwise, and
    `Adapter.from_run` writes, byte for byte, the Diffusers file an adapter
    bound the same way in process writes."""
    from dew.interop.pretrained import load_diffusion_source

    directory = pipelines["flux"].source
    config, state = _recipe_run(tmp_path, 16, "--pretrained", str(directory), "preset:none",
                                "--objective.unconditional-prob", "0.0", "--lora.rank", "2", "--lora.modules",
                                *DENOISER_MODULES)
    assert config.lora == LoRA(rank=2, modules=DENOISER_MODULES)

    source = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla", size=(16, 16))
    tuned = source.adapt(config.lora, key=config.trainer.key)
    moved = flat_tree(state.variables["params"])
    assert moved and all(name.endswith(lora.FACTORS) for name in moved)
    assert all(bool(jnp.any(leaf != flat_tree(tuned.variables["params"])[name]))
               for name, leaf in moved.items() if name.endswith("lora_B"))
    for collection in (FROZEN, "encoders", "autoencoder"):
        for before, after in zip(jax.tree.leaves(tuned.variables[collection]),
                                 jax.tree.leaves(state.variables[collection]), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=collection)
    assert tuned.adapter is not None
    tuned.adapter.save(state.variables, tmp_path / "in-process")
    rebuilt = Adapter.from_run(tmp_path / "run")
    rebuilt.save(rebuilt.variables, tmp_path / "from-run")
    _same_files(tmp_path / "from-run", tmp_path / "in-process")


def test_a_custom_objective_trains_only_an_adapters_factors(decoder, reference):
    """An objective written against the plain model, with no filter of its
    own, trains an adapter: the adapted model folds the frozen base back in
    itself, so the optimizer moves the factors alone and the base comes
    back bitwise."""
    from dew.inputs import Field, InputSpec
    from dew.objectives.base import Objective

    tokens = np.asarray(reference["input_ids"])

    class Squared(Objective):
        """The mean squared logit, over whatever tree it is handed."""

        inputs = InputSpec(Field("text", tokens.shape[1:]))

        def __init__(self, model, variables):
            self.model, self.variables = model, variables

        def init(self, key, variables=None):
            return self.variables if variables is None else variables

        def loss(self, variables, batch, step):
            return squared_output(self.model, variables, batch["text"])

    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(decoder.model, decoder.variables, key=0,
                                                               layouts=decoder.layouts)
    rows = jax.device_count()
    data = Loader(train=lambda partition: iter([{"text": tokens[np.arange(rows) % 2]}] * 2), val=None,
                   records=rows, batch=rows)
    trainer = Trainer(Squared(adapter.model, adapter.variables), optax.sgd(0.1), key=0)
    initial = trainer.initial_state()
    state = trainer.fit(data, steps=2, log_every=2)

    assert set(flat_tree(state.variables["params"])) == {
        f"{'.'.join(target[1:])}.{factor}" for target in adapter.targets for factor in lora.FACTORS}
    before = flat_tree(initial.variables["params"])
    assert all(bool(jnp.any(leaf != before[name]))
               for name, leaf in flat_tree(state.variables["params"]).items() if name.endswith("lora_B"))
    for before, after in zip(jax.tree.leaves(initial.variables[FROZEN]),
                             jax.tree.leaves(state.variables[FROZEN]), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_a_scratch_diffusion_run_from_the_command_line_trains_its_lora(tmp_path):
    """`--lora.rank 2 --lora.modules final_proj` on a diffusion run from
    scratch, through the recipe's argv: the run binds the adapter to the
    objective's own fresh draw of the denoiser from the run's key, with the
    factors from the key folded with 1. Two steps move every factor and leave
    the drawn weights and the text tower bitwise, and `Adapter.from_run`
    writes, byte for byte, the PEFT directory an adapter bound the same way
    in process writes. The DiT's blocks are adaLN-Zero, so with their
    modulation frozen at zero only the output projection carries a gradient;
    it is the one target."""
    config, state = _recipe_run(
        tmp_path, 8, "--model", "simple_dit", "--model.patch_size", "2", "--model.emb_features", "16",
        "--model.num_layers", "1", "--model.num_heads", "2",
        "--text.encoder", "char_table", "--text.checkpoint", "char_table", "--lora.rank", "2",
        "--lora.modules", "final_proj")
    assert config.lora == LoRA(rank=2, modules=("final_proj",))

    moved = flat_tree(state.variables["params"])
    assert set(moved) == {f"output.final_proj.{factor}" for factor in lora.FACTORS}
    plain = dataclasses.replace(config, lora=None).build()
    key = jax.random.key(config.trainer.key)
    drawn = plain.model_variables(plain.init(key))
    for before, after in zip(jax.tree.leaves(drawn["params"]), jax.tree.leaves(state.variables[FROZEN]),
                             strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    adapter = config.lora.apply(plain.model, drawn, key=jax.random.fold_in(key, 1))
    assert all(bool(jnp.any(leaf != flat_tree(adapter.variables["params"])[name]))
               for name, leaf in moved.items() if name.endswith("lora_B"))
    adapter.save(state.variables, tmp_path / "in-process")
    rebuilt = Adapter.from_run(tmp_path / "run")
    rebuilt.save(rebuilt.variables, tmp_path / "from-run")
    _same_files(tmp_path / "from-run", tmp_path / "in-process")


def test_an_imported_adapter_binds_to_its_bundle_as_a_fresh_one_does(decoder, tmp_path):
    """`source.adapt(LoRA(..., source=<PEFT directory>))` binds an imported adapter to the bundle as a
    fresh spec's does: its factors are the file's (the adapted model computes what `LoRA.load` binds,
    bit for bit), the adapter saves its factors back under the source's names, and the bundle saves the
    merged weights a plain load computes. A spec stating another rank, alpha, target, rslora or dropout
    than the file holds is refused, as is the PEFT fixture whose modules take ranks and alphas of their
    own, which one spec cannot state."""
    spec = LoRA(rank=2, modules=("q_proj", "v_proj"), alpha=4.0, dropout=0.0)
    drawn = decoder.adapt(spec, key=0)
    # Factors that move the logits, so the import is no identity.
    trained = {**drawn.variables, "params": jax.tree.map(lambda leaf: leaf + 0.05, drawn.variables["params"])}
    drawn.adapter.save(trained, tmp_path / "imported")
    expected = LoRA.load(decoder.model, decoder.variables, tmp_path / "imported", layouts=decoder.layouts)
    stated = dataclasses.replace(spec, source=str(tmp_path / "imported"))
    bundled = decoder.adapt(stated, key=1)
    ids = jnp.arange(12, dtype=jnp.int32).reshape(2, 6) % decoder.model.vocab_size
    found = np.asarray(bundled.model.apply(bundled.variables, ids))
    np.testing.assert_array_equal(found, np.asarray(expected.model.apply(expected.variables, ids)))
    assert not np.array_equal(found, np.asarray(decoder.model.apply(decoder.variables, ids)))
    bundled.adapter.save(bundled.variables, tmp_path / "again")
    again = LoRA.load(decoder.model, decoder.variables, tmp_path / "again", layouts=decoder.layouts)
    jax.tree.map(np.testing.assert_array_equal, thaw(again.variables), thaw(expected.variables))
    bundled.save(tmp_path / "merged")
    merged = Pretrained.load(tmp_path / "merged", dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(np.asarray(merged.model.apply(merged.variables, ids)),
                                  np.asarray(decoder.model.apply(expected.merge(expected.variables), ids)))
    for other in ({"rank": 4}, {"alpha": 8.0}, {"modules": ("q_proj",)}, {"rslora": True},
                  {"dropout": 0.1}):
        with pytest.raises(ValueError):
            decoder.adapt(dataclasses.replace(stated, **other), key=0)
    with pytest.raises(ValueError):
        decoder.adapt(LoRA(rank=4, modules=("q_proj", "v_proj", "down_proj"), alpha=8, dropout=0.1,
                           source=str(ADAPTER)), key=0)


TRAIN_IMPORTED = """
import sys
from dew.config.lm import LMRunConfig
from dew.data import Loading, presets
from dew.data.sources.tokens import TokenFiles
from dew.registry import ModelConfig

tokens, root, name = sys.argv[1:4]
chain = TokenFiles(tokens).pipe(presets.lm, None, seq_len=8, batch=8, pack=None).loading(Loading(workers=0))
model = ModelConfig("causal_transformer",
                    {"emb_features": 16, "num_layers": 1, "num_heads": 2, "dtype": "float32"})
LMRunConfig.cli([*sys.argv[4:], "--trainer.name", name, "--trainer.checkpoint-dir", root,
                 "--trainer.steps", "1",
                 "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False",
                 "--trainer.eval-every", "None", "--trainer.checkpoint-every", "1", "--max-new-tokens", "0",
                 "--optim.learning-rate", "0"], default=LMRunConfig(data=chain, model=model)).run()
"""
"""A recorded LM run in a process of its own, which reads `HF_ENDPOINT` as it imports the Hub client:
`tokens`, the runs' root and the run's name, then its own flags."""


def test_a_recorded_run_imports_its_adapter_from_the_hub_and_replays_it_offline(tmp_path):
    """A run whose `--lora.source` names a Hub repo (here on the loopback) starts from that adapter's
    factors and records the repo at the commit it read (`RunConfig.pinned`); with the Hub gone, the same
    command reads the cached commit and records it alike. At a zero learning rate each run's factors
    are the served adapter's, bit for bit."""
    import json
    import subprocess
    import sys

    from local_hub import LocalHub, hub_environment
    from token_support import write_token_corpus

    ids = np.random.RandomState(0).randint(1, 250, 640).astype(np.uint8)
    tokens = write_token_corpus(tmp_path / "tokens", ids)
    spec = ["--lora.rank", "2", "--lora.modules", "q_proj", "v_proj", "--lora.alpha", "4.0"]

    def trained(name, *flags, hub=None, offline=False):
        done = subprocess.run([sys.executable, "-c", TRAIN_IMPORTED, tokens.train, str(tmp_path / "runs"),
                               name, *spec, *flags], env=hub_environment(hub, tmp_path, offline=offline),
                              capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr[-3000:]
        return tmp_path / "runs" / name

    # A fresh adapter's run writes the factors the Hub then serves, moved off zero so they compute.
    fresh = Adapter.from_run(trained("fresh"))
    moved = {**fresh.variables, "params": jax.tree.map(lambda leaf: leaf + 0.05, fresh.variables["params"])}
    fresh.save(moved, tmp_path / "served")
    served = {name: tmp_path / "served" / name for name in (lora.PEFT_CONFIG, lora.PEFT_WEIGHTS)}
    with LocalHub({"local/adapter": served}) as hub:
        online = trained("online", "--lora.source", "local/adapter", hub=hub)
        commit = hub.commit("local/adapter")
    offline = trained("offline", "--lora.source", "local/adapter", offline=True)
    for run in (online, offline):
        recorded = json.loads((run / "run.json").read_text())["fields"]["lora"]
        assert recorded["source"] == f"local/adapter@{commit}"
        restored = Adapter.from_run(run, step=1)
        jax.tree.map(np.testing.assert_array_equal, thaw(restored.variables)["params"],
                     thaw(moved)["params"])


ADAPT_IMPORTED = """
import sys
import jax.numpy as jnp
import numpy as np
from dew.interop.pretrained import Pretrained
from dew.lora import LoRA

decoder = Pretrained.load(sys.argv[1], dtype="float32", attention_impl="reference")
spec = LoRA(rank=2, modules=("q_proj", "v_proj"), alpha=4.0, source=sys.argv[2], weight_name=sys.argv[3])
bundled = decoder.adapt(spec, key=0)
ids = jnp.arange(12, dtype=jnp.int32).reshape(2, 6) % decoder.model.vocab_size
np.save(sys.argv[4], np.asarray(bundled.model.apply(bundled.variables, ids)))
"""
"""An imported adapter bound to a decoder in a process of its own, which reads `HF_ENDPOINT` as it
imports the Hub client: the decoder, the adapter's source and its weight name, and where its logits go."""


def test_an_adapter_named_by_its_folder_fetches_the_files_it_reads_alone(decoder, tmp_path):
    """An adapter inside a checkpoint folder (`--lora.weight-name checkpoint-10`) beside the trainer's
    own state loads from a local directory and from a Hub repo alike, computing the same logits, bit for
    bit, as `LoRA.load` of that folder; from the Hub only its PEFT pair is fetched, the optimizer state
    and the root's other files left there."""
    import subprocess
    import sys

    from local_hub import LocalHub, hub_environment

    drawn = decoder.adapt(LoRA(rank=2, modules=("q_proj", "v_proj"), alpha=4.0), key=0)
    moved = {**drawn.variables, "params": jax.tree.map(lambda leaf: leaf + 0.05, drawn.variables["params"])}
    checkpoint = tmp_path / "local" / "checkpoint-10"
    drawn.adapter.save(moved, checkpoint)
    (checkpoint / "optimizer.pt").write_bytes(b"\x80" * 64)
    (tmp_path / "local" / "other.safetensors").write_bytes(b"\x00" * 16)
    ids = jnp.arange(12, dtype=jnp.int32).reshape(2, 6) % decoder.model.vocab_size
    expected = LoRA.load(decoder.model, decoder.variables, checkpoint, layouts=decoder.layouts)
    logits = np.asarray(expected.model.apply(expected.variables, ids))
    spec = LoRA(rank=2, modules=("q_proj", "v_proj"), alpha=4.0, source=str(tmp_path / "local"),
                weight_name="checkpoint-10")
    local = decoder.adapt(spec, key=1)
    np.testing.assert_array_equal(np.asarray(local.model.apply(local.variables, ids)), logits)
    served = {path.relative_to(tmp_path / "local").as_posix(): path
              for path in (tmp_path / "local").rglob("*") if path.is_file()}
    with LocalHub({"local/checkpoints": served}) as hub:
        done = subprocess.run([sys.executable, "-c", ADAPT_IMPORTED, str(LLAMA), "local/checkpoints",
                               "checkpoint-10", str(tmp_path / "hub.npy")],
                              env=hub_environment(hub, tmp_path), capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr[-3000:]
        fetched = {read.file for read in hub.served if read.repo == "local/checkpoints" and read.file}
    np.testing.assert_array_equal(np.load(tmp_path / "hub.npy"), logits)
    assert fetched <= {"checkpoint-10/adapter_config.json", "checkpoint-10/adapter_model.safetensors"}
