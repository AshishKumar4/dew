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
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference
from safetensors.numpy import load_file, save_file

from dew import lora
from dew.data import Dataset
from dew.diffusion.process import DenoisingCondition
from dew.inputs.diffusion import _text_features
from dew.interop.pretrained import Pretrained
from dew.interop.safetensors_io import read_file, write_file
from dew.lora import Adapter, LoRA, Target
from dew.objectives.base import FROZEN, Step, freeze, merge, thaw
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import LMObjective
from dew.training import Layout, MeshSpec, Trainer

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


def test_the_config_patterns_decide_each_target(loaded):
    """The PEFT config's r and lora_alpha apply except where a rank_pattern
    or alpha_pattern names the module the way PEFT's get_pattern_key does."""
    adapter = loaded
    variables = adapter.variables
    ranks = {"/".join(path[1:]): target.rank for path, target in adapter.targets.items()}
    alphas = {"/".join(path[1:]): target.alpha for path, target in adapter.targets.items()}
    assert ranks == {
        "layers_0/self_attn/q_proj": 4,
        "layers_0/self_attn/v_proj": 4,
        "layers_0/mlp/down_proj": 4,
        "layers_1/self_attn/q_proj": 4,
        "layers_1/self_attn/v_proj": 2,
        "layers_1/mlp/down_proj": 4,
    }
    assert alphas == {
        "layers_0/self_attn/q_proj": 8,
        "layers_0/self_attn/v_proj": 8,
        "layers_0/mlp/down_proj": 3,
        "layers_1/self_attn/q_proj": 8,
        "layers_1/self_attn/v_proj": 8,
        "layers_1/mlp/down_proj": 3,
    }
    assert adapter.dropout == 0.1 and not adapter.rslora
    assert variables["params"]["layers_1"]["self_attn"]["v_proj"]["lora_A"].shape == (64, 2)
    assert variables["params"]["layers_1"]["self_attn"]["v_proj"]["lora_B"].shape == (2, 32)


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


def test_adapting_an_adapted_model_is_refused(decoder, loaded):
    with pytest.raises(ValueError, match="already adapted"):
        LoRA(rank=2, modules=("q_proj",)).apply(loaded.model, loaded.variables, key=0,
                                                layouts=decoder.layouts)


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

    data = Dataset(train=lambda partition: iter([{"text": tokens[np.arange(rows) % 2]}] * 3), val=None,
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
    reference's step on the adapter parameters."""
    adapter = _rebound(loaded, decoder.model, dropout=0.0)
    meta = json.loads((FIXTURES / "llama-tiny" / "meta.json").read_text())
    tokens = np.asarray(reference["input_ids"])
    rows = 2 * jax.device_count()
    objective = LMObjective(adapter.model, tokens.shape[1] - 1, variables=adapter.variables, ema_decay=None)
    data = Dataset(
        train=lambda partition: iter([{"text": tokens[np.arange(rows) % 2]}]),
        val=None,
        records=rows,
        batch=rows,
    )
    trainer = Trainer(objective, optax.sgd(meta["learning_rate"]), key=jax.random.key(3),
                      mesh=MeshSpec(), layout=Layout(min_shard=2**30))
    initial = trainer.initial_state()

    state = trainer.fit(data, steps=1, log_every=1)

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
        Dataset(train=lambda partition: iter([batch, batch]), val=None, records=rows, batch=rows),
        steps=2,
        log_every=2,
    )

    assert set(_flat(state.variables["params"])) == {
        f"{'.'.join(target[1:])}.{factor}" for target in tuned.adapter.targets for factor in lora.FACTORS}
    for before, after in zip(
        jax.tree.leaves(initial.variables[FROZEN]), jax.tree.leaves(state.variables[FROZEN]), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    trained = thaw(state.variables)
    tuned.adapter.save(state.variables, tmp_path / "adapter")
    read = LoRA.load(source.model, source.variables, tmp_path / "adapter", layouts=source.layouts)
    for name, leaf in _flat(thaw(read.variables)).items():
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(_flat(trained)[name]), err_msg=name)
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


def test_an_adapted_bundle_refuses_a_second_adapter(decoder):
    tuned = decoder.adapt(LoRA(rank=2, modules=("q_proj",)), key=0)
    with pytest.raises(ValueError, match="already carries an adapter"):
        tuned.adapt(LoRA(rank=2, modules=("v_proj",)), key=1)


def test_a_text_request_without_a_processor_names_the_argument(decoder):
    from dew.inference import TextGeneration

    with pytest.raises(ValueError, match="processor="):
        TextGeneration(decoder.model, decoder.variables)("hello", 2, key=0)


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


def test_a_run_records_the_spec_it_was_asked_for_and_the_adapter_it_bound(decoder):
    """`RunConfig.lora` is the spec, PEFT's fields, and round-trips as one.
    The model record carries the bound adapter with each target's binding,
    the name its source writes it under, so `Adapter.recorded` rebuilds an
    adapter that saves under those names with no source at hand, and a
    binding the checkpoint's factors do not fit is refused by name."""
    from dew.config import ModelConfig, RunConfig

    spec = LoRA(rank=2, modules=("q_proj", "down_proj"), alpha=3.0, dropout=0.1)
    record = json.loads(json.dumps(RunConfig(lora=spec).to_dict()))
    assert record["lora"] == {"rank": 2, "modules": ["q_proj", "down_proj"], "alpha": 3.0, "rslora": False,
                              "dropout": 0.1}
    assert RunConfig.from_dict(record).lora == spec

    adapter = decoder.adapt(spec, key=0).adapter
    assert adapter is not None
    model = json.loads(json.dumps(RunConfig(model=ModelConfig.from_model(adapter.model)).to_dict()))["model"]
    recorded = model["adapter"]
    assert recorded["layouts"]["params/layers_0/self_attn/q_proj"] == {
        "name": "model.layers.0.self_attn.q_proj.weight", "shape": [64, 64], "transpose": [1, 0]}
    rebuilt = Adapter.recorded(ModelConfig.from_dict(model).build(), adapter.variables, recorded)
    assert (rebuilt.targets, rebuilt.layouts) == (adapter.targets, adapter.layouts)

    narrow = json.loads(json.dumps(recorded))
    narrow["layouts"]["params/layers_0/self_attn/q_proj"]["shape"] = [32, 64]
    with pytest.raises(ValueError, match=r"the recorded binding model\.layers\.0\.self_attn\.q_proj"):
        Adapter.recorded(adapter.model, adapter.variables, narrow)


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
    with pytest.raises(ValueError, match="w_proj"):
        LoRA(rank=2, modules=("q_proj", "w_proj"), alpha=2.0).apply(
            decoder.model, decoder.variables, key=0, layouts=decoder.layouts)


def test_the_branch_drops_out_only_under_a_dropout_stream(decoder, loaded, reference):
    adapter, variables = loaded, loaded.variables
    tokens = jnp.asarray(reference["input_ids"])
    plain = adapter.model.apply(variables, tokens)
    dropped = adapter.model.apply(variables, tokens, rngs={"dropout": jax.random.key(1)})
    assert np.max(np.abs(np.asarray(dropped) - np.asarray(plain))) > 1e-3
    kept = _rebound(adapter, decoder.model, dropout=0.0).model.apply(
        variables, tokens, rngs={"dropout": jax.random.key(1)})
    np.testing.assert_array_equal(np.asarray(kept), np.asarray(plain))


def test_rslora_scales_by_the_square_root_of_the_rank(decoder, loaded, reference):
    adapter, variables = loaded, loaded.variables
    tokens = jnp.asarray(reference["input_ids"])
    scaled = _rebound(adapter, decoder.model, rslora=True)
    alpha_times_root = {path: dataclasses.replace(target, alpha=target.alpha * np.sqrt(target.rank))
                       for path, target in adapter.targets.items()}
    equivalent = _rebound(adapter, decoder.model, targets=alpha_times_root)
    np.testing.assert_allclose(
        np.asarray(scaled.model.apply(variables, tokens)),
        np.asarray(equivalent.model.apply(variables, tokens)),
        atol=1e-5,
        rtol=0,
    )


def _write_peft(directory: Path, tensors, config_updates=None):
    config = json.loads((ADAPTER / lora.PEFT_CONFIG).read_text())
    config.update(config_updates or {})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / lora.PEFT_CONFIG).write_text(json.dumps(config))
    save_file(tensors, directory / lora.PEFT_WEIGHTS)
    return directory


def test_refusals_name_the_reason(decoder, tmp_path):
    fixture = load_file(ADAPTER / lora.PEFT_WEIGHTS)
    prefix = lora.PEFT_PREFIX + "model.layers.0.self_attn.q_proj"
    a, b = fixture[prefix + ".lora_A.weight"], fixture[prefix + ".lora_B.weight"]

    def load(name, tensors, config=None):
        directory = _write_peft(tmp_path / name, tensors, config)
        return LoRA.load(decoder.model, decoder.variables, directory, layouts=decoder.layouts)

    with pytest.raises(ValueError, match=r"layers.5.self_attn.q_proj, which this source does not bind"):
        load("unbound", {lora.PEFT_PREFIX + "model.layers.5.self_attn.q_proj.lora_A.weight": a,
                         lora.PEFT_PREFIX + "model.layers.5.self_attn.q_proj.lora_B.weight": b})
    with pytest.raises(ValueError, match="stores rank 2 but its config declares 4"):
        load("rank", {prefix + ".lora_A.weight": a[:2], prefix + ".lora_B.weight": b[:, :2]})
    with pytest.raises(ValueError, match=r"delta of \(64, 32\) on a weight the source stores as \(64, 64\)"):
        load("shape", {prefix + ".lora_A.weight": a[:, :32], prefix + ".lora_B.weight": b})
    with pytest.raises(ValueError, match="not a projection weight"):
        load("norm", {lora.PEFT_PREFIX + "model.norm.lora_A.weight": a,
                      lora.PEFT_PREFIX + "model.norm.lora_B.weight": b})
    with pytest.raises(ValueError, match="use_dora"):
        load("dora", {prefix + ".lora_A.weight": a, prefix + ".lora_B.weight": b}, {"use_dora": True})
    with pytest.raises(ValueError, match="kohya"):
        load("kohya", {"lora_unet_down_blocks_0.alpha": np.full((), 4, np.float32)})
    with pytest.raises(ValueError, match="lora_A without its partner"):
        load("half", {prefix + ".lora_A.weight": a})


def test_a_per_expert_source_tensor_takes_no_adapter():
    """A stacked expert leaf answers for every expert's tensor, so one
    expert's delta has no leaf of its own."""
    mixtral = Pretrained.load(ROOT / "tests" / "fixtures" / "hf" / "mixtral-tiny", dtype="float32",
                              attention_impl="reference")
    with pytest.raises(ValueError, match=r"experts.0.w1 is assembled from several leaves"):
        LoRA(rank=2, modules=("w1",), alpha=2.0).apply(mixtral.model, mixtral.variables, key=0,
                                                       layouts=mixtral.layouts)


def test_a_target_that_is_not_a_dense_is_refused_when_called(decoder, reference):
    """A record names native module paths, which no binding checks against
    a module's kind; the branch refuses a target that is not a projection."""
    record = {"rank": 2, "alpha": 2.0, "rslora": False, "dropout": 0.0, "modules": ["params/embed_tokens"],
              "layouts": {}}
    with pytest.raises(TypeError, match=r"params/embed_tokens.*targets nn.Dense and nn.DenseGeneral kernels"):
        lora.adapted(decoder.model, record).apply(decoder.variables, jnp.asarray(reference["input_ids"]))


def test_freeze_refuses_a_filter_that_splits_nothing(decoder):
    with pytest.raises(ValueError, match="trains nothing"):
        freeze(decoder.variables, lambda path: False)
    with pytest.raises(ValueError, match="freezes nothing"):
        freeze(decoder.variables, lambda path: True)


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
    features = tower.apply({"params": _subtree(merged, TEXT_ROOT)}, ids, method=_text_features)
    np.testing.assert_allclose(np.asarray(features.last), sd_reference["context"], atol=1e-4, rtol=0)
    plain = tower.apply({"params": _subtree(pipeline.variables, TEXT_ROOT)}, ids, method=_text_features)
    assert np.max(np.abs(np.asarray(plain.last) - sd_reference["context"])) > 1e-3


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


def test_a_file_without_a_header_scales_by_one_as_diffusers_does(pipeline, tmp_path):
    tensors, _ = read_file(SD / lora.DIFFUSERS_WEIGHTS)
    write_file(tensors, tmp_path / lora.DIFFUSERS_WEIGHTS, {"format": "pt"})
    adapter = LoRA.load(pipeline.model, pipeline.variables, tmp_path, layouts=pipeline.layouts)
    assert all(target.alpha == target.rank for target in adapter.targets.values())
    assert {target.rank for path, target in adapter.targets.items() if path[0] == "params"} == {4}
    assert {target.rank for path, target in adapter.targets.items() if path[0] == "encoders"} == {2}


# --------------------------------------------------------------------------
# An adapter on a model the registry built, from a run config
# --------------------------------------------------------------------------


REGISTRY_FIELDS = {"vocab_size": 256, "emb_features": 16, "num_layers": 2, "num_heads": 2,
                   "head_dim": 8, "mlp_features": 32, "max_seq_len": 16}


def _flat(tree):
    """Every leaf by its dotted path, for comparing two trees leaf by leaf."""
    return {".".join(str(entry.key) for entry in path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def _registry_decoder():
    """A tiny causal transformer and its variables, built from the registry:
    no published checkpoint, so no weight layouts to resolve names through."""
    from dew.config import ModelConfig

    config = ModelConfig("causal_transformer", {**REGISTRY_FIELDS,
                                              "dtype": "float32", "attention_impl": "reference"})
    model = config.build()
    return config, model, model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))


def test_a_registry_model_adapts_through_the_names_its_own_kernels_carry(tmp_path):
    """No loader in sight: the module paths under `params` are the names
    `target_modules` matches, the fresh adapter is the identity, and the
    factors write and read back in PEFT's layout under those names."""
    _, model, variables = _registry_decoder()
    adapter = LoRA(rank=2, modules=("q_proj", "layers_1.mlp.up_proj"), alpha=4.0).apply(
        model, variables, key=1)
    adapted = adapter.variables
    assert set(adapter.targets) == {("params", "layers_0", "self_attn", "q_proj"),
                                    ("params", "layers_1", "self_attn", "q_proj"),
                                    ("params", "layers_1", "mlp", "up_proj")}
    tokens = jnp.asarray([[3, 4, 5, 6, 7, 8]])
    np.testing.assert_array_equal(np.asarray(adapter.model.apply(adapted, tokens)),
                                  np.asarray(model.apply(variables, tokens)))

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

    with pytest.raises(ValueError, match="to_q match no projection"):
        LoRA(rank=2, modules=("to_q",), alpha=2.0).apply(model, variables, key=0)

    # Targets bound with no names say so instead of writing a file under
    # names they never resolved.
    unnamed = Adapter.bound(model, adapted, adapter.targets, rslora=False, dropout=0.0, layouts={})
    with pytest.raises(ValueError, match="is not a projection this adapter bound"):
        unnamed.save(adapted, tmp_path / "unnamed")


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
    with pytest.raises(ValueError, match="targets differ"):
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


def _wider(tree):
    """Every floating leaf of `tree` in float64."""
    return jax.tree.map(
        lambda leaf: leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, tree)


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
                                   variables=_wider(source.variables))
        truth = sampled(wide, _wider(prepared))
    assert_as_exact_as_the_reference(sampled(tuned, prepared), sampled(source, prepared), truth, family)
    with pytest.raises(ValueError, match="q_proj match no projection"):
        source.adapt(LoRA(rank=2, modules=("q_proj",)), key=0)
    with pytest.raises(ValueError, match="already carries an adapter"):
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
    data = Dataset(train=lambda partition: iter([batch] * 3), val=None, records=rows, batch=rows)
    state = trainer.fit(data, steps=3, log_every=3)

    assert set(_flat(state.variables["params"])) == {
        f"{'.'.join(path[1:])}.{factor}" for path in tuned.adapter.targets for factor in lora.FACTORS}
    for collection in (FROZEN, "encoders", "autoencoder"):
        for before, after in zip(jax.tree.leaves(initial.variables[collection]),
                                 jax.tree.leaves(state.variables[collection]), strict=True):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after), err_msg=collection)
    assert all(bool(jnp.any(leaf)) for name, leaf in _flat(state.variables["params"]).items()
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
    for name, leaf in _flat(thaw(loaded.variables)).items():
        if "lora_" in name:
            np.testing.assert_array_equal(np.asarray(leaf), np.asarray(_flat(thaw(variables))[name]),
                                          err_msg=name)

    tuned.save(tmp_path / "merged", variables=variables)
    reloaded = Pretrained.load(tmp_path / "merged", dtype="float32", attention_impl="xla")
    merged = dataclasses.replace(source, variables=tuned.adapter.merge(variables))
    np.testing.assert_array_equal(_sampled(reloaded.text_to_image()), _sampled(merged.text_to_image()))
    assert np.abs(_sampled(merged.text_to_image()) - _sampled(source.text_to_image())).max() > 0


EXPORTS = ROOT / "tests" / "fixtures" / "lora_exports"


def _recorded(name: str):
    with np.load(EXPORTS / f"{name}.npz") as data:
        arrays = {key: data[key] for key in data}
    return arrays, json.loads(arrays.pop("meta").tobytes())


def _assert_written_as_read(name: str, directory: Path, meta) -> None:
    from tools.lora_export_reference import digests

    assert digests(directory) == meta["digests"], (
        "Dew's adapter changed: rerun both halves of tools/lora_export_reference.py, "
        "`consume` in its PEFT environment, so PEFT and Diffusers read the new files")


@pytest.mark.parametrize("name", ["llama", "llama-rslora", "llama-patterns"])
def test_peft_reads_the_adapter_dew_writes_at_dews_logits(name, decoder, tmp_path):
    """tools/lora_export_reference.py records PEFT 0.20.0 reading the
    adapter a case writes (a fresh one at a rank and alpha, one with
    rsLoRA, and the committed one whose rank and alpha patterns Dew writes
    back, each with its factors moved) onto transformers' llama-tiny: every
    tensor of the file lands, and the adapted and `merge_and_unload` logits
    are recorded in float32 and float64. The adapter written here is the
    one that was read, and Dew's adapted and merged logits hold the float64
    rule against PEFT's."""
    from tools.lora_export_reference import CASES, export_case

    adapter, variables = export_case(CASES[name], decoder, tmp_path)
    arrays, meta = _recorded(name)
    _assert_written_as_read(name, tmp_path, meta)
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
    adapter, variables = export_case(CASES[family], source, tmp_path)
    arrays, meta = _recorded(family)
    _assert_written_as_read(family, tmp_path, meta)
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
def test_an_explicit_none_clears_what_a_pipeline_supplies(pipelines):
    """`DiffusionObjective(pipe, autoencoder=None)` trains in pixel space and
    `variables=None` draws the denoiser, as they do without a pipeline; only
    an omitted keyword takes the pipeline's."""
    pipe = pipelines["flux"]
    cleared = DiffusionObjective(pipe, autoencoder=None, variables=None)
    assert cleared.autoencoder is None and cleared.variables is None
    kept = DiffusionObjective(pipe)
    assert kept.autoencoder is pipe.autoencoder and kept.variables is pipe.variables


def test_a_loss_head_trains_beside_an_adapted_pipelines_factors(pipelines):
    """A head the objective adds to what trains, EDM2's uncertainty weighting,
    lands under `params` beside the factors, and the base stays frozen."""
    from dew.objectives.diffusion.objective import UNCERTAINTY

    tuned = pipelines["sd3"].adapt(LoRA(rank=2, modules=("to_q",)), key=0)
    state = DiffusionObjective(tuned, uncertainty=8).init(jax.random.key(0))
    assert UNCERTAINTY in state["params"]
    assert all(_is_factor(path) for path, _ in jax.tree_util.tree_leaves_with_path(state["params"])
               if path[0].key != UNCERTAINTY)
    assert not any(_is_factor(path) for path, _ in jax.tree_util.tree_leaves_with_path(state[FROZEN]))


def _recipe_run(tmp_path, image_size, *args):
    """A diffusion recipe run from argv as a user types it: two steps on the
    TFDS image fixture, with `args` ending in the run's LoRA flags
    (`--lora.rank`, `--lora.modules`) and no subcommand. Returns the parsed
    config and the trained state."""
    from dew.objectives.diffusion import DiffusionRunConfig

    config = DiffusionRunConfig.cli([
        "--model.dtype", "float32", "--model.attention-impl", "xla",
        "data:tfds-images", "--data.path", str(IMAGES), "--data.image-size", str(image_size),
        "--data.augmentation", "none", "--data.val-batches", "None", "--objective.solver",
        '{"class": "euler"}', "--objective.guidance", "None", "--objective.steps", "2",
        "--objective.ema-decay", "None", "--val-metrics",
        "--trainer.checkpoint-dir", str(tmp_path), "--trainer.batch-size", str(jax.device_count()),
        "--trainer.steps", "2", "--trainer.eval-every", "None", "--trainer.checkpoint-every", "2",
        "--trainer.compilation-cache-dir", "None", "--trainer.multi-host", "False", "--trainer.name", "run",
        *args])
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
    moved = _flat(state.variables["params"])
    assert moved and all(name.endswith(lora.FACTORS) for name in moved)
    assert all(bool(jnp.any(leaf != _flat(tuned.variables["params"])[name]))
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
    from dew.objectives.base import Objective, Ratio

    tokens = np.asarray(reference["input_ids"])

    class Squared(Objective):
        """The mean squared logit, over whatever tree it is handed."""

        inputs = InputSpec(Field("text", tokens.shape[1:]))

        def __init__(self, model, variables):
            self.model, self.variables = model, variables

        def init(self, key, variables=None):
            return self.variables if variables is None else variables

        def loss(self, variables, batch, step):
            logits = self.model.apply(variables, batch["text"])
            return Ratio(jnp.sum(logits.astype(jnp.float32) ** 2), jnp.asarray(logits.size, jnp.float32))

    adapter = LoRA(rank=2, modules=("q_proj", "v_proj")).apply(decoder.model, decoder.variables, key=0,
                                                               layouts=decoder.layouts)
    rows = jax.device_count()
    data = Dataset(train=lambda partition: iter([{"text": tokens[np.arange(rows) % 2]}] * 2), val=None,
                   records=rows, batch=rows)
    trainer = Trainer(Squared(adapter.model, adapter.variables), optax.sgd(0.1), key=0)
    initial = trainer.initial_state()
    state = trainer.fit(data, steps=2, log_every=2)

    assert set(_flat(state.variables["params"])) == {
        f"{'.'.join(target[1:])}.{factor}" for target in adapter.targets for factor in lora.FACTORS}
    before = _flat(initial.variables["params"])
    assert all(bool(jnp.any(leaf != before[name])) for name, leaf in _flat(state.variables["params"]).items()
               if name.endswith("lora_B"))
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

    moved = _flat(state.variables["params"])
    assert set(moved) == {f"output.final_proj.{factor}" for factor in lora.FACTORS}
    plain = dataclasses.replace(config, lora=None).build()
    key = jax.random.key(config.trainer.key)
    drawn = plain.model_variables(plain.init(key))
    for before, after in zip(jax.tree.leaves(drawn["params"]), jax.tree.leaves(state.variables[FROZEN]),
                             strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    adapter = config.lora.apply(plain.model, drawn, key=jax.random.fold_in(key, 1))
    assert all(bool(jnp.any(leaf != _flat(adapter.variables["params"])[name])) for name, leaf in moved.items()
               if name.endswith("lora_B"))
    adapter.save(state.variables, tmp_path / "in-process")
    rebuilt = Adapter.from_run(tmp_path / "run")
    rebuilt.save(rebuilt.variables, tmp_path / "from-run")
    _same_files(tmp_path / "from-run", tmp_path / "in-process")


def test_an_nnx_model_behind_flaxs_bridge_is_refused_rather_than_adapted_in_name_only():
    """An NNX `Linear` behind `flax.nnx.bridge.ToLinen` makes no `nn.Dense`
    call for the branch to join, so its factors would train while the
    forward never read them: the forward stayed the base model's with a
    nonzero B, and every factor's gradient was zero."""
    from flax import nnx
    from flax.nnx import bridge

    class Projection(nnx.Module):
        def __init__(self, *, rngs):
            self.q_proj = nnx.Linear(8, 8, rngs=rngs)

        def __call__(self, x):
            return self.q_proj(x)

    model = bridge.ToLinen(Projection)
    variables = model.init(jax.random.key(0), jnp.ones((2, 8)))
    with pytest.raises(TypeError, match=r"Projection is an NNX model behind flax\.nnx\.bridge\.ToLinen"):
        LoRA(rank=2, modules=("q_proj",)).apply(model, variables, key=1)
