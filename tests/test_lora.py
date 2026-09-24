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
  pipeline, whose UNet approximates the GELU; the torch reference computes
  it exactly, so the UNet runs with `approximate_gelu=False` here.
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
from safetensors.numpy import load_file, save_file

from dew import lora
from dew.data import Dataset
from dew.diffusion.process import DenoisingCondition
from dew.inputs.diffusion import _text_features
from dew.interop.pretrained import load_pretrained
from dew.interop.safetensors_io import read_file, write_file
from dew.lora import LoRA
from dew.objectives.base import FROZEN, Step, freeze, merge, thaw
from dew.objectives.lm import LMObjective
from dew.training import Layout, MeshSpec, Trainer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "lora"
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"
ADAPTER = FIXTURES / "llama-tiny" / "adapter"


@pytest.fixture(scope="module")
def decoder():
    return load_pretrained(LLAMA, dtype="float32", attention_impl="reference")


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURES / "llama-tiny" / "reference.npz") as data:
        return {key: data[key] for key in data}


@pytest.fixture(scope="module")
def loaded(decoder):
    return LoRA.load(decoder.model, decoder.variables, decoder.layouts, ADAPTER)


def _factors(adapter, tree, directory) -> dict[str, np.ndarray]:
    """The adapter's leaves in `tree`, in PEFT's layout under PEFT's names."""
    adapter.save(tree, directory)
    return {key.removeprefix(lora.PEFT_PREFIX): value for key, value in load_file(directory / lora.PEFT_WEIGHTS).items()}


def _source_tensor(source, tree, name: str) -> np.ndarray:
    return next(layout for layout in source.weight_layouts if layout.name == name).export(tree)


def test_the_config_patterns_decide_each_target(loaded):
    """The PEFT config's r and lora_alpha apply except where a rank_pattern
    or alpha_pattern names the module the way PEFT's get_pattern_key does."""
    adapter, variables = loaded
    ranks = {"/".join(path[1:]): target.rank for path, target in adapter.targets.items()}
    alphas = {"/".join(path[1:]): target.alpha for path, target in adapter.targets.items()}
    assert ranks == {"layers_0/self_attn/q_proj": 4, "layers_0/self_attn/v_proj": 4, "layers_0/mlp/down_proj": 4,
                     "layers_1/self_attn/q_proj": 4, "layers_1/self_attn/v_proj": 2, "layers_1/mlp/down_proj": 4}
    assert alphas == {"layers_0/self_attn/q_proj": 8, "layers_0/self_attn/v_proj": 8, "layers_0/mlp/down_proj": 3,
                      "layers_1/self_attn/q_proj": 8, "layers_1/self_attn/v_proj": 8, "layers_1/mlp/down_proj": 3}
    assert adapter.dropout == 0.1 and not adapter.rslora
    assert variables["params"]["layers_1"]["self_attn"]["v_proj"]["lora_A"].shape == (64, 2)
    assert variables["params"]["layers_1"]["self_attn"]["v_proj"]["lora_B"].shape == (2, 32)


def test_the_unmerged_forward_matches_peft(decoder, loaded, reference):
    adapter, variables = loaded
    logits = adapter.adapt(decoder.model).apply(variables, jnp.asarray(reference["input_ids"]))
    np.testing.assert_allclose(np.asarray(logits), reference["adapted_logits"], atol=1e-4, rtol=0)
    assert np.max(np.abs(reference["adapted_logits"] - reference["base_logits"])) > 1


def test_eval_prediction_disables_adapter_dropout(decoder, loaded, reference):
    adapter, variables = loaded
    ids = jnp.asarray(reference["input_ids"])
    objective = LMObjective(adapter.adapt(decoder.model), ids.shape[1] - 1, ema_decay=None)
    predictions = {}
    for train in (False, True):
        predictions[train] = [
            objective.predict(variables, {"text": ids},
                              Step(step=jnp.int32(0), key=jax.random.key(seed), ema=None),
                              train=train)[2].logits
            for seed in (0, 1)
        ]
    np.testing.assert_array_equal(*predictions[False])
    np.testing.assert_allclose(predictions[False][0], reference["adapted_logits"][:, :-1],
                               atol=1e-4, rtol=0)
    assert not np.array_equal(*predictions[True])


def test_adapting_an_adapted_model_is_refused(decoder, loaded):
    adapter, _ = loaded
    adapted = adapter.adapt(decoder.model)
    with pytest.raises(ValueError, match="already adapted"):
        adapter.adapt(adapted)


def test_the_merge_matches_peft_weights_and_logits(decoder, loaded, reference):
    """W + scale * B A into every kernel, the factors gone, the plain model
    over the merged tree agreeing with merge_and_unload."""
    adapter, variables = loaded
    merged = adapter.merge(variables)
    assert not any(adapter.trainable(tuple(entry.key for entry in path))
                   for path, _ in jax.tree_util.tree_leaves_with_path(merged))
    for key in reference:
        if key.startswith("merged/"):
            np.testing.assert_allclose(_source_tensor(decoder, merged, key.removeprefix("merged/")),
                                       reference[key], atol=1e-6, rtol=0)
    logits = decoder.model.apply(merged, jnp.asarray(reference["input_ids"]))
    np.testing.assert_allclose(np.asarray(logits), reference["merged_logits"], atol=1e-4, rtol=0)


def test_adapter_gradients_of_the_token_loss_match_peft(decoder, loaded, reference, tmp_path):
    """The mean next-token cross entropy through the LM objective over the
    frozen split, differentiated with respect to the adapter alone."""
    adapter, variables = loaded
    # The reference stepped in eval mode; the objective's forward is a
    # training one, so the branch's dropout is turned off to compare.
    adapter = dataclasses.replace(adapter, dropout=0.0)
    tokens = jnp.asarray(reference["input_ids"])
    objective = LMObjective(adapter.adapt(decoder.model), tokens.shape[1] - 1, pretrained=variables,
                            ema_decay=None, trainable=adapter.trainable)
    params = objective.init(jax.random.key(0))
    assert sorted(params) == [FROZEN, "params"]
    assert all(adapter.trainable(("params", *(entry.key for entry in path)))
               for path, _ in jax.tree_util.tree_leaves_with_path(params["params"]))

    def loss(moving):
        stats, _ = objective.loss({**params, "params": moving}, {"text": tokens},
                                  Step(step=jnp.int32(0), key=jax.random.key(1), ema=None))
        return objective.reduce_loss(stats)[0]

    value, gradient = jax.jit(jax.value_and_grad(loss))(params["params"])
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    exported = _factors(adapter, merge(variables, {"params": gradient}), tmp_path)
    for key in reference:
        if key.startswith("grad/"):
            np.testing.assert_allclose(exported[key.removeprefix("grad/")], reference[key], atol=1e-4, rtol=0)


def test_one_trainer_step_moves_the_adapter_and_nothing_else(decoder, loaded, reference, tmp_path):
    """A real SGD step through the Trainer: the frozen collection comes back
    bitwise, every factor moves, and the logits and factors agree with the
    reference's step on the adapter parameters."""
    adapter, variables = loaded
    adapter = dataclasses.replace(adapter, dropout=0.0)
    meta = json.loads((FIXTURES / "llama-tiny" / "meta.json").read_text())
    tokens = np.asarray(reference["input_ids"])
    rows = 2 * jax.device_count()
    objective = LMObjective(adapter.adapt(decoder.model), tokens.shape[1] - 1, pretrained=variables,
                            ema_decay=None, trainable=adapter.trainable)
    data = Dataset(train=lambda partition: iter([{"text": tokens[np.arange(rows) % 2]}]), val=None, records=rows, batch=rows)
    trainer = Trainer(objective, optax.sgd(meta["learning_rate"]), key=jax.random.key(3),
                      mesh=MeshSpec(), layout=Layout(min_shard=2**30))
    initial = trainer.initial_state()

    state = trainer.fit(data, steps=1, log_every=1)

    for before, after in zip(jax.tree.leaves(initial.params[FROZEN]), jax.tree.leaves(state.params[FROZEN]), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    assert all(bool(jnp.any(before != after)) for before, after in
               zip(jax.tree.leaves(initial.params["params"]), jax.tree.leaves(state.params["params"]), strict=True))
    trained = thaw(state.params)
    logits = adapter.adapt(decoder.model).apply(trained, jnp.asarray(tokens))
    np.testing.assert_allclose(np.asarray(logits), reference["updated_logits"], atol=1e-4, rtol=0)
    exported = _factors(adapter, trained, tmp_path / "adapter")
    for key in reference:
        if key.startswith("updated/"):
            np.testing.assert_allclose(exported[key.removeprefix("updated/")], reference[key], atol=1e-4, rtol=0)
    # The merged full export reloads as a plain source at the stepped logits.
    decoder.save(tmp_path / "merged", variables=adapter.merge(trained))
    reloaded = load_pretrained(tmp_path / "merged", dtype="float32", attention_impl="reference")
    np.testing.assert_allclose(np.asarray(reloaded.model.apply(reloaded.variables, jnp.asarray(tokens))),
                               reference["updated_logits"], atol=1e-4, rtol=0)


def test_export_writes_the_peft_file_back(decoder, loaded, tmp_path):
    """The factors land bitwise where PEFT wrote them, under its names, and
    the config resolves every module to the same rank and alpha on reload."""
    adapter, variables = loaded
    adapter.save(variables, tmp_path)
    ours = load_file(tmp_path / lora.PEFT_WEIGHTS)
    theirs = load_file(ADAPTER / lora.PEFT_WEIGHTS)
    assert ours.keys() == theirs.keys()
    for key in theirs:
        np.testing.assert_array_equal(ours[key], theirs[key])
    config = json.loads((tmp_path / lora.PEFT_CONFIG).read_text())
    assert (config["r"], config["lora_alpha"], config["use_rslora"], config["lora_dropout"]) == (4, 8, False, 0.1)
    assert config["rank_pattern"] == {"model.layers.1.self_attn.v_proj": 2}
    assert config["alpha_pattern"] == {"model.layers.0.mlp.down_proj": 3, "model.layers.1.mlp.down_proj": 3}
    again, variables_again = LoRA.load(decoder.model, decoder.variables, decoder.layouts, tmp_path)
    assert again == adapter
    for ours_leaf, theirs_leaf in zip(jax.tree.leaves(variables_again), jax.tree.leaves(variables), strict=True):
        np.testing.assert_array_equal(np.asarray(ours_leaf), np.asarray(theirs_leaf))


def test_a_run_record_holds_an_adapters_targets_not_its_source_bindings(loaded):
    """A loaded adapter carries the source projections it was bound over,
    which follow from the source the run record already names. The record
    carries the adapter itself, and a record without bindings rebuilds it,
    unbound, as a run config declares it."""
    from dew.config import RunConfig

    adapter, _ = loaded
    assert adapter.layouts
    record = json.loads(json.dumps(RunConfig(lora=adapter).to_dict()))
    assert "layouts" not in record["lora"]
    rebuilt = RunConfig.from_dict(record).lora
    assert rebuilt == adapter and not rebuilt.layouts


def test_a_fresh_adapter_is_the_identity_and_matches_by_suffix(decoder, reference):
    """PEFT's target_modules: a suffix names every projection under it; B
    starts at zero so the adapted forward is the base forward; A is drawn
    on +-1/sqrt(fan_in). An unset alpha is PEFT's own default, twice the rank."""
    adapter, variables = LoRA.fresh(decoder.model, decoder.variables, decoder.layouts, rank=3,
                                    modules=("q_proj", "layers.1.mlp.up_proj"), key=jax.random.key(0))
    assert set(adapter.targets) == {("params", "layers_0", "self_attn", "q_proj"),
                                    ("params", "layers_1", "self_attn", "q_proj"),
                                    ("params", "layers_1", "mlp", "up_proj")}
    assert {target.alpha for target in adapter.targets.values()} == {6.0}
    tokens = jnp.asarray(reference["input_ids"])
    np.testing.assert_array_equal(np.asarray(adapter.adapt(decoder.model).apply(variables, tokens)),
                                  np.asarray(decoder.model.apply(decoder.variables, tokens)))
    a = variables["params"]["layers_1"]["mlp"]["up_proj"]["lora_A"]
    assert a.shape == (64, 3) and 0 < float(jnp.abs(a).max()) <= 1 / 8
    assert not bool(jnp.any(variables["params"]["layers_1"]["mlp"]["up_proj"]["lora_B"]))
    with pytest.raises(ValueError, match="w_proj"):
        LoRA.fresh(decoder.model, decoder.variables, decoder.layouts, rank=2, alpha=2.0,
                   modules=("q_proj", "w_proj"), key=jax.random.key(0))


def test_the_branch_drops_out_only_under_a_dropout_stream(decoder, loaded, reference):
    adapter, variables = loaded
    tokens = jnp.asarray(reference["input_ids"])
    model = adapter.adapt(decoder.model)
    plain = model.apply(variables, tokens)
    dropped = model.apply(variables, tokens, rngs={"dropout": jax.random.key(1)})
    assert np.max(np.abs(np.asarray(dropped) - np.asarray(plain))) > 1e-3
    kept = dataclasses.replace(adapter, dropout=0.0).adapt(decoder.model).apply(
        variables, tokens, rngs={"dropout": jax.random.key(1)})
    np.testing.assert_array_equal(np.asarray(kept), np.asarray(plain))


def test_rslora_scales_by_the_square_root_of_the_rank(decoder, loaded, reference):
    adapter, variables = loaded
    tokens = jnp.asarray(reference["input_ids"])
    scaled = dataclasses.replace(adapter, rslora=True)
    alpha_times_root = {path: dataclasses.replace(target, alpha=target.alpha * np.sqrt(target.rank))
                       for path, target in adapter.targets.items()}
    equivalent = dataclasses.replace(adapter, targets=alpha_times_root)
    np.testing.assert_allclose(np.asarray(scaled.adapt(decoder.model).apply(variables, tokens)),
                               np.asarray(equivalent.adapt(decoder.model).apply(variables, tokens)), atol=1e-5, rtol=0)


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

    with pytest.raises(ValueError, match="layers.5.self_attn.q_proj, which this source does not bind"):
        LoRA.load(decoder.model, decoder.variables, decoder.layouts, _write_peft(tmp_path / "unbound", {
            lora.PEFT_PREFIX + "model.layers.5.self_attn.q_proj.lora_A.weight": a,
            lora.PEFT_PREFIX + "model.layers.5.self_attn.q_proj.lora_B.weight": b}))
    with pytest.raises(ValueError, match="stores rank 2 but its config declares 4"):
        LoRA.load(decoder.model, decoder.variables, decoder.layouts, _write_peft(tmp_path / "rank", {
            prefix + ".lora_A.weight": a[:2], prefix + ".lora_B.weight": b[:, :2]}))
    with pytest.raises(ValueError, match=r"delta of \(64, 32\) on a weight the source stores as \(64, 64\)"):
        LoRA.load(decoder.model, decoder.variables, decoder.layouts, _write_peft(tmp_path / "shape", {
            prefix + ".lora_A.weight": a[:, :32], prefix + ".lora_B.weight": b}))
    with pytest.raises(ValueError, match="not a projection weight"):
        LoRA.load(decoder.model, decoder.variables, decoder.layouts, _write_peft(tmp_path / "norm", {
            lora.PEFT_PREFIX + "model.norm.lora_A.weight": a, lora.PEFT_PREFIX + "model.norm.lora_B.weight": b}))
    with pytest.raises(ValueError, match="use_dora"):
        LoRA.load(decoder.model, decoder.variables, decoder.layouts, _write_peft(tmp_path / "dora", {
            prefix + ".lora_A.weight": a, prefix + ".lora_B.weight": b}, {"use_dora": True}))
    with pytest.raises(ValueError, match="kohya"):
        LoRA.load(decoder.model, decoder.variables, decoder.layouts, _write_peft(tmp_path / "kohya", {"lora_unet_down_blocks_0.alpha": np.full((), 4, np.float32)}))
    with pytest.raises(ValueError, match="lora_A without its partner"):
        LoRA.load(decoder.model, decoder.variables, decoder.layouts, _write_peft(tmp_path / "half", {prefix + ".lora_A.weight": a}))


def test_a_per_expert_source_tensor_takes_no_adapter():
    """A stacked expert leaf answers for every expert's tensor, so one
    expert's delta has no leaf of its own."""
    mixtral = load_pretrained(ROOT / "tests" / "fixtures" / "hf" / "mixtral-tiny", dtype="float32",
                              attention_impl="reference")
    with pytest.raises(ValueError, match="experts.0.w1 is assembled from several leaves"):
        LoRA.fresh(mixtral.model, mixtral.variables, mixtral.layouts, rank=2, alpha=2.0,
                   modules=("w1",), key=jax.random.key(0))


def test_a_target_that_is_not_a_dense_is_refused_when_called(decoder, reference):
    adapter = LoRA({("params", "embed_tokens"): lora.Target(2, 2.0)})
    with pytest.raises(TypeError, match="params/embed_tokens.*targets nn.Dense and nn.DenseGeneral kernels"):
        adapter.adapt(decoder.model).apply(decoder.variables, jnp.asarray(reference["input_ids"]))


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
    adapter = LoRA({("params", "proj"): lora.Target(rank, alpha)})
    keys = iter(jax.random.split(jax.random.key(seed), 2))
    shape = tree["params"]["proj"]["kernel"].shape
    factors = {"lora_A": jnp.asarray(jax.random.normal(next(keys), shape[:contracted] + (rank,))),
               "lora_B": jnp.asarray(jax.random.normal(next(keys), (rank,) + shape[contracted:]))}
    adapted_tree = merge(tree, {"params": {"proj": factors}})
    return adapter, adapter.adapt(model), adapted_tree, adapter.merge(adapted_tree)


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
    source = load_pretrained(destination / "sd", dtype="float32", attention_impl="reference")
    # The torch reference computes the GELU the Flax class approximates.
    return dataclasses.replace(source, model=dataclasses.replace(source.model, approximate_gelu=False))


@pytest.fixture(scope="module")
def sd_reference():
    with np.load(SD / "reference.npz") as data:
        return {key: data[key] for key in data}


def _subtree(tree, path):
    for part in path:
        tree = tree[part]
    return tree


def test_the_text_encoder_component_adapts_the_conditioning_tower(pipeline, sd_reference):
    adapter, variables = LoRA.load(pipeline.model, pipeline.variables, pipeline.layouts, SD)
    tower = pipeline.inputs.conditions["conditioning"].encoder.towers[0]
    ids = jnp.asarray(sd_reference["prompt_ids"])
    features = adapter.adapt(tower, root=TEXT_ROOT).apply({"params": _subtree(variables, TEXT_ROOT)}, ids,
                                                          method=_text_features)
    np.testing.assert_allclose(np.asarray(features.last), sd_reference["context"], atol=1e-4, rtol=0)
    plain = tower.apply({"params": _subtree(pipeline.variables, TEXT_ROOT)}, ids, method=_text_features)
    assert np.max(np.abs(np.asarray(plain.last) - sd_reference["context"])) > 1e-3


def test_the_unet_component_matches_the_pipeline_adapted_and_fused(pipeline, sd_reference):
    """to_q's [in, heads, depth] and to_out.0's [heads, depth, features]
    DenseGeneral kernels take their factors in kernel layout and merge back
    to the fused torch weights."""
    adapter, variables = LoRA.load(pipeline.model, pipeline.variables, pipeline.layouts, SD)
    condition = DenoisingCondition(jnp.asarray(sd_reference["context"]), None, None)
    latent, time = jnp.asarray(sd_reference["latent"]), jnp.asarray(sd_reference["time"])
    predicted = adapter.adapt(pipeline.model).apply({"params": variables["params"]}, latent, time, conditioning=condition)
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
    adapter, variables = LoRA.load(pipeline.model, pipeline.variables, pipeline.layouts, SD)
    adapter.save(variables, tmp_path)
    ours, metadata = read_file(tmp_path / lora.DIFFUSERS_WEIGHTS)
    theirs, _ = read_file(SD / lora.DIFFUSERS_WEIGHTS)
    assert ours.keys() == theirs.keys()
    for key in theirs:
        np.testing.assert_array_equal(ours[key], theirs[key])
    header = json.loads(metadata[lora.DIFFUSERS_METADATA])
    assert (header["unet.r"], header["unet.lora_alpha"], header["text_encoder.r"], header["text_encoder.lora_alpha"]) == (4, 6, 2, 5)
    assert LoRA.load(pipeline.model, pipeline.variables, pipeline.layouts, tmp_path)[0] == adapter


def test_a_file_without_a_header_scales_by_one_as_diffusers_does(pipeline, tmp_path):
    tensors, _ = read_file(SD / lora.DIFFUSERS_WEIGHTS)
    write_file(tensors, tmp_path / lora.DIFFUSERS_WEIGHTS, {"format": "pt"})
    adapter, _ = LoRA.load(pipeline.model, pipeline.variables, pipeline.layouts, tmp_path)
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

    config = ModelConfig("causal_transformer", dict(REGISTRY_FIELDS), dtype="float32",
                         attention_impl="reference")
    model = config.build()
    return config, model, model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))


def test_a_registry_model_adapts_through_the_names_its_own_kernels_carry(tmp_path):
    """No loader in sight: the module paths under `params` are the names
    `target_modules` matches, the fresh adapter is the identity, and the
    factors write and read back in PEFT's layout under those names."""
    _, model, variables = _registry_decoder()
    adapter, adapted = LoRA.fresh(model, variables, {}, rank=2, alpha=4.0,
                                  modules=("q_proj", "layers_1.mlp.up_proj"), key=jax.random.key(1))
    assert set(adapter.targets) == {("params", "layers_0", "self_attn", "q_proj"),
                                    ("params", "layers_1", "self_attn", "q_proj"),
                                    ("params", "layers_1", "mlp", "up_proj")}
    tokens = jnp.asarray([[3, 4, 5, 6, 7, 8]])
    np.testing.assert_array_equal(np.asarray(adapter.adapt(model).apply(adapted, tokens)),
                                  np.asarray(model.apply(variables, tokens)))

    adapter.save(adapted, tmp_path)
    written = load_file(tmp_path / lora.PEFT_WEIGHTS)
    assert set(written) == {f"{lora.PEFT_PREFIX}{name}.lora_{factor}.weight"
                            for name in ("layers_0.self_attn.q_proj", "layers_1.self_attn.q_proj",
                                         "layers_1.mlp.up_proj") for factor in "AB"}
    assert written[f"{lora.PEFT_PREFIX}layers_1.mlp.up_proj.lora_A.weight"].shape == (2, 16)
    again, restored = LoRA.load(model, variables, {}, tmp_path)
    assert again == adapter
    for ours, theirs in zip(jax.tree.leaves(restored), jax.tree.leaves(adapted), strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))

    with pytest.raises(ValueError, match="to_q match no projection"):
        LoRA.fresh(model, variables, {}, rank=2, alpha=2.0, modules=("to_q",), key=jax.random.key(0))

    # A target set a person declares binds nothing, so it says so instead of
    # writing a file under names it never resolved.
    declared = LoRA(dict(adapter.targets))
    assert declared == adapter, "the bindings are the adapter's baggage, not its identity"
    with pytest.raises(ValueError, match="binds no source names"):
        declared.save(adapted, tmp_path / "declared")


def test_a_run_config_adapter_trains_its_factors_and_nothing_else(tmp_path):
    """`--lora` on a run: `RunConfig.train` adapts the module the objective
    trains, freezes every other leaf through the adapter's own filter, and
    two steps move the factors and leave the base weights bitwise."""
    from dew.config import RunConfig, TrainerConfig

    config_model, model, variables = _registry_decoder()
    adapter, _ = LoRA.fresh(model, variables, {}, rank=2, alpha=4.0, modules=("q_proj", "v_proj"),
                            key=jax.random.key(1))
    rows = 2 * jax.device_count()
    batch = {"text": np.random.RandomState(0).randint(1, 250, (rows, 9)).astype(np.int32)}
    data = Dataset(train=lambda partition: iter([batch, batch]), val=None, records=rows, batch=rows)
    config = RunConfig(model=config_model, lora=dataclasses.replace(adapter, dropout=0.0),
                       trainer=TrainerConfig(checkpoint_dir=str(tmp_path), batch_size=rows, steps=2,
                                             eval_every=None, checkpoint_every=None,
                                             compilation_cache_dir=None))
    objective = LMObjective(config_model.build(), 8, ema_decay=None)

    state = config.train(objective, data, name="run")

    selects = objective.trainable
    assert selects is not None
    assert selects(("params", "layers_0", "self_attn", "q_proj", "lora_A"))
    assert not selects(("params", "layers_0", "self_attn", "q_proj", "kernel"))
    initial = Trainer(objective, optax.sgd(0.0), key=jax.random.key(config.trainer.seed)).initial_state()
    moved = _flat(state.params["params"])
    assert set(moved) == {f"{'.'.join(target[1:])}.{factor}"
                          for target in config.lora.targets for factor in lora.FACTORS}
    for name, leaf in moved.items():
        assert bool(jnp.any(leaf != _flat(initial.params["params"])[name])), f"{name} did not move"
    for before, after in zip(jax.tree.leaves(initial.params[FROZEN]),
                             jax.tree.leaves(state.params[FROZEN]), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert record["lora"]["targets"]["params/layers_0/self_attn/q_proj"] == {"rank": 2, "alpha": 4.0}
    assert RunConfig.from_dict(record).lora == config.lora


def test_an_objective_that_selects_its_own_leaves_refuses_a_config_adapter():
    """One filter decides what trains: an objective that already has one, or
    one that keeps no model to adapt, is refused by name."""
    from dew.config import RunConfig, TrainerConfig

    _, model, variables = _registry_decoder()
    adapter, _ = LoRA.fresh(model, variables, {}, rank=2, alpha=4.0, modules=("q_proj",),
                            key=jax.random.key(1))
    rows = jax.device_count()
    data = Dataset(train=lambda partition: iter([]), val=None, records=rows, batch=rows)
    config = RunConfig(lora=adapter, trainer=TrainerConfig(batch_size=rows, steps=1))

    held = LMObjective(model, 8, ema_decay=None, trainable=lambda path: True)
    with pytest.raises(ValueError, match="LMObjective already selects what trains"):
        config.train(held, data, name="run")

    class Modelless:
        """Something that trains other than one module: there is nothing to adapt."""

    with pytest.raises(ValueError, match="Modelless keeps no `model`"):
        config.train(Modelless(), data, name="run")


def test_the_branch_reaches_every_layer_of_a_scanned_run(decoder, reference):
    """A scanned stack runs its like layers as one module, `layers_0_1`; the
    adapter's targets are named per layer, so the branch resolves the run to
    its layers and every layer's factors carry gradient. Before this, a
    stack cloned to `scan_layers=True` after adapting silently trained only
    the layers that ran alone."""
    adapter, variables = LoRA.fresh(decoder.model, decoder.variables, decoder.layouts, rank=2, alpha=4.0,
                                    modules=("q_proj",), key=jax.random.key(1))
    scanned = adapter.adapt(decoder.model.clone(scan_layers=True))
    tokens = jnp.asarray(reference["input_ids"])
    plain = adapter.adapt(decoder.model).apply(variables, tokens)
    np.testing.assert_allclose(np.asarray(scanned.apply(variables, tokens)), np.asarray(plain), rtol=1e-5, atol=1e-5)

    def loss(params):
        return jnp.mean(scanned.apply({**variables, "params": params}, tokens) ** 2)

    grads = jax.grad(loss)(variables["params"])
    for layer in ("layers_0", "layers_1"):
        factors = grads[layer]["self_attn"]["q_proj"]
        assert float(jnp.abs(factors["lora_B"]).max()) > 0, layer
    assert adapter.target_at(("params", "layers_0_1", "self_attn", "q_proj")) == adapter.targets[("params", "layers_0", "self_attn", "q_proj")]
    assert adapter.target_at(("params", "layers_0_1", "mlp", "up_proj")) is None
    uneven = LoRA({**adapter.targets, ("params", "layers_1", "self_attn", "q_proj"): lora.Target(3, 4.0)})
    with pytest.raises(ValueError, match="targets differ"):
        uneven.target_at(("params", "layers_0_1", "self_attn", "q_proj"))
