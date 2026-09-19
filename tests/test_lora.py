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
from dew.inputs.diffusion import _text_features
from dew.interop.pretrained import load_pretrained
from dew.interop.safetensors_io import read_file, write_file
from dew.nn.backbones.unet_condition import DenoisingCondition
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
    return lora.load(decoder, ADAPTER)


def _factors(source, adapter, tree, directory) -> dict[str, np.ndarray]:
    """The adapter's leaves in `tree`, in PEFT's layout under PEFT's names."""
    lora.save(source, adapter, tree, directory)
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
    exported = _factors(decoder, adapter, merge(variables, {"params": gradient}), tmp_path)
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
    data = Dataset(train=lambda: iter([{"text": tokens[np.arange(rows) % 2]}]), val=None, records=rows, batch=rows)
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
    exported = _factors(decoder, adapter, trained, tmp_path / "adapter")
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
    lora.save(decoder, adapter, variables, tmp_path)
    ours = load_file(tmp_path / lora.PEFT_WEIGHTS)
    theirs = load_file(ADAPTER / lora.PEFT_WEIGHTS)
    assert ours.keys() == theirs.keys()
    for key in theirs:
        np.testing.assert_array_equal(ours[key], theirs[key])
    config = json.loads((tmp_path / lora.PEFT_CONFIG).read_text())
    assert (config["r"], config["lora_alpha"], config["use_rslora"], config["lora_dropout"]) == (4, 8, False, 0.1)
    assert config["rank_pattern"] == {"model.layers.1.self_attn.v_proj": 2}
    assert config["alpha_pattern"] == {"model.layers.0.mlp.down_proj": 3, "model.layers.1.mlp.down_proj": 3}
    again, variables_again = lora.load(decoder, tmp_path)
    assert again == adapter
    for ours_leaf, theirs_leaf in zip(jax.tree.leaves(variables_again), jax.tree.leaves(variables), strict=True):
        np.testing.assert_array_equal(np.asarray(ours_leaf), np.asarray(theirs_leaf))


def test_a_fresh_adapter_is_the_identity_and_matches_by_suffix(decoder, reference):
    """PEFT's target_modules: a suffix names every projection under it; B
    starts at zero so the adapted forward is the base forward; A is drawn
    on +-1/sqrt(fan_in)."""
    adapter, variables = lora.fresh(decoder, rank=3, alpha=6.0, modules=("q_proj", "layers.1.mlp.up_proj"),
                                    key=jax.random.key(0))
    assert set(adapter.targets) == {("params", "layers_0", "self_attn", "q_proj"),
                                    ("params", "layers_1", "self_attn", "q_proj"),
                                    ("params", "layers_1", "mlp", "up_proj")}
    tokens = jnp.asarray(reference["input_ids"])
    np.testing.assert_array_equal(np.asarray(adapter.adapt(decoder.model).apply(variables, tokens)),
                                  np.asarray(decoder.model.apply(decoder.variables, tokens)))
    a = variables["params"]["layers_1"]["mlp"]["up_proj"]["lora_A"]
    assert a.shape == (64, 3) and 0 < float(jnp.abs(a).max()) <= 1 / 8
    assert not bool(jnp.any(variables["params"]["layers_1"]["mlp"]["up_proj"]["lora_B"]))
    with pytest.raises(ValueError, match="w_proj"):
        lora.fresh(decoder, rank=2, alpha=2.0, modules=("q_proj", "w_proj"), key=jax.random.key(0))


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
        lora.load(decoder, _write_peft(tmp_path / "unbound", {
            lora.PEFT_PREFIX + "model.layers.5.self_attn.q_proj.lora_A.weight": a,
            lora.PEFT_PREFIX + "model.layers.5.self_attn.q_proj.lora_B.weight": b}))
    with pytest.raises(ValueError, match="stores rank 2 but its config declares 4"):
        lora.load(decoder, _write_peft(tmp_path / "rank", {
            prefix + ".lora_A.weight": a[:2], prefix + ".lora_B.weight": b[:, :2]}))
    with pytest.raises(ValueError, match=r"delta of \(64, 32\) on a weight the source stores as \(64, 64\)"):
        lora.load(decoder, _write_peft(tmp_path / "shape", {
            prefix + ".lora_A.weight": a[:, :32], prefix + ".lora_B.weight": b}))
    with pytest.raises(ValueError, match="not a projection weight"):
        lora.load(decoder, _write_peft(tmp_path / "norm", {
            lora.PEFT_PREFIX + "model.norm.lora_A.weight": a, lora.PEFT_PREFIX + "model.norm.lora_B.weight": b}))
    with pytest.raises(ValueError, match="use_dora"):
        lora.load(decoder, _write_peft(tmp_path / "dora", {
            prefix + ".lora_A.weight": a, prefix + ".lora_B.weight": b}, {"use_dora": True}))
    with pytest.raises(ValueError, match="kohya"):
        lora.load(decoder, _write_peft(tmp_path / "kohya", {"lora_unet_down_blocks_0.alpha": np.full((), 4, np.float32)}))
    with pytest.raises(ValueError, match="lora_A without its partner"):
        lora.load(decoder, _write_peft(tmp_path / "half", {prefix + ".lora_A.weight": a}))


def test_a_per_expert_source_tensor_takes_no_adapter():
    """A stacked expert leaf answers for every expert's tensor, so one
    expert's delta has no leaf of its own."""
    mixtral = load_pretrained(ROOT / "tests" / "fixtures" / "hf" / "mixtral-tiny", dtype="float32",
                              attention_impl="reference")
    with pytest.raises(ValueError, match="experts.0.w1 is assembled from several leaves"):
        lora.fresh(mixtral, rank=2, alpha=2.0, modules=("w1",), key=jax.random.key(0))


def test_a_target_that_is_not_a_dense_is_refused_when_called(decoder, reference):
    adapter = lora.LoRA({("params", "embed_tokens"): lora.Target(2, 2.0)})
    with pytest.raises(TypeError, match="embed_tokens is a Embed"):
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
    adapter = lora.LoRA({("params", "proj"): lora.Target(rank, alpha)})
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
    adapter, variables = lora.load(pipeline, SD)
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
    adapter, variables = lora.load(pipeline, SD)
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
    adapter, variables = lora.load(pipeline, SD)
    lora.save(pipeline, adapter, variables, tmp_path)
    ours, metadata = read_file(tmp_path / lora.DIFFUSERS_WEIGHTS)
    theirs, _ = read_file(SD / lora.DIFFUSERS_WEIGHTS)
    assert ours.keys() == theirs.keys()
    for key in theirs:
        np.testing.assert_array_equal(ours[key], theirs[key])
    header = json.loads(metadata[lora.DIFFUSERS_METADATA])
    assert (header["unet.r"], header["unet.lora_alpha"], header["text_encoder.r"], header["text_encoder.lora_alpha"]) == (4, 6, 2, 5)
    assert lora.load(pipeline, tmp_path)[0] == adapter


def test_a_file_without_a_header_scales_by_one_as_diffusers_does(pipeline, tmp_path):
    tensors, _ = read_file(SD / lora.DIFFUSERS_WEIGHTS)
    write_file(tensors, tmp_path / lora.DIFFUSERS_WEIGHTS, {"format": "pt"})
    adapter, _ = lora.load(pipeline, tmp_path)
    assert all(target.alpha == target.rank for target in adapter.targets.values())
    assert {target.rank for path, target in adapter.targets.items() if path[0] == "params"} == {4}
    assert {target.rank for path, target in adapter.targets.items() if path[0] == "encoders"} == {2}
