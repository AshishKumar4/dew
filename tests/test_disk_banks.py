"""Disk-backed MoE banks read at execution, not once at trace or load time."""

import json
import re
import shutil
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, widened

from dew.inference.banks import HeldBanks
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.interop.safetensors_io import read_weights, save_sharded
from dew.registry import models
from dew.sampling.text import Sampling, generate
from dew.training import Layout, MeshSpec

import_module("dew.nn.backbones")  # registers the fixture kind


FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "mixtral-tiny"
DEVICE = Layout(min_shard=1, tolerance=1.0)


def single_mesh():
    return MeshSpec().build([jax.devices()[0]])


def decoder(config, bank_layers, dtype="float32"):
    record = {**translate_config(config), "dtype": dtype, "attention_impl": "reference"}
    return models.build("causal_transformer", {**record, "scan_layers": True,
                                               "bank_layers": bank_layers})


BANKED_SEEDS = 8


@pytest.mark.parametrize("bank_layers", [None, 1])
@pytest.mark.parametrize("read_ahead", [True, False])
def test_disk_moe_logits_and_cached_decode_equal_resident(bank_layers, read_ahead):
    """A decoder streamed from disk scores what the resident one does: bit for bit where a bank holds
    each layer, so the two run one scan; as exact against float64 (check 1) where the banks hold the
    stack whole, whose effectful one-trip scan XLA:CUDA cannot peel as it does the resident two-layer
    one (ledger 2540). Loading and tracing read no weight, and a compiled program rereads each one."""
    from dew.inference.banks import SafetensorsBanks

    with SafetensorsBanks(FIXTURE, cache_bytes=0, param_dtype="float32", read_ahead=read_ahead) as source:
        model = decoder(source.config, bank_layers)
        weights = translate_weights(read_weights(FIXTURE), translate_config(source.config), "mixtral")
        resident = HeldBanks(weights).place(model, mesh=single_mesh(), layout=DEVICE)
        streamed = source.stream(model, mesh=single_mesh(), layout=DEVICE)
        assert source.stats().misses == 0  # No decoder weight is read by loading or tracing.
        tokens = jax.device_put(np.load(FIXTURE / "input_ids.npy"), jax.devices()[0])
        score = jax.jit(model.apply)
        score.lower(streamed, tokens)
        assert source.stats().misses == 0
        actual = score(streamed, tokens)
        np.testing.assert_allclose(actual, np.load(FIXTURE / "logits.npy"), atol=1e-4, rtol=1e-4)
        before = source.stats().misses
        if bank_layers == 1:
            np.testing.assert_array_equal(actual, score(resident, tokens))
            np.testing.assert_array_equal(score(streamed, tokens[:, ::-1]), score(resident, tokens[:, ::-1]))
        else:
            vocab = translate_config(source.config)["vocab_size"]
            with jax.enable_x64(new_val=True):
                wide = decoder(source.config, None, "float64")
                wide_variables = widened(jax.device_get(resident))

            def truth(drawn):
                with jax.enable_x64(new_val=True):
                    return wide.apply(wide_variables, jnp.asarray(drawn))

            def draw(seed):
                return (np.random.default_rng([2540, seed]).integers(0, vocab, tokens.shape, np.int32),)

            assert_as_exact(lambda drawn: model.apply(streamed, drawn),
                            lambda drawn: model.apply(resident, drawn), draw,
                            "a disk-banked MoE decoder's logits", seeds=BANKED_SEEDS, truth_fn=truth)
        assert source.stats().misses > before  # A compiled executable rereads uncached rows.

        def drawn(variables):
            return generate(model, variables, tokens[:1, :2], key=jax.random.key(1),
                            max_new_tokens=3, sampling=Sampling(temperature=0.0))

        np.testing.assert_array_equal(drawn(streamed).tokens, drawn(resident).tokens)
        assert source.stats().cache_bytes == 0


def experts_fused(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`tensors` with each layer's per-expert projections as transformers 5 stores them fused, the
    layout a packed family splits: `experts.gate_up_proj` [E, 2I, H] and `experts.down_proj` [E, H, I]."""
    experts: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    held = {}
    for name, tensor in tensors.items():
        found = re.fullmatch(r"(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name)
        if found is None:
            held[name] = tensor
        else:
            experts.setdefault((found[1], found[3]), {})[int(found[2])] = np.asarray(tensor)
    for stem in {stem for stem, _ in experts}:
        def stacked(projection, stem=stem):
            return np.stack([experts[stem, projection][index] for index in sorted(experts[stem, projection])])
        held[f"{stem}.gate_up_proj"] = np.concatenate([stacked("gate_proj"), stacked("up_proj")], axis=1)
        held[f"{stem}.down_proj"] = stacked("down_proj")
    return held


@pytest.mark.parametrize("fused_experts", [False, True], ids=["per-expert", "fused-experts"])
def test_disk_source_reads_shards_and_bounds_cache(fused_experts, tmp_path):
    """A row read from a sharded checkpoint is the resident translation's, under a cache bound: Mixtral's
    per-expert tensors, and Qwen2-MoE's fused experts (`experts_fused`), which a lazy read splits into
    views and reads only when the row is placed."""
    from dew.inference.banks import SafetensorsBanks

    fixture = FIXTURE.parent / "qwen2-moe-tiny" if fused_experts else FIXTURE
    tensors = experts_fused(read_weights(fixture)) if fused_experts else read_weights(fixture)
    (tmp_path / "config.json").write_bytes((fixture / "config.json").read_bytes())
    save_sharded(tensors, tmp_path, max_shard_size=12_000)
    with SafetensorsBanks(tmp_path, cache_bytes=1, param_dtype="float32") as source:
        row = source.read(0)
        expected = translate_weights(read_weights(fixture), translate_config(source.config),
                                     str(source.config["model_type"]))
        for actual, reference in zip(
            jax.tree.leaves(row),
            jax.tree.leaves(
                {name: tree["layers_0"] for name, tree in expected.items() if "layers_0" in tree}
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, reference)
            assert not actual.flags.writeable
        assert source.stats().cache_bytes <= 1
        assert source.stats().bytes_read > 0


@pytest.mark.parametrize("bank_layers", [None, 1])
def test_disk_source_also_builds_ordinary_resident_banks(bank_layers):
    from dew.inference.banks import SafetensorsBanks

    with SafetensorsBanks(FIXTURE, cache_bytes=0) as source:
        model = decoder(source.config, bank_layers)
        actual = source.place(model, mesh=single_mesh(), layout=DEVICE)
        expected = HeldBanks(translate_weights(
            read_weights(FIXTURE), translate_config(source.config), "mixtral"
        )).place(model, mesh=single_mesh(), layout=DEVICE)
        for leaf, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(leaf, reference)


def test_disk_rows_preserve_requested_physical_formats():
    from jax.experimental.layout import Format, Layout as DeviceLayout

    from dew.inference.banks import SafetensorsBanks, one_layer

    with SafetensorsBanks(FIXTURE, cache_bytes=0) as source:
        shapes = one_layer(source.shapes(), 0)
        placement = DEVICE.offloaded(single_mesh(), shapes)
        formats = jax.tree.map(lambda shape, target: Format(
            DeviceLayout(major_to_minor=tuple(reversed(range(len(shape.shape))))), target), shapes, placement)
        row = source.bank((0,), formats)
        expected = source.read(0)
        for actual, reference, target in zip(jax.tree.leaves(row), jax.tree.leaves(expected),
                                             jax.tree.leaves(formats), strict=True):
            np.testing.assert_array_equal(actual, reference)
            assert actual.format.layout.major_to_minor == target.layout.major_to_minor
            assert actual.format.sharding == target.sharding


def test_streaming_keeps_captured_layers_and_refuses_training_and_closed_sources():
    from dew.inference.banks import SafetensorsBanks

    with SafetensorsBanks(FIXTURE, cache_bytes=0) as source:
        model = decoder(source.config, None)
        variables = source.stream(model, mesh=single_mesh(), layout=DEVICE)
        tokens = jnp.ones((1, 2), jnp.int32)
        with pytest.raises(ValueError):
            model.apply(variables, tokens, train=True)
        _, kept = model.apply(variables, tokens, capture_intermediates=model.capture_layers,
                              mutable=list(model.layer_collections))
        states = [model.layer_state(kept, index) for index in range(model.num_layers)]
        assert all(state.shape[:2] == tokens.shape for state in states)
    with pytest.raises(RuntimeError):
        source.read(0)


def test_read_ahead_prepares_the_next_row_and_close_drains_it():
    from dew.inference.banks import SafetensorsBanks

    source = SafetensorsBanks(FIXTURE, cache_bytes=0)
    source.read(0, following=1)
    following = source.read(1)
    assert source.stats().misses == 2
    expected = translate_weights(read_weights(FIXTURE), translate_config(source.config), "mixtral")
    for actual, reference in zip(jax.tree.leaves(following), jax.tree.leaves(
            {name: tree["layers_1"] for name, tree in expected.items() if "layers_1" in tree}), strict=True):
        np.testing.assert_array_equal(actual, reference)
    source.read(0, following=1)
    source.close()
    assert source.stats().misses == 4


def test_an_incomplete_disk_checkpoint_is_refused_before_inference(tmp_path):
    from dew.inference.banks import SafetensorsBanks

    (tmp_path / "config.json").write_bytes((FIXTURE / "config.json").read_bytes())
    tensors = read_weights(FIXTURE)
    del tensors["model.layers.0.block_sparse_moe.experts.0.w1.weight"]
    save_sharded(tensors, tmp_path)
    with pytest.raises(ValueError):
        SafetensorsBanks(tmp_path)


def test_disk_banks_read_safetensors_alone_before_converting_the_model(tmp_path):
    """A source that ships its weights as a pickle alone is refused before any translation, which would
    hold the whole model; the same source with its safetensors streams."""
    import torch
    from safetensors.torch import load_file

    from dew.inference.banks import SafetensorsBanks

    original = Path(__file__).parent / "fixtures/hf/qwen38-dense-tiny"
    shutil.copy2(original / "config.json", tmp_path / "config.json")
    torch.save(load_file(original / "model.safetensors"), tmp_path / "pytorch_model.bin")
    with pytest.raises(FileNotFoundError):
        SafetensorsBanks(tmp_path)
    shutil.copy2(original / "model.safetensors", tmp_path / "model.safetensors")
    with SafetensorsBanks(tmp_path) as source:
        assert source.stats().misses == 0


def test_disk_loading_rejects_unbounded_layouts_before_reading_rows():
    from dew.inference.banks import SafetensorsBanks

    with SafetensorsBanks(FIXTURE, cache_bytes=0) as source:
        model = decoder(source.config, None)
        with pytest.raises(ValueError):
            source.stream(model.clone(scan_layers=False), mesh=single_mesh(), layout=DEVICE)
        with pytest.raises(ValueError):
            source.stream(model, mesh=single_mesh(), layout=Layout(
                min_shard=1, tolerance=1.0, host_parameters=("params/layers_*",)))
        assert source.stats().misses == 0


def test_disk_bank_memory_plan_has_no_decoder_weight_arguments():
    from dew.inference.banks import SafetensorsBanks

    with SafetensorsBanks(FIXTURE, cache_bytes=0) as source:
        model = decoder(source.config, None)
        streamed = source.stream(model, mesh=single_mesh(), layout=DEVICE)
        resident = HeldBanks(translate_weights(
            read_weights(FIXTURE), translate_config(source.config), "mixtral"
        )).place(model, mesh=single_mesh(), layout=DEVICE)
        tokens = jnp.ones((1, 2), jnp.int32)
        resident_plan = jax.jit(model.apply).lower(resident, tokens).compile().memory_analysis()
        disk_plan = jax.jit(model.apply).lower(streamed, tokens).compile().memory_analysis()
        rows = sum(source.layer_bytes(index) for index in range(model.num_layers))
        assert resident_plan.argument_size_in_bytes - disk_plan.argument_size_in_bytes == rows
        assert source.stats().misses == 0


def test_a_long_disk_scan_is_the_resident_moe(tmp_path):
    from dew.inference.banks import SafetensorsBanks

    config = json.loads((FIXTURE / "config.json").read_text())
    config["num_hidden_layers"] = 4
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensors = read_weights(FIXTURE)
    tensors.update({name.replace("model.layers.0.", "model.layers.2."): value
                    for name, value in tuple(tensors.items()) if name.startswith("model.layers.0.")})
    tensors.update({name.replace("model.layers.1.", "model.layers.3."): value
                    for name, value in tuple(tensors.items()) if name.startswith("model.layers.1.")})
    save_sharded(tensors, tmp_path, max_shard_size=24_000)
    with SafetensorsBanks(tmp_path, cache_bytes=0, param_dtype="float32") as source:
        model = decoder(source.config, None)
        resident = HeldBanks(translate_weights(
            tensors, translate_config(source.config), "mixtral"
        )).place(model, mesh=single_mesh(), layout=DEVICE)
        streamed = source.stream(model, mesh=single_mesh(), layout=DEVICE)
        tokens = jnp.asarray(np.load(FIXTURE / "input_ids.npy"))
        score = jax.jit(model.apply)
        np.testing.assert_array_equal(score(streamed, tokens), score(resident, tokens))


def test_a_decision_release_streams_its_wrappers_float64_layers_as_the_resident_load(tmp_path):
    """A slot release over a media wrapper streams its native backbone's layers from disk: the banked
    float64 forward, its blocks read both ways, is the resident load's bit for bit, and reads every
    row it runs."""
    from dew.inference.banks import SafetensorsBanks
    from dew.interop import Pretrained

    directory = tmp_path / "source"
    shutil.copytree(Path(__file__).parent / "fixtures/hf/qwen38-dense-tiny", directory)
    (directory / "readout.json").write_text(json.dumps({
        "readout": "slot_lm", "n_slots": 16, "head_dim": 8, "attention": "noncausal"}))
    with jax.enable_x64(new_val=True):
        bundle = Pretrained.load(directory, weights=False, dtype="float64", param_dtype="float64",
                                  attention_impl="reference")
        model = bundle.model.clone(language_model=bundle.model.language_model.clone(
            scan_layers=True, bank_layers=1))
        with SafetensorsBanks(directory, cache_bytes=0, param_dtype="float64", read_ahead=False) as source:
            streamed = source.stream(model, mesh=single_mesh(), layout=DEVICE)
            assert source.stats().misses == 0
            resident = Pretrained.load(directory, dtype="float64", param_dtype="float64",
                                       attention_impl="reference")
            placed = HeldBanks(resident.variables).place(model, mesh=single_mesh(), layout=DEVICE)
            tokens = jnp.asarray([[1, 8, 4, 6, 7, 2]], jnp.int32)
            groups = jnp.asarray([[0, 0, 0, 1, 1, 1]], jnp.int32)

            def score(variables):
                return model.apply(variables, tokens, block_groups=groups)

            expected = jax.jit(score)(placed)
            actual = jax.jit(score)(streamed)
            assert actual.dtype == expected.dtype == jnp.float64
            np.testing.assert_array_equal(actual, expected)
            assert source.stats().misses >= len(model.bank_sites(resident.variables)[0].view.groups)


def test_a_config_naming_expert_dtype_alone_streams_as_the_unquantized_load(tmp_path):
    """`expert_dtype` names a quantized source's expert storage only beside its quantization config
    (`codecs.source_quantization`): a config that states it alone is the unquantized checkpoint the
    loader reads, and the banks stream it as the resident load computes it."""
    from dew.inference.banks import SafetensorsBanks

    config = {**json.loads((FIXTURE / "config.json").read_text()), "expert_dtype": "fp4"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    shutil.copy2(FIXTURE / "model.safetensors", tmp_path / "model.safetensors")
    with SafetensorsBanks(tmp_path, cache_bytes=0, param_dtype="float32") as source:
        model = decoder(source.config, None)
        resident = HeldBanks(translate_weights(
            read_weights(FIXTURE), translate_config(config), "mixtral"
        )).place(model, mesh=single_mesh(), layout=DEVICE)
        streamed = source.stream(model, mesh=single_mesh(), layout=DEVICE)
        tokens = jnp.asarray(np.load(FIXTURE / "input_ids.npy"))
        score = jax.jit(model.apply)
        np.testing.assert_array_equal(score(streamed, tokens), score(resident, tokens))
