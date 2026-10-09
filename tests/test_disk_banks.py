"""Disk-backed MoE banks read at execution, not once at trace or load time."""

import json
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def decoder(config, bank_layers):
    record = {**translate_config(config), "dtype": "float32", "attention_impl": "reference"}
    return models.build("causal_transformer", {**record, "scan_layers": True,
                                               "bank_layers": bank_layers})


@pytest.mark.parametrize("bank_layers", [None, 1])
@pytest.mark.parametrize("read_ahead", [True, False])
def test_disk_moe_logits_and_cached_decode_equal_resident(bank_layers, read_ahead):
    from dew.inference.banks import SafetensorsBanks

    with SafetensorsBanks(FIXTURE, cache_bytes=0, param_dtype="float32", read_ahead=read_ahead) as source:
        model = decoder(source.config, bank_layers)
        resident = HeldBanks(translate_weights(
            read_weights(FIXTURE), translate_config(source.config), "mixtral"
        )).place(model, mesh=single_mesh(), layout=DEVICE)
        singleton = decoder(source.config, 1)
        singleton_store = HeldBanks(translate_weights(
            read_weights(FIXTURE), translate_config(source.config), "mixtral"
        )).place(singleton, mesh=single_mesh(), layout=DEVICE)
        streamed = source.stream(model, mesh=single_mesh(), layout=DEVICE)
        assert source.stats().misses == 0  # No decoder weight is read by loading or tracing.
        tokens = jax.device_put(np.load(FIXTURE / "input_ids.npy"), jax.devices()[0])
        score = jax.jit(model.apply)
        score.lower(streamed, tokens)
        assert source.stats().misses == 0
        expected = score(resident, tokens)
        actual = score(streamed, tokens)
        def equal_up_to_resident_fusion(actual, expected, inputs):
            if bank_layers == 1 or jax.default_backend() == "cpu":
                np.testing.assert_array_equal(actual, expected)
                return
            # An effectful one-trip scan cannot be peeled. XLA's resident
            # two-layer scan can, and its RMSNorm tiles become [1,2,32]
            # rather than [1,1,32]. Bound this case by the elementwise
            # spread of the same resident model's two legitimate fusions,
            # not by a chosen output tolerance.
            alternate = jax.jit(singleton.apply)(singleton_store, inputs)
            gap = np.abs(np.asarray(actual, np.float64) - np.asarray(expected, np.float64))
            spread = np.abs(np.asarray(alternate, np.float64) - np.asarray(expected, np.float64))
            assert np.all(gap <= spread), (float(gap.max()), float(spread.max()))

        equal_up_to_resident_fusion(actual, expected, tokens)
        np.testing.assert_allclose(actual, np.load(FIXTURE / "logits.npy"), atol=1e-4, rtol=1e-4)
        before = source.stats().misses
        equal_up_to_resident_fusion(score(streamed, tokens[:, ::-1]), score(resident, tokens[:, ::-1]),
                                   tokens[:, ::-1])
        assert source.stats().misses > before  # A compiled executable rereads uncached rows.

        def draw(variables):
            return generate(model, variables, tokens[:1, :2], key=jax.random.key(1),
                            max_new_tokens=3, sampling=Sampling(temperature=0.0))

        np.testing.assert_array_equal(draw(streamed).tokens, draw(resident).tokens)
        assert source.stats().cache_bytes == 0


def test_disk_source_reads_shards_and_bounds_cache(tmp_path):
    from dew.inference.banks import SafetensorsBanks

    (tmp_path / "config.json").write_bytes((FIXTURE / "config.json").read_bytes())
    save_sharded(read_weights(FIXTURE), tmp_path, max_shard_size=12_000)
    with SafetensorsBanks(tmp_path, cache_bytes=1, param_dtype="float32") as source:
        row = source.read(0)
        expected = translate_weights(read_weights(FIXTURE), translate_config(source.config), "mixtral")
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


@pytest.mark.parametrize("direct", [False, True])
def test_parallel_reads_return_every_stored_byte_wherever_it_sits_in_its_file(tmp_path, direct):
    """`ParallelReader` reads a mapped tensor's own bytes from its file,
    through the page cache or past it: off the page grid, smaller than a
    page, across several spans of its chunk, at the end of the file, and a
    view of one expert of a stacked tensor. An array that is not a mapped
    view comes back as it is."""
    from dew.interop.safetensors_io import read_file, save_params
    from dew.interop.streaming import ParallelReader
    from dew.training.host import stored_range

    rng = np.random.default_rng(0)
    params = {"a_small": rng.standard_normal(3, np.float32),
              "b_experts": rng.standard_normal((4, 37, 129), np.float32),
              "c_tail": rng.integers(0, 255, 70_001, np.uint8)}
    save_params(params, tmp_path / "model.safetensors")
    tensors, _ = read_file(tmp_path / "model.safetensors")
    path, offset = stored_range(tensors["b_experts"][2])
    with open(path, "rb") as stored:
        stored.seek(offset)
        assert stored.read(tensors["b_experts"][2].nbytes) == tensors["b_experts"][2].tobytes()
    assert stored_range(tensors["b_experts"][:, 1]) is None and stored_range(params["a_small"]) is None
    wanted = [tensors["a_small"], tensors["b_experts"], tensors["b_experts"][2], tensors["c_tail"],
              params["a_small"]]
    with ParallelReader(threads=3, chunk=16_384, direct=direct) as reader:
        loaded = reader.load(wanted)
    for array, read in zip(wanted, loaded, strict=True):
        assert read.shape == array.shape and read.dtype == array.dtype and stored_range(read) is None
        np.testing.assert_array_equal(read, array)
    assert loaded[-1] is params["a_small"]


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


def test_cache_retains_a_prefix_without_cyclic_scan_thrash():
    from dew.inference.banks import SafetensorsBanks

    with SafetensorsBanks(FIXTURE, cache_bytes=0) as shape_source:
        budget = shape_source.layer_bytes(0)
    with SafetensorsBanks(FIXTURE, cache_bytes=budget) as source:
        first = source.read(0)
        source.read(1)
        misses = source.stats().misses
        assert source.read(0) is first
        assert source.stats().misses == misses
        assert source.stats().cache_bytes == budget


def test_streaming_refuses_training_and_closed_sources():
    from dew.inference.banks import SafetensorsBanks

    with SafetensorsBanks(FIXTURE, cache_bytes=0) as source:
        model = decoder(source.config, None)
        variables = source.stream(model, mesh=single_mesh(), layout=DEVICE)
        tokens = jnp.ones((1, 2), jnp.int32)
        with pytest.raises(ValueError, match="inference"):
            model.apply(variables, tokens, train=True)
    with pytest.raises(RuntimeError, match="closed"):
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
    with pytest.raises(ValueError, match=r"expert 0|checkpoint does not fit"):
        SafetensorsBanks(tmp_path)


def test_disk_loading_rejects_unbounded_layouts_before_reading_rows():
    from dew.inference.banks import SafetensorsBanks

    with SafetensorsBanks(FIXTURE, cache_bytes=0) as source:
        model = decoder(source.config, None)
        with pytest.raises(ValueError, match="scan_layers=True"):
            source.stream(model.clone(scan_layers=False), mesh=single_mesh(), layout=DEVICE)
        with pytest.raises(ValueError, match="bounded host cache"):
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


def test_runtime_scan_fetches_exactly_the_stored_expert_weights():
    from dew.inference.banks import SafetensorsBanks
    from dew.nn.backbones.decoder_stack import _fetched_layer

    with SafetensorsBanks(FIXTURE, cache_bytes=0) as source:
        model = decoder(source.config, None)
        streamed = source.stream(model, mesh=single_mesh(), layout=DEVICE)
        resident = HeldBanks(translate_weights(
            read_weights(FIXTURE), translate_config(source.config), "mixtral"
        )).place(model, mesh=single_mesh(), layout=DEVICE)
        reader = streamed["streaming"]["layers_0_1"]["bank"]
        bank = {name: tree["layers_0_1"] for name, tree in resident.items() if "layers_0_1" in tree}

        def check(bank):
            def step(carry, index):
                fetched = reader.fetch(index, carry)
                expected = _fetched_layer(bank, index)
                equals = jax.tree.map(
                    lambda actual, reference: jnp.all(actual == reference), fetched, expected
                )
                return carry + 1, jnp.all(jnp.stack(jax.tree.leaves(equals)))

            return jax.lax.scan(step, jnp.zeros((1, 1, 1)), jnp.arange(model.num_layers))[1]

        np.testing.assert_array_equal(jax.jit(check)(bank), np.ones(model.num_layers, bool))


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
