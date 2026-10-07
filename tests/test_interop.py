"""safetensors round-trips for Flax parameter trees, and the hub round trip.

The tree is nested and the file is flat, so what these tests hold onto is the
'/'-joined naming: nesting, values and dtypes have to survive a save and a
load, the names on disk have to be readable by a safetensors reader that knows
nothing about dew, and a missing optional dependency has to say so. The hub
pair is tested against a recording stand-in for the hub client: what matters
is the call it makes and the directory it hands over, and no test reaches the
network.
"""

import json
import sys
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

import dew
from dew.interop import hub, load_params, save_hf_layout, save_params
from dew.interop.hub import pull_from_hub
from dew.interop.safetensors_io import read_file, write_file
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.dit import TextContext

safetensors_numpy = pytest.importorskip("safetensors.numpy")
safetensors = import_module("safetensors")



@pytest.fixture
def params(rng):
    model = SimpleDiT(
        patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1
    )
    x = jax.random.normal(rng, (1, 8, 8, 3))
    return model.init(
        rng,
        x,
        jnp.ones((1,)),
        TextContext(jnp.ones((1, 77, 768)), jnp.ones((1, 77), bool)),
    )


def flat_names(tree):
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {"/".join(entry.key for entry in path) for path, _ in leaves}


def test_round_trip_keeps_the_tree_and_the_values(params, tmp_path):
    path = tmp_path / "model.safetensors"
    save_params(params, path)
    loaded = load_params(path)

    assert jax.tree_util.tree_structure(loaded) == jax.tree_util.tree_structure(params)
    for saved, restored in zip(jax.tree.leaves(params), jax.tree.leaves(loaded), strict=True):
        assert np.array_equal(np.asarray(saved), restored)


def test_round_trip_keeps_bfloat16(params, tmp_path):
    """The point of writing bf16 is the bytes, so a load must not widen them."""
    narrowed = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), params)
    path = tmp_path / "bf16.safetensors"
    save_params(narrowed, path)
    loaded = load_params(path)

    for saved, restored in zip(
        jax.tree.leaves(narrowed), jax.tree.leaves(loaded), strict=True
    ):
        assert restored.dtype == jnp.bfloat16
        assert np.array_equal(np.asarray(saved), restored)


def test_names_on_disk_are_the_slash_joined_paths(params, tmp_path):
    """A reader outside dew sees flat names, and they are the module paths."""
    path = tmp_path / "model.safetensors"
    save_params(params, path)

    names = set(safetensors_numpy.load_file(str(path)))
    assert names == flat_names(params)


def test_hf_layout_writes_the_pair_a_loader_looks_for(params, tmp_path):
    config = {"architecture": "simple_dit", "patch_size": 4, "emb_features": 32}
    export = tmp_path / "export"
    save_hf_layout(params, config, export)

    assert json.loads((export / "config.json").read_text()) == config
    loaded = load_params(export / "model.safetensors")
    assert jax.tree_util.tree_structure(loaded) == jax.tree_util.tree_structure(params)


def test_a_key_holding_the_separator_is_refused(tmp_path):
    with pytest.raises(ValueError, match="'/'"):
        save_params({"params": {"a/b": jnp.ones((2,))}}, tmp_path / "model.safetensors")


def test_missing_safetensors_names_the_extra(params, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "safetensors", None)
    with pytest.raises(ImportError, match=r"dewml\[interop\]"):
        save_params(params, tmp_path / "model.safetensors")


def test_missing_safetensors_reader_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "safetensors", None)
    with pytest.raises(ImportError, match=r"dewml\[interop\]"):
        read_file(tmp_path / "model.safetensors")


# ---------------------------------------------------------------------------------
# The memory-mapped reader
# ---------------------------------------------------------------------------------


def test_every_stored_dtype_reads_back_exactly(tmp_path):
    """Scalar, nonempty and empty tensors keep their own dtype and bytes."""
    stored = {
        "bf16": np.asarray(jnp.asarray([1.5, -2.0], jnp.bfloat16)),
        "f16": np.asarray([0.5, -0.25], np.float16),
        "f32": np.asarray([0.5, -0.25], np.float32),
        "f64": np.asarray([[3.0]], np.float64),
        "i64": np.asarray([7], np.int64),
        "bytes": np.asarray([0, 127, 128, 255], np.uint8),
        "empty": np.zeros((0, 4), np.uint8),
        "flag": np.ones((), dtype=np.bool_),
        "scalar": np.asarray(0.25, np.float32),
    }
    path = tmp_path / "model.safetensors"
    safetensors_numpy.save_file(stored, str(path), metadata={"writer": "dew"})

    tensors, metadata = read_file(path)

    assert metadata == {"writer": "dew"}
    for name, expected in stored.items():
        tensor = tensors[name]
        assert tensor.dtype == expected.dtype and tensor.shape == expected.shape
        np.testing.assert_array_equal(tensor, expected)


def test_float8_payloads_are_mapped_without_numpy_dtype_conversion(tmp_path):
    """FP8 array, scalar and empty payloads use the official format tag.
    Compare stored bits, including signed zero, rather than widened values."""
    from dew.interop.codecs import E4M3

    stored = {
        "weight": np.asarray([0.0, -0.0, 1.5, -2.0, 448.0], dtype=E4M3),
        "scalar": np.asarray(-1.5, dtype=E4M3),
        "empty": np.empty((0, 3), dtype=E4M3),
        "bf16": np.asarray([1.5, -2.0], dtype=ml_dtypes.bfloat16),
        "packed": np.asarray([0, 128, 255], dtype=np.uint8),
    }
    path = tmp_path / "fp8.safetensors"
    safetensors_numpy.save_file(stored, str(path))
    tensors, _ = read_file(path)
    for name, expected in stored.items():
        actual = tensors[name]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert isinstance(actual.base, np.memmap) and not actual.flags.writeable
        assert actual.tobytes() == expected.tobytes()


def raw_safetensors(path, entries):
    """Write a safetensors file byte for byte, for the dtype tags no NumPy writer emits."""
    header, payload = {}, b""
    for name, (tag, shape, data) in entries.items():
        header[name] = {"dtype": tag, "shape": list(shape),
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def test_block_scale_exponents_and_packed_fp4_are_mapped_as_stored(tmp_path):
    """DeepSeek-V4's shards carry F8_E8M0 scales and F4 experts (the header of
    DeepSeek-V4-Flash shard 5). An E8M0 byte is the exponent 2**(b - 127); an
    F4 tensor of n elements is n / 2 bytes, returned as that byte view, the
    way PyTorch writes a float4_e2m1fn_x2 [1, 2] as F4 [1, 4]."""
    path = tmp_path / "v4.safetensors"
    raw_safetensors(path, {"w.scale": ("F8_E8M0", (3,), bytes([127, 128, 120])),
                           "w": ("F4", (1, 4), bytes([0x21, 0x43]))})

    tensors, _ = read_file(path)

    scale = tensors["w.scale"]
    assert scale.dtype == ml_dtypes.float8_e8m0fnu
    np.testing.assert_array_equal(scale.astype(np.float32), [1.0, 2.0, 2.0 ** -7])
    packed = tensors["w"]
    assert packed.dtype == np.uint8 and packed.shape == (1, 2)
    assert packed.tobytes() == bytes([0x21, 0x43]) and isinstance(packed.base, np.memmap)


def test_an_fp4_tensor_with_an_odd_last_axis_is_refused_by_name(tmp_path):
    path = tmp_path / "odd.safetensors"
    raw_safetensors(path, {"w": ("F4", (2, 3), bytes(3))})

    with pytest.raises(ValueError, match=r"'w'.*F4 with shape \(2, 3\)"):
        read_file(path)


def test_the_arrays_are_read_only_views_of_the_file(tmp_path):
    """The map outlives the reader context, and the arrays refuse writes."""
    path = tmp_path / "model.safetensors"
    safetensors_numpy.save_file(
        {"w": np.asarray([1.0, 2.0, 3.0], np.float32)}, str(path)
    )

    tensors, _ = read_file(path)

    tensor = tensors["w"]
    assert isinstance(tensor.base, np.memmap) and not tensor.flags.writeable
    np.testing.assert_array_equal(tensor, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="read-only"):
        tensor[0] = 9.0


_PUBLISHERS = [
    pytest.param(
        lambda value, path: save_params({"params": {"w": value}}, path),
        id="save_params",
    ),
    pytest.param(
        lambda value, path: write_file({"params/w": value}, path, {}), id="write_file"
    ),
]


def test_the_same_tensors_and_metadata_write_the_same_bytes(tmp_path):
    """safetensors 0.8.0 keeps a header's metadata in a Rust HashMap, whose
    order is drawn per map, so the same table and metadata wrote different
    bytes from one write to the next (6 distinct files in 20 writes of three
    keys). A written file is a function of what it holds: an adapter saved
    twice, or a file and its digest, agree byte for byte."""
    path = tmp_path / "model.safetensors"
    tensors = {"a": np.ones((2, 2), np.float32), "b": np.zeros(3, np.int32)}
    metadata = {"format": "pt", "lora_adapter_metadata": '{"r": 2}', "z": "1"}
    written = set()
    for _ in range(20):
        write_file(tensors, path, metadata)
        written.add(path.read_bytes())
    assert len(written) == 1
    header = json.loads(next(iter(written))[8:8 + int.from_bytes(next(iter(written))[:8], "little")])
    assert list(header["__metadata__"]) == sorted(metadata)
    loaded, read = read_file(path)
    assert read == metadata and all(np.array_equal(loaded[name], tensors[name]) for name in tensors)


@pytest.mark.parametrize("publish", _PUBLISHERS)
def test_overwrite_retains_the_previously_loaded_tree(tmp_path, publish):
    path = tmp_path / "model.safetensors"
    save_params({"params": {"w": np.asarray([1.0, 2.0], np.float32)}}, path)
    original = load_params(path)

    publish(np.asarray([9.0, 8.0], np.float32), path)

    np.testing.assert_array_equal(original["params"]["w"], [1.0, 2.0])
    np.testing.assert_array_equal(load_params(path)["params"]["w"], [9.0, 8.0])
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("publish", _PUBLISHERS)
def test_failed_publication_preserves_the_file_and_cleans_the_temporary(
    tmp_path, monkeypatch, publish
):
    path = tmp_path / "model.safetensors"
    save_params({"params": {"w": np.asarray([1.0, 2.0], np.float32)}}, path)
    original = load_params(path)
    before = path.read_bytes()

    def partial_write(tensors, filename, metadata=None):
        Path(filename).write_bytes(b"incomplete payload")
        raise OSError("injected write failure")

    monkeypatch.setattr(safetensors_numpy, "save_file", partial_write)
    with pytest.raises(OSError, match="injected write failure"):
        publish(np.asarray([9.0, 8.0], np.float32), path)

    assert path.read_bytes() == before
    np.testing.assert_array_equal(original["params"]["w"], [1.0, 2.0])
    assert set(tmp_path.iterdir()) == {path}


def test_a_truncated_file_fails_in_the_official_parser(tmp_path):
    path = tmp_path / "model.safetensors"
    safetensors_numpy.save_file({"w": np.ones((4,), np.float32)}, str(path))
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])

    with pytest.raises(safetensors.SafetensorError):
        read_file(path)


def test_a_metadata_only_file_reads_empty(tmp_path):
    path = tmp_path / "model.safetensors"
    safetensors_numpy.save_file({}, str(path), metadata={"layout": "dew"})

    tensors, metadata = read_file(path)

    assert tensors == {} and metadata == {"layout": "dew"}


def test_a_bf16_checkpoint_still_loads_as_fp32_parameters(tmp_path):
    """A bfloat16 checkpoint reads in its stored dtype and widens exactly,
    so the public fp32 default is unchanged."""
    from dew.interop import Pretrained

    source = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny"
    tensors, _ = read_file(source / "model.safetensors")
    bf16 = {
        name: np.asarray(jnp.asarray(tensor, jnp.bfloat16))
        for name, tensor in tensors.items()
    }
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "config.json").write_text((source / "config.json").read_text())
    write_file(bf16, directory / "model.safetensors", {"format": "pt"})

    loaded = Pretrained.load(str(directory), dtype="float32", attention_impl="xla")

    leaves, _ = jax.tree_util.tree_flatten(loaded.variables)
    assert all(np.asarray(leaf).dtype == np.float32 for leaf in leaves)
    np.testing.assert_array_equal(
        np.asarray(loaded.variables["params"]["embed_tokens"]["embedding"]),
        np.asarray(bf16["model.embed_tokens.weight"], np.float32),
    )


def assert_parameter_storage(reference, actual, is_parameter):
    """The public storage contract on full variable paths, not implementation calls."""
    assert jax.tree.structure(reference) == jax.tree.structure(actual)
    for (path, before), after in zip(jax.tree_util.tree_leaves_with_path(reference),
                                    jax.tree.leaves(actual), strict=True):
        names = tuple(entry.key for entry in path)
        before, after = np.asarray(before), np.asarray(after)
        floating = jnp.issubdtype(before.dtype, jnp.floating)
        if floating:
            assert before.dtype == np.float32, names
        dtype = np.dtype(ml_dtypes.bfloat16) if floating and is_parameter(names) else before.dtype
        assert after.dtype == dtype, names
        np.testing.assert_array_equal(after, before.astype(dtype), err_msg=str(names))


@pytest.mark.parametrize("family", [
    "llama-tiny", "gemma4-tiny-mm", "gemma-4-audio-tiny", "diffusion-gemma-workflow",
])
def test_public_parameter_storage_is_independent_of_compute_and_roundtrips(tmp_path, family):
    from dew.interop import Pretrained
    from dew.nn.diffusion_gemma import DiffusionGemma

    directory = Path(__file__).resolve().parent / "fixtures" / "hf" / family
    masters = Pretrained.load(directory, dtype="bfloat16", attention_impl="xla")
    native = Pretrained.load(directory, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
    master_model = masters.model.text if isinstance(masters.model, DiffusionGemma) else masters.model
    native_model = native.model.text if isinstance(native.model, DiffusionGemma) else native.model
    assert jnp.dtype(master_model.dtype) == jnp.dtype(jnp.bfloat16)
    assert jnp.dtype(native_model.dtype) == jnp.dtype(jnp.float32)
    assert_parameter_storage(masters.variables, native.variables, lambda path: path[0] == "params")
    destination = tmp_path / "export"
    native.save(destination)
    restored = Pretrained.load(destination, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
    assert jax.tree.structure(native.variables) == jax.tree.structure(restored.variables)
    for before, after in zip(
        jax.tree.leaves(native.variables), jax.tree.leaves(restored.variables), strict=True
    ):
        assert np.asarray(before).dtype == np.asarray(after).dtype
        np.testing.assert_array_equal(before, after)


def test_public_loader_rejects_aliases_hidden_by_bfloat16_rounding(tmp_path):
    from dew.interop import Pretrained

    source = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny"
    tensors, _ = read_file(source / "model.safetensors")
    config = json.loads((source / "config.json").read_text())
    config["tie_word_embeddings"] = True
    shape = tensors["model.embed_tokens.weight"].shape
    tensors["model.embed_tokens.weight"] = np.full(shape, 1., np.float32)
    tensors["lm_head.weight"] = np.full(shape, 1. + 1. / 1024, np.float32)
    directory = tmp_path / "different-aliases"
    save_hf_layout(tensors, config, directory)
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        Pretrained.load(directory, param_dtype="bfloat16")


# ---------------------------------------------------------------------------------
# Hub pull
# ---------------------------------------------------------------------------------


class _RecordingApi:
    """Stands in for HfApi and keeps every call a push makes, with the files
    the uploaded folder held while the call ran."""

    def __init__(self):
        self.created = []
        self.uploaded = []
        self.files = []

    def create_repo(self, repo_id, **kwargs):
        self.created.append((repo_id, kwargs))

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)
        self.files.append({entry.name for entry in Path(kwargs["folder_path"]).iterdir()})


def test_a_bundle_pushes_what_it_saves_to_a_created_repo(tmp_path, monkeypatch):
    """`push_to_hub` is `save` into a staging directory and that directory
    uploaded, to a repo created when missing; the privacy flag and the commit
    message pass through."""
    import huggingface_hub

    from dew.interop import PretrainedDecoder
    from dew.nn.backbones import CausalTransformer

    api = _RecordingApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=2, mlp_features=16,
                              max_seq_len=8, attention_impl="reference")
    bundle = PretrainedDecoder.from_model(model, model.init(jax.random.key(0), np.zeros((1, 2), np.int32)),
                                          tokenizer="byte")
    bundle.push_to_hub("acme/dew-export", private=True, commit_message="step 1000")
    bundle.save(tmp_path / "saved")

    assert api.created == [("acme/dew-export", {"private": True, "exist_ok": True})]
    assert [(call["repo_id"], call["commit_message"]) for call in api.uploaded] == [
        ("acme/dew-export", "step 1000")]
    assert api.files == [{entry.name for entry in (tmp_path / "saved").iterdir()}]


def test_pull_returns_the_snapshot_directory(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)

    assert pull_from_hub("acme/dew-export", revision="v2") == tmp_path / "snapshot"
    assert pull_from_hub("acme/dew-export") == tmp_path / "snapshot"
    assert calls == [
        {"repo_id": "acme/dew-export", "revision": "v2"},
        {"repo_id": "acme/dew-export", "revision": None},
    ]


# ---------------------------------------------------------------------------------
# A trained run out to the published layout
# ---------------------------------------------------------------------------------


def assert_transformers_reads_the_run(export, task, ids, *, ordered_export=None):
    """transformers' own class for the export's model_type loads its files
    with a clean report (`decoder_export_reference.reference_model`) and,
    over the same ids, holds the run's own logits to tests/reference_error.py's
    rule: in fp32 the reference, in float64 the truth. An `ordered_export`
    holds the trained run over 52 residual orders when one draw can flake;
    every reference order in float64 must compute the identity's truth.
    """
    from reference_error import (
        ORDERS,
        assert_as_exact_as_the_reference,
        assert_as_exact_over_orders,
        assert_computes_the_oracle,
        distance,
    )
    from residual_orders import orders, permuted, residual_width

    from dew.interop import Pretrained
    from tools.decoder_export_reference import Case, reference_logits

    case = Case("run export", str(export))
    truth = reference_logits(case, export, ids, wide=True)
    if ordered_export is None:
        ours = np.asarray(task.model.apply(task.variables, jnp.asarray(ids, jnp.int32)))
        assert_as_exact_as_the_reference(ours, reference_logits(case, export, ids), truth,
                                         "run export logits")
        return
    source = Pretrained.load(export, dtype="float32", attention_impl="reference")
    forward = jax.jit(lambda variables: task.model.apply(variables, jnp.asarray(ids, jnp.int32)))
    mine, theirs = [], []
    # Two attention projections, a key reduction, two MLP projections and
    # the norms/residuals per layer, then the output head, bound the chain.
    model = source.model
    roundings = model.num_layers * (4 * model.emb_features + model.mlp_features + ids.shape[1] + 16)
    roundings += 2 * model.emb_features
    for order in orders(residual_width(task.variables), ORDERS, seed=0):
        variables = permuted(task.variables, order)
        source.save(ordered_export, variables=variables)
        assert_computes_the_oracle(reference_logits(case, ordered_export, ids, wide=True), truth,
                                   "run export logits in order", roundings=roundings)
        mine.append(distance(forward(variables), truth))
        theirs.append(distance(reference_logits(case, ordered_export, ids), truth))
    assert_as_exact_over_orders(mine, theirs, "run export logits")


def test_a_trained_lm_run_exports_and_reloads_at_its_own_logits(tmp_path):
    """The whole way out of a run directory: run.json and the checkpoint in,
    a Hugging Face directory out, and `Pretrained.load` reads it back at the
    logits the run's own task computes, as transformers does."""
    from test_inference import make_lm_run

    from dew.interop import Pretrained

    run = tmp_path / "run"
    run.mkdir()
    make_lm_run(run)
    task = dew.pipeline(str(run))
    destination = tmp_path / "export"

    Pretrained.from_run(str(run)).save(destination)

    assert {entry.name for entry in destination.iterdir()} == {
        "config.json", "generation_config.json", "model.safetensors"}
    assert json.loads((destination / "generation_config.json").read_text())["tokenizer_name"] == "byte"
    reloaded = Pretrained.load(destination, dtype="float32", attention_impl="reference")
    ids = jnp.asarray([[3, 4, 5, 6]], jnp.int32)
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)),
                                  np.asarray(task.model.apply(task.variables, ids)))
    assert_transformers_reads_the_run(destination, task, np.asarray([[3, 4, 5, 6, 7, 8, 9, 10]]),
                                      ordered_export=tmp_path / "orders")


def test_exporting_a_run_whose_model_has_no_published_layout_names_it(tmp_path):
    """A latent diffusion run: the denoiser is a native model with no file
    to be written back into, so building its export bundle is refused,
    naming the model and the task that loads the run."""
    from test_inference import make_run

    from dew.interop import Pretrained

    make_run(tmp_path)
    with pytest.raises(TypeError, match="SimpleDiT has no maintained exported bundle layout"):
        Pretrained.from_run(str(tmp_path))


def test_writing_a_model_no_decoder_family_describes_as_a_decoder_names_it():
    """A decoder family's config is derived from a decoder's fields, so a
    denoiser handed to the decoder writer is refused by name rather than
    read as fields it does not have."""
    from dew.interop import PretrainedDecoder
    from dew.registry import models

    denoiser = models.build("simple_dit")
    with pytest.raises(TypeError, match="SimpleDiT has no Hugging Face decoder layout"):
        PretrainedDecoder.from_model(denoiser, {})


FRESH_TRAINING = """
import sys

import grain.python as grain
import jax.numpy as jnp
import numpy as np
import optax

from dew.checkpoints import Checkpoints
from dew.data import ByteTokenizer, Dataset, Loading
from dew.inference import RunProcessor
from dew.nn.backbones import CausalTransformer
from dew.objectives.{module} import {objective}
from dew.training import Trainer

model = CausalTransformer(vocab_size=256, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                          max_seq_len=16, dtype=jnp.float32, attention_impl="xla")
objective = {objective}(model, 8, ema_decay=None, processor=RunProcessor(ByteTokenizer()))
rows = [{{"text": np.arange(9, dtype=np.int32)}} for _ in range(8)]
data = Dataset.from_grain(grain.MapDataset.source(rows), batch=8, loading=Loading(workers=0))
checkpoints = Checkpoints(sys.argv[1])
Trainer(objective, optax.sgd(.01), key=0, checkpoints=checkpoints).fit(data, steps=1, checkpoint_every=1)
checkpoints.wait()
"""


def test_a_run_trained_in_one_process_exports_from_a_fresh_one(tmp_path):
    """`dew export` in a new process reads the run by its record alone: the
    registry imports the module that registers each name the record holds,
    so nothing the training process happened to import is needed. What it
    writes, transformers reads at the run's own logits."""
    import os
    import subprocess


    root = Path(__file__).resolve().parents[1]
    env = {**{name: value for name, value in os.environ.items() if name != "XLA_FLAGS"},
           "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(root / "src")}
    program = FRESH_TRAINING.format(module="lm", objective="LMObjective")
    trained = subprocess.run([sys.executable, "-c", program, str(tmp_path / "run")],
                             capture_output=True, text=True, env=env, timeout=600)
    assert trained.returncode == 0, trained.stderr[-2000:]
    exported = subprocess.run([sys.executable, "-m", "dew.cli.main", "export", str(tmp_path / "run"),
                               str(tmp_path / "export")],
                              capture_output=True, text=True, env=env, timeout=600)
    assert exported.returncode == 0, exported.stderr[-2000:]
    assert (tmp_path / "export" / "model.safetensors").is_file()
    assert_transformers_reads_the_run(tmp_path / "export", dew.pipeline(str(tmp_path / "run")),
                                      np.arange(9).reshape(1, 9))


def test_a_registry_imports_the_module_that_registers_a_name_and_no_other():
    """A lookup of a name nothing has registered yet imports the module whose
    decorator registers it, found in Dew's sources, with what that module
    imports, and no other registering module (JEPA's objective, the eval
    harness); a name no module registers still raises."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(root / "src")}
    program = ("import sys\n"
               "from dew.registry import objectives, solvers\n"
               "assert 'dew.objectives.rl.ppo' not in sys.modules\n"
               "print(objectives['ppo'].__module__, solvers['heun'].__name__)\n"
               "print('dew.objectives.jepa' in sys.modules, 'dew.eval.harness' in sys.modules)\n"
               "try:\n"
               "    objectives['no_such_objective']\n"
               "except KeyError as error:\n"
               "    print('refused', 'no_such_objective' in str(error))\n")
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env=env,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.splitlines() == ["dew.objectives.rl.ppo Heun", "False False", "refused True"]


ALIASES = """
import json

import dew.registry as registry

drift = {}
for kind in registry.KINDS:
    for alias, path in kind.paths.items():
        try:
            member = kind[alias]
        except ModuleNotFoundError as missing:
            if missing.name is None or missing.name.split(".")[0] == "dew":
                raise
            continue  # an optional dependency this environment lacks
        if registry.import_path(member) != path:
            drift[f"{kind.kind} {alias}"] = registry.import_path(member)
print(json.dumps(drift))
"""


def test_every_alias_names_the_class_at_its_path():
    """Each alias imports, in a fresh process, a class whose own import path
    is the path the alias names, so a class moved or renamed without its
    alias fails here rather than in a user's run."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(root / "src")}
    done = subprocess.run([sys.executable, "-c", ALIASES], capture_output=True, text=True, env=env,
                          timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.splitlines()[-1]) == {}


def test_the_cli_exports_a_run_and_refuses_a_directory_that_is_not_one(tmp_path, capsys):
    """`dew export <run> <dest>` is the same call with two positional names."""
    from test_inference import make_lm_run

    from dew.cli.main import main

    run = tmp_path / "run"
    run.mkdir()
    make_lm_run(run)
    destination = tmp_path / "export"

    assert main(["export", str(run), str(destination)]) == 0
    assert (destination / "model.safetensors").is_file()
    assert "exported" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        main(["export", str(tmp_path / "nothing"), str(tmp_path / "other")])


def test_a_block_diffusion_run_writes_no_export_transformers_cannot_read(tmp_path):
    """transformers' DiffusionGemmaForBlockDiffusion, the implementation the
    checkpoint layout is for, builds experts and a vision tower, and takes
    generation_config.json as a closed set of fields. Google's dense
    text-only model has neither, and Dew's byte vocabulary has no files and
    no field to be named in, so a run of either is refused before a file is
    written (a run of the image-reading model on its own tokenizer exports:
    test_diffusion_gemma_workflow.py)."""
    from test_inference import make_block_run

    from dew.interop import Pretrained

    dense = tmp_path / "dense"
    dense.mkdir()
    make_block_run(dense, fixture="diffusion-gemma-sft")
    with pytest.raises(ValueError, match="cannot read an export"):
        Pretrained.from_run(str(dense), ema=False).save(tmp_path / "refused")
    assert not (tmp_path / "refused").exists()

    byte = tmp_path / "byte"
    byte.mkdir()
    make_block_run(byte)
    with pytest.raises(ValueError, match="no tokenizer files"):
        Pretrained.from_run(str(byte), ema=False).save(tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_the_fid_converter_refuses_a_pickle_that_is_missing_a_key(tmp_path):
    """The extractor's own variables tree says which arrays the conversion
    wants and what they are called, so a jax-fid pickle that cannot answer for
    one of them is refused by the name it failed on rather than landing a tree
    with a hole in it."""
    import pickle

    from dew.interop.inception_fid import convert, upstream_names

    tree: dict = {}
    for upstream in upstream_names().values():
        node = tree
        for step in upstream[:-1]:
            node = node.setdefault(step, {})
        node[upstream[-1]] = np.zeros(1, np.float32)
    del tree["Mixed_7c"]["branch_pool"]["bn"]["var"]

    path = tmp_path / "inception_v3_fid.pickle"
    path.write_bytes(pickle.dumps(tree))
    with pytest.raises(ValueError, match="Mixed_7c/branch_pool/bn/var"):
        convert(path)


def test_a_written_tensor_reads_back_in_its_own_order_whatever_its_memory_layout(tmp_path):
    """A column-major or transposed view is written as the values it holds.
    safetensors writes memory as C order, and a TPU returns column-major host
    views of some arrays: a bf16 checkpoint written from one read back
    transposed on the TPU lane (16,364 of 16,384 values wrong)."""
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    layouts = {"fortran": np.asfortranarray(values), "transposed": values.T.copy().T,
               "bf16": np.asfortranarray(values.astype(ml_dtypes.bfloat16))}
    write_file(layouts, tmp_path / "model.safetensors", {"format": "pt"})
    read, _ = read_file(tmp_path / "model.safetensors")
    for name in layouts:
        np.testing.assert_array_equal(np.asarray(read[name], np.float32), values, err_msg=name)
