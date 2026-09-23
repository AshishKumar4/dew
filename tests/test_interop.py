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
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

import dew
from dew.interop import hub, load_params, pull_from_hub, push_to_hub, save_hf_layout, save_params
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.dit import TextContext

safetensors_numpy = pytest.importorskip("safetensors.numpy")
import safetensors

from dew.interop.safetensors_io import read_file, write_file


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
    for saved, restored in zip(jax.tree.leaves(params), jax.tree.leaves(loaded)):
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
    with pytest.raises(ImportError, match=r"dew-ml\[interop\]"):
        save_params(params, tmp_path / "model.safetensors")


def test_missing_safetensors_reader_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "safetensors", None)
    with pytest.raises(ImportError, match=r"dew-ml\[interop\]"):
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
        "flag": np.asarray(True, np.bool_),
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
    from dew.interop.quantized import E4M3

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

    with pytest.raises(ValueError, match="'w'.*F4 with shape \\(2, 3\\)"):
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
    from dew.interop import load_pretrained

    source = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama-tiny"
    tensors, _ = read_file(source / "model.safetensors")
    bf16 = {
        name: np.asarray(jnp.asarray(tensor, jnp.bfloat16))
        for name, tensor in tensors.items()
    }
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "config.json").write_text((source / "config.json").read_text())
    safetensors_numpy.save_file(bf16, str(directory / "model.safetensors"))

    loaded = load_pretrained(str(directory), dtype="float32", attention_impl="xla")

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
    from dew.interop import load_pretrained
    from dew.nn.diffusion_gemma import DiffusionGemma

    directory = Path(__file__).resolve().parent / "fixtures" / "hf" / family
    masters = load_pretrained(directory, dtype="bfloat16", attention_impl="xla")
    native = load_pretrained(directory, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
    master_model = masters.model.text if isinstance(masters.model, DiffusionGemma) else masters.model
    native_model = native.model.text if isinstance(native.model, DiffusionGemma) else native.model
    assert jnp.dtype(master_model.dtype) == jnp.dtype(jnp.bfloat16)
    assert jnp.dtype(native_model.dtype) == jnp.dtype(jnp.float32)
    assert_parameter_storage(masters.variables, native.variables, lambda path: path[0] == "params")
    destination = tmp_path / "export"
    native.save(destination)
    restored = load_pretrained(destination, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
    assert jax.tree.structure(native.variables) == jax.tree.structure(restored.variables)
    for before, after in zip(jax.tree.leaves(native.variables), jax.tree.leaves(restored.variables), strict=True):
        assert np.asarray(before).dtype == np.asarray(after).dtype
        np.testing.assert_array_equal(before, after)


def test_public_loader_rejects_aliases_hidden_by_bfloat16_rounding(tmp_path):
    from dew.interop import load_pretrained

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
        load_pretrained(directory, param_dtype="bfloat16")


# ---------------------------------------------------------------------------------
# Hub push and pull
# ---------------------------------------------------------------------------------


class _RecordingApi:
    """Stands in for HfApi and keeps every call push_to_hub makes."""

    def __init__(self):
        self.created = []
        self.uploaded = []
        self.files = []

    def create_repo(self, repo_id, **kwargs):
        self.created.append((repo_id, kwargs))

    def upload_folder(self, **kwargs):
        # The real client reads the folder during the call, and a staged
        # export is gone by the time the test looks, so the listing is taken
        # here, where the client would take it.
        self.uploaded.append(kwargs)
        self.files.append({entry.name for entry in Path(kwargs["folder_path"]).iterdir()})


@pytest.fixture
def api(monkeypatch):
    recording = _RecordingApi()
    monkeypatch.setattr(hub, "HfApi", lambda: recording)
    return recording


def test_push_creates_the_repo_and_uploads_the_export_directory(params, tmp_path, api):
    export = tmp_path / "export"
    save_hf_layout(params, {"architecture": "simple_dit"}, export)

    push_to_hub(export, "acme/dew-export")

    assert api.created == [("acme/dew-export", {"private": False, "exist_ok": True})]
    assert api.uploaded == [
        {
            "repo_id": "acme/dew-export",
            "folder_path": str(export),
            "commit_message": "Upload dew export",
        }
    ]
    uploaded = Path(api.uploaded[0]["folder_path"])
    assert {entry.name for entry in uploaded.iterdir()} == {
        "model.safetensors",
        "config.json",
    }


def test_push_passes_the_private_flag_and_the_commit_message_through(
    params, tmp_path, api
):
    export = tmp_path / "export"
    save_hf_layout(params, {"architecture": "simple_dit"}, export)

    push_to_hub(export, "acme/held-back", private=True, commit_message="step 1000")

    assert api.created == [("acme/held-back", {"private": True, "exist_ok": True})]
    assert api.uploaded[0]["commit_message"] == "step 1000"


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


def test_a_trained_lm_run_exports_and_reloads_at_its_own_logits(tmp_path):
    """The whole way out of a run directory: run.json and the checkpoint in,
    a Hugging Face directory out, and `load_pretrained` reads it back at the
    logits the run's own task computes."""
    from test_inference import make_lm_run

    import dew
    from dew.interop import export_run, load_pretrained

    run = tmp_path / "run"
    run.mkdir()
    make_lm_run(run)
    task = dew.pipeline(str(run))
    destination = tmp_path / "export"

    export_run(str(run), destination)

    assert {entry.name for entry in destination.iterdir()} == {
        "config.json", "generation_config.json", "model.safetensors"}
    assert json.loads((destination / "generation_config.json").read_text())["tokenizer_name"] == "byte"
    reloaded = load_pretrained(destination, dtype="float32", attention_impl="reference")
    ids = jnp.asarray([[3, 4, 5, 6]], jnp.int32)
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)),
                                  np.asarray(task.model.apply(task.variables, ids)))


def test_exporting_a_run_whose_model_has_no_published_layout_names_it(tmp_path):
    """A latent diffusion run: the denoiser is a native model with no file
    to be written back into, so the refusal names the model and what does
    export."""
    from test_inference import make_run

    from dew.interop import export_run

    make_run(tmp_path)
    with pytest.raises(ValueError, match="SimpleDiT has no published layout"):
        export_run(str(tmp_path), tmp_path / "export")


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


def test_push_exports_a_run_directory_and_uploads_that(tmp_path, api):
    """A run directory is Dew's format and nothing on the Hub reads it, so
    the push uploads what `export_run` writes; `raw` uploads the run itself,
    which is the form `from_pretrained` pulls back."""
    from test_inference import make_lm_run

    run = tmp_path / "run"
    run.mkdir()
    make_lm_run(run)

    push_to_hub(run, "acme/lm")

    assert Path(api.uploaded[0]["folder_path"]) != run
    assert api.files[0] == {"config.json", "generation_config.json", "model.safetensors"}

    push_to_hub(run, "acme/lm-raw", raw=True)
    assert api.uploaded[1]["folder_path"] == str(run)


def test_a_block_diffusion_run_exports_under_its_published_config(tmp_path):
    """DiffusionGemma writes the reference's own encoder/decoder names, over
    the published config the run recorded rather than a derived one."""
    from test_inference import make_block_run

    from dew.interop import export_run, load_pretrained

    run = tmp_path / "run"
    run.mkdir()
    make_block_run(run)
    destination = tmp_path / "export"

    export_run(str(run), destination, ema=False)

    reloaded = load_pretrained(destination, dtype="float32", attention_impl="xla", max_seq_len=32)
    task = dew.pipeline(str(run), ema=False)
    # The export writes the layer scalars into the reference's buffers, so
    # the reloaded tree is the source's shape, not the run's; what has to
    # survive is the canvas the two decode.
    wanted = task([[1, 5, 7]], 3, seed=4).host()
    actual = reloaded.block_generation()([[1, 5, 7]], 3, seed=4).host()
    np.testing.assert_array_equal(actual.tokens, wanted.tokens)
    np.testing.assert_array_equal(actual.decoder_steps, wanted.decoder_steps)


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


def test_layer_bytes_reads_the_headers_of_a_checkpoint():
    """The bytes a host-streamed run pins are the numbered layers' tensors;
    the count comes from the shard headers alone, so a launcher can size
    the pinned pool before any backend starts, and it equals what the
    loaded tensors weigh."""
    from dew.interop.safetensors_io import layer_bytes, read_file
    directory = Path(__file__).parent / "fixtures/hf/diffusion-gemma-sft"
    tensors, _ = read_file(directory / "model.safetensors")
    expected = sum(tensor.nbytes for name, tensor in tensors.items() if re.search(r"\.layers\.\d+\.", name))
    assert expected > 0
    assert layer_bytes(directory) == expected


def test_layer_bytes_covers_a_pipeline_component_and_its_index(tmp_path):
    """A pipeline keeps its decoder in a component directory whose shards an
    index names; the count follows the index and ignores files that are not
    the component's weights."""
    from dew.interop.safetensors_io import layer_bytes, write_file
    component = tmp_path / "transformer"
    component.mkdir()
    layer = np.ones((4, 8), np.float32)
    write_file({"model.layers.0.mlp.kernel": layer, "model.embed.weight": np.ones((3, 8), np.float32)},
               component / "model-00001-of-00002.safetensors", {})
    write_file({"model.layers.1.mlp.kernel": layer}, component / "model-00002-of-00002.safetensors", {})
    write_file({"model.layers.9.mlp.kernel": layer}, component / "stray.safetensors", {})
    (component / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.layers.0.mlp.kernel": "model-00001-of-00002.safetensors",
        "model.embed.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.mlp.kernel": "model-00002-of-00002.safetensors"}}))
    assert layer_bytes(tmp_path) == 2 * layer.nbytes
