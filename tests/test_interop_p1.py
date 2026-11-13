"""Behavioural proofs for interop metadata-only dispatch (`weights=False` across directory, DDUF,
single-file, FLUX.2 latent statistics, composite and `pack-quantized` compressed-tensors sources),
beam override admission, unified generation-config controls, and `medias` processor path/byte
atomicity."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from dew.interop import Pretrained, PretrainedDecoder, PretrainedPipeline, codecs
from dew.interop.generation_config import generation_config_of, source_decoding
from dew.interop.safetensors_io import read_file, write_file
from dew.nn.autoencoders.api import AutoEncoder
from dew.registry import to_record
from dew.sampling import Beam, Sampling

FIXTURES = Path(__file__).resolve().parent / "fixtures"
KIMI_K25 = FIXTURES / "hf" / "kimi-k25-tiny"
QWEN_TINY = FIXTURES / "hf" / "qwen3-tiny"
JOINT_TINY = FIXTURES / "joint_checkpoint" / "tiny"
SINGLE_FILE = FIXTURES / "single_file" / "sd"


PROBE = Path(__file__).resolve().parent / "interop_hub_probe.py"


def _files(directory: Path) -> dict[str, Path]:
    """A directory's files by their repo-relative names, as a Hub repo of them serves them."""
    return {path.relative_to(directory).as_posix(): path for path in sorted(directory.rglob("*"))
            if path.is_file()}


def _payloads(path: Path) -> list[tuple[str, int, int]]:
    """Each stored tensor's name and its bytes' range in `path`: a safetensors file's, from its header as
    the format lays it (8 bytes of its length, then the header), and a DDUF archive's safetensors
    entries' at their offsets (huggingface_hub's `read_dduf_file`); none for any other file."""
    from huggingface_hub import read_dduf_file

    if path.suffix == ".dduf":
        places = [entry.offset for name, entry in read_dduf_file(path).items()
                  if name.endswith(".safetensors")]
    elif path.suffix == ".safetensors":
        places = [0]
    else:
        return []
    ranges = []
    with open(path, "rb") as stream:
        for offset in places:
            stream.seek(offset)
            length = int.from_bytes(stream.read(8), "little")
            start = offset + 8 + length
            for tensor, entry in json.loads(stream.read(length)).items():
                if tensor != "__metadata__":
                    begin, end = entry["data_offsets"]
                    ranges.append((tensor, start + begin, start + end))
    return ranges


def _weightless(hub, repo: str, files: dict[str, Path], declared: tuple[str, ...] = ()) -> None:
    """No weight file left the Hub for `repo` whole, and none of a tensor's payload but the `declared`
    metadata tensors' (a sidecar, a latent's statistics) beyond a safetensors file's header read: each
    range it served (`Served.asked`) clear of every other tensor's payload, save one from a safetensors
    file's first byte, which huggingface_hub's public header read (`HfApi.parse_safetensors_file_metadata`)
    asks in one range of its own size. A DDUF archive's every range is clear: its headers and metadata
    entries are read by their exact ranges."""
    for served in hub.served:
        if served.repo != repo or served.file is None or served.sent == 0:
            continue
        payloads = [(start, stop) for tensor, start, stop in _payloads(files[served.file])
                    if not tensor.endswith(declared)]
        if payloads:
            assert served.asked is not None, served
            begin, end = served.asked
            if begin == 0 and served.file.endswith(".safetensors"):
                continue
            assert not any(begin < stop and start < end for start, stop in payloads), served


def _hub_loaded(root: Path, hub, repo: str, name: str, *, offline: bool = False, **request) -> dict:
    """`repo` loaded without its weights from `hub` (None: the Hub gone) in a process of its own
    (`interop_hub_probe.py`), its Hub caches and recorded answers under `root`, offline where asked: the
    probe's answer (`shapes`, or `refused`)."""
    import subprocess
    import sys

    from local_hub import hub_environment

    asked = root / name / "request.json"
    asked.parent.mkdir(parents=True)
    asked.write_text(json.dumps({"repo": repo, **request}))
    done = subprocess.run([sys.executable, str(PROBE), str(asked)],
                          env=hub_environment(hub, root, offline=offline), capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(asked.with_name("answer.json").read_text())


def _shapes_and_dtypes(variables: object) -> object:
    return jax.tree.map(lambda leaf: [list(leaf.shape), str(leaf.dtype)], variables)


def test_pipeline_dduf_single_file_and_composite_loads_without_weights_download_no_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipeline directory, a DDUF archive and a joint checkpoint load without their weights from a Hub
    (on the loopback), each variable's shape and dtype the load with weights holds, and no byte of a
    weight payload leaves the Hub: a safetensors file is read by its header's range, an archive by its
    directory's, its entries' headers' and its metadata entries' (B1). The archive loads again with the
    Hub gone from what the first load recorded, and one it never read is refused."""
    from huggingface_hub import export_folder_as_dduf
    from local_hub import LocalHub

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    root = tmp_path / "hub"

    sd3_root = tmp_path / "sd3"
    with tarfile.open(FIXTURES / "sd3_source.tar.xz") as archive:
        archive.extractall(sd3_root, filter="data")
    sd3_dir = sd3_root / "pipeline"
    expected_pipe = Pretrained.load(sd3_dir, dtype="float32", attention_impl="xla", on_host=True)

    flux_root = tmp_path / "flux"
    with tarfile.open(FIXTURES / "flux_source.tar.xz") as archive:
        archive.extractall(flux_root, filter="data")
    dduf_repo = tmp_path / "dduf_repo"
    dduf_repo.mkdir()
    dduf_path = dduf_repo / "pipeline.dduf"
    export_folder_as_dduf(dduf_path, flux_root / "pipeline")
    expected_dduf = Pretrained.load(flux_root / "pipeline", dtype="float32", attention_impl="xla",
                                    on_host=True)
    local_dduf = Pretrained.load(
        dduf_repo, dduf_file="pipeline.dduf", dtype="float32", attention_impl="xla", weights=False
    )
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(local_dduf.variables))
    assert not [p for p in (tmp_path / "xdg" / "dew" / "dduf").rglob("*") if p.suffix == ".safetensors"]
    assert _shapes_and_dtypes(local_dduf.variables) == _shapes_and_dtypes(expected_dduf.variables)

    expected_joint = Pretrained.load(JOINT_TINY, dtype="float32", attention_impl="xla", on_host=True)
    repos = {"acme/sd3": _files(sd3_dir), "acme/flux-dduf": _files(dduf_repo),
             "acme/joint": _files(JOINT_TINY)}
    options = {"dtype": "float32", "attention_impl": "xla"}
    with LocalHub(repos) as hub:
        loaded_pipe = _hub_loaded(root, hub, "acme/sd3", "sd3", options=options)
        hub_dduf = _hub_loaded(root, hub, "acme/flux-dduf", "dduf",
                               options={**options, "dduf_file": "pipeline.dduf"})
        loaded_joint = _hub_loaded(root, hub, "acme/joint", "joint", options=options)
    for repo in repos:
        _weightless(hub, repo, repos[repo])
    assert loaded_pipe["shapes"] == _shapes_and_dtypes(expected_pipe.variables)
    assert hub_dduf["shapes"] == _shapes_and_dtypes(expected_dduf.variables)
    assert loaded_joint["shapes"] == _shapes_and_dtypes(expected_joint.variables)

    offline_dduf = _hub_loaded(root, None, "acme/flux-dduf", "dduf-offline", offline=True,
                               options={**options, "dduf_file": "pipeline.dduf"})
    assert offline_dduf.get("shapes") == _shapes_and_dtypes(expected_dduf.variables), offline_dduf
    unrecorded = _hub_loaded(root, None, "acme/flux-dduf", "unrecorded", offline=True,
                             options={**options, "dduf_file": "unrecorded.dduf"})
    assert unrecorded["refused"] == "FileNotFoundError"


def test_single_file_loads_without_weights_across_storage_dtypes_and_borrowed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flux_root = tmp_path / "flux"
    with tarfile.open(FIXTURES / "flux_source.tar.xz") as archive:
        archive.extractall(flux_root, filter="data")
    expected_flux = Pretrained.load(
        flux_root / "pipeline", dtype="float32", attention_impl="xla", on_host=True
    )

    sf_repo = tmp_path / "sf_repo"
    shutil.copytree(SINGLE_FILE / "configs", sf_repo)
    sd_tensors, sd_meta = read_file(SINGLE_FILE / "sd.safetensors")
    kw = {"dtype": "float32", "attention_impl": "xla"}
    pos_key = "cond_stage_model.transformer.text_model.embeddings.position_ids"
    pos_ids = np.arange(77, dtype=np.int64)[None, :]
    sd_with_i64 = {**{k: np.asarray(v) for k, v in sd_tensors.items()}, pos_key: pos_ids}
    write_file(sd_with_i64, sf_repo / "sd.safetensors", sd_meta)
    expected_sf = Pretrained.load(sf_repo, single_file="sd.safetensors", on_host=True, **kw)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_sf"))
    local_sf = Pretrained.load(sf_repo, single_file="sd.safetensors", weights=False, **kw)
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(local_sf.variables))
    sf_cache_root = tmp_path / "xdg_sf" / "dew" / "single_file"
    assert not [p for p in sf_cache_root.rglob("*") if p.suffix == ".safetensors"]
    assert _shapes_and_dtypes(local_sf.variables) == _shapes_and_dtypes(expected_sf.variables)


    sharded_flux_repo = tmp_path / "sharded_flux_sf"
    shutil.copytree(flux_root / "pipeline", sharded_flux_repo)
    t5_dir = sharded_flux_repo / "text_encoder_2"
    t5_tensors, t5_meta = read_file(t5_dir / "model.safetensors")
    (t5_dir / "model.safetensors").unlink()
    t5_items = list(t5_tensors.items())
    mid = len(t5_items) // 2
    shard1 = {k: np.asarray(v) for k, v in t5_items[:mid]}
    shard2 = {k: np.asarray(v) for k, v in t5_items[mid:]}
    write_file(shard1, t5_dir / "model-00001-of-00002.safetensors", t5_meta)
    write_file(shard2, t5_dir / "model-00002-of-00002.safetensors", t5_meta)
    weight_map = dict.fromkeys(shard1, "model-00001-of-00002.safetensors") | dict.fromkeys(
        shard2, "model-00002-of-00002.safetensors"
    )
    (t5_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    tr_cfg = json.loads((sharded_flux_repo / "transformer" / "config.json").read_text()) | {
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "axes_dims_rope": [16, 56, 56],
        "num_layers": 1,
        "num_single_layers": 1,
    }
    (sharded_flux_repo / "transformer" / "config.json").write_text(json.dumps(tr_cfg))
    head = int(tr_cfg["attention_head_dim"])
    width = head * int(tr_cfg["num_attention_heads"])
    mlp = 4 * width
    in_ch = int(tr_cfg["in_channels"])
    joint = int(tr_cfg["joint_attention_dim"])
    pooled = int(tr_cfg["pooled_projection_dim"])
    linear_shapes: dict[str, tuple[int, int]] = {
        "time_in.in_layer": (width, 256),
        "time_in.out_layer": (width, width),
        "vector_in.in_layer": (width, pooled),
        "vector_in.out_layer": (width, width),
        "guidance_in.in_layer": (width, 256),
        "guidance_in.out_layer": (width, width),
        "txt_in": (width, joint),
        "img_in": (width, in_ch),
        "final_layer.linear": (in_ch, width),
        "final_layer.adaLN_modulation.1": (2 * width, width),
    }
    norm_names: list[str] = []
    for idx in range(int(tr_cfg["num_single_layers"])):
        linear_shapes[f"single_blocks.{idx}.modulation.lin"] = (3 * width, width)
        linear_shapes[f"single_blocks.{idx}.linear1"] = (3 * width + mlp, width)
        linear_shapes[f"single_blocks.{idx}.linear2"] = (width, width + mlp)
        norm_names += [
            f"single_blocks.{idx}.norm.query_norm.scale",
            f"single_blocks.{idx}.norm.key_norm.scale",
        ]
    for idx in range(int(tr_cfg["num_layers"])):
        for side in ("img", "txt"):
            prefix = f"double_blocks.{idx}.{side}"
            linear_shapes[f"{prefix}_mod.lin"] = (6 * width, width)
            linear_shapes[f"{prefix}_attn.qkv"] = (3 * width, width)
            linear_shapes[f"{prefix}_attn.proj"] = (width, width)
            linear_shapes[f"{prefix}_mlp.0"] = (mlp, width)
            linear_shapes[f"{prefix}_mlp.2"] = (width, mlp)
            norm_names += [f"{prefix}_attn.norm.query_norm.scale", f"{prefix}_attn.norm.key_norm.scale"]
    flux_sf_tensors = (
        {f"{k}.weight": np.zeros(sh, np.float16) for k, sh in linear_shapes.items()}
        | {f"{k}.bias": np.zeros((sh[0],), np.float16) for k, sh in linear_shapes.items()}
        | {k: np.ones((head,), np.float16) for k in norm_names}
    )
    write_file(flux_sf_tensors, sharded_flux_repo / "flux_tr.safetensors", {})
    from local_hub import LocalHub

    repos = {"acme/sd-sf": _files(sf_repo), "acme/flux-sharded-sf": _files(sharded_flux_repo)}
    root = tmp_path / "hub"
    with LocalHub(repos) as hub:
        hub_sf = _hub_loaded(root, hub, "acme/sd-sf", "sd", options={**kw, "single_file": "sd.safetensors"})
        hub_sharded_sf = _hub_loaded(root, hub, "acme/flux-sharded-sf", "flux",
                                     options={**kw, "single_file": "flux_tr.safetensors"})
    for repo in repos:
        _weightless(hub, repo, repos[repo])
    assert hub_sf["shapes"] == _shapes_and_dtypes(expected_sf.variables)
    for part in ("encoders", "autoencoder"):
        assert hub_sharded_sf["shapes"][part] == _shapes_and_dtypes(expected_flux.variables[part])


def test_flux2_metadata_only_load_preserves_latent_statistics_on_restore(
    tmp_path: Path
) -> None:
    flux2_root = tmp_path / "flux2"
    with tarfile.open(FIXTURES / "flux2_source.tar.xz") as archive:
        archive.extractall(flux2_root, filter="data")
    flux2_dir = flux2_root / "pipeline"
    expected = PretrainedPipeline.load(flux2_dir, dtype="float32", attention_impl="xla", on_host=True)
    assert expected.autoencoder is not None

    from local_hub import LocalHub

    from dew.interop.autoencoders.flux2 import LATENT_STATISTICS

    rng = np.random.default_rng(11)
    sample = jnp.asarray(rng.uniform(-1.0, 1.0, (1, 16, 16, 3)).astype(np.float32))
    np.save(tmp_path / "sample.npy", np.asarray(sample))
    vae_params = expected.variables["autoencoder"]
    expected_latent = np.asarray(expected.autoencoder.encode_batch(vae_params, sample))
    repos = {"acme/flux2": _files(flux2_dir)}
    with LocalHub(repos) as hub:
        loaded = _hub_loaded(tmp_path / "hub", hub, "acme/flux2", "flux2", pipeline=True,
                             options={"dtype": "float32", "attention_impl": "xla"},
                             encode={"directory": str(flux2_dir), "sample": str(tmp_path / "sample.npy")})
    _weightless(hub, "acme/flux2", repos["acme/flux2"], declared=LATENT_STATISTICS)
    assert loaded["shapes"] == _shapes_and_dtypes(expected.variables)
    np.testing.assert_array_equal(np.load(tmp_path / "hub" / "flux2" / "latent.npy"), expected_latent)

    local = PretrainedPipeline.load(flux2_dir, dtype="float32", attention_impl="xla", weights=False)
    assert local.autoencoder is not None
    rebound = AutoEncoder.bound(to_record(local.autoencoder, AutoEncoder), params=vae_params)
    np.testing.assert_array_equal(np.asarray(rebound.encode_batch(vae_params, sample)), expected_latent)

    from huggingface_hub import export_folder_as_dduf

    m3_dir = tmp_path / "m3_flux2"
    shutil.copytree(flux2_dir, m3_dir)
    shutil.rmtree(m3_dir / "text_encoder")
    shutil.rmtree(m3_dir / "tokenizer")
    shutil.copytree(flux2_root / "mistral3" / "text_encoder", m3_dir / "text_encoder")
    shutil.copytree(flux2_root / "mistral3" / "tokenizer", m3_dir / "tokenizer")
    tok_cfg = json.loads((m3_dir / "tokenizer" / "tokenizer_config.json").read_text())
    tok_cfg["chat_template"] = (m3_dir / "tokenizer" / "chat_template.jinja").read_text()
    (m3_dir / "tokenizer" / "tokenizer_config.json").write_text(json.dumps(tok_cfg))
    m3_idx = json.loads((m3_dir / "model_index.json").read_text()) | {
        "_class_name": "Flux2Pipeline",
        "text_encoder": ["transformers", "Mistral3ForConditionalGeneration"],
        "tokenizer": ["transformers", "PreTrainedTokenizerFast"],
    }
    (m3_dir / "model_index.json").write_text(json.dumps(m3_idx))
    kw = {"dtype": "float32", "attention_impl": "xla"}
    expected_m3 = PretrainedPipeline.load(m3_dir, on_host=True, **kw)
    flux2_dduf_repo = tmp_path / "flux2_dduf_repo"
    flux2_dduf_repo.mkdir()
    export_folder_as_dduf(flux2_dduf_repo / "flux2.dduf", m3_dir)
    dduf_loaded = PretrainedPipeline.load(flux2_dduf_repo, dduf_file="flux2.dduf", weights=False, **kw)
    assert dduf_loaded.autoencoder is not None
    np.testing.assert_array_equal(
        np.asarray(dduf_loaded.autoencoder.encode_batch(vae_params, sample)), expected_latent
    )
    saved_dir = tmp_path / "flux2_saved"
    dduf_loaded.save(saved_dir, variables=expected_m3.variables)
    saved_te, _ = read_file(saved_dir / "text_encoder" / "model.safetensors")
    assert any("vision" in key for key in saved_te)
    reloaded_saved = PretrainedPipeline.load(saved_dir, on_host=True, **kw)
    assert reloaded_saved.autoencoder is not None
    np.testing.assert_array_equal(
        np.asarray(reloaded_saved.autoencoder.encode_batch(vae_params, sample)), expected_latent
    )
    assert _shapes_and_dtypes(reloaded_saved.variables) == _shapes_and_dtypes(expected_m3.variables)


def test_pack_quantized_compressed_tensors_loads_shapes_without_downloading_weights(
    tmp_path: Path
) -> None:
    ct_dir = tmp_path / "ct_qwen"
    shutil.copytree(QWEN_TINY, ct_dir)
    cfg = json.loads((ct_dir / "config.json").read_text())
    cfg["intermediate_size"] = 36
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "group",
                    "group_size": 4,
                    "dynamic": False,
                },
            }
        },
    }
    (ct_dir / "config.json").write_text(json.dumps(cfg))
    tensors, header_meta = read_file(ct_dir / "model.safetensors")
    rng = np.random.default_rng(3)
    packed_tensors: dict[str, np.ndarray] = {}
    packed_zero_word = np.uint32(0x99999999).view(np.int32)
    for name, raw_tensor in tensors.items():
        dense = np.asarray(raw_tensor)
        if "gate_proj.weight" in name or "up_proj.weight" in name:
            dense = rng.standard_normal((36, dense.shape[1]), dtype=np.float32).astype(np.float16)
        elif "down_proj.weight" in name:
            dense = rng.standard_normal((dense.shape[0], 36), dtype=np.float32).astype(np.float16)
        if "up_proj.weight" in name or "down_proj.weight" in name:
            out_dim, in_dim = dense.shape
            groups = in_dim // 4
            scales = np.full((out_dim, groups), 0.05, dtype=np.float16)
            packed_zeros = np.full(((out_dim + 7) // 8, groups), packed_zero_word, dtype=np.int32)
            grid = {
                name.removesuffix(".weight") + ".weight_scale": scales,
                name.removesuffix(".weight") + ".weight_zero_point": packed_zeros,
                name.removesuffix(".weight") + ".weight_shape": np.asarray([out_dim, in_dim], np.int32),
            }
            quant = codecs.source_quantization(cfg, grid=grid)
            assert quant is not None
            packed_tensors.update(quant.encode(name, dense))
        else:
            packed_tensors[name] = dense
    write_file(packed_tensors, ct_dir / "model.safetensors", header_meta)

    expected = Pretrained.load(ct_dir, dtype="float32", attention_impl="reference", on_host=True)
    local_shapes = Pretrained.load(ct_dir, dtype="float32", attention_impl="reference", weights=False)
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(local_shapes.variables))
    assert _shapes_and_dtypes(local_shapes.variables) == _shapes_and_dtypes(expected.variables)

    from local_hub import LocalHub

    repos = {"acme/ct-qwen": _files(ct_dir)}
    with LocalHub(repos) as hub:
        hub_shapes = _hub_loaded(tmp_path / "hub", hub, "acme/ct-qwen", "ct",
                                 options={"dtype": "float32", "attention_impl": "reference"})
    _weightless(hub, "acme/ct-qwen", repos["acme/ct-qwen"], declared=(".weight_shape",))
    assert hub_shapes["shapes"] == _shapes_and_dtypes(expected.variables)


def test_sampling_override_governs_beam_admission(tmp_path: Path) -> None:
    stochastic_beam_dir = tmp_path / "stochastic_beam"
    shutil.copytree(QWEN_TINY, stochastic_beam_dir)
    (stochastic_beam_dir / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "num_beams": 2, "top_p": 0.9, "eos_token_id": 2, "pad_token_id": 0})
    )
    loaded = PretrainedDecoder.load(stochastic_beam_dir, dtype="float32", attention_impl="reference")
    gen = loaded.text_generation(sampling=Sampling(temperature=0.0))
    assert isinstance(gen.strategy, Beam) and gen.strategy.width == 2

    det_beam_dir = tmp_path / "det_beam"
    shutil.copytree(QWEN_TINY, det_beam_dir)
    (det_beam_dir / "generation_config.json").write_text(
        json.dumps({"do_sample": False, "num_beams": 2, "eos_token_id": 2, "pad_token_id": 0})
    )
    det_loaded = PretrainedDecoder.load(det_beam_dir, dtype="float32", attention_impl="reference")
    with pytest.raises(ValueError):
        det_loaded.text_generation(sampling=Sampling(temperature=0.7))


def test_policy_controls_round_trip_through_generation_config() -> None:
    loaded = Pretrained.load(QWEN_TINY, dtype="float32", attention_impl="reference", weights=False)
    rich = Sampling(
        temperature=0.6,
        top_k=25,
        top_p=0.85,
        min_p=0.05,
        typical_p=0.92,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        min_new_tokens=4,
        stop=("END", "STOP"),
        eos_token_ids=(2, 3),
        pad_token_id=0,
    )
    exported = generation_config_of(rich, max_new_tokens=16)
    restored, transforms, strategy = source_decoding({}, exported, loaded.model, 1, None)
    assert restored == rich and exported["max_new_tokens"] == 16 and transforms is None and strategy is None


def test_medias_processor_keeps_image_paths_and_bytes_atomic(tmp_path: Path) -> None:
    from tool_support import run_from

    source = tmp_path / "kimi"
    shutil.copytree(KIMI_K25, source)
    tiny = json.loads((source / "processor_config.json").read_text())["image_processor"]
    (source / "processor_config.json").unlink()
    with run_from("remote_code_reference", "tools") as reference:
        for file in reference.PROCESSORS["kimi-k25-tiny"][2]:
            shutil.copy(reference.release_file("kimi-k25-tiny", file), source / file)
    released = json.loads((source / "preprocessor_config.json").read_text())
    side = tiny["size"]["max_height"] // tiny["patch_size"]
    released["media_proc_cfg"].update(
        patch_size=tiny["patch_size"],
        merge_kernel_size=tiny["merge_size"],
        in_patch_limit=tiny["max_patches"],
        in_patch_limit_each_frame=None,
        patch_limit_on_one_side=side,
    )
    (source / "preprocessor_config.json").write_text(json.dumps(released))
    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference", trust_remote_code=True)
    assert loaded.processor is not None

    rng = np.random.default_rng(7)
    arr0 = (rng.random((12, 8, 3)) * 255).astype(np.uint8)
    arr1 = (rng.random((8, 16, 3)) * 255).astype(np.uint8)
    path0 = tmp_path / "img0.png"
    path1 = tmp_path / "img1.png"
    Image.fromarray(arr0).save(path0)
    Image.fromarray(arr1).save(path1)
    buf0 = io.BytesIO()
    Image.fromarray(arr0).save(buf0, format="PNG")
    raw_bytes0 = buf0.getvalue()
    buf1 = io.BytesIO()
    Image.fromarray(arr1).save(buf1, format="PNG")
    raw_bytes1 = buf1.getvalue()

    ref_single = loaded.processor("w5 <|media_pad|> w7", images=arr0)
    by_str = loaded.processor("w5 <|media_pad|> w7", images=str(path0))
    by_path = loaded.processor("w5 <|media_pad|> w7", images=path0)
    by_bytes = loaded.processor("w5 <|media_pad|> w7", images=raw_bytes0)
    by_bytearray = loaded.processor("w5 <|media_pad|> w7", images=bytearray(raw_bytes0))
    for candidate in (by_str, by_path, by_bytes, by_bytearray):
        np.testing.assert_array_equal(np.asarray(candidate.tokens), np.asarray(ref_single.tokens))
        np.testing.assert_array_equal(
            np.asarray(candidate.conditioning["pixel_values"]),
            np.asarray(ref_single.conditioning["pixel_values"]),
        )

    rows = ["w5 <|media_pad|> w7", "<|media_pad|> w2"]
    ref_batch = loaded.processor(rows, images=[arr0, arr1])
    by_flat_paths = loaded.processor(rows, images=[str(path0), str(path1)])
    by_flat_bytearrays = loaded.processor(rows, images=[bytearray(raw_bytes0), bytearray(raw_bytes1)])
    for batch_candidate in (by_flat_paths, by_flat_bytearrays):
        np.testing.assert_array_equal(np.asarray(batch_candidate.tokens), np.asarray(ref_batch.tokens))
        np.testing.assert_array_equal(
            np.asarray(batch_candidate.conditioning["pixel_values"]),
            np.asarray(ref_batch.conditioning["pixel_values"]),
        )
