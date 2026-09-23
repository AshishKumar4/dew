"""What a Hub source resolves to before any family reads it.

Each case is a published repo's own files, committed as fixtures at a pinned
revision: its config.json, its index and, where the file selection is the
claim, its Hub listing (`source.json`). No weights are committed; the
recording stand-in for the Hub below serves the metadata the fixture holds
and an empty placeholder for any other file, and records every weight
transfer, which is what the selection tests hold onto.
"""

import json
import os
import shutil
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from dew.interop import codecs, hf_decoders, pretrained

safetensors_numpy = pytest.importorskip("safetensors.numpy")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

# What a load fetches besides weights: configs, indexes, tokenizer files.
METADATA = ("*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja")


def fixture_config(name):
    return json.loads((FIXTURES / name / "config.json").read_text())


class FakeHub:
    """`huggingface_hub.snapshot_download` over one fixture's listing.

    Metadata files come from the fixture and every other file is an empty
    placeholder; `fetched` records each weights file a call transferred.
    """

    def __init__(self, fixture: Path, root: Path):
        source = json.loads((fixture / "source.json").read_text())
        self.fixture, self.files, self.commit = fixture, source["files"], source["revision"]
        self.snapshot = root / "snapshots" / self.commit
        self.fetched: list[str] = []

    def __call__(self, repo_id, *, revision=None, allow_patterns=None, dry_run=False):
        chosen = [name for name in self.files
                  if allow_patterns is None or any(fnmatch(name, pattern) for pattern in allow_patterns)]
        if dry_run:
            return [SimpleNamespace(filename=name) for name in chosen]
        for name in chosen:
            target = self.snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if (self.fixture / name).is_file():
                shutil.copyfile(self.fixture / name, target)
            else:
                target.write_bytes(b"")
            if not any(fnmatch(name, pattern) for pattern in METADATA):
                self.fetched.append(name)
        return str(self.snapshot)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    import huggingface_hub

    def serve(name):
        fake = FakeHub(FIXTURES / name, tmp_path / name)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
        return fake

    return serve


@pytest.mark.parametrize("name, repo", [
    ("mistral-7b-instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("mamba-codestral-7b", "mistralai/Mamba-Codestral-7B-v0.1"),
])
def test_a_decoder_downloads_only_the_shards_its_index_names(hub, name, repo):
    """Both repos also ship Mistral's consolidated.safetensors, the same
    weights under mistral-inference names; a glob fetched it (twice the
    bytes) and then failed on `unknown tensor name`."""
    fake = hub(name)

    directory = hf_decoders._snapshot(repo, None)

    assert directory == fake.snapshot
    assert sorted(fake.fetched) == [f"model-0000{shard}-of-00003.safetensors" for shard in (1, 2, 3)]


@pytest.mark.parametrize("name, repo, components", [
    ("sdxl-base-1.0", "stabilityai/stable-diffusion-xl-base-1.0",
     ("text_encoder", "text_encoder_2", "unet", "vae")),
    ("sd-1.5", "stable-diffusion-v1-5/stable-diffusion-v1-5",
     ("safety_checker", "text_encoder", "unet", "vae")),
])
def test_a_pipeline_downloads_only_the_component_files_it_reads(hub, monkeypatch, name, repo, components):
    """No fp16 or non-EMA variant, no root single-file checkpoint, no ONNX or
    OpenVINO copy, and no folder model_index.json does not declare
    (SDXL's vae_1_0): the non-variant weights of each declared component."""
    fake = hub(name)
    monkeypatch.setattr(pretrained, "_load_diffusion_source", lambda directory, index, **kwargs:
                        pretrained.Pretrained(None, {}, None, index, directory, {}))

    loaded = pretrained.load_pretrained(repo)

    assert loaded.source == fake.snapshot and loaded.revision == fake.commit
    assert sorted(fake.fetched) == sorted(
        f"{component}/{'model' if component.startswith(('text', 'safety')) else 'diffusion_pytorch_model'}"
        ".safetensors" for component in components)


def write_index(directory, weight_map, stem="model"):
    (directory / f"{stem}.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_a_local_checkpoint_reads_the_indexed_shards_and_not_a_consolidated_copy(tmp_path):
    one, two = np.ones((2, 2), np.float32), np.zeros((2,), np.float32)
    safetensors_numpy.save_file({"model.embed_tokens.weight": one}, str(tmp_path / "model-00001-of-00002.safetensors"))
    safetensors_numpy.save_file({"model.norm.weight": two}, str(tmp_path / "model-00002-of-00002.safetensors"))
    safetensors_numpy.save_file({"tok_embeddings.weight": one}, str(tmp_path / "consolidated.safetensors"))
    write_index(tmp_path, {"model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                           "model.norm.weight": "model-00002-of-00002.safetensors"})

    tensors = hf_decoders._load_shards(tmp_path)

    assert sorted(tensors) == ["model.embed_tokens.weight", "model.norm.weight"]


def test_a_tensor_in_two_shards_is_refused_by_name(tmp_path):
    safetensors_numpy.save_file({"model.norm.weight": np.ones(2, np.float32)}, str(tmp_path / "a.safetensors"))
    safetensors_numpy.save_file({"model.norm.weight": np.zeros(2, np.float32)}, str(tmp_path / "b.safetensors"))
    write_index(tmp_path, {"model.norm.weight": "a.safetensors", "model.embed_tokens.weight": "b.safetensors"})

    with pytest.raises(ValueError, match="'model.norm.weight' is stored in both a.safetensors and b.safetensors"):
        hf_decoders._load_shards(tmp_path)


def test_a_text_encoder_reads_its_weights_and_not_the_fp16_variant_beside_them(tmp_path):
    """A pipeline's text_encoder folder holds model.safetensors and
    model.fp16.safetensors, the same names at two precisions."""
    from dew.nn import text_encoders

    weight = np.arange(4, dtype=np.float32)
    safetensors_numpy.save_file({"text_model.final_layer_norm.weight": weight}, str(tmp_path / "model.safetensors"))
    safetensors_numpy.save_file({"text_model.final_layer_norm.weight": weight.astype(np.float16)},
                                str(tmp_path / "model.fp16.safetensors"))

    tensors = text_encoders._read_tensors(tmp_path)

    assert tensors["text_model.final_layer_norm.weight"].dtype == np.float32


def test_a_sharded_vae_reads_through_its_index(tmp_path):
    from dew.nn.autoencoders.vae import load_pretrained_vae

    (tmp_path / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKL"}))
    safetensors_numpy.save_file({"encoder.conv_in.bias": np.ones(2, np.float32)},
                                str(tmp_path / "diffusion_pytorch_model-00001-of-00002.safetensors"))
    safetensors_numpy.save_file({"decoder.conv_out.bias": np.zeros(3, np.float32)},
                                str(tmp_path / "diffusion_pytorch_model-00002-of-00002.safetensors"))
    write_index(tmp_path, {"encoder.conv_in.bias": "diffusion_pytorch_model-00001-of-00002.safetensors",
                           "decoder.conv_out.bias": "diffusion_pytorch_model-00002-of-00002.safetensors"},
                stem="diffusion_pytorch_model")

    params = load_pretrained_vae(str(tmp_path))["params"]

    assert params["encoder"]["conv_in"]["bias"].shape == (2,)
    assert params["decoder"]["conv_out"]["bias"].shape == (3,)


class Discussion(SimpleNamespace):
    pass


@pytest.fixture
def conversion(monkeypatch):
    """SFconvertbot's pull request on state-spaces/mamba2-130m, as the Hub API
    lists it: refs/pr/1, whose parent is the main commit 3a5aea0c."""
    import huggingface_hub

    parents = {"refs/pr/1": "3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49"}

    def discussions(self, repo_id, **kwargs):
        assert kwargs["author"] == "SFconvertbot"
        return [Discussion(title="Adding `safetensors` variant of this model", git_reference="refs/pr/1")]

    def commits(self, repo_id, *, revision):
        return [SimpleNamespace(commit_id="ea6060f6"), SimpleNamespace(commit_id=parents[revision])]

    monkeypatch.setattr(huggingface_hub.HfApi, "get_repo_discussions", discussions)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_commits", commits)
    return parents


def test_a_pickle_repo_names_the_safetensors_conversion_that_loads(hub, conversion):
    hub("mamba2-130m-ssm")

    with pytest.raises(FileNotFoundError, match=r"pytorch_model\.bin.*revision='refs/pr/1'"):
        pretrained.load_pretrained("state-spaces/mamba2-130m")


def test_a_conversion_of_another_commit_is_not_offered(hub, conversion):
    """transformers' rule: the pull request's parent is the commit loaded."""
    hub("mamba2-130m-ssm")
    conversion["refs/pr/1"] = "0" * 40

    with pytest.raises(FileNotFoundError, match="convert them to safetensors") as error:
        pretrained.load_pretrained("state-spaces/mamba2-130m")
    assert "refs/pr/1" not in str(error.value)


@pytest.mark.network
def test_the_hub_lists_the_mamba2_130m_conversion():
    with pytest.raises(FileNotFoundError, match=r"revision='refs/pr/1'"):
        hf_decoders._snapshot("state-spaces/mamba2-130m", "3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49")


def test_a_gguf_repo_is_named_as_gguf_before_any_weight_downloads(hub):
    fake = hub("qwen3-4b-gguf")

    with pytest.raises(FileNotFoundError, match=r"GGUF files \(Qwen3-4B-BF16\.gguf, .*base_model"):
        pretrained.load_pretrained("unsloth/Qwen3-4B-GGUF")
    assert fake.fetched == []


@pytest.mark.parametrize("name, stated", [
    ("llama-3.1-8b-instruct-mlx-4bit", r"4-bit weights in groups of 64"),
    ("bonsai-27b-mlx-1bit", r"1-bit weights in groups of 128"),
])
def test_mlx_quantization_is_refused_by_name(name, stated):
    with pytest.raises(ValueError, match=f"is MLX quantization \\({stated}\\)"):
        codecs.source_quantization(fixture_config(name))


@pytest.mark.parametrize("name", [
    "qwen3-8b-fp8",                      # fmt e4m3, qwen3
    "qwen3-30b-a3b-instruct-2507-fp8",   # fmt e4m3, qwen3_moe, modules_to_not_convert
    "qwen3-coder-next-fp8",              # no fmt, qwen3_next
])
def test_a_qwen3_fp8_config_is_read_by_the_codec_and_translates(name):
    """The codec reads quantization_config (128-square blocks, float32
    scales); the family translator does not refuse it as a field it lacks."""
    config = fixture_config(name)

    quantization = codecs.source_quantization(config)
    record = hf_decoders.translate_config(config)

    assert quantization is not None
    assert record["vocab_size"] == config["vocab_size"]


def test_a_format_the_codec_cannot_read_is_still_refused_on_the_same_config():
    config = fixture_config("qwen3-8b-fp8")
    config["quantization_config"] = {**config["quantization_config"], "quant_method": "awq"}

    with pytest.raises(ValueError, match="quant_method 'awq'"):
        codecs.source_quantization(config)


@pytest.mark.parametrize("name, inert", [
    ("qwen2.5-0.5b", {"use_mrope"}),
    ("smollm2-135m-instruct", {"transformers.js_config", "is_llama_config", "rope_interleaved"}),
    # Bare `Infinity` in time_step_limit, plus rms_norm.
    ("mamba2-130m-hf", {"rms_norm"}),
    # Plus the rest of mamba_ssm's fields.
    ("mamba-codestral-7b", {"rms_norm", "norm_before_gate", "intermediate_size",
                            "time_step_init_scheme", "time_step_scale"}),
])
def test_a_config_whose_extra_fields_the_reference_ignores_translates(name, inert):
    config = fixture_config(name)

    hf_decoders.translate_config(config)

    assert inert <= set(config)
    assert hf_decoders._inert(config["model_type"], config) == inert


def test_mamba2s_open_time_step_bound_reads_as_infinity():
    record = hf_decoders.translate_config(fixture_config("mamba2-130m-hf"))

    assert record["mixer"].time_step_limit == (0.0, float("inf"))


@pytest.mark.parametrize("name, field, value", [
    ("qwen2.5-0.5b", "use_mrope", True),
    ("mamba2-130m-hf", "rms_norm", False),
    ("mamba-codestral-7b", "intermediate_size", 4096),
])
def test_an_inert_field_holding_another_model_is_refused_by_name(name, field, value):
    config = {**fixture_config(name), field: value}

    with pytest.raises(ValueError, match=f"^{field}={value!r} is not expressible"):
        hf_decoders.translate_config(config)


def test_a_field_outside_the_inert_policy_is_still_refused_by_name():
    config = {**fixture_config("smollm2-135m-instruct"), "residual_multiplier": 0.22}

    with pytest.raises(ValueError, match=r"config fields \['residual_multiplier'\] is not expressible"):
        hf_decoders.translate_config(config)


def test_every_inert_field_is_one_the_reference_config_class_does_not_declare():
    """The policy's premise, against the installed reference: the family's
    config class (and PreTrainedConfig, for the fields every family shares)
    declares none of them, so transformers keeps them as bare attributes
    that its modeling code never reads."""
    transformers = pytest.importorskip("transformers")
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    def declared(cls):
        return {name for klass in cls.__mro__ for name in getattr(klass, "__annotations__", {})}

    assert {"expand", "time_step_limit"} <= declared(CONFIG_MAPPING["mamba2"])
    for model_type, fields in hf_decoders._INERT_FIELDS.items():
        cls = transformers.PreTrainedConfig if model_type is None else CONFIG_MAPPING[model_type]
        assert not set(fields) & declared(cls), (model_type, set(fields) & declared(cls))


def test_the_r1_0528_qwen3_yarn_is_the_references_table():
    """deepseek-ai/DeepSeek-R1-0528-Qwen3-8B's rope_scaling is YaRN factor 4
    over 32768 positions at base 1e6, plus vLLM's attn_factor, which
    transformers neither validates nor reads. The frequencies and the cos/sin
    scale are the reference's own `ROPE_INIT_FUNCTIONS['yarn']` values."""
    pytest.importorskip("torch")
    from transformers import Qwen3Config
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from dew.nn.rope import YarnScaling, yarn_attention_factor, yarn_inv_freq

    released = fixture_config("deepseek-r1-0528-qwen3-8b")
    record = hf_decoders.translate_config(released)
    expected, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](Qwen3Config.from_dict(released), None)

    scaling = YarnScaling(**record["yarn"])
    assert record["yarn"]["factor"] == 4.0 and record["yarn"]["original_max_position_embeddings"] == 32768
    assert np.max(np.abs(np.asarray(yarn_inv_freq(128, 1e6, scaling)) - expected.numpy())) < 1e-7
    assert yarn_attention_factor(scaling) == pytest.approx(attention_factor)


def bf16_source(tmp_path, **stated):
    """qwen3-tiny with its tensors stored in bfloat16 and `stated` as its dtype fields."""
    source = tmp_path / "source"
    shutil.copytree(FIXTURES / "qwen3-tiny", source)
    tensors = hf_decoders._load_shards(source)
    safetensors_numpy.save_file({name: np.asarray(value, np.float32).astype(ml_dtypes.bfloat16)
                                 for name, value in tensors.items()}, str(source / "model.safetensors"))
    config = {key: value for key, value in fixture_config("qwen3-tiny").items()
              if key not in ("dtype", "torch_dtype")}
    (source / "config.json").write_text(json.dumps({**config, **stated}))
    return source


def leaf_dtypes(variables):
    import jax

    return {np.dtype(leaf.dtype) for leaf in jax.tree.leaves(variables["params"])}


@pytest.mark.parametrize("stated, stored", [
    ({}, ml_dtypes.bfloat16),                       # the first floating tensor's
    ({"dtype": "float16"}, np.float16),             # config.json's, which wins
    ({"torch_dtype": "float32"}, np.float32),       # the pre-5.0 spelling
])
def test_param_dtype_auto_stores_the_checkpoints_dtype(tmp_path, stated, stored):
    """transformers' dtype='auto' rule: config.json's dtype, else the
    dtype of the first floating tensor."""
    loaded = pretrained.load_pretrained(bf16_source(tmp_path, **stated), dtype="float32",
                                        param_dtype="auto", attention_impl="xla")

    assert leaf_dtypes(loaded.variables) == {np.dtype(stored)}


def test_the_pipeline_places_a_source_in_its_own_dtype(tmp_path):
    import dew

    task = dew.pipeline(str(bf16_source(tmp_path)), dtype="float32", param_dtype="auto")

    assert leaf_dtypes(task.variables) == {np.dtype(ml_dtypes.bfloat16)}


@pytest.mark.parametrize("name", ["qwen3-tiny", "llama-tiny", "mistral-tiny", "gemma3-tiny",
                                  "olmo3-yarn-tiny", "mixtral-tiny", "deepseek-v3-tiny"])
def test_a_saved_source_writes_its_config_back_unchanged(tmp_path, name):
    """Qwen/Qwen3-0.6B states max_position_embeddings 40960 and loads at the
    8192-token cache Dew allocates by default; the export's config is the
    source's, so a server reading it (vLLM's --max-model-len) sees 40960 and
    every other field as published."""
    source = tmp_path / "source"
    shutil.copytree(FIXTURES / name, source)
    config = {**fixture_config(name), "max_position_embeddings": 40960}
    (source / "config.json").write_text(json.dumps(config))

    loaded = pretrained.load_pretrained(source, dtype="float32", attention_impl="xla")
    loaded.save(tmp_path / "saved")

    assert loaded.model.max_seq_len == 8192
    assert json.loads((tmp_path / "saved" / "config.json").read_text()) == config


def test_a_cached_repo_id_loads_offline_when_the_commit_lists_files_never_downloaded(tmp_path):
    """HF_HUB_OFFLINE=1 with the source in the Hub cache, as a load that
    fetched only what it reads leaves it: the commit's cached listing names
    a README the snapshot never downloaded. Offline, huggingface_hub's dry
    run raises LocalEntryNotFoundError for that file, and the load reads
    the snapshot it has, pinned or at the cached `main`."""
    import subprocess
    import sys

    commit = "0123456789abcdef0123456789abcdef01234567"
    repo = tmp_path / "hub" / "models--dew--qwen3-tiny"
    shutil.copytree(FIXTURES / "qwen3-tiny", repo / "snapshots" / commit)
    listed = {path.name: {"size": path.stat().st_size, "blob_id": f"{index:040x}"}
              for index, path in enumerate(sorted((repo / "snapshots" / commit).iterdir()))}
    listed["README.md"] = {"size": 1, "blob_id": "f" * 40}
    (repo / "trees").mkdir()
    (repo / "trees" / f"{commit}.json").write_text(json.dumps({"format_version": 1, "files": listed}))
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit)
    script = """
import sys
from dew.interop import load_pretrained
for revision in (None, sys.argv[1]):
    print(load_pretrained("dew/qwen3-tiny", revision=revision, dtype="float32").revision)
"""
    run = subprocess.run([sys.executable, "-c", script, commit], capture_output=True, text=True,
                         env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(tmp_path / "hub"),
                              "JAX_PLATFORMS": "cpu"})
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.split() == [commit, commit]
