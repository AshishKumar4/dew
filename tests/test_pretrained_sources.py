"""What a Hub source resolves to before any family reads it.

Each case is a published repo's own files, committed as fixtures at a pinned
revision: its config.json, its index and, where the file selection is the
claim, its Hub listing (`source.json`). No weights are committed; the
recording stand-in for the Hub below serves the metadata the fixture holds
and an empty placeholder for any other file, and records every weight
transfer, which is what the selection tests hold onto.
"""

import json
import shutil
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dew.interop import hf_decoders, pretrained

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
    loaded = []
    monkeypatch.setattr(pretrained, "_load_diffusion_source",
                        lambda directory, index, **kwargs: loaded.append(directory))

    pretrained.load_pretrained(repo)

    assert loaded == [fake.snapshot]
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
        pretrained._source_quantization(fixture_config(name))


@pytest.mark.parametrize("name", [
    "qwen3-8b-fp8",                      # fmt e4m3, qwen3
    "qwen3-30b-a3b-instruct-2507-fp8",   # fmt e4m3, qwen3_moe, modules_to_not_convert
    "qwen3-coder-next-fp8",              # no fmt, qwen3_next
])
def test_a_qwen3_fp8_config_is_read_by_the_codec_and_translates(name):
    """The codec reads quantization_config (128-square blocks, float32
    scales); the family translator does not refuse it as a field it lacks."""
    config = fixture_config(name)

    quantization = pretrained._source_quantization(config)
    record = hf_decoders.translate_config(config)

    assert quantization is not None
    assert record["vocab_size"] == config["vocab_size"]


def test_a_format_the_codec_cannot_read_is_still_refused_on_the_same_config():
    config = fixture_config("qwen3-8b-fp8")
    config["quantization_config"] = {**config["quantization_config"], "quant_method": "awq"}

    with pytest.raises(ValueError, match="quant_method 'awq'"):
        pretrained._source_quantization(config)


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
