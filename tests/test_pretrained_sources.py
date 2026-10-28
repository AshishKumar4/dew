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
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
import safetensors.numpy as safetensors_numpy
from interop_support import tiny_config

from dew.interop import hf_decoders, sources

# Tier 3's admission traces its forward through torchax (`torchax_fallback.admitted`).
pytestmark = pytest.mark.torchax

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

def fixture_config(name):
    return json.loads((FIXTURES / name / "config.json").read_text())


def test_an_explicit_readout_tensor_file_uses_the_native_safe_weight_reader(tmp_path):
    """A named .pt head is converted by the same safe tensor reader as native checkpoints,
    without requiring the backbone's conventional weight filename."""
    import torch

    from dew.interop.sources import tensor_file

    tensors = {"slot": torch.arange(24, dtype=torch.float32).reshape(3, 8),
               "q.weight": torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16,
               "k.weight": torch.zeros((2, 8))}
    path = tmp_path / "readout_head.pt"
    torch.save(tensors, path)
    restored = tensor_file(path)
    assert set(restored) == set(tensors)
    for name, value in tensors.items():
        np.testing.assert_array_equal(restored[name], value.numpy())


def test_a_saved_processor_record_reloads_its_selected_folder(tmp_path):
    """A processor selected below a repository root survives the task-settings JSON round trip;
    the unrelated root config cannot replace its vocabulary and preprocessing."""
    from token_support import byte_level_tokenizer

    from dew.inference.tasks import TextSettings
    from dew.interop import Pretrained
    from dew.interop.pretrained import load_processor
    from dew.registry import from_record, to_record
    from dew.sampling import Sampling

    shutil.copytree(FIXTURES / "qwen3-tiny", tmp_path / "release")
    byte_level_tokenizer(tmp_path / "release")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "not_the_selected_release"}))
    bundle = Pretrained.load(tmp_path, subfolder="release", dtype="float32")
    settings = TextSettings(seq_len=8, max_new_tokens=2, tokenizer=None,
                            processor=bundle.processor.source, sampling=Sampling())
    restored = from_record(TextSettings, json.loads(json.dumps(to_record(settings, TextSettings))))
    processor = load_processor(restored.processor, bundle.model)
    for text in ("hello", "hello world"):
        np.testing.assert_array_equal(processor(text).tokens, bundle.processor(text).tokens)


"""The commit of SFconvertbot's refs/pr/1 on state-spaces/mamba2-130m."""


def test_the_r1_0528_qwen3_yarn_is_the_references_table():
    """deepseek-ai/DeepSeek-R1-0528-Qwen3-8B's rope_scaling is YaRN factor 4
    over 32768 positions at base 1e6, plus vLLM's attn_factor, which
    transformers neither validates nor reads. The frequencies and the cos/sin
    scale are the reference's own `ROPE_INIT_FUNCTIONS['yarn']` values."""
    from transformers import Qwen3Config
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from dew.nn.rope import YarnScaling, yarn_attention_factor, yarn_inv_freq

    released = fixture_config("deepseek-r1-0528-qwen3-8b")
    record = hf_decoders.translate_config(released)
    expected, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](Qwen3Config.from_dict(released), None)

    scaling = YarnScaling(**record["yarn"])
    assert record["yarn"]["factor"] == 4.0 and record["yarn"]["original_max_position_embeddings"] == 32768
    assert (
        np.max(np.abs(np.asarray(yarn_inv_freq(128, 1e6, scaling, dtype=np.float32)) - expected.numpy()))
        < 1e-7
    )
    assert yarn_attention_factor(scaling) == pytest.approx(attention_factor)


def bf16_source(tmp_path, **stated):
    """qwen3-tiny with its tensors stored in bfloat16 and `stated` as its dtype fields."""
    source = tmp_path / "source"
    shutil.copytree(FIXTURES / "qwen3-tiny", source)
    tensors = sources.load_shards(source)
    safetensors_numpy.save_file({name: np.asarray(value, np.float32).astype(ml_dtypes.bfloat16)
                                 for name, value in tensors.items()}, str(source / "model.safetensors"))
    config = {key: value for key, value in fixture_config("qwen3-tiny").items()
              if key not in ("dtype", "torch_dtype")}
    (source / "config.json").write_text(json.dumps({**config, **stated}))
    return source


def test_pipeline_preserves_a_bf16_residual_before_its_norm(tmp_path):
    """A fresh inference process observes the same stored BF16 sum as training."""
    import subprocess
    import sys

    source = bf16_source(tmp_path)
    script = """
import sys
import dew
task = dew.pipeline(sys.argv[1], dtype="bfloat16", param_dtype="auto")
import jax
import jax.numpy as jnp
import numpy as np
from dew.nn.attention import RMSNorm
a = jax.random.normal(jax.random.key(0), (512, 64), jnp.bfloat16)
b = (jax.random.normal(jax.random.key(1), (512, 64)) * 0.37).astype(jnp.bfloat16)
norm = RMSNorm(epsilon=1e-5, scale_after_cast=True, dtype=jnp.bfloat16)
variables = norm.init(jax.random.key(2), a)
stored = jax.jit(jnp.add)(a, b)
expected = jax.jit(norm.apply)(variables, stored)
actual = jax.jit(lambda a, b: norm.apply(variables, a + b))(a, b)
np.testing.assert_array_equal(actual, expected)
"""
    flags = " ".join(flag for flag in os.environ.get("XLA_FLAGS", "").split()
                     if not flag.startswith("--xla_allow_excess_precision"))
    result = subprocess.run([sys.executable, "-c", script, str(source)], capture_output=True, text=True,
                            env={**os.environ, "XLA_FLAGS": flags, "JAX_PLATFORMS": "cpu"}, timeout=90)
    assert result.returncode == 0, result.stderr[-4000:]


@pytest.mark.parametrize("entry", ["typed", "trainer"])
def test_import_policy_covers_typed_loading_and_plain_trainer(tmp_path, entry):
    """Real callers that do not run pipeline/prepare_process keep declared BF16 sums."""
    import subprocess
    import sys

    source = bf16_source(tmp_path)
    script = """
import sys
import dew
import jax
import jax.numpy as jnp
import numpy as np
if sys.argv[2] == "typed":
    from dew.interop.pretrained import Pretrained
    from dew.sampling import Sampling
    task = Pretrained.load(sys.argv[1], dtype="bfloat16", param_dtype="auto").text_generation(
        sampling=Sampling(temperature=0, eos_token_ids=None))
    generated = task(np.asarray([[1,2,3]], np.int32), 3, key=0).host()
    assert int(generated.lengths[0]) == 3
else:
    import optax
    from dew.objectives import Objective
    from dew.objectives.base import Aux
    from dew.training import Trainer
    class Probe(Objective):
        def init(self, key, variables=None):
            return {"params": {"w": jnp.ones(1)}}
        def loss(self, variables, batch, step):
            return jnp.sum(variables["params"]["w"] ** 2), Aux({})
    trainer = Trainer(Probe(), optax.sgd(0.1), key=0)
    trainer.initial_state()
from dew.nn.attention import RMSNorm
a = jax.random.normal(jax.random.key(0), (512,64), jnp.bfloat16)
b = (jax.random.normal(jax.random.key(1), (512,64)) * 0.37).astype(jnp.bfloat16)
norm = RMSNorm(epsilon=1e-5, scale_after_cast=True, dtype=jnp.bfloat16)
variables = norm.init(jax.random.key(2), a)
stored = jax.jit(jnp.add)(a,b)
expected = jax.jit(norm.apply)(variables,stored)
actual = jax.jit(lambda a,b: norm.apply(variables,a+b))(a,b)
np.testing.assert_array_equal(actual,expected)
"""
    flags = " ".join(flag for flag in os.environ.get("XLA_FLAGS", "").split()
                     if not flag.startswith("--xla_allow_excess_precision"))
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), entry],
        capture_output=True,
        text=True,
        env={**os.environ, "XLA_FLAGS": flags, "JAX_PLATFORMS": "cpu"},
        timeout=90,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def tier_two_config():
    """A tiny config of a model_type no registered family is named for that a registered family's
    convention computes (CWM, Llama's): the verifier checks it against transformers' own class."""
    return tiny_config("cwm", sliding_window=4).to_dict()


def test_a_source_is_routed_on_its_config_alone_as_the_loader_reads_it():
    """`source_route` is the loader's decision before any weight downloads, which a census of configs
    reads too: a registered family's config is its decoder, translated; a media wrapper's is the
    wrapper; DiffusionGemma's its own; an unregistered model_type a registered family's convention
    computes is verified against transformers (tier 2), an unregistered causal LM none computes runs
    as transformers' own forward (tier 3), and one transformers has no class for is refused, as are a
    storage format the codecs cannot decode and a family's refusal."""
    from dew.interop.pretrained import (
        DecoderRoute,
        DiffusionGemmaRoute,
        TorchRoute,
        VerifiedRoute,
        WrapperRoute,
        source_route,
    )

    route = source_route(fixture_config("llama-tiny"))
    assert isinstance(route, DecoderRoute)
    assert route.record == hf_decoders.translate_config(fixture_config("llama-tiny"))
    assert isinstance(source_route(fixture_config("gemma3-tiny-mm")), WrapperRoute)
    assert isinstance(source_route(fixture_config("diffusion-gemma-sft")), DiffusionGemmaRoute)
    verified = source_route(tier_two_config())
    assert isinstance(verified, VerifiedRoute) and verified.verified.error <= verified.verified.bound
    torch_route = source_route(tiny_config("cohere").to_dict())
    assert isinstance(torch_route, TorchRoute) and torch_route.admitted.model_type == "cohere"
    llama = fixture_config("llama-tiny")
    for refused in ({**llama, "model_type": "an_unregistered_decoder"},
                    {**llama, "quantization_config": {"quant_method": "mlx", "bits": 4, "group_size": 64,
                                                       "mode": "affine"}},
                    {"model_type": "an_unregistered_decoder", "quantization": {"bits": 4, "group_size": 64}},
                    {**llama, "hidden_act": "a_nonexistent_activation"}):
        with pytest.raises(ValueError):
            source_route(refused)


def test_tier_three_admits_a_causal_lm_it_can_train_and_refuses_the_rest():
    """Tier 3 runs an unregistered type's forward only where it is a causal LM tier 3 can train, each
    refusal its own: BERT configured as a decoder (`is_decoder`) is one, each position reading the
    tokens before it, and configured as an encoder it reads the later tokens too and is refused; a
    speculative drafter is no model on its own (`DrafterRefused`), whatever its type; a rotary whose
    tables transformers recomputes from each call's positions (dynamic NTK) is admitted, its tables
    computed in the trace (`functional_rotaries`); BioGPT's
    training forward draws its layer drop on the host, which a traced step would fix; a quantized
    source is refused, tier 3 reading float weights; and one whose shrunken model does not run at the
    probe's length (a BERT decoder of 8 positions) is `ProbeUnavailable`, not a model none computes."""
    from dew.interop import decoder_parts, verify
    from dew.interop.pretrained import TorchRoute, source_route

    assert isinstance(source_route(tiny_config("bert", is_decoder=True).to_dict()), TorchRoute)
    dynamic = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": 2.0}
    assert isinstance(source_route(tiny_config("cohere", rope_parameters=dynamic).to_dict()), TorchRoute)
    # Longcat Flash routes its top 12 of routed and identity experts; the probe shrinks all three
    # under the config's own keys (`moe_topk`), so its shrunken router routes within its experts.
    longcat = tiny_config("longcat_flash", n_routed_experts=16, zero_expert_num=8, moe_topk=12,
                          num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=8,
                          qk_rope_head_dim=8, head_dim=8, v_head_dim=8, expert_ffn_hidden_size=32,
                          num_layers=1, num_hidden_layers=2)
    assert isinstance(source_route(longcat.to_dict()), TorchRoute)
    cohere = tiny_config("cohere").to_dict()
    for refused in (tiny_config("bert", is_decoder=False).to_dict(),
                    tiny_config("biogpt").to_dict(),
                    {**cohere, "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 64,
                                                       "desc_act": False, "sym": True}}):
        with pytest.raises(ValueError):
            source_route(refused)
    with pytest.raises(decoder_parts.DrafterRefused):
        source_route({**cohere, "num_target_layers": 4})
    with pytest.raises(verify.ProbeUnavailable):
        source_route(tiny_config("bert", is_decoder=True, max_position_embeddings=8).to_dict())


def test_tier_three_runs_dews_attention_where_every_layer_asks_the_interface_for_it():
    """Tier 3 observes, on the probe's model, whether every decoder layer asks transformers' attention
    interface for its attention with a mask Dew's visibility reproduces: StableLM's do, though its
    class declares no attention backend, and run on Dew's; OpenAI GPT's and GPT-NeoX Japanese's compute
    their own, and keep them, saying why."""
    from dew.interop.pretrained import source_route

    stablelm = source_route(tiny_config("stablelm").to_dict()).admitted
    assert stablelm.attends and not stablelm.why
    openai = source_route(tiny_config("openai-gpt").to_dict()).admitted
    assert not openai.attends and openai.why
    # GPT-NeoX Japanese adds the mask it is given to its scores in its own code: the probe's mask is
    # transformers' eager one, so its attention stays causal and the type is admitted on its own.
    japanese = source_route(tiny_config("gpt_neox_japanese").to_dict()).admitted
    assert not japanese.attends and japanese.why


@pytest.mark.parametrize("spelled", ["its own names", "the standard names"])
def test_the_probe_shrinks_a_config_by_whichever_names_its_class_reads(spelled):
    """OpenAI-GPT's class reads its width, heads and depth as `n_embd`, `n_head` and `n_layer`, and the
    standard names as aliases of them: a config spelled either way shrinks to a small model, the
    transformers config built from it as small, and is probed rather than refused for want of a name."""
    from transformers import AutoConfig

    from dew.interop import verify

    sizes = ({"n_embd": 256, "n_head": 4, "n_layer": 2} if spelled == "its own names"
             else {"hidden_size": 256, "num_attention_heads": 4, "num_hidden_layers": 2})
    config = {**{key: value for key, value in tiny_config("openai-gpt").to_dict().items()
                 if key not in ("n_embd", "n_head", "n_layer", "hidden_size", "num_attention_heads",
                                "num_hidden_layers", "num_key_value_heads", "head_dim")}, **sizes}
    shrunk = verify.shrink(config)
    rebuilt = AutoConfig.for_model("openai-gpt", **{key: value for key, value in shrunk.items()
                                                    if key != "model_type"})
    assert rebuilt.hidden_size < 256 and rebuilt.num_hidden_layers == 2
    assert isinstance(verify.verify_mapping(config), verify.Unmapped)


def test_the_hub_census_routes_a_config_as_the_loader_does(tmp_path):
    """tools/hf_coverage.py classifies a census of configs by the loader's own route (`source_route`),
    with the real verifier: a storage format the codecs refuse is refused before transformers is
    asked about the type, as is one a codec's own reader refuses (GPTQ at three bits); a type
    transformers has no class for, asked only after those, is remote code; a registered family's
    config is tier 1; a type a registered convention computes, verified, is tier 2; an unregistered
    causal LM none computes is tier 3; and each configuration of a type is its own verdict, in either
    order (BERT as a decoder is tier 3, as an encoder refused; Cohere asking for an attention backend
    transformers has none of is refused), every route counted in the summary."""
    from tool_support import load

    census = load("hf_coverage")
    llama = fixture_config("llama-tiny")
    configs = {
        "storage": {"model_type": "a_custom_decoder", "quantization": {"bits": 4, "group_size": 64}},
        "gptq": {"model_type": "a_custom_decoder",
                 "quantization_config": {"quant_method": "gptq", "bits": 3}},
        "custom": {**llama, "model_type": "a_custom_decoder"},
        "llama": llama,
        "cwm": tier_two_config(),
        "cohere": tiny_config("cohere").to_dict(),
        "bert decoder": tiny_config("bert", is_decoder=True).to_dict(),
        "bert encoder": tiny_config("bert", is_decoder=False).to_dict(),
        "cohere, no backend": {**tiny_config("cohere").to_dict(), "attn_implementation": "no_such_backend"},
    }
    expected = {"storage": "refused", "gptq": "refused", "custom": "remote code", "llama": "tier 1",
                "cwm": "tier 2", "cohere": "tier 3", "bert decoder": "tier 3", "bert encoder": "refused",
                "cohere, no backend": "refused"}
    for ordered in (list(configs), list(reversed(configs))):
        rows = [{"id": name, "model_type": configs[name]["model_type"], "downloads": len(configs) - index,
                 "config": configs[name]} for index, name in enumerate(ordered)]
        (tmp_path / "census.json").write_text(json.dumps(rows))
        census.classify(tmp_path)
        coverage = json.loads((tmp_path / "coverage.json").read_text())
        assert {model["id"]: model["route"] for model in coverage["models"]} == expected
        shares = coverage["summary"]["all types"]
        assert sum(entry["models"] for entry in shares.values()) == len(configs)
        assert shares["tier 3"]["models"] == 2



def test_a_tensor_two_shards_both_hold_is_refused_by_both_readers(tmp_path):
    """An index whose two shards both store `a`: reading the weights and reading their headers each
    refuse it, where a merge would keep whichever came last."""
    from dew.interop.safetensors_io import read_weights

    first, second = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    safetensors_numpy.save_file({"a": np.zeros(2, np.float32), "b": np.ones(2, np.float32)},
                                str(tmp_path / first))
    safetensors_numpy.save_file({"a": np.full(2, 2.0, np.float32)}, str(tmp_path / second))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {"a": second, "b": first}}))
    stored = sources.StoredWeights(str(tmp_path), tmp_path)
    for read in (lambda: read_weights(tmp_path), lambda: stored.specs("")):
        with pytest.raises(ValueError):
            read()


_CONFIG_ONLY = """
import sys
import jax
from dew.interop import Pretrained
loaded = Pretrained.load(sys.argv[1], revision=sys.argv[2], weights=False, dtype="bfloat16")
leaves = jax.tree.leaves(loaded.variables)
assert leaves and all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in leaves)
print(sum(leaf.size for leaf in leaves))
"""


@pytest.mark.network
def test_a_hub_decoder_loaded_without_its_weights_downloads_none(tmp_path):
    """`Pretrained.load(weights=False)` of a Hub decoder reads its config and its weights' safetensors
    headers over HTTP, none of its weights downloaded, so a config-only check of a large base runs
    on a small disk (ledger 1888): a fresh cache holds no weight file after it, and its variables are
    the shapes of SmolLM2-135M's parameters."""
    import subprocess
    import sys

    repo, revision = "HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "cache"), "PYTHONPATH": src,
           "DEW_HUB_ANSWERS": str(tmp_path / "answers")}
    done = subprocess.run([sys.executable, "-c", _CONFIG_ONLY, repo, revision], env=env, check=False,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
    assert 130_000_000 < int(done.stdout.split()[-1]) < 140_000_000
    assert not [path for path in (tmp_path / "cache").rglob("*") if path.suffix in (".safetensors", ".bin")]


def test_a_runs_base_loads_without_its_weights_under_the_runs_source_options():
    """A run's source options (`SourceOptions.keywords`) and `weights=False` load the base as the run
    builds it, its variables the shapes and dtypes a load with its weights builds under the same
    options: a wrapper's language model alone, its first prediction depth, its full attention reading
    blocks (ledger 1888)."""
    import jax

    from dew.config import SourceOptions
    from dew.interop import Pretrained

    base = Path(__file__).resolve().parent / "fixtures" / "hf" / "qwen35-tiny-mm"
    options = SourceOptions(text_only=True, prediction_depths=0,
                            attention={"full_attention": "block_bidirectional"})
    shapes = Pretrained.load(base, dtype="float32", weights=False, **options.keywords())
    loaded = Pretrained.load(base, dtype="float32", **options.keywords())
    assert shapes.model == loaded.model and shapes.config == loaded.config
    assert jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), shapes.variables) == jax.tree.map(
        lambda leaf: (leaf.shape, leaf.dtype), loaded.variables)
    assert shapes.model.kind_of("full_attention").mixer.bidirectional_blocks


WITHOUT_WEIGHTS = ("hf/gpt2-tiny", "hf/gpt-neox-tiny", "hf/bloom-tiny", "hf/falcon-tiny", "hf/phi3-tiny",
                   "hf/mamba2-tiny", "hf/nemotron-h-tiny", "hf/kimi-linear-tiny", "hf/deepseek-v4-tiny",
                   "hf/gpt-oss-tiny", "hf/minimax-m2-fp8-tiny", "hf/qwen35-tiny-mm",
                   "hf/gemma4-unified-native-tiny", "codecs/modelopt/tiny", "codecs/modelopt/mixed",
                   "codecs/nvfp4_qdq/e4m3", "codecs/storage/fp8", "codecs/storage/nvfp4")
"""Sources whose readers each compute on stored values a load without its weights does not read: legacy
buffers (GPT-2, GPT-NeoX), packed and split projections, a prepare hook (Kimi Linear, DeepSeek V4),
media towers and projectors (Qwen 3.5, Gemma 4 Unified), and each decoded format: MXFP4 (GPT OSS), block
FP8 (MiniMax-M2), ModelOpt NVFP4 and FP8 with input scales, compressed-tensors NVFP4 and storage."""


def test_a_tensor_a_load_without_its_weights_holds_has_shapes_and_no_values():
    """An `Unread` tensor goes through a reader's NumPy as its shape and dtype (a split, a transpose, a
    cast, a lookup, a decode's arithmetic), and a value asked of it, a branch on one, a Python number,
    an array, raises PayloadUnread, so no reader computes on a value it was not given (ledger 1904)."""
    import numpy as np

    from dew.interop.unread import PayloadUnread, Unread, lookup

    codes = Unread((4, 8), np.uint8)
    halves = np.split(codes.reshape(4, 2, 4).transpose(0, 2, 1), 2, axis=-1)
    assert [half.shape for half in halves] == [(4, 4, 1), (4, 4, 1)]
    decoded = lookup(np.arange(256, dtype=np.float32), codes) * np.float32(0.5)
    assert (decoded.shape, decoded.dtype) == ((4, 8), np.float32)
    assert np.shape(decoded) == (4, 8) and decoded.astype(np.float16).dtype == np.float16
    # NumPy's own dtypes and geometry: an int32 beside a float32 is float64, a weak Python scalar keeps
    # the array's dtype, and a byte view widens the last axis by the item's bytes.
    assert np.add(Unread((2,), np.int32), np.ones(2, np.float32)).dtype == np.float64
    assert (Unread((2,), np.uint8) + 1).dtype == np.uint8
    assert np.concatenate([Unread((2,), np.int32), np.ones(3, np.float32)]).dtype == np.float64
    assert Unread((2, 3), np.float32).view(np.uint8).shape == (2, 12)
    assert Unread((4, 8), np.float32).dot(Unread((8, 2), np.float32)).shape == (4, 2)
    for asked in (lambda: bool(np.array_equal(codes, codes)), lambda: float(decoded[0, 0]),
                  lambda: np.asarray(codes), lambda: np.isfinite(decoded).all() or None,
                  lambda: np.zeros((4, 8)) + np.asarray(decoded)):
        with pytest.raises(PayloadUnread):
            asked()


def test_a_gpt2_release_holding_its_legacy_buffers_loads_without_its_weights(tmp_path):
    """The original GPT-2 release stores each block's causal mask and mask sentinel, which a load checks
    and drops: without its weights the load holds their shapes alone, so it checks those, not values
    it was not given (a mask of fabricated zeros is no causal mask), and builds what a load with them
    builds (ledger 1918)."""
    import shutil

    import jax

    from dew.interop import Pretrained

    directory = tmp_path / "gpt2"
    shutil.copytree(Path(__file__).resolve().parent / "fixtures" / "hf" / "gpt2-tiny", directory)
    stored = safetensors_numpy.load_file(directory / "model.safetensors")
    positions = json.loads((directory / "config.json").read_text())["n_positions"]
    for block in range(2):
        stored[f"transformer.h.{block}.attn.bias"] = np.tril(np.ones((positions, positions), np.float32))[
            None, None]
        stored[f"transformer.h.{block}.attn.masked_bias"] = np.asarray(-1e4, np.float32)
    safetensors_numpy.save_file(stored, directory / "model.safetensors", metadata={"format": "pt"})
    shapes = Pretrained.load(directory, dtype="float32", weights=False)
    loaded = Pretrained.load(directory, dtype="float32")
    assert repr(shapes.model) == repr(loaded.model)
    assert jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), shapes.variables) == jax.tree.map(
        lambda leaf: (leaf.shape, leaf.dtype), loaded.variables)


@pytest.mark.parametrize("name", WITHOUT_WEIGHTS)
def test_a_load_without_its_weights_builds_what_a_load_with_them_builds(name):
    """A checkpoint loaded without its weights is the source a load with them reads, its config and
    model the same, its variables the same shapes and dtypes, from the stored tensors' shapes alone:
    no reader decodes a value it was not given (GPT-2's legacy causal buffers, a quantized release's
    weights and scales), as none is read (ledger 1918, 1904)."""
    import jax

    from dew.interop import Pretrained

    directory = Path(__file__).resolve().parent / "fixtures" / name
    shapes = Pretrained.load(directory, dtype="float32", weights=False)
    loaded = Pretrained.load(directory, dtype="float32")
    # The model's printed fields, which hold a quantized release's scales as arrays that compare by value.
    assert shapes.config == loaded.config and repr(shapes.model) == repr(loaded.model)
    assert jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), shapes.variables) == jax.tree.map(
        lambda leaf: (leaf.shape, leaf.dtype), loaded.variables)
