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

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

def fixture_config(name):
    return json.loads((FIXTURES / name / "config.json").read_text())


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
    torch_route = source_route(tiny_config("biogpt").to_dict())
    assert isinstance(torch_route, TorchRoute) and torch_route.admitted.model_type == "biogpt"
    llama = fixture_config("llama-tiny")
    for refused in ({**llama, "model_type": "an_unregistered_decoder"},
                    {**llama, "quantization_config": {"quant_method": "mlx", "bits": 4, "group_size": 64,
                                                       "mode": "affine"}},
                    {"model_type": "an_unregistered_decoder", "quantization": {"bits": 4, "group_size": 64}},
                    {**llama, "hidden_act": "a_nonexistent_activation"}):
        with pytest.raises(ValueError):
            source_route(refused)


def test_tier_three_admits_a_causal_lm_alone_and_shrinks_a_config_by_its_own_names():
    """Tier 3 runs an unregistered type's forward only where it is a causal LM: BERT configured as a
    decoder (`is_decoder`) is one, each position reading the tokens before it, and configured as an
    encoder it reads the later tokens too and is refused. The probe reads a config's sizes under its
    own names (OpenAI-GPT's `n_embd`, `n_head`, `n_layer`), so such a config is probed rather than
    refused for want of `hidden_size`."""
    from dew.interop import verify
    from dew.interop.pretrained import TorchRoute, source_route

    decoder = source_route(tiny_config("bert", is_decoder=True).to_dict())
    assert isinstance(decoder, TorchRoute)
    with pytest.raises(ValueError):
        source_route(tiny_config("bert", is_decoder=False).to_dict())
    openai = tiny_config("openai-gpt", hidden_size=None, num_attention_heads=None, num_hidden_layers=None,
                         num_key_value_heads=None, head_dim=None, n_embd=256, n_head=4, n_layer=2).to_dict()
    assert "hidden_size" not in openai and openai["n_embd"] == 256
    shrunk = verify.shrink(openai)
    assert shrunk["n_embd"] < 256 and "hidden_size" not in shrunk
    assert isinstance(verify.verify_mapping(openai), verify.Unmapped)


def test_the_hub_census_routes_a_config_as_the_loader_does(tmp_path):
    """tools/hf_coverage.py classifies a census of configs by the loader's own route (`source_route`),
    with the real verifier: a storage format the codecs refuse is refused before transformers is
    asked about the type, as is one a codec's own reader refuses (GPTQ at three bits); a type
    transformers has no class for, asked only after those, is remote code; a registered family's
    config is tier 1; a type a registered convention computes, verified, is tier 2; and an unregistered
    causal LM none computes is tier 3."""
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
        "biogpt": tiny_config("biogpt").to_dict(),
    }
    rows = [{"id": name, "model_type": config["model_type"], "downloads": len(configs) - index,
             "config": config} for index, (name, config) in enumerate(configs.items())]
    (tmp_path / "census.json").write_text(json.dumps(rows))
    census.classify(tmp_path)
    routes = {model["id"]: model["route"]
              for model in json.loads((tmp_path / "coverage.json").read_text())["models"]}
    assert routes == {"storage": "refused", "gptq": "refused", "custom": "remote code", "llama": "tier 1",
                      "cwm": "tier 2", "biogpt": "tier 3"}



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
