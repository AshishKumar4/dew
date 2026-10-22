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


def test_a_source_is_routed_on_its_config_alone_as_the_loader_reads_it():
    """`source_route` is the loader's decision before any weight downloads, which a census of configs
    reads too: a registered family's config is its decoder, translated; a media wrapper's is the
    wrapper; DiffusionGemma's its own; an unregistered model_type goes to the verifier given; a
    storage format the codecs cannot decode and a family's refusal are refused on the config, as the
    loader refuses them."""
    from dew.interop.codecs import StorageRefused
    from dew.interop.pretrained import source_route

    route = source_route(fixture_config("llama-tiny"))
    assert route.kind == "decoder" and route.record is not None
    assert route.record == hf_decoders.translate_config(fixture_config("llama-tiny"))
    assert source_route(fixture_config("gemma3-tiny-mm")).kind == "wrapper"
    assert source_route(fixture_config("diffusion-gemma-sft")).kind == "diffusion_gemma"
    asked = []

    def verifier(unregistered):
        asked.append(unregistered["model_type"])
        raise ValueError("no registered family's convention computes it")

    llama = fixture_config("llama-tiny")
    with pytest.raises(ValueError):
        source_route({**llama, "model_type": "an_unregistered_decoder"}, verified=verifier)
    assert asked == ["an_unregistered_decoder"]
    with pytest.raises(StorageRefused):
        source_route({**llama, "quantization_config": {"quant_method": "mlx", "bits": 4, "group_size": 64,
                                                        "mode": "affine"}})
    with pytest.raises(StorageRefused):
        # A storage format is refused before the verifier is asked about an unregistered type.
        source_route({"model_type": "an_unregistered_decoder", "quantization": {"bits": 4, "group_size": 64}},
                     verified=verifier)
    assert asked == ["an_unregistered_decoder"]
    with pytest.raises(ValueError):
        source_route({**fixture_config("llama-tiny"), "hidden_act": "a_nonexistent_activation"})

