"""Muse Glimmer's text decoder against transformers 5.16.1's, on a checkpoint built here.

transformers ships no causal-LM class for the text model alone: its logits are
`MuseGlimmerForConditionalGeneration`'s, the text model's final states through
`lm_head`, times `output_multiplier`, under a tanh softcap
(modeling_muse_glimmer.py:1176-1181). So the reference runs `MuseGlimmerTextModel`
in float64, with every weight drawn away from its initial value so the (1 + w)
block norms, the plain final norm and the gate projection all count, and applies
that head itself.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from dew.interop import Pretrained  # noqa: E402

TEXT = {
    "model_type": "muse_glimmer_text", "architectures": ["MuseGlimmerTextModel"], "vocab_size": 256,
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "hidden_activation": "silu", "max_position_embeddings": 256,
    "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8, "qk_scale_factor": 3.87, "sliding_window": 8,
    "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
    "layer_rope_theta": [500000.0] * 3 + [0.0],
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    "final_logit_softcapping": 20.0, "output_multiplier": 0.19611613513818404,
    "tie_word_embeddings": False, "attention_bias": False,
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A tiny text checkpoint, its input ids and the reference's float64 logits."""
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextModel

    torch.manual_seed(0)
    config = MuseGlimmerTextConfig(**{key: value for key, value in TEXT.items()
                                      if key not in ("model_type", "architectures")})
    model = MuseGlimmerTextModel(config).to(torch.float64).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    head = torch.randn(TEXT["vocab_size"], TEXT["hidden_size"], dtype=torch.float64) * 0.1
    ids = torch.randint(0, TEXT["vocab_size"], (2, 24))
    with torch.no_grad():
        states = model(input_ids=ids).last_hidden_state
    softcap = TEXT["final_logit_softcapping"]
    logits = softcap * torch.tanh(states @ head.T * TEXT["output_multiplier"] / softcap)

    directory = tmp_path_factory.mktemp("muse-glimmer-text")
    weights = {f"model.{name}": tensor.to(torch.float32).contiguous()
               for name, tensor in model.state_dict().items()}
    weights["lm_head.weight"] = head.to(torch.float32)
    safetensors_torch.save_file(weights, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(TEXT))
    return directory, ids.numpy(), logits.numpy()


def test_the_text_decoder_computes_the_reference_logits(checkpoint):
    """Sliding and NoPE full layers, the own gate projection, the per-site norms and the softcapped,
    scaled head: Dew's fp32 logits lie within 1e-4 of the reference's float64 ones."""
    directory, ids, truth = checkpoint
    pretrained = Pretrained.load(str(directory), dtype="float32", attention_impl="reference")
    ours = np.asarray(pretrained.model.apply(pretrained.variables, ids))
    assert np.max(np.abs(ours - truth)) < 1e-4


def test_the_export_writes_back_the_source_tensors_and_config(checkpoint, tmp_path):
    """Saving the loaded model writes every source tensor back unchanged, and a config that loads to
    the same model and the same logits."""
    directory, ids, _ = checkpoint
    pretrained = Pretrained.load(str(directory), dtype="float32", attention_impl="reference")
    pretrained.save(tmp_path)
    written = safetensors_torch.load_file(str(tmp_path / "model.safetensors"))
    source = safetensors_torch.load_file(str(directory / "model.safetensors"))
    assert sorted(written) == sorted(source)
    for name, tensor in source.items():
        assert torch.equal(written[name].to(tensor.dtype), tensor), name
    exported = json.loads((tmp_path / "config.json").read_text())
    for key in ("layer_rope_theta", "post_norm_eps", "qk_scale_factor", "final_logit_softcapping",
                "output_multiplier"):
        assert exported[key] == pytest.approx(TEXT[key]), key
    reloaded = Pretrained.load(str(tmp_path), dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)),
                                  np.asarray(pretrained.model.apply(pretrained.variables, ids)))
