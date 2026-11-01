"""GPT-1's post-LN blocks over GPT-2's Conv1D layers against transformers."""

import json

import pytest
from interop_support import assert_exported_decoder_follows_transformers, token_draw

from dew.interop import Pretrained


@pytest.mark.parametrize("afn", ["gelu", "relu"])
def test_a_gpt1_decoder_computes_transformers_logits_and_greedy_continuation(afn, tmp_path):
    """GPT-1's decoder (modeling_openai.py: learned positions, each block norming its stream after each
    sublayer joins it, no final norm, its q, k and v side by side in one Conv1D), drawn by transformers'
    own init, its tanh GELU or ReLU: Dew loads it as the openai-gpt family and computes transformers'
    logits on par from its float64 twin (check 1), its export reloads in transformers to the same logits,
    bit for bit, and a greedy continuation draws the tokens transformers' generate draws."""
    import torch
    from transformers import OpenAIGPTConfig, OpenAIGPTLMHeadModel

    config = OpenAIGPTConfig(vocab_size=96, n_embd=64, n_layer=2, n_head=4, n_positions=32, afn=afn)
    torch.manual_seed(0)
    OpenAIGPTLMHeadModel(config).eval().save_pretrained(tmp_path / "source")
    loaded = Pretrained.load(tmp_path / "source", dtype="float32", attention_impl="reference")
    assert_exported_decoder_follows_transformers(loaded, tmp_path, token_draw(0x3079, 96, (2, 9)),
                                                 "GPT-1 logits", lambda model, ids: model(ids).logits)


def test_gpt1_scales_by_the_head_width_whatever_its_config_carries(tmp_path):
    """GPT-1's blocks always scale their attention by the head width (modeling_openai.py builds each with
    `scale=True`): a config carrying GPT-2's `scale_attn_weights: false` loads with that scale, as
    transformers runs it, not unscaled."""
    import torch
    from transformers import OpenAIGPTConfig, OpenAIGPTLMHeadModel

    torch.manual_seed(0)
    OpenAIGPTLMHeadModel(OpenAIGPTConfig(vocab_size=96, n_embd=64, n_layer=2, n_head=4, n_positions=32)
                         ).eval().save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "scale_attn_weights": False}))
    loaded = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    assert loaded.model.attention_scale is None
