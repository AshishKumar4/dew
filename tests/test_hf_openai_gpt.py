"""GPT-1's post-LN blocks over GPT-2's Conv1D layers against transformers."""

import numpy as np
import pytest
from interop_support import token_draw, transformers_arms, transformers_model
from reference_error import assert_as_exact, seeds_for

from dew.interop import Pretrained
from dew.sampling import Sampling, generate


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
    reference, truth = transformers_arms(transformers_model(tmp_path / "source"),
                                         lambda model, ids: model(ids).logits)
    draw = token_draw(0x3079, 96, (2, 9))
    seeds = seeds_for(reference, draw, "GPT-1 logits", truth_fn=truth)
    assert_as_exact(lambda ids: loaded.model.apply(loaded.variables, ids), reference, draw,
                    "GPT-1 logits", seeds=seeds, truth_fn=truth)
    loaded.save(tmp_path / "export")
    exported, source = (transformers_model(tmp_path / name) for name in ("export", "source"))
    ids = torch.as_tensor(np.asarray(draw(0)[0]), dtype=torch.long)
    with torch.no_grad():
        np.testing.assert_array_equal(exported(ids).logits.numpy(), source(ids).logits.numpy())
        drawn = source.generate(ids[:1, :4], max_new_tokens=6, min_new_tokens=6, do_sample=False)
    generated = generate(loaded.model, loaded.variables, np.asarray(ids[:1, :4]), 6, key=0,
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens), drawn.numpy())
