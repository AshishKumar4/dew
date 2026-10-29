"""Released checkpoints of families Dew already reads, each at its own pinned config (tests/fixtures/hf,
tools/hf_reference.py `write_released_config`) shrunk to a tiny width and depth, its weights drawn by the
source's own initialization: each loads through its family's reader and computes the source's logits on
par with the source's own float32 run from its float64 twin (check 1)."""

import json
from pathlib import Path

import numpy as np
import pytest
from interop_support import transformers_arms, transformers_model
from reference_error import assert_as_exact, seeds_for

from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config

FIXTURES = Path(__file__).parent / "fixtures" / "hf"


def shrunk(name: str, **dims) -> dict:
    """The released config of fixture `name` with `dims` over its own: every other field as released."""
    return {**json.loads((FIXTURES / name / "config.json").read_text()), **dims}


def ids_over(vocabulary: int, shape=(2, 9)):
    """Check 1's draw of token ids over a whole vocabulary, the same on every run."""
    def draw(seed):
        return (np.random.default_rng([0x1D5, seed]).integers(0, vocabulary, shape).astype(np.int32),)
    return draw


def test_minicpm5_computes_transformers_logits_through_the_llama_reader(tmp_path):
    """MiniCPM5-2B's released config (a llama: grouped queries 16 over 2, a 128-wide head over a
    2048-wide stream, rope theta 5e6, an untied head, two end tokens and its whole 130560-token
    vocabulary) at two layers of width 64, its weights transformers' own draw, loads through the llama
    reader and computes transformers' logits on par from its float64 twin (check 1)."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    config = shrunk("minicpm5-2b", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(**config)).to(torch.float32).save_pretrained(tmp_path)
    loaded = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    reference, truth = transformers_arms(transformers_model(tmp_path),
                                         lambda model, ids: model(ids, use_cache=False).logits)
    draw = ids_over(config["vocab_size"])
    seeds = seeds_for(reference, draw, "MiniCPM5 logits", truth_fn=truth)
    assert_as_exact(lambda ids: loaded.model.apply(loaded.variables, ids), reference, draw, "MiniCPM5 logits",
                    seeds=seeds, truth_fn=truth)


@pytest.mark.parametrize("name", ["minimax-m2.5", "minimax-m2.7"])
def test_a_later_minimax_release_reads_as_m2_at_its_own_context(name):
    """MiniMax-M2.5's and M2.7's released configs leave out fields M2's states at the values M2's own
    defaults give (dropout, the unread linear-attention betas, the shared-expert mode) and M2.7 widens the
    context: each reads through the M2 reader as M2's model at its own context, the model whose
    computation M2's source qualifies (tests/test_hf_decoder_moe.py)."""
    later = json.loads((FIXTURES / name / "config.json").read_text())
    m2 = json.loads((FIXTURES / "minimax-m2" / "config.json").read_text())
    assert translate_config(later) == translate_config(
        {**m2, "max_position_embeddings": later["max_position_embeddings"]})


def test_diffucoder_reads_through_the_dream_reader_as_dream():
    """DiffuCoder-7B's released config is Dream-v0-7B's but for its stored dtype, and its model code is
    Dream-v0's byte for byte (tools/hf_reference.py): it reads through the Dream reader as Dream's model,
    whose logits Dream's own code qualifies (tests/test_hf_decoders.py, dream-tiny)."""
    diffucoder = json.loads((FIXTURES / "diffucoder-7b" / "config.json").read_text())
    dream = json.loads((FIXTURES / "dream-7b" / "config.json").read_text())
    assert translate_config(diffucoder) == translate_config({**dream, "torch_dtype": diffucoder["torch_dtype"]})
