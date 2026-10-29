"""Released checkpoints of families Dew already reads, each at its own pinned config (tests/fixtures/hf,
tools/hf_reference.py `write_released_config`) shrunk to a tiny width and depth, its weights drawn by the
source's own initialization: each loads through its family's reader and computes the source's logits on
par with the source's own float32 run from its float64 twin (check 1)."""

import json
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import fixture_ids, transformers_arms, transformers_model
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, seeds_for, widened

from dew.interop import Pretrained

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


def at_tiny_dims(release: str, tiny: str, dims: tuple[str, ...], directory: Path, **beside) -> Path:
    """Fixture `release`'s released config at fixture `tiny`'s `dims` (and `beside`), over `tiny`'s weights,
    written to `directory`: the release's every other field as it states it, its quantization left with
    the weights it covers."""
    stated = json.loads((FIXTURES / release / "config.json").read_text())
    small = json.loads((FIXTURES / tiny / "config.json").read_text())
    shutil.copytree(FIXTURES / tiny, directory)
    config = {**{key: value for key, value in stated.items() if key != "quantization_config"},
              **{key: small[key] for key in dims}, **beside}
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def assert_recorded_logits(directory: Path, tiny: Path, label: str) -> None:
    """Check 1 of the checkpoint at `directory`'s logits against the source's recorded over `tiny`'s drawn
    ids (its recorded_logits.npz, `indexed_recording`), Dew's float64 run held to the recording's sketch."""
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    path = tiny / "recorded_logits.npz"
    seeds = declared_seeds(path)["logits"]
    draw = fixture_ids(tiny)
    wide, held = loaded.model.clone(dtype=jnp.float64), widened(loaded.variables)
    drawn, reference, truth = indexed_recording(
        recording(path), "logits", seeds, leaves=("logits",),
        truth_fn=lambda inputs: {"logits": wide.apply(held, inputs["input_ids"])},
        inputs_of=lambda seed: {"input_ids": draw(seed)[0]})
    def ours(index, inputs):
        return {"logits": loaded.model.apply(loaded.variables, inputs["input_ids"])}

    assert_as_exact(ours, reference, drawn, label, seeds=seeds, truth_fn=truth)


M2_DIMS = ("head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "num_attention_heads",
           "num_experts_per_tok", "num_hidden_layers", "num_key_value_heads", "num_local_experts",
           "rotary_dim", "vocab_size")
"""The fields that size a MiniMax-M2 checkpoint, which its tiny states at its own small values."""


@pytest.mark.parametrize("name", ["minimax-m2.5", "minimax-m2.7"])
def test_a_later_minimax_release_computes_m2s_source_logits(name, tmp_path):
    """MiniMax-M2.5's and M2.7's released configs (which leave out fields M2's states at the values the
    family's defaults give, and M2.7 widens the context) at minimax-m2-tiny's sizes, over its weights:
    each reads through the M2 reader and computes the logits MiniMax-M2's source computed over them
    (transformers' MiniMaxM2ForCausalLM, recorded by tools/hf_reference.py), on par with them over the
    recording's seeds (check 1). The released configs' prediction depths are absent from the weights, so
    the checkpoint loads as its trunk."""
    tiny = json.loads((FIXTURES / "minimax-m2-tiny" / "config.json").read_text())
    directory = at_tiny_dims(name, "minimax-m2-tiny", M2_DIMS, tmp_path / name,
                             attn_type_list=[1] * tiny["num_hidden_layers"])
    assert_recorded_logits(directory, FIXTURES / "minimax-m2-tiny", f"{name} logits")


DREAM_DIMS = ("hidden_size", "intermediate_size", "max_position_embeddings", "mask_token_id",
              "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "pad_token_id", "vocab_size")
"""The fields that size a Dream checkpoint, its token ids among them, which dream-tiny states small."""


def test_diffucoder_computes_dreams_source_logits(tmp_path):
    """DiffuCoder-7B's released config (Dream-v0-7B's but for its stored dtype; its configuration_dream.py
    and modeling_dream.py are Dream-v0's at 6572adb byte for byte, tools/hf_reference.py) at dream-tiny's
    sizes, over its weights, reads through the Dream reader and computes the logits Dream's own code
    computed over them (tools/remote_code_reference.py --record), on par with them over the recording's
    seeds (check 1). Its end tokens, ids past the tiny vocabulary, are left with its sizes."""
    from model_support import remote_code_site
    from remote_code_draws import draw

    directory = at_tiny_dims("diffucoder-7b", "dream-tiny", DREAM_DIMS, tmp_path / "diffucoder",
                             bos_token_id=None, eos_token_id=None)
    source = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    path, label = FIXTURES / "dream-tiny" / "released.npz", "dream-tiny logits"
    seeds = declared_seeds(path)[label]
    vocabulary = json.loads((directory / "config.json").read_text())["vocab_size"]
    drawn, reference, truth = indexed_recording(
        recording(path), label, seeds, inputs_of=lambda seed: draw("logits", seed, vocabulary),
        truth_fn=remote_code_site(source, "logits", wide=True))
    forward = remote_code_site(source, "logits")
    assert_as_exact(lambda index, inputs: forward(inputs), reference, drawn, "DiffuCoder logits", seeds=seeds,
                    truth_fn=truth)
