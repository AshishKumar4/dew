"""A `Mamba2ForCausalLM` checkpoint through `dew.interop.mamba2` against the
transformers 5.16.1 logits in tests/fixtures/hf/mamba2-tiny.

Observed on CPU: fp32 logits to 9.4e-07 on logits of magnitude 1.7,
tolerance 1e-5; a bfloat16 run of the same weights to 1.5e-02 from the fp32
reference and 2.9e-02 from the reference's own bfloat16 forward (which sits
2.3e-02 from its fp32), tolerance 1e-1 on each.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact, assert_as_exact_as_the_reference
from safetensors.numpy import load_file, save_file

from dew.interop.mamba2 import config_from_hf
from dew.interop.pretrained import Pretrained

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hf" / "mamba2-tiny"


@pytest.fixture(scope="module")
def tensors():
    return load_file(FIXTURE / "model.safetensors")


def largest(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, np.float32) - np.asarray(right, np.float32))))


def test_the_public_loader_reads_the_fixture(tensors):
    """`Pretrained.load` dispatches the `mamba2` model type through the
    decoder family table and reproduces the reference logits, its config's
    unbounded time step written as transformers writes an infinity."""
    source = Pretrained.load(FIXTURE, dtype="float32", attention_impl="reference")
    ids = jnp.asarray(np.load(FIXTURE / "input_ids.npy"))
    logits = source.model.apply(source.variables, ids)
    reference = np.load(FIXTURE / "logits.npy")
    assert largest(logits, reference) < 1e-5


@pytest.mark.parametrize("bound", [{"__float__": "NaN"}, {"__float__": "nan"}, {"__float__": "inf"},
                                   {"__float__": "Infinity", "note": 1}, {"value": 1.0}])
def test_a_time_step_bound_that_is_no_number_or_infinity_refuses_the_load(tmp_path, bound):
    """The fixture with its time step's upper bound spelled as a NaN, or as a record transformers does
    not write: the load refuses it rather than build a mixer that clips to NaN."""
    for path in FIXTURE.iterdir():
        (tmp_path / path.name).symlink_to(path)
    config = json.loads((FIXTURE / "config.json").read_text())
    config["time_step_limit"] = [0.0, bound]
    (tmp_path / "config.json").unlink()
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")


STREAM_SEEDS = 8
"""The bf16 residual stream's declared draws of token ids against transformers' bf16 forward (check 1),
sized by `seeds_for` paired with the stream on the CPU."""


def test_a_bf16_mamba2_sums_its_residual_stream_in_fp32(tmp_path, tensors):
    """`residual_in_fp32` (transformers' Mamba2Block and mamba_ssm's Block): in bf16 each layer reads
    the stream in bf16 and adds its output to it in fp32. With the fixture's weights rounded to bf16
    values (so the float64 port runs the same ones) and its embeddings scaled 256 times, so that the
    stream dwarfs the mixers' outputs as Mamba-2 130M's massive activations do, each layer's stream
    is on par with transformers' own bf16 forward from the float64 port's over drawn token ids (check
    1). A stream rounded to bf16 sat 27 and 34 times farther."""
    import torch
    import transformers
    narrow = {name: np.asarray(jnp.asarray(value, jnp.bfloat16), np.float32)
              for name, value in tensors.items()}
    narrow["backbone.embeddings.weight"] = narrow["backbone.embeddings.weight"] * 256
    save_file(narrow, tmp_path / "model.safetensors")
    config = (FIXTURE / "config.json").read_text()
    (tmp_path / "config.json").write_text(config)
    shape, vocabulary = np.load(FIXTURE / "input_ids.npy").shape, json.loads(config)["vocab_size"]
    layers = json.loads(config)["num_hidden_layers"]

    def port(dtype):
        model = transformers.Mamba2ForCausalLM.from_pretrained(tmp_path, dtype=dtype).eval()

        def streams(ids):
            with torch.no_grad():
                out = model(torch.from_numpy(np.asarray(ids).astype(np.int64)), output_hidden_states=True)
            # Each layer's output; the last hidden state is the final norm's.
            return {f"layer {layer}": state.double().numpy()
                    for layer, state in enumerate(out.hidden_states[:layers])}

        return as_given(streams)

    loaded = Pretrained.load(tmp_path, dtype="bfloat16", param_dtype="bfloat16")

    def streams(ids):
        _, state = loaded.model.apply(loaded.variables, ids, capture_intermediates=True,
                                      mutable=["intermediates"])
        return {f"layer {layer}": state["intermediates"][f"layers_{layer}"]["__call__"][0]
                for layer in range(layers)}

    def draw(seed):
        return (np.random.default_rng(seed).integers(0, vocabulary, shape).astype(np.int32),)

    assert_as_exact(streams, port(torch.bfloat16), draw, "a bf16 Mamba-2's residual stream",
                    seeds=STREAM_SEEDS, truth_fn=port(torch.float64))


def mamba_ssm_config() -> dict:
    """The mamba2-tiny fixture as mamba_ssm's own config.json spells it
    (mamba_ssm/models/config_mamba.py), its vocabulary unpadded."""
    return {"d_model": 8, "d_intermediate": 0, "n_layer": 2, "vocab_size": 30,
            "ssm_cfg": {"layer": "Mamba2", "d_state": 4, "headdim": 8, "chunk_size": 4},
            "attn_layer_idx": [], "attn_cfg": {}, "rms_norm": True, "residual_in_fp32": True,
            "fused_add_norm": True, "pad_vocab_size_multiple": 16, "tie_embeddings": False}


def test_a_mamba_ssm_checkpoint_loads_and_saves_as_its_hf_port(tmp_path, tensors):
    """mamba_ssm's tensor names differ from the port's in the embedding
    alone. The load reproduces the reference logits within the fixture's
    fp32 bound, and a save writes the port's names and config."""
    source = tmp_path / "ssm"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(mamba_ssm_config()))
    save_file({name.replace("backbone.embeddings.", "backbone.embedding."): array
               for name, array in tensors.items()}, source / "model.safetensors")

    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    loaded.save(tmp_path / "saved")

    logits = loaded.model.apply(loaded.variables, jnp.asarray(np.load(FIXTURE / "input_ids.npy")))
    assert largest(logits, np.load(FIXTURE / "logits.npy")) < 1e-5
    assert sorted(load_file(tmp_path / "saved" / "model.safetensors")) == sorted(tensors)
    saved = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert config_from_hf(saved) == config_from_hf(json.loads((FIXTURE / "config.json").read_text()))


MAMBA2_130M = "3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49"
MAMBA2_130M_HF = "05e8773fc4ac1cd067e8a18a5c45372ce5178405"


@pytest.mark.network
def test_state_spaces_mamba2_130m_computes_its_hf_ports_logits():
    """state-spaces/mamba2-130m (mamba_ssm config and names, pickles on
    main, loaded from SFconvertbot's refs/pr/1) against transformers 5.16.1
    running AntonV/mamba2-130m-hf in fp32 on CPU, on two prompts cut to a
    common 13 tokens.

    Observed on CPU: fp32 4.7e-4 on logits of magnitude 170, every argmax
    equal; tolerance 1e-3, about twice that. In bfloat16 Dew hands the scan
    the conv's bf16 output, as the reference's own bfloat16 forward does, and
    is held to that forward's RMS distance from the fp32 logits
    (`assert_as_exact_as_the_reference`).
    """
    import torch
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("AntonV/mamba2-130m-hf", revision=MAMBA2_130M_HF)
    prompts = ["The capital of France is Paris, and the capital of Germany is",
               "def fibonacci(n):\n    if n < 2:\n        return n\n    return"]
    rows = [tokenizer(prompt)["input_ids"] for prompt in prompts]
    ids = np.asarray([row[:min(map(len, rows))] for row in rows], np.int32)
    port = transformers.Mamba2ForCausalLM.from_pretrained("AntonV/mamba2-130m-hf", revision=MAMBA2_130M_HF,
                                                          dtype=torch.float32).eval()
    with torch.no_grad():
        reference = port(torch.from_numpy(ids.astype(np.int64))).logits.numpy()

    fp32 = Pretrained.load("state-spaces/mamba2-130m", revision=MAMBA2_130M, dtype="float32")
    bf16 = Pretrained.load("state-spaces/mamba2-130m", revision=MAMBA2_130M,
                           dtype="bfloat16", param_dtype="bfloat16")

    logits = np.asarray(fp32.model.apply(fp32.variables, ids), np.float32)
    assert fp32.revision == "ea6060f68a4289e9c06f80effa896629ba519216"
    assert largest(logits, reference) < 1e-3
    assert np.array_equal(logits.argmax(-1), reference.argmax(-1))
    with torch.no_grad():
        theirs = port.to(torch.bfloat16)(torch.from_numpy(ids.astype(np.int64))).logits.float().numpy()
    mine = np.asarray(bf16.model.apply(bf16.variables, ids), np.float32)
    assert_as_exact_as_the_reference(mine, theirs, reference, "bf16 logits against the fp32 port's")
