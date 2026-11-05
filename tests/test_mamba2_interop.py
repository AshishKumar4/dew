"""A `Mamba2ForCausalLM` checkpoint through `dew.interop.mamba2` against transformers' own model of it
(tests/fixtures/hf/mamba2-tiny), and state-spaces/mamba2-130m against its port, each held by check 1 over
seeded rows of tokens from Dew's float64 load."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording
from safetensors.numpy import load_file, save_file
from ssd_support import MAMBA2_130M_HF, mamba2_130m, mamba2_130m_logits, mamba2_interop_draw

from dew.interop.mamba2 import config_from_hf
from dew.interop.pretrained import Pretrained

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hf" / "mamba2-tiny"


@pytest.fixture(scope="module")
def tensors():
    return load_file(FIXTURE / "model.safetensors")


FIXTURE_SEEDS = 16
"""The fixture's check-1 seeds."""


def fixture_tokens(seed):
    """Two rows of 12 token ids of the fixture's vocabulary of 32."""
    return (np.random.default_rng([seed, 2003]).integers(0, 32, (2, 12)).astype(np.int32),)


def assert_logits_of_the_fixture(path: Path, label: str) -> None:
    """The checkpoint at `path`, loaded (`Pretrained.load`), on par with transformers' fp32 model of the
    fixture over seeded rows of tokens, its truth the load in float64 (check 1)."""
    import torch
    import transformers

    port = transformers.Mamba2ForCausalLM.from_pretrained(FIXTURE, dtype=torch.float32).eval()
    loaded = Pretrained.load(path, dtype="float32", attention_impl="reference")
    with jax.enable_x64(new_val=True):
        wide = Pretrained.load(path, dtype="float64", param_dtype="float64", attention_impl="reference")

    def reference(ids):
        with torch.no_grad():
            return port(torch.from_numpy(np.asarray(ids).astype(np.int64))).logits.numpy()

    @as_given
    def truth(ids):
        with jax.enable_x64(new_val=True):
            return np.asarray(wide.model.apply(wide.variables, jnp.asarray(ids)))

    assert_as_exact(jax.jit(lambda ids: loaded.model.apply(loaded.variables, ids)), as_given(reference),
                    fixture_tokens, label, seeds=FIXTURE_SEEDS, truth_fn=truth)


def test_the_public_loader_reads_the_fixture():
    """`Pretrained.load` dispatches the `mamba2` model type through the decoder family table, its config's
    unbounded time step written as transformers writes an infinity: its logits on par with transformers'
    model of the fixture (check 1)."""
    assert_logits_of_the_fixture(FIXTURE, "the fixture's logits")


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
    """mamba_ssm's tensor names differ from the port's in the embedding alone: the load's logits on par
    with transformers' model of the fixture (check 1), and a save writes the port's names and config."""
    source = tmp_path / "ssm"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(mamba_ssm_config()))
    save_file({name.replace("backbone.embeddings.", "backbone.embedding."): array
               for name, array in tensors.items()}, source / "model.safetensors")

    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    loaded.save(tmp_path / "saved")

    assert_logits_of_the_fixture(source, "mamba_ssm's names loaded")
    assert sorted(load_file(tmp_path / "saved" / "model.safetensors")) == sorted(tensors)
    saved = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert config_from_hf(saved) == config_from_hf(json.loads((FIXTURE / "config.json").read_text()))


INTEROP = Path(__file__).resolve().parent / "fixtures" / "mamba2" / "interop_seeded.npz"
"""transformers' bf16 and float64 logits of state-spaces/mamba2-130m's port on seeded rows of tokens, recorded
(tools/mamba2_reference.py's `write_interop`)."""


@pytest.mark.network
def test_state_spaces_mamba2_130m_computes_its_hf_ports_logits():
    """state-spaces/mamba2-130m (mamba_ssm config and names, pickles on main, loaded from SFconvertbot's
    refs/pr/1) against transformers 5.17.0's port of it (AntonV/mamba2-130m-hf) over seeded rows of tokens
    ('interop', `ssd_support.mamba2_interop_draw`), from float64 (check 1), its truth Dew's float64 load,
    which the recorder held to the port's float64 twin at every seed: in fp32 on par with the port running
    in fp32 on CPU, and in bfloat16, Dew handing the scan the conv's bf16 output as the port's own
    bfloat16 forward does, on par with that forward, recorded."""
    import torch
    import transformers

    port = transformers.Mamba2ForCausalLM.from_pretrained("AntonV/mamba2-130m-hf", revision=MAMBA2_130M_HF,
                                                          dtype=torch.float32).eval()

    def ported(index, inputs):
        del index
        with torch.no_grad():
            ids = torch.from_numpy(np.asarray(inputs["ids"]).astype(np.int64))
            return {"logits": port(ids).logits.numpy()}

    seeds = declared_seeds(INTEROP)["interop"]
    draw, recorded, truth = indexed_recording(
        recording(INTEROP), "interop", seeds, inputs_of=mamba2_interop_draw,
        truth_fn=jax.jit(mamba2_130m_logits(mamba2_130m(jnp.float64))))
    fp32 = mamba2_130m_logits(mamba2_130m(jnp.float32))
    assert_as_exact(jax.jit(lambda index, inputs: fp32(inputs)), as_given(ported), draw,
                    "fp32 logits against the port's in fp32", seeds=seeds, truth_fn=truth)
    bf16 = mamba2_130m_logits(mamba2_130m(jnp.bfloat16))
    assert_as_exact(jax.jit(lambda index, inputs: bf16(inputs)), recorded, draw,
                    "bf16 logits against the port's recorded in bfloat16", seeds=seeds, truth_fn=truth)
