"""`param_dtype='auto'` keeps every tensor in the dtype its checkpoint stores it
in: a bf16 checkpoint's fp32 state stays fp32, a pipeline's components keep
their own dtypes, and only a quantized source's packed weights take a dtype,
the one the checkpoint declares."""

import json
import shutil
import tarfile
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from dew.interop import Pretrained, codecs
from dew.interop.safetensors_io import read_weights, write_file

FIXTURES = Path(__file__).parent / "fixtures"


def leaf(variables, path):
    for key in path:
        variables = variables[key]
    return variables


def bf16_except(source: Path, directory: Path, kept: tuple[str, ...]) -> dict[str, np.ndarray]:
    """`source`'s checkpoint stored as bf16 but for the tensors `kept` names,
    which stay fp32, its config declaring bfloat16: the tensors written."""
    shutil.copytree(source, directory)
    tensors = {name: value if any(part in name for part in kept) else value.astype(ml_dtypes.bfloat16)
               for name, value in read_weights(source).items()}
    for stale in directory.glob("*.safetensors"):
        stale.unlink()
    write_file(tensors, directory / "model.safetensors", {"format": "pt"})
    config = json.loads((directory / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps({**config, "dtype": "bfloat16"}))
    return tensors


@pytest.mark.parametrize("fixture, kept", [
    ("mamba2-tiny", ("mixer.A_log", "mixer.D")),
    ("nemotron-h-moe-tiny", ("mixer.A_log", "mixer.D", "e_score_correction_bias")),
    ("deepseek-v3-tiny", ("e_score_correction_bias",)),
])
def test_auto_keeps_a_bf16_checkpoints_fp32_state_fp32(tmp_path, fixture, kept):
    """Mamba's A_log and D and a router's score-correction bias are stored in
    fp32 beside bf16 weights; transformers' one-dtype rule would have cast
    them to the config's bfloat16."""
    written = bf16_except(FIXTURES / "hf" / fixture, tmp_path / fixture, kept)
    loaded = Pretrained.load(tmp_path / fixture, dtype="float32", param_dtype="auto")
    exported = {layout.name: np.asarray(layout.export(loaded.variables)) for layout in loaded.weight_layouts
                if layout.name in written}
    assert {name for name, value in exported.items() if value.dtype == np.float32} == {
        name for name in exported if any(part in name for part in kept)}
    for name, value in exported.items():
        assert value.dtype == written[name].dtype, name
        np.testing.assert_array_equal(value.view(np.uint8), written[name].view(np.uint8), err_msg=name)


def test_auto_keeps_each_pipeline_component_in_its_own_dtype(tmp_path):
    """A pipeline whose denoiser is stored in bf16 beside fp32 text encoders
    and VAE loads each component as stored, where the old rule stored all of
    them at the denoiser's dtype."""
    with tarfile.open(FIXTURES / "flux_source.tar.xz") as archive:
        archive.extractall(tmp_path, filter="data")
    pipeline = tmp_path / "pipeline"
    transformer = read_weights(pipeline / "transformer")
    for stale in (pipeline / "transformer").glob("*.safetensors"):
        stale.unlink()
    write_file({name: value.astype(ml_dtypes.bfloat16) for name, value in transformer.items()},
               pipeline / "transformer" / "diffusion_pytorch_model.safetensors", {"format": "pt"})
    loaded = Pretrained.load(pipeline, dtype="float32", param_dtype="auto")
    seen = set()
    for layout in loaded.weight_layouts:
        component, _, name = layout.name.partition("/")
        stored = read_weights(pipeline / component)
        if name in stored:
            seen.add(component)
            assert np.asarray(layout.export(loaded.variables)).dtype == stored[name].dtype, layout.name
    assert {"transformer", "vae", "text_encoder"} <= seen


def test_auto_decodes_packed_weights_to_the_declared_dtype(tmp_path):
    """ModelOpt's packed NVFP4 weights have no storage dtype of their own;
    they decode to the bfloat16 its config declares, and its unpacked
    tensors keep theirs."""
    directory = FIXTURES / "codecs" / "modelopt" / "tiny"
    stored = read_weights(directory)
    codec = codecs.source_quantization(json.loads((directory / "config.json").read_text()))
    assert codec is not None
    packed = set(codec.names(stored))
    loaded = Pretrained.load(directory, dtype="float32", param_dtype="auto")
    checked = 0
    for layout in loaded.weight_layouts:
        dtype = leaf(loaded.variables, layout.paths[0]).dtype
        if layout.name in packed:
            assert dtype == ml_dtypes.bfloat16, layout.name
            checked += 1
        elif layout.name in stored and len(layout.paths) == 1:
            assert dtype == stored[layout.name].dtype, layout.name
    assert checked == len(packed)
