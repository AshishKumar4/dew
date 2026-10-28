"""safetensors round-trips for Flax parameter trees, and the hub round trip.

The tree is nested and the file is flat, so what these tests hold onto is the
'/'-joined naming: nesting, values and dtypes have to survive a save and a
load, the names on disk have to be readable by a safetensors reader that knows
nothing about dew, and a missing optional dependency has to say so. The hub
pair is tested against a recording stand-in for the hub client: what matters
is the call it makes and the directory it hands over, and no test reaches the
network.
"""

import json
import sys
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

import dew
from dew.interop import load_params, save_params
from dew.interop.safetensors_io import read_file, write_file
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.dit import TextContext

safetensors_numpy = import_module("safetensors.numpy")
safetensors = import_module("safetensors")



@pytest.fixture
def params(rng):
    model = SimpleDiT(
        patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1
    )
    x = jax.random.normal(rng, (1, 8, 8, 3))
    return model.init(
        rng,
        x,
        jnp.ones((1,)),
        TextContext(jnp.ones((1, 77, 768)), jnp.ones((1, 77), bool)),
    )


def test_round_trip_keeps_the_tree_and_the_values(params, tmp_path):
    path = tmp_path / "model.safetensors"
    save_params(params, path)
    loaded = load_params(path)

    assert jax.tree_util.tree_structure(loaded) == jax.tree_util.tree_structure(params)
    for saved, restored in zip(jax.tree.leaves(params), jax.tree.leaves(loaded), strict=True):
        assert np.array_equal(np.asarray(saved), restored)


def test_round_trip_keeps_bfloat16(params, tmp_path):
    """The point of writing bf16 is the bytes, so a load must not widen them."""
    narrowed = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), params)
    path = tmp_path / "bf16.safetensors"
    save_params(narrowed, path)
    loaded = load_params(path)

    for saved, restored in zip(
        jax.tree.leaves(narrowed), jax.tree.leaves(loaded), strict=True
    ):
        assert restored.dtype == jnp.bfloat16
        assert np.array_equal(np.asarray(saved), restored)


# ---------------------------------------------------------------------------------
# The memory-mapped reader
# ---------------------------------------------------------------------------------


def test_every_stored_dtype_reads_back_exactly(tmp_path):
    """Scalar, nonempty and empty tensors keep their own dtype and bytes."""
    stored = {
        "bf16": np.asarray(jnp.asarray([1.5, -2.0], jnp.bfloat16)),
        "f16": np.asarray([0.5, -0.25], np.float16),
        "f32": np.asarray([0.5, -0.25], np.float32),
        "f64": np.asarray([[3.0]], np.float64),
        "i64": np.asarray([7], np.int64),
        "bytes": np.asarray([0, 127, 128, 255], np.uint8),
        "empty": np.zeros((0, 4), np.uint8),
        "flag": np.ones((), dtype=np.bool_),
        "scalar": np.asarray(0.25, np.float32),
    }
    path = tmp_path / "model.safetensors"
    safetensors_numpy.save_file(stored, str(path), metadata={"writer": "dew"})

    tensors, metadata = read_file(path)

    assert metadata == {"writer": "dew"}
    for name, expected in stored.items():
        tensor = tensors[name]
        assert tensor.dtype == expected.dtype and tensor.shape == expected.shape
        np.testing.assert_array_equal(tensor, expected)


def test_float8_payloads_are_mapped_without_numpy_dtype_conversion(tmp_path):
    """FP8 array, scalar and empty payloads use the official format tag.
    Compare stored bits, including signed zero, rather than widened values."""
    from dew.interop.codecs import E4M3

    stored = {
        "weight": np.asarray([0.0, -0.0, 1.5, -2.0, 448.0], dtype=E4M3),
        "scalar": np.asarray(-1.5, dtype=E4M3),
        "empty": np.empty((0, 3), dtype=E4M3),
        "bf16": np.asarray([1.5, -2.0], dtype=ml_dtypes.bfloat16),
        "packed": np.asarray([0, 128, 255], dtype=np.uint8),
    }
    path = tmp_path / "fp8.safetensors"
    safetensors_numpy.save_file(stored, str(path))
    tensors, _ = read_file(path)
    for name, expected in stored.items():
        actual = tensors[name]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert isinstance(actual.base, np.memmap) and not actual.flags.writeable
        assert actual.tobytes() == expected.tobytes()


def raw_safetensors(path, entries):
    """Write a safetensors file byte for byte, for the dtype tags no NumPy writer emits."""
    header, payload = {}, b""
    for name, (tag, shape, data) in entries.items():
        header[name] = {"dtype": tag, "shape": list(shape),
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


_A = '"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}'
_B = '"b":{"dtype":"I32","shape":[2,2],"data_offsets":[16,32]}'
_PAYLOAD = np.arange(4, dtype=np.float32).tobytes() + np.arange(4, dtype=np.int32).tobytes()

def _table(*entries: str) -> str:
    return "{" + ",".join(entries) + "}"


CORRUPTIONS = {
    "valid": (_table(_A, _B), _PAYLOAD),
    "gap": (_table(_A, _B.replace("[16,32]", "[24,40]")), _PAYLOAD + bytes(8)),
    "overlap": (_table(_A, _B.replace("[16,32]", "[8,24]")), _PAYLOAD[:24]),
    "short": (_table(_A, _B), _PAYLOAD[:-1]),
    "trailing": (_table(_A, _B), _PAYLOAD + bytes(4)),
    "span": (_table(_A.replace("[2,2]", "[2,3]"), _B), _PAYLOAD),
    "dtype": (_table(_A.replace("F32", "Q9"), _B), _PAYLOAD),
    "repeated": (_table(_B.replace("I32", "F32"), _A, _B), _PAYLOAD),
    "null-metadata": (_table('"__metadata__":null', _A, _B), _PAYLOAD),
    "unknown-field": (_table(_A.replace("}", ',"extension":1}'), _B), _PAYLOAD),
    "repeated-field": (_table(_A.replace('{"dtype":"F32"', '{"dtype":"F32","dtype":"F32"'), _B), _PAYLOAD),
    "repeated-metadata": (_table('"__metadata__":{"x":"1"}', '"__metadata__":{"y":"2"}', _A, _B), _PAYLOAD),
    "metadata": (_table('"__metadata__":{"x":"1","x":"2"}', _A, _B), _PAYLOAD),
    "utf-16": (_table(_A, _B), _PAYLOAD),
    "entry": (_table('"c":[1]', _A, _B), _PAYLOAD),
    "object-shape": (_table(_A.replace("[2,2]", "{}"), _B), _PAYLOAD),
    "sequence": (_table('"a":["F32",[2,2],[0,16]]', _B), _PAYLOAD),
    "surrogate": (_table('"__metadata__":{"x":"\\ud800"}', _A, _B), _PAYLOAD),
}
"""Each one change to a valid two-tensor file: a hole between tensors, two
tensors on the same bytes, a missing byte, bytes after the tensors, a shape
its bytes do not hold, an unknown dtype tag, a tensor named twice with another
dtype first, null metadata, a field TensorInfo does not know, a TensorInfo
field named twice, metadata named twice, a metadata key named twice, the
header in UTF-16, a tensor that is no object, a shape that is an object, a
tensor stated as a sequence, and metadata holding an unpaired surrogate."""


@pytest.mark.parametrize("corruption", [*CORRUPTIONS, "length"])
def test_a_read_accepts_and_refuses_what_safetensors_does(tmp_path, corruption):
    """Dew validates the header of the file it holds open through safe_open: a
    file safe_open reads reads to the same tensors and metadata, and each one
    safe_open refuses, Dew refuses with safe_open's error. `length` states a
    header longer than the file."""
    text, payload = CORRUPTIONS.get(corruption, CORRUPTIONS["valid"])
    header = text.encode("utf-16" if corruption == "utf-16" else "utf-8")
    stated = 10 ** 9 if corruption == "length" else len(header)
    path = tmp_path / "model.safetensors"
    path.write_bytes(stated.to_bytes(8, "little") + header + payload)
    try:
        with safetensors.safe_open(str(path), "np") as reader:
            names = reader.keys()
            expected = {name: reader.get_tensor(name) for name in names}
            stored = reader.metadata() or {}
    except safetensors.SafetensorError:
        with pytest.raises(safetensors.SafetensorError):
            read_file(path)
        return
    tensors, metadata = read_file(path)
    assert metadata == stored and list(tensors) == sorted(expected)
    for name, value in expected.items():
        assert tensors[name].dtype == value.dtype
        np.testing.assert_array_equal(tensors[name], value)


def test_block_scale_exponents_and_packed_fp4_are_mapped_as_stored(tmp_path):
    """DeepSeek-V4's shards carry F8_E8M0 scales and F4 experts (the header of
    DeepSeek-V4-Flash shard 5). An E8M0 byte is the exponent 2**(b - 127); an
    F4 tensor of n elements is n / 2 bytes, returned as that byte view, the
    way PyTorch writes a float4_e2m1fn_x2 [1, 2] as F4 [1, 4]."""
    path = tmp_path / "v4.safetensors"
    raw_safetensors(path, {"w.scale": ("F8_E8M0", (3,), bytes([127, 128, 120])),
                           "w": ("F4", (1, 4), bytes([0x21, 0x43]))})

    tensors, _ = read_file(path)

    scale = tensors["w.scale"]
    assert scale.dtype == ml_dtypes.float8_e8m0fnu
    np.testing.assert_array_equal(scale.astype(np.float32), [1.0, 2.0, 2.0 ** -7])
    packed = tensors["w"]
    assert packed.dtype == np.uint8 and packed.shape == (1, 2)
    assert packed.tobytes() == bytes([0x21, 0x43]) and isinstance(packed.base, np.memmap)


_PUBLISHERS = [
    pytest.param(
        lambda value, path: save_params({"params": {"w": value}}, path),
        id="save_params",
    ),
    pytest.param(
        lambda value, path: write_file({"params/w": value}, path, {}), id="write_file"
    ),
]


def test_the_same_tensors_and_metadata_write_the_same_bytes(tmp_path):
    """safetensors 0.8.0 keeps a header's metadata in a Rust HashMap, whose
    order is drawn per map, so the same table and metadata wrote different
    bytes from one write to the next (6 distinct files in 20 writes of three
    keys). A written file is a function of what it holds: an adapter saved
    twice, or a file and its digest, agree byte for byte."""
    path = tmp_path / "model.safetensors"
    tensors = {"a": np.ones((2, 2), np.float32), "b": np.zeros(3, np.int32)}
    metadata = {"format": "pt", "lora_adapter_metadata": '{"r": 2}', "z": "1"}
    written = set()
    for _ in range(20):
        write_file(tensors, path, metadata)
        written.add(path.read_bytes())
    assert len(written) == 1
    header = json.loads(next(iter(written))[8:8 + int.from_bytes(next(iter(written))[:8], "little")])
    assert list(header["__metadata__"]) == sorted(metadata)
    loaded, read = read_file(path)
    assert read == metadata and all(np.array_equal(loaded[name], tensors[name]) for name in tensors)


@pytest.mark.parametrize("publish", _PUBLISHERS)
def test_overwrite_retains_the_previously_loaded_tree(tmp_path, publish):
    path = tmp_path / "model.safetensors"
    save_params({"params": {"w": np.asarray([1.0, 2.0], np.float32)}}, path)
    original = load_params(path)

    publish(np.asarray([9.0, 8.0], np.float32), path)

    np.testing.assert_array_equal(original["params"]["w"], [1.0, 2.0])
    np.testing.assert_array_equal(load_params(path)["params"]["w"], [9.0, 8.0])
    assert set(tmp_path.iterdir()) == {path}


def test_a_read_racing_overwrites_holds_one_whole_file(tmp_path):
    """A writer replaces the file between two tables of other dtypes and
    shapes while a reader reads it. Each read holds one table whole: its
    header, its offsets and its bytes come from the one file the reader
    opened, never one file's header over the other's bytes. Each read is
    checked and dropped before the next: its map holds a descriptor, and
    2000 held at once pass a container's 1024."""
    import threading

    path = tmp_path / "model.safetensors"
    tables = ({"w": np.arange(64, dtype=np.float32).reshape(8, 8), "b": np.ones(4, np.float32)},
              {"w": np.full((2, 3), 7, np.int64), "b": np.zeros(1000, np.float16)})
    write_file(tables[0], path, {})
    done = threading.Event()

    def overwrite():
        count = 0
        while not done.is_set():
            count += 1
            write_file(tables[count % 2], path, {})

    def whole(read):
        return any(read.keys() == table.keys() and all(
            read[name].dtype == table[name].dtype and np.array_equal(read[name], table[name])
            for name in table) for table in tables)

    writer = threading.Thread(target=overwrite)
    writer.start()
    try:
        mixed = sum(not whole(read_file(path)[0]) for _ in range(2000))
    finally:
        done.set()
        writer.join()
    assert not mixed, f"{mixed} of 2000 reads paired one file's header with another's bytes"


@pytest.mark.parametrize("publish", _PUBLISHERS)
def test_failed_publication_preserves_the_file_and_cleans_the_temporary(
    tmp_path, monkeypatch, publish
):
    path = tmp_path / "model.safetensors"
    save_params({"params": {"w": np.asarray([1.0, 2.0], np.float32)}}, path)
    original = load_params(path)
    before = path.read_bytes()

    def partial_write(tensors, filename, metadata=None):
        Path(filename).write_bytes(b"incomplete payload")
        raise OSError("injected write failure")

    monkeypatch.setattr(safetensors_numpy, "save_file", partial_write)
    with pytest.raises(OSError):
        publish(np.asarray([9.0, 8.0], np.float32), path)

    assert path.read_bytes() == before
    np.testing.assert_array_equal(original["params"]["w"], [1.0, 2.0])
    assert set(tmp_path.iterdir()) == {path}


def assert_parameter_storage(reference, actual, is_parameter):
    """The public storage contract on full variable paths, not implementation calls."""
    assert jax.tree.structure(reference) == jax.tree.structure(actual)
    for (path, before), after in zip(jax.tree_util.tree_leaves_with_path(reference),
                                    jax.tree.leaves(actual), strict=True):
        names = tuple(entry.key for entry in path)
        before, after = np.asarray(before), np.asarray(after)
        floating = jnp.issubdtype(before.dtype, jnp.floating)
        if floating:
            assert before.dtype == np.float32, names
        dtype = np.dtype(ml_dtypes.bfloat16) if floating and is_parameter(names) else before.dtype
        assert after.dtype == dtype, names
        np.testing.assert_array_equal(after, before.astype(dtype), err_msg=str(names))


@pytest.mark.parametrize("family", [
    "llama-tiny", "gemma4-tiny-mm", "gemma-4-audio-tiny", "diffusion-gemma-workflow",
])
def test_public_parameter_storage_is_independent_of_compute_and_roundtrips(tmp_path, family):
    from dew.interop import Pretrained
    from dew.nn.diffusion_gemma import DiffusionGemma

    directory = Path(__file__).resolve().parent / "fixtures" / "hf" / family
    masters = Pretrained.load(directory, dtype="bfloat16", attention_impl="xla")
    native = Pretrained.load(directory, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
    master_model = masters.model.text if isinstance(masters.model, DiffusionGemma) else masters.model
    native_model = native.model.text if isinstance(native.model, DiffusionGemma) else native.model
    assert jnp.dtype(master_model.dtype) == jnp.dtype(jnp.bfloat16)
    assert jnp.dtype(native_model.dtype) == jnp.dtype(jnp.float32)
    assert_parameter_storage(masters.variables, native.variables, lambda path: path[0] == "params")
    destination = tmp_path / "export"
    native.save(destination)
    restored = Pretrained.load(destination, dtype="float32", param_dtype="bfloat16", attention_impl="xla")
    assert jax.tree.structure(native.variables) == jax.tree.structure(restored.variables)
    for before, after in zip(
        jax.tree.leaves(native.variables), jax.tree.leaves(restored.variables), strict=True
    ):
        assert np.asarray(before).dtype == np.asarray(after).dtype
        np.testing.assert_array_equal(before, after)

    if family == "llama-tiny":
        widened = Pretrained.load(destination, dtype="float32", attention_impl="xla")
        for before, after in zip(jax.tree.leaves(native.variables),
                                 jax.tree.leaves(widened.variables), strict=True):
            assert np.asarray(after).dtype == np.float32
            np.testing.assert_array_equal(np.asarray(after), np.asarray(before, np.float32))


# ---------------------------------------------------------------------------------
# Hub pull
# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# A trained run out to the published layout
# ---------------------------------------------------------------------------------


RUN_EXPORT_SEEDS = 8
"""A run's export's declared draws of ids against transformers' fp32 read of it (check 1)."""


def assert_transformers_reads_the_run(export, task, ids, *, ordered_export=None):
    """transformers' own class for the export's model_type loads its files
    with a clean report (`decoder_export_reference.reference_model`) and holds
    the run's own logits on par with its fp32 forward from its float64 one over
    drawn ids of the shape of `ids` (check 1). An `ordered_export`
    holds the trained run over 52 residual orders when one draw can flake,
    each order of the run's canonical tree exported and, packed as the run's
    task holds it, run by Dew; every reference order in float64 must compute
    the identity's truth.
    """
    from reference_error import (
        ORDERS,
        as_given,
        assert_as_exact,
        assert_as_exact_over_orders,
        assert_computes_the_oracle,
        distance,
    )
    from residual_orders import orders, permuted, residual_width

    from dew.inference.projections import inference_projections
    from dew.interop import Pretrained
    from tools.decoder_export_reference import Case, reference_logits

    case = Case("run export", str(export))
    if ordered_export is None:
        vocabulary = json.loads((Path(export) / "config.json").read_text())["vocab_size"]

        def draw(seed):
            return (np.random.default_rng(seed).integers(0, vocabulary, np.shape(ids)).astype(np.int32),)

        def read(wide):
            return as_given(lambda drawn: reference_logits(case, export, np.asarray(drawn), wide=wide))

        assert_as_exact(lambda drawn: task.model.apply(task.variables, drawn), read(wide=False), draw,
                        "run export logits", seeds=RUN_EXPORT_SEEDS, truth_fn=read(wide=True))
        return
    truth = reference_logits(case, export, ids, wide=True)
    source = Pretrained.load(export, dtype="float32", attention_impl="reference")
    forward = jax.jit(lambda variables: task.model.apply(variables, jnp.asarray(ids, jnp.int32)))
    mine, theirs = [], []
    # Two attention projections, a key reduction, two MLP projections and
    # the norms/residuals per layer, then the output head, bound the chain.
    model = source.model
    roundings = model.num_layers * (4 * model.emb_features + model.mlp_features + ids.shape[1] + 16)
    roundings += 2 * model.emb_features
    for order in orders(residual_width(source.variables), ORDERS, seed=0):
        variables = permuted(source.variables, order)
        source.save(ordered_export, variables=variables)
        assert_computes_the_oracle(reference_logits(case, ordered_export, ids, wide=True), truth,
                                   "run export logits in order", roundings=roundings)
        mine.append(distance(forward(inference_projections(task.model, variables)), truth))
        theirs.append(distance(reference_logits(case, ordered_export, ids), truth))
    assert_as_exact_over_orders(mine, theirs, "run export logits")


def test_a_trained_lm_run_exports_and_reloads_at_its_own_logits(tmp_path):
    """The whole way out of a run directory: run.json and the checkpoint in,
    a Hugging Face directory out, and `Pretrained.load` reads it back at the
    logits the run's weights compute, as transformers does; read with its
    projections packed, as the run's own task holds them, at the task's."""
    from test_inference import make_lm_run

    from dew.inference.projections import inference_projections
    from dew.interop import Pretrained

    run = tmp_path / "run"
    run.mkdir()
    make_lm_run(run)
    task = dew.pipeline(str(run))
    destination = tmp_path / "export"

    exported = Pretrained.from_run(str(run))
    exported.save(destination)

    assert {entry.name for entry in destination.iterdir()} == {
        "config.json", "generation_config.json", "model.safetensors"}
    assert json.loads((destination / "generation_config.json").read_text())["tokenizer_name"] == "byte"
    reloaded = Pretrained.load(destination, dtype="float32", attention_impl="reference")
    ids = jnp.asarray([[3, 4, 5, 6]], jnp.int32)
    np.testing.assert_array_equal(np.asarray(reloaded.model.apply(reloaded.variables, ids)),
                                  np.asarray(exported.model.apply(exported.variables, ids)))
    np.testing.assert_array_equal(
        np.asarray(reloaded.model.apply(inference_projections(reloaded.model, reloaded.variables), ids)),
        np.asarray(task.model.apply(task.variables, ids)))
    assert_transformers_reads_the_run(destination, task, np.asarray([[3, 4, 5, 6, 7, 8, 9, 10]]),
                                      ordered_export=tmp_path / "orders")


FRESH_TRAINING = """
import sys

import grain.python as grain
import jax.numpy as jnp
import numpy as np
import optax

from dew.checkpoints import Checkpoints
from dew.data import ByteTokenizer, Loading, Memory
from dew.data.compile import compile
from dew.data.pipeline import Pipeline
from dew.inference import RunProcessor
from dew.nn.backbones import CausalTransformer
from dew.objectives.{module} import {objective}
from dew.training import Trainer

model = CausalTransformer(vocab_size=256, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                          max_seq_len=16, dtype=jnp.float32, attention_impl="xla")
objective = {objective}(model, 8, ema_decay=None, processor=RunProcessor(ByteTokenizer()))
rows = [{{"text": np.arange(9, dtype=np.int32)}} for _ in range(8)]
data = compile(Pipeline(Memory(rows)).repeat().batch(8), read="memory", loading=Loading(workers=0))
checkpoints = Checkpoints(sys.argv[1])
Trainer(objective, optax.sgd(.01), key=0, checkpoints=checkpoints).fit(data, steps=1, checkpoint_every=1)
checkpoints.wait()
"""


def test_a_run_trained_in_one_process_exports_from_a_fresh_one(tmp_path):
    """`dew export` in a new process reads the run by its record alone: the
    registry imports the module that registers each name the record holds,
    so nothing the training process happened to import is needed. What it
    writes, transformers reads at the run's own logits."""
    import os
    import subprocess


    root = Path(__file__).resolve().parents[1]
    env = {**{name: value for name, value in os.environ.items() if name != "XLA_FLAGS"},
           "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(root / "src")}
    program = FRESH_TRAINING.format(module="lm", objective="LMObjective")
    trained = subprocess.run([sys.executable, "-c", program, str(tmp_path / "run")],
                             capture_output=True, text=True, env=env, timeout=600)
    assert trained.returncode == 0, trained.stderr[-2000:]
    exported = subprocess.run([sys.executable, "-m", "dew.cli.main", "export", str(tmp_path / "run"),
                               str(tmp_path / "export")],
                              capture_output=True, text=True, env=env, timeout=600)
    assert exported.returncode == 0, exported.stderr[-2000:]
    assert (tmp_path / "export" / "model.safetensors").is_file()
    assert_transformers_reads_the_run(tmp_path / "export", dew.pipeline(str(tmp_path / "run")),
                                      np.arange(9).reshape(1, 9))

    from dew.cli.main import main

    with pytest.raises(FileNotFoundError):
        main(["export", str(tmp_path / "nothing"), str(tmp_path / "other")])


def test_a_written_tensor_reads_back_in_its_own_order_whatever_its_memory_layout(tmp_path):
    """A column-major or transposed view is written as the values it holds.
    safetensors writes memory as C order, and a TPU returns column-major host
    views of some arrays: a bf16 checkpoint written from one read back
    transposed on the TPU lane (16,364 of 16,384 values wrong)."""
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    layouts = {"fortran": np.asfortranarray(values), "transposed": values.T.copy().T,
               "bf16": np.asfortranarray(values.astype(ml_dtypes.bfloat16))}
    write_file(layouts, tmp_path / "model.safetensors", {"format": "pt"})
    read, _ = read_file(tmp_path / "model.safetensors")
    for name in layouts:
        np.testing.assert_array_equal(np.asarray(read[name], np.float32), values, err_msg=name)
