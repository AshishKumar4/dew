"""A convolution compiled without its GPU keeps algorithm -1, cuDNN's pick at each run, which an A100 does not
repeat (`report.DEVICELESS_CONVOLUTION_PICKS`, row 7941): named programs holding one are refused, read for
their memory alone they say so, and a step file holding one anywhere is compiled on the run's devices."""

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from model_support import TINY_DECODER
from test_compile_ahead import BATCH, SEQ_LEN, TINY_SHARD, VOCAB, gpus
from token_support import token_rows

from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer, TrainerAdvanced


def test_a_convolution_compiled_ahead_without_its_gpu_is_compiled_on_it(tmp_path, monkeypatch):
    """A compile without a device cannot pick a cuDNN convolution's algorithm, which is then picked at run
    time, on an A100 not the same each run (`report.DEVICELESS_CONVOLUTION_PICKS`, row 7941): over a node's
    topology, autotuner off as a repeatable lane runs, and over the generation's canonical GPUs alike, a
    program holding a convolution is refused before any compiles (`report.CompiledOnDevice`), named, where a
    product alone compiles."""
    from jax.experimental import topologies

    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.templates import report
    from dew.training.ahead import AheadPrograms, BuildTarget, compile_programs

    serialized = tmp_path / "topology.bin"
    serialized.write_bytes(topologies.TopologyDescription(list(compile_only_gpus("sm80", 1))).serialize())
    monkeypatch.setenv("XLA_FLAGS", f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0")
    image = jax.ShapeDtypeStruct((1, 8, 8, 4), jnp.float32)
    kernel = jax.ShapeDtypeStruct((3, 3, 4, 4), jnp.float32)
    square = jax.ShapeDtypeStruct((128, 128), jnp.float32)

    def convolved(x, w):
        return jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))

    with pytest.raises(report.CompiledOnDevice) as refused:
        compile_programs({"conv": (convolved, [image, kernel])}, "sm80", topology=serialized)
    assert refused.value.programs == ("conv",)
    assert list(compile_programs({"dot": (lambda x, y: x @ y, [square, square])}, "sm80",
                                 topology=serialized).programs) == ["dot"]
    with pytest.raises(report.CompiledOnDevice) as refused:
        compile_programs({"conv": (convolved, [image, kernel]),
                          "dot": (lambda x, y: x @ y, [square, square])}, "sm80")
    assert refused.value.programs == ("conv",)
    # Compiled offline, for its memory or a step file's program, it holds the algorithm left to run time
    # (`report.left_at_run`, by which a step file is never taken: `AheadOfTime.at_run`); the product not.
    placed = jax.sharding.SingleDeviceSharding(compile_only_gpus("sm80", 1)[0])

    def offline(function, *shapes):
        lowered = jax.jit(function).trace(*(jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=placed)
                                            for shape in shapes)).lower()
        with report.read_offline():
            return report.xla_compiled(lowered, None)

    assert report.left_at_run(offline(convolved, image, kernel))
    assert not report.left_at_run(offline(lambda x, y: x @ y, square, square))
    # Read for its memory alone, its file says so, which every GPU's load refuses (row 8688).
    read = compile_programs({"conv": (convolved, [image, kernel]),
                             "dot": (lambda x, y: x @ y, [square, square])}, "sm80", memory_only=True)
    assert read.at_run == ("conv",)
    # Saved and loaded, as a process of their target takes them, at the autotuner's level 0 (where an
    # untuned file is taken): the file holding the convolution is refused, its product alone's is loaded
    # (row 8785). This host's own target stands for the GPU's.
    here = BuildTarget.of(jax.devices()[:1])
    for name, at_run, loads in (("conv", ("conv",), False), ("dot", (), True)):
        dataclasses.replace(read, target=here, programs={}, at_run=at_run).save(tmp_path / f"{name}.dew-aot")
        handed = AheadPrograms.load(tmp_path / f"{name}.dew-aot")
        if loads:
            assert handed.loaded() == {}
        else:
            with pytest.raises(ValueError):
                handed.loaded()


class ConvolvedStart(LMObjective):
    """A decoder's objective whose start alone holds a convolution: its initializer draws the run's
    variables and adds nothing of a convolution of its key's noise, which its step never runs."""

    @property
    def initializer(self):
        return jax.tree_util.Partial(_convolved_draw, super().initializer)


def _convolved_draw(drawn, key):
    """`drawn(key)`, one leaf moved by nothing of a convolution XLA cannot fold away."""
    variables = drawn(key)
    noise = jax.random.normal(key, (1, 8, 8, 4))
    convolved = jax.lax.conv_general_dilated(noise, jnp.ones((3, 3, 4, 4)), (1, 1), "SAME",
                                             dimension_numbers=("NHWC", "HWIO", "NHWC"))
    leaves, structure = jax.tree.flatten(variables)
    return jax.tree.unflatten(structure, [leaves[0] + 0 * jnp.sum(convolved), *leaves[1:]])


def test_a_step_file_whose_start_alone_holds_a_convolution_is_compiled_on_the_runs_devices(tmp_path,
                                                                                         monkeypatch):
    """A step compiled ahead for a generation's canonical GPUs whose convolution sits in its start alone (its
    initializer), its step holding none, leaves that convolution's algorithm to run time
    (`report.DEVICELESS_CONVOLUTION_PICKS`): its file says so (`AheadOfTime.at_run`, read over every
    executable it holds), and a run never takes it, at the autotuner's level 0 too, where an untuned step
    is taken (row 8687)."""
    from dew._cache import uncached
    from dew.training.ahead import AheadOfTime, BuildTarget, Executable, compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
    trainer = Trainer(ConvolvedStart(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0),
                      mesh=MeshSpec().build(gpus("sm90")),
                      advanced=TrainerAdvanced(layout=Layout(min_shard=TINY_SHARD)))
    compile_ahead(trainer, batch).save(tmp_path / "step.dew-aot")
    handed = AheadOfTime.load(tmp_path / "step.dew-aot")
    assert handed.at_run
    here = jax.devices()[:1]
    monkeypatch.setenv("XLA_FLAGS", f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0")
    assert dataclasses.replace(handed, built=BuildTarget.of(here)).runnable(here) is None
    # Loaded direct, as a tool or a caller of `taken` loads it, it is refused alike (row 8957): a step of
    # this host's own, which loads here as it is, refused where it says it leaves a choice to run time.
    every = jax.devices()
    replicated = jax.sharding.NamedSharding(jax.sharding.Mesh(np.asarray(every), ("devices",)),
                                            jax.sharding.PartitionSpec())
    with uncached():
        doubled = jax.jit(lambda x: x * 2.0, out_shardings=replicated).lower(
            jax.ShapeDtypeStruct((4,), jnp.float32, sharding=replicated)).compile()
    own = dataclasses.replace(handed, built=BuildTarget.of(every), executable=Executable.of(doubled))
    assert dataclasses.replace(own, at_run=False).loaded() is not None
    with pytest.raises(ValueError):
        own.loaded()
