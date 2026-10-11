"""A training step lowered for a CUDA GPU on a host without one (`dew.training._lowering`)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.sharding import Mesh

from dew.nn.kernels.generation import (
    DEVICE_KINDS,
    KERNELS,
    PALLAS_CHOICES,
    assumed,
    backend,
    bf16_dot_runs,
    device_generation,
    gpu_mesh,
    measured_kernel,
    triton_runs,
)
from dew.training._lowering import UNLOWERABLE, triton_kernels

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("generation, triton, bf16", [
    ("sm75", False, False), ("sm80", True, True), ("sm120", True, True), ("v5e", False, True)])
def test_an_assumed_generation_answers_for_the_device(generation, triton, bf16):
    """Within `assumed`, the generation, its bf16 dot and its Triton kernels are the assumed device's,
    on this CPU, and so is every KERNELS choice; outside it, and under `assumed(None)`, this
    device's again."""
    own = device_generation(), triton_runs(), bf16_dot_runs()
    with assumed(generation):
        assert (device_generation(), triton_runs(), bf16_dot_runs()) == (generation, triton, bf16)
        rule = measured_kernel('gated_delta_rule', 'xla')
        assert rule == KERNELS['gated_delta_rule'].get(generation, 'xla')
        with assumed(None):
            assert (device_generation(), triton_runs(), bf16_dot_runs()) == own
    assert (device_generation(), triton_runs(), bf16_dot_runs()) == own


@pytest.mark.parametrize("generation, platform", [("sm75", "gpu"), ("sm120", "gpu"), ("v5e", "tpu"),
                                                  ("cpu", "cpu")])
def test_an_assumed_generation_names_its_platform(generation, platform):
    own = backend()
    with assumed(generation):
        assert backend() == platform
    assert backend() == own == jax.default_backend()


def test_a_platform_keyed_choice_takes_the_assumed_gpus_path():
    """On this CPU, jax.nn's xla attention sums a long fp32 value product in one chain, so 'auto' takes
    the reference path, and an explicit Pallas grouped matmul interprets. Under an assumed T4 both
    act as on the GPU: the xla path, and XLA's ragged dot, as a T4 compiles no Triton kernel."""
    from dew.nn.attention import VALUE_BLOCK, _xla_kernel_chains
    from dew.nn.moe import grouped_matmul_kernel

    query, key = jnp.zeros((1, 8, 2, 64)), jnp.zeros((1, VALUE_BLOCK + 1, 2, 64))
    operands = (jnp.bfloat16, jnp.bfloat16)
    assert _xla_kernel_chains(query, key)
    assert grouped_matmul_kernel('pallas', jnp.bfloat16, operands, None) == 'pallas'
    with assumed('sm75'):
        assert not _xla_kernel_chains(query, key)
        assert grouped_matmul_kernel('pallas', jnp.bfloat16, operands, None) == 'xla'


def test_the_gpu_mesh_keeps_the_axes_over_the_generations_devices():
    """The device mesh's axes, sizes and types, over abstract devices of the generation's kind, which
    JAX's Pallas lowering reads the compute capability from; a generation without a kind is refused."""
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ('data', 'tensor'))
    target = gpu_mesh(mesh, 'sm120')
    assert (target.axis_names, target.axis_sizes, target.axis_types) == (
        mesh.axis_names, mesh.axis_sizes, mesh.axis_types)
    assert target.abstract_device.device_kind == DEVICE_KINDS['sm120']
    assert target.abstract_device.platform == 'cuda'
    with pytest.raises(ValueError, match="'sm81'"):
        gpu_mesh(mesh, 'sm81')


def _doubled(x):
    """A Pallas kernel that interprets wherever Dew's kernels do (`triton_runs`)."""
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                          interpret=not triton_runs(), name="doubled")(x)


def test_a_kernel_traced_for_an_assumed_gpu_lowers_to_a_named_triton_call():
    """Traced under an assumed A100 on that GPU's mesh, a kernel lowers on this CPU to one Triton call,
    by its name; traced for the CPU it interprets, and calls none."""
    x = jax.ShapeDtypeStruct((64,), jnp.float32)
    mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ('data',)), 'sm80')
    with assumed('sm80'), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(_doubled).trace(x).lower(lowering_platforms=('cuda',))
    assert triton_kernels(lowered) == {"doubled": 1}
    assert triton_kernels(jax.jit(_doubled).lower(x)) == {}


# The Triton kernels each Pallas choice a KERNELS row can name lowers to, by their `pallas_call` names,
# and the probe that runs them: the Clef recipe's step (tools/clef_step.py), or an expert layer's
# forward and backward (`_experts_lowered`). None for a kernel whose Triton lowering no CPU reaches
# (`dew.training._lowering.UNLOWERABLE`).
FAMILIES = {
    ('gated_delta_rule', 'pallas'): ("step", {"gated_delta_states", "gated_delta_states_backward",
                                              "gated_delta_prep", "gated_delta_prep_backward"}),
    ('grouped_matmul', 'pallas'): ("experts", {"pallas_triton_ragged_dot", "tgmm_ragged_dot"}),
    ('tokamax_grouped_matmul', 'triton'): ("experts", None),
}

GPUS = sorted({generation for row in KERNELS.values() for generation in row if generation.startswith('sm')}
              | {'sm80', 'sm120'}, key=lambda generation: int(generation[2:]))
"""Every CUDA generation a KERNELS row names, and the A100's and the G4's, which Dew's sessions take."""


def chosen(generation: str, probe: str) -> set[tuple[str, str]]:
    """The Pallas choices `generation`'s rows name whose kernels `probe` runs."""
    return {(choice, row[generation]) for choice, row in KERNELS.items()
            if (choice, row.get(generation)) in FAMILIES and FAMILIES[choice, row[generation]][0] == probe}


def test_every_pallas_choice_a_row_names_has_a_lowering_probe():
    """A Pallas kernel KERNELS chooses for some GPU is one this file lowers for that GPU on a CPU: a new
    row needs its kernels in FAMILIES, so no Triton lowering first runs in a GPU session's compile."""
    named = {(choice, kernel) for choice, row in KERNELS.items() for kernel in row.values()
             if kernel in PALLAS_CHOICES}
    assert named <= set(FAMILIES), sorted(named - set(FAMILIES))
    assert {key for key, (_, names) in FAMILIES.items() if names is None} == set(UNLOWERABLE)


@pytest.mark.parametrize("generation", ["sm80", "sm120"])
def test_the_clef_step_lowers_for_the_gpu_with_every_kernel_its_generation_chooses(generation, tmp_path):
    """c58's and c57's step, the Clef recipe's decision step over a Qwen 3.5 hybrid, lowered on this CPU
    for their GPUs, an A100 and a G4 (tools/clef_step.py): it lowers, and it calls the Triton kernels
    of every Pallas choice the generation's KERNELS row names for its layers, so none was
    interpreted or refused."""
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "clef_step.py"), "--lower-for", generation,
                           "--work", str(tmp_path)], capture_output=True, text=True, cwd=ROOT, timeout=1500,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr[-4000:]
    kernels = json.loads(done.stdout.splitlines()[-1])["kernels"]
    assert set(kernels) == set().union(*(FAMILIES[key][1] for key in chosen(generation, "step")))


def _experts_lowered(generation: str, implementation: str):
    """An expert layer's bf16 forward and backward, as a MoE step runs them, lowered for `generation`."""
    from dew.nn.moe import ExpertMLP

    experts = ExpertMLP(num_experts=8, hidden_features=128, out_features=64, implementation=implementation,
                        dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((256, 64), jnp.bfloat16)
    indices = jnp.asarray(np.arange(512).reshape(256, 2) % 8, jnp.int32)
    weights = jnp.full((256, 2), 0.5, jnp.float32)
    variables = jax.eval_shape(lambda: experts.init(jax.random.key(1), jnp.zeros(x.shape, x.dtype), weights,
                                                    indices))

    def loss(variables, x):
        return experts.apply(variables, x, weights, indices).astype(jnp.float32).sum()

    mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ('data',)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
        return jax.jit(jax.value_and_grad(loss)).trace(variables, x).lower(lowering_platforms=('cuda',))


@pytest.mark.parametrize("generation", GPUS)
def test_an_expert_layer_lowers_for_the_gpu_with_the_grouped_matmul_its_generation_chooses(generation):
    """'auto' takes the generation's grouped matmul: its forward and both backward products lower to
    the Pallas kernels where the row names them, and to none where it does not."""
    expected = set().union(*(FAMILIES[key][1] for key in chosen(generation, "experts")
                             if key[0] == 'grouped_matmul'))
    assert set(triton_kernels(_experts_lowered(generation, 'auto'))) == expected


@pytest.mark.parametrize("generation", [generation for generation in GPUS
                                        if 'tokamax_grouped_matmul' in dict(chosen(generation, "experts"))])
def test_tokamax_lowers_for_the_gpu_its_row_names(generation):
    """'tokamax', an opt-in implementation 'auto' never takes, names its Triton kernel per generation;
    tokamax registers that kernel only on a host with a GPU, so a CPU's dry run skips it, and the
    kernel lab's verify on the GPU runs it (tests/test_moe.py)."""
    api = pytest.importorskip("tokamax._src.ops.ragged_dot.api")
    if 'triton' not in api.IMPLEMENTATIONS:
        pytest.skip("tokamax registers its Triton ragged dot only where a GPU is present")
    assert triton_kernels(_experts_lowered(generation, 'tokamax'))
