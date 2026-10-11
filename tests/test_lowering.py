"""A training step lowered for a CUDA GPU on a host without one (`dew.training.lowering`)."""

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
    KERNELS,
    assumed,
    backend,
    bf16_dot_runs,
    device_generation,
    measured_kernel,
    triton_runs,
)
from dew.training.lowering import DEVICE_KINDS, gpu_mesh, triton_kernels

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
    the reference path, and an explicit Pallas RMSNorm interprets. Under an assumed T4 both act as
    on the GPU: the xla path, and XLA's RMSNorm, as a T4 compiles no Triton kernel."""
    from dew.nn.attention import VALUE_BLOCK, _xla_kernel_chains
    from dew.nn.kernels import norm_swiglu

    query, key = jnp.zeros((1, 8, 2, 64)), jnp.zeros((1, VALUE_BLOCK + 1, 2, 64))
    x = jnp.zeros((4, 64))
    assert _xla_kernel_chains(query, key)
    assert norm_swiglu.rms_kernel('pallas', x, None, jnp.float32, fp32_statistics=True) == 'pallas'
    with assumed('sm75'):
        assert not _xla_kernel_chains(query, key)
        assert norm_swiglu.rms_kernel('pallas', x, None, jnp.float32, fp32_statistics=True) == 'xla'


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


# The kernels of each Pallas choice a generation's KERNELS row names, as the step calls them.
FAMILIES = {
    ('rms_norm', 'pallas'): {"rms_norm_forward", "rms_norm_backward"},
    ('swiglu', 'pallas'): {"swiglu_forward", "swiglu_backward"},
    ('gated_delta_rule', 'fla'): {"fla_chunk_local_cumsum", "fla_kkt_solve", "fla_recompute_w_u",
                                  "fla_chunk_fwd_h", "fla_chunk_fwd_o", "fla_chunk_bwd_dhu",
                                  "fla_chunk_bwd_dv_local", "fla_chunk_bwd_dqkwg", "fla_prepare_wy_repr_bwd"},
}


@pytest.mark.parametrize("generation", ["sm80", "sm120"])
def test_the_clef_step_lowers_for_the_gpu_with_every_kernel_its_generation_chooses(generation, tmp_path):
    """c58's and c57's step, the Clef recipe's decision step over a Qwen 3.5 hybrid, lowered on this CPU
    for their GPUs, an A100 and a G4 (tools/clef_step.py): it lowers, and it calls the Triton kernels
    of every Pallas choice the generation's KERNELS row names, so none was interpreted or refused."""
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "clef_step.py"), "--lower-for", generation,
                           "--work", str(tmp_path)], capture_output=True, text=True, cwd=ROOT, timeout=1500,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr[-4000:]
    kernels = json.loads(done.stdout.splitlines()[-1])["kernels"]
    chosen = {(choice, row[generation]) for choice, row in KERNELS.items() if generation in row}
    expected = set().union(*(names for key, names in FAMILIES.items() if key in chosen))
    assert set(kernels) == expected
