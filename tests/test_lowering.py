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

from dew.nn.kernels.generation import PALLAS_CHOICES, TARGETS, assumed, gpu_mesh, gpu_target, triton_runs
from dew.training._lowering import triton_kernels

ROOT = Path(__file__).resolve().parents[1]


def _doubled(x):
    """A Pallas kernel that interprets wherever Dew's kernels do (`triton_runs`)."""
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                          interpret=not triton_runs(), name="doubled")(x)


def test_a_kernel_traced_for_an_assumed_gpu_lowers_to_a_named_triton_call():
    """Traced under an assumed A100 on that GPU's mesh, a kernel lowers on any host to one Triton call,
    by its name; traced for a CPU it interprets, and calls none."""
    x = jax.ShapeDtypeStruct((64,), jnp.float32)
    mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ('data',)), 'sm80')
    with assumed('sm80'), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(_doubled).trace(x).lower(lowering_platforms=('cuda',))
    assert triton_kernels(lowered) == {"doubled": 1}
    with jax.default_device(jax.devices('cpu')[0]):
        assert triton_kernels(jax.jit(_doubled).lower(x)) == {}


# The Triton kernels each Pallas choice a KERNELS row can name lowers to, by their `pallas_call` names,
# and the probe that runs them: the Clef recipe's step (tools/clef_step.py), or an expert layer's
# forward and backward (`_experts_lowered`).
FAMILIES = {
    ('grouped_matmul', 'pallas'): ("experts", {"pallas_triton_ragged_dot", "tgmm_ragged_dot"}),
    ('rms_norm', 'pallas'): ("step", {"rms_norm_forward", "rms_norm_backward"}),
    ('swiglu', 'pallas'): ("step", {"swiglu_forward", "swiglu_backward"}),
    ('gated_delta_rule', 'fla'): ("step", {"fla_chunk_local_cumsum", "fla_kkt_solve", "fla_recompute_w_u",
                                           "fla_chunk_fwd_h", "fla_chunk_fwd_o", "fla_chunk_bwd_dv_local",
                                           "fla_chunk_bwd_dhu", "fla_chunk_bwd_dqkwg",
                                           "fla_prepare_wy_repr_bwd"}),
}

GPUS = sorted({generation for generation, target in TARGETS.items()
               if generation.startswith('sm') and target.kernels}
              | {'sm80', 'sm120'}, key=lambda generation: int(generation[2:]))
"""Every CUDA generation a KERNELS row names, and the A100's and the G4's, which Dew's sessions take."""

SHORT_CONV = {"short_conv_forward", "short_conv_backward"}
"""The short conv's kernels (`causal_conv1d`), which every GPU step with a recurrent mixer calls."""


def chosen(generation: str, probe: str) -> set[tuple[str, str]]:
    """The Pallas choices `generation`'s rows name whose kernels `probe` runs."""
    return {(choice, kernel) for choice, kernel in gpu_target(generation).kernels.items()
            if (choice, kernel) in FAMILIES and FAMILIES[choice, kernel][0] == probe}


def test_every_pallas_choice_a_row_names_has_a_lowering_probe():
    """A Pallas kernel KERNELS chooses for some GPU is one this file lowers for that GPU on a CPU: a new
    row needs its kernels in FAMILIES, so no Triton lowering first runs in a GPU session's compile."""
    named = {(choice, kernel) for target in TARGETS.values() for choice, kernel in target.kernels.items()
             if kernel in PALLAS_CHOICES}
    assert named <= set(FAMILIES), sorted(named - set(FAMILIES))


@pytest.mark.parametrize("generation", ["sm80", "sm120"])
def test_the_compensated_norm_vjp_lowers_for_both_gpu_generations(generation):
    """Explicit Pallas checks the candidate even before that generation opts in.

    The reverse pass's compensated sums and their rounded assembly must lower
    too; interpreting on CPU cannot expose broken multi-result GPU assembly.
    The programs' partial sums meet in an XLA reduction, not a third kernel.
    """
    from dew.nn.attention import rms_normalized

    def forward_backward(x, scale, seed):
        def norm(x, scale):
            return rms_normalized(x, scale, 1e-5, jnp.bfloat16, scale_offset=True,
                                   scale_after_cast=True, fp32_statistics=True, implementation='pallas')
        y, backward = jax.vjp(norm, x, scale)
        return y, *backward(seed)

    x = jax.ShapeDtypeStruct((32, 17), jnp.bfloat16)
    scale = jax.ShapeDtypeStruct((17,), jnp.float32)
    mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ('data',)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(forward_backward).trace(x, scale, x).lower(lowering_platforms=('cuda',))
    assert triton_kernels(lowered) == {'rms_norm_forward': 1, 'rms_norm_backward': 1}


@pytest.mark.parametrize("generation", ["sm80", "sm120"])
def test_the_clef_step_lowers_for_the_gpu_with_every_kernel_its_generation_chooses(generation, tmp_path):
    """c58's and c57's step, the Clef recipe's decision step over a Qwen 3.5 hybrid, lowered on this CPU
    for their GPUs, an A100 and a G4 (tools/clef_step.py): it lowers, and it calls the Triton kernels
    of the short conv and of every Pallas choice the generation's KERNELS row names for its layers, the
    gated delta rule's port among them, so none was interpreted or refused."""
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "clef_step.py"), "--lower-for", generation,
                           "--work", str(tmp_path)], capture_output=True, text=True, cwd=ROOT, timeout=1500,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr[-4000:]
    kernels = json.loads(done.stdout.splitlines()[-1])["kernels"]
    assert set(kernels) == SHORT_CONV.union(*(FAMILIES[key][1] for key in chosen(generation, "step")))


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
