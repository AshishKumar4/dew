"""A training step lowered for a CUDA GPU on a host without one (`dew.training._lowering`)."""

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.sharding import Mesh

from dew.nn.kernels.generation import (
    PALLAS_CHOICES,
    TARGETS,
    assumed,
    chosen as kernel_choices,
    gpu_mesh,
    gpu_target,
    measured_kernel,
    triton_runs,
)
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
# and the probe that runs them: an expert layer's forward and backward (`_experts_lowered`), a
# Mamba-2 layer's (`_mamba2_lowered`), or a model step ("step").
FAMILIES = {
    ('grouped_matmul', 'pallas'): ("experts", {"pallas_triton_ragged_dot", "tgmm_ragged_dot"}),
    ('rms_norm', 'pallas'): ("step", {"rms_norm_forward", "rms_norm_backward"}),
    ('swiglu', 'pallas'): ("step", {"swiglu_forward", "swiglu_backward"}),
    ('gated_delta_rule', 'fla'): ("step", {"fla_chunk_local_cumsum", "fla_kkt_solve", "fla_recompute_w_u",
                                           "fla_chunk_fwd_h", "fla_chunk_fwd_o", "fla_chunk_bwd_dv_local",
                                           "fla_chunk_bwd_dhu", "fla_chunk_bwd_dqkwg",
                                           "fla_prepare_wy_repr_bwd"}),
    ('ssd', 'pallas'): ("mamba2", {f"mamba_{name}" for name in (
        "chunk_cumsum_fwd", "chunk_state_fwd", "state_passing_fwd", "bmm_chunk_fwd", "chunk_scan_fwd",
        "chunk_scan_bwd_dstates", "state_passing_bwd", "chunk_scan_chunk_state_bwd_dx", "chunk_state_bwd_db",
        "chunk_scan_bwd_dc", "chunk_scan_bwd_dcb", "bmm_chunk_bwd", "chunk_scan_bwd_ddacs_stable",
        "chunk_cumsum_bwd")}),
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


def _step_lowered(generation: str):
    """A bf16 Qwen 3.5 hybrid's language-model training step (tests/fixtures/hf/qwen35-tiny: three gated
    delta net layers and one attention layer, its delta nets' heads 32 wide, a power of two as a real
    model's and the short conv's kernels take), lowered for `generation` from abstract state as
    tools/kernel_picks.py lowers one (`Trainer(lower_for=...)`)."""
    import optax

    from dew.interop.hf_decoders import translate_config
    from dew.nn.backbones import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer
    from dew.training._lowering import Lowered

    config = json.loads((ROOT / "tests" / "fixtures" / "hf" / "qwen35-tiny" / "config.json").read_text())
    fields = translate_config({**config.get("text_config", config), "linear_key_head_dim": 32,
                               "linear_value_head_dim": 32})
    model = CausalTransformer(**{**fields, "dtype": jnp.bfloat16})
    trainer = Trainer(LMObjective(model, 128), optax.adamw(1e-4), key=jax.random.key(0),
                      lower_for=generation)
    state = jax.eval_shape(trainer.initial_state, trainer.objective.initializer, trainer.key)
    with pytest.raises(Lowered) as lowered:
        # A row for each of the host's devices, which the trainer's data axis splits.
        trainer.compile(state, {"text": jax.ShapeDtypeStruct((jax.device_count(), 129), jnp.int32)})
    return lowered.value.lowered


@pytest.mark.parametrize("generation", GPUS)
def test_a_hybrid_language_model_step_lowers_for_the_gpu_with_every_kernel_its_generation_chooses(generation):
    """A Qwen 3.5 hybrid's training step, lowered on this CPU for the GPU, calls the short conv's
    kernels and those of every Pallas choice the generation's row names for a step's layers (the
    norms, SwiGLU and the gated delta rule's port), so none was interpreted or refused there."""
    expected = SHORT_CONV.union(*(FAMILIES[key][1] for key in chosen(generation, "step")))
    assert set(triton_kernels(_step_lowered(generation))) == expected


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


def _mamba2_lowered(generation: str):
    """A bf16 Mamba-2 layer's forward and backward, as a hybrid's step runs them, lowered for
    `generation` after its launches are settled as the trainer settles a step's (`tuning.tune`, which
    lowers it with every launch's description and reads them back)."""
    from dew.nn.kernels import tuning
    from dew.nn.mixers.mamba2 import Mamba2

    layer = Mamba2(emb_features=128, num_heads=4, head_dim=64, state_size=64, chunk_size=64,
                   dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((2, 128, 128), jnp.bfloat16)
    variables = jax.eval_shape(lambda: layer.init(jax.random.key(1), jnp.zeros(x.shape, x.dtype)))

    def loss(variables, x):
        return jnp.asarray(layer.apply(variables, x)).astype(jnp.float32).sum()

    def lower():
        return jax.jit(jax.value_and_grad(loss)).trace(variables, x).lower(lowering_platforms=('cuda',))

    mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ('data',)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
        tuning.tune(lower)
        return lower()


@pytest.mark.parametrize("generation", GPUS)
def test_a_mamba2_layer_lowers_for_the_gpu_with_the_ssd_scan_its_generation_chooses(generation):
    """A Mamba-2 layer, its launches settled as the trainer settles them, lowers to the short conv's
    kernels and, where the row names the port for the scan, mamba_ssm's fourteen ported kernels; where
    it does not, to none of them."""
    expected = SHORT_CONV.union(*(FAMILIES[key][1] for key in chosen(generation, "mamba2")))
    assert set(triton_kernels(_mamba2_lowered(generation))) == expected


def test_chosen_kernels_keep_other_choices_and_restore_the_row_after_an_exception():
    with assumed("sm80"):
        measured = measured_kernel("attention", "xla")
        with kernel_choices(attention="xla", ssd="pallas"):
            assert measured_kernel("attention", "flash") == "xla"
            with pytest.raises(RuntimeError), kernel_choices(attention="flash"):
                assert measured_kernel("attention", "xla") == "flash"
                assert measured_kernel("ssd", "xla") == "pallas"
                raise RuntimeError("the comparison ended")
            assert measured_kernel("attention", "flash") == "xla"
        assert measured_kernel("attention", "xla") == measured


def test_a_call_the_dispatch_sends_off_flash_is_logged_at_info_and_a_float32_call_not_at_all():
    """FlashAttention takes no float32 query, so a float32 call declines nothing and says so nowhere; a
    bf16 call the dispatch sends off the kernel measured fastest is the dispatch's own choice, logged
    once at INFO, which a run's console does not show."""
    program = textwrap.dedent('''
        import json
        import logging
        import jax.numpy as jnp
        from dew.nn import attention
        from dew.nn.kernels import generation

        records = []
        class Capture(logging.Handler):
            def emit(self, record):
                records.append([record.levelno, record.args[:3]])

        logger = logging.getLogger(generation.__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(Capture())
        with generation.assumed("sm80"):
            selected = {}
            for dtype in (jnp.float32, jnp.bfloat16, jnp.bfloat16):
                query = jnp.zeros((1, 8, 2, 64), dtype)
                selected[str(dtype)] = attention.resolve_implementation(
                    "auto", query, query, segment_ids=jnp.ones((1, 8)), max_segments=2)
        print(json.dumps({"selected": selected[str(jnp.bfloat16)], "records": records}))
    ''')
    fresh = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True,
                           timeout=120, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    logged = json.loads(fresh.stdout)
    assert logged["records"] == [[logging.INFO, ["attention", logged["selected"], "flash"]]]
