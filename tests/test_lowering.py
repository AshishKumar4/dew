"""A training step lowered for a CUDA GPU on a host without one (`dew.training._lowering`)."""

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.sharding import Mesh

from dew.nn.kernels.generation import (
    PALLAS_CHOICES,
    TARGETS,
    GpuTarget,
    Refusal,
    Route,
    architecture,
    assumed,
    chosen as kernel_choices,
    device_generation,
    family,
    fp8_tensor_cores,
    gpu_mesh,
    gpu_target,
    measured_kernel,
    route,
    routes,
    runtime_facts,
    triton_runs,
)
from dew.nn.kernels.templates import bundles
from dew.training._lowering import custom_calls, triton_kernels

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
    ('gated_delta_rule', 'fla'): ("step", {"recurrence_cumulated", "fla_kkt_solve", "fla_recompute_w_u",
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
    kernels, those of every Pallas choice the generation's row names for a step's layers (the norms and
    SwiGLU), and the gated delta rule's training kernels (`recurrence_chunk.GATED_DELTA`, no row read):
    the recurrence template's Mosaic launches on a warpgroup-MMA GPU, whose training cell it takes, no
    Triton kernel among them, else the port's, so none was interpreted or refused there."""
    delta = set() if family(generation) == 'wgmma' else FAMILIES['gated_delta_rule', 'fla'][1]
    expected = SHORT_CONV | delta | set().union(*(FAMILIES[key][1] for key in chosen(generation, "step")))
    assert set(triton_kernels(_step_lowered(generation))) == expected


@pytest.mark.parametrize("bound", [None, 64], ids=["undeclared", "declared"])
def test_a_packed_delta_net_lowers_onto_the_port_whatever_documents_its_rows_declare(bound):
    """A bf16 gated delta net over packed rows of 128 tokens in 8 documents, their bound undeclared
    (so the row's length, 128, far more than its two chunks) or declared at 64, lowered for an A100 on
    this CPU: its forward and backward run on the port's kernels, which keep only the memories passing
    between a document's chunks, whatever the bound."""
    from dew.batch import coordinates
    from dew.nn.inputs import AttentionMetadata, MixerCall
    from dew.nn.linear import GatedDeltaNet

    layer = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=2, head_k_dim=32, head_v_dim=32,
                          dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((2, 128, 64), jnp.bfloat16)
    segments = jnp.asarray(np.repeat(np.arange(1, 9), 16)[None].repeat(2, 0), jnp.int32)
    call = MixerCall(AttentionMetadata(coordinates=coordinates(segments=segments, max_segments=bound)))
    variables = jax.eval_shape(lambda: layer.init(jax.random.key(0), jnp.zeros(x.shape, x.dtype)))

    def loss(variables, x):
        return layer.apply(variables, x, call).astype(jnp.float32).sum()

    mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ('data',)), "sm80")
    with assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(jax.value_and_grad(loss)).trace(variables, x).lower(lowering_platforms=('cuda',))
    assert FAMILIES[('gated_delta_rule', 'fla')][1] <= set(triton_kernels(lowered))


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
    """'auto' runs the grouped matmul's kernels on every GPU generation, each forward and backward
    launch of each projection the route its cell takes there (its route says which): the template's
    Mosaic GPU launch, or the old route `grouped_matmul.DECLARED` lists. On an Ampere-class GPU the
    forward runs Mosaic GPU's and the gradients Triton's, whose two kernels lower; on one whose family has
    no bundle (`bundles.of`: consumer Blackwell's) Triton's runs all three; on Hopper Triton's runs only
    where a launch runs it, Mosaic GPU's lowering as no Triton kernel. The experts'
    SiLU product takes the fused SwiGLU where the row names it, forward and backward, as every fused
    kernel is differentiated through its portable arithmetic (`generation.linear_tangent`)."""
    with routes() as taken:
        lowered = _experts_lowered(generation, 'auto')
    # The layer's own launches, as traced for `generation`; its initialization runs on the CPU.
    kernels = {route.kernel for route in taken
               if route.choice == 'grouped_matmul' and route.target == generation}
    grouped = {"pallas_triton_ragged_dot", "tgmm_ragged_dot"}
    assert kernels and kernels <= {'triton', 'mosaic_gpu'}
    lowered_grouped = set(triton_kernels(lowered)) & grouped
    if not gpu_target(generation).wgmma:
        bundled = bundles.of(generation, jnp.bfloat16) is not None
        assert kernels == ({'mosaic_gpu', 'triton'} if bundled else {'triton'}) and lowered_grouped == grouped
    assert bool(lowered_grouped) == ('triton' in kernels)
    swiglu = set().union(*(FAMILIES[key][1] for key in chosen(generation, "step") if key[0] == 'swiglu'))
    assert set(triton_kernels(lowered)) - grouped == swiglu


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
        return tuning.tune(lower)[0]


@pytest.mark.parametrize("generation", GPUS)
def test_a_mamba2_layer_lowers_for_the_gpu_with_the_ssd_scan_its_generation_chooses(generation):
    """A Mamba-2 layer, its launches settled as the trainer settles them, lowers to the short conv's
    kernels and mamba_ssm's fourteen ported kernels, which take the scan on any GPU Dew's Triton kernels
    run on (`recurrence_chunk.SCAN`, no row read). Its unweighted norm takes the fused RMSNorm where the
    row names it."""
    expected = SHORT_CONV | FAMILIES['ssd', 'pallas'][1]
    expected |= set().union(*(FAMILIES[key][1] for key in chosen(generation, "step") if key[0] == "rms_norm"))
    assert set(triton_kernels(_mamba2_lowered(generation))) == expected


def test_a_mamba2_layer_lowers_its_ssd_scan_on_each_shard_of_a_tpu_mesh():
    """On a TPU mesh splitting its rows, a Mamba-2 layer's Pallas scan lowers to Mosaic on each shard
    (`manual_map`), as Mosaic needs: lowered for an assumed TPU v5p on an abstract two-device mesh,
    forward and backward, each scan's custom call holds one row of the two, at the layer's widths."""
    from jax.sharding import AbstractDevice, AbstractMesh, AxisType

    from dew.nn.mixers.mamba2 import Mamba2

    layer = Mamba2(emb_features=128, num_heads=4, head_dim=64, state_size=64, chunk_size=64,
                   dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((2, 128, 128), jnp.bfloat16)
    variables = jax.eval_shape(lambda: layer.init(jax.random.key(1), jnp.zeros(x.shape, x.dtype)))

    def loss(variables, x):
        return jnp.asarray(layer.apply(variables, x)).astype(jnp.float32).sum()

    device = AbstractDevice(device_kind=TARGETS["v5p"].device_kind, num_cores=None, platform="tpu")
    mesh = AbstractMesh((2,), ("data",), axis_types=(AxisType.Auto,), abstract_device=device)
    with assumed("v5p"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(jax.value_and_grad(loss)).trace(variables, x).lower(lowering_platforms=("tpu",))
    calls = [call for call in custom_calls(lowered) if call.target == "tpu_custom_call"]
    assert len(calls) == 2, calls
    for call in calls:
        assert {(1, 128, 4, 64), (1, 4, 128, 64)} & {*call.operands, *call.results}, call


@pytest.mark.parametrize("generation, cores", [
    ("sm80", False), ("sm86", False), ("sm89", True), ("sm90", True), ("sm100", True), ("sm120", True),
    ("v6e", False), ("cpu", False)])
def test_fp8_tensor_cores_are_a_fact_of_the_compute_capability(generation, cores):
    """Capability 8.9 and later multiplies FP8, whether or not the target table names the generation."""
    assert fp8_tensor_cores(generation) is cores


@pytest.mark.parametrize("generation, claimed", [
    ("sm75", None), ("sm80", "mma.sync"), ("sm86", "mma.sync"), ("sm87", "mma.sync"),
    ("sm89", "mma.sync"), ("sm90", "wgmma"), ("sm100", "tcgen05"), ("sm103", "tcgen05"),
    ("sm110", "tcgen05"), ("sm120", "mma.sync"), ("sm121", "mma.sync"), ("sm70", None), ("sm60", None),
    ("v5e", "tpu"), ("cpu", None)])
def test_a_generations_instruction_family_is_its_compute_capabilitys(generation, claimed):
    """The family a template's mainloop claims comes from the compute capability alone, a GPU the table
    names or not (sm87, sm110 and sm121 it does not); a Turing GPU (no bf16 MMA), one before it or a CPU
    is in none."""
    assert family(generation) == claimed


def test_an_unseen_member_of_a_family_takes_its_familys_facts_and_its_own_device_attributes():
    """A Blackwell GPU no row names (sm110) ranks as the family's: its tensor rates, tensor memory and
    model latency are tcgen05's, and its L2 bandwidth is the family's bytes a multiprocessor a clock over
    the multiprocessors and clock its device reports, so the cost model ranks its configs."""
    from dew.nn.kernels.templates import cost

    target = GpuTarget("sm110", "an unseen Blackwell", 227 * 1024, multiprocessors=20, clock_hz=1.5e9,
                       memory_bandwidth=273e9)
    blackwell = architecture("sm100")
    assert blackwell is not None
    assert (target.tensor_rates, target.tmem_columns, target.wgmma) == (blackwell.tensor_rates, 512, False)
    assert target.load_latency == blackwell.load_latency
    assert target.l2_bandwidth == blackwell.l2_bytes_per_clock * 20 * 1.5e9
    assert cost.seconds(cost.Cost(programs=40, device_bytes=1e6, program_flops=1e9), target) > 0


def test_a_gpu_the_table_does_not_name_keeps_the_resources_its_device_reports():
    """A CUDA device of a generation no row names (a B200, sm100) takes its multiprocessors and its
    block's shared memory from the device, and Hopper's warpgroup MMA from its capability, 9.0 alone,
    so a launch sized by them sizes itself rather than taking a default for want of a row."""
    blackwell = runtime_facts(GpuTarget("sm100", "NVIDIA B200"), SimpleNamespace(
        compute_capability="10.0", core_count=148, shared_memory_per_block_optin=232448, local_hardware_id=0,
        device_kind="NVIDIA B200"))
    hopper = runtime_facts(GpuTarget("sm90", "NVIDIA H100 80GB HBM3"), SimpleNamespace(
        compute_capability="9.0", core_count=132, shared_memory_per_block_optin=232448, local_hardware_id=0,
        device_kind="NVIDIA H100 80GB HBM3"))
    assert (blackwell.multiprocessors, blackwell.shared_limit, blackwell.wgmma) == (148, 232448, False)
    assert (hopper.multiprocessors, hopper.shared_limit, hopper.wgmma) == (132, 232448, True)


@pytest.mark.gpu(reason="reads a CUDA device's attributes")
def test_a_gpu_reports_its_block_and_multiprocessors_whether_or_not_a_row_names_it():
    """The default device's shared memory a block and multiprocessors come from JAX's device: a bare
    target of its generation, holding none of the table's resources, takes the same ones the
    default device's target reports, and the row's own where the table records them."""
    device = jax.devices()[0]
    named = gpu_target()
    bare = runtime_facts(GpuTarget(device_generation(), device.device_kind), device)
    assert bare.shared_limit is not None and bare.multiprocessors == device.core_count
    assert (bare.shared_limit, bare.multiprocessors, bare.wgmma) == (
        named.shared_limit, named.multiprocessors, named.wgmma)
    row = TARGETS.get(device_generation())
    if row is not None and row.shared_limit is not None:
        assert named.shared_limit == row.shared_limit


@pytest.mark.gpu(reason="reads a CUDA device's clocks from its driver")
def test_a_gpu_reports_its_clock_and_memory_bandwidth_from_its_driver():
    """The default device's multiprocessor clock and memory bandwidth, which JAX's device does not
    carry, come from the CUDA driver at the device's own ordinal (`local_hardware_id`, which a process
    seeing a subset of the GPUs numbers from zero): positive, and a GPU's bandwidth between that of the
    slowest and the fastest memory Dew's targets name."""
    target = gpu_target()
    assert target.clock_hz > 1e8
    rows = [row.memory_bandwidth for row in TARGETS.values() if row.memory_bandwidth]
    assert min(rows) / 2 < target.memory_bandwidth < 2 * max(rows)


@pytest.mark.gpu(reason="two processes, each seeing one of two GPUs")
@pytest.mark.skipif(jax.default_backend() != "gpu" or len(jax.devices()) < 2, reason="needs two GPUs")
def test_each_process_reads_the_facts_of_the_gpu_it_sees(tmp_path):
    """Two processes, each seeing one GPU (`CUDA_VISIBLE_DEVICES`): the second's device is global id 1
    and its own ordinal 0, and each process's kernel target holds the CUDA driver's clock, bandwidth and
    multiprocessors for that GPU, the same in both, as GPUs of one kind give."""
    from process_support import ProcessGroup, free_port, report_of, start_process

    worker = Path(__file__).with_name("device_facts_worker.py")
    coordinator = f"127.0.0.1:{free_port()}"
    outs = [tmp_path / f"process{rank}.json" for rank in range(2)]
    # The test's own two GPUs, as the run allotted them, one to each process, which allocates on demand
    # beside this process's preallocation.
    visible = [str(device.local_hardware_id) for device in jax.local_devices()[:2]]
    allotted = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allotted:
        visible = [allotted.split(",")[int(ordinal)] for ordinal in visible]
    with ProcessGroup() as pool:
        for rank, out in enumerate(outs):
            pool.running.append(start_process(
                [sys.executable, str(worker), str(rank), coordinator, str(out)], devices=1,
                environment={"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": visible[rank],
                             "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}))
        reports = [report_of(process, out, timeout=300, pool=pool.running)
                   for process, out in zip(pool.running, outs, strict=True)]
    placed = [(report["processes"], report["id"], report["ordinal"]) for report in reports]
    assert placed == [(2, 0, 0), (2, 1, 0)]
    for report in reports:
        assert report["clock_hz"] == report["driver_clock_hz"]
        assert report["memory_bandwidth"] == report["driver_bandwidth"]
        assert report["multiprocessors"] == report["driver_multiprocessors"]
    assert reports[0]["clock_hz"] == reports[1]["clock_hz"]


def test_a_run_over_gpus_of_two_kinds_is_refused():
    """Each process's kernel choices read its first GPU's facts and must agree with every other's, so a
    run whose processes' first GPUs are of two kinds is refused; first GPUs of one kind pass, whatever
    else a process holds."""
    from dew.nn.kernels.generation import _one_kind

    def gpu(identifier, process, kind):
        return SimpleNamespace(id=identifier, process_index=process, device_kind=kind)

    h100, a100 = "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"
    _one_kind([gpu(0, 0, h100), gpu(1, 1, h100)])
    # One process's second GPU of another kind reads no facts.
    _one_kind([gpu(0, 0, h100), gpu(1, 0, a100), gpu(2, 1, h100), gpu(3, 1, a100)])
    with pytest.raises(ValueError):
        _one_kind([gpu(0, 0, h100), gpu(1, 1, a100)])


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


def test_a_function_jitted_before_a_choice_traces_again_within_it():
    @jax.jit
    def ported(x):
        return x + (measured_kernel("ssd", "xla") == "pallas")

    x = jnp.zeros((), jnp.int32)
    with kernel_choices(ssd="xla"):
        assert ported(x) == 0
        with kernel_choices(ssd="pallas"):
            assert ported(x) == 1
        assert ported(x) == 0


def test_a_choice_no_dispatch_reads_is_refused_by_name():
    with pytest.raises(ValueError), kernel_choices(gated_delta="fla"):
        pass


def test_a_call_the_dispatch_sends_off_flash_is_logged_at_info_and_a_float32_call_not_at_all():
    """FlashAttention takes no float32 query, so a float32 call declines nothing and says so nowhere; a
    bf16 call the dispatch sends off the kernel measured fastest for the call's own facts (packed
    documents with no bound on a row's, on an A100) is the dispatch's own choice, logged once at INFO,
    which a run's console does not show. The kernel it names is the target's, xla, as cuDNN takes no
    segment ids."""
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
        with generation.assumed("sm80"), generation.routes() as found:
            for dtype in (jnp.float32, jnp.bfloat16, jnp.bfloat16):
                query = jnp.zeros((1, 8, 2, 64), dtype)
                attention.resolve_implementation("auto", query, query, segment_ids=jnp.ones((1, 8)))
        print(json.dumps({"selected": found[-1].kernel, "records": records}))
    ''')
    fresh = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True,
                           timeout=120, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    logged = json.loads(fresh.stdout)
    assert logged["selected"] == "xla"
    assert logged["records"] == [[logging.INFO, ["attention", "xla", "flash"]]]


def test_a_route_records_the_targets_decision_where_its_dispatch_reports_it():
    """`route` records a dispatch's decision in every `routes` block open, an inner block's in the
    outer's too, as the target decides it: its generation within `assumed`, the kernel the row or
    `chosen` prefers, a refusal's stable code and the row that tracks it, what the host ran, and the
    reporting function."""
    def dispatch(kernel, refusal=None, ran=None):
        return route("ssd", kernel=kernel, refusal=refusal, ran=ran)

    with assumed("sm90"), routes() as outer:
        assert dispatch("pallas").kernel == "pallas"
        row = measured_kernel("ssd", "pallas")
        head_dim = Refusal("head_dim", "a 320-wide head", "inherent")
        with kernel_choices(ssd="pallas"), routes() as inner:
            dispatch("xla", head_dim, ran="xla")
    test = test_a_route_records_the_targets_decision_where_its_dispatch_reports_it
    site = f"{__name__}:{test.__qualname__}.<locals>.dispatch"
    assert inner == [Route("ssd", "xla", "pallas", head_dim, "xla", "sm90", site)]
    assert outer == [Route("ssd", "pallas", row, None, "pallas", "sm90", site), *inner]


def test_a_dispatch_sending_a_call_off_its_preferred_kernel_reports_the_route(without_deterministic_ops):
    """The attention dispatch, sending a packed bf16 call with no bound on a row's documents off the
    flash kernel an A100 measured fastest, reports through the attention selector the target's route:
    xla, as cuDNN takes no segment ids, the refusal's code and its declaration (inherent), and xla as
    what ran, never a host fact as a refusal; a call flash takes reports flash and no refusal."""
    from dew.nn import attention

    query = jnp.zeros((1, 8, 2, 64), jnp.bfloat16)
    with assumed("sm80"), routes() as found:
        ran = attention.resolve_implementation("auto", query, query, segment_ids=jnp.ones((1, 8)))
        attention.resolve_implementation("auto", query, query, segment_ids=jnp.ones((1, 8)), max_segments=2)
    off, taken = [found_route for found_route in found if found_route.choice == "attention"]
    assert (off.kernel, off.preferred, off.ran, off.target) == ("xla", "flash", ran, "sm80")
    assert (off.refusal.code, off.refusal.row) == ("document_bound", "inherent")
    assert off.site == "dew.nn.attention:resolve_implementation"
    assert (taken.kernel, taken.preferred, taken.refusal) == ("flash", "flash", None)


def test_a_call_off_a_kernel_for_its_cost_is_a_tracked_gap_and_one_past_its_tiles_inherent(
        without_deterministic_ops):
    """A packed call past FlashAttention's measured document cap goes off it for its cost, which the route
    report counts as a gap tracked by its measurement's row, not as a limit; heads wider than any fused
    kernel tiles are a limit no implementation of flash lifts, inherent (`tools/kernel_routes.verdict`)."""
    from dew.nn import attention
    from tools.kernel_routes import verdict

    packed, wide = jnp.zeros((1, 8, 2, 64), jnp.bfloat16), jnp.zeros((1, 8, 2, 512), jnp.bfloat16)
    with assumed("sm80"), routes() as found:
        attention.resolve_implementation("auto", packed, packed, segment_ids=jnp.ones((1, 8)),
                                         max_segments=99)
        attention.resolve_implementation("auto", wide, wide, causal=True)
    capped, widest = [found_route for found_route in found if found_route.choice == "attention"]
    assert capped.refusal.code == "packed_documents" and verdict(capped).startswith("gap: measured")
    assert widest.refusal.code == "head_dim" and verdict(widest) == "inherent"


def test_a_function_traced_before_a_routes_block_reports_its_routes_within_it():
    """A jitted function traced and cached before a `routes` block, warm in an inner jit too, reports
    its routes again in each block: every block is a fresh collection JAX keys its traces on."""
    @jax.jit
    def inner(x):
        route("ssd", kernel="xla")
        return x + 1

    @jax.jit
    def outer(x):
        return inner(x) * 2

    x = jnp.zeros(3)
    jax.block_until_ready(outer(x))
    for _ in range(2):
        with routes() as found:
            jax.block_until_ready(outer(x))
        assert [found_route.choice for found_route in found] == ["ssd"]
