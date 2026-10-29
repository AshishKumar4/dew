"""A training step lowered for a CUDA GPU on a host without one (`dew.training._lowering`)."""

import json
import logging
import math
import os
import re
import sys
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
    chosen_kernel,
    device_generation,
    family,
    fp8_tensor_cores,
    gpu_mesh,
    gpu_target,
    route,
    routes,
    runtime_facts,
    triton_runs,
)
from dew.telemetry import instrumentation
from dew.training._lowering import custom_calls, kernel_report, triton_kernels

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


# The Triton kernels each Pallas kernel a selector can take lowers to, by their `pallas_call` names,
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

GPUS = ['sm80', 'sm89', 'sm90', 'sm120']
"""The CUDA generations Dew's sessions take and its measured cells name: the A100's, the L4's, the H100's
and the G4's."""

GMM = 2 * 256 * 128 * 128
"""The multiply-adds of each grouped product of the step `_reported` lowers: forward, input and weight
gradients of 256 rows by 128 by 128."""


def _reported(generation: str):
    """A grouped GEMM's forward and backward beside a Triton kernel, lowered on this CPU for `generation`'s
    GPU, with the routes its dispatches reported (`routes`)."""
    from dew.nn.kernels.templates.gemm import template

    def step(x, w, sizes):
        def loss(x, w):
            return template.gemm(x, w, sizes, out_dtype=jnp.bfloat16).astype(jnp.float32).sum()

        return jax.grad(loss, argnums=(0, 1))(x, w), _doubled(x)

    x, w = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16), jax.ShapeDtypeStruct((2, 128, 128), jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((2,), jnp.int32)
    mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ('data',)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh), routes() as found:
        lowered = jax.jit(step).trace(x, w, sizes).lower(lowering_platforms=("cuda",))
    return lowered, found


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_steps_kernel_report_names_its_launches_and_the_flop_count_sizes_them_as_it_does(generation):
    """A grouped GEMM's step lowered for the GPU (row 1895): its kernel report lists the program's
    launches, the template's by kernel name with the FLOPs each states, a Triton launch's stating none,
    and the FLOP count of the same program counts the stated FLOPs and leaves out, as unsized, the
    launches the report does."""
    from dew.telemetry.instrumentation import hlo_flops

    lowered, found = _reported(generation)
    record = kernel_report(found, lowered).record()
    launches = {entry["name"]: entry for entry in record["launches"]}
    templated = [entry for entry in record["launches"] if entry["kind"] == "mosaic_gpu"]
    assert all(entry["flops"] == entry["calls"] * GMM for entry in templated)
    assert launches["doubled"]["kind"] == "triton" and launches["doubled"]["flops"] is None
    unstated = sum(entry["calls"] for entry in record["launches"] if entry["flops"] is None)
    assert record["unsized"] == unstated
    flops = hlo_flops(lowered.as_text(dialect="hlo"))
    assert flops is not None
    assert (flops.hardware, flops.unsized) == (sum(entry["flops"] for entry in templated), unstated)


def test_a_cutlass_grouped_gemm_is_sized_by_its_shapes_and_an_unknown_call_is_unsized():
    """dew_cutlass's grouped GEMMs, lowered for the GPU, count each token row's product once in the FLOP
    count and in the kernel report; a custom call nothing sizes counts as an unsized launch in both."""
    from dew.telemetry.instrumentation import hlo_flops

    tokens, kernel = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16), jax.ShapeDtypeStruct((2, 128, 64),
                                                                                          jnp.bfloat16)
    cotangent, sizes = jax.ShapeDtypeStruct((256, 64), jnp.bfloat16), jax.ShapeDtypeStruct((2,), jnp.int32)

    def step(tokens, kernel, cotangent, sizes):
        out = jax.ffi.ffi_call("dew_cutlass_grouped_gemm", jax.ShapeDtypeStruct((256, 64), jnp.bfloat16))(
            tokens, kernel, sizes, transpose_w=False)
        gradient = jax.ffi.ffi_call("dew_cutlass_grouped_gemm_k", jax.ShapeDtypeStruct(
            (2, 128, 64), jnp.bfloat16))(tokens, cotangent, sizes)
        unknown = jax.ffi.ffi_call("an_unknown_kernel", jax.ShapeDtypeStruct((256, 64), jnp.bfloat16))(out)
        return unknown, gradient

    lowered = jax.jit(step).trace(tokens, kernel, cotangent, sizes).lower(lowering_platforms=("cuda",))
    record = kernel_report((), lowered).record()
    assert [(entry["kind"], entry["name"], entry["calls"]) for entry in record["launches"]] == [
        (None, "an_unknown_kernel", 1), ("grouped_gemm", "dew_cutlass_grouped_gemm", 1),
        ("grouped_gemm", "dew_cutlass_grouped_gemm_k", 1)]
    assert record["unsized"] == 1
    flops = hlo_flops(lowered.as_text(dialect="hlo"))
    assert flops is not None
    assert (flops.hardware, flops.unsized) == (2 * 2 * 256 * 128 * 64, 1)


def test_a_tpu_grouped_gemm_states_its_cost_to_the_flop_count():
    """The TPU grouped GEMM's two launches, lowered for a TPU on this CPU, state their products'
    FLOPs (`templates.gemm.cost`), which the kernel report and the FLOP count both read."""
    from dew.nn.kernels.templates.gemm import mosaic_tpu
    from dew.telemetry.instrumentation import hlo_flops

    blocks = mosaic_tpu.Blocks(128, 128, 128)
    lhs, rhs = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16), jax.ShapeDtypeStruct((2, 128, 128),
                                                                                    jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((2,), jnp.int32)

    def step(lhs, rhs, sizes):
        out = mosaic_tpu.gmm(lhs, rhs, sizes, blocks=blocks, transpose_rhs=False, out_dtype=jnp.bfloat16)
        return out, mosaic_tpu.tgmm(lhs, out, sizes, blocks=blocks, out_dtype=jnp.bfloat16)

    with assumed("v6e"):
        lowered = jax.jit(step).trace(lhs, rhs, sizes).lower(lowering_platforms=("tpu",))
    record = kernel_report((), lowered).record()
    assert [(entry["kind"], entry["name"], entry["flops"]) for entry in record["launches"]] == [
        ("mosaic_tpu", "dew_gmm", GMM), ("mosaic_tpu", "dew_tgmm", GMM)]
    assert record["unsized"] == 0
    flops = hlo_flops(lowered.as_text(dialect="hlo"))
    assert flops is not None
    assert (flops.hardware, flops.unsized) == (2 * GMM, 0)


def test_every_pallas_kernel_a_measured_cell_runs_has_a_lowering_probe():
    """A Pallas kernel a selector's measured cell runs on some GPU (`Op.declared`) is one this file lowers
    for that GPU on a CPU: a new cell needs its kernels in FAMILIES, so no Triton lowering first runs in a
    GPU session's compile."""
    from dew.nn.kernels import grouped_matmul, norm_swiglu

    named = {(op.choice, kernel) for op in (norm_swiglu.NORM, norm_swiglu.SWIGLU, grouped_matmul.GROUPED)
             for kernel, _ in op.declared.values() if kernel in PALLAS_CHOICES}
    assert named <= set(FAMILIES), sorted(named - set(FAMILIES))


@pytest.mark.parametrize("generation", ["sm80", "sm120"])
def test_the_compensated_norm_vjp_lowers_for_both_gpu_generations(generation):
    """Explicit Pallas checks the candidate even before that generation opts in.

    The reverse pass's compensated sums and their rounded assembly must lower
    too; interpreting on CPU cannot expose broken multi-result GPU assembly.
    """
    from dew.nn.norms import rms_normalized

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
        jax.jit(forward_backward).trace(x, scale, x).lower(lowering_platforms=('cuda',))


def _step_lowered(generation: str):
    """A bf16 Qwen 3.5 hybrid's language-model training step (tests/fixtures/hf/qwen35-tiny: three gated
    delta net layers and one attention layer, its delta nets' heads 32 wide, a power of two as a real
    model's and the short conv's kernels take), lowered for `generation` from abstract state as
    tools/kernel_picks.py lowers one (`Trainer(lower_for=...)`): the `Lowered` its dry run raises."""
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
    return lowered.value


@pytest.mark.parametrize("generation", GPUS)
def test_a_hybrid_language_model_step_lowers_for_every_gpu(generation):
    """A Qwen 3.5 hybrid's training step, its gated delta nets, attention, norms and SwiGLU, lowers on this
    CPU for every GPU Dew's sessions take (`GPUS`): its dry run raises the `Lowered` program."""
    _step_lowered(generation)


@pytest.mark.parametrize("bound", [None, 64], ids=["undeclared", "declared"])
def test_a_packed_delta_net_lowers_for_an_a100_whatever_documents_its_rows_declare(bound):
    """A bf16 gated delta net over packed rows of 128 tokens in 8 documents, their bound undeclared
    (so the row's length, 128, far more than its two chunks) or declared at 64: its forward and backward
    lower for an A100 on this CPU, whatever the bound."""
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
        jax.jit(jax.value_and_grad(loss)).trace(variables, x).lower(lowering_platforms=('cuda',))


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


COLLECTIVES = re.compile(r"stablehlo\.(all_gather|all_reduce|all_to_all|reduce_scatter|collective_permute|"
                         r"collective_broadcast)")
"""The collectives a lowered program states itself (a `shard_map`'s; XLA's partitioner adds its own later)."""

@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_kernel_launches_only_where_every_axis_of_its_mesh_is_manual(generation):
    """Row 1923: the toolkit's launch point refuses, as it traces, a kernel under a mesh axis larger than
    one device that XLA's partitioner would split around it (`sharding.per_shard`); inside a `shard_map`
    holding that axis manual, each device's launch reads its own rows and lowers."""
    from jax.sharding import AbstractMesh, AxisType, PartitionSpec

    from dew.nn.kernels import grouped_matmul
    from dew.nn.kernels.templates.gemm import mosaic_gpu

    x, w = jax.ShapeDtypeStruct((512, 128), jnp.bfloat16), jax.ShapeDtypeStruct((2, 128, 128), jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((2,), jnp.int32)
    with assumed(generation):
        target = gpu_target(generation)
        launch = grouped_matmul.MosaicGmm()
        bound = launch.bind(x, w, sizes, interpret=False, out_dtype="bfloat16", trans_rhs=False)
        tile = launch.derived(bound, target)

    def launched(x, w, sizes):
        return mosaic_gpu.gmm(x, w, sizes, tiles=tile, transpose_rhs=False, out_dtype=jnp.bfloat16,
                              multiprocessors=target.multiprocessors)

    mesh = gpu_mesh(AbstractMesh((2,), ("data",), axis_types=(AxisType.Auto,)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
        with pytest.raises(ValueError):
            jax.jit(launched).trace(x, w, sizes)
        per_device = jax.shard_map(launched, mesh=mesh, in_specs=(PartitionSpec("data"), PartitionSpec(),
                                                                  PartitionSpec()),
                                   out_specs=PartitionSpec("data"), check_vma=False)
        jax.jit(per_device).trace(x, w, sizes).lower(lowering_platforms=("cuda",))


def _expert_layer_step(rows: int):
    """An expert layer's loss and its arguments over `rows` tokens, top-2 of 8 experts: its grouped GEMM
    launches' rows are the routed slots, two a token."""
    from dew.nn.moe import ExpertMLP

    experts = ExpertMLP(num_experts=8, hidden_features=128, out_features=64, dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((rows, 64), jnp.bfloat16)
    indices = jnp.asarray(np.arange(rows * 2).reshape(rows, 2) % 8, jnp.int32)
    weights = jnp.full((rows, 2), 0.5, jnp.float32)
    variables = jax.eval_shape(lambda: experts.init(jax.random.key(1), jnp.zeros(x.shape, x.dtype), weights,
                                                    indices))

    def loss(variables, x):
        return experts.apply(variables, x, weights, indices).astype(jnp.float32).sum()

    return loss, (variables, x)


def _norm_step(rows: int):
    """A residual stream's fused RMSNorm over `[2, rows / 2, 256]`, its scale replicated."""
    from dew.nn.kernels import norm_swiglu

    def portable(x, scale):
        wide = x.astype(jnp.float32)
        return (wide * jax.lax.rsqrt(jnp.mean(wide * wide, -1, keepdims=True) + 1e-6) * scale).astype(x.dtype)

    def loss(x, scale):
        y = norm_swiglu.rms_norm(x, scale, 1e-6, jnp.bfloat16, offset=False, after_cast=False,
                                 portable=portable,
                                 axes=("activation_batch", "activation_length", "activation_embed"))
        return y.astype(jnp.float32).sum()

    return loss, (jax.ShapeDtypeStruct((2, rows // 2, 256), jnp.bfloat16),
                  jax.ShapeDtypeStruct((256,), jnp.float32))


def _swiglu_step(rows: int):
    """A feed-forward's fused SiLU product over `[2, rows / 2, 512]`."""
    from dew.nn.kernels import norm_swiglu

    def loss(gate, up):
        def portable(gate, up):
            return (jax.nn.silu(gate.astype(jnp.float32)) * up).astype(gate.dtype)

        y = norm_swiglu.swiglu(gate, up, portable,
                               axes=("activation_batch", "activation_length", "activation_mlp"))
        return y.astype(jnp.float32).sum()

    shaped = jax.ShapeDtypeStruct((2, rows // 2, 512), jnp.bfloat16)
    return loss, (shaped, shaped)


def _short_conv_step(rows: int):
    """A depthwise causal conv over `[2, rows / 2, 256]` with its taps and bias replicated."""
    from dew.nn.linear import causal_conv1d

    def loss(x, taps, bias):
        return causal_conv1d(x, taps, activation=True, bias=bias).astype(jnp.float32).sum()

    return loss, (jax.ShapeDtypeStruct((2, rows // 2, 256), jnp.bfloat16),
                  jax.ShapeDtypeStruct((256, 4), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))


PER_SHARD_STEPS = {"expert layer": _expert_layer_step, "rms norm": _norm_step, "swiglu": _swiglu_step,
                   "short conv": _short_conv_step}
"""Each op whose kernels run per shard (row 1923, bible §7), its step at a number of rows. Each lane adds its
ops here."""


def _kernel_launches(loss, args, generation: str, devices: int) -> tuple[list, str]:
    """The Pallas launches (`custom_calls`) of `loss`'s value and gradient lowered for `generation`'s GPU on
    an abstract mesh of `devices` that splits the rows, and the program's text."""
    from jax.sharding import AbstractMesh, AxisType

    from dew.telemetry.instrumentation import custom_call_kind

    mesh = gpu_mesh(AbstractMesh((devices,), ("data",), axis_types=(AxisType.Auto,)), generation)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(jax.value_and_grad(loss, argnums=tuple(range(len(args))))).trace(*args).lower(
            lowering_platforms=("cuda",))
    calls = [call for call in custom_calls(lowered)
             if custom_call_kind(call.target) in ("triton", "mosaic_gpu")]
    return calls, lowered.as_text()


@pytest.mark.parametrize("op", PER_SHARD_STEPS)
@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_every_kernel_launch_takes_its_own_devices_rows_on_a_two_device_mesh(generation, op):
    """Bible §7, row 1923: an op's forward and backward, lowered for the GPU on an abstract two-device mesh
    that splits its rows, launches the kernels it launches on one device, each reading fewer elements than
    there (its device's rows, its replicated parameters whole), never the whole batch; and the program's
    only collective is the reduction of its replicated parameters' gradients over the data axis."""
    loss, args = PER_SHARD_STEPS[op](384)
    whole, _ = _kernel_launches(loss, args, generation, 1)
    split, text = _kernel_launches(loss, args, generation, 2)

    def elements(call) -> int:
        return sum(math.prod(shape) for shape in call.operands)

    pairs = list(zip(split, whole, strict=True)) if len(split) == len(whole) else []
    assert whole and pairs, (op, [call.target for call in whole], [call.target for call in split])
    assert all(elements(on_two) < elements(on_one) for on_two, on_one in pairs), (
        op, [(elements(on_two), elements(on_one)) for on_two, on_one in pairs])
    assert set(COLLECTIVES.findall(text)) <= {"all_reduce"}


@pytest.mark.parametrize("generation", GPUS)
def test_an_expert_layer_lowers_for_every_gpu(generation):
    """An expert layer's bf16 forward and backward under 'auto', its grouped matmul and SiLU product as
    the generation's cells run them, lowers on this CPU for every GPU a row names, and the A100's and the
    G4's."""
    _experts_lowered(generation, 'auto')


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
def test_a_mamba2_layer_lowers_for_every_gpu(generation):
    """A Mamba-2 layer, its launches settled as the trainer settles them, lowers on this CPU for every GPU
    a row names, and the A100's and the G4's, its scan, short conv and unweighted norm included."""
    _mamba2_lowered(generation)


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


Shapes = set[tuple[int, ...]]


def _packed_step(rows: int, devices: int) -> tuple[jax.stages.Lowered, Shapes]:
    """A bf16 gated delta net's forward and gradient over `rows` rows of 128 tokens, each three packed
    documents, its token coordinates built in the step as a trainer builds them once a batch
    (`dew.batch.coordinates`), lowered for an assumed H100 over an abstract mesh of `devices` splitting its
    rows (the recurrence template takes the H100's training cell), and its parameters' shapes."""
    from jax.sharding import AbstractMesh, AxisType

    from dew.batch import coordinates
    from dew.nn.inputs import AttentionMetadata, MixerCall
    from dew.nn.linear import GatedDeltaNet

    layer = GatedDeltaNet(256, 1, 2, 128, 128, dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((rows, 128, 256), jnp.bfloat16)
    segments = np.repeat(np.searchsorted([40, 100], np.arange(128), side="right")[None] + 1, rows, axis=0)
    variables = jax.eval_shape(lambda: layer.init(jax.random.key(1), jnp.zeros(x.shape, x.dtype)))

    def loss(variables, x, segments):
        metadata = AttentionMetadata(coordinates=coordinates(segments=segments, max_segments=3))
        return jnp.asarray(layer.apply(variables, x, MixerCall(metadata=metadata))).astype(jnp.float32).sum()

    device = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90").abstract_device
    mesh = AbstractMesh((devices,), ("data",), axis_types=(AxisType.Auto,), abstract_device=device)
    given = jax.ShapeDtypeStruct(segments.shape, jnp.int32)
    with assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        traced = jax.jit(jax.value_and_grad(loss)).trace(variables, x, given)
    shapes = {_squeezed(leaf.shape) for leaf in jax.tree.leaves(variables)}
    return traced.lower(lowering_platforms=("cuda",)), shapes


Operation = tuple[str, str, tuple, tuple]
"""A lowered operation: its name, its callee or custom call's target, and its operands' and results'
`(shape, element type)`."""


def _operations(lowered: jax.stages.Lowered) -> list[Operation]:
    """Every operation of a lowered program in program order (its StableHLO module's walk, as
    `dew.training._lowering.triton_calls` reads it)."""
    from jax.extend.mlir import ir

    found: list[Operation] = []

    def typed(values) -> tuple:
        tensors = [value.type for value in values if isinstance(value.type, ir.RankedTensorType)]
        return tuple((tuple(tensor.shape), str(tensor.element_type)) for tensor in tensors)

    def visit(operation: ir.Operation) -> ir.WalkResult:
        attributes = operation.attributes
        target = ""
        if "call_target_name" in attributes:
            target = ir.StringAttr(attributes["call_target_name"]).value
        elif "callee" in attributes:
            target = attributes["callee"].value
        found.append((operation.name, target, typed(operation.operands), typed(operation.results)))
        return ir.WalkResult.ADVANCE

    module = lowered.compiler_ir(dialect="stablehlo")
    assert module is not None
    with module.context:
        module.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return found


def _mosaic_calls(operations: list[Operation]) -> list[tuple]:
    """The Mosaic GPU calls among a lowered program's `operations` (`_operations`), each its operands' and
    results' shapes."""
    return [(operands, results) for name, target, operands, results in operations
            if name == "stablehlo.custom_call" and target.startswith("mosaic_gpu")]


def _collective(name: str) -> str | None:
    """The collective a lowered operation `name` is (`instrumentation.collective`, its StableHLO dialect's
    name in HLO's spelling), None for any other."""
    return instrumentation.collective(name.removeprefix("stablehlo.").replace("_", "-"))


def _squeezed(shape) -> tuple[int, ...]:
    """`shape` without its unit dimensions: a gradient's sum may reduce a parameter reshaped so (the short
    conv's `[512, 1, 4]` weight as `[512, 4]`)."""
    return tuple(size for size in shape if size != 1)


def _template_launches(policy) -> int:
    """The recurrence template's Mosaic GPU calls in a bf16 gated delta net's forward and gradient,
    lowered for an assumed H100 (whose training cell the template takes), the loss under the remat rung
    `policy` (`decoder_block.RematPolicy`; None, none)."""
    from dew.nn.backbones.decoder_block import RematPolicy
    from dew.nn.linear import GatedDeltaNet

    layer = GatedDeltaNet(256, 1, 2, 128, 128, dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((1, 128, 256), jnp.bfloat16)
    variables = jax.eval_shape(lambda: layer.init(jax.random.key(1), jnp.zeros(x.shape, x.dtype)))

    def loss(variables, x):
        return jnp.asarray(layer.apply(variables, x)).astype(jnp.float32).sum()

    if policy is not None:
        loss = jax.checkpoint(loss, policy=RematPolicy(save=policy).checkpoint_policy())
    mesh = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(jax.value_and_grad(loss)).trace(variables, x).lower(lowering_platforms=("cuda",))
    return sum(name == "stablehlo.custom_call" and target.startswith("mosaic_gpu")
               for name, target, *_ in _operations(lowered))


def test_a_remat_rung_keeping_the_delta_rules_residuals_runs_the_templates_forward_once():
    """A gated delta net under a remat rung that saves its delta rule's names (`DELTA_RULE_RESIDUALS`: its
    output and the template's inverse and memory, named as the port names its own) runs the template's
    forward launches once, as the layer without remat does, where the rung that saves nothing ('full')
    replays them in the backward: a Qwen3.5-4B training step at 4x6,144 tokens a device under 'full' ran
    each GDN layer's forward twice, 63.6 ms of its 1.6 s on an H100 (row 2023)."""
    from dew.nn.mixer_base import DELTA_RULE_RESIDUALS

    unrematted, saved, full = (_template_launches(policy) for policy in (None, DELTA_RULE_RESIDUALS, ()))
    assert unrematted == saved < full, (unrematted, saved, full)


def test_a_packed_recurrence_runs_on_each_devices_own_rows():
    """A packed gated delta net on a mesh splitting its rows runs on each device's rows, as its lowered GPU
    program states it: lowered for an H100 over two devices and two rows, its recurrence template's Mosaic
    GPU calls are one device's over one row, operand for operand (`recurrence_chunk.template_rule`, a
    per-shard map); no scan (a cumsum, a reduce window) runs along the batch's flattened tokens, whose split
    over a mesh XLA's GPU partitioner expands into a loop a token (the packed documents' offsets come from
    the batch's `Coordinates.firsts`, a sort of each row's slots); and its only collectives are its
    parameters' gradients' sums. On the whole batch every device of a 2xH100 FSDP run ran the
    template on the global rows, and each layer's offsets took 97 loops of a token an iteration (row 1897,
    bible 7's multi-device lowering). The final executable's loops need a GPU's compile; this proves their
    cause absent from the program it compiles."""
    (sharded, parameters), (local, _) = _packed_step(rows=2, devices=2), _packed_step(rows=1, devices=1)
    operations = _operations(sharded)
    calls = _mosaic_calls(operations)
    assert calls and calls == _mosaic_calls(_operations(local))
    scanned = [operands for name, target, operands, _ in operations
               if name == "stablehlo.reduce_window" or (name == "func.call" and target.startswith("cumsum"))]
    assert scanned and all(2 * 128 not in shape for operand in scanned for shape, _ in operand), scanned
    reduced = {_squeezed(shape) for name, _, _, results in operations if _collective(name)
               for shape, _ in results}
    assert reduced <= parameters, (reduced, parameters)


def _split_delta_forward(length: int, devices: int) -> jax.stages.Lowered:
    """A bf16 gated delta net's forward over one row of `length` tokens, lowered for an assumed H100 over an
    abstract mesh whose sequence axis splits it `devices` ways."""
    from jax.sharding import AbstractMesh, AxisType

    from dew.nn.linear import GatedDeltaNet

    layer = GatedDeltaNet(256, 1, 2, 128, 128, dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((1, length, 256), jnp.bfloat16)
    variables = jax.eval_shape(lambda: layer.init(jax.random.key(1), jnp.zeros(x.shape, x.dtype)))
    device = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90").abstract_device
    mesh = AbstractMesh((devices,), ("sequence",), axis_types=(AxisType.Auto,), abstract_device=device)
    with assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        return jax.jit(layer.apply).trace(variables, x).lower(lowering_platforms=("cuda",))


def test_a_gated_delta_net_split_over_the_sequence_runs_each_devices_own_slice():
    """A gated delta net on a mesh splitting its sequence runs the recurrence on each device's slice, as its
    lowered H100 program states it: over two devices and 384 tokens, every kernel call of the layer over one
    device's 192 tokens is among its calls, operand for operand, but the conv's, which reads its three-token
    halo too, beside its slice's summary (the rule from `[0 | I]`), where the rule ran the whole sequence on
    every device before (row 1437); and its collectives are the halo's and the summaries' gathers alone, no
    activation reduced (`recurrence_chunk.entering_memory`, `linear.sequence_conv`)."""
    def kernels(operations: list[Operation]) -> list[tuple]:
        return [(target, operands) for name, target, operands, _ in operations
                if name == "stablehlo.custom_call"]

    operations = _operations(_split_delta_forward(384, 2))
    calls, local = kernels(operations), kernels(_operations(_split_delta_forward(192, 1)))
    convolved = [(target, (((1, 195, 512), dtype), *rest)) for target, ((shape, dtype), *rest) in local
                 if shape == (1, 192, 512)]
    assert convolved and all(call in calls for call in convolved), (calls, local)
    assert all(call in calls for call in local if call[1][0][0] != (1, 192, 512)), (calls, local)
    collectives = {_collective(name) for name, *_ in operations if _collective(name)}
    assert collectives <= {"all-gather", "collective-permute"}, collectives


def _ssd_forward(generation: str, rows: int, devices: int) -> jax.stages.Lowered:
    """A bf16 Mamba-2 layer's forward over `rows` rows of 128 tokens, its scan on the recurrence template
    (`SCAN`'s 'mosaic_gpu', named: a declared cell may give the port the generation's forward), lowered for
    an assumed `generation` over an abstract mesh of `devices` splitting its rows."""
    from jax.sharding import AbstractMesh, AxisType

    from dew.nn.mixers.mamba2 import Mamba2

    layer = Mamba2(emb_features=128, num_heads=4, head_dim=64, state_size=64, chunk_size=64,
                   dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((rows, 128, 128), jnp.bfloat16)
    variables = jax.eval_shape(lambda: layer.init(jax.random.key(1), jnp.zeros(x.shape, x.dtype)))
    device = gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation).abstract_device
    mesh = AbstractMesh((devices,), ("data",), axis_types=(AxisType.Auto,), abstract_device=device)
    with assumed(generation), jax.sharding.use_abstract_mesh(mesh), kernel_choices(ssd="mosaic_gpu"):
        return jax.jit(layer.apply).trace(variables, x).lower(lowering_platforms=("cuda",))


def test_a_mamba2_forward_runs_its_template_scan_on_each_devices_own_rows():
    """A Mamba-2 layer's forward on a mesh splitting its rows runs the recurrence template's scan on each
    device's rows (`SCAN`'s template, `recurrence_chunk.template_rule`, a per-shard map), as its lowered H100
    program states it: lowered over two devices and two rows, its Mosaic GPU calls are one device's over one
    row, operand for operand, and it holds no collective (bible 7's multi-device lowering, row 1963). An
    A100's template refuses this scan: no config of mma.sync's fits it."""
    sharded = _operations(_ssd_forward("sm90", rows=2, devices=2))
    calls = _mosaic_calls(sharded)
    assert calls and calls == _mosaic_calls(_operations(_ssd_forward("sm90", rows=1, devices=1)))
    assert not [name for name, *_ in sharded if _collective(name)]


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


def test_chosen_kernels_keep_other_choices_and_restore_them_after_an_exception():
    with assumed("sm80"):
        with kernel_choices(attention="xla", ssd="pallas"):
            assert chosen_kernel("attention", "flash") == "xla"
            with pytest.raises(RuntimeError), kernel_choices(attention="flash"):
                assert chosen_kernel("attention", "xla") == "flash"
                assert chosen_kernel("ssd", "xla") == "pallas"
                raise RuntimeError("the comparison ended")
            assert chosen_kernel("attention", "flash") == "xla"
        assert chosen_kernel("attention", "flash") == "flash"


def test_a_function_jitted_before_a_choice_traces_again_within_it():
    @jax.jit
    def ported(x):
        return x + (chosen_kernel("ssd", "xla") == "pallas")

    x = jnp.zeros((), jnp.int32)
    with kernel_choices(ssd="xla"):
        assert ported(x) == 0
        with kernel_choices(ssd="pallas"):
            assert ported(x) == 1
        assert ported(x) == 0


def test_a_choice_no_dispatch_reads_is_refused_by_name():
    with pytest.raises(ValueError), kernel_choices(gated_delta="fla"):
        pass


def test_a_route_records_the_targets_decision_where_its_dispatch_reports_it():
    """`route` records a dispatch's decision in every `routes` block open, an inner block's in the
    outer's too, as the target decides it: its generation within `assumed`, the kernel `chosen` prefers
    (else the one it ran), a refusal's stable code and the row that tracks it, what the host ran, and the
    reporting function."""
    def dispatch(kernel, refusal=None, ran=None):
        return route("ssd", kernel=kernel, refusal=refusal, ran=ran)

    with assumed("sm90"), routes() as outer:
        assert dispatch("pallas").kernel == "pallas"
        head_dim = Refusal("head_dim", "a 320-wide head", "inherent")
        with kernel_choices(ssd="pallas"), routes() as inner:
            dispatch("xla", head_dim, ran="xla")
    test = test_a_route_records_the_targets_decision_where_its_dispatch_reports_it
    site = f"{__name__}:{test.__qualname__}.<locals>.dispatch"
    assert inner == [Route("ssd", "xla", "pallas", head_dim, "xla", "sm90", site)]
    assert outer == [Route("ssd", "pallas", "pallas", None, "pallas", "sm90", site), *inner]


def test_a_route_off_its_preferred_kernel_is_logged_once_at_info_and_one_on_it_not_at_all():
    """A dispatch off the kernel it prefers is its own choice, logged once a process at INFO for each kernel
    and refusal, naming both (`route`); a dispatch that runs its preferred kernel logs nothing, so a run's
    console shows neither."""
    from dew.nn.kernels import generation

    logger, records = logging.getLogger(generation.__name__), []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append((record.levelno, record.args[:3]))

    capture, level = Capture(), logger.level
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)
    try:
        off = Refusal("head_dim", "this test's own 300-wide head", "inherent")
        for _ in range(2):
            route("ssd", kernel="xla", preferred="pallas", refusal=off)
        route("ssd", kernel="pallas", preferred="pallas")
    finally:
        logger.removeHandler(capture)
        logger.setLevel(level)
    assert records == [(logging.INFO, ("ssd", "xla", "pallas"))]


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


@pytest.mark.parametrize("scan", ["sum", "max", "min"])
def test_a_blocked_prefix_scan_is_the_running_scan(scan):
    """`prefix_sum`, `prefix_max` and `prefix_min` (forward and from the end) are numpy's running sum,
    maximum and minimum, integer for integer, along either axis, over lengths short of one block, short of
    whole blocks and of whole ones (`SCAN_BLOCKS`)."""
    from dew.batch import SCAN_BLOCKS, prefix_max, prefix_min, prefix_sum

    rng = np.random.default_rng(0)
    for length in (5, SCAN_BLOCKS * 3 + 7, SCAN_BLOCKS * 8):
        for axis in (0, 1):
            values = rng.integers(-50, 50, (3, length) if axis else (length, 3)).astype(np.int32)
            if scan == "sum":
                np.testing.assert_array_equal(prefix_sum(values, axis), np.cumsum(values, axis))
            elif scan == "max":
                np.testing.assert_array_equal(prefix_max(values, axis), np.maximum.accumulate(values, axis))
            else:
                np.testing.assert_array_equal(prefix_min(values, axis), np.minimum.accumulate(values, axis))
                reversed_min = np.flip(np.minimum.accumulate(np.flip(values, axis), axis), axis)
                np.testing.assert_array_equal(prefix_min(values, axis, reverse=True), reversed_min)
    empty = np.zeros((2, 0), np.int32)
    for scanned in (prefix_sum(empty), prefix_max(empty), prefix_min(empty), prefix_min(empty, reverse=True)):
        assert scanned.shape == (2, 0)


def test_a_blocked_prefix_scan_keeps_cumsums_dtype_and_infinities():
    """A blocked running sum sums in `cumsum`'s dtype throughout, so a requested int8 wraps as `cumsum`'s
    does and a bool counts in the default integer; a running maximum or minimum of infinities is
    infinite, as numpy's is."""
    import jax.numpy as jnp

    from dew.batch import prefix_max, prefix_min, prefix_sum

    wrapped = prefix_sum(np.full((1, 200), 1, np.int8), dtype=jnp.int8)
    expected = jnp.cumsum(jnp.full((1, 200), 1, jnp.int8), axis=1, dtype=jnp.int8)
    np.testing.assert_array_equal(wrapped, expected)
    assert wrapped.dtype == jnp.int8
    counted = prefix_sum(np.ones((1, 130), bool))
    assert counted.dtype == jnp.cumsum(jnp.ones((1, 130), bool), axis=1).dtype and int(counted[0, -1]) == 130
    lows, highs = np.full((1, 130), -np.inf, np.float32), np.full((1, 130), np.inf, np.float32)
    np.testing.assert_array_equal(prefix_max(lows), lows)
    np.testing.assert_array_equal(prefix_min(highs), highs)
    np.testing.assert_array_equal(prefix_min(highs, reverse=True), highs)


def test_a_batchs_token_layout_split_over_its_tokens_compiles_no_loop_a_token():
    """The token layout a step builds from a packed, padded batch (its documents' starts, each token's
    rank in its document and in its row, the layers' counted places, block numbers and ends, and learned
    positions' places), compiled by XLA's CUDA pipeline for two H100s with the batch's token axis split
    between them (`compile_only_gpus`, no GPU attached), runs no loop: a running scan along a split axis
    became a loop of one iteration a token in XLA's GPU partitioner, a 2xH100 training step's tenfold
    slowdown (ledger 1905); each runs in blocks no mesh splits (`dew.batch.prefix_sum`)."""
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.batch import coordinates, real_places
    from dew.decision.model import key_runs
    from dew.nn.inputs import counted_places, numbered_blocks
    from dew.nn.visibility import block_ends
    from dew.training.ahead import compile_only_gpus

    def layout(segments, valid):
        tokens = coordinates(segments=segments, valid=valid, max_segments=4)
        return (tokens.positions, tokens.starts, real_places(valid), counted_places(segments, valid, None),
                numbered_blocks(tokens, segments.shape, 16), block_ends(segments),
                key_runs(jnp.minimum(segments, 2), segments, 2))

    mesh = Mesh(np.array(compile_only_gpus("sm90", 2)), ("sequence",))
    split = NamedSharding(mesh, PartitionSpec(None, "sequence"))
    shapes = (jax.ShapeDtypeStruct((4, 8192), jnp.int32, sharding=split),
              jax.ShapeDtypeStruct((4, 8192), jnp.bool_, sharding=split))
    program = jax.jit(layout, in_shardings=(split, split)).lower(*shapes).compile().as_text()
    assert not re.findall(r"\bwhile\(", program), re.findall(r"known_trip_count[^\n]*", program)[:4]
