"""tools/attribute_step.py: where a profiled step's device time goes, by part and kind.

The rows here are a Qwen3.5-0.8B training step's own, on an A100: XProf's hlo_stats
rows and the optimized HLO they ran, cut to the instructions each case reads. A kind is read
from the op, never from its operands' names, which its expression also holds: FlashAttention's
backward takes an operand called `%input_concatenate_fusion.8` and is attention, not a layout
move, and a weight-gradient GEMM traced at `up_proj/transpose` is a GEMM.
"""

import json

import pytest
from jax.profiler import ProfileData
from tool_support import load

attribute = load("attribute_step")

MODULE = """HloModule jit_step

%fused_multiply_transpose.8 (param_0: f32[8,16,3072], param_1: f32[8,16,48,64]) -> f32[8,16,3072] {
  %param_1 = f32[8,16,48,64]{3,2,1,0} parameter(1)
  %exp.178 = f32[8,16,48,64]{3,2,1,0} exponential(%param_1)
  %transpose.12 = f32[8,48,64,16]{3,2,1,0} transpose(%exp.178), dimensions={0,2,3,1}
  %param_0 = f32[8,16,3072]{2,1,0} parameter(0)
  ROOT %multiply.3 = f32[8,16,3072]{2,1,0} multiply(%param_0, %param_0)
}

%fused_convert.83 (param_0.1: f32[8,3072,8,256]) -> bf16[8,3072,8,256] {
  %param_0.1 = f32[8,3072,8,256]{3,2,1,0} parameter(0)
  ROOT %convert.4 = bf16[8,3072,8,256]{3,2,1,0} convert(%param_0.1)
}

%fused_reduce.89 (param_0.2: f32[8,16,3072]) -> f32[8,16] {
  %param_0.2 = f32[8,16,3072]{2,1,0} parameter(0)
  %constant.1 = f32[] constant(0)
  ROOT %reduce.7 = f32[8,16]{1,0} reduce(%param_0.2, %constant.1), dimensions={2}, to_apply=%add
}
"""
ENTRY = [
    "  %p0 = bf16[8,3072,8,256]{3,2,1,0} parameter(0)",
    "  %ffi_call.114 = (bf16[8,3072,8,256]{3,2,1,0}, bf16[8,3072,8,256]{3,2,1,0}) custom-call("
    "%loop_convert_fusion.83, %input_concatenate_fusion.8), custom_call_target=\"dew_flash_mha_bwd\", "
    "api_version=API_VERSION_TYPED_FFI",
    "  %dot_product_attention_bwd.144 = (bf16[8,16,2800,64]{3,1,2,0}, u8[0]{0}) custom-call("
    "%bitcast.13109.0, %bitcast.13113.0), custom_call_target=\"__cudnn$fmhaSoftmaxBackward\"",
    "  %pallas_call.525 = (f32[6144,64,128]{2,1,0}) custom-call(%bitcast.478.0), "
    "custom_call_target=\"__gpu$xla.gpu.triton\"",
    "  %custom-call.1589 = (bf16[1024,3584]{1,0}, s8[4194304]{0}) custom-call(%bitcast.remat.237, "
    "%bitcast.2097.0), custom_call_target=\"__cublas$lt$matmul\"",
    "  %input_multiply_transpose_fusion.8 = f32[8,16,3072]{2,1,0} fusion(%a, %b), kind=kInput, "
    "calls=%fused_multiply_transpose.8",
    "  %loop_convert_fusion.83 = bf16[8,3072,8,256]{3,2,1,0} fusion(%p0), kind=kLoop, "
    "calls=%fused_convert.83",
    "  %input_reduce_fusion.89 = f32[8,16]{1,0} fusion(%c), kind=kInput, calls=%fused_reduce.89",
    "  ROOT %copy.5 = bf16[8,3072,8,256]{2,3,1,0} copy(%p0)",
]
MODULE += "ENTRY %main.1 (p0: bf16[8,3072,8,256]) -> bf16[8,3072,8,256] {\n" + "\n".join(ENTRY) + "\n}\n"

LAYER = ("jit(step)/transpose(jvp(MultimodalTransformer.hidden_states))/language_model.hidden_states/"
         "language_model.stack/layers_11/layers_11._forward/layers_11._mix/self_attn/")


def row(name, expression, framework, category="custom-call"):
    """An hlo_stats row as XProf writes it, its expression cut where XProf cuts it."""
    return {"hlo_op_name": name, "hlo_op_expression": expression[:400], "tf_op_name": framework,
            "category": category}


ROWS = {
    "flash backward": (row("ffi_call.114", "%ffi_call.114 = (bf16[8,3072,8,256]{3,2,1,0}, bf16[8,3072,8,256]"
                           "{3,2,1,0}) custom-call(bf16[8,3072,8,256]{3,2,1,0} %loop_convert_fusion.83, bf16"
                           "[8,3072,8,256]{3,2,1,0} %input_concatenate_fusion.8, bf16[8,3072,2,2",
                           LAYER + "self_attn._attended/ffi_call:"), "attention"),
    "cuDNN attention backward": (row("dot_product_attention_bwd.144", "%dot_product_attention_bwd.144 = "
                                     "(bf16[8,16,2800,64]{3,1,2,0}, u8[0]{0}) custom-call(bf16[8,2800,16,64]"
                                     "{3,2,1,0} %bitcast.13109.0",
                                     "jit(step)/transpose(jvp(JointSchemaHead))/evidence_layers_1/attention/"
                                     "dot_product_attention_bwd:"), "attention"),
    "Pallas recurrence": (row("pallas_call.525", "%pallas_call.525 = (bf16[8,3072,16,128]{3,2,1,0}) "
                              "custom-call(bf16[8,3072,16,128]{3,2,1,0} %bitcast.478.0",
                              LAYER + "fla_chunk_bwd_dqkwg/pallas_call:"), "pallas"),
    "weight-gradient GEMM": (row("custom-call.1589", "%custom-call.1589 = (bf16[1024,3584]{1,0}, "
                                 "s8[4194304]{0}) custom-call(bf16[24576,1024]{1,0} %bitcast.remat.237",
                                 LAYER.replace("_mix/self_attn/", "mlp/up_proj/") + "transpose:"), "gemm"),
    "transposing fusion": (row("input_multiply_transpose_fusion.8", "%input_multiply_transpose_fusion.8 = "
                               "f32[8,16,3072]{2,1,0} fusion(f32[8,16,3072]{2,1,0} "
                               "%input_transpose_fusion.225",
                               LAYER + "jit(softplus)/mul:", "input fusion"), "layout"),
    "dtype cast": (row("loop_convert_fusion.83", "%loop_convert_fusion.83 = bf16[8,3072,8,256]{3,2,1,0} "
                       "fusion("
                       "f32[8,3072,8,256]{3,2,1,0} %p0", LAYER + "convert_element_type:", "loop fusion"),
                   "elementwise"),
    "reduction": (row("input_reduce_fusion.89", "%input_reduce_fusion.89 = f32[8,16]{1,0} fusion("
                      "f32[8,16,3072]"
                      "{2,1,0} %c", LAYER + "reduce_sum:", "input fusion"), "reduce"),
    "layout copy": (row("copy.5", "%copy.5 = bf16[8,3072,8,256]{2,3,1,0} copy(bf16[8,3072,8,256]{3,2,1,0} "
                        "%p0)",
                        LAYER + "copy:", "data formatting"), "layout"),
}


TRACED = ('metadata={op_name="jit(step)/transpose(jvp(MultimodalTransformer.hidden_states))/'
          'language_model.hidden_states/layers_3/input_layernorm/OP" '
          'source_file="/root/dew/src/dew/nn/attention.py" source_line=292}')
NAMELESS = "\n".join([
    "HloModule jit_step",
    "",
    "%fused_norm.4 (param_0: f32[8,3072,1024]) -> (f32[8,3072], f32[8,3072,1024]) {",
    "  %param_0 = f32[8,3072,1024]{2,1,0} parameter(0)",
    "  %multiply.9 = f32[8,3072,1024]{2,1,0} multiply(%param_0, %param_0), " + TRACED.replace("OP", "mul"),
    "  %constant.1 = f32[] constant(0)",
    "  %reduce.2 = f32[8,3072]{1,0} reduce(%multiply.9, %constant.1), dimensions={2}, to_apply=%add, "
    + TRACED.replace("OP", "reduce_sum"),
    "  ROOT %tuple.1 = (f32[8,3072]{1,0}, f32[8,3072,1024]{2,1,0}) tuple(%reduce.2, %multiply.9)",
    "}",
    "",
    "ENTRY %main.2 (p0: f32[8,3072,1024]) -> (f32[8,3072], f32[8,3072,1024]) {",
    "  %p0 = f32[8,3072,1024]{2,1,0} parameter(0)",
    "  ROOT %input_reduce_fusion.4 = (f32[8,3072]{1,0}, f32[8,3072,1024]{2,1,0}) fusion(%p0), kind=kInput, "
    "calls=%fused_norm.4",
    "}",
    "",
])


COLUMNS = ["program_id", "category", "hlo_op_name", "hlo_op_expression", "tf_op_name", "occurrences",
           "total_self_time", "source_info"]
"""XProf's hlo_stats columns every capture here reports, in its order."""


def captured(tmp_path, cells, peak: float, extra: tuple[str, ...] = ()):
    """A capture's reports under `tmp_path`/profile, as XProf writes them: `cells`, rows of `COLUMNS` and
    then the `extra` columns, and a roofline of `peak` GFLOP/s. Returns the reports directory."""
    reports = tmp_path / "profile" / "capture-a" / "reports"
    reports.mkdir(parents=True)
    (reports / "hlo_stats.json").write_text(json.dumps([{
        "cols": [{"id": name} for name in (*COLUMNS, *extra)],
        "rows": [{"c": [{"v": value} for value in values]} for values in cells]}]))
    (reports / "roofline_model.json").write_text(json.dumps({"p": {"peak_flop_rate": peak}}))
    return reports


def traced(tmp_path, kernels):
    """The capture's trace beside its reports, as JAX writes it (plugins/profile/<run>/<host>.xplane.pb):
    `kernels`, each (stream, name, start ms, end ms, program id or None for a transfer no program
    launches), on one GPU's plane, its events carrying XLA's program_id stat."""
    names = sorted({name for _, name, *_ in kernels})
    lines = []
    for number, stream in enumerate(sorted({stream for stream, *_ in kernels}), 1):
        events = "".join(
            f"events {{ metadata_id: {names.index(name) + 1} offset_ps: {round(start * 1e9)} "
            f"duration_ps: {round((end - start) * 1e9)}"
            + ("" if program is None else f" stats {{ metadata_id: 1 int64_value: {program} }}") + " } "
            for each, name, start, end, program in kernels if each == stream)
        lines.append(f'lines {{ id: {number} name: "{stream}" timestamp_ns: 0 {events}}}')
    metadata = "".join(f'event_metadata {{ key: {index} value {{ id: {index} name: "{name}" }} }} '
                       for index, name in enumerate(names, 1))
    text = (f'planes {{ id: 1 name: "/device:GPU:0" {" ".join(lines)} {metadata}'
            'stat_metadata { key: 1 value { id: 1 name: "program_id" } } }')
    run = tmp_path / "profile" / "capture-a" / "plugins" / "profile" / "run"
    run.mkdir(parents=True)
    (run / "host.xplane.pb").write_bytes(ProfileData.text_proto_to_serialized_xspace(text))


def dumped(tmp_path, name: str, text: str):
    """An XLA dump directory under `tmp_path` holding one optimized module `name`."""
    dump = tmp_path / "hlo"
    dump.mkdir()
    (dump / name).write_text(text)
    return dump


def test_a_nameless_fusion_is_attributed_by_the_ops_it_fuses(tmp_path):
    """A multi-output fusion carries no op_name of its own (that step's 50 ms of
    "other"): its part and direction come from the ops fused inside it."""
    fusion = ("%input_reduce_fusion.4 = (f32[8,3072]{1,0}, f32[8,3072,1024]{2,1,0}) "
              "fusion(f32[8,3072,1024]{2,1,0} %p0)")
    captured(tmp_path, [("15", "input fusion", "input_reduce_fusion.4", fusion, "", "1", 2000.0, "")],
             311869.0)
    dump = dumped(tmp_path, "module_0015.jit_step.sm_8.0_gpu_after_optimizations.txt", NAMELESS)
    out = tmp_path / "attribution.json"
    assert attribute.main(["--trace", str(tmp_path / "profile"), "--hlo", str(dump), "--out", str(out)]) == 0
    time = json.loads(out.read_text())["time"]
    assert time["by_part_kind_ms"] == {"norm": {"reduce": 2.0}}
    assert time["by_part_ms"] == {"norm": {"backward": 2.0}}


UPDATE = "\n".join([
    "HloModule jit_step",
    "",
    "%fused_update.7 (param_0: f32[1024], param_1: f32[1024], param_2: f32[1024]) -> bf16[1024] {",
    "  %param_0 = f32[1024]{0} parameter(0)",
    "  %param_1 = f32[1024]{0} parameter(1)",
    "  %param_2 = f32[1024]{0} parameter(2)",
    "  %divide.3 = f32[1024]{0} divide(%param_1, %param_2)",
    "  %subtract.4 = f32[1024]{0} subtract(%param_0, %divide.3)",
    "  ROOT %convert.5 = bf16[1024]{0} convert(%subtract.4), "
    "metadata={op_name=\"jit(step)/convert_element_type\" "
    "source_file=\"/root/dew/src/dew/training/narrow.py\" source_line=94}",
    "}",
    "",
    "ENTRY %main.3 (w: f32[1024], mu: f32[1024], nu: f32[1024]) -> bf16[1024] {",
    "  %w = f32[1024]{0} parameter(0), metadata={op_name=\"current.variables['params']['w']\"}",
    "  %mu = f32[1024]{0} parameter(1), metadata={op_name=\"current.opt_state[0].mu['w']\"}",
    "  %nu = f32[1024]{0} parameter(2), metadata={op_name=\"current.opt_state[0].nu['w']\"}",
    "  %bitcast.6 = f32[1024]{0} bitcast(%nu)",
    "  ROOT %loop_convert_fusion.7 = bf16[1024]{0} fusion(%w, %mu, %bitcast.6), kind=kLoop, "
    "calls=%fused_update.7, metadata={op_name=\"jit(step)/convert_element_type\" "
    "source_file=\"/root/dew/src/dew/training/narrow.py\" source_line=94}",
    "}",
    "",
])


def test_a_fusion_that_reads_optimizer_state_is_the_optimizers(tmp_path):
    """AdamW's update of a leaf is one fusion, named by its root, the parameter narrowed for
    the next step's compute (that step's 12 ms of "other" on an A100): the moments it reads say it is
    the optimizer's."""
    fusion = ("%loop_convert_fusion.7 = bf16[1024]{0} fusion(f32[1024]{0} %w, f32[1024]{0} %mu, "
              "f32[1024]{0} %bitcast.6)")
    captured(tmp_path, [("3", "loop fusion", "loop_convert_fusion.7", fusion,
                         "jit(step)/convert_element_type", "1", 3000.0, "src/dew/training/narrow.py:94")],
             311869.0)
    dump = dumped(tmp_path, "module_0003.jit_step.sm_8.0_gpu_after_optimizations.txt", UPDATE)
    out = tmp_path / "attribution.json"
    assert attribute.main(["--trace", str(tmp_path / "profile"), "--hlo", str(dump), "--out", str(out)]) == 0
    time = json.loads(out.read_text())["time"]
    assert time["by_part_kind_ms"] == {"optimizer": {"elementwise": 3.0}}
    # A capture with no accelerator timeline (no trace here) says so: its device time is unknown, not 0.
    assert time["device_busy_ms_per_step"] is None and time["op_self_time_ms_per_step"] == 3.0


def test_a_capture_attributes_its_time_by_part_and_kind(tmp_path):
    """Through the command: the step's time by kind and by part and kind, each op once."""
    times = {"flash backward": 6000.0, "Pallas recurrence": 2000.0, "transposing fusion": 1000.0,
             "weight-gradient GEMM": 500.0}
    cells = [("15", found["category"], found["hlo_op_name"], found["hlo_op_expression"], found["tf_op_name"],
              "2", time * 2, "", 1500.0) for case, time in times.items() for found in [ROWS[case][0]]]
    cells.append(("7", "custom-call", "other.1", "", "other program", "1", 15000.0, "", 0))
    captured(tmp_path, cells, 311869.0, extra=("hbm_bw",))
    # Two steps on the device: the step program's kernels on two streams, overlapping from 4 to 6 ms,
    # a batch's transfer, which no program launches, and another program's kernel in its gap.
    compute, other, copies = "Stream #14(Compute)", "Stream #20(Compute)", "Stream #15(MemcpyH2D)"
    traced(tmp_path, [(compute, "flash_bwd", 0, 6, 15), (compute, "gemm", 10, 12, 15),
                      (other, "pallas_recurrence", 4, 8, 15), (compute, "other", 8, 9, 7),
                      (copies, "MemcpyH2D", 9.5, 10, None)])
    dump = dumped(tmp_path, "module_0015.jit_step.sm_8.0_gpu_after_optimizations.txt", MODULE)
    out = tmp_path / "attribution.json"
    assert attribute.main(["--trace", str(tmp_path / "profile"), "--hlo", str(dump),
                           "--out", str(out), "--steps", "2"]) == 0
    time = json.loads(out.read_text())["time"]
    assert time["step_program"] == "15"
    assert time["programs_ms"] == {"15": 9.5, "7": 7.5}
    # Its window is 12 ms over two steps; busy the union of its kernels and the transfer, 8 + 2.5 ms,
    # though their self times sum to 19 ms, as concurrent kernels each count there; of the 1.5 ms gap,
    # the other program ran 1 and the device was idle 0.5.
    assert {key: time[f"device_{key}_ms_per_step"] for key in (
        "window", "busy", "compute", "exposed_communication", "other_programs", "idle")} == {
        "window": 6.0, "busy": 5.25, "compute": 5.25, "exposed_communication": 0.0, "other_programs": 0.5,
        "idle": 0.25}
    assert time["op_self_time_ms_per_step"] == 9.5
    assert time["by_kind_ms"] == {"attention": 6.0, "pallas": 2.0, "layout": 1.0, "gemm": 0.5}
    assert time["by_part_kind_ms"]["attention"] == {"attention": 6.0}
    assert time["by_part_kind_ms"]["mlp"] == {"gemm": 0.5}
    # 1,500 GiB/s over the transposing fusion's millisecond is 1.5 GiB a step.
    assert {record["us"]: record["hbm_mib"] for record in time["top_ops"]}[1000.0] == 1536.0


def test_the_peak_is_dews_dense_bf16_figure_for_a_named_device(tmp_path):
    """XProf's roofline gives an RTX PRO 6000 a 3,742 TFLOP/s peak, a sparse low-precision figure:
    every MFU field reads Dew's dense bf16 table where the entry names the device, and none is
    written where it does not, XProf's figure kept beside for reference."""
    gemm = "%custom-call.1 = bf16[8,8]{1,0} custom-call()"
    captured(tmp_path, [("15", "custom-call", "custom-call.1", gemm, "jit(step)/dot_general", "1", 1000.0,
                         "")], 3742430.0)
    traced(tmp_path, [("Stream #14(Compute)", "cublas_gemm", 0, 1, 15)])
    entry = tmp_path / "entry.json"
    entry.write_text(json.dumps({"model_flops_per_step": 2.5e11,
                                 "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}))
    out = tmp_path / "attribution.json"
    command = ["--trace", str(tmp_path / "profile"), "--out", str(out), "--entry", str(entry)]
    assert attribute.main(command) == 0
    time = json.loads(out.read_text())["time"]
    assert time["peak_gflops"] == 500e3
    assert time["mfu_bf16_of_device_time"] == time["mfu_mixed_of_device_time"] == 0.5
    assert time["bf16_equiv_throughput_tflops"] == 250.0
    entry.write_text(json.dumps({"model_flops_per_step": 2.5e11}))
    assert attribute.main(command) == 0
    time = json.loads(out.read_text())["time"]
    assert time["xprof_peak_gflops"] == 3742430.0
    assert (time["peak_gflops"], time["mfu_bf16_of_device_time"], time["mfu_mixed_of_device_time"],
            time["gemm_lost_ms"], time["gemm_fraction_of_peak"]) == (None, None, None, None, None)
    assert time["device_busy_ms_per_step"] == 1.0


CONTRACT_1 = ('backend_config={"gemm_backend_config":{"dot_dimension_numbers":'
              '{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"]}}}')
GEMMS = "\n".join([
    "HloModule jit_step",
    "",
    "ENTRY %main.9 (a: f32[5000,5000], b: f32[5000,5000], c: f8e4m3fn[10000,8000], "
    "d: f8e4m3fn[8000,5000]) -> bf16[10000,5000] {",
    "  %a = f32[5000,5000]{1,0} parameter(0)",
    "  %b = f32[5000,5000]{1,0} parameter(1)",
    "  %c = f8e4m3fn[10000,8000]{1,0} parameter(2)",
    "  %d = f8e4m3fn[8000,5000]{1,0} parameter(3)",
    "  %custom-call.1 = f32[5000,5000]{1,0} custom-call(f32[5000,5000]{1,0} %a, f32[5000,5000]{1,0} %b), "
    f'custom_call_target="__cublas$gemm", {CONTRACT_1}',
    "  ROOT %custom-call.2 = bf16[10000,5000]{1,0} custom-call(f8e4m3fn[10000,8000]{1,0} %c, "
    f'f8e4m3fn[8000,5000]{{1,0}} %d), custom_call_target="__cublas$lt$matmul$f8", {CONTRACT_1}',
    "}",
    ""])


def test_a_gemm_is_judged_at_its_compiled_flops_against_its_own_precisions_peak(tmp_path):
    """A GEMM's work is its compiled FLOPs, counted from the op as the step's count counts it, over
    the measured time; XProf's rates are not read. Its model rate put a cuBLAS weight-gradient GEMM
    of that step at 4x its work, and its normalized rate scales by bf16's peak over its own guess of the
    op's precision (a later run's GEMMs at 1.127 of the A100's). Each GEMM's fraction is of its own
    precision's peak in Dew's table, as mfu_mixed counts it."""
    cells = [("15", "custom-call", "custom-call.1", '%custom-call.1 = f32[8,8]{1,0} custom-call('
              'f32[8,8]{1,0} %a, f32[8,8]{1,0} %b), custom_call_target="__cublas$gemm"',
              "jit(step)/dot_general", "1", 1000.0, "", 8.0e6, 1.0e6),
             ("15", "custom-call", "custom-call.2", '%custom-call.2 = bf16[8,8]{1,0} custom-call('
              'f8e4m3fn[8,8]{1,0} %a, f8e4m3fn[8,8]{1,0} %b), custom_call_target="__cublas$lt$matmul$f8"',
              "jit(step)/dot_general", "1", 1000.0, "", 4.0e5, 3.2e6)]
    captured(tmp_path, cells, 3742430.0, extra=("normalized_flop_rate", "model_flop_rate"))
    traced(tmp_path, [("Stream #14(Compute)", "cublas_gemm", 0, 1, 15),
                      ("Stream #14(Compute)", "cublasLt_f8_gemm", 1, 2, 15)])
    entry = tmp_path / "entry.json"
    entry.write_text(json.dumps({"model_flops_per_step": 1.05e12, "device": "NVIDIA RTX PRO 6000 Blackwell"}))
    dump = dumped(tmp_path, "module_0015.jit_step.sm_12.0_gpu_after_optimizations.txt", GEMMS)
    out = tmp_path / "attribution.json"
    command = ["--trace", str(tmp_path / "profile"), "--hlo", str(dump), "--out", str(out),
               "--entry", str(entry)]
    assert attribute.main(command) == 0
    time = json.loads(out.read_text())["time"]
    # 2.5e11 and 8e11 FLOPs a millisecond: 2.5e5 of the 5e5 GFLOP/s bf16 peak and 8e5 of the 1e6
    # FP8 peak, whatever XProf's rows say (4x those).
    assert {record["hlo"]: record["fraction_of_peak"] for record in time["gemms_by_lost_time"]} == {
        "custom-call.1": 0.5, "custom-call.2": 0.8}
    assert time["gemm_fraction_of_peak"] == 0.65 and time["gemm_lost_ms"] == 0.7
    assert time["fp8_gemm_flops_per_step"] == 8e11
    # All the step is GEMM time, so mfu_mixed is the GEMMs' own fraction.
    assert time["mfu_mixed_of_device_time"] == 0.65


def test_the_dump_reads_the_compile_the_capture_ran_from_an_archive(tmp_path):
    """A dump holds every compile of the step, a plan's candidates among them; only the four
    files of the one the capture ran are extracted (the newest, without a capture), and the
    memory breakdown reads them a line at a time. A capture of a compile the dump lacks is
    refused."""
    import tarfile

    dump = tmp_path / "hlo"
    dump.mkdir()
    for number, peak in (("0003", 1 << 30), ("0015", 2 << 30)):
        stem = f"module_{number}.jit_step.sm_8.0_gpu_after_optimizations"
        (dump / f"{stem}.txt").write_text(MODULE)
        (dump / f"{stem}-live-range.txt").write_text(
            f"Stack trace breakdown for peak usage: {peak} bytes\n"
            f"  main ({100.0:.1f}%, total: {peak} bytes, current: 0 bytes)\n"
            f"    p/params/kernel (50.0%, total: {peak // 2} bytes, current: {peak // 2} bytes)\n")
        (dump / f"{stem}-buffer-assignment.txt").write_text(f"allocation 0: size {peak}, parameter 0\n")
        (dump / f"{stem}-memory-usage-report.txt").write_text(f"Total bytes: {peak}\n")
    (dump / "module_0016.jit_other.sm_8.0_gpu_after_optimizations.txt").write_text(MODULE)
    archive = tmp_path / "hlo.tgz"
    with tarfile.open(archive, "w:gz") as writer:
        writer.add(dump, arcname="hlo")
    for source in (archive, dump):
        for program, gib in ((None, 2.0), ("3", 1.0)):
            with attribute.dump_files(source, "jit_step", program) as files:
                memory = attribute.attribute_memory(files)
            assert (memory["peak_gib"], memory["total_bytes_gib"], memory["allocations_gib"]) == (
                gib, gib, {"parameter": gib})
        with pytest.raises(SystemExit), attribute.dump_files(source, "jit_step", "9"):
            pass
