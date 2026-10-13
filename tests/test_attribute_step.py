"""tools/attribute_step.py: where a profiled step's device time goes, by part and kind.

The rows here are the Qwen3.5-0.8B step's own (c50, a decision step on an A100): XProf's hlo_stats
rows and the optimized HLO they ran, cut to the instructions each case reads. A kind is read
from the op, never from its operands' names, which its expression also holds: FlashAttention's
backward takes an operand called `%input_concatenate_fusion.8` and is attention, not a layout
move, and a weight-gradient GEMM traced at `up_proj/transpose` is a GEMM.
"""

import json

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


def test_a_nameless_fusion_is_attributed_by_the_ops_it_fuses(tmp_path):
    """A multi-output fusion carries no op_name of its own (Qwen3.5-0.8B's 50 ms of
    "other"): its part and direction come from the ops fused inside it."""
    reports = tmp_path / "profile" / "capture-a" / "reports"
    reports.mkdir(parents=True)
    columns = ["program_id", "category", "hlo_op_name", "hlo_op_expression", "tf_op_name", "occurrences",
               "total_self_time", "source_info"]
    cells = [("15", "input fusion", "input_reduce_fusion.4", "%input_reduce_fusion.4 = (f32[8,3072]{1,0}, "
              "f32[8,3072,1024]{2,1,0}) fusion(f32[8,3072,1024]{2,1,0} %p0)", "", "1", 2000.0, "")]
    (reports / "hlo_stats.json").write_text(json.dumps([{"cols": [{"id": name} for name in columns],
                                                         "rows": [{"c": [{"v": v} for v in cells[0]]}]}]))
    (reports / "roofline_model.json").write_text(json.dumps({"p": {"peak_flop_rate": 311869.0}}))
    dump = tmp_path / "hlo"
    dump.mkdir()
    (dump / "module_0015.jit_step.sm_8.0_gpu_after_optimizations.txt").write_text(NAMELESS)
    out = tmp_path / "attribution.json"
    assert attribute.main(["--trace", str(tmp_path / "profile"), "--hlo", str(dump), "--out", str(out)]) == 0
    time = json.loads(out.read_text())["time"]
    assert time["by_part_kind_ms"] == {"norm": {"reduce": 2.0}}
    assert time["by_part_ms"] == {"norm": {"backward": 2.0}}


def test_a_capture_attributes_its_time_by_part_and_kind(tmp_path):
    """Through the command: the step's time by kind and by part and kind, each op once."""
    reports = tmp_path / "profile" / "capture-a" / "reports"
    reports.mkdir(parents=True)
    columns = ["program_id", "category", "hlo_op_name", "hlo_op_expression", "tf_op_name", "occurrences",
               "total_self_time", "source_info", "hbm_bw"]
    times = {"flash backward": 6000.0, "Pallas recurrence": 2000.0, "transposing fusion": 1000.0,
             "weight-gradient GEMM": 500.0}
    cells = [("15", found["category"], found["hlo_op_name"], found["hlo_op_expression"], found["tf_op_name"],
              "1", time, "", 1500.0) for case, time in times.items() for found in [ROWS[case][0]]]
    rows = [{"c": [{"v": value} for value in values]} for values in cells]
    (reports / "hlo_stats.json").write_text(json.dumps([{"cols": [{"id": name} for name in columns],
                                                         "rows": rows}]))
    (reports / "roofline_model.json").write_text(json.dumps({"p": {"peak_flop_rate": 311869.0}}))
    dump = tmp_path / "hlo"
    dump.mkdir()
    (dump / "module_0015.jit_step.sm_8.0_gpu_after_optimizations.txt").write_text(MODULE)
    out = tmp_path / "attribution.json"
    assert attribute.main(["--trace", str(tmp_path / "profile"), "--hlo", str(dump), "--out", str(out)]) == 0
    time = json.loads(out.read_text())["time"]
    assert time["by_kind_ms"] == {"attention": 6.0, "pallas": 2.0, "layout": 1.0, "gemm": 0.5}
    assert time["by_part_kind_ms"]["attention"] == {"attention": 6.0}
    assert time["by_part_kind_ms"]["mlp"] == {"gemm": 0.5}
    # 1,500 GiB/s over the transposing fusion's millisecond is 1.5 GiB a step.
    assert {record["us"]: record["hbm_mib"] for record in time["top_ops"]}[1000.0] == 1536.0


def test_the_peak_is_dews_dense_bf16_figure_for_a_named_device(tmp_path):
    """XProf's roofline gives an RTX PRO 6000 a 3,742 TFLOP/s peak, a sparse low-precision figure:
    every MFU field reads Dew's dense bf16 table where the entry names the device, and none is
    written where it does not, XProf's figure kept beside for reference."""
    reports = tmp_path / "profile" / "capture-a" / "reports"
    reports.mkdir(parents=True)
    columns = ["program_id", "category", "hlo_op_name", "hlo_op_expression", "tf_op_name", "occurrences",
               "total_self_time", "source_info"]
    cells = ("15", "custom-call", "custom-call.1", "%custom-call.1 = bf16[8,8]{1,0} custom-call()",
             "jit(step)/dot_general", "1", 1000.0, "")
    (reports / "hlo_stats.json").write_text(json.dumps([{"cols": [{"id": name} for name in columns],
                                                         "rows": [{"c": [{"v": v} for v in cells]}]}]))
    (reports / "roofline_model.json").write_text(json.dumps({"p": {"peak_flop_rate": 3742430.0}}))
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
    assert time["device_ms_per_step"] == 1.0


def test_a_gemm_is_judged_at_its_compiled_flops_against_its_own_precisions_peak(tmp_path):
    """XProf's normalized rate scales a GEMM's FLOPs by bf16's peak over its precision's, so an fp32
    dot it takes for non-tensor-core arithmetic reads above any peak (the later Qwen3.5-0.8B step's GEMMs at 1.127 of
    the A100's). The compiler's FLOPs over the measured time are the work done, and each GEMM's
    fraction is of its own precision's peak in Dew's table, as mfu_mixed counts it."""
    reports = tmp_path / "profile" / "capture-a" / "reports"
    reports.mkdir(parents=True)
    columns = ["program_id", "category", "hlo_op_name", "hlo_op_expression", "tf_op_name", "occurrences",
               "total_self_time", "source_info", "normalized_flop_rate", "model_flop_rate"]
    cells = [("15", "custom-call", "custom-call.1", '%custom-call.1 = f32[8,8]{1,0} custom-call('
              'f32[8,8]{1,0} %a, f32[8,8]{1,0} %b), custom_call_target="__cublas$gemm"',
              "jit(step)/dot_general", "1", 1000.0, "", 8.0e6, 2.5e5),
             ("15", "custom-call", "custom-call.2", '%custom-call.2 = bf16[8,8]{1,0} custom-call('
              'f8e4m3fn[8,8]{1,0} %a, f8e4m3fn[8,8]{1,0} %b), custom_call_target="__cublas$lt$matmul$f8"',
              "jit(step)/dot_general", "1", 1000.0, "", 4.0e5, 8.0e5)]
    rows = [{"c": [{"v": v} for v in row]} for row in cells]
    (reports / "hlo_stats.json").write_text(json.dumps([{"cols": [{"id": name} for name in columns],
                                                         "rows": rows}]))
    (reports / "roofline_model.json").write_text(json.dumps({"p": {"peak_flop_rate": 3742430.0}}))
    entry = tmp_path / "entry.json"
    entry.write_text(json.dumps({"model_flops_per_step": 1.05e12, "device": "NVIDIA RTX PRO 6000 Blackwell"}))
    out = tmp_path / "attribution.json"
    command = ["--trace", str(tmp_path / "profile"), "--out", str(out), "--entry", str(entry)]
    assert attribute.main(command) == 0
    time = json.loads(out.read_text())["time"]
    # 2.5e5 of the 5e5 GFLOP/s bf16 peak and 8e5 of the 1e6 FP8 peak, a millisecond each.
    assert {record["hlo"]: record["fraction_of_peak"] for record in time["gemms_by_lost_time"]} == {
        "custom-call.1": 0.5, "custom-call.2": 0.8}
    assert time["gemm_fraction_of_peak"] == 0.65 and time["gemm_lost_ms"] == 0.7
    assert time["fp8_gemm_flops_per_step"] == 8e11
    # All the step is GEMM time, so mfu_mixed is the GEMMs' own fraction.
    assert time["mfu_mixed_of_device_time"] == 0.65


def test_the_dump_reads_only_its_newest_module_from_an_archive(tmp_path):
    """A dump holds every compile of the step; only the newest one's four files are extracted,
    and the memory breakdown reads them a line at a time."""
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
        with attribute.dump_files(source, "jit_step") as files:
            memory = attribute.attribute_memory(files)
        assert (memory["peak_gib"], memory["total_bytes_gib"], memory["allocations_gib"]) == (
            2.0, 2.0, {"parameter": 2.0})
