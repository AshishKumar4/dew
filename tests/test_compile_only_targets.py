"""A compile-only GPU's target as XLA compiles for it (`generation.target_config`), against the host GPU of
its part: the program it compiles is the host's own compile of it."""

import functools

import jax
import numpy as np


def test_a_compile_only_gpu_fuses_a_program_as_the_host_gpu_of_its_part_does(tmp_path):
    """Dew's portable recurrence gradient (the compile-ahead tier cell's), compiled for a compile-only A100
    of its multiprocessors (`compile_only_gpus`) and for an A100 SXM4 40 GB host's own serialized topology
    (`host_gpus`), takes the same instructions: XLA fuses by the L2 the target states, and a compile-only
    A100 that stated none fused the gradient's sums apart, 271 instructions off, its run 2e-7 from the host's
    compile of it (row 9609, M-FS-21b-a100)."""
    import base64
    import collections
    import re
    from pathlib import Path

    import test_recurrence_stability as stability
    from arm_arithmetic import summed_as_run
    from arm_store import _arm_program
    from reference_error import with_gradients
    from truths_ahead import named

    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import report
    from dew.training.ahead import host_gpus

    hooks = stability.PORTABLE_SITES["scalar delta"]
    dew = with_gradients(stability.restarting(functools.partial(
        stability.chunk_on, "xla", hooks, chunk_size=stability.CHUNK, **stability.bounded("aligned keys")),
        "aligned keys"))
    inputs = stability.cotangents(hooks, "aligned keys", 0)
    written = Path(__file__).parent / "fixtures" / "topologies" / "a100-sxm4-40gb.b64"
    (tmp_path / "topology.bin").write_bytes(base64.b64decode(written.read_text()))

    def instructions(gpu) -> collections.Counter:
        # The optimized module's instructions by opcode, shape and fusion kind, its names and their order
        # aside.
        mesh = jax.sharding.Mesh(np.asarray([gpu]), ("devices",))
        placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        with summed_as_run():
            programs, held = {}, {}
            _arm_program(programs, held, "dew", dew, inputs, reference=False)
            function, shapes = programs[named("dew", inputs)]
            with (generation.assumed(generation.built_target("sm80", 108), portable=True),
                  jax.sharding.use_abstract_mesh(mesh.abstract_mesh), report.certifying(gpu)):
                lowered = jax.jit(function, out_shardings=placed).lower(
                    *(jax.ShapeDtypeStruct(shape.shape, shape.dtype, sharding=placed) for shape in shapes))
                text = lowered.compile(compiler_options=report.targeted(lowered, None)).as_text()
        found = collections.Counter()
        for line in text.splitlines():
            if (made := re.match(r"\s*(?:ROOT )?%[\w.\-]+ = (\S+(?: \S+)?) ([\w\-]+)\(", line)):
                kind = re.search(r"kind=(\w+)", line)
                found[made.group(2), made.group(1).split("{")[0], kind.group(1) if kind else ""] += 1
        return found

    (host,) = host_gpus("sm80", tmp_path / "topology.bin", 1)
    (compile_only,) = generation.compile_only_gpus("sm80", 1, multiprocessors=108)
    assert instructions(compile_only) == instructions(host)


def test_a_compile_only_hopper_lowers_fp8_dots_to_cublaslt_as_its_host_gpu_does(tmp_path):
    """Both blockwise FP8 cuBLASLt references (`blockwise_matrix.cublaslt`,
    `test_gemm_blockwise._gradient_cublaslt`), compiled for a compile-only Hopper of a GH200's multiprocessors
    (`compile_only_gpus`) and for a GH200 host's own serialized topology (`host_gpus`), lower their FP8 dots
    to cuBLASLt's FP8 target (`__cublas$lt$matmul$f8`) with no FP16 widening and no XLA dot: XLA's
    `GemmRewriter` requires CUDA runtime >= 12.0.0 (`device_description.runtime_version`), and a compile-only
    Hopper stating none (`0.0.0`) widened both FP8 operands to fp16 and ran `__cublas$lt$matmul`, 2331x–3112x
    more exact than Hopper's FP8 WGMMA accumulator (rows 12983, 12984)."""
    import base64
    from pathlib import Path

    import blockwise_matrix
    import generation_support
    import test_gemm_blockwise

    from dew.nn.kernels import generation
    from dew.telemetry.instrumentation import hlo_instructions
    from dew.training.ahead import host_gpus

    written = Path(__file__).parent / "fixtures" / "topologies" / "gh200.b64"
    (tmp_path / "topology.bin").write_bytes(base64.b64decode(written.read_text()))

    (host,) = host_gpus("sm90", tmp_path / "topology.bin", 1)
    (compile_only,) = generation.compile_only_gpus("sm90", 1, multiprocessors=132)

    case = dict(zip(("generation", "columns", "groups", "out_dtype"), blockwise_matrix.CASES[0], strict=True))
    product_fn = blockwise_matrix.cublaslt(blockwise_matrix._sizes(int(case["groups"])), case["out_dtype"])
    cases = (
        ("product", product_fn, blockwise_matrix._operands(case, 0)),
        ("gradient", test_gemm_blockwise._gradient_cublaslt, test_gemm_blockwise._gradient_operands(0)),
    )

    def lowered_targets(gpu, fn, inputs):
        mesh = jax.sharding.Mesh(np.asarray([gpu]), ("devices",))
        placed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        shapes = tuple(jax.ShapeDtypeStruct(np.shape(x), x.dtype, sharding=placed) for x in inputs)
        with (generation.assumed(generation.built_target("sm90", 132), portable=True),
              jax.sharding.use_abstract_mesh(mesh.abstract_mesh)):
            compiled = jax.jit(fn, out_shardings=placed,
                               compiler_options=generation_support.CUBLAS).lower(*shapes).compile()
        instructions = tuple(hlo_instructions(compiled).values())
        targets = {each.target for each in instructions} - {""}
        dots = [each for each in instructions if each.op == "dot" or "dot" in each.fused]
        f16_converts = [each for each in instructions
                        if each.element == "f16" and (each.op == "convert" or "convert" in each.fused)]
        return targets, dots, f16_converts

    for name, fn, inputs in cases:
        for kind, gpu in (("host", host), ("compile_only", compile_only)):
            targets, dots, f16_converts = lowered_targets(gpu, fn, inputs)
            assert targets == {"__cublas$lt$matmul$f8"}, (name, kind, sorted(targets))
            assert not dots, (name, kind, dots)
            assert not f16_converts, (name, kind, f16_converts)
