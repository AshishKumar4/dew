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
    from reference_error import _arm_program, with_gradients
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
