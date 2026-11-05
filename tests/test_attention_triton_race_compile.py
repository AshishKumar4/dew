"""The Pallas-Triton attention backward's dK/dV launch compiled for an H100 on the CPU: no asynchronous copy
overwrites a shared tile an unfinished asynchronous MMA reads. Triton's pipeliner gave the query tile, loaded
once and read twice (prescaled for the scores, raw for dK's product), one buffer at two stages, so the next
query block's copy overwrote it under dK's product: dK nondeterministic and wrong on an H100 at every derived
config (FS-TRITON-RACE1-h100), fixed in the pinned Triton's pipeliner (jax 0.12.0.1.post45)."""

import re

import jax
import jax.numpy as jnp
import kernel_compile

from dew.nn.kernels.templates import attention as template, attention_triton


def _overwritten_mma_operands(module: str) -> tuple[list[str], int, int]:
    """Single-buffered MMA operands overwritten before the outstanding MMA groups complete, with the
    asynchronous MMAs and copies the schedule holds (none read means another IR's text, not a safe one)."""
    roots: dict[str, str] = {}
    products = copies = 0
    single: set[str] = set()
    pending: set[str] = set()
    hazards: list[str] = []
    for line in module.splitlines():
        defined = re.match(r"\s*(%\w+)(?::\d+)? =", line)
        operands = re.search(r'"\w+\.\w+"\(([^)]*)\)', line)
        values = re.findall(r"%\w+", operands[1]) if operands else []
        if defined and '"ttg.local_alloc"' in line:
            roots[defined[1]] = defined[1]
            if re.search(r"memdesc<1x\d+x\d+xbf16,", line):
                single.add(defined[1])
        elif defined and any(op in line for op in ('"ttg.memdesc_index"', '"ttg.memdesc_trans"')):
            roots[defined[1]] = roots.get(values[0], values[0])
        elif '"ttng.warp_group_dot_wait"' in line and "pendings = 0" in line:
            pending.clear()
        elif '"ttng.warp_group_dot"' in line and "isAsync = true" in line:
            products += 1
            pending.update(roots.get(value, value) for value in values[:2]
                           if roots.get(value, value) in single)
        elif '"ttg.async_copy_global_to_local"' in line:
            copies += 1
            destination = roots.get(values[1], values[1])
            if destination in pending:
                hazards.append(f"{destination}: {line.strip()}")
    return hazards, products, copies


def test_hopper_attention_backward_preserves_its_multiuse_query_tile(tmp_path):
    """At 64 by 64, each launch two stages deep, the raw query tile dK's product reads stays its own until
    that product completes, after its prescaled copy fed the scores."""
    config = attention_triton.BackwardTiles(64, 64, attention_triton.WARPS, attention_triton.STAGES,
                                            attention_triton.WARPS, attention_triton.STAGES)
    power, scale = template.prescale(jnp.bfloat16, 128, None)
    op = attention_triton._backward_op(score_mod=None, score_kinds=(), score_dtypes=(), scale=scale,
                                       prescale=power, split=True, natural=False, accurate=template.ACCURATE,
                                       initial=False)
    query = jax.ShapeDtypeStruct((1, 1024, 32, 128), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, 1024, 8, 128), jnp.bfloat16)
    row = jax.ShapeDtypeStruct((1, 32, 1024), jnp.float32)
    index = jax.ShapeDtypeStruct((1, 32, 1024), jnp.int32)
    span = jax.ShapeDtypeStruct((1, 1024), jnp.int32)

    def launch(q, k, v, dout, peak, total, anchor, delta, key_index, start, end):
        return attention_triton._dkv_launch(
            q, k, v, dout, peak, total, anchor, delta, key_index, start, end, None, None, (),
            score_mod=op.score_mod, score_kinds=op.score_kinds, scale=op.scale, prescale=op.prescale,
            grad_scale=op.scale * op.prescale, how=op.how, block_q=config.block_q, block_k=config.block_k,
            warps=config.dkv_warps, stages=config.dkv_stages, accurate=op.accurate, interpret=False)

    lowered = kernel_compile.lowered_for(
        launch, (query, key, key, query, row, row, row, row, index, span, span), "sm90")
    lowered.compile(compiler_options={"xla_dump_to": str(tmp_path), "xla_dump_emitter_re": "triton"})

    dumps = list(tmp_path.glob("*attention_dkv.triton-to-llvm.txt"))
    assert len(dumps) == 1, [path.name for path in tmp_path.iterdir()]
    marker = "// -----// IR Dump Before ConvertTritonGPUToLLVM:"
    text = dumps[0].read_text()
    assert marker in text, "the compiler did not dump the scheduled Triton GPU module"
    module = text.split(marker, 1)[1].split("// -----// IR Dump Before ", 1)[0]
    hazards, products, copies = _overwritten_mma_operands(module)
    assert products and copies, "the scheduled module holds no asynchronous MMA or copy this test reads"
    assert not hazards, "async copies overwrite unfinished MMA operands:\n" + "\n".join(hazards)
