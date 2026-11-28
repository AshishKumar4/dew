"""Probe (not for landing; run on a GH200): where the XLA route of a packed NVFP4 prefill loses the columns of
a subnormal column scale (M-NG-68c2: test_gemm_nvfp4's K-F5 cell, 'apart' +-0 where float64 is ~1e-37)."""

import jax
import jax.numpy as jnp
import numpy as np


def test_probe_kf5_stages():
    from dew.algebra import bits
    from dew.nn.kernels import formats, gemm, generation
    from dew.nn.kernels.templates.gemm import factored

    rng = np.random.default_rng(13743)
    codes = rng.integers(-2, 3, (2, 256, 128)).astype(np.float32) / 2
    scales = np.tile(np.asarray([1.0, 2.0 ** -127, 2.0 ** 121], np.float32), 43)[:128]
    scale = np.stack([scales, np.roll(scales, 1)])
    bank = formats.NvFp4.of(jnp.asarray(codes), jnp.ones((2, 16, 128), jnp.float8_e4m3fn), jnp.asarray(scale))
    rows = rng.integers(-1, 2, (256, 256)).astype(np.float32)
    sizes = jnp.asarray([128, 128], jnp.int32)
    truth = np.concatenate([rows[g * 128:(g + 1) * 128].astype(np.float64) @ codes[g]
                            * scale[g].astype(np.float64) for g in range(2)]).astype(np.float32)
    lines = [f"backend {jax.default_backend()} devices {jax.devices()[:1]} owed {bits.NVFP4_OWED}"]
    subnormal_cols = np.where(scale[0] == np.float32(2.0 ** -127))[0][:3]
    lines.append(f"subnormal columns (group 0) {subnormal_cols}")
    # (1) The decoded bank and the plain product before any scale.
    decoded = np.asarray(bank.decoded()).astype(np.float32)
    lines.append(f"decoded[0,:4,{subnormal_cols[0]}] {decoded[0, :4, subnormal_cols[0]]} dtype {bank.decoded().dtype}")
    # (2) A plain jitted multiply of a subnormal by 2^8, and of a product by it.
    sub = jnp.full((4,), 2.0 ** -127, jnp.float32)
    lines.append(f"jit sub*256 bits {np.asarray(jax.jit(lambda a: a * np.float32(256))(sub)).view(np.uint32)}")
    lines.append(f"jit owed_native(3, 2^-127, 8) bits "
                 f"{np.asarray(jax.jit(lambda p, f: factored.owed_native(p, f, raised=8))(jnp.full((4,), 3.0, jnp.float32), sub)).view(np.uint32)}")
    lines.append(f"jit owed_exactly(3, 2^-127, 8) bits "
                 f"{np.asarray(jax.jit(lambda p, f: factored.owed_exactly(p, f, 8))(jnp.full((4,), 3.0, jnp.float32), sub)).view(np.uint32)}")
    # (3) per_row of the column scales.
    expanded = np.asarray(jax.jit(lambda s: factored.per_row(s, sizes, 256))(jnp.asarray(scale)))
    lines.append(f"per_row[0, subnormal cols] bits {expanded[0, subnormal_cols].view(np.uint32)}")
    # (4) The whole XLA route, and where it is zero against the truth.
    with generation.chosen(nvfp4="xla"):
        out = np.asarray(jax.jit(lambda x: gemm.nvfp4_product(x, bank, sizes, out_dtype=jnp.float32))(
            jnp.asarray(rows, jnp.bfloat16)))
    bad = np.argwhere(out.view(np.uint32) != truth.view(np.uint32))
    lines.append(f"xla route mismatches {len(bad)}; first {bad[:3].tolist()}; out {out[tuple(bad[0])] if len(bad) else None} "
                 f"truth {truth[tuple(bad[0])] if len(bad) else None}")
    lines.append(f"mismatch columns {sorted(set(bad[:, 1].tolist()))[:12]}")
    text = jax.jit(lambda x: gemm.nvfp4_product(x, bank, sizes, out_dtype=jnp.float32)).lower(
        jnp.asarray(rows, jnp.bfloat16)).compile().as_text() if False else ""
    raise AssertionError("\n".join(lines) + text)
