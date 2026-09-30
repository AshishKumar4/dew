"""Forward-mode attention on a CUDA GPU: cuDNN's fused kernel, reverse-mode
only, takes a JVP inside `forward_mode_attention`, its value the kernel's and
its tangent the reference path's.

The test suite's cuda lane runs under --xla_gpu_deterministic_ops, where
Dew refuses cuDNN attention (openxla/xla#46500), so this check runs on its
own, without the flag:

    PYTHONPATH=src python tools/attention_jvp_check.py

It prints the largest difference of the cuDNN value and tangent from XLA's
kernel, whose JVP JAX derives itself, on bf16 inputs of batch 2, 128 tokens,
4 heads of 64.
"""

import jax
import jax.numpy as jnp

from dew.nn.attention import forward_mode_attention, scaled_dot_product_attention


def main() -> None:
    q, k, v = (jax.random.normal(jax.random.PRNGKey(i), (2, 128, 4, 64), jnp.bfloat16) for i in range(3))
    tangents = tuple(jax.random.normal(jax.random.PRNGKey(10 + i), q.shape, jnp.bfloat16) for i in range(3))

    def run(implementation):
        def attend(q, k, v):
            return scaled_dot_product_attention(q, k, v, implementation=implementation)
        with forward_mode_attention():
            return jax.jit(lambda: jax.jvp(attend, (q, k, v), tangents))()

    try:
        jax.jvp(lambda q: scaled_dot_product_attention(q, k, v, implementation="cudnn"), (q,), (tangents[0],))
        print("cudnn took a JVP outside forward_mode_attention")
    except TypeError as error:
        print(f"outside the context: {error}")
    out, tangent = run("cudnn")
    reference_out, reference_tangent = run("xla")

    def gap(a, b):
        return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
    print(f"value gap {gap(out, reference_out):.4g} (max |value| {float(jnp.abs(reference_out).max()):.3g}), "
          f"tangent gap {gap(tangent, reference_tangent):.4g} "
          f"(max |tangent| {float(jnp.abs(reference_tangent.astype(jnp.float32)).max()):.3g})")


if __name__ == "__main__":
    main()
