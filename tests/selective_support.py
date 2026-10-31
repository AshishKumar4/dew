"""Mamba's selective scan held to recorded mamba_ssm (tools/mamba_selective_reference.py): a recorded case's
seeded inputs drawn again without torch, and Dew's portable scan over them with every gradient, the
recorder's sizing proxy and its float64 cross-check."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import with_gradients
from tool_support import load

from dew.nn.mixers.mamba import selective_scan

_recorder = load("mamba_selective_reference")
SEEDED_CASES = _recorder.SEEDED_CASES


def selective_draw(seed: int, case: str) -> dict[str, np.ndarray]:
    """`case`'s inputs at `seed` as the recorder drew them, its narrow operands' values stored in fp32."""
    *shape, dtype, regime = SEEDED_CASES[case]
    return _recorder.operands(np.random.default_rng(seed), *shape,
                              lambda value: value.astype(jnp.dtype(dtype)).astype(np.float32), regime)


def starts_of(arrays) -> jax.Array | None:
    """Each token's whether it starts a document, from the recorded `seq`, or None where none is packed."""
    seq = arrays.get("seq")
    if seq is None:
        return None
    return jnp.concatenate([jnp.ones_like(seq[:, :1], bool), seq[:, 1:] > seq[:, :-1]], axis=1)


def selective_xla(arrays, dtype=None, scan=selective_scan) -> dict[str, jax.Array]:
    """`scan` (Dew's portable selective scan unless given, its operands `(x, dt, A, B, C, D, state, starts)`)
    over a case's inputs, `x`, the step, `B`, `C` and the cotangent in `dtype` where given: its output and
    each operand's gradient under the recorded cotangent, named as the recording names them."""
    names = ("x", "dt", "A", "B", "C", "D")
    narrow = ("x", "dt", "B", "C")
    operands = tuple(jnp.asarray(arrays[name], dtype if name in narrow else None) for name in names)
    starts = starts_of(arrays)

    def run(*operands):
        return scan(*operands, None, starts)[0]

    out = jax.eval_shape(run, *operands)
    found = with_gradients(run)(*operands, jnp.asarray(arrays["dy"], out.dtype))
    return dict(zip(("y", *names), found, strict=True))
