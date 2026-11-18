"""Mamba-3's SISO sites held to mamba_ssm's own kernels, `mamba3_siso_combined` (state-spaces/mamba
e9594ce1c, mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py, Triton) recorded on a CUDA GPU by
tools/mamba3_reference.py: each site's seeded draws in the kernel's own parametrization, its queries and keys
(`C`, `B`), values `x`, `ADT` and `DT`, trapezoid logits `trap` and raw angles `angles` (a token turns its
pairs by `tanh(angles) pi DT`, angle_dt.py:94), and Dew's chunked form on them (`dew_form`, `chunk_mamba3`).

The kernel casts its queries, keys, values, trapezoid and angles to bf16 (mamba3_siso_combined.py:384-392):
its sites are a bf16 layer's, each of those drawn as a bf16 value, so the cast keeps it and both arms read
the same operands. `ADT` is drawn as a bf16 `A` times a bf16 `DT`, exact in fp32, so Dew's `A = ADT / DT`
is `A` itself. The kernel's biases are zero (a site's keys and queries are their sum already), and it
reads no skip, no gate and no state entering: the recurrence alone, its output and every operand's
gradient. Packed documents run as the kernel runs them, one row of sequences (`cu_seqlens`). The truth is
Dew's recurrence a token at a time in float64 (`recurrent_mamba3`, its transcription of mamba_ssm's
modules), the recording's float64 arm."""

import dataclasses
import functools
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

RECORDED = Path(__file__).resolve().parent / "fixtures" / "mamba3" / "siso_seeded.npz"
"""The recording (tools/mamba3_reference.py)."""

OPERANDS = ("x", "B", "C", "DT", "ADT", "trap", "angles")
"""A site's differentiated operands, in `dew_form`'s order."""

LEAVES = ("out", *OPERANDS)
"""A site's leaves: the output and each operand's gradient."""

GROUPS, HEADS, WIDTH, STATE, PAIRS, CHUNK = 1, 4, 64, 64, 16, 64
"""A site's key groups and heads, a head's values and its state's keys, the pairs that turn (half the
state's, mamba_ssm's `rope_fraction` 0.5) and the kernel's chunk."""


@dataclasses.dataclass(frozen=True)
class Site:
    """A recorded site: its rows and tokens, each row's documents' first tokens where packed (`documents`,
    the kernel's one row, `cu_seqlens`), and its draws by seed (`arrays`)."""

    rows: int
    length: int
    documents: tuple[int, ...] | None = None

    def arrays(self, seed: int) -> dict[str, np.ndarray]:
        """`seed`'s operands (`OPERANDS`, each a bf16 value but `DT` and `ADT`'s fp32 product) and its
        output's cotangent `d_out`, with `starts` where packed."""
        normal = np.random.default_rng([seed, 7923]).standard_normal

        def bf16(x) -> np.ndarray:
            return np.asarray(jnp.asarray(x, jnp.float32).astype(jnp.bfloat16).astype(jnp.float32))

        tokens, heads = (self.rows, self.length), (self.rows, self.length, HEADS)
        dt = bf16(np.logaddexp(0.0, normal(heads) - 1.0))
        a = bf16(-np.exp(0.5 * normal(heads)))
        drawn = {"x": bf16(normal((*heads, WIDTH))),
                 **{name: bf16(normal((*tokens, GROUPS, STATE)) / np.sqrt(STATE)) for name in ("B", "C")},
                 "DT": dt, "ADT": (a * dt).astype(np.float32), "trap": bf16(normal(heads)),
                 "angles": bf16(normal((*heads, PAIRS))), "d_out": normal((*heads, WIDTH)).astype(np.float32)}
        if self.documents is not None:
            starts = np.zeros(tokens, bool)
            starts[:, list(self.documents)] = True
            drawn["starts"] = starts
        return drawn


def sites() -> dict[str, Site]:
    """The recorded sites by name: rows of one sequence each, and one row of packed documents."""
    return {"mamba3/siso/bf16": Site(2, 200), "mamba3/siso/bf16 packed": Site(1, 300, (0, 61, 150, 211))}


def dew_form(chunked: Callable, arrays: dict, leaves: tuple[str, ...] = LEAVES) -> dict[str, jax.Array]:
    """`chunked(x, dt, A, B, C, lam, theta, starts)` (Mamba-3's form, `chunk_mamba3`'s operands, its output
    `[B, S, R, H, P]`) on a site's `arrays` in the kernel's parametrization: `A = ADT / DT`, `lam =
    sigmoid(trap)`, `theta = tanh(angles) pi`, a rank of one; its output and its operands' gradients
    (`leaves`) under the output's cotangent `d_out`. Its `x`, `B` and `C` in bf16 where the arrays are
    float32 (a bf16 layer's), float64 where they are."""
    wide = jnp.asarray(arrays["DT"]).dtype == jnp.float64
    narrow = jnp.float64 if wide else jnp.bfloat16
    starts = arrays.get("starts")

    def forward(x, B, C, DT, ADT, trap, angles):
        theta = jnp.tanh(angles) * np.pi
        out = chunked(x[:, :, None], DT, ADT / DT, B[:, :, None], C[:, :, None], jax.nn.sigmoid(trap), theta,
                      starts)
        return out[:, :, 0]

    primals = tuple(jnp.asarray(arrays[name], narrow if name in ("x", "B", "C") else None)
                    for name in OPERANDS)
    out, pullback = jax.vjp(forward, *primals)
    gradients = pullback(jnp.asarray(arrays["d_out"], out.dtype))
    found = {"out": out, **dict(zip(OPERANDS, gradients, strict=True))}
    return {leaf: found[leaf] for leaf in leaves}


def chunked_on(kernel: str | None, x, dt, A, B, C, lam, theta, starts):
    """`chunk_mamba3`'s output from a zero state at the kernel's chunk, on `kernel` (`generation.chosen`'s
    name for the chunked op's) or, None, on each product's selected kernel, as a layer runs it (a bf16
    call's forward on the template, its gradient on the portable math until the template's backward takes
    it, row 2076)."""
    import contextlib

    from dew.nn.kernels import generation
    from dew.nn.mixers.mamba3 import chunk_mamba3

    with contextlib.nullcontext() if kernel is None else generation.chosen(gated_delta_rule=kernel):
        return chunk_mamba3(x, dt, A, B, C, lam, theta, starts=starts, chunk_size=CHUNK)[0]


def recurrent(x, dt, A, B, C, lam, theta, starts):
    """`recurrent_mamba3`'s output from a zero state: the float64 truth's form."""
    from dew.nn.mixers.mamba3 import recurrent_mamba3

    return recurrent_mamba3(x, dt, A, B, C, lam, theta, starts=starts)[0]


def truth(arrays: dict, leaves: tuple[str, ...] = LEAVES) -> dict[str, jax.Array]:
    """A site's float64 truth: `dew_form` over `recurrent` in float64 (the caller's `enable_x64`)."""
    widened = {name: np.asarray(value, np.float64) if np.issubdtype(np.asarray(value).dtype, np.floating)
               else value for name, value in arrays.items()}
    return dew_form(recurrent, widened, leaves)


def the_chunked_form_is_as_exact_as_mamba_ssm(kernel: str | None, site: str,
                                              leaves: tuple[str, ...] = LEAVES):
    """Mamba-3's chunked form on `kernel` (`chunked_on`) on par from float64 with recorded mamba_ssm at `site`'s declared
    seeds (check 1), its leaves `leaves`, the truth the recurrence a token at a time in float64."""
    from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording

    seeds = declared_seeds(RECORDED)[site]
    with jax.enable_x64(new_val=True):
        exact = jax.jit(functools.partial(truth, leaves=leaves))
    draw, reference, truths = indexed_recording(recording(RECORDED), site, seeds,
                                                inputs_of=sites()[site].arrays, leaves=leaves, truth_fn=exact)

    def run(index, arrays):
        del index
        return dew_form(functools.partial(chunked_on, kernel), arrays, leaves)

    assert_as_exact(run, as_given(reference), draw, f"mamba3 {kernel} against mamba_ssm, {site}", seeds=seeds,
                    truth_fn=as_given(truths))
