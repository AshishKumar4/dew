"""A coupled momentum's sites the chunked op is held to fla at (row 1454, `recurrence.Hooks.coupled`): each
site's draw as host arrays (`Site.arrays`), which tools/fla_momentum_reference.py hands fla-core 0.6.0's
`chunk_momentum_delta_rule` on a GPU and records (`RECORDED`), and Dew's leaves of it (`ruled`).

- `momentum/bf16`: one sequence a row in bf16, its erase keys its own (fla's `p`, taken as given:
  `use_p_times_alpha=False`), its keys and queries as given (`use_qk_l2norm_in_kernel=False`). fla's chunked
  rule takes no fp32 call and no packed documents (it refuses both), so no site holds either.
"""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk

RECORDED = Path(__file__).resolve().parent / "fixtures" / "fla" / "momentum_seeded.npz"
"""fla-core 0.6.0's recording at every site (tools/fla_momentum_reference.py)."""

ROWS, LENGTH, HEADS, KEYS, VALUES = 4, 288, 4, 32, 16
"""Four whole chunks of fla's 64 tokens and a part one, keys wider than values, over four rows and four
heads: at two rows of 160 tokens over two heads, fla's reference against itself resolved no RMS ratio
within 512 seeds at the write gates', the log decays', the entering pair's and the keys' gradients (its
recorder's seed sizing, M-JS-265b on an A100)."""

HOOKS = recurrence.Hooks(decay='scalar', coupled='momentum', erase='untied', write_bound=1.0,
                         scale=KEYS ** -0.5)
"""fla's chunked momentum as the sites call it: its scale its default, `Dk^-1/2`, its write gates a
sigmoid's."""

LEAVES = ("out", "final", "dq", "dk", "dv", "dp", "dg", "dmu", "dbeta", "dh0")
"""A site's leaves: the output and final pair, then each operand's gradient."""


class Site(NamedTuple):
    draw: Callable[[int], dict]
    """A seed's operands and the cotangents of the output and the final pair, by name."""
    leaves: tuple[str, ...] = LEAVES

    def arrays(self, seed: int) -> dict[str, np.ndarray]:
        """The seed's draw as host arrays, the recording's inputs."""
        return {name: np.asarray(value) for name, value in self.draw(seed).items()}


def _drawn(seed: int, dtype) -> dict:
    """A seed's operands as fla's layer forms them, queries, keys, values and erase keys in `dtype`: unit
    queries, keys and erase keys, the log decays `log sigmoid` of normals (the memory's near one, the
    momentum's held at -2 or above, fla's `min_log_mu`), the write gate a sigmoid's; an entering pair `[B,
    H, 2, Dk, Dv]`, and the cotangents of the output and the final pair: normals the same on every host, as
    a recorded site draws them (testing.md, `input_digest`)."""
    from recorded_inputs import normals
    from test_recurrence_chunk import hosted
    from test_recurrence_momentum import momentum_draw

    draws = iter(range(12))

    def normal(shape):
        """The seed's next normals, the same bits on every host (`recorded_inputs.normals`)."""
        return normals([1454, 7, seed, next(draws)], shape, np.float64)

    drawn = momentum_draw(normal, ROWS, LENGTH, HEADS, KEYS, VALUES)
    return {name: hosted(value, dtype if name in ("q", "k", "v", "p", "dy") else jnp.float32)
            for name, value in drawn.items()}


def chunk_of(arrays: dict) -> tuple[Chunk, tuple]:
    """A site's arrays as the chunked op's operands and the cotangents they were drawn with."""
    return (Chunk(arrays["q"], arrays["k"], arrays["v"], log_decay=arrays["g"], beta=arrays["beta"],
                  momentum_decay=arrays["mu"], erase_key=arrays["p"], state=arrays["h0"]),
            (arrays["dy"], arrays["dfinal"]))


def ruled(rule: Callable[[Chunk], tuple], arrays: dict, leaves: tuple[str, ...] = LEAVES) -> dict:
    """`rule`'s leaves on a site's arrays: its output and final pair and each operand's gradient under the
    drawn cotangents."""
    operands, (dy, dfinal) = chunk_of(arrays)
    (out, final), pull = jax.vjp(rule, operands)
    (grads,) = pull((jnp.asarray(dy).astype(out.dtype), jnp.asarray(dfinal).astype(final.dtype)))
    found = {"out": out, "final": final, "dq": grads.query, "dk": grads.key, "dv": grads.value,
             "dp": grads.erase_key, "dg": grads.log_decay, "dmu": grads.momentum_decay, "dbeta": grads.beta,
             "dh0": grads.state}
    return {name: found[name] for name in leaves}


def token_by_token(given: Chunk) -> tuple:
    """The recurrence a token at a time (`test_recurrence_momentum.tokenwise`) on a site's operands."""
    from test_recurrence_momentum import tokenwise

    return tokenwise(given.query, given.key, given.value, given.log_decay, given.beta, given.momentum_decay,
                     given.erase_key, given.state)


def sites() -> dict[str, Site]:
    """Every recorded site by name."""
    return {"momentum/bf16": Site(functools.partial(_drawn, dtype=jnp.bfloat16))}


def the_chunks_are_as_exact_as_fla(rule: Callable[[Chunk], tuple], site: str) -> None:
    """`rule` (a chunked call) on par from float64 with recorded fla-core at `site`'s declared seeds (check
    1), the truth the recurrence a token at a time in float64, which the recorder held to fla's own reference
    at every seed."""
    from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording

    seeds, leaves = declared_seeds(RECORDED)[site], sites()[site].leaves
    draw, reference, truth = indexed_recording(
        recording(RECORDED), site, seeds, inputs_of=sites()[site].arrays, leaves=leaves,
        truth_fn=jax.jit(lambda arrays: ruled(token_by_token, arrays, leaves)))

    def run(index, arrays):
        del index
        return ruled(rule, arrays, leaves)

    assert_as_exact(run, as_given(reference), draw, f"the chunked momentum against fla, {site}", seeds=seeds,
                    truth_fn=as_given(truth))


def dew_chunks(given: Chunk) -> tuple:
    """The chunked op under the sites' hooks (`HOOKS`) in fla's chunks of 64 tokens."""
    from dew.nn.kernels.recurrence_chunk import chunk

    return chunk(HOOKS, given, chunk_size=64)
