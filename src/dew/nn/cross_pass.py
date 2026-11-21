"""Cross-pass messages: a looped decoder's passes coupled through the block summaries the pass before left.

Under `Loop.messages` every pass enters the looped block at the prelude's output, and the block's layers
are cut into blocks of `block_size` (the last shorter where they do not divide). Each pass leaves each
block's residual summary (`CausalTransformer.residual_summary`) in a bank `[B, S, blocks, D]`, and every
sublayer input of a later pass merges that bank in before its norm or projection (`CrossPassMerge`), one
merge per input site: each of its mixer's (`dew.nn.protocols.InputSites`) and its feed-forward's. The
first pass reads no bank, so it is the decoder's own forward; its gates drawn at zero, every later pass
is the first again, and a pretrained decoder gains passes without changing what it computes. A merge
(Inkling 2 Mini's cross-pass controller, generalized past its standard residual):

    m      = rms_normed(depth_attention(bank))                    # the passes' message, unit RMS
    σ      = sqrt(mean(u²) + eps),  û = u / σ                     # the sublayer input u, at unit RMS
    h      = û W_u + m W_m                                        # a rank-r controller state
    forget = tanh(h W_f),  gain = tanh(h W_g),  add = tanh(h W_a)
    out    = (1 - forget) u + σ (gain m + add)

in at least fp32 whatever the run computes in, cast back to `u`'s dtype; `additive` drops `add`, and
without `scale_by_input_rms` the message joins at unit scale rather than at the input's.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .attention_residuals import DepthAttention
from .flops import projection_flops
from .norms import unweighted_rmsnorm
from .precision import at_least_fp32
from .sharding import declared

MERGE_PREFIX = "cross_pass_"
"""The name of a block's merges, each `MERGE_PREFIX` and its site (`DecoderBlock.cross_pass`)."""

MESSAGES_UNWRITTEN = ("the source's format holds no cross-pass messages, which this decoder has trained; a "
                      "run's checkpoint or release keeps them")
"""Why a decoder holding merges is written in no family's layout (`Pretrained.export`)."""


def holds_merges(params) -> bool:
    """Whether `params` holds a block's merges anywhere (`MERGE_PREFIX`)."""
    paths = jax.tree_util.tree_flatten_with_path(params)[0]
    return any(str(getattr(key, "key", "")).startswith(MERGE_PREFIX) for path, _ in paths for key in path)


@dataclasses.dataclass(frozen=True)
class CrossPass:
    """How a looped block's passes pass messages (`Loop.messages`, the module doc): the layers a bank
    summary closes (`block_size`), the controller's rank, whether the message adds a term of its own
    (`additive`), whether the depth mixture sums the summaries at unit RMS (`normalize_values`), whether
    the message joins at the input's RMS (`scale_by_input_rms`), and the merge's norm epsilon (None: the
    decoder's `norm_eps`)."""

    block_size: int = 4
    rank: int = 64
    additive: bool = True
    normalize_values: bool = True
    scale_by_input_rms: bool = True
    norm_eps: float | None = None

    def __post_init__(self):
        if self.block_size < 1 or self.rank < 1:
            raise ValueError(f"cross-pass messages summarize blocks of at least one layer through a "
                             f"controller of rank at least 1, got block_size={self.block_size}, "
                             f"rank={self.rank}")

    def blocks(self, first: int, end: int) -> tuple[tuple[int, int], ...]:
        """The blocks layers [first, end) cut into, each `[start, stop)`, the last shorter where
        `block_size` does not divide them."""
        return tuple((start, min(start + self.block_size, end))
                     for start in range(first, end, self.block_size))


class CrossPassMerge(nn.Module):
    """One input site's merge of the bank the pass before left into its sublayer input (the module doc).
    Its parameters are declared in `setup`, so the tree holds them whether or not a pass reads a bank:
    a decoder initialized at one pass resumes at more."""

    spec: CrossPass
    emb_features: int
    norm_eps: float

    def setup(self):
        width, rank = self.emb_features, self.spec.rank
        self.depth = DepthAttention(width, norm_eps=self.norm_eps,
                                    normalize_values=self.spec.normalize_values, name='depth')
        down = declared(nn.initializers.normal(1 / math.sqrt(2 * width)), ('embed', None))
        up = declared(nn.initializers.zeros, (None, 'embed'))
        self.input_down = self.param('input_down', down, (width, rank), jnp.float32)
        self.message_down = self.param('message_down', down, (width, rank), jnp.float32)
        self.forget = self.param('forget', up, (rank, width), jnp.float32)
        self.gain = self.param('gain', up, (rank, width), jnp.float32)
        if self.spec.additive:
            self.added = self.param('added', up, (rank, width), jnp.float32)

    def declare(self) -> tuple[jax.Array, jax.Array]:
        """The merge's parameters declared where no pass reads a bank: its depth mixture's scale and its
        input's down projection, whose reading runs both `setup`s."""
        return self.depth.scale, self.input_down

    def __call__(self, read: jax.Array, bank: jax.Array) -> jax.Array:
        """`read` `[B, S, D]` with `bank` `[B, S, blocks, D]` merged in, in `read`'s dtype."""
        wide = at_least_fp32(read.dtype)
        highest = jax.lax.Precision.HIGHEST

        def times(x, kernel):
            return jnp.matmul(x, kernel.astype(wide), precision=highest)

        u = read.astype(wide)
        message = unweighted_rmsnorm(self.depth(bank.astype(wide)), self.norm_eps, implementation='xla')
        sigma = jnp.sqrt(jnp.mean(jnp.square(u), axis=-1, keepdims=True) + jnp.asarray(self.norm_eps, wide))
        state = times(u / sigma, self.input_down) + times(message, self.message_down)
        sent = jnp.tanh(times(state, self.gain)) * message
        if self.spec.additive:
            sent = sent + jnp.tanh(times(state, self.added))
        kept = (1 - jnp.tanh(times(state, self.forget))) * u
        return (kept + (sigma * sent if self.spec.scale_by_input_rms else sent)).astype(read.dtype)

    @nn.nowrap
    def merge_flops(self, lengths, blocks: int, *, trained: bool = False) -> float:
        """A merge's work over `lengths` tokens and `blocks` summaries: its depth mixture's (its learned
        score's alone where `trained` counts learned matrices' work) and its controller's five projections
        (four with no added term) through rank `rank`."""
        depth = DepthAttention(self.emb_features, parent=None)
        mixed = (depth.depth_parameter_flops if trained else depth.depth_flops)(lengths, blocks)
        projections = 4 + int(self.spec.additive)
        return mixed + projections * projection_flops(lengths, self.emb_features, self.spec.rank)
