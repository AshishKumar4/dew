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

its products in the sublayer's compute dtype under the run's precision (`dew.nn.precision.computed_dot`),
its statistics, gates and mix in at least fp32, cast back to `u`'s dtype; `additive` drops `add`, and
without `scale_by_input_rms` the message joins at unit scale rather than at the input's.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .attention_residuals import DepthAttention, depth_read
from .dense import watched_or_owned
from .derivatives import portable_formed
from .flops import projection_flops
from .norms import unweighted_rmsnorm
from .precision import at_least_fp32, computed_dot
from .sharding import declared

MERGE_PREFIX = "cross_pass_"
"""The name of a block's merges, each `MERGE_PREFIX` and its site (`DecoderBlock.cross_pass`)."""

MESSAGES_UNWRITTEN = ("the source's format holds no cross-pass messages, which this decoder has trained; a "
                      "run's checkpoint or release keeps them")
"""Why a decoder holding merges is written in no family's layout (`Pretrained.export`)."""


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

    def merge_flops(self, lengths, width: int, blocks: int, *, trained: bool = False) -> float:
        """One merge's work (`CrossPassMerge`) over `lengths` tokens of `width` and `blocks` summaries: its
        depth mixture's (its learned score's alone where `trained` counts learned matrices' work) and its
        controller's five projections (four with no added term) through rank `rank`."""
        depth = DepthAttention(width, parent=None)
        mixed = (depth.depth_parameter_flops if trained else depth.depth_flops)(lengths, blocks)
        projections = 4 + int(self.additive)
        return mixed + projections * projection_flops(lengths, width, self.rank)

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

    def controller(self) -> tuple[jax.Array, ...]:
        """The controller's projections: input and message down, gain, forget, and an additive's added."""
        return (self.input_down, self.message_down, self.gain, self.forget,
                *((self.added,) if self.spec.additive else ()))

    def __call__(self, read: jax.Array, bank: jax.Array) -> jax.Array:
        """`read` `[B, S, D]` with `bank` `[B, S, blocks, D]` merged in, in `read`'s dtype."""
        return merged_sites((self,), read, bank)[0]


def merged_sites(merges: Sequence[CrossPassMerge], read: jax.Array, bank: jax.Array) -> tuple[jax.Array, ...]:
    """Each of `merges`' merge of `bank` `[B, S, blocks, D]` into `read` `[B, S, D]` (a block's input sites
    of one sublayer input, `DecoderBlock`'s), in `read`'s dtype: the bank read and normalized once for them
    all and every site's depth query scored in one product (`depth_read`, row 12923), each site's controller
    its own (`_merged`). Each merge's parameters are its own as `CrossPassMerge` declares them, so a saved
    decoder loads as it is."""
    first = merges[0]
    if any(merge.spec != first.spec or merge.norm_eps != first.norm_eps for merge in merges):
        raise ValueError("merges read one bank together only under one message spec and norm epsilon")
    spec, eps, depth = first.spec, first.norm_eps, first.depth
    if any(watched_or_owned(merge) or watched_or_owned(merge.depth) for merge in merges):
        # A merge a quantization may own (a rule over its depth mixture's path among them) or a caller
        # watches, it or its depth mixture, runs as it is, each apart.
        return tuple(_merged(read, merge.depth(bank), merge.controller(), spec, eps) for merge in merges)

    # The read's and the bank's cotangents leave only with the merges' parameters' gradients formed, their
    # states recomputed from their operands in the reverse mode (`portable_formed`): XLA otherwise puts the
    # controllers' low-rank products' and the depth queries' gradients off to the step's end, holding those
    # states live across layers and an exit-gated loop's unrolled passes (row 11713).
    def merged(read, bank, parameters):
        wide = at_least_fp32(read.dtype)
        queries = jnp.stack([scale.astype(wide) * kernel[:, 0].astype(wide)
                             for scale, kernel, *_ in parameters], axis=1)
        mixed = depth_read(bank.astype(read.dtype), queries, norm_eps=depth.norm_eps,
                           normalize_values=depth.normalize_values)
        return tuple(_merged(read, mixed[:, :, site], controller, spec, eps)
                     for site, (_, _, *controller) in enumerate(parameters))
    parameters = tuple((merge.depth.scale, merge.depth.kernel, *merge.controller()) for merge in merges)
    return portable_formed(merged)(read, bank, parameters)


def _merged(read: jax.Array, mixed: jax.Array, controller, spec: CrossPass, eps: float) -> jax.Array:
    """`CrossPassMerge`'s merge of `read` `[B, S, D]` with the bank's depth mixture `mixed` `[B, S, D]` by
    its `controller` (input and message down projections, gain, forget, and an additive's added), in
    `read`'s dtype: its two down projections one product over the joined input and message, its up
    projections one product over their joined kernels (row 12922), each in `read`'s compute dtype under
    the run's precision (`dew.nn.precision.computed_dot`: a bf16 run's Tensor Cores, `ACCURATE`'s exact
    float32; row 12918), its norms' statistics, gates and mix in at least fp32."""
    input_down, message_down, *ups = controller
    compute = read.dtype
    wide = at_least_fp32(compute)
    u = read.astype(wide)
    message = unweighted_rmsnorm(mixed.astype(wide), eps, implementation='xla')
    sigma = jnp.sqrt(jnp.mean(jnp.square(u), axis=-1, keepdims=True) + jnp.asarray(eps, wide))
    state = computed_dot(jnp.concatenate([u / sigma, message], axis=-1),
                         jnp.concatenate([input_down, message_down], axis=0), dtype=compute)
    gates = jnp.tanh(computed_dot(state, jnp.concatenate(ups, axis=1), dtype=compute))
    gain, forget, *added = jnp.split(gates, len(ups), axis=-1)
    sent = gain * message + (added[0] if added else 0)
    kept = (1 - forget) * u
    return (kept + (sigma * sent if spec.scale_by_input_rms else sent)).astype(read.dtype)
