import functools
from collections.abc import Sequence
from typing import ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.mesh import LayoutRefused

from ..dit import ROPE_THETA, ModulatedBlock, _DiTStackOptions, remat_block, rope_for_scan
from ..flops import MissingFlopsCount, counted_flops, patch_lengths
from ..precision import at_least_fp32
from ..rope import rotary_freqs
from ..scatter import gather_positions
from .decoder_stack import pipelined_blocks, pipelines


def scatter_tokens(held: jax.Array, kept: jax.Array, tokens: jax.Array) -> jax.Array:
    """`held` with the `[B, K, F]` tokens written back at `kept`: TREAD's
    `Router.end_route`."""
    return held.at[jnp.arange(held.shape[0])[:, None], kept].set(tokens)


class SimpleDiT(_DiTStackOptions):
    """The standard DiT: a plain stack of adaLN-Zero attention blocks.

    - `adaln_silu=False` and `text_pooling="all"` give FlaxDiff 0.2's
      conditioning, as in `SimpleUDiT`.
    - `routes` is TREAD's token routing (Krause et al. 2025) during training.
      Each `(ratio, start, end)` draws `int(tokens * ratio)` tokens per
      example, from the `dropout` stream, that skip blocks `start` to `end`
      and rejoin after `end` with the values they had on entry, as
      CompVis/tread's `Router` does; the kept tokens stay in order at their
      own positions. Sampling runs every token through every block.
    - `patch_bottleneck` is JiT's bottleneck patch embedding (Li and He 2025),
      128 in its models.
    - `interval` adds the `duration` of the predicted interval next to the
      time, for MeanFlow and shortcut models (`Process.interval`).
    - `time_scale` scales the Fourier frequencies; keep it small for a model
      trained through a JVP in time.
    """
    adaln_silu: bool = True
    text_pooling: Literal["real", "all"] = "real"
    routes: Sequence[Sequence[float]] = ()
    patch_bottleneck: int | None = None
    interval: bool = False
    time_scale: float = 16

    block_prefixes: ClassVar[tuple[str, ...]] = ("dit_block_",)
    """The names the stack's blocks are stored under, `{prefix}N`, one prefix a mixer."""

    @nn.nowrap
    def counted_blocks(self) -> list[ModulatedBlock]:
        """The layers `stack` builds, unbound and without lifted remat, for host-side work
        accounting; a configuration `stack` refuses is refused here too."""
        return self.stack(counted=True)

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """Input channels and conditioning width belong to the objective's geometry."""
        raise MissingFlopsCount("SimpleDiT requires image_flops(lengths, input_channels, text_features)")

    @nn.nowrap
    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray:
        """Patch-token lengths from the actual latent/image shape, never a guessed channel count."""
        return patch_lengths(sample_shape, (self.patch_size, self.patch_size), rows)

    @nn.nowrap
    def image_flops(self, lengths, input_channels: int, text_features: int | None = None) -> float:
        """Every matmul, including patch/time/text input maps; only lookups excluded."""
        embed = self._embedding(self.patch_size, self.emb_features, bottleneck=self.patch_bottleneck)
        condition = self._conditioning(self.emb_features, interval=self.interval)
        total = embed.model_flops(lengths, input_channels) + condition.model_flops(lengths, text_features)
        for index, block in enumerate(self.counted_blocks()):
            active = np.asarray(lengths)
            for ratio, start, end in self.checked_routes():
                if start <= index <= end:
                    active = active - np.floor(active * ratio).astype(np.int64)
            total += counted_flops(block, active)
        return total + self._output(self.patch_size, self.output_channels).model_flops(
            lengths, self.emb_features)


    def setup(self):
        self.embed = self._embedding(self.patch_size, self.emb_features, self.scan_order,
                                     bottleneck=self.patch_bottleneck)
        self.conditioning = self._conditioning(self.emb_features, text_pooling=self.text_pooling,
                                              interval=self.interval, time_scale=self.time_scale)
        self.blocks = self.stack()
        self.output = self._output(self.patch_size, self.output_channels)

    def stack(self, *, counted: bool = False) -> list[ModulatedBlock]:
        """The layers between the patch embedding and the output, all attention; `counted`,
        unbound and without remat (`counted_blocks`)."""
        return [self.layer(index, kind, counted=counted) for index, kind in enumerate(self.kinds())]

    @nn.nowrap
    def kinds(self) -> list[str]:
        """Each layer's mixer, in order."""
        return ['attn'] * self.num_layers

    @nn.nowrap
    def layer(self, index: int, kind: str, *, counted: bool = False) -> ModulatedBlock:
        """Layer `index`'s block, of mixer `kind`, stored as `dit_block_{index}`; `counted`, unbound
        and without remat."""
        block = functools.partial(ModulatedBlock if counted else remat_block(ModulatedBlock, self.remat),
                                  features=self.emb_features, num_heads=self.num_heads, mixer='attention',
                                  adaln_silu=self.adaln_silu, **self._block_options(),
                                  name=f"dit_block_{index}")
        return block(parent=None) if counted else block()

    def __call__(self, x, temb, textcontext=None, train: bool = False, duration=None):
        _, H, W, _ = x.shape
        x_seq, inv_idx = self.embed(x)
        cond_emb = self.conditioning(temb, textcontext, duration)
        freqs_cis = rope_for_scan(x_seq, self.emb_features // self.num_heads, self.scan_order)

        starts = (
            {int(start): (ratio, int(end)) for ratio, start, end in self.checked_routes()} if train else {}
        )
        if pipelines(self, self.kinds()):
            if starts:
                raise LayoutRefused(
                    "TREAD's routes take part of the tokens through a span of the blocks, and a pipeline's "
                    "stages over the stage axis run one program over every token; train the routed model "
                    "on a mesh without a stage axis")
            return self.output(self.pipelined(x_seq, cond_emb, freqs_cis, train), inv_idx, H, W)
        rotation, held, kept, end = freqs_cis, None, None, None
        for index, block in enumerate(self.blocks):
            if index in starts:
                ratio, end = starts[index]
                held, kept = x_seq, self.kept_tokens(x_seq, ratio, index)
                # TREAD's `Router.start_route`: the tokens the router keeps.
                x_seq = gather_positions(x_seq, kept)
                if freqs_cis is not None:
                    rotation = rotary_freqs(kept, self.emb_features // self.num_heads, ROPE_THETA,
                                            dtype=at_least_fp32(x_seq.dtype))
            x_seq = block(x_seq, cond_emb, rotation, train)
            if index == end:
                x_seq = scatter_tokens(held, kept, x_seq)
                rotation, held, kept, end = freqs_cis, None, None, None

        return self.output(x_seq, inv_idx, H, W)

    def pipelined(self, tokens, conditioning, freqs_cis, train: bool):
        """The blocks over `tokens` as a pipeline over the stage axis (`pipelined_blocks`), each row
        modulated by its own `conditioning`, every one rotated alike."""
        kinds = self.kinds()

        def run(blocks, x, rows):
            for block in blocks:
                x = block(x, rows, freqs_cis, train)
            return x

        return pipelined_blocks(self, self.block_prefixes, len(kinds), tokens,
                                lambda index: self.layer(index, kinds[index]), run, rows=conditioning)

    def checked_routes(self) -> list[tuple[float, int, int]]:
        """Return `routes` as `(ratio, start, end)` tuples, after checking them.

        Each ratio must be in (0, 1), and the spans must be ordered, disjoint
        and inside the stack."""
        routes = [(float(ratio), int(start), int(end)) for ratio, start, end in self.routes]
        following = 0
        for ratio, start, end in routes:
            if not 0.0 < ratio < 1.0 or not following <= start <= end < self.num_layers:
                raise ValueError(f"a route is (ratio in (0, 1), start, end) over ordered, disjoint "
                                 f"spans of the {self.num_layers} blocks; got {self.routes}")
            following = end + 1
        return routes

    def kept_tokens(self, tokens: jax.Array, ratio: float, start: int) -> jax.Array:
        """Return the `[B, keep]` indices of the tokens a route at `start` computes on.

        As in the reference, these are the first `keep` of a uniform shuffle,
        in sequence order."""
        batch, count, _ = tokens.shape
        noise = jax.random.uniform(self.make_rng("dropout"), (batch, count))
        kept = jnp.sort(jnp.argsort(noise, axis=1)[:, :count - int(count * ratio)], axis=1)
        self.sow("intermediates", f"route_{start}", kept)
        return kept
