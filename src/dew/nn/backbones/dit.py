from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp

from dew.registry import models

from ..dit import ROPE_THETA, ModulatedBlock, _DiTStackOptions, remat_block, rope_for_scan
from ..precision import at_least_fp32
from ..rope import rotary_freqs


def gather_tokens(tokens: jax.Array, kept: jax.Array) -> jax.Array:
    """The `[B, K, F]` tokens at the `[B, K]` indices `kept`: TREAD's
    `Router.start_route`."""
    return jnp.take_along_axis(tokens, kept[..., None], axis=1)


def scatter_tokens(held: jax.Array, kept: jax.Array, tokens: jax.Array) -> jax.Array:
    """`held` with the `[B, K, F]` tokens written back at `kept`: TREAD's
    `Router.end_route`."""
    return held.at[jnp.arange(held.shape[0])[:, None], kept].set(tokens)


@models("simple_dit")
class SimpleDiT(_DiTStackOptions):
    """Standard DiT: a plain stack of adaLN-Zero attention blocks.

    `adaln_silu=False` and `text_pooling="all"` are FlaxDiff 0.2's conditioning,
    as in `SimpleUDiT`. `routes` is TREAD's token routing (Krause et al. 2025)
    in training: each `(ratio, start, end)` draws `int(tokens * ratio)` tokens
    per example, from the `dropout` stream, that skip blocks `start` to `end`
    and rejoin after `end` with the values they entered with, as CompVis/tread's
    `Router` does; kept tokens stay in order at their own positions. Sampling
    runs every token through every block. `patch_bottleneck` is JiT's bottleneck
    patch embedding (Li & He 2025), 128 in its models. `interval` reads the
    `duration` of the predicted interval beside the time (MeanFlow, shortcut
    models; `Process.interval`), and `time_scale` scales the Fourier
    frequencies, small for a model trained through a JVP in time.
    """
    adaln_silu: bool = True
    text_pooling: Literal["real", "all"] = "real"
    routes: Sequence[Sequence[float]] = ()
    patch_bottleneck: int | None = None
    interval: bool = False
    time_scale: float = 16


    def setup(self):
        self.embed = self._embedding(self.patch_size, self.emb_features, self.scan_order,
                                     bottleneck=self.patch_bottleneck)
        self.conditioning = self._conditioning(self.emb_features, text_pooling=self.text_pooling,
                                              interval=self.interval, time_scale=self.time_scale)
        self.blocks = self.stack()
        self.output = self._output(self.patch_size, self.output_channels)

    def stack(self) -> list[ModulatedBlock]:
        """The layers between the patch embedding and the output, all attention."""
        return [
            remat_block(ModulatedBlock, self.remat)(
                features=self.emb_features,
                num_heads=self.num_heads,
                mixer='attention',
                adaln_silu=self.adaln_silu,
                **self._block_options(),
                name=f"dit_block_{i}"
            ) for i in range(self.num_layers)
        ]

    def __call__(self, x, temb, textcontext=None, train: bool = False, duration=None):
        _, H, W, _ = x.shape
        x_seq, inv_idx = self.embed(x)
        cond_emb = self.conditioning(temb, textcontext, duration)
        freqs_cis = rope_for_scan(x_seq, self.emb_features // self.num_heads, self.scan_order)

        starts = (
            {int(start): (ratio, int(end)) for ratio, start, end in self.checked_routes()} if train else {}
        )
        rotation, held, kept, end = freqs_cis, None, None, None
        for index, block in enumerate(self.blocks):
            if index in starts:
                ratio, end = starts[index]
                held, kept = x_seq, self.kept_tokens(x_seq, ratio, index)
                x_seq = gather_tokens(x_seq, kept)
                if freqs_cis is not None:
                    rotation = rotary_freqs(kept, self.emb_features // self.num_heads, ROPE_THETA,
                                            dtype=at_least_fp32(x_seq.dtype))
            x_seq = block(x_seq, cond_emb, rotation, train)
            if index == end:
                x_seq = scatter_tokens(held, kept, x_seq)
                rotation, held, kept, end = freqs_cis, None, None, None

        return self.output(x_seq, inv_idx, H, W)

    def checked_routes(self) -> list[tuple[float, int, int]]:
        """`routes` as `(ratio, start, end)`, refused unless each ratio is in
        (0, 1) and the spans are ordered, disjoint and inside the stack."""
        routes = [(float(ratio), int(start), int(end)) for ratio, start, end in self.routes]
        following = 0
        for ratio, start, end in routes:
            if not 0.0 < ratio < 1.0 or not following <= start <= end < self.num_layers:
                raise ValueError(f"a route is (ratio in (0, 1), start, end) over ordered, disjoint "
                                 f"spans of the {self.num_layers} blocks; got {self.routes}")
            following = end + 1
        return routes

    def kept_tokens(self, tokens: jax.Array, ratio: float, start: int) -> jax.Array:
        """The `[B, keep]` indices of the tokens a route at `start` computes on:
        the reference's first `keep` of a uniform shuffle, in sequence order."""
        batch, count, _ = tokens.shape
        noise = jax.random.uniform(self.make_rng("dropout"), (batch, count))
        kept = jnp.sort(jnp.argsort(noise, axis=1)[:, :count - int(count * ratio)], axis=1)
        self.sow("intermediates", f"route_{start}", kept)
        return kept
