from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew.registry import models

from ..dit import (
    ROPE_THETA,
    ConditioningEmbed,
    ModulatedBlock,
    PatchSequenceEmbed,
    PatchSequenceOutput,
    RematChoice,
    remat_block,
    rope_for_scan,
)
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
class SimpleDiT(nn.Module):
    """Standard DiT: a plain stack of adaLN-Zero attention blocks.

    `adaln_silu=False` and `text_pooling="all"` use FlaxDiff 0.2's
    conditioning, as in `SimpleUDiT`.

    `routes` is TREAD's token routing (Krause et al. 2025, "TREAD: Token
    Routing for Efficient Architecture-agnostic Diffusion Training"), in
    training only: each `(ratio, start, end)` draws `int(tokens * ratio)`
    tokens uniformly per example that skip blocks `start` to `end`
    inclusive, which the rest pass through, and rejoin after block `end`
    holding the values they entered block `start` with, as CompVis/tread's
    `Router` gathers and scatters them. The kept tokens stay in sequence
    order and rotate at their own positions. The draw reads the `dropout`
    stream. Sampling runs every token through every block.

    `patch_bottleneck` is JiT's bottleneck patch embedding (Li & He 2025,
    "Back to Basics: Let Denoising Generative Models Denoise"), for large
    pixel patches: 128 in its models.

    `interval` makes it an interval model (`Process.interval`), reading the
    `duration` of the interval it predicts over beside the time: MeanFlow's
    and shortcut models' network.
    """
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout_rate: float = 0.0  # Typically 0 for diffusion
    dtype: Dtype | None = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    qk_norm: bool = False
    attention_impl: str = "auto"  # an AttentionImpl
    remat: RematChoice = False
    scan_order: Literal["raster", "hilbert", "zigzag"] = "raster"
    adaln_silu: bool = True
    text_pooling: Literal["real", "all"] = "real"
    routes: Sequence[Sequence[float]] = ()
    patch_bottleneck: int | None = None
    interval: bool = False


    def setup(self):
        self.embed = PatchSequenceEmbed(
            patch_size=self.patch_size,
            emb_features=self.emb_features,
            scan_order=self.scan_order,
            dtype=self.dtype,
            precision=self.precision,
            bottleneck=self.patch_bottleneck,
        )
        self.conditioning = ConditioningEmbed(
            emb_features=self.emb_features,
            mlp_ratio=self.mlp_ratio,
            dtype=self.dtype,
            precision=self.precision,
            text_pooling=self.text_pooling,
            interval=self.interval,
        )
        self.blocks = self.stack()
        self.output = PatchSequenceOutput(
            patch_size=self.patch_size,
            output_channels=self.output_channels,
            norm_epsilon=self.norm_epsilon,
            dtype=self.dtype,
            precision=self.precision,
        )

    def stack(self) -> list[ModulatedBlock]:
        """The layers between the patch embedding and the output, all attention."""
        return [
            remat_block(ModulatedBlock, self.remat)(
                features=self.emb_features,
                num_heads=self.num_heads,
                mixer='attention',
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                norm_epsilon=self.norm_epsilon,
                adaln_silu=self.adaln_silu,
                qk_norm=self.qk_norm,
                attention_impl=self.attention_impl,
                name=f"dit_block_{i}"
            ) for i in range(self.num_layers)
        ]

    def __call__(self, x, temb, textcontext=None, train: bool = False, duration=None):
        _, H, W, _ = x.shape
        x_seq, inv_idx = self.embed(x)
        cond_emb = self.conditioning(temb, textcontext, duration)
        freqs_cis = rope_for_scan(x_seq, self.emb_features // self.num_heads, self.scan_order)

        starts = {int(start): (ratio, int(end)) for ratio, start, end in self.checked_routes()} if train else {}
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
