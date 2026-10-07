"""Video DiT with factorized spatial-temporal attention, built from the shared DiT blocks.

Each layer is a spatial ModulatedBlock over the patch tokens of every frame,
followed by a temporal ModulatedBlock over the frame axis of every patch
position. This is the standard factorized design, so the spatial half's compute
grows linearly in T and the temporal half's linearly in S.
"""

import jax.numpy as jnp

from ..dit import ROPE_THETA, ModulatedBlock, _DiTStackOptions, remat_block, rope_for_scan
from ..precision import at_least_fp32
from ..rope import rotary_freqs


class VideoDiT(_DiTStackOptions):
    """Factorized spatial-temporal DiT over (B, T, H, W, C) inputs."""
    def setup(self):
        self.embed = self._embedding(self.patch_size, self.emb_features, self.scan_order)
        self.conditioning = self._conditioning(self.emb_features)

        def block(name):
            return remat_block(ModulatedBlock, self.remat)(
                features=self.emb_features,
                num_heads=self.num_heads,
                mixer='attention',
                **self._block_options(),
                name=name,
            )

        self.spatial_blocks = [block(f"spatial_block_{i}") for i in range(self.num_layers)]
        self.temporal_blocks = [block(f"temporal_block_{i}") for i in range(self.num_layers)]

        self.output = self._output(self.patch_size, self.output_channels)

    def __call__(self, x, temb, textcontext=None, train: bool = False):
        B, T, H, W, C = x.shape

        # Per-frame patchify; the permutation is identical for every frame
        frames = x.reshape(B * T, H, W, C)
        tokens, inv_idx = self.embed(frames)  # [B*T, S, F]
        S = tokens.shape[1]

        cond_emb = self.conditioning(temb, textcontext)  # [B, F]
        cond_spatial = jnp.repeat(cond_emb, T, axis=0)   # [B*T, F]
        cond_temporal = jnp.repeat(cond_emb, S, axis=0)  # [B*S, F]

        dim_head = self.emb_features // self.num_heads
        freqs_spatial = rope_for_scan(tokens, dim_head, self.scan_order)
        # Time is a genuine 1D axis, RoPE applies directly
        freqs_temporal = rotary_freqs(jnp.arange(T), dim_head, ROPE_THETA,
                                      dtype=at_least_fp32(tokens.dtype))

        for spatial, temporal in zip(self.spatial_blocks, self.temporal_blocks, strict=True):
            tokens = spatial(tokens, cond_spatial, freqs_spatial, train)
            # [B*T, S, F] -> [B*S, T, F]
            tokens = tokens.reshape(B, T, S, -1).transpose(0, 2, 1, 3).reshape(B * S, T, -1)
            tokens = temporal(tokens, cond_temporal, freqs_temporal, train)
            # back to [B*T, S, F]
            tokens = tokens.reshape(B, S, T, -1).transpose(0, 2, 1, 3).reshape(B * T, S, -1)

        out_frames = self.output(tokens, inv_idx, H, W)  # [B*T, H, W, C]
        return out_frames.reshape(B, T, H, W, self.output_channels)
