"""
Hybrid SSM-attention DiT: interleaves linear-time S5 blocks with attention
blocks in a configurable ratio. The S5 layers live in ssm.py; the block and
the patchify/conditioning/output machinery live in dit.py.
"""

from collections.abc import Sequence

from dew.registry import models

from ..dit import ModulatedBlock, build_block_pattern, remat_block
from .dit import SimpleDiT

DEFAULT_SSM_RATIO = "3:1"


@models("hybrid_dit")
class HybridSSMAttentionDiT(SimpleDiT):
    """DiT that interleaves SSM blocks with attention blocks.

    The mixer of every layer comes from `ssm_attention_ratio`, a shorthand
    that reads the same at any depth ("3:1", "all-ssm"), or from
    `block_pattern`, which names each layer. Setting both raises a ValueError
    at setup. Everything around the layers is `SimpleDiT`'s.
    """
    ssm_state_dim: int = 64
    block_pattern: Sequence[str] | None = None  # e.g., ['ssm','ssm','ssm','attn']
    ssm_attention_ratio: str = DEFAULT_SSM_RATIO  # "3:1", "1:1", "all-ssm", "all-attn"
    bidirectional_ssm: bool = True
    use_2d_fusion: bool = False  # 2D state fusion in SSM blocks (see SpatialFusionConv)

    def block(self, index: int, block_type: str) -> ModulatedBlock:
        """Build layer `index`'s block, an SSM mixer or an attention one.

        The two share the width, the MLP ratio and the norms. They differ
        in the mixer they name, the fields only that mixer reads, the remat
        policy, and the name a checkpoint stores them under.
        """
        def build(block_cls, policy: str | None, **fields) -> ModulatedBlock:
            return remat_block(block_cls, self.remat, policy=policy)(
                features=self.emb_features,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision,
                norm_epsilon=self.norm_epsilon,
                **fields,
            )

        if block_type == 'ssm':
            return build(
                ModulatedBlock, None,
                mixer='ssm',
                ssm_state_dim=self.ssm_state_dim,
                bidirectional_ssm=self.bidirectional_ssm,
                use_2d_fusion=self.use_2d_fusion,
                scan_order=self.scan_order,
                name=f"ssm_block_{index}",
            )
        return build(
            ModulatedBlock, 'dots',
            mixer='attention',
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            qk_norm=self.qk_norm,
            attention_impl=self.attention_impl,
            name=f"dit_block_{index}",
        )

    def stack(self) -> list[ModulatedBlock]:
        """Layer `i` is the block `ssm_attention_ratio` or `block_pattern` names."""
        if self.block_pattern is not None and self.ssm_attention_ratio != DEFAULT_SSM_RATIO:
            raise ValueError(
                f"block_pattern names every layer's mixer and ssm_attention_ratio "
                f"{self.ssm_attention_ratio!r} names them by ratio; set one, not both")
        pattern = build_block_pattern(
            self.num_layers, self.ssm_attention_ratio, self.block_pattern)
        return [self.block(index, block_type) for index, block_type in enumerate(pattern)]
