import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from typing import Dict, Callable, Sequence, Any, Union, Optional
import einops
from ..blocks import kernel_init, ConvLayer, Downsample, Upsample, FourierEmbedding, TimeProjection, ResidualBlock
from ..attention import Stage, TransformerBlock
from functools import partial
from dew.registry import models

@models("unet")
class Unet(nn.Module):
    output_channels:int=3
    emb_features:int=64*4
    feature_depths:list=(64, 128, 256, 512)
    attention_configs: Sequence[Optional[Stage]] = (
        Stage(heads=8), Stage(heads=8), Stage(heads=8), Stage(heads=8))
    """Attention per resolution stage, one entry per feature depth; None is a
    stage with no attention."""
    num_res_blocks:int=2
    num_middle_res_blocks:int=1
    activation:Callable = jax.nn.swish
    norm_groups:int=8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    attention_impl: Optional[str] = None

    def setup(self):
        if self.norm_groups > 0:
            norm = partial(nn.GroupNorm, self.norm_groups)
            self.conv_out_norm = norm()
        else:
            norm = partial(nn.RMSNorm, 1e-5)
            self.conv_out_norm = norm()
        
    @nn.compact
    def __call__(self, x, temb, textcontext=None, train: bool = False):
        # print("embedding features", self.emb_features)
        temb = FourierEmbedding(features=self.emb_features)(temb)
        temb = TimeProjection(features=self.emb_features)(temb)

        # Without text the attention blocks fall back to self-attention
        # (TransformerBlock's context defaults to its input), so there is
        # nothing to unpack. Cross-attention reads the whole sequence; the
        # mask weights the pooling the DiT family does and has no place in
        # it here.
        text = None if textcontext is None else textcontext.hidden
        
        # print("time embedding", temb.shape)
        feature_depths = self.feature_depths
        attention_configs = self.attention_configs

        conv_type = up_conv_type = down_conv_type = middle_conv_type = "conv"
        # middle_conv_type = "separable"
        
        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.dtype,
            precision=self.precision
        )(x)
        downs = [x]

        # Downscaling blocks
        for i, (dim_out, attention_config) in enumerate(zip(feature_depths, attention_configs)):
            dim_in = x.shape[-1]
            # dim_in = dim_out
            for j in range(self.num_res_blocks):
                x = ResidualBlock(
                    down_conv_type,
                    name=f"down_{i}_residual_{j}",
                    features=dim_in,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision,
                )(x, temb)
                if attention_config is not None and j == self.num_res_blocks - 1:   # Apply attention only on the last block
                    x = TransformerBlock(heads=attention_config.heads, dtype=attention_config.dtype, attention_impl=self.attention_impl,
                                        dim_head=dim_in // attention_config.heads,
                                        use_projection=attention_config.use_projection,
                                        use_self_and_cross=attention_config.use_self_and_cross,
                                        precision=attention_config.precision or self.precision,
                                        only_pure_attention=attention_config.only_pure_attention,
                                        force_fp32_for_softmax=attention_config.force_fp32_for_softmax,
                                        norm_inputs=attention_config.norm_inputs,
                                        explicitly_add_residual=attention_config.explicitly_add_residual,
                                        use_linear_attention=attention_config.use_linear_attention,
                                        norm_epsilon=attention_config.norm_epsilon,
                                        name=f"down_{i}_attention_{j}")(x, text)
                # print("down residual for feature level", i, "is of shape", x.shape, "features", dim_in)
                downs.append(x)
            if i != len(feature_depths) - 1:
                # print("Downsample", i, x.shape)
                x = Downsample(
                    features=dim_out,
                    scale=2,
                    activation=self.activation,
                    name=f"down_{i}_downsample",
                    dtype=self.dtype,
                    precision=self.precision
                )(x)

        # Middle Blocks
        middle_dim_out = self.feature_depths[-1]
        middle_attention = self.attention_configs[-1]
        for j in range(self.num_middle_res_blocks):
            x = ResidualBlock(
                middle_conv_type,
                name=f"middle_res1_{j}",
                features=middle_dim_out,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision,
            )(x, temb)
            if middle_attention is not None and j == self.num_middle_res_blocks - 1:   # Apply attention only on the last block
                x = TransformerBlock(heads=middle_attention.heads, dtype=middle_attention.dtype, attention_impl=self.attention_impl,
                                    dim_head=middle_dim_out // middle_attention.heads,
                                    use_projection=middle_attention.use_projection,
                                    use_self_and_cross=False,
                                    precision=middle_attention.precision or self.precision,
                                    only_pure_attention=middle_attention.only_pure_attention,
                                    force_fp32_for_softmax=middle_attention.force_fp32_for_softmax,
                                    norm_inputs=middle_attention.norm_inputs,
                                    explicitly_add_residual=middle_attention.explicitly_add_residual,
                                    use_linear_attention=middle_attention.use_linear_attention,
                                    norm_epsilon=middle_attention.norm_epsilon,
                                    name=f"middle_attention_{j}")(x, text)
            x = ResidualBlock(
                middle_conv_type,
                name=f"middle_res2_{j}",
                features=middle_dim_out,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision,
            )(x, temb)

        # Upscaling Blocks
        for i, (dim_out, attention_config) in enumerate(zip(reversed(feature_depths), reversed(attention_configs))):
            # print("Upscaling", i, "features", dim_out)
            for j in range(self.num_res_blocks):
                x = jnp.concatenate([x, downs.pop()], axis=-1)
                # print("concat==> ", i, "concat", x.shape)
                # kernel_size = (1 + 2 * (j + 1), 1 + 2 * (j + 1))
                kernel_size = (3, 3)
                x = ResidualBlock(
                    up_conv_type,# if j == 0 else "separable",
                    name=f"up_{i}_residual_{j}",
                    features=dim_out,
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision,
                )(x, temb)
                if attention_config is not None and j == self.num_res_blocks - 1:   # Apply attention only on the last block
                    x = TransformerBlock(heads=attention_config.heads, dtype=attention_config.dtype, attention_impl=self.attention_impl, 
                                        dim_head=dim_out // attention_config.heads,
                                        use_projection=attention_config.use_projection,
                                        use_self_and_cross=attention_config.use_self_and_cross,
                                        precision=attention_config.precision or self.precision,
                                        only_pure_attention=attention_config.only_pure_attention,
                                        force_fp32_for_softmax=attention_config.force_fp32_for_softmax,
                                        norm_inputs=attention_config.norm_inputs,
                                        explicitly_add_residual=attention_config.explicitly_add_residual,
                                        use_linear_attention=attention_config.use_linear_attention,
                                        norm_epsilon=attention_config.norm_epsilon,
                                        name=f"up_{i}_attention_{j}")(x, text)
            # print("Upscaling ", i, x.shape)
            if i != len(feature_depths) - 1:
                x = Upsample(
                    features=feature_depths[-i],
                    scale=2,
                    activation=self.activation,
                    name=f"up_{i}_upsample",
                    dtype=self.dtype,
                    precision=self.precision
                )(x)

        # x = self.last_up_norm(x)
        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.dtype,
            precision=self.precision
        )(x)
    
        x = jnp.concatenate([x, downs.pop()], axis=-1)

        x = ResidualBlock(
            conv_type,
            name="final_residual",
            features=self.feature_depths[0],
            kernel_size=(3,3),
            strides=(1, 1),
            activation=self.activation,
            norm_groups=self.norm_groups,
            dtype=self.dtype,
            precision=self.precision,
        )(x, temb)

        x = self.conv_out_norm(x)
        x = self.activation(x)

        noise_out = ConvLayer(
            conv_type,
            features=self.output_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            # activation=jax.nn.mish
            # kernel_init=self.kernel_init(scale=0.0),
            dtype=self.dtype,
            precision=self.precision
        )(x)
        return noise_out#, attentions