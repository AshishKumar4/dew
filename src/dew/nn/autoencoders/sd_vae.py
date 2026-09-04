import jax
import jax.numpy as jnp
from functools import partial

from flax import linen as nn
from .api import AutoEncoder
from .vae import FlaxEncoder, FlaxDecoder, load_pretrained_vae

"""
This module contains an Autoencoder implementation which uses the Stable Diffusion VAE model from the HuggingFace model hub.
The actual model was not trained by me. I have only implemented the wrapper (and vendored the diffusers Flax modules
in vae.py, as diffusers dropped Flax support upstream) to make it compatible with our library.
All credits for the model go to the developers of Stable Diffusion VAE and for the modules to the Diffusers library.
"""

class StableDiffusionVAE(AutoEncoder):
    """A pretrained AutoencoderKL behind the `AutoEncoder` seam.

    `modelname` is a Hub repo or a local directory. The config decides the
    latent space, so the same class carries SD1's four channels and the
    sixteen of SD3.5 and Flux; those newer configs also set `use_quant_conv`
    and `use_post_quant_conv` false, and then there are no such layers to
    apply. `params` overrides the loaded weights, which is how a test mutates
    one.
    """

    def __init__(self, modelname="CompVis/stable-diffusion-v1-4", revision="bf16",
                 dtype=jnp.bfloat16, latent_shift=None, latent_scale=None, params=None):

        pretrained = load_pretrained_vae(modelname, revision=revision)
        config = pretrained["config"]
        params = pretrained["params"] if params is None else params

        self.modelname = modelname
        self.revision = revision
        self.dtype = dtype
        self.params = params

        enc = FlaxEncoder(
            in_channels=config["in_channels"],
            out_channels=config["latent_channels"],
            down_block_types=config["down_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            act_fn=config["act_fn"],
            norm_num_groups=config["norm_num_groups"],
            double_z=True,
            dtype=dtype,
        )

        dec = FlaxDecoder(
            in_channels=config["latent_channels"],
            out_channels=config["out_channels"],
            up_block_types=config["up_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            norm_num_groups=config["norm_num_groups"],
            act_fn=config["act_fn"],
            dtype=dtype,
        )

        # SD3.5 and Flux fold the quantisation convolutions away, and their
        # configs say so; SD1-era configs predate the keys and carry both.
        use_quant_conv = config.get("use_quant_conv", True)
        use_post_quant_conv = config.get("use_post_quant_conv", True)
        one_by_one = partial(nn.Conv, kernel_size=(1, 1), strides=(1, 1),
                             padding="VALID", dtype=dtype)
        quant_conv = one_by_one(2 * config["latent_channels"]) if use_quant_conv else None
        post_quant_conv = one_by_one(config["latent_channels"]) if use_post_quant_conv else None

        # The VAE's own latent normalization rides on the AutoEncoder seam, so a
        # caller can override it with per-dataset statistics without a second
        # scaling path. Older configs predate these keys; 0.0 and 0.18215 are
        # the SD defaults.
        self.latent_shift = config.get("shift_factor", 0.0) if latent_shift is None else latent_shift
        self.latent_scale = config.get("scaling_factor", 0.18215) if latent_scale is None else latent_scale
        print(f"Latent shift: {self.latent_shift}, latent scale: {self.latent_scale}")

        def encode_single_frame(params, x, rngkey=None):
            latents = enc.apply({"params": params['encoder']}, x, deterministic=True)
            if quant_conv is not None:
                latents = quant_conv.apply({"params": params['quant_conv']}, latents)
            # apply returns the output alone unless mutable collections were
            # asked for, and none were.
            assert not isinstance(latents, tuple)
            if rngkey is not None:
                mean, log_std = jnp.split(latents, 2, axis=-1)
                log_std = jnp.clip(log_std, -30, 20)
                std = jnp.exp(0.5 * log_std)
                latents = mean + std * jax.random.normal(rngkey, mean.shape, dtype=mean.dtype)
            else:
                latents, _ = jnp.split(latents, 2, axis=-1)
            return latents

        def decode_single_frame(params, z):
            if post_quant_conv is not None:
                z = post_quant_conv.apply({"params": params['post_quant_conv']}, z)
            return dec.apply({"params": params['decoder']}, z)

        self.encode_single_frame = jax.jit(encode_single_frame)
        self.decode_single_frame = jax.jit(decode_single_frame)

        # The latent geometry, from one frame through the encoder. It is the
        # weights' own shape, so it is read here and not on every call.
        dummy_input = jnp.ones((1, 128, 128, config["in_channels"]), dtype=dtype)
        dummy_latents = self.encode_single_frame(params, dummy_input)
        self.__downscale_factor__ = dummy_input.shape[1] // dummy_latents.shape[1]
        self.__latent_channels__ = dummy_latents.shape[-1]

    def encode_batch(self, params, x, key=None):
        return self.encode_single_frame(params, x, key)

    def decode_batch(self, params, z):
        return self.decode_single_frame(params, z)

    @property
    def downscale_factor(self) -> int:
        """Returns the downscale factor for the encoder."""
        return self.__downscale_factor__
    
    @property
    def latent_channels(self) -> int:
        """Returns the number of channels in the latent space."""
        return self.__latent_channels__
    
    @property
    def name(self) -> str:
        """Get the name of the autoencoder model."""
        return "stable_diffusion"
    
    def serialize(self):
        """Serialize the model to a dictionary format."""
        return {
            "modelname": self.modelname,
            "revision": self.revision,
            "dtype": str(self.dtype),
        }
    