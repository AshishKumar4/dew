"""Check 1's autoencoder walks, shared by their recorder (tools/reference_vjp.py `record_walk`, whose paired
sizing runs Dew's walk beside the source's) and the test that reads the recording (interop_support.py
`assert_walk_as_recorded`): each walk's draws (`walk_draw`: its operand and a cotangent probe for each output,
channels first as the sources take them) and Dew's walk of a draw (`dew_walk`) over each autoencoder's call
(`dc_ae_call`, `posterior`, `rae_call`)."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

Draw = Callable[[int], dict[str, np.ndarray]]


def walk_draw(operand: Sequence[int], outputs: Sequence[Sequence[int]], *, stream: int,
              uniform: tuple[float, float] | None = None) -> Draw:
    """A walk's draws: at a seed, its `operand` of that shape (uniform over `uniform`, an image's pixels,
    else standard normal, a latent) and a standard normal probe for each of its `outputs`' shapes, from
    the generator of the seed in `stream` (one walk's draws apart from another's)."""
    def draw(seed: int) -> dict[str, np.ndarray]:
        generator = np.random.default_rng([stream, seed])
        operand_of = (generator.standard_normal(operand) if uniform is None
                      else generator.uniform(*uniform, size=operand))
        return {"operand": operand_of.astype(np.float32),
                **{f"probe_{index}": generator.standard_normal(shape).astype(np.float32)
                   for index, shape in enumerate(outputs)}}
    return draw


def _image_walks(image, latent, *, stream: int, pixels: tuple[float, float]) -> dict[str, Draw]:
    """An autoencoder's encode of an `image` (uniform over `pixels`) to its `latent`, and its decode of a
    standard normal latent back to the image's shape."""
    return {"encode": walk_draw(image, [latent], stream=stream, uniform=pixels),
            "decode": walk_draw(latent, [image], stream=stream + 1)}


DC_AE = {
    variant: {**_image_walks((2, 3, 32, 48), (2, 8, 8, 12), stream=100 + 10 * index, pixels=(-1.0, 1.0)),
              **{f"small {walk}": draw for walk, draw in _image_walks(
                  (2, 3, 8, 16), (2, 8, 2, 4), stream=105 + 10 * index, pixels=(-1.0, 1.0)).items()}}
    for index, variant in enumerate(("conv", "shuffle", "plain"))}
"""Each tiny DC-AE's walks (tools/diffusers_dc_ae_reference.py): a rectangular batch's, and one small enough
that the deepest level's attention takes its quadratic path."""

QWEN_IMAGE_VAE = {"encode": walk_draw((2, 4, 32, 48), [(2, 8, 2, 3)] * 2, stream=200, uniform=(-1.0, 1.0)),
                  "decode": walk_draw((2, 8, 4, 5), [(2, 4, 64, 80)], stream=201)}
"""The tiny Qwen-Image VAE's walks (tools/diffusers_qwen_image_vae_reference.py): its posterior's mean and
standard deviation of an RGBA batch, and its decode of a latent of 4x5 places: a 2x3 latent's norm scales'
gradients sum few places' heavy-tailed incoming cotangents, and stayed unresolved over 512 draws (the
decoder's middle and last norms), where 4x5 resolves every leaf."""

RAE = {variant: _image_walks(image, latent, stream=300 + 10 * index, pixels=(0.0, 1.0))
       for index, (variant, image, latent) in enumerate((
           ("dinov2", (1, 3, 128, 128), (1, 128, 8, 8)), ("siglip2", (2, 3, 64, 64), (2, 64, 8, 8)),
           ("mae", (2, 3, 64, 64), (2, 96, 8, 8))))}
"""Each tiny RAE's walks (tools/diffusers_rae_reference.py): its frozen encoder's latent of an image in [0,
1], and its decoder's pixels of a latent."""


def dew_walk(call, params, layouts, *, dtype, collection: str = "autoencoder", prefix: str = "vae/"):
    """Dew's walk of a draw at `dtype`, as the source's recording keeps it: `call(params, x)`'s outputs (a
    tuple) over the draw's operand moved channels last, and against its probes the gradients of their
    probed sum in the operand and in each of `layouts`' parameters, written into its source tensor's layout
    (`source_array`, by its name less `prefix`), each array channels first again; compiled once."""
    names = [layout.name.removeprefix(prefix) for layout in layouts]

    @jax.jit
    def walk(inputs):
        operand = jnp.moveaxis(jnp.asarray(inputs["operand"], dtype), 1, -1)
        probes = tuple(jnp.moveaxis(jnp.asarray(inputs[f"probe_{index}"], dtype), 1, -1)
                       for index in range(sum(key.startswith("probe_") for key in inputs)))
        outputs, pullback = jax.vjp(lambda held, x: tuple(call(held, x)), params, operand)
        grad_params, grad_operand = pullback(probes)
        return {**{f"output_{index}": jnp.moveaxis(output, -1, 1) for index, output in enumerate(outputs)},
                "grad_operand": jnp.moveaxis(grad_operand, -1, 1),
                **{f"grad_param.{name}": layout.source_array({collection: grad_params})
                   for name, layout in zip(names, layouts, strict=True)}}
    return walk


def trained(layouts):
    """The layouts of a source's parameters: a batch norm's running statistics are buffers in the source,
    with no gradient."""
    return [layout for layout in layouts if not layout.name.endswith(("running_mean", "running_var"))]


def dc_ae_call(model, walk: str):
    """A DC-AE's call of `walk` (an encode or decode, of either size)."""
    method = walk.split()[-1]
    return lambda params, x: (model.apply({"params": params}, x, method=getattr(model, method)),)


def posterior(model, params, image):
    """A KL autoencoder's posterior mean and standard deviation, split from its encoder's moments as
    Diffusers' `DiagonalGaussianDistribution` splits them."""
    from dew.nn.autoencoders.kl import posterior_moments

    mean, log_variance = posterior_moments(model.apply({"params": params}, image,
                                                       method=lambda vae, x: vae.quant_conv(vae.encoder(x))))
    return mean, jnp.exp(0.5 * log_variance)


def qwen_image_vae_call(model, walk: str):
    """The Qwen-Image VAE's call of `walk`: its posterior's mean and deviation, or its decode."""
    if walk == "encode":
        return lambda params, x: posterior(model, params, x)
    return lambda params, x: (model.apply({"params": params}, x, method=model.decode),)


def rae_call(coder, walk: str):
    """An RAE's call of `walk`: its normalized latent of pixels in [0, 1], or its decode to [0, 1]."""
    if walk == "encode":
        return lambda params, image: (coder.encode(params, 2 * image - 1),)
    return lambda params, code: ((coder.decode(params, code) + 1) / 2,)


def wide_rae(coder):
    """An RAE computing in float64, its latent statistics the checkpoint's own."""
    from dew.nn.autoencoders.rae import RAEAutoencoder

    return RAEAutoencoder(model=coder.model.clone(dtype=jnp.float64), params=coder.params,
                          **coder.record_arguments())


def key_biases(names) -> list[str]:
    """The attention key biases among `names` (source tensor names): each shifts every logit of a softmax
    row alike, so its gradient cancels exactly, its truth zero (tools/reference_vjp.py `cancelled_truth`)."""
    return [name for name in names if name.endswith(".attention.to_k.bias")]
