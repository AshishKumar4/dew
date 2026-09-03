"""Reproductions for findings 3 (PSNR/SSIM), 17 (learn_sigma), 19 (CLIP constants), 24 (mean pooling)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np

def verdict(tag, ok, detail):
    print(f"[{tag}] {'CONFIRMED' if ok else 'NOT REPRODUCED'}: {detail}")

# 3. PSNR/SSIM compare [-1, 1] samples with a 0..255 batch image.
from dew.eval.psnr import get_psnr_metric
from dew.eval.ssim import get_ssim_metric
img = (np.random.RandomState(0).rand(4, 32, 32, 3) * 255).astype(np.uint8)
generated = jnp.asarray(img, jnp.float32) / 127.5 - 1.0   # exactly the same image, library convention
psnr_val = float(get_psnr_metric().function(generated, {"image": img}))
ssim_val = float(get_ssim_metric().function(generated, {"image": img}))
verdict("3 psnr-ssim-range", psnr_val < 0 and ssim_val < 0.5,
        f"identical image scores PSNR={psnr_val:.1f} dB (should be +inf), SSIM={ssim_val:.3f} (should be 1.0)")

# 17. learn_sigma=True: output has no variance channels and the sigma half of the head gets zero gradient.
from dew.nn.backbones.dit import SimpleDiT
dit = SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1, learn_sigma=True)
x = jnp.ones((2, 8, 8, 3)); temb = jnp.ones((2,))
params = dit.init(jax.random.PRNGKey(0), x, temb)
out = dit.apply(params, x, temb)
kernel = params["params"]["output"]["final_proj"]["kernel"]
def loss(p):
    return jnp.sum(dit.apply(p, x, temb) ** 2)
# zero-init head gives zero grad everywhere; perturb the kernel so the live half has gradient
params_live = jax.tree.map(lambda a: a, params)
params_live["params"]["output"]["final_proj"]["kernel"] = kernel + 0.1
g = jax.grad(loss)(params_live)["params"]["output"]["final_proj"]["kernel"]
half = g.shape[1] // 2
verdict("17 learn-sigma", out.shape[-1] == 3 and float(jnp.abs(g[:, half:]).max()) == 0.0 and float(jnp.abs(g[:, :half]).max()) > 0,
        f"output channels={out.shape[-1]} (head width {kernel.shape[1]}), grad max live half={float(jnp.abs(g[:, :half]).max()):.3g}, sigma half={float(jnp.abs(g[:, half:]).max()):.3g}")

# 19. CLIP weights enter a jitted function as closed-over constants.
from dew.nn.text_encoders import CLIPTextModel
clip = CLIPTextModel.from_pretrained("tests/fixtures/clip/tiny")
ids = jnp.zeros((2, 8), jnp.int32)
def f(ids):
    return clip(ids).last_hidden_state
jaxpr = jax.make_jaxpr(f)(ids)
n_weights = sum(int(np.prod(l.shape)) for l in jax.tree.leaves(clip.variables))
n_consts = sum(int(np.prod(np.shape(c))) for c in jaxpr.consts)
verdict("19 clip-constants", n_consts >= n_weights,
        f"jaxpr of a function calling the encoder carries {n_consts} constant elements; the encoder holds {n_weights} weight elements")

# 24. Text conditioning mean-pools padding positions.
from dew.nn.dit import ConditioningEmbed
ce = ConditioningEmbed(emb_features=16)
text = jnp.zeros((1, 77, 8)).at[:, :7].set(1.0)   # 7 real tokens, 70 padded rows
p = ce.init(jax.random.PRNGKey(0), jnp.ones((1,)), text)
full = ce.apply(p, jnp.ones((1,)), text)
masked = ce.apply(p, jnp.ones((1,)), text[:, :7])
verdict("24 mean-pool-padding", not bool(jnp.allclose(full, masked, atol=1e-5)),
        f"conditioning with 70 padded rows differs from the 7-token mean by max {float(jnp.abs(full-masked).max()):.3g}")
