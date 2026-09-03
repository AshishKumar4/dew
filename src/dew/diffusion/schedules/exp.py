import numpy as np
import jax.numpy as jnp
from .discrete import DiscreteNoiseScheduler

def exp_beta_schedule(timesteps, start_angle=0.008, end_angle=0.999):
    ts = np.linspace(0, 1, timesteps + 1, dtype=np.float64)
    alphas_bar = np.exp(ts * -12.0)
    alphas_bar = alphas_bar/alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 0, end_angle)

class ExpNoiseScheduler(DiscreteNoiseScheduler):
    def __init__(self, timesteps, beta_start=0.008, beta_end=0.999, p2_loss_weight_k=1, p2_loss_weight_gamma=1,
                 dtype=jnp.float32, clip_min=-1.0, clip_max=1.0, min_snr_gamma=None, prediction_transform=None):
        super().__init__(timesteps, beta_start, beta_end, schedule_fn=exp_beta_schedule,
                         p2_loss_weight_k=p2_loss_weight_k, p2_loss_weight_gamma=p2_loss_weight_gamma,
                         dtype=dtype, clip_min=clip_min, clip_max=clip_max,
                         min_snr_gamma=min_snr_gamma, prediction_transform=prediction_transform)
