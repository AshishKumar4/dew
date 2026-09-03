import numpy as np
import jax.numpy as jnp
from .discrete import DiscreteNoiseScheduler

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
    return betas

class LinearNoiseScheduler(DiscreteNoiseScheduler):
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02, p2_loss_weight_k=1, p2_loss_weight_gamma=1,
                 dtype=jnp.float32, clip_min=-1.0, clip_max=1.0, min_snr_gamma=None, prediction_transform=None):
        super().__init__(timesteps, beta_start, beta_end, schedule_fn=linear_beta_schedule,
                         p2_loss_weight_k=p2_loss_weight_k, p2_loss_weight_gamma=p2_loss_weight_gamma,
                         dtype=dtype, clip_min=clip_min, clip_max=clip_max,
                         min_snr_gamma=min_snr_gamma, prediction_transform=prediction_transform)
