import jax
import jax.numpy as jnp
from typing import Union
from dew.random_state import RandomMarkovState  
from .common import NoiseScheduler, reshape_rates

class DiscreteNoiseScheduler(NoiseScheduler):
    """
    Variance Preserving Noise Scheduler
    signal_rate**2 + noise_rate**2 = 1
    """
    def __init__(self, timesteps,
                    beta_start=0.0001,
                    beta_end=0.02,
                    schedule_fn=None, 
                    p2_loss_weight_k:float=1,
                    p2_loss_weight_gamma:float=1,
                    dtype=jnp.float32,
                    clip_min=-1.0,
                    clip_max=1.0,
                    min_snr_gamma=None,
                    prediction_transform=None):
        super().__init__(timesteps, dtype=dtype, clip_min=clip_min, clip_max=clip_max,
                         min_snr_gamma=min_snr_gamma, prediction_transform=prediction_transform)
        betas = schedule_fn(timesteps, beta_start, beta_end)
        alphas = 1 - betas
        alpha_cumprod = jnp.cumprod(alphas, axis=0)
        alpha_cumprod_prev = jnp.append(1.0, alpha_cumprod[:-1])
        
        self.betas = jnp.array(betas, dtype=jnp.float32)
        self.alphas = alphas.astype(jnp.float32)
        self.alpha_cumprod = alpha_cumprod.astype(jnp.float32)
        self.alpha_cumprod_prev = alpha_cumprod_prev.astype(jnp.float32)

        self.sqrt_alpha_cumprod = jnp.sqrt(alpha_cumprod).astype(jnp.float32)
        self.sqrt_one_minus_alpha_cumprod = jnp.sqrt(1 - alpha_cumprod).astype(jnp.float32)

        self.p2_loss_weights = self.get_p2_weights(p2_loss_weight_k, p2_loss_weight_gamma)
    
    def generate_timesteps(self, batch_size, state:RandomMarkovState) -> tuple[jnp.ndarray, RandomMarkovState]:
        state, rng = state.get_random_key()
        timesteps = jax.random.randint(rng, (batch_size,), 0, self.max_timesteps)
        return timesteps, state
    
    def get_p2_weights(self, k, gamma):
        return (k + self.alpha_cumprod / (1 - self.alpha_cumprod)) ** -gamma
    
    def get_schedule_weights(self, steps, shape=(-1, 1, 1, 1)):
        steps = jnp.int16(steps)
        return self.p2_loss_weights[steps].reshape(shape)

    def get_rates(self, steps, shape=(-1, 1, 1, 1)) -> tuple[jnp.ndarray, jnp.ndarray]:
        steps = jnp.int16(steps)
        signal_rates = self.sqrt_alpha_cumprod[steps]
        noise_rates = self.sqrt_one_minus_alpha_cumprod[steps]
        return reshape_rates((signal_rates, noise_rates), shape=shape)