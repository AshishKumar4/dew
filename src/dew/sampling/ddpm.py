import jax
import jax.numpy as jnp
from .common import DiffusionSampler
from dew.diffusion.schedules import get_coeff_shapes_tuple
from dew.random_state import MarkovState, RandomMarkovState

class DDPMSampler(DiffusionSampler):
    """Exact ancestral sampler for the reverse diffusion SDE.

    One step draws from the forward posterior q(x_s | x_t, x_0) for
    x_t = alpha_t x_0 + sigma_t eps, written in signal and noise rates so it
    holds for any schedule and any step stride, not just t -> t-1. The
    posterior mean is alpha_s x_0 + alpha_t sigma_s^2 / (alpha_s sigma_t) eps
    and its variance is sigma_s^2 (1 - alpha_t^2 sigma_s^2 / (alpha_s^2 sigma_t^2)).
    """
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, sample_model_fn, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        state, rng = state.get_random_key()
        noise = jax.random.normal(rng, reconstructed_samples.shape, dtype=jnp.float32)

        shape = get_coeff_shapes_tuple(current_samples)
        current_signal_rate, current_noise_rate = self.noise_schedule.get_rates(current_step, shape)
        next_signal_rate, next_noise_rate = self.noise_schedule.get_rates(next_step, shape)

        pred_noise_coeff = ((next_noise_rate ** 2) * current_signal_rate) / (current_noise_rate * next_signal_rate)

        noise_ratio_squared = (next_noise_rate ** 2) / (current_noise_rate ** 2)
        signal_ratio_squared = (current_signal_rate ** 2) / (next_signal_rate ** 2)
        gamma = next_noise_rate * jnp.sqrt(1 - signal_ratio_squared * noise_ratio_squared)

        next_samples = next_signal_rate * reconstructed_samples + pred_noise_coeff * pred_noise + noise * gamma
        return next_samples, state
