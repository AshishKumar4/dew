"""Reproductions for the diffusion findings. Each block prints a verdict line."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np
from test_samplers import (make_karras_sampler, make_vp_sampler, generate, DATA_STD,
                           KarrasOracle)
from dew.sampling.ddpm import DDPMSampler
from dew.sampling.multistep_dpm import MultiStepDPM
from dew.sampling.euler import EulerAncestralSampler
from dew.diffusion.schedules import (CosineNoiseScheduler, KarrasVENoiseScheduler,
                                     EDMNoiseScheduler, CosineGeneralNoiseScheduler,
                                     FlowMatchingScheduler)
from dew.diffusion.transforms import VPredictionTransform
from dew.nn.blocks import FourierEmbedding
from dew.random_state import RandomMarkovState

def verdict(tag, ok, detail):
    print(f"[{tag}] {'CONFIRMED' if ok else 'NOT REPRODUCED'}: {detail}")

# 1. DDPMSampler on a VE schedule is deterministic (gamma == 0) and off in std.
model, sampler = make_karras_sampler(DDPMSampler)
a = generate(model, sampler)  # test helper seeds via rngstate? check determinism explicitly
params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)))
def run(seed):
    return sampler.generate_samples(params=params, num_samples=64, resolution=8,
                                    diffusion_steps=100, rngstate=RandomMarkovState(jax.random.PRNGKey(seed)))
x1, x2 = run(1), run(2)
same_across_seeds = bool(jnp.allclose(x1, x2, atol=1e-6))
sched = sampler.noise_schedule
shape = (-1, 1, 1, 1)
a_t, s_t = sched.get_rates(jnp.array([0.5]), shape)
a_n, s_n = sched.get_rates(jnp.array([0.4]), shape)
gamma = jnp.sqrt((s_n**2 / s_t**2) * (1 - a_t**2 / a_n**2))
true_var = s_n**2 * (1 - a_t**2 * s_n**2 / (a_n**2 * s_t**2))
verdict("1 ddpm-ve", same_across_seeds and float(gamma.ravel()[0]) == 0.0,
        f"gamma={float(gamma.ravel()[0]):.4g} (true posterior std {float(jnp.sqrt(true_var).ravel()[0]):.4g}), "
        f"different seeds give identical samples={same_across_seeds}, sample std={float(jnp.std(x1)):.3f} vs data std {DATA_STD}")

# 7. get_posterior_variance: untraceable int() on arrays, and returns std not variance.
cs = CosineNoiseScheduler(1000)
try:
    cs.get_posterior_variance(jnp.array([10, 20]))
    vec_ok = False
except TypeError as e:
    vec_ok = True
scalar = float(cs.get_posterior_variance(10).ravel()[0])
verdict("7 posterior-variance", vec_ok and np.isclose(scalar, float(jnp.sqrt(cs.posterior_variance[10]))),
        f"array input raises TypeError={vec_ok}; scalar returns {scalar:.4g}, sqrt(var)={float(jnp.sqrt(cs.posterior_variance[10])):.4g}, var={float(cs.posterior_variance[10]):.4g}")

# 8. MultiStepDPM on a VP schedule integrates the wrong ODE (no guard).
model_vp, ms = make_vp_sampler(MultiStepDPM)
xs = generate(model_vp, ms)
std = float(jnp.std(xs)); mean = float(jnp.mean(xs))
verdict("8 multistep-vp", abs(std - DATA_STD) > 0.05 or not np.isfinite(std),
        f"constructed without error on CosineNoiseScheduler; samples mean={mean:.3f} std={std:.3f} vs expected std {DATA_STD}")

# 9. Fourier time embedding: adjacent timesteps in the discrete/flow domain are near-orthogonal.
emb = FourierEmbedding(features=512)
p = emb.init(jax.random.PRNGKey(0), jnp.zeros((1,)))
def cos(a, b):
    ea, eb = emb.apply(p, jnp.array([a])), emb.apply(p, jnp.array([b]))
    return float(jnp.sum(ea * eb) / (jnp.linalg.norm(ea) * jnp.linalg.norm(eb)))
flow = FlowMatchingScheduler()
_, tf0 = flow.transform_inputs(None, jnp.array([0.500]))
_, tf1 = flow.transform_inputs(None, jnp.array([0.501]))
kv = KarrasVENoiseScheduler(1)
steps = jnp.linspace(1, 0, 100)
_, c0 = kv.transform_inputs(None, steps[50:51]); _, c1 = kv.transform_inputs(None, steps[51:52])
print(f"   discrete t=500 vs 501: cos={cos(500., 501.):.3f}; t=500 vs 550: cos={cos(500., 550.):.3f}")
print(f"   flow t=0.500 vs 0.501 (model time {float(tf0[0]):.1f} vs {float(tf1[0]):.1f}): cos={cos(float(tf0[0]), float(tf1[0])):.3f}")
print(f"   EDM adjacent grid sigmas (c_noise {float(c0[0]):.4f} vs {float(c1[0]):.4f}): cos={cos(float(c0[0]), float(c1[0])):.3f}")
verdict("9 fourier-aliasing", cos(500., 501.) < 0.2 and cos(float(c0[0]), float(c1[0])) > 0.9,
        "discrete/flow adjacent steps decorrelated while EDM adjacent steps stay correlated")

# 10. Default p2 weighting turns the cosine/v preset into the plain x0 loss: w(t) * (SNR+1) == 1.
cs_v = CosineNoiseScheduler(1000, beta_end=1, prediction_transform=VPredictionTransform())
t = jnp.array([10, 300, 600, 900])
w = cs_v.get_weights(t, shape=(-1,))
snr = cs_v.get_snr(t)
prod = w * (snr + 1)
verdict("10 p2-default", bool(jnp.allclose(prod, 1.0, atol=1e-4)),
        f"weight*(SNR+1) = {np.array(prod).round(4).tolist()} (v-loss * w equals x0-loss exactly)")

# 11. Validation sampling uses a fixed seed: identical samples from consecutive calls without rngstate.
model_k, ea = make_karras_sampler(EulerAncestralSampler)
params_k = model_k.init(jax.random.PRNGKey(1), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)))
y1 = ea.generate_samples(params=params_k, num_samples=8, resolution=8, diffusion_steps=20)
y2 = ea.generate_samples(params=params_k, num_samples=8, resolution=8, diffusion_steps=20)
verdict("11 fixed-seed", bool(jnp.allclose(y1, y2)), "two ancestral runs without rngstate are bit-identical")

# 15. Schedule constructors swallow unknown kwargs.
try:
    KarrasVENoiseScheduler(1, sigma_mn=0.01)
    EDMNoiseScheduler(1, P_meen=0.0)
    CosineNoiseScheduler(1000, beta_stat=0.1)
    swallowed = True
except TypeError:
    swallowed = False
verdict("15 kwargs-swallowed", swallowed, "misspelled sigma_mn / P_meen / beta_stat accepted silently")

# 22. GeneralizedNoiseScheduler.get_schedule_weights is live for CosineGeneralNoiseScheduler and is not EDM's lambda.
cg = CosineGeneralNoiseScheduler()
tt = jnp.array([0.2, 0.5, 0.8])
w_cg = cg.get_weights(tt, shape=(-1,))
sig = cg.get_sigmas(tt)
lam = 1 / cg.sigma_data**2 + 1 / sig**2
verdict("22 uncited-weight", not bool(jnp.allclose(w_cg, lam, rtol=1e-3)),
        f"CosineGeneral weights {np.array(w_cg).round(4).tolist()} vs EDM lambda {np.array(lam).round(4).tolist()}")
