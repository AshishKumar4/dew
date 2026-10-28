#!/usr/bin/env python3
"""Write the RL parity fixtures under tests/fixtures/rl/.

Two references, run on fixed tensors, both Apache-2.0 and both read as clones
under /tmp/design (docs/design/post-training.md, appendix B):

- verl at commit 896a9bb, `verl/trainer/ppo/core_algos.py`:
  `compute_grpo_outcome_advantage`, `compute_rloo_outcome_advantage`,
  `compute_gae_advantage_return`, `compute_policy_loss_vanilla`,
  `compute_policy_loss_gspo`, `kl_penalty_forward` and `agg_loss`, plus the
  masked reductions in `verl/utils/torch_functional.py`. Torch autograd also
  gives the gradient of each loss with respect to the policy log-probabilities,
  which is the only thing that pins a stop-gradient down.
- Tunix at commit b9f5e65, `tunix/rl/algo_core.py` and `tunix/rl/common.py`:
  the same estimators and the same k3 KL in numpy and JAX. Two independent
  implementations of one equation is a stronger claim than one.

Writes advantage.npz and surrogate.npz. tests/test_rl_advantage.py and
tests/test_rl_surrogate.py hold the tolerances and the largest differences
observed. Run it with an interpreter that has torch and jax:

  ~/Desktop/dew/.venv/bin/python tools/parity_rl.py

`recording` records the advantage estimators over check 1's seeds instead
(tools/recorded_references.py), each reference's float32 run's evidence from
verl's float64 run, with that run's sketch, each
seed's inputs drawn again by tests/test_rl_advantage.py (`advantage_inputs`):

  PYTHONPATH=src:tools python tools/parity_rl.py recording

Both references import their whole training stack at module scope (ray,
tensordict and omegaconf for verl; metrax, jaxtyping and the SFT trainer for
Tunix), so the reference files are compiled directly with those names stubbed.
A stub either satisfies the import machinery or carries a constant the
reference reads; every number written here comes from the reference's own
arithmetic.
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rl"
VERL = Path("/tmp/design/verl")
TUNIX = Path("/tmp/design/tunix")

GROUP = 4
"""Completions per prompt. The rollout lays a group out in adjacent rows, so
verl's `uid` column is `[0, 0, 0, 0, 1, 1, 1, 1, ...]` and Dew's reshape by
`GROUP` sees the same groups."""

REWARDS = np.array([
    # A group one completion solved, the shape a verifiable reward has.
    1.0, 0.0, 0.0, 0.0,
    # A group with nothing between its members: the eps under the standard
    # deviation is all that stands between this and a division by zero.
    0.5, 0.5, 0.5, 0.5,
    # Mixed signs and a fractional reward, so centring cannot hide in a sign.
    2.0, -1.0, 0.25, 1.0,
    # A group whose mean is already zero.
    -1.5, 1.5, -0.5, 0.5,
], dtype=np.float32)
UIDS = np.repeat(np.arange(REWARDS.shape[0] // GROUP), GROUP).astype(np.int64)

TOKEN_REWARDS = np.array([
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -0.5],
    [2.0, 0.0, 0.0, 0.0, 0.0],
], dtype=np.float32)
VALUES = np.array([
    # The 7.0 and -7.0 sit behind the mask: a recursion that reads them moves
    # every advantage in the row.
    [0.1, -0.2, 0.3, 7.0, -7.0],
    [0.5, 0.4, -0.1, 0.2, 0.25],
    [1.0, 3.0, 3.0, 3.0, 3.0],
], dtype=np.float32)
RESPONSE_MASK = np.array([
    [1.0, 1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
], dtype=np.float32)
GAMMA = 0.99
LAM = 0.95

OLD_LOG_PROBS = np.full((4, 6), -1.0, dtype=np.float32)
LOG_RATIO = np.array([
    # Inside the band, above the high clip, below the low clip, and equal.
    [0.0, 0.1, 0.3, -0.3, 0.05, 0.0],
    # exp(1.5) = 4.48 against a negative advantage: the dual clip at 3.0 fires.
    [0.0, 0.5, 1.5, -0.5, 0.2, 0.0],
    [0.2, -0.2, 0.0, 0.4, -0.4, 0.1],
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
], dtype=np.float32)
LOG_PROBS = OLD_LOG_PROBS + LOG_RATIO
KL_DIFF = np.array([0.0, 0.2, -0.2, 1.0, -1.0, 0.5], dtype=np.float32)
REF_LOG_PROBS = LOG_PROBS + KL_DIFF
SURROGATE_MASK = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    # A row nobody generated, which every reduction has to survive.
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
], dtype=np.float32)
# One advantage per completion, including a zero and a negative one.
ADVANTAGES = np.array([1.0, -1.0, 0.0, 0.5], dtype=np.float32)
EPSILON_LOW = 0.2
EPSILON_HIGH = 0.28
"""Asymmetric on purpose, DAPO's clip-higher. A surrogate that reads one
epsilon for both sides disagrees with the reference here."""
DUAL_CLIP = 3.0

# The k3 estimator clamps its exponent to +-20 and its result to +-10. These
# three differences land above the input clamp, below it with room to spare,
# and just under the output clamp.
EXTREME_KL_DIFF = np.array([[30.0, -5.0, 2.5]], dtype=np.float32)
EXTREME_LOG_PROBS = np.full((1, 3), -1.0, dtype=np.float32)
EXTREME_REF_LOG_PROBS = EXTREME_LOG_PROBS + EXTREME_KL_DIFF


ESTIMATORS = ("masked_mean_rows", "masked_whiten", "group_advantage", "group_advantage_unnormalised",
              "rloo_advantage", "gae_advantage", "gae_return")
"""The advantage recording's leaves, one per estimator output."""


DRAWN_MASK = np.tile(RESPONSE_MASK, (16, 1))
"""The recorded draws' response mask: the fixed fixture's three rows (one row cut short, one whole,
one a single token) sixteen times over."""
DRAWN_GROUPS = 64
"""The recorded draws' groups of GROUP rewards."""
WHITENED = 32
"""The recorded draws' tensors whitened apart: a whitening shares one mean and one deviation over its
tensor, one rounding a draw would otherwise give."""


def advantage_inputs(seed: int) -> dict[str, np.ndarray]:
    """A seed's inputs to every estimator, the same on every run: DRAWN_GROUPS groups of GROUP rewards,
    the second group's equal (the case only the eps under the deviation divides), and token rewards and
    values under DRAWN_MASK, the values large behind it (which a recursion that reads them moves).
    Check 1's draw for the recorded references, each draw many rows, so a few seeds resolve it."""
    rng = np.random.default_rng([0xADB, seed])
    rewards = rng.normal(size=DRAWN_GROUPS * GROUP).astype(np.float32)
    rewards[GROUP:2 * GROUP] = rewards[GROUP]

    def values(*shape):
        drawn = rng.normal(size=(*shape, *DRAWN_MASK.shape))
        return np.where(DRAWN_MASK > 0, drawn, 7 * drawn).astype(np.float32)

    return {"rewards": rewards, "token_rewards": rng.normal(size=DRAWN_MASK.shape).astype(np.float32),
            "values": values(), "whitened": values(WHITENED)}


def verl_advantages(verl, inputs: dict[str, np.ndarray], dtype) -> dict[str, np.ndarray]:
    """verl's estimators on `inputs` at the torch `dtype`."""
    import torch

    reductions = sys.modules["verl.utils.torch_functional"]

    def tensor(x):
        return torch.tensor(np.asarray(x), dtype=dtype)

    rewards = inputs["rewards"]
    uids = np.repeat(np.arange(rewards.shape[0] // GROUP), GROUP).astype(np.int64)
    token_level, ones = tensor(rewards[:, None]), tensor(np.ones((rewards.shape[0], 1)))
    values, mask = tensor(inputs["values"]), tensor(DRAWN_MASK)
    found = {"masked_mean_rows": reductions.masked_mean(values, mask, axis=-1),
             "masked_whiten": torch.stack([reductions.masked_whiten(tensor(one), mask)
                                           for one in inputs["whitened"]])}
    found["group_advantage"] = verl.compute_grpo_outcome_advantage(token_level.clone(), ones, uids)[0][:, 0]
    found["group_advantage_unnormalised"] = verl.compute_grpo_outcome_advantage(
        token_level.clone(), ones, uids, norm_adv_by_std_in_grpo=False)[0][:, 0]
    found["rloo_advantage"] = verl.compute_rloo_outcome_advantage(token_level.clone(), ones, uids)[0][:, 0]
    found["gae_advantage"], found["gae_return"] = verl.compute_gae_advantage_return(
        tensor(inputs["token_rewards"]), values, mask, GAMMA, LAM)
    return {name: found[name].detach().numpy() for name in ESTIMATORS}


def tunix_advantages(tunix, inputs: dict[str, np.ndarray], dtype) -> dict[str, np.ndarray]:
    """Tunix's estimators on `inputs` at the JAX `dtype`."""
    import jax.numpy as jnp

    rewards, values, mask = (jnp.asarray(inputs.get(name, DRAWN_MASK), dtype)
                             for name in ("rewards", "values", "response_mask"))
    found = {"masked_mean_rows": tunix.masked_mean(values, mask, axis=-1),
             "masked_whiten": jnp.stack([tunix.masked_whiten(jnp.asarray(one, dtype), mask)
                                         for one in inputs["whitened"]]),
             "group_advantage": tunix.compute_advantages(rewards, GROUP),
             "group_advantage_unnormalised": tunix.compute_drgrpo_advantages(rewards, GROUP),
             "rloo_advantage": tunix.compute_rloo_advantages(rewards, GROUP)}
    found["gae_advantage"], found["gae_return"] = tunix.compute_gae_advantages(
        jnp.asarray(inputs["token_rewards"], dtype), values, mask, GAMMA, LAM)
    return {name: np.asarray(found[name]) for name in ESTIMATORS}


def dew_estimators(inputs: dict) -> dict:
    """Dew's estimators on a draw (`advantage_inputs`), by the recording's leaf names: the count the
    recording plans on, and the test's arm and, in float64, its truth."""
    import jax
    import jax.numpy as jnp

    from dew.rl.advantage import gae, group_advantage, masked_mean, masked_whiten, rloo_advantage

    mask = jnp.asarray(DRAWN_MASK, inputs["values"].dtype)
    advantages, returns = gae(inputs["token_rewards"], inputs["values"], mask, gamma=GAMMA, lam=LAM)
    return {"masked_mean_rows": masked_mean(inputs["values"], mask, axis=-1),
            "masked_whiten": jax.vmap(lambda one: masked_whiten(one, mask))(inputs["whitened"]),
            "group_advantage": group_advantage(inputs["rewards"], GROUP),
            "group_advantage_unnormalised": group_advantage(inputs["rewards"], GROUP, normalise_by_std=False),
            "rloo_advantage": rloo_advantage(inputs["rewards"], GROUP),
            "gae_advantage": advantages, "gae_return": returns}


def write_advantage_recording(verl, tunix, command: str, out: Path) -> None:
    """recorded_advantage_<reference>.npz: each reference's estimators recorded over check 1's seeds
    (`record_case`'s summary), its float32 run's evidence from verl's float64 run, which keeps float64
    through every estimator, and that run's sketch, the count planned on Dew's own estimators
    (`dew_estimators`)."""
    import jax
    import jax.numpy as jnp
    import torch
    from fetched_fixtures import save
    from recorded_references import provenance, record_case

    def truth(inputs):
        return verl_advantages(verl, inputs, torch.float64)

    def tunix_float32(inputs):
        # The recording runs where x64 is enabled, where a constant Tunix makes without a dtype (its
        # scan's carry) would be float64; its float32 run is the one without x64.
        with jax.enable_x64(new_val=False):
            return tunix_advantages(tunix, inputs, jnp.float32)

    evaluations = {"verl": lambda inputs: (verl_advantages(verl, inputs, torch.float32), truth(inputs)),
                   "tunix": lambda inputs: (tunix_float32(inputs), truth(inputs))}
    for name, evaluate in evaluations.items():
        with jax.enable_x64(new_val=True):
            recorded = record_case(name, advantage_inputs, evaluate, dew_estimators,
                                   truth_proxy=dew_estimators,
                                   store_inputs=False, summary=True)
        commit = {"verl": "896a9bb", "tunix": "b9f5e65"}[name]
        recorded[f"{name}.provenance"] = provenance(name, commit, "cpu", command, torch=torch.__version__,
                                                    truth="verl float64")
        written = save(out / f"recorded_advantage_{name}.npz", recorded, reference=f"{name} {commit}",
                       license="Apache-2.0")
        print(f"{written}: {written.stat().st_size} bytes", flush=True)


def stub(name, **attributes):
    """Register a module that exists only to satisfy an import a reference
    performs at load time."""
    module = types.ModuleType(name)
    module.__path__ = []
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    sys.modules[name] = module
    parent, _, leaf = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], leaf, module)
    return module


class AlgoConfig:
    """Stands in for verl's `AlgoConfig`, whose real definition needs omegaconf
    and the worker stack. The policy losses only assert a config is not one."""


class ActorConfig:
    """Stands in for verl's `ActorConfig`, which reaches the engine and model
    configs. The policy losses read the clip constants and `global_batch_info`,
    and `compute_policy_loss_gspo` asserts the class."""

    def __init__(self, clip_ratio, clip_ratio_low, clip_ratio_high, clip_ratio_c):
        self.clip_ratio = clip_ratio
        self.clip_ratio_low = clip_ratio_low
        self.clip_ratio_high = clip_ratio_high
        self.clip_ratio_c = clip_ratio_c
        # Empty is verl's single-process case: token-mean then divides by the
        # microbatch's own token count.
        self.global_batch_info = {}

    def get(self, name, default=None):
        return getattr(self, name, default)


class Registry:
    """Stands in for Tunix's function registry. `algo_core` decorates its
    estimators with it at import and never reads them back through it here."""

    def register(self, *_):
        return lambda function: function


def registering(*_):
    """Tunix's `register_policy_loss_fn` and `register_advantage_estimator`."""
    return lambda function: function


def load_verl(verl=VERL):
    """verl's core_algos and the masked reductions it calls, from the clone at `verl`."""
    from reference_source import loaded

    stub("ray")
    stub("tensordict", TensorDict=dict)
    stub("omegaconf", DictConfig=dict, MISSING=None)
    stub("verl")
    stub("verl.utils", as_torch_index=None, group_mean_std=None)
    stub("verl.utils.device", get_device_name=lambda: "cpu",
         get_torch_device=lambda: None)
    stub("verl.utils.import_utils", deprecated=lambda *_: (lambda f: f))
    stub("verl.trainer")
    stub("verl.trainer.config", AlgoConfig=AlgoConfig)
    stub("verl.trainer.ppo")
    stub("verl.workers")
    stub("verl.workers.config", ActorConfig=ActorConfig)

    loaded("verl.utils.torch_functional", verl / "verl" / "utils" / "torch_functional.py")
    return loaded("verl.trainer.ppo.core_algos", verl / "verl" / "trainer" / "ppo" / "core_algos.py")


def load_tunix():
    """Tunix's algo_core and common, compiled from the clone.

    Importing `tunix` runs a package `__init__` that reaches metrax, jaxtyping
    and the SFT trainer, none of which the estimators touch, so the four files
    they do need are compiled directly. Only the function registry is stubbed,
    because the decorators run at import and nothing here calls a function
    back through them.
    """
    from reference_source import loaded

    stub("tunix")
    stub("tunix.rl")
    stub("tunix.sft")
    stub("tunix.oss")
    stub("tunix.rl.function_registry", default_registry=Registry(),
         register_policy_loss_fn=registering, register_value_loss_fn=registering,
         register_advantage_estimator=registering)
    loaded("tunix.oss.utils", TUNIX / "tunix" / "oss" / "utils.py")
    loaded("tunix.sft.utils", TUNIX / "tunix" / "sft" / "utils.py")
    common = loaded("tunix.rl.common", TUNIX / "tunix" / "rl" / "common.py")
    algo_core = loaded("tunix.rl.algo_core", TUNIX / "tunix" / "rl" / "algo_core.py")
    return algo_core, common


def write_advantage(verl, tunix):
    """Advantages and returns from both references, on the same rewards."""
    import jax.numpy as jnp
    import torch

    def tensor(x):
        return torch.tensor(np.asarray(x), dtype=torch.float32)

    written = {
        "group": np.asarray(GROUP),
        "rewards": REWARDS,
        "uids": UIDS,
        "token_rewards": TOKEN_REWARDS,
        "values": VALUES,
        "response_mask": RESPONSE_MASK,
        "gamma": np.asarray(GAMMA, np.float32),
        "lam": np.asarray(LAM, np.float32),
    }

    # verl reads one scalar per completion as the row sum of a token-level
    # reward matrix, so a single column of rewards is the same input.
    token_level = tensor(REWARDS[:, None])
    ones = tensor(np.ones((REWARDS.shape[0], 1), np.float32))
    advantages, _ = verl.compute_grpo_outcome_advantage(token_level.clone(), ones, UIDS)
    written["verl_group_advantage"] = advantages[:, 0].numpy()
    advantages, _ = verl.compute_grpo_outcome_advantage(
        token_level.clone(), ones, UIDS, norm_adv_by_std_in_grpo=False)
    written["verl_group_advantage_unnormalised"] = advantages[:, 0].numpy()
    advantages, _ = verl.compute_rloo_outcome_advantage(token_level.clone(), ones, UIDS)
    written["verl_rloo_advantage"] = advantages[:, 0].numpy()

    advantages, returns = verl.compute_gae_advantage_return(
        tensor(TOKEN_REWARDS), tensor(VALUES), tensor(RESPONSE_MASK), GAMMA, LAM)
    written["verl_gae_advantage"] = advantages.numpy()
    written["verl_gae_return"] = returns.numpy()
    reductions = sys.modules["verl.utils.torch_functional"]
    written["verl_masked_whiten"] = reductions.masked_whiten(
        tensor(VALUES), tensor(RESPONSE_MASK)).numpy()
    written["verl_masked_mean_rows"] = reductions.masked_mean(
        tensor(VALUES), tensor(RESPONSE_MASK), axis=-1).numpy()

    written["tunix_group_advantage"] = np.asarray(tunix.compute_advantages(REWARDS, GROUP))
    written["tunix_group_advantage_unnormalised"] = np.asarray(
        tunix.compute_drgrpo_advantages(REWARDS, GROUP))
    written["tunix_rloo_advantage"] = np.asarray(
        tunix.compute_rloo_advantages(REWARDS, GROUP))
    advantages, returns = tunix.compute_gae_advantages(
        jnp.asarray(TOKEN_REWARDS), jnp.asarray(VALUES),
        jnp.asarray(RESPONSE_MASK), GAMMA, LAM)
    written["tunix_gae_advantage"] = np.asarray(advantages)
    written["tunix_gae_return"] = np.asarray(returns)
    written["tunix_masked_whiten"] = np.asarray(
        tunix.masked_whiten(jnp.asarray(VALUES), jnp.asarray(RESPONSE_MASK)))
    written["tunix_masked_mean_rows"] = np.asarray(
        tunix.masked_mean(jnp.asarray(VALUES), jnp.asarray(RESPONSE_MASK), axis=-1))
    return written


def write_surrogate(verl, tunix_common):
    """Losses, metrics and gradients from verl, and the k3 KL from both."""
    import jax.numpy as jnp
    import torch

    torch_functional = sys.modules["verl.utils.torch_functional"]

    def leaf(x):
        return torch.tensor(np.asarray(x), dtype=torch.float32, requires_grad=True)

    def tensor(x):
        return torch.tensor(np.asarray(x), dtype=torch.float32)

    def gradient(x: torch.Tensor) -> np.ndarray:
        """The gradient a backward pass left on a leaf, which is the point
        of running the reference under autograd."""
        if x.grad is None:
            raise RuntimeError("the reference loss did not reach the log-probabilities")
        return x.grad.numpy()

    config = ActorConfig(EPSILON_LOW, EPSILON_LOW, EPSILON_HIGH, DUAL_CLIP)
    mask = tensor(SURROGATE_MASK)
    # verl broadcasts the per-completion advantage over the sequence itself.
    advantages = tensor(np.repeat(ADVANTAGES[:, None], SURROGATE_MASK.shape[1], axis=1))

    written = {
        "old_log_probs": OLD_LOG_PROBS,
        "log_probs": LOG_PROBS,
        "ref_log_probs": REF_LOG_PROBS,
        "response_mask": SURROGATE_MASK,
        "advantages": ADVANTAGES,
        "epsilon_low": np.asarray(EPSILON_LOW, np.float32),
        "epsilon_high": np.asarray(EPSILON_HIGH, np.float32),
        "dual_clip": np.asarray(DUAL_CLIP, np.float32),
        "extreme_log_probs": EXTREME_LOG_PROBS,
        "extreme_ref_log_probs": EXTREME_REF_LOG_PROBS,
    }

    log_probs = leaf(LOG_PROBS)
    loss, metrics = verl.compute_policy_loss_vanilla(
        tensor(OLD_LOG_PROBS), log_probs, advantages, mask,
        loss_agg_mode="token-mean", config=config)
    loss.backward()
    written["verl_clipped_loss"] = loss.detach().numpy()
    written["verl_clipped_grad"] = gradient(log_probs)
    written["verl_clipped_pg_clipfrac"] = np.asarray(metrics["actor/pg_clipfrac"], np.float32)
    written["verl_clipped_pg_clipfrac_lower"] = np.asarray(
        metrics["actor/pg_clipfrac_lower"], np.float32)
    written["verl_clipped_ppo_kl"] = np.asarray(metrics["actor/ppo_kl"], np.float32)

    log_probs = leaf(LOG_PROBS)
    loss, metrics = verl.compute_policy_loss_gspo(
        tensor(OLD_LOG_PROBS), log_probs, advantages, mask,
        loss_agg_mode="token-mean", config=config)
    loss.backward()
    written["verl_gspo_loss"] = loss.detach().numpy()
    written["verl_gspo_grad"] = gradient(log_probs)
    written["verl_gspo_pg_clipfrac"] = np.asarray(metrics["actor/pg_clipfrac"], np.float32)

    log_probs = leaf(LOG_PROBS)
    kl = verl.kl_penalty_forward(log_probs, tensor(REF_LOG_PROBS), "k3")
    penalty = verl.agg_loss(loss_mat=kl, loss_mask=mask, loss_agg_mode="token-mean")
    penalty.backward()
    written["verl_k3_kl"] = kl.detach().numpy()
    written["verl_k3_penalty"] = penalty.detach().numpy()
    written["verl_k3_grad"] = gradient(log_probs)
    written["verl_k3_kl_extreme"] = verl.kl_penalty_forward(
        tensor(EXTREME_LOG_PROBS), tensor(EXTREME_REF_LOG_PROBS), "k3").detach().numpy()

    written["verl_token_mean_log_probs"] = verl.agg_loss(
        loss_mat=tensor(LOG_PROBS), loss_mask=mask,
        loss_agg_mode="token-mean").detach().numpy()
    written["verl_masked_mean_log_probs"] = torch_functional.masked_mean(
        tensor(LOG_PROBS), mask).numpy()

    written["tunix_k3_kl"] = np.asarray(tunix_common.compute_kl_divergence(
        jnp.asarray(LOG_PROBS), jnp.asarray(REF_LOG_PROBS), "low_var_kl"))
    return written


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("recording",))
    parser.add_argument("--out", type=Path, default=FIXTURES)
    parsed = parser.parse_args(argv)
    out = parsed.out
    verl = load_verl()
    tunix, tunix_common = load_tunix()
    if parsed.mode == "recording":
        write_advantage_recording(verl, tunix, "PYTHONPATH=src:tools python tools/parity_rl.py recording",
                                  out)
        return

    out.mkdir(parents=True, exist_ok=True)
    for name, written in (("advantage", write_advantage(verl, tunix)),
                          ("surrogate", write_surrogate(verl, tunix_common))):
        path = out / f"{name}.npz"
        np.savez(path, **written)
        print(f"{path}: {path.stat().st_size / 1e3:.1f} kB")
        for key, array in sorted(written.items()):
            print(f"  {key}: {np.asarray(array).shape} {np.asarray(array).dtype}")


if __name__ == "__main__":
    main()
