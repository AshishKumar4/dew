#!/usr/bin/env python3
"""Write tests/fixtures/rl/agentic.npz: the agentic loss variants run by their references.

References, all Apache-2.0 or MIT, read from the research clones at the
commits the agentic RL memo pins (~/.cache/dew/research/src):

- verl 12ebe0cb4d300c58449fb6c675379e8700015c51,
  `verl/trainer/ppo/core_algos.py`: `compute_policy_loss_vanilla`,
  `compute_policy_loss_gspo`, `compute_policy_loss_cispo`, `agg_loss`;
  `verl/trainer/ppo/rollout_corr_helper.py`:
  `compute_rollout_correction_weights` (token TIS and the IcePop band),
  `compute_rollout_rejection_mask` (`seq_sum_k1`, `seq_mean_k1`),
  `compute_offpolicy_metrics` (`kl`, `k3_kl`) and the ESS inside
  `compute_is_metrics`.
- Agent Lightning ff9457587fb6ec900e16e93be9ad2d77409afa08,
  `agentlightning/verl/per_rollout_loss.py`: `normalize_advantages_by_rollout`
  and `compute_policy_loss_per_rollout_mean`.

Each row below is one sequence as verl sees it; Dew's tests lay the same
sequences out as packed chains, two to a row, and compare. Torch autograd
gives each loss's gradient with respect to the current log-probabilities,
which pins every stop-gradient. The reference files import their training
stacks at module scope, so they are compiled with those names stubbed, as
tools/parity_rl.py does; every number written comes from the reference's
own arithmetic. Run with an interpreter that has torch:

  ~/Desktop/dew/.venv/bin/python tools/parity_agentic.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_rl import ActorConfig, AlgoConfig, compile_reference, stub

SOURCES = Path.home() / ".cache/dew/research/src"
VERL = SOURCES / "verl"
LIGHTNING = SOURCES / "agent-lightning"
VERL_REVISION = "12ebe0cb4d300c58449fb6c675379e8700015c51"
LIGHTNING_REVISION = "ff9457587fb6ec900e16e93be9ad2d77409afa08"
OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rl" / "agentic.npz"

# Four sequences of six tokens. Row 0 sits inside every band. Row 1 has a
# token whose proximal/behavior ratio is exp(2) = 7.39, above IcePop's 5 and
# the TIS cap, and a masked one at exp(-1.2); both sequence masks reject it.
# Row 2 drifts steadily: its summed k1 (log behavior - log proximal) is -0.9
# and its mean -0.15. The sequence and geometric bands are asymmetric in log
# space, so each rejects row 2 on its negative side and would keep +0.9 and
# +0.15; a flipped k1 sign keeps row 2. Row 3 is short.
# Current-minus-old spans both clip sides and the dual clip.
OLD = np.array([
    [-1.0, -0.5, -2.0, -0.3, -1.2, -0.7],
    [-0.4, -2.5, -0.9, -1.1, -0.2, -3.0],
    [-0.8, -0.6, -1.4, -0.9, -1.0, -0.5],
    [-0.3, -1.7, -0.6, -2.2, -0.1, -0.4],
], np.float32)
BEHAVIOR = OLD - np.array([
    [0.05, -0.02, 0.01, 0.0, -0.03, 0.02],
    [2.0, 0.1, -1.2, 0.0, 0.3, -0.1],
    [0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    [-0.1, 0.4, 0.0, 0.0, 0.0, 0.0],
], np.float32)
CURRENT = OLD + np.array([
    [0.0, 0.1, 0.3, -0.3, 0.05, 0.0],
    [0.0, 0.5, 1.5, -0.5, 0.2, 0.0],
    [0.2, -0.2, 0.0, 0.4, -0.4, 0.1],
    [0.05, -0.6, 0.05, 0.05, 0.0, 0.0],
], np.float32)
MASK = np.array([
    [1, 1, 1, 1, 1, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0],
], np.float32)
ADVANTAGES = np.array([1.0, -1.0, 0.5, 1.0], np.float32)
"""One per sequence. Rows 0 and 3 share an advantage because the per-rollout
check puts them in one rollout."""
ROLLOUTS = ["A", "B", "C", "A"]
EPSILON_LOW = 0.2
EPSILON_HIGH = 0.28
DUAL_CLIP = 3.0
TIS_CAP = 2.0
BAND = (0.5, 5.0)
SEQUENCE_BAND = (0.5, 3.0)
GEOMETRIC_BAND = (0.87, 2.0)


def revision(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def load():
    """verl's core_algos and rollout_corr_helper, then Agent Lightning's per-rollout loss."""
    for path, wanted in ((VERL, VERL_REVISION), (LIGHTNING, LIGHTNING_REVISION)):
        if revision(path) != wanted:
            raise RuntimeError(f"{path} is not at {wanted}")
    stub("ray")
    stub("tensordict", TensorDict=dict)
    stub("omegaconf", DictConfig=dict, MISSING=None)
    stub("verl")
    stub("verl.utils", as_torch_index=None, group_mean_std=None)
    stub("verl.utils.device", get_device_name=lambda: "cpu", get_torch_device=lambda: None)
    stub("verl.utils.import_utils", deprecated=lambda *_: (lambda f: f))
    stub("verl.protocol", DataProto=object)
    stub("verl.trainer")
    stub("verl.trainer.config", AlgoConfig=AlgoConfig)
    stub("verl.trainer.config.algorithm", RolloutCorrectionConfig=object)
    stub("verl.trainer.ppo")
    stub("verl.workers")
    stub("verl.workers.config", ActorConfig=ActorConfig)
    stub("verl.workers.config.actor", PolicyLossConfig=object)
    functional = compile_reference("verl.utils.torch_functional", VERL / "verl/utils/torch_functional.py")
    core = compile_reference("verl.trainer.ppo.core_algos", VERL / "verl/trainer/ppo/core_algos.py")
    helper = compile_reference("verl.trainer.ppo.rollout_corr_helper",
                               VERL / "verl/trainer/ppo/rollout_corr_helper.py")
    stub("agentlightning")
    stub("agentlightning.verl")
    lightning = compile_reference("agentlightning.verl.per_rollout_loss",
                                  LIGHTNING / "agentlightning/verl/per_rollout_loss.py")
    return functional, core, helper, lightning


def main() -> None:
    import torch

    torch.set_num_threads(1)
    _, core, helper, lightning = load()
    old, behavior, mask = (torch.tensor(value) for value in (OLD, BEHAVIOR, MASK))
    advantages = torch.tensor(ADVANTAGES)[:, None].expand_as(old).contiguous()
    config = ActorConfig(clip_ratio=EPSILON_LOW, clip_ratio_low=EPSILON_LOW,
                         clip_ratio_high=EPSILON_HIGH, clip_ratio_c=DUAL_CLIP)
    out: dict[str, np.ndarray] = {}

    def run(name, loss_fn, agg, weights=None, response_mask=mask, proximal=old):
        current = torch.tensor(CURRENT).requires_grad_()
        loss, metrics = loss_fn(proximal, current, advantages, response_mask, agg, config, weights)
        loss.backward()
        assert current.grad is not None
        out[f"{name}_loss"] = loss.detach().numpy()
        out[f"{name}_grad"] = current.grad.numpy()
        for key, value in metrics.items():
            out[f"{name}_{key.split('/')[-1]}"] = np.asarray(value, np.float32)

    for agg, suffix in (("token-mean", "token"), ("seq-mean-token-mean", "sequence")):
        run(f"ppo_{suffix}", core.compute_policy_loss_vanilla, agg)
        run(f"gspo_{suffix}", core.compute_policy_loss_gspo, agg)
        run(f"cispo_{suffix}", core.compute_policy_loss_cispo, agg)

    log_ratio = old - behavior
    tis, tis_metrics = helper.compute_rollout_correction_weights(
        log_ratio, mask, rollout_is="token", rollout_is_threshold=TIS_CAP)
    band, band_metrics = helper.compute_rollout_correction_weights(
        log_ratio, mask, rollout_is="token", rollout_is_threshold=f"{BAND[0]}_{BAND[1]}")
    out.update(tis_weights=tis.numpy(), band_weights=band.numpy(),
               tis_ess=np.float32(tis_metrics["rollout_is_eff_sample_size"]),
               band_ess=np.float32(band_metrics["rollout_is_eff_sample_size"]),
               band_oob=np.float32(band_metrics["rollout_is_oob_ratio"]))
    run("ppo_band_token", core.compute_policy_loss_vanilla, "token-mean", band)
    run("ppo_tis_token", core.compute_policy_loss_vanilla, "token-mean", tis)

    # Bypass mode (compute_policy_loss_bypass_mode, ppo_clip): behavior is the
    # old policy, no IS weight reaches the loss, and the band is token-level
    # rejection on the current policy against behavior. verl's k1 is
    # log(mu / pi), so the ratio band (low, high) is threshold "1/high_1/low".
    bypass, _ = helper.compute_rollout_rejection_mask(
        torch.tensor(CURRENT) - behavior, mask, rollout_rs="token_k1",
        rollout_rs_threshold=f"{1 / BAND[1]}_{1 / BAND[0]}")
    out["bypass_band_mask"] = bypass.numpy()
    run("ppo_bypass_band_token", core.compute_policy_loss_vanilla, "token-mean",
        response_mask=bypass, proximal=behavior)

    for option, bounds, name in (("seq_sum_k1", SEQUENCE_BAND, "sequence"),
                                 ("seq_mean_k1", GEOMETRIC_BAND, "geometric")):
        rejected, _ = helper.compute_rollout_rejection_mask(
            log_ratio, mask, rollout_rs=option, rollout_rs_threshold=f"{bounds[0]}_{bounds[1]}")
        out[f"{name}_mask"] = rejected.numpy()
        run(f"ppo_{name}_token", core.compute_policy_loss_vanilla, "token-mean", response_mask=rejected)
        run(f"ppo_{name}_sequence", core.compute_policy_loss_vanilla, "seq-mean-token-mean",
            response_mask=rejected)

    offpolicy = helper.compute_offpolicy_metrics(old, behavior, mask)
    out.update(offpolicy_kl=np.float32(offpolicy["kl"]), offpolicy_k3_kl=np.float32(offpolicy["k3_kl"]))

    scaled = lightning.normalize_advantages_by_rollout(advantages, mask, ROLLOUTS, num_trained_rows=len(ROLLOUTS))
    current = torch.tensor(CURRENT).requires_grad_()
    per_rollout, _ = core.compute_policy_loss_vanilla(old, current, scaled, mask, "token-sum", config, None)
    per_rollout.backward()
    assert current.grad is not None
    out.update(per_rollout_loss=per_rollout.detach().numpy(), per_rollout_grad=current.grad.numpy())

    np.savez(OUTPUT, allow_pickle=False, verl_revision=np.asarray(VERL_REVISION),
             lightning_revision=np.asarray(LIGHTNING_REVISION), torch_version=np.asarray(torch.__version__),
             old=OLD, behavior=BEHAVIOR, current=CURRENT, mask=MASK, advantages=ADVANTAGES,
             rollouts=np.asarray(ROLLOUTS), epsilon_low=np.float32(EPSILON_LOW),
             epsilon_high=np.float32(EPSILON_HIGH), dual_clip=np.float32(DUAL_CLIP),
             tis_cap=np.float32(TIS_CAP), band=np.asarray(BAND, np.float32),
             sequence_band=np.asarray(SEQUENCE_BAND, np.float32),
             geometric_band=np.asarray(GEOMETRIC_BAND, np.float32), **out)
    for key in sorted(out):
        if key.endswith("_loss"):
            print(f"{key}: {float(out[key]):.9g}")


if __name__ == "__main__":
    main()
