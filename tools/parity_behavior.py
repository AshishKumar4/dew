"""Run verl's token TIS and corrected PPO loss on fixed CPU tensors.

Install verl from the exact revision below into an isolated reference env.
Dew does not import verl or torch at runtime. The VCS installation metadata
is checked before recording outputs, so a different reference cannot silently
rewrite the fixture.
"""

from importlib.metadata import distribution
import json
from pathlib import Path

import numpy as np

REVISION = "d040717b21af2e23e8e789a3e354cff2394ae2de"


def main() -> None:
    import torch
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights
    from verl.workers.config.actor import ActorConfig

    direct = distribution("verl").read_text("direct_url.json")
    if direct is None or json.loads(direct)["vcs_info"]["commit_id"] != REVISION:
        raise RuntimeError(f"install verl from git revision {REVISION}")
    torch.set_num_threads(1)
    old = torch.tensor([[-.1, -1., -2., -.3], [-30., -.1, -.5, -1.]], dtype=torch.float32)
    behavior = torch.tensor([[-2., -.4, -2.5, -.2], [-.1, -30., -.6, -.5]], dtype=torch.float32)
    current = (old + torch.tensor([[.5, -.2, .1, 0.], [.1, -.2, -.1, .2]])).requires_grad_()
    advantage = torch.tensor([[1., -1., .7, -.2], [-.5, 2., -1.3, .1]])
    mask = torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 0.]])
    cap = 2.
    weights, _ = compute_rollout_correction_weights(old - behavior, mask, rollout_is="token",
                                                    rollout_is_threshold=cap, rollout_is_batch_normalize=False)
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    corrected, _ = compute_policy_loss_vanilla(old, current, advantage, mask,
                                              "token-mean", config, weights)
    plain, _ = compute_policy_loss_vanilla(old, current, advantage, mask, "token-mean", config, None)
    corrected.backward()
    assert current.grad is not None
    output = Path(__file__).resolve().parents[1] / "tests/fixtures/rl/behavior.npz"
    np.savez(output, allow_pickle=False, revision=np.asarray(REVISION), cap=np.asarray(cap),
             old=old.numpy(), behavior=behavior.numpy(), current=current.detach().numpy(),
             advantages=advantage.numpy(), mask=mask.numpy(), weights=weights.numpy(),
             corrected=corrected.detach().numpy(), plain=plain.detach().numpy(),
             gradient=current.grad.numpy())
    print(f"{output}: corrected={corrected.item():.9g}, plain={plain.item():.9g}")


if __name__ == "__main__":
    main()
