"""Record PPO actor/critic losses, gradients and GAE using installed pinned verl."""

from importlib.metadata import distribution
import json
from pathlib import Path

import numpy as np

REVISION = "d040717b21af2e23e8e789a3e354cff2394ae2de"


def main() -> None:
    import torch
    from verl.trainer.ppo.core_algos import (
        compute_gae_advantage_return, compute_policy_loss_vanilla, compute_value_loss,
    )
    from verl.workers.config.actor import ActorConfig

    direct = distribution("verl").read_text("direct_url.json")
    if direct is None or json.loads(direct)["vcs_info"]["commit_id"] != REVISION:
        raise RuntimeError(f"install verl from git revision {REVISION}")
    torch.set_num_threads(1)
    mask = torch.tensor([[1., 1., 0., 1., 1.], [1., 1., 1., 0., 0.]])
    rewards = torch.tensor([[0., 0., 0., 0., 1.], [0., 0., -.5, 0., 0.]])
    old_values = torch.tensor([[.2, -.1, .4, .5, -.3], [.1, .4, -.2, .2, .1]])
    predicted = (old_values + torch.tensor([[.6, -.1, 1., -.7, .4], [-.6, .1, .2, -.8, .3]])).requires_grad_()
    old = torch.tensor([[-.2, -.9, -1., -.5, -.3], [-.8, -.6, -.4, -.9, -.2]])
    current = (old + torch.tensor([[.5, -.6, 0., .2, -.1], [-.4, .5, .1, .2, .3]])).requires_grad_()
    gamma, lam, clip, coefficient = .97, .9, .2, .5
    advantages, returns = compute_gae_advantage_return(rewards, old_values, mask,
                                                       torch.tensor(gamma), torch.tensor(lam))
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    actor, _ = compute_policy_loss_vanilla(old, current, advantages, mask, "token-mean", config, None)
    critic, _ = compute_value_loss(predicted, returns, old_values, mask, clip)
    unclipped, _ = compute_value_loss(predicted, returns, old_values, mask, 100.)
    loss = actor + coefficient * critic
    loss.backward()
    assert current.grad is not None and predicted.grad is not None
    from importlib import import_module
    import jax

    native = import_module("test_ppo")
    trainer, rollout = native.build_ppo()
    state = trainer.initial_state()
    episodes = rollout.episodes.collect(state, {"task_id": np.array([31], np.int32)}, jax.random.key(23))
    batch = rollout.episodes.tensors(episodes)
    values = np.asarray(rollout.objective.values(state.params, batch))
    token_mask = batch["response_mask"].reshape(len(episodes), -1)
    episode_rewards = np.zeros_like(token_mask)
    for row, episode in enumerate(episodes):
        episode_rewards[row, np.flatnonzero(token_mask[row])[-1]] = episode.reward
    episode_adv, episode_returns = compute_gae_advantage_return(
        torch.tensor(episode_rewards), torch.tensor(values.reshape(token_mask.shape)), torch.tensor(token_mask),
        torch.tensor(gamma), torch.tensor(lam))
    # The small native models have a bigram policy and a diagonal feature
    # scale followed by Dense(1). Evaluate those same weights with torch,
    # then let verl compute every loss term and its autograd pullback.
    policy_table = torch.tensor(np.asarray(state.params["params"]["policy"]["table"]))
    policy_table = (policy_table + .4 * torch.sin(torch.arange(policy_table.numel()).reshape(policy_table.shape))).requires_grad_()
    weights = state.params["params"]["critic"]
    kernel = torch.tensor(np.asarray(weights["value"]["kernel"]))
    kernel = (1.2 * kernel + torch.linspace(.5, -.4, kernel.shape[0])[:, None]).requires_grad_()
    scale = torch.tensor(np.asarray(weights["backbone"]["scale"]), requires_grad=True)
    bias = torch.tensor(np.asarray(weights["value"]["bias"]), requires_grad=True)
    previous = torch.tensor(batch["input_ids"][:, native.PROMPT - 1:native.PROMPT + native.RESPONSE - 1], dtype=torch.long)
    actions = torch.tensor(batch["input_ids"][:, native.PROMPT:], dtype=torch.long)
    policy_probs = torch.log_softmax(policy_table, dim=-1)[previous, actions]
    ref_table = torch.tensor(np.asarray(state.params["params"]["policy"]["table"]))
    ref_probs = torch.log_softmax(ref_table, dim=-1)[previous, actions]
    critic_values = (scale * kernel[:, 0])[previous] + bias[0]
    response_mask = torch.tensor(batch["response_mask"])
    pg, _ = compute_policy_loss_vanilla(torch.tensor(batch["old_log_probs"]), policy_probs,
        episode_adv.reshape(values.shape), response_mask, "token-mean", config, None)
    vf, _ = compute_value_loss(critic_values, episode_returns.reshape(values.shape), torch.tensor(values), response_mask, clip)
    core = import_module("verl.trainer.ppo.core_algos")
    kl = (core.kl_penalty_forward(policy_probs, ref_probs, "k3") * response_mask).sum() / response_mask.sum()
    objective_loss = pg + coefficient * vf + .03 * kl
    objective_loss.backward()
    assert all(value.grad is not None for value in (policy_table, kernel, scale, bias))
    objective_fixture = {"objective_loss": objective_loss.detach().numpy()}
    for name, value in (("policy", policy_table), ("kernel", kernel), ("scale", scale), ("bias", bias)):
        assert value.grad is not None
        objective_fixture[f"objective_{name}"] = value.detach().numpy()
        objective_fixture[f"objective_{name}_gradient"] = value.grad.numpy()
    output = Path(__file__).resolve().parents[1] / "tests/fixtures/rl/ppo.npz"
    np.savez(output, revision=REVISION, mask=mask.numpy(), rewards=rewards.numpy(),
             old_values=old_values.numpy(), predicted=predicted.detach().numpy(), old=old.numpy(),
             current=current.detach().numpy(), gamma=gamma, lam=lam, clip=clip, coefficient=coefficient,
             advantages=advantages.numpy(), returns=returns.numpy(), actor=actor.detach().numpy(),
             critic=critic.detach().numpy(), unclipped=unclipped.detach().numpy(), loss=loss.detach().numpy(),
             actor_gradient=current.grad.numpy(), critic_gradient=predicted.grad.numpy(),
             episode_old_values=values, episode_advantages=episode_adv.numpy().reshape(values.shape),
             episode_returns=episode_returns.numpy().reshape(values.shape), **objective_fixture)
    print(f"verl PPO: actor={actor.item()}, critic={critic.item()}, combined={loss.item()}")


if __name__ == "__main__":
    main()
