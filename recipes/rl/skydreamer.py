"""SkyDreamer's recipe (Verraest et al., arXiv 2510.14783) on its race (`dew.rl.skydreamer`): informed
DreamerV3 learning online from scratch to fly four motor commands from a mask of the gates, the body rates,
the motor speeds and the flight plan, its world model decoding the privileged state.

    python recipes/rl/skydreamer.py --trainer.checkpoint-dir runs/skydreamer --trainer.steps 125000

At its settings (Sec III-A): DreamerV3's size12m model (an RSSM of 2048, 256 hidden, 16 classes; 16 deep
convolutions and 256 units), `train_ratio` 128, a replay of 10M steps over 16 environments, the imagined
returns bootstrapped from the slow critic (`slowtar`), and the policy's mean smoothed at 0.002 (Eq 11);
LaProp at 4e-5 after a 1000-update warmup. 125,000 updates replay 128 steps for each of 1M taken. Each
validation flies the greedy policy through a 2000-step episode in each environment, scoring its return, its
steps and the gates it passed. Not SkyDreamer's: `replay_context` (16 there; each window here starts its
latent afresh), its later phases (batch length 256 after 8M steps, entropy and rate lowered after 13M) and
the track (`dew.rl.skydreamer.CIRCLE`)."""

from dew.checkpoints import Keep
from dew.config import ObjectiveConfig, TrainerConfig
from dew.objectives.rl.config import EnvConfig, EnvRunConfig, EpisodesConfig, RolloutConfig
from dew.optim import Linear, OptimConfig
from dew.registry import ModelConfig

ENVIRONMENTS, EPISODE = 16, 2000
SENSED = (("rates", 3), ("motors", 4), ("plan", 24))
"""What the encoder reads beside the mask, each entry and its width."""
DECODED = (("state", 16, 256), *((name, width, 256) for name, width in SENSED))
"""What the world model decodes besides the mask (the informed decoder): the privileged state and the
sensed entries, each two layers of 256 then its width."""

SKYDREAMER = EnvRunConfig(
    env=EnvConfig("skydreamer", {}, count=ENVIRONMENTS, episode_length=EPISODE),
    model=ModelConfig("dew.objectives.rl.dreamer:WorldModel", {
        "reads": "mask", "proprio": [list(entry) for entry in SENSED],
        "auxiliary": [list(head) for head in DECODED], "deter": 2048, "hidden": 256, "stoch": 32,
        "classes": 16, "blocks": 8, "units": 256, "conv_depth": 16, "dtype": "bfloat16"}),
    objective=ObjectiveConfig("dew.objectives.rl.dreamer:DreamerV3Objective",
                              {"slow_targets": True, "smoothness": 0.002}),
    rollout=RolloutConfig("dreamer", {"batch": 16, "length": 64, "train_ratio": 128.0,
                                      "capacity": 10_000_000 // ENVIRONMENTS, "prefill": 64}),
    optim=OptimConfig(optimizer="laprop", learning_rate=Linear(peak_value=4e-5, warmup_steps=1000,
                                                               end_value=4e-5)),
    episodes=EpisodesConfig(steps=EPISODE, reached=("passed",)),
    trainer=TrainerConfig(steps=125_000, log_every=500, eval_every=12_500, checkpoint_every=12_500,
                          keep=Keep(latest=2)))

if __name__ == "__main__":
    EnvRunConfig.cli(default=SKYDREAMER).run()
