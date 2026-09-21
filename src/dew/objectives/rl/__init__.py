"""Online RL objectives: rollouts, preference losses and group updates.

These compose the array math in `dew.rl`; the import gate in
tests/test_rl_imports.py keeps that arrow one way. `dew.rl` may read `dew`,
and nothing under `dew` outside these two packages may read `dew.rl`.
"""

from dew.data.preferences import IDS_KEY as PREFERENCE_IDS_KEY, MASK_KEY as PREFERENCE_MASK_KEY

from .episodes import (
                       Action,
                       Environment,
                       EnvironmentFactory,
                       Episode,
                       EpisodeCancelled,
                       EpisodeFailure,
                       EpisodeId,
                       EpisodeInference,
                       EpisodeRecorder,
                       EpisodeRollout,
                       EpisodeStatus,
                       Observation,
                       RecoverableEnvironment,
                       Transition,
                       Verifier,
)
from .flow import FlowGRPOObjective, FlowReward, FlowRollout
from .grpo import GRPOObjective
from .journal import EpisodeJournal
from .ppo import PPOObjective, PPORollout, ValueHead
from .preference import DPOObjective
from .rollout import (
                       ADVANTAGES_KEY,
                       IDS_KEY,
                       OLD_LOG_PROBS_KEY,
                       RESPONSE_MASK_KEY,
                       REWARDS_KEY,
                       Reward,
                       SampledRollout,
)
from .sandbox import SandboxLimits, SubprocessEnvironment

__all__ = ["ADVANTAGES_KEY", "IDS_KEY", "OLD_LOG_PROBS_KEY", "PREFERENCE_IDS_KEY", "PREFERENCE_MASK_KEY",
           "RESPONSE_MASK_KEY", "REWARDS_KEY", "Action", "DPOObjective", "Environment", "EnvironmentFactory",
           "Episode", "EpisodeCancelled", "EpisodeFailure", "EpisodeId", "EpisodeInference", "EpisodeJournal",
           "EpisodeRecorder", "EpisodeRollout", "EpisodeStatus", "FlowGRPOObjective", "FlowReward",
           "FlowRollout", "GRPOObjective", "Observation", "PPOObjective", "PPORollout",
           "RecoverableEnvironment", "Reward", "SampledRollout", "SandboxLimits", "SubprocessEnvironment",
           "Transition", "ValueHead", "Verifier"]
