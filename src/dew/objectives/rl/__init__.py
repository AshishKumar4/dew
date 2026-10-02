"""Online RL objectives: rollouts, preference losses and group updates.

These compose the array math in `dew.rl`; the import gate in
tests/test_rl_imports.py keeps that arrow one way. `dew.rl` may read `dew`,
and nothing under `dew` outside these two packages may read `dew.rl`.
"""

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
                       session_of,
)
from .flow import REWARDS_KEY, FlowGRPOObjective, FlowReward, FlowRollout
from .grpo import GRPOObjective
from .journal import EpisodeJournal
from .ppo import PPOObjective, PPORollout, ValueHead
from .preference import DPOObjective
from .rollout import Reward, SampledRollout
from .scheduler import Publisher, RolloutScheduler, SchedulerRecord
from .sessions import (
                       ADVANTAGES_KEY,
                       IDS_KEY,
                       OLD_LOG_PROBS_KEY,
                       RESPONSE_MASK_KEY,
                       Call,
                       Session,
                       SessionSource,
                       Status,
                       Task,
                       pack,
)
from .sources import EnvironmentSource, PromptSource, prompt_tasks

__all__ = [
                       "ADVANTAGES_KEY",
                       "IDS_KEY",
                       "OLD_LOG_PROBS_KEY",
                       "RESPONSE_MASK_KEY",
                       "REWARDS_KEY",
                       "Action",
                       "Call",
                       "DPOObjective",
                       "Environment",
                       "EnvironmentFactory",
                       "EnvironmentSource",
                       "Episode",
                       "EpisodeCancelled",
                       "EpisodeFailure",
                       "EpisodeId",
                       "EpisodeInference",
                       "EpisodeJournal",
                       "EpisodeRecorder",
                       "EpisodeRollout",
                       "EpisodeStatus",
                       "FlowGRPOObjective",
                       "FlowReward",
                       "FlowRollout",
                       "GRPOObjective",
                       "Observation",
                       "PPOObjective",
                       "PPORollout",
                       "PromptSource",
                       "Publisher",
                       "RecoverableEnvironment",
                       "Reward",
                       "RolloutScheduler",
                       "SampledRollout",
                       "SchedulerRecord",
                       "Session",
                       "SessionSource",
                       "Status",
                       "Task",
                       "Transition",
                       "ValueHead",
                       "Verifier",
                       "pack",
                       "prompt_tasks",
                       "session_of",
]
