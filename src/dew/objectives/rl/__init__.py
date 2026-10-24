"""Online RL objectives: rollouts, preference losses and group updates.

These objectives combine the array math in `dew.rl`. Imports go one way:
`dew.rl` may import from `dew`, and nothing in `dew` outside `dew.rl` and
this package may import `dew.rl`.
"""

# tests/test_rl_imports.py enforces the one-way import rule above.
from dew.data.prompts import PromptRecord

from .episodes import (
                       Environment,
                       EnvironmentFactory,
                       Episode,
                       EpisodeCancelled,
                       EpisodeCollector,
                       EpisodeFailure,
                       EpisodeId,
                       EpisodeInference,
                       EpisodeRecorder,
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
from .rollout import PolicyCollector, Reward, prompt_tasks
from .scheduler import Asynchronous, Publisher, RolloutScheduler, SchedulerRecord, Synchronous, task_ids
from .sessions import (
                       ADVANTAGES_KEY,
                       IDS_KEY,
                       OLD_LOG_PROBS_KEY,
                       RESPONSE_MASK_KEY,
                       TOKEN_REWARDS_KEY,
                       Call,
                       Collector,
                       Session,
                       SessionSource,
                       Snapshot,
                       Status,
                       Task,
                       Transcript,
                       pack,
)
from .sources import EnvironmentSource, PromptSource

__all__ = [
                       "ADVANTAGES_KEY",
                       "IDS_KEY",
                       "OLD_LOG_PROBS_KEY",
                       "RESPONSE_MASK_KEY",
                       "REWARDS_KEY",
                       "TOKEN_REWARDS_KEY",
                       "Asynchronous",
                       "Call",
                       "Collector",
                       "DPOObjective",
                       "Environment",
                       "EnvironmentFactory",
                       "EnvironmentSource",
                       "Episode",
                       "EpisodeCancelled",
                       "EpisodeCollector",
                       "EpisodeFailure",
                       "EpisodeId",
                       "EpisodeInference",
                       "EpisodeJournal",
                       "EpisodeRecorder",
                       "EpisodeStatus",
                       "FlowGRPOObjective",
                       "FlowReward",
                       "FlowRollout",
                       "GRPOObjective",
                       "Observation",
                       "PPOObjective",
                       "PPORollout",
                       "PolicyCollector",
                       "PromptRecord",
                       "PromptSource",
                       "Publisher",
                       "RecoverableEnvironment",
                       "Reward",
                       "RolloutScheduler",
                       "SchedulerRecord",
                       "Session",
                       "SessionSource",
                       "Snapshot",
                       "Status",
                       "Synchronous",
                       "Task",
                       "Transcript",
                       "Transition",
                       "ValueHead",
                       "Verifier",
                       "pack",
                       "prompt_tasks",
                       "session_of",
                       "task_ids",
]
