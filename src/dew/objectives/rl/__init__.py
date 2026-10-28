"""Online RL objectives: rollouts, preference losses and group updates.

These objectives combine the array math in `dew.rl`. Imports go one way:
`dew.rl` may import from `dew`, and nothing in `dew` outside `dew.rl` and
this package may import `dew.rl`.

Each name is imported from its module when first used, so importing one of the package's
modules, or a name from one, reaches only what that module imports.
"""

# tests/test_rl_imports.py enforces the one-way import rule above.
from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.data.prompts import PromptRecord
    from dew.objectives.rl.episodes import (
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
    from dew.objectives.rl.flow import REWARDS_KEY, FlowGRPOObjective, FlowReward, FlowRollout
    from dew.objectives.rl.grpo import GRPOObjective
    from dew.objectives.rl.journal import EpisodeJournal
    from dew.objectives.rl.ppo import PPOObjective, PPORollout, ValueHead
    from dew.objectives.rl.preference import DPOObjective
    from dew.objectives.rl.rollout import PolicyCollector, Reward, prompt_tasks
    from dew.objectives.rl.scheduler import (
        Asynchronous,
        Publisher,
        RolloutScheduler,
        SchedulerRecord,
        Synchronous,
        task_ids,
    )
    from dew.objectives.rl.sessions import (
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
    from dew.objectives.rl.sources import EnvironmentSource, PromptSource

_EXPORTS = {
    "PromptRecord": "dew.data.prompts",
    "Environment": "dew.objectives.rl.episodes", "EnvironmentFactory": "dew.objectives.rl.episodes",
    "Episode": "dew.objectives.rl.episodes", "EpisodeCancelled": "dew.objectives.rl.episodes",
    "EpisodeCollector": "dew.objectives.rl.episodes", "EpisodeFailure": "dew.objectives.rl.episodes",
    "EpisodeId": "dew.objectives.rl.episodes", "EpisodeInference": "dew.objectives.rl.episodes",
    "EpisodeRecorder": "dew.objectives.rl.episodes", "EpisodeStatus": "dew.objectives.rl.episodes",
    "Observation": "dew.objectives.rl.episodes", "RecoverableEnvironment": "dew.objectives.rl.episodes",
    "Transition": "dew.objectives.rl.episodes", "Verifier": "dew.objectives.rl.episodes",
    "session_of": "dew.objectives.rl.episodes",
    "REWARDS_KEY": "dew.objectives.rl.flow", "FlowGRPOObjective": "dew.objectives.rl.flow",
    "FlowReward": "dew.objectives.rl.flow", "FlowRollout": "dew.objectives.rl.flow",
    "GRPOObjective": "dew.objectives.rl.grpo",
    "EpisodeJournal": "dew.objectives.rl.journal",
    "PPOObjective": "dew.objectives.rl.ppo", "PPORollout": "dew.objectives.rl.ppo",
    "ValueHead": "dew.objectives.rl.ppo",
    "DPOObjective": "dew.objectives.rl.preference",
    "PolicyCollector": "dew.objectives.rl.rollout", "Reward": "dew.objectives.rl.rollout",
    "prompt_tasks": "dew.objectives.rl.rollout",
    "Asynchronous": "dew.objectives.rl.scheduler", "Publisher": "dew.objectives.rl.scheduler",
    "RolloutScheduler": "dew.objectives.rl.scheduler", "SchedulerRecord": "dew.objectives.rl.scheduler",
    "Synchronous": "dew.objectives.rl.scheduler", "task_ids": "dew.objectives.rl.scheduler",
    "ADVANTAGES_KEY": "dew.objectives.rl.sessions", "IDS_KEY": "dew.objectives.rl.sessions",
    "OLD_LOG_PROBS_KEY": "dew.objectives.rl.sessions", "RESPONSE_MASK_KEY": "dew.objectives.rl.sessions",
    "TOKEN_REWARDS_KEY": "dew.objectives.rl.sessions", "Call": "dew.objectives.rl.sessions",
    "Collector": "dew.objectives.rl.sessions", "Session": "dew.objectives.rl.sessions",
    "SessionSource": "dew.objectives.rl.sessions", "Snapshot": "dew.objectives.rl.sessions",
    "Status": "dew.objectives.rl.sessions", "Task": "dew.objectives.rl.sessions",
    "Transcript": "dew.objectives.rl.sessions", "pack": "dew.objectives.rl.sessions",
    "EnvironmentSource": "dew.objectives.rl.sources", "PromptSource": "dew.objectives.rl.sources",
}
__getattr__ = _lookup(__name__, _EXPORTS)

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


def __dir__() -> list[str]:
    return list(__all__)
