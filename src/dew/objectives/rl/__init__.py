"""Online RL objectives: rollouts, preference losses and group updates.

These compose the array math in `dew.rl`; the import gate in
tests/test_rl_imports.py keeps that arrow one way. `dew.rl` may read `dew`,
and nothing under `dew` outside these two packages may read `dew.rl`.
"""

from dew.data.preferences import IDS_KEY as PREFERENCE_IDS_KEY, MASK_KEY as PREFERENCE_MASK_KEY

from .asynchronous import POLICY_VERSION_KEY, AsyncRollout, RolloutRecord
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
                       rollout_of,
)
from .fleet import ContainerRunner, Outcome, ProcessRunner, Program, Runner, SandboxFleet, Verdict
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
from .rollouts import Call, Rollout, RolloutSource, Status, Task, pack
from .sandbox import SandboxLimits, SubprocessEnvironment
from .verifiers import CodeReward, MathReward, code_block

__all__ = ["ADVANTAGES_KEY", "IDS_KEY", "OLD_LOG_PROBS_KEY", "POLICY_VERSION_KEY", "PREFERENCE_IDS_KEY",
           "PREFERENCE_MASK_KEY", "RESPONSE_MASK_KEY", "REWARDS_KEY", "Action", "AsyncRollout", "Call", "CodeReward",
           "ContainerRunner", "DPOObjective", "Environment", "EnvironmentFactory",
           "Episode", "EpisodeCancelled", "EpisodeFailure", "EpisodeId", "EpisodeInference", "EpisodeJournal",
           "EpisodeRecorder", "EpisodeRollout", "EpisodeStatus", "FlowGRPOObjective", "FlowReward",
           "FlowRollout", "GRPOObjective", "MathReward", "Observation", "Outcome", "PPOObjective", "PPORollout",
           "ProcessRunner", "Program", "RecoverableEnvironment", "Reward", "Rollout", "RolloutRecord", "RolloutSource", "Runner",
           "SampledRollout", "SandboxFleet", "SandboxLimits", "Status", "SubprocessEnvironment",
           "Task", "Transition", "ValueHead", "Verdict", "Verifier", "code_block", "pack", "rollout_of"]
