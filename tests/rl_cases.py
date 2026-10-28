"""The rows an RL objective trains on (tests/objective_cases.py's GRPO and PPO cases): the sessions a
rollout packs, apart from the other objectives' cases."""

from objective_data import SEQ_LEN

from dew.objectives.rl.sessions import Call, Session, Status, pack


def packed(objective_for):
    """Rows `sessions.pack` writes, what every GRPO rollout and the RLVR
    example's scheduler train on, at the width the objective names."""
    width = SEQ_LEN + 1
    sessions = [Session("t", str(sample // 2), sample, 0,
                        (Call((1, 2, 3), (4, 5), (-0.1, -0.2), "stop", 0),),
                        Status.COMPLETED, float(sample % 2)) for sample in range(4)]
    return objective_for(width - 1), pack(sessions, width)
