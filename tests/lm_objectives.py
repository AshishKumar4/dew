"""An LM objective of a user's own, which a run names by its import path."""

from dew.objectives.lm import LMObjective


class Anchored(LMObjective):
    """The language model's loss with a fixed z-loss anchoring the log-partition, a class Dew does
    not ship."""

    def __init__(self, model, seq_len, **arguments):
        super().__init__(model, seq_len, z_loss=1e-4, **arguments)
