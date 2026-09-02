from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EvaluationMetric:
    """A per-batch measurement and the reduction across validation batches."""

    function: Callable
    name: str
    higher_is_better: bool = False
    reducer: Callable = np.mean