import logging
import enum
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..sampling import lhs_sampling, uniform_sampling


_logger = logging.getLogger("Hybrid_Distance_UR")

class Behavior(enum.IntEnum):
    """Behavior when falsifying case for system is encountered.

    Attributes:
        FALSIFICATION: Stop searching when the first falsifying case is encountered
        MINIMIZATION: Continue searching after encountering a falsifying case until iteration
                      budget is exhausted
    """

    FALSIFICATION = enum.auto()
    MINIMIZATION = enum.auto()

@dataclass(frozen=True)
class InitializationPhase:
    initial_samples_x: NDArray[np.double]
    initial_samples_y: NDArray[np.double]

def _is_falsification(evaluation: NDArray) -> bool:
    return evaluation[0] == 0 and evaluation[1] < 0


class Fn:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *arg):
        self.count = self.count + 1
        
        hybrid_dist = self.func(*arg)
        # print(self.count, arg[0], hybrid_dist)
        
        return hybrid_dist

def _evaluate_samples(
                samples: NDArray[np.double], 
                fn: Callable[[NDArray[np.double]], NDArray[np.double]],
                behavior: Behavior
) -> NDArray:

    evaluations = []
    
    modified_x_train = []
    for sample in samples:
        evaluation = np.array(fn(sample), dtype=np.double)
        evaluations.append(evaluation)
        modified_x_train.append(sample)
        if _is_falsification(evaluation) and (behavior is Behavior.FALSIFICATION):
            print(evaluation)
            break

    return np.array(modified_x_train), np.array(evaluations)


##### v9 ####### add user defined parameters to input, break once falsified
def UR_HD(
    nSamples: int,
    inpRanges: ArrayLike, 
    test_fn: Callable[[NDArray[np.double]], NDArray[np.double]],
    seed: int, 
    behavior: Behavior = Behavior.FALSIFICATION
):
    inpRanges = np.array(inpRanges)
    test_fn = Fn(test_fn)
    if inpRanges.ndim != 2:
        raise ValueError("input ranges should be 2-dimensional")

    if inpRanges.shape[1] != 2:
        raise ValueError("input range 2nd dimension should be equal to 2")

    rng = np.random.default_rng(seed)
    np.random.seed(seed+1000)

    tf_dim = inpRanges.shape[0]
    if nSamples > nSamples:
        raise ValueError(f"Received n_0({nSamples}) > nSamples ({nSamples}): Initial samples (n_0) cannot be greater than Maximum Evaluations Budget (nSamples)")

    initial_samples = lhs_sampling(nSamples, inpRanges, tf_dim, rng)
    # inital_samples_hd = initial_samples
    initial_samples, initial_sample_distances = _evaluate_samples(initial_samples, test_fn, behavior)
    
    initial_points = InitializationPhase(
                        initial_samples_x = initial_samples,
                        initial_samples_y = initial_sample_distances
                    )
    algo_journey = [initial_points]
    print(initial_samples.shape)
    return algo_journey