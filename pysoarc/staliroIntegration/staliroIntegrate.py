from dataclasses import dataclass
from typing import Any, List, Sequence, Callable
from attr import frozen

import numpy as np
from numpy.typing import NDArray
from staliro.core import Interval, Optimizer, ObjectiveFn, Sample

from ..coreAlgorithm import PySOARC

Bounds = Sequence[Interval]

@frozen(slots=True)
class PySOARCResult:
    """Data class that represents the result of a uniform random optimization.

    Attributes:
        average_cost: The average cost of all the samples selected.
    """

    algorithm_points: List


@dataclass(frozen=True)
class run_pysoarc(Optimizer[PySOARCResult, None]):
    """The PySOARC optimizer"""

    # inpRanges: 
    
    n_0: int
    trs_max_budget: int
    max_loc_iter:int
    alpha_lvl_set: float
    eta0: float
    eta1: float
    delta: float
    gamma: float
    eps_tr: float
    min_tr_size: float
    TR_threshold: float
    gpr_model: Callable
    local_search: str
    behavior: str

    def optimize(self, func: ObjectiveFn, bounds: Bounds, budget:int, seed: int) -> PySOARCResult:
        region_support = np.array((tuple(bound.astuple() for bound in bounds),))[0]
        
        def test_function(sample: np.ndarray) -> float:
            return func.eval_sample(Sample(sample))
        
        samples = PySOARC(
                        n_0=self.n_0, 
                        nSamples=budget, 
                        trs_max_budget=self.trs_max_budget,
                        max_loc_iter=self.max_loc_iter,
                        inpRanges= region_support,
                        alpha_lvl_set=self.alpha_lvl_set, 
                        eta0=self.eta0, 
                        eta1=self.eta1, 
                        delta=self.delta, 
                        gamma=self.gamma, 
                        eps_tr=self.eps_tr, 
                        min_tr_size=self.min_tr_size,
                        TR_threshold=self.TR_threshold,
                        prob= test_function,
                        gpr_model=self.gpr_model,
                        seed = seed,
                        local_search=self.local_search,
                        behavior=self.behavior
                )
        return PySOARCResult(samples)                                