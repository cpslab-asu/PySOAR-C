from dataclasses import dataclass
from typing import List
from attr import frozen

import numpy as np
from numpy.typing import NDArray
from staliro.optimizers import Optimizer, ObjFunc

from ..coreAlgorithm import PySOARC
from ..gprInterface import GaussianProcessRegressor


@frozen(slots=True)
class PySOARCResult:
    """Data class that represents the result of a uniform random optimization.

    Attributes:
        average_cost: The average cost of all the samples selected.
    """

    algorithm_points: List


@dataclass(frozen=True)
class run_pysoarc(Optimizer[NDArray[np.double], PySOARCResult]):
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
    gpr_model: GaussianProcessRegressor
    local_search: str
    behavior: PySOARC.Behavior = PySOARC.Behavior.MINIMIZATION

    def optimize(self, func: ObjFunc[NDArray[np.double]], params: Optimizer.Params) -> PySOARCResult:
        region_support = np.array([bound for bound in params.input_bounds])
        samples = PySOARC.PySOARC(
            n_0=self.n_0, 
            nSamples=params.budget, 
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
            test_fn=lambda sample: func.eval_sample(sample),
            gpr_model=self.gpr_model,
            seed=params.seed,
            local_search=self.local_search,
            behavior=self.behavior
        )

        return PySOARCResult(samples)                                
