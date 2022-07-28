from cgi import test
from dataclasses import dataclass
from typing import Any, List, Sequence, Callable

import numpy as np
from staliro.core import Interval, Optimizer, ObjectiveFn, Sample

from ..coreAlgorithm import PySOAR

Bounds = Sequence[Interval]
PySOARResult = List[Any]

@dataclass(frozen=True)
class run_pysoar(Optimizer[PySOARResult]):
    """The PartX optimizer provides statistical guarantees about the existence of falsifying behaviour in a system."""

    # inpRanges: 
    
    n_0: int

    alpha_lvl_set: float
    eta0: float
    eta1: float
    delta: float
    gamma: float
    eps_tr: float

    local_search: str
    folder_name: str
    benchmark_name: str

    def optimize(self, func: ObjectiveFn, bounds: Bounds, budget:int, seed: int) -> PySOARResult:
        region_support = np.array((tuple(bound.astuple() for bound in bounds),))
        
        def test_function(sample: np.ndarray) -> float:
            return func.eval_sample(Sample(sample))
        
        return PySOAR(n_0=self.n_0, 
            nSamples=budget, 
            inpRanges= region_support,
            alpha_lvl_set=self.alpha_lvl_set, 
            eta0=self.eta0, 
            eta1=self.eta1, 
            delta=self.delta, 
            gamma=self.gamma, 
            eps_tr=self.eps_tr, 
            seed=seed,
            prob= test_function,
            local_search=self.local_search,
            folder_name=self.folder_name, 
            benchmark_name=self.benchmark_name
        )