from dataclasses import dataclass
from typing import Sequence, Union
from numpy.typing import NDArray
import numpy as np
from staliro.optimizers import ObjFunc, Optimizer
from .optimizer import soarc as _soaroptimizer
from .optimizer import Behavior, LocalBest, LocalPhase, InitializationPhase, GlobalPhase
from .gpr import GaussianProcessRegressor

@dataclass(frozen=True)
class SoarResult:
    algoJourney: list[InitializationPhase | GlobalPhase | LocalPhase | LocalBest]

class SoarOptimizer(Optimizer[Union[float, tuple[float,float]], SoarResult]):
    
    n_0: int
    trs_max_budget: int
    max_loc_iter: int
    alpha_lvl_set: float
    eta0: float
    eta1: float
    delta: float
    gamma: float
    eps_tr: float
    min_tr_size: float
    TR_threshold: float
    gpr_model: GaussianProcessRegressor
    behavior: Behavior
    local_search: str = "gp_local_search"
    
    def optimize(self, func: ObjFunc[float|tuple[float,float]], params: Optimizer.Params) -> SoarResult:
    # def optimize(self, func: ObjFunc[tuple[float, float]], bounds: Sequence[Interval], budget: int, seed: int) -> PySoarResult:
        # Figure this out
        inpRanges = np.array(params.input_bounds)

        # def test_function(sample: np.ndarray) -> float:
        #     return func.eval_sample(Sample(sample))
        
        
        return SoarResult(
            _soaroptimizer(
                n_0= self.n_0,
                nSamples = params.budget,
                trs_max_budget = self.trs_max_budget,
                max_loc_iter=self.max_loc_iter,
                inpRanges = inpRanges,
                alpha_lvl_set = self.alpha_lvl_set,
                eta0 = self.eta0,
                eta1 = self.eta1,
                delta = self.delta,
                gamma = self.gamma,
                eps_tr = self.eps_tr,
                min_tr_size=self.min_tr_size,
                TR_threshold=self.TR_threshold,
                test_fn = func.eval_sample,
                gpr_model = self.gpr_model,
                seed = params.seed,
                local_search= self.local_search,
                behavior = self.behavior
            )
        )
