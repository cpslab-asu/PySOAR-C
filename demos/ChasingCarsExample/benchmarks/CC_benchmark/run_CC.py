
import logging
from .CC_specifications import load_specification_dict
from models import CCModel
from Benchmark import Benchmark

import staliro
from staliro import TestOptions
from soar import SoarOptimizer, Behavior
from soar.gpr import InternalGPR

logging.basicConfig(level=logging.DEBUG)

# Define Signals and Specification
class Benchmark_CC(Benchmark):
    def __init__(self, benchmark, instance, results_folder) -> None:
        
        self.results_folder = results_folder
        self.specification, self.signals = load_specification_dict(benchmark, instance)
        

        
        self.MAX_BUDGET = 2000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        self.model = CCModel()

        
        self.optimizer = SoarOptimizer(
            n_0 = 20,
            trs_max_budget = 10,
            max_loc_iter = 15,
            alpha_lvl_set = 0.95,
            eta0 = 0.25,
            eta1 = 0.75,
            delta = 0.75,
            gamma = 1.25,
            eps_tr = 0.01,
            min_tr_size = 2.0,
            TR_threshold = 0.05,
            gpr_model = InternalGPR(),
            behavior = Behavior.FALSIFICATION,
        )
        

        self.options = TestOptions(tspan=(0,100), iterations=self.MAX_BUDGET, runs = 1, signals=self.signals)
        
    def run(self):
        result = staliro.test(self.model, self.specification, self.optimizer, self.options)
        # result = staliro(self.model, self.specification, self.optimizer, self.options)