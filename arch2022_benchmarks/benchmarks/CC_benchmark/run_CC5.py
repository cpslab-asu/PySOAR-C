
from CC_benchmark.CC_specifications import load_specification_dict
from models import CCModel
from Benchmark import Benchmark
from pysoar import run_pysoar
from pysoar.gprInterface import InternalGPR
from staliro.staliro import staliro
from staliro.options import Options

# Define Signals and Specification
class Benchmark_CC5(Benchmark):
    def __init__(self, benchmark, results_folder) -> None:
        if benchmark != "CC5":
            raise ValueError("Inappropriate Benchmark name")

        self.results_folder = results_folder
        self.specification, self.signals = load_specification_dict(benchmark)
        print(self.specification)
        print(self.signals)
        self.MAX_BUDGET = 2000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        self.model = CCModel()
        self.optimizer = run_pysoar(
             n_0= 50,
             trs_max_budget = 10,
             max_loc_iter = 10,
             alpha_lvl_set = 0.05,
             eta0 = .25,
             eta1 = .75,
             delta = .75,
             gamma = 1.25,
             eps_tr = 0.01,
             gpr_model=InternalGPR(),
             local_search= "gp_local_search",
             folder_name= self.results_folder,
             benchmark_name= f"{benchmark}_budget_{self.MAX_BUDGET}_{self.NUMBER_OF_MACRO_REPLICATIONS}_reps_actual",
             behavior = "Falsification"
            )
         
        self.options = Options(runs=self.NUMBER_OF_MACRO_REPLICATIONS, iterations=self.MAX_BUDGET, interval=(0, 100),  signals=self.signals)

    def run(self):
        result = staliro(self.model, self.specification, self.optimizer, self.options)