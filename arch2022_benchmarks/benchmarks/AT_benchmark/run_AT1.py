
from AT_benchmark.AT_specifications import load_specification_dict
from models import AutotransModel
from Benchmark import Benchmark
from pysoar import run_pysoar

from staliro.staliro import staliro
from staliro.options import Options

# Define Signals and Specification
class Benchmark_AT1(Benchmark):
    def __init__(self, benchmark, results_folder) -> None:
        if benchmark != "AT1":
            raise ValueError("Inappropriate Benchmark name")

        self.results_folder = results_folder
        self.specification, self.signals = load_specification_dict(benchmark)
        print(self.specification)
        print(self.signals)
        self.MAX_BUDGET = 2000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        self.model = AutotransModel()
        self.optimizer = run_pysoar(
             n_0= 50,
             alpha_lvl_set = .05,
             eta0 = .25,
             eta1 = .75,
             delta = .75,
             gamma = 1.25,
             eps_tr = 0.01,
             local_search= '',
             folder_name= self.results_folder,
             benchmark_name= f"{benchmark}_budget_{self.MAX_BUDGET}_{self.NUMBER_OF_MACRO_REPLICATIONS}_reps"
            )

        self.options = Options(runs=self.NUMBER_OF_MACRO_REPLICATIONS, iterations=self.MAX_BUDGET, interval=(0, 50),  signals=self.signals)

    def run(self):
        result = staliro(self.model, self.specification, self.optimizer, self.options)