import math
from staliro.core.model import Model, ModelData, Failure, StaticInput, Signals
from staliro.options import Options, SignalOptions
from staliro.core.interval import Interval
import numpy as np
from numpy.typing import NDArray
from staliro.staliro import staliro
from staliro.options import Options
from staliro.specifications import RTAMTDense

from pysoar import run_pysoar
# from test_functions import Himmelblau_2d

import numpy as np

# nInputs = 2
# n_0 = 10 * nInputs

# DEMO


# # run macroreplications
# macrorep = 2
# falsified = np.zeros((macrorep, 1))
# best_rob = np.zeros((macrorep, 1))
# for repNum in range(macrorep):
#     run, history = PySOAR(inpRanges, prob, repNum, local_search='gradient')
#     best_rob[repNum, 0] = run['bestRob']
#     if best_rob[repNum, 0] <= 0:
#         falsified[repNum, 0] = 1
# print(falsified, best_rob)

NLFDataT = NDArray[np.float_]
NLFResultT = ModelData[NLFDataT, None]


class NLFModel(Model[NLFResultT, None]):
    def simulate(
        self, static: StaticInput, signals: Signals, intrvl: Interval
    ) -> NLFResultT:

        timestamps_array = np.array(1.0).flatten()
        X = static[0]
        Y = static[1]
        d1 = X**3
        d2 = math.sin(X/2) + math.sin(Y/2) + 2
        d3 = math.sin((X-3)/2) + math.sin((Y-3)/2) + 4
        d4 = (math.sin((X - 6)/2)/2) + (math.sin((Y-6)/2)/2) + 2
        # print(f"True val = {d2}, {d3}, {d4}")
        data_array = np.hstack((d1, d2,d3, d4)).reshape((-1,1))
        
        return ModelData(data_array, timestamps_array)


model = NLFModel()

initial_conditions = [
    np.array([-5,5]),
    np.array([-5,5]),
]


phi_2 = "x>=0"
phi_3 = "y>=2"
phi_4 = "z>=1"

fn_list_1 = f"{phi_2} and {phi_3} and {phi_4}"
specification = RTAMTDense(fn_list_1, {"x":1, "y":2, "z":3})

results_folder = "TEST123"
benchmark = "NLF"

print(specification)

MAX_BUDGET = 2000
NUMBER_OF_MACRO_REPLICATIONS = 10

optimizer = run_pysoar(
        n_0= 50,
        alpha_lvl_set = .05,
        eta0 = .25,
        eta1 = .75,
        delta = .75,
        gamma = 1.25,
        eps_tr = 0.01,
        local_search= "",
        folder_name= results_folder,
        benchmark_name= f"{benchmark}_budget_{MAX_BUDGET}_{NUMBER_OF_MACRO_REPLICATIONS}_reps"
    )

options = Options(runs=NUMBER_OF_MACRO_REPLICATIONS, iterations=MAX_BUDGET, interval=(0, 50),  static_parameters = initial_conditions,signals=[])

staliro(model, specification, optimizer, options)