import math
# from pysoar.pysoar.coreAlgorithm.PySOAR import PySOAR
from staliro.core.model import Model, ModelData, Failure, StaticInput, Signals
from staliro.options import Options, SignalOptions
from staliro.core.interval import Interval
import numpy as np
from numpy.typing import NDArray
from staliro.staliro import staliro
from staliro.options import Options
from staliro.specifications import RTAMTDense

from pysoar import PySOAR
# from test_functions import Himmelblau_2d
from pysoar.gprInterface import InternalGPR

import numpy as np


# NLFDataT = NDArray[np.float_]
# NLFResultT = ModelData[NLFDataT, None]


# class NLFModel(Model[NLFResultT, None]):
#     def simulate(
#         self, static: StaticInput, signals: Signals, intrvl: Interval
#     ) -> NLFResultT:

#         timestamps_array = np.array(1.0).flatten()
#         X = static[0]
#         Y = static[1]
#         d1 = X**3
#         d2 = math.sin(X/2) + math.sin(Y/2) + 2
#         d3 = math.sin((X-3)/2) + math.sin((Y-3)/2) + 4
#         d4 = (math.sin((X - 6)/2)/2) + (math.sin((Y-6)/2)/2) + 2
#         # print(f"True val = {d2}, {d3}, {d4}")
#         data_array = np.hstack((d1, d2,d3, d4)).reshape((-1,1))
        
#         return ModelData(data_array, timestamps_array)


# model = NLFModel()

# initial_conditions = [
#     np.array([-5,5]),
#     np.array([-5,5]),
# ]


# phi_2 = "x>=0"
# phi_3 = "y>=2"
# phi_4 = "z>=1"

# fn_list_1 = f"{phi_2} and {phi_3} and {phi_4}"
# specification = RTAMTDense(fn_list_1, {"x":1, "y":2, "z":3})

results_folder = "TEST123"
benchmark = "NLF"

# print(specification)

def test_function(X):
    return (X[0] ** 2 + X[1] - 11) ** 2 + (X[1] ** 2 + X[0] - 7) ** 2 - 5.

MAX_BUDGET = 2000
NUMBER_OF_MACRO_REPLICATIONS = 10
# n_0, nSamples, inpRanges, alpha_lvl_set, eta0, eta1, delta, gamma, eps_tr, prob, seed, local_search, folder_name, benchmark_name
point_history, modes, simultation_time = PySOAR(
    n_0= 20,
    nSamples = 500,
    trs_max_budget = 20,
    inpRanges = np.array([[-6.,0.],[-6.,6.]]),
    alpha_lvl_set = .05,
    eta0 = .25,
    eta1 = .75,
    delta = .75,
    gamma = 1.25,
    eps_tr = 0.01,
    prob = test_function,
    gpr_model = InternalGPR(),
    seed = 123456,
    local_search= "gp_local_search",
    folder_name= results_folder,
    benchmark_name= f"{benchmark}_budget_{MAX_BUDGET}_{NUMBER_OF_MACRO_REPLICATIONS}_reps",
    behavior = "Minimization"
    )

import pickle

with open("TEST123/NLF_budget_2000_10_reps/NLF_budget_2000_10_reps_seed_123456.pkl", "rb") as f:
    point_history, modes, simultation_time = pickle.load(f)

point_history = np.array(point_history, dtype = object)
distinction = np.array(point_history[:,2])
modes = np.array(point_history[:,3])
print(point_history)
subset_1 = point_history[modes == 0, :]
# print(np.array(subset_1[:,1]))


from matplotlib import pyplot as plt
fig = plt.figure()
        
ax = fig.add_subplot(111)

for i in np.unique(distinction):
    print(distinction == i)
    
    subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
    if subset_1.shape[0] != 0:
        points = np.stack(subset_1[:,1])
        ax.plot(points[:,0], points[:,1], markersize = 5)
        ax.plot(points[0,0], points[0,1], '*', markersize = 5)
    
plt.show()
# subset_1 = point_history[ == 0, :]
# print(subset_1)