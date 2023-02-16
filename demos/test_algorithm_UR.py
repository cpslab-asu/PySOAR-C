from pysoarc.coreAlgorithm.UR import Behavior, UR_HD

import numpy as np


results_folder = "NonLinearFunction_mo"
benchmark = "Himmelblaus"

# print(specification)

# def test_function(X):
#     return (X[0] ** 2 + X[1] - 11) ** 2 + (X[1] ** 2 + X[0] - 7) ** 2 - 5.
#     # return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 
    
from ha_tf import HA

ha = HA()

MAX_BUDGET = 500
NUMBER_OF_MACRO_REPLICATIONS = 100
trs_max_budget = 10
import pickle

for i in range(NUMBER_OF_MACRO_REPLICATIONS):
    starting_seed = 1234565+i
    point_history  = UR_HD(
        nSamples = MAX_BUDGET,
        inpRanges = np.array([[-1.,1.],[-1.,1.]]),
        test_fn = ha.get_cost,
        seed = starting_seed,
        behavior = Behavior.FALSIFICATION
        )
    with open(f"/home/daittan/RA_Work/pysoar-c/demos/UR_res/UR_max500_rep100_seed{starting_seed}.pickle", "wb") as f:
        pickle.dump(point_history, f)

