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

from pysoar.coreAlgorithm import PySOAR_Timed
from pysoar import PySOAR
# from test_functions import Himmelblau_2d
from pysoar.gprInterface import InternalGPR

import numpy as np


results_folder = "Thesis"
benchmark = "2state_ha_pysoar"

# print(specification)

# def test_function(X):
#     return (X[0] ** 2 + X[1] - 11) ** 2 + (X[1] ** 2 + X[0] - 7) ** 2 - 5.
#     # return (X[0]**2 + X[1] - 11)**2 + (X[1]**2 + X[0] - 7)**2 - 40 
    
from ha_tf import HA
ha = HA()

MAX_BUDGET = 300
NUMBER_OF_MACRO_REPLICATIONS = 50
trs_max_budget = 15
seed = 352427
for i in range(NUMBER_OF_MACRO_REPLICATIONS):
    point_history, modes, simultation_time = PySOAR(
        n_0= 20,
        nSamples = MAX_BUDGET,
        trs_max_budget = trs_max_budget,
        max_loc_iter=15,
        inpRanges = np.array([[-1.,1.],[-1.,1.]]),
        alpha_lvl_set = .05,
        eta0 = .25,
        eta1 = .75,
        delta = .75,
        gamma = 1.25,
        eps_tr = 0.01,
        prob = ha.get_robustness,
        gpr_model = InternalGPR(),
        seed = seed + i,
        local_search= "gp_local_search",
        folder_name= results_folder,
        benchmark_name= f"{benchmark}_budget_{MAX_BUDGET}_{trs_max_budget}_maxtrs_1",
        behavior = "Falsification"
        )
