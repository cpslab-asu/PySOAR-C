import math
import numpy as np

from pysoarc import Behavior, PySOARC
from pysoarc.gprInterface import InternalGPR

    
from ha_tf import HA

ha = HA()

MAX_BUDGET = 500
NUMBER_OF_MACRO_REPLICATIONS = 1
trs_max_budget = 15
import pickle

for i in range(NUMBER_OF_MACRO_REPLICATIONS):
    starting_seed = 1234565+i
    point_history  = PySOARC(
        n_0= 20,
        nSamples = MAX_BUDGET,
        trs_max_budget = trs_max_budget,
        max_loc_iter=10,
        inpRanges = np.array([[-1.,1.],[-1.,1.]]),
        alpha_lvl_set = .05,
        eta0 = .25,
        eta1 = .75,
        delta = .75,
        gamma = 1.25,
        eps_tr = 0.01,
        min_tr_size=5.0,
        TR_threshold=0.05,
        test_fn = ha.get_cost,
        gpr_model = InternalGPR(),
        seed = starting_seed,
        local_search= "gp_local_search",
        behavior = Behavior.FALSIFICATION
        )
    with open(f"pysoarc_max500_rep_50_seed{starting_seed}.pickle", "wb") as f:
        pickle.dump(point_history, f)


