import math
import numpy as np

from soar.optimizer import Behavior, PySOARC
from soar.gpr import InternalGPR

import pickle    
from ha_tf import HA

import logging

logging.basicConfig(level=logging.DEBUG)

ha = HA()

MAX_BUDGET = 1000
NUMBER_OF_MACRO_REPLICATIONS = 1
trs_max_budget = 10


for i in range(NUMBER_OF_MACRO_REPLICATIONS):
    starting_seed = 1234565+i
    point_history  = PySOARC(
        n_0= 20,
        nSamples = MAX_BUDGET,
        trs_max_budget = trs_max_budget,
        max_loc_iter=15,
        inpRanges = np.array([[-1.,1.],[-1.,1.]]),
        alpha_lvl_set = .95,
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


