import math
import numpy as np

from soar.optimizer import Behavior, soarc
from soar.gpr import InternalGPR

import pickle    
# from ha_tf import HA

import logging

logging.basicConfig(level=logging.DEBUG)


import math
import numpy as np
# import torch
# from torch.autograd import Variable

def ackley(curxvec):
    a=20
    b=0.2
    c=2*np.pi
    d = len(curxvec)
    sum1 = np.sum(curxvec**2)
    sum2 = np.sum(np.cos(c*curxvec))
    term1 = -a * np.exp(-b*np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)
    exval =  term1 + term2 + a + np.exp(1)
    return exval

MAX_BUDGET = 10000
NUMBER_OF_MACRO_REPLICATIONS = 1
trs_max_budget = 100


for i in range(NUMBER_OF_MACRO_REPLICATIONS):
    starting_seed = 1234565+i
    point_history  = soarc(
        n_0= 100,
        nSamples = MAX_BUDGET,
        trs_max_budget = trs_max_budget,
        max_loc_iter=10,
        inpRanges = np.array([[-32.,32.] for _ in range(10)]),
        alpha_lvl_set = .95,
        eta0 = .25,
        eta1 = .75,
        delta = .75,
        gamma = 1.25,
        eps_tr = 0.0000001,
        min_tr_size=1.28,
        TR_threshold=1,
        test_fn = ackley,
        gpr_model = InternalGPR(),
        seed = starting_seed,
        local_search= "gp_local_search",
        behavior = Behavior.MINIMIZATION
        )
    with open(f"pysoarc_max500_rep_50_seed{starting_seed}_2.pickle", "wb") as f:
        pickle.dump(point_history, f)

    print(point_history)
