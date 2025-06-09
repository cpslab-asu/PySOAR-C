import math
import os
from pathlib import Path
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
function = "ackley"
dim = 20
NUMBER_OF_MACRO_REPLICATIONS = 10
trs_max_budget = 100

randomstate_main = np.random.RandomState(seed = 4572329)
random_seed_sim = randomstate_main.randint(1976,size = NUMBER_OF_MACRO_REPLICATIONS)

for starting_seed in random_seed_sim:
    
    point_history  = soarc(
        n_0= 100,
        nSamples = MAX_BUDGET,
        trs_max_budget = trs_max_budget,
        max_loc_iter=10,
        inpRanges = np.array([[-32.,32.] for _ in range(dim)]),
        alpha_lvl_set = .5,
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

    folder_path = Path.cwd() / Path(f"LS-exp/{function}/dim_{dim}")
    folder_path.mkdir(parents=True, exist_ok=True)
    
    file_path = folder_path / f"seed_{starting_seed}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(point_history, f)

    print(point_history)
