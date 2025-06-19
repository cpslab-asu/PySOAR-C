# from ha_tf import HA
import logging
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable

from soar.gpr import InternalGPR
from soar.optimizer import Behavior, soarc

logging.basicConfig(level=logging.DEBUG)


def branin(x):
    x = torch.asarray(x)
    flat = x.dim() == 1
    if flat:
        x = x.view(1, -1)
    ndim = x.size(1)
    n_repeat = np.int32(ndim / 2)
    n_dummy = ndim % 2

    shift = torch.cat([torch.FloatTensor([2.5, 7.5]).repeat(n_repeat), torch.zeros(n_dummy)])

    if hasattr(x, 'data'):
        x.data = x.data * 7.5 + shift.type_as(x.data)
    else:
        x = x * 7.5 + shift.type_as(x)
    a = 1
    b = 5.1 / (4 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6
    s = 10
    t = 1.0 / (8 * math.pi)
    output = 0
    for i in range(n_repeat):
        output += a * (x[:, 2 * i + 1] - b * x[:, 2 * i] ** 2 + c * x[:, 2 * i] - r) ** 2 + s * (1 - t) * torch.cos(x[:, 2 * i]) + s
    output /= float(n_repeat)
    if flat:
        return float(output.squeeze(0))
    else:
        return np.array(output)[0]


MAX_BUDGET = 10000
function = "branin"
dim = 10
NUMBER_OF_MACRO_REPLICATIONS = 10
trs_max_budget = 100

randomstate_main = np.random.RandomState(seed = 4572329)
random_seed_sim = randomstate_main.randint(1976,size = NUMBER_OF_MACRO_REPLICATIONS)

for starting_seed in random_seed_sim:
    
    point_history  = soarc(
        n_0= 100,
        n_samples = MAX_BUDGET,
        trs_max_budget = trs_max_budget,
        max_loc_iter=100,
        inp_ranges = np.array([[-1.,1.] for _ in range(dim)]),
        alpha_lvl_set = 0.95,
        eta0 = .25,
        eta1 = .75,
        delta = .75,
        gamma = 1.25,
        eps_tr = 0.000001,
        min_tr_size=64,
        tr_threshold=0.0001,
        test_fn = branin,
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
