import math
# from pysoar.pysoar.coreAlgorithm.PySOAR import PySOAR
import pickle

import numpy as np
from numpy.typing import NDArray

from pysoar import PySOAR
# from test_functions import Himmelblau_2d
from pysoar.gprInterface import InternalGPR

import numpy as np


results_folder = "NonLinearFunction_mo"
benchmark = "Himmelblaus"



def test_function(X):
    a = (100 * (X[1] - X[0]**2)**2 + ((X[0]-1)**2))
    b = (100 * (X[2] - X[1]**2)**2 + ((X[1]-1)**2))
    c = (100 * (X[3] - X[2]**2)**2 + ((X[2]-1)**2))
    d = (100 * (X[4] - X[3]**2)**2 + ((X[3]-1)**2))
    return a + b + c + d
    # return "f(X[0], X[1], X[2], X[3], X[4])"
    
    

MAX_BUDGET = 1200
NUMBER_OF_MACRO_REPLICATIONS = 10
trs_max_budget = 50

for replication in range(NUMBER_OF_MACRO_REPLICATIONS):
    point_history, modes, sim_times = PySOAR(
        n_0= 100,
        nSamples = MAX_BUDGET,
        trs_max_budget = trs_max_budget,
        max_loc_iter=30,
        inpRanges = np.array([[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.],[-2.,2.]]),
        alpha_lvl_set = .05,
        eta0 = .25,
        eta1 = .75,
        delta = .75,
        gamma = 1.25,
        eps_tr = 0.001,
        prob = test_function,
        gpr_model = InternalGPR(),
        seed = 1234567,
        local_search= "gp_local_search",
        folder_name= results_folder,
        benchmark_name= f"{benchmark}_budget_{MAX_BUDGET}_{trs_max_budget}_maxtrs",
        behavior = "Minimization"
        )

    with open(f"Pysor_res_{replication}", "wb") as f:
        pickle.dump((point_history, modes, sim_times), f)


point_history = np.array(point_history, dtype = object)
distinction = np.array(point_history[:,2])
modes = np.array(point_history[:,3])
regions = np.array(point_history[:,4])


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
fig = plt.figure()
        
ax = fig.add_subplot(111)



for i in np.unique(distinction):
    # print(distinction == i)

    subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
    # subset_1 = point_history[distinction == i, :]

    if subset_1.shape[0] != 0:
        all_points = np.stack(subset_1[:,1])
        ax.plot(all_points[:,0], all_points[:,1], markersize = 0.1)
        for itera,p in enumerate(subset_1):
            points = p[1]
            region = p[4]
            # print(points)
            # print(region)
            x = region[0,0]
            y = region[1,0]
            w = region[0,1] - region[0,0]
            h = region[1,1] - region[1,0]

            if itera == 0:
                ax.plot(points[0], points[1], 'b*', markersize = 7)
                ax.annotate(f"{i}-{itera}",xy=(points[0]+0.001, points[1]+0.001))
            else:
                ax.plot(points[0], points[1], 'b.', markersize = 4)
                ax.annotate(f"{i}-{itera}",xy=(points[0]+0.001, points[1]+0.001))
            # ax.add_patch( Rectangle((x,y),
            #                     w, h,
            #                     ec = 'black',
            #                     fc ='green',
            #                     alpha = 0.1))    


for i in np.unique(distinction):
    # print(distinction == i)

    # subset_1 = point_history[np.logical_and(np.logical_or(modes == 1, modes == 3), distinction == i), :]
    subset_1 = point_history[distinction == i, :]

    if subset_1.shape[0] != 0:
        all_points = np.stack(subset_1[:,1])
        ax.plot(all_points[:,0], all_points[:,1], "b.", markersize = 4)
        ax.plot(all_points[0,0], all_points[0,1], "b*", markersize = 7)
# ax.set_aspect('equal', adjustable = 'box')
# ax2.set_aspect('equal', adjustable = 'box')
plt.show()
plt.savefig(f"test.png", bbox_inches='tight', dpi = 500, transparent = False)